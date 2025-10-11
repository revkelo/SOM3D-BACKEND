# s3_manager.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import os

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError as _BotoClientError
from botocore.exceptions import EndpointConnectionError as _BotoEndpointError


@dataclass
class S3Config:
    """
    Configuración del cliente S3/MinIO.
    - endpoint: p.ej. "http://192.168.0.10:9010" (para MinIO local con HTTP)
    - insecure: True si usas HTTP sin TLS (MinIO local); False si usas HTTPS
    - bucket: nombre del bucket
    - prefix: prefijo raíz para tus objetos (se normaliza para terminar en '/')
    - region: región (para S3 real); en MinIO da igual, usa "us-east-1"
    - access_key / secret_key: credenciales
    """
    endpoint: Optional[str] = None
    insecure: bool = False
    bucket: str = "som3d"
    prefix: str = "jobs/"
    region: str = "us-east-1"
    access_key: Optional[str] = None
    secret_key: Optional[str] = None

    def __post_init__(self):
        # normaliza prefix: sin dobles // y terminando en /
        p = (self.prefix or "").strip()
        while "//" in p:
            p = p.replace("//", "/")
        if p and not p.endswith("/"):
            p = p + "/"
        self.prefix = p


class S3Manager:
    """
    Wrapper simple sobre boto3 para S3/MinIO, con helpers que usa tu backend.
    """
    # Exponer excepciones para captura externa
    ClientError = _BotoClientError
    EndpointError = _BotoEndpointError

    def __init__(self, cfg: S3Config):
        self.cfg = cfg

        # SSL según flag insecure
        use_ssl = not bool(cfg.insecure)
        endpoint_url = cfg.endpoint or os.getenv("S3_ENDPOINT")

        # Addressing path-style ayuda mucho con MinIO detrás de proxies
        bc = BotoConfig(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            retries={"max_attempts": 5, "mode": "standard"},
        )

        session = boto3.session.Session()
        self.client = session.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=cfg.region,
            aws_access_key_id=cfg.access_key or os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=cfg.secret_key or os.getenv("AWS_SECRET_ACCESS_KEY"),
            use_ssl=use_ssl,
            verify=True if use_ssl else False,  # si es http, desactiva verify
            config=bc,
        )

    # ------------------ Normalización de keys ------------------

    @staticmethod
    def _norm(*parts: str) -> str:
        """
        Une partes con "/" y limpia dobles "//".
        No fuerza slash final (eso lo hacen helpers específicos).
        """
        items: List[str] = []
        for p in parts:
            if p is None:
                continue
            s = str(p).strip()
            if not s:
                continue
            items.append(s)
        key = "/".join(items)
        while "//" in key:
            key = key.replace("//", "/")
        return key

    def join_key(self, prefix: Optional[str], *parts: str) -> str:
        """
        Une respetando "/" y evitando dobles "//".
        Si prefix es None/"" usa sólo parts. No fuerza slash final.
        """
        if prefix is None or prefix == "":
            return self._norm(*parts)
        return self._norm(prefix, *parts)

    # ------------------ Bucket ------------------

    def ensure_bucket(self) -> None:
        """
        Crea el bucket si no existe. Para MinIO/region-less no hace falta LocationConstraint.
        """
        try:
            self.client.head_bucket(Bucket=self.cfg.bucket)
            return
        except self.ClientError as e:
            code = int(e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
            if code not in (404, 301, 400):
                # Errores de auth/permiso, etc.
                raise

        # Intentar crear el bucket (maneja carrera si otro proceso lo crea)
        if self.cfg.region and self.cfg.region.lower() != "us-east-1":
            self.client.create_bucket(
                Bucket=self.cfg.bucket,
                CreateBucketConfiguration={"LocationConstraint": self.cfg.region},
            )
        else:
            self.client.create_bucket(Bucket=self.cfg.bucket)

    # ------------------ IO primitivas ------------------

    def upload_file(self, local_path: str, key: str) -> None:
        k = self._norm(key)
        self.client.upload_file(local_path, self.cfg.bucket, k)

    def upload_bytes(self, data: bytes, key: str, content_type: Optional[str] = None) -> None:
        k = self._norm(key)
        extra: Dict[str, Any] = {}
        if content_type:
            extra["ContentType"] = content_type
        self.client.put_object(Bucket=self.cfg.bucket, Key=k, Body=data, **extra)

    def download_bytes(self, key: str) -> bytes:
        k = self._norm(key)
        obj = self.client.get_object(Bucket=self.cfg.bucket, Key=k)
        return obj["Body"].read()

    def exists(self, key: str) -> bool:
        k = self._norm(key)
        try:
            self.client.head_object(Bucket=self.cfg.bucket, Key=k)
            return True
        except self.ClientError as e:
            status = int(e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
            if status == 404:
                return False
            code = (e.response or {}).get("Error", {}).get("Code")
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    # ------------------ Listado ------------------

    def list(self, prefix: Optional[str] = None) -> List[str]:
        """
        Lista recursivamente todos los objetos bajo `prefix`.
        Devuelve lista de keys (strings). Si no hay resultados, devuelve [].
        """
        pfx = self._norm(prefix or self.cfg.prefix or "")
        if pfx and not pfx.endswith("/"):
            pfx = pfx + "/"

        paginator = self.client.get_paginator("list_objects_v2")
        keys: List[str] = []
        for page in paginator.paginate(Bucket=self.cfg.bucket, Prefix=pfx):
            contents = page.get("Contents")
            if not contents:
                continue
            for it in contents:
                key = it.get("Key")
                if key:
                    keys.append(key)
        return keys

    def list_meta(self, prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Igual que list(), pero con metadatos: [{Key, Size, LastModified, ETag, StorageClass}, ...]
        Útil si quieres poblar tabla con tamaño/fecha en el front.
        """
        pfx = self._norm(prefix or self.cfg.prefix or "")
        if pfx and not pfx.endswith("/"):
            pfx = pfx + "/"

        paginator = self.client.get_paginator("list_objects_v2")
        out: List[Dict[str, Any]] = []
        for page in paginator.paginate(Bucket=self.cfg.bucket, Prefix=pfx):
            contents = page.get("Contents")
            if not contents:
                continue
            out.extend(contents)
        return out

    # ------------------ Pre-signed ------------------

    def presign_get(self, key: str, expires: int = 3600) -> str:
        """
        Genera URL presignada GET para descargar un objeto.
        """
        k = self._norm(key)
        return self.client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": self.cfg.bucket, "Key": k},
            ExpiresIn=int(expires),
        )
