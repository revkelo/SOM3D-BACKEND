#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, List, Dict

import boto3
from botocore.config import Config as _S3Config
from botocore.exceptions import ClientError as _S3ClientError, EndpointConnectionError as _S3EndpointError


@dataclass
class S3Config:
    endpoint: Optional[str]
    insecure: bool
    bucket: str
    prefix: str
    region: str
    access_key: Optional[str]
    secret_key: Optional[str]


class S3Manager:
    """
    Envoltorio simple para S3/MinIO:
    - ensure_bucket()
    - upload_file(local_path, key)
    - join_key(*parts)
    - list_objects(prefix)  (útil para debug)
    - presign_get_object(key, expires_seconds=3600) (opcional)
    """
    def __init__(self, cfg: Optional[S3Config] = None) -> None:
        cfg = cfg or self._from_env()
        self.cfg = cfg
        session = boto3.session.Session(
            aws_access_key_id=cfg.access_key,
            aws_secret_access_key=cfg.secret_key,
            region_name=cfg.region or "us-east-1",
        )
        s3_extra = {"addressing_style": "path"} if cfg.endpoint else {}
        self.client = session.client(
            "s3",
            endpoint_url=cfg.endpoint,   # None → AWS; URL → MinIO
            config=_S3Config(signature_version="s3v4", s3=s3_extra),
            verify=not cfg.insecure,
        )

    # ---------- Utils ----------
    @staticmethod
    def _from_env() -> S3Config:
        return S3Config(
            endpoint=os.getenv("S3_ENDPOINT"),              # ej: http://127.0.0.1:9000
            insecure=bool(os.getenv("S3_INSECURE")),        # "1" → True
            bucket=os.getenv("S3_BUCKET", "som3d"),
            prefix=os.getenv("S3_PREFIX", "jobs/"),
            region=os.getenv("AWS_REGION", "us-east-1"),
            access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )

    @staticmethod
    def join_key(*parts: str) -> str:
        cleaned = [p.strip("/") for p in parts if p]
        return "/".join(cleaned)

    # ---------- Ops ----------
    def ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.cfg.bucket)
            return
        except _S3ClientError as e:
            code = (e.response.get("Error", {}) or {}).get("Code")
            if code in ("404", "NotFound", "NoSuchBucket"):
                # En AWS, buckets fuera de us-east-1 requieren LocationConstraint
                if not self.cfg.endpoint and (self.cfg.region or "us-east-1") != "us-east-1":
                    self.client.create_bucket(
                        Bucket=self.cfg.bucket,
                        CreateBucketConfiguration={"LocationConstraint": self.cfg.region or "us-east-1"},
                    )
                else:
                    self.client.create_bucket(Bucket=self.cfg.bucket)
            else:
                raise

    def upload_file(self, local_path: str, key: str) -> None:
        self.client.upload_file(local_path, self.cfg.bucket, key)

    def list_objects(self, prefix: Optional[str] = None) -> List[Dict]:
        p = prefix.strip("/") if prefix else ""
        out: List[Dict] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.cfg.bucket, Prefix=p):
            for it in page.get("Contents", []):
                out.append(it)
        return out

    def presign_get_object(self, key: str, expires_seconds: int = 3600) -> str:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.cfg.bucket, "Key": key},
            ExpiresIn=expires_seconds,
        )

    # ---------- Errores típicos a exponer si quieres loguear con detalle ----------
    ClientError = _S3ClientError
    EndpointError = _S3EndpointError
