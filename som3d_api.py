#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SOM3D Backend 100% S3 (stateless)
- Entrada ZIP -> S3
- Manifest/estado/log -> S3
- STL/ZIP resultados -> S3
- Endpoints leen SIEMPRE de S3 (sin memoria persistente local)
"""
from __future__ import annotations

import asyncio
import io
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, asdict
from multiprocessing import Process
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import traceback

import psutil
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# ==== Importa tus clases de negocio ====
try:
    from ClaseTotalsegmentor import TotalSegmentatorRunner  # type: ignore
except Exception as e:
    raise RuntimeError("No se pudo importar TotalSegmentatorRunner desde ClaseTotalsegmentor.py") from e

try:
    from ClaseGenerator import NiftiToSTLConverter  # type: ignore
except Exception as e:
    raise RuntimeError("No se pudo importar NiftiToSTLConverter desde ClaseGenerator.py") from e

# ==== S3 manager aislado ====
from s3_manager import S3Manager, S3Config  # asegúrate del nombre de archivo


# ==========================
# Carga .env (si existe)
# ==========================
dotenv_path = find_dotenv(usecwd=True)
if dotenv_path:
    load_dotenv(dotenv_path, override=False)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


# ==========================
# App
# ==========================
app = FastAPI(
    title="SOM3D Backend S3-only",
    version="4.0.0",
)

# Instancia de S3 (una sola, para toda la app)
S3: Optional[S3Manager] = None

DEFAULT_PREFIX = os.getenv("S3_PREFIX", "jobs/")
RESULT_SUBDIR = "stls"
LOG_KEY = "logs/job.log"
MANIFEST_KEY = "manifest.json"
INPUT_ZIP = "input.zip"
RESULT_ZIP = "stl_result.zip"


@app.on_event("startup")
async def _on_startup():
    global S3
    S3 = S3Manager(S3Config(
        endpoint=os.getenv("S3_ENDPOINT"),
        insecure=_env_bool("S3_INSECURE", False),
        bucket=os.getenv("S3_BUCKET", "som3d"),
        prefix=os.getenv("S3_PREFIX", DEFAULT_PREFIX),
        region=os.getenv("AWS_REGION", "us-east-1"),
        access_key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    ))
    try:
        S3.ensure_bucket()
    except Exception as e:
        print(f"[WARN] No se pudo verificar/crear bucket en startup: {e}")


# ==========================
# Utils S3 (manifest / logs)
# ==========================
def _s3_job_base(job_id: str) -> str:
    return S3.join_key(DEFAULT_PREFIX, job_id)  # type: ignore


def _s3_key(job_id: str, *parts: str) -> str:
    base = _s3_job_base(job_id)
    return S3.join_key(base, *parts)  # type: ignore


def _now_ts() -> float:
    return time.time()


def _zip_dir_to_bytes(dir_path: Path) -> bytes:
    mem = io.BytesIO()
    with tempfile.TemporaryDirectory(prefix="stl_zip_") as td:
        temp_zip_path = Path(td) / "result_stl.zip"
        shutil.make_archive(str(temp_zip_path.with_suffix('')), 'zip', root_dir=dir_path)
        data = temp_zip_path.read_bytes()
        mem.write(data)
    mem.seek(0)
    return mem.getvalue()


def _collect_nifti_root(ts_out_dir: Path) -> Path:
    if ts_out_dir.exists() and any(ts_out_dir.rglob("*.nii.gz")):
        return ts_out_dir
    for sub in ts_out_dir.iterdir():
        if sub.is_dir() and any(sub.rglob("*.nii.gz")):
            return sub
    return ts_out_dir


# ==========================
# Modelo Job + Manifest (S3)
# ==========================
@dataclass
class JobManifest:
    job_id: str
    created_at: float
    updated_at: float
    status: str  # queued|running|done|error|canceled
    phase: str   # queued|totalsegmentator|converting|zipping|finished|error|canceled
    percent: float
    params: Dict[str, Any]
    metrics: Dict[str, Any]
    error: Optional[str] = None
    log_tail: List[str] = None  # pequeño tail
    bucket: Optional[str] = None
    s3_prefix: Optional[str] = None
    s3_keys_results: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["percent"] = round(float(self.percent), 1)
        return d


def _load_manifest(job_id: str) -> Optional[JobManifest]:
    key = _s3_key(job_id, MANIFEST_KEY)
    try:
        data = S3.download_bytes(key)  # type: ignore
        obj = json.loads(data.decode("utf-8"))
        return JobManifest(**{
            "job_id": job_id,
            **obj
        })
    except Exception:
        return None


def _save_manifest(m: JobManifest) -> None:
    key = _s3_key(m.job_id, MANIFEST_KEY)
    payload = json.dumps(m.to_dict(), ensure_ascii=False, indent=2).encode("utf-8")
    S3.upload_bytes(payload, key)  # type: ignore


def _append_log(job_id: str, line: str, manifest: Optional[JobManifest] = None):
    # Subimos log completo (append “ingenuo”: descargamos, concatenamos, subimos)
    log_key = _s3_key(job_id, LOG_KEY)
    try:
        existing = S3.download_bytes(log_key)  # type: ignore
    except Exception:
        existing = b""
    new = existing + (line.rstrip() + "\n").encode("utf-8")
    S3.upload_bytes(new, log_key)  # type: ignore
    if manifest:
        tail = (manifest.log_tail or [])
        tail.append(line.rstrip())
        if len(tail) > 200:
            tail = tail[-200:]
        manifest.log_tail = tail
        manifest.updated_at = _now_ts()
        _save_manifest(manifest)


# ==========================
# Proceso worker (sube TODO a S3)
# ==========================
def _worker_entry(
    job_id: str,
    s3_cfg: Dict[str, Any],
    params: Dict[str, Any],
):
    """
    - Descarga/lee input.zip desde S3 si hace falta (ya subido por create_job).
    - Ejecuta TS + conversión a STL en temp local.
    - Sube STLs/ZIP a S3.
    - Actualiza manifest/log en S3.
    """
    from pathlib import Path
    import tempfile
    import json
    import traceback

    # Re-crear cliente S3 dentro del proceso
    s3 = S3Manager(S3Config(**s3_cfg))

    def s3_key(*parts: str) -> str:
        base = s3.join_key(s3_cfg.get("prefix") or DEFAULT_PREFIX, job_id)
        return s3.join_key(base, *parts)

    def write_manifest_patch(patch: Dict[str, Any]):
        key = s3_key(MANIFEST_KEY)
        try:
            current = json.loads(s3.download_bytes(key).decode("utf-8"))
        except Exception:
            current = {}
        current.update(patch)
        if "percent" in current:
            current["percent"] = float(current["percent"])
        current["updated_at"] = _now_ts()
        s3.upload_bytes(json.dumps(current, ensure_ascii=False, indent=2).encode("utf-8"), key)

    def wlog(msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        # append en S3 + tail en manifest
        # Descargar, concatenar y subir (simple, robusto)
        log_key = s3_key(LOG_KEY)
        try:
            existing = s3.download_bytes(log_key)
        except Exception:
            existing = b""
        s3.upload_bytes(existing + (line + "\n").encode("utf-8"), log_key)
        # actualizar tail en manifest
        try:
            key = s3_key(MANIFEST_KEY)
            try:
                mani = json.loads(s3.download_bytes(key).decode("utf-8"))
            except Exception:
                mani = {}
            tail = mani.get("log_tail") or []
            tail.append(line)
            if len(tail) > 200:
                tail = tail[-200:]
            mani["log_tail"] = tail
            mani["updated_at"] = _now_ts()
            s3.upload_bytes(json.dumps(mani, ensure_ascii=False, indent=2).encode("utf-8"), key)
        except Exception:
            pass

    def save_worker_error(phase_name: str, exc: BaseException):
        wlog(f"ERROR en fase '{phase_name}': {type(exc).__name__}: {exc}")
        write_manifest_patch({
            "status": "error",
            "phase": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "percent": 99.0
        })

    try:
        write_manifest_patch({"status": "running", "phase": "totalsegmentator"})
        with tempfile.TemporaryDirectory(prefix=f"som3d_job_{job_id}_") as td:
            tdir = Path(td)
            ts_out = tdir / "TS_OUT"
            stl_out = tdir / "STL_OUT"
            ts_out.mkdir(parents=True, exist_ok=True)
            stl_out.mkdir(parents=True, exist_ok=True)

            # Input.zip ya fue subido por create_job
            input_key = s3_key(INPUT_ZIP)
            input_zip = tdir / "input.zip"
            input_zip.write_bytes(s3.download_bytes(input_key))
            wlog("ZIP de entrada listo.")

            # 1) TotalSegmentator
            try:
                wlog("Iniciando TotalSegmentator ...")
                runner = TotalSegmentatorRunner(
                    robust_import=True,
                    enable_ortopedia=bool(params.get("enable_ortopedia", True)),
                    enable_appendicular=bool(params.get("enable_appendicular", False)),
                    enable_muscles=bool(params.get("enable_muscles", False)),
                    enable_hip_implant=bool(params.get("enable_hip_implant", False)),
                    extra_tasks=list(params.get("extra_tasks", [])),
                    on_log=lambda s: wlog(str(s)),
                )
                runner.run(input_zip, ts_out)
                wlog("TotalSegmentator finalizado.")
                write_manifest_patch({"phase": "converting", "percent": 70.0})
            except Exception as e:
                save_worker_error("totalsegmentator", e)
                return

            # 2) NIfTI → STL
            try:
                wlog("Convirtiendo NIfTI a STL ...")
                conv = NiftiToSTLConverter(progress_cb=lambda s: wlog(str(s)))
                nifti_root = _collect_nifti_root(ts_out)
                try:
                    conv.convert_folder(
                        input_dir=nifti_root,
                        output_dir=stl_out,
                        recursive=True,
                        clip_min=None,
                        clip_max=None,
                        exclude_zeros=True,
                        min_voxels=10,
                        log_name=None,
                    )
                except TypeError:
                    conv.convert_folder(nifti_root, stl_out, True, None, None, True, 10, None)
                wlog("Conversión a STL finalizada.")
                write_manifest_patch({"phase": "zipping", "percent": 97.0})
            except Exception as e:
                save_worker_error("convert_to_stl", e)
                return

            # 3) Subir STLs a S3 (HQ si existe)
            try:
                hq_dir = stl_out / "hq_suavizado_decimado"
                uploaded_keys: List[str] = []
                target_dir = hq_dir if hq_dir.exists() else stl_out
                for p in sorted(target_dir.rglob("*.stl")):
                    rel_name = p.name
                    key = s3_key(RESULT_SUBDIR, rel_name)
                    s3.upload_file(str(p), key)
                    uploaded_keys.append(key)
                if uploaded_keys:
                    wlog(f"STLs subidos ({len(uploaded_keys)}).")
                write_manifest_patch({"s3_keys_results": uploaded_keys})
            except Exception as e:
                save_worker_error("upload_stls", e)
                return

            # 4) Empaquetar ZIP y subir
            try:
                data = _zip_dir_to_bytes(target_dir)  # type: ignore
                s3.upload_bytes(data, s3_key(RESULT_ZIP))
                wlog("ZIP final subido.")
            except Exception as e:
                save_worker_error("zip_result", e)
                return

            # 5) Finalizar
            write_manifest_patch({
                "status": "done",
                "phase": "finished",
                "percent": 100.0,
                "error": None
            })
            wlog("Job finalizado correctamente.")
    except Exception as e:
        # Falla inesperada
        save_worker_error("unexpected", e)


# ==========================
# Endpoints
# ==========================
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/jobs")
async def create_job(
    file: UploadFile = File(..., description=".zip con DICOMs"),
    enable_ortopedia: bool = Form(True),
    enable_appendicular: bool = Form(False),
    enable_muscles: bool = Form(False),
    enable_hip_implant: bool = Form(False),
    teeth: bool = Form(False),
    cranio: bool = Form(False),
    extra_tasks: Optional[str] = Form(None),
):
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Se espera un .zip con DICOMs")

    job_id = uuid.uuid4().hex
    s3_prefix = _s3_job_base(job_id)

    # Subir input.zip DIRECTO a S3 (streamed)
    input_key = _s3_key(job_id, INPUT_ZIP)
    tmp = Path(tempfile.mkdtemp(prefix=f"upload_{job_id}_")) / file.filename
    try:
        with tmp.open("wb") as f:
            while True:
                chunk = await file.read(2 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        S3.upload_file(str(tmp), input_key)  # type: ignore
    finally:
        try:
            shutil.rmtree(tmp.parent, ignore_errors=True)
        except Exception:
            pass

    # Construir lista final de tasks extra
    extra_list: List[str] = []
    if extra_tasks:
        extra_list += [t.strip() for t in extra_tasks.split(",") if t.strip()]
    if teeth:
        extra_list.append("teeth")
    if cranio:
        extra_list.append("craniofacial_structures")
    # dedupe manteniendo orden
    seen = set()
    extra_list = [t for t in extra_list if not (t in seen or seen.add(t))]

    # Crear manifest inicial en S3
    manifest = JobManifest(
        job_id=job_id,
        created_at=_now_ts(),
        updated_at=_now_ts(),
        status="running",
        phase="totalsegmentator",
        percent=0.0,
        params={
            "enable_ortopedia": enable_ortopedia,
            "enable_appendicular": enable_appendicular,
            "enable_muscles": enable_muscles,
            "enable_hip_implant": enable_hip_implant,
            "extra_tasks": extra_list,
        },
        metrics={},
        error=None,
        log_tail=[],
        bucket=S3.cfg.bucket if S3 else None,  # type: ignore
        s3_prefix=s3_prefix,
        s3_keys_results=[],
    )
    _save_manifest(manifest)
    _append_log(job_id, "Job creado. Lanzando proceso...", manifest)

    # Lanzar worker (stateless: le pasamos S3 config y params)
    s3_cfg = dict(
        endpoint=S3.cfg.endpoint,  # type: ignore
        insecure=S3.cfg.insecure,  # type: ignore
        bucket=S3.cfg.bucket,      # type: ignore
        prefix=S3.cfg.prefix,      # type: ignore
        region=S3.cfg.region,      # type: ignore
        access_key=S3.cfg.access_key,  # type: ignore
        secret_key=S3.cfg.secret_key,  # type: ignore
    )
    proc = Process(
        target=_worker_entry,
        args=(job_id, s3_cfg, manifest.params),
        daemon=True,
    )
    proc.start()

    # Guardamos un pid marker (opcional, solo informativo)
    _append_log(job_id, f"PID worker: {proc.pid}")

    return JSONResponse(content=manifest.to_dict(), status_code=201)


@app.get("/jobs")
async def list_jobs():
    """
    Lista jobs buscando manifest.json bajo el prefijo.
    """
    try:
        keys = S3.list(DEFAULT_PREFIX)  # type: ignore
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"List S3 error: {e}")

    # Filtrar solo manifests
    manifests: List[Dict[str, Any]] = []
    for k in keys:
        if not k.endswith("/" + MANIFEST_KEY):
            continue
        # job_id es la carpeta anterior al manifest
        try:
            parts = k.strip("/").split("/")
            job_id = parts[-2]
            m = _load_manifest(job_id)
            if m:
                manifests.append(m.to_dict())
        except Exception:
            continue

    return {"jobs": manifests}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    m = _load_manifest(job_id)
    if not m:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    return m.to_dict()


@app.get("/jobs/{job_id}/progress")
async def job_progress(job_id: str):
    m = _load_manifest(job_id)
    if not m:
        raise HTTPException(status_code=404, detail="Job no encontrado")

    return {
        "job_id": job_id,
        "status": m.status,
        "phase": m.phase,
        "percent": round(float(m.percent), 1),
        "metrics": m.metrics,
        "message": (m.log_tail[-1] if (m.log_tail) else None),
        "error": m.error,
        "log_tail": (m.log_tail or [])[-200:],
        "s3_bucket": m.bucket,
        "s3_prefix": m.s3_prefix,
        "s3_keys_results": m.s3_keys_results or [],
    }


@app.get("/jobs/{job_id}/log")
async def job_log(job_id: str, tail: int = 200):
    # Descarga del log desde S3
    tail = max(1, min(int(tail), 2000))
    log_key = _s3_key(job_id, LOG_KEY)
    try:
        data = S3.download_bytes(log_key)  # type: ignore
        lines = data.decode("utf-8", errors="ignore").splitlines()
        return {"job_id": job_id, "lines": lines[-tail:]}
    except Exception:
        # fallback: manifest tail
        m = _load_manifest(job_id)
        if not m:
            raise HTTPException(status_code=404, detail="Job no encontrado")
        return {"job_id": job_id, "lines": (m.log_tail or [])[-tail:]}


@app.get("/jobs/{job_id}/stls")
async def job_stls(job_id: str):
    """
    Lista TODOS los .stl bajo el job:
      - {prefix}/{job_id}/**/*.stl
      - {prefix}/{job_id}/stls/**/*.stl
    Busca con el prefix REAL cargado en S3.cfg.prefix para evitar desalineaciones.
    """
    if not S3:
        raise HTTPException(status_code=500, detail="S3 no inicializado")

    # Prefijo base según la config real (NO uses DEFAULT_PREFIX aquí)
    base1 = S3.join_key(S3.cfg.prefix or "", job_id)          # ej: "jobs/94bb...bf2"
    base2 = S3.join_key(base1, RESULT_SUBDIR)                 # ej: "jobs/94bb...bf2/stls"

    # Asegurar barra final para list_objects_v2
    if not base1.endswith("/"):
        base1 += "/"
    if not base2.endswith("/"):
        base2 += "/"

    try:
        keys1 = S3.list(base1)  # recursivo
        keys2 = S3.list(base2)  # recursivo
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 list error: {e}")

    # Unir y filtrar STL (case-insensitive)
    all_keys = (keys1 or []) + (keys2 or [])
    stl_keys = sorted([k for k in all_keys if k.lower().endswith(".stl")])

    if not stl_keys:
        # 404 explícito cuando no hay STL
        raise HTTPException(
            status_code=404,
            detail={
                "reason": "stl_not_found",
                "checked_prefixes": [base1, base2],
                "message": "No se encontraron STL para este job"
            },
        )

    items = [{"filename": k.rsplit("/", 1)[-1], "s3_key": k} for k in stl_keys]

    # bucket y s3_prefix informativos
    return {
        "job_id": job_id,
        "source": "s3",
        "bucket": S3.cfg.bucket,
        "s3_prefix": base1.rstrip("/"),
        "count": len(items),
        "items": items,
    }




@app.get("/jobs/{job_id}/result")
async def get_result(job_id: str, expires: int = 3600):
    """
    Devuelve URL presignada para descargar {job_id}/stl_result.zip desde S3.
    """
    m = _load_manifest(job_id)
    if not m:
        raise HTTPException(status_code=404, detail="Job no encontrado")

    if m.status != "done":
        detail = m.error or "Resultado no disponible aún"
        raise HTTPException(status_code=409, detail=detail)

    key = _s3_key(job_id, RESULT_ZIP)
    if not S3.exists(key):  # type: ignore
        raise HTTPException(status_code=404, detail="ZIP no encontrado en S3")

    try:
        url = S3.presign_get(key, expires=expires)  # type: ignore
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Presign error: {e}")

    return {"job_id": job_id, "url": url, "expires_in": expires}


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """
    Marca el job como cancelado en S3. Intenta matar el proceso si corre en este host (best-effort).
    """
    m = _load_manifest(job_id)
    if not m:
        return {"status": "not_found"}

    # Intento de leer PID del log (si se guardó)
    # Nota: en este diseño stateless no rastreamos PIDs de forma confiable
    # pero hacemos best-effort buscando una línea "PID worker: NNN".
    pid_to_kill = None
    try:
        log_key = _s3_key(job_id, LOG_KEY)
        data = S3.download_bytes(log_key)  # type: ignore
        for line in data.decode("utf-8", errors="ignore").splitlines():
            if "PID worker:" in line:
                try:
                    pid_to_kill = int(line.split("PID worker:")[1].strip())
                except Exception:
                    pass
    except Exception:
        pass

    if pid_to_kill:
        try:
            parent = psutil.Process(pid_to_kill)
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except Exception:
                    pass
            try:
                parent.kill()
            except Exception:
                pass
        except Exception:
            pass

    # Actualizar manifest
    m.status = "canceled"
    m.phase = "canceled"
    m.percent = min(m.percent or 0.0, 99.0)
    m.error = "Cancelado por el usuario"
    (m.log_tail or []).append("Job cancelado por el usuario.")
    m.updated_at = _now_ts()
    _save_manifest(m)
    _append_log(job_id, "Job cancelado.", m)

    return m.to_dict()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("som3d_api:app", host="0.0.0.0", port=8000, reload=False)
