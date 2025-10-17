from __future__ import annotations

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

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse

from app.services.s3_manager import S3Manager, S3Config
from app.services.som3d.totalsegmentator import TotalSegmentatorRunner
from app.services.som3d.generator import NiftiToSTLConverter
from sqlalchemy.orm import Session
from ..db import get_db
from ..core.security import get_current_user, get_current_user_optional
from ..models import JobConv, JobSTL, Paciente, Medico
from ..schemas import FinalizeJobIn, JobSTLOut
from ..core.config import mysql_url


router = APIRouter(prefix="/som3d", tags=["som3d"])


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


S3: Optional[S3Manager] = None
DEFAULT_PREFIX = os.getenv("S3_PREFIX", "jobs/")
RESULT_SUBDIR = "stls"
LOG_KEY = "logs/job.log"
MANIFEST_KEY = "manifest.json"
INPUT_ZIP = "input.zip"
RESULT_ZIP = "stl_result.zip"


def _ensure_s3() -> S3Manager:
    global S3
    if S3 is None:
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
        except Exception:
            pass
    return S3


def _s3_job_base(job_id: str) -> str:
    s3 = _ensure_s3()
    return s3.join_key(DEFAULT_PREFIX, job_id)


def _s3_key(job_id: str, *parts: str) -> str:
    s3 = _ensure_s3()
    base = _s3_job_base(job_id)
    return s3.join_key(base, *parts)


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
    s3 = _ensure_s3()
    key = _s3_key(job_id, MANIFEST_KEY)
    try:
        data = s3.download_bytes(key)
        obj = json.loads(data.decode("utf-8"))
        return JobManifest(**{"job_id": job_id, **obj})
    except Exception:
        return None


def _save_manifest(m: JobManifest) -> None:
    s3 = _ensure_s3()
    key = _s3_key(m.job_id, MANIFEST_KEY)
    payload = json.dumps(m.to_dict(), ensure_ascii=False, indent=2).encode("utf-8")
    s3.upload_bytes(payload, key)


def _append_log(job_id: str, line: str, manifest: Optional[JobManifest] = None):
    s3 = _ensure_s3()
    log_key = _s3_key(job_id, LOG_KEY)
    try:
        existing = s3.download_bytes(log_key)
    except Exception:
        existing = b""
    new = existing + (line.rstrip() + "\n").encode("utf-8")
    s3.upload_bytes(new, log_key)
    if manifest:
        tail = (manifest.log_tail or [])
        tail.append(line.rstrip())
        if len(tail) > 200:
            tail = tail[-200:]
        manifest.log_tail = tail
        manifest.updated_at = _now_ts()
        _save_manifest(manifest)


def _worker_entry(job_id: str, s3_cfg: Dict[str, Any], params: Dict[str, Any], db_url: Optional[str] = None):
    from pathlib import Path
    import tempfile
    from sqlalchemy import create_engine, text as _sql_text

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
        log_key = s3_key(LOG_KEY)
        try:
            existing = s3.download_bytes(log_key)
        except Exception:
            existing = b""
        s3.upload_bytes(existing + (line + "\n").encode("utf-8"), log_key)
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

    def _update_jobconv(status: str, set_finished: bool = False):
        if not db_url:
            return
        try:
            eng = create_engine(db_url, pool_pre_ping=True, future=True)
            with eng.begin() as conn:
                if set_finished:
                    conn.execute(_sql_text(
                        "UPDATE JobConv SET status=:s, finished_at=NOW(), updated_at=NOW() WHERE job_id=:jid"
                    ), {"s": status, "jid": job_id})
                else:
                    conn.execute(_sql_text(
                        "UPDATE JobConv SET status=:s, updated_at=NOW() WHERE job_id=:jid"
                    ), {"s": status, "jid": job_id})
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
        _update_jobconv("ERROR", set_finished=True)

    try:
        write_manifest_patch({"status": "running", "phase": "totalsegmentator"})
        with tempfile.TemporaryDirectory(prefix=f"som3d_job_{job_id}_") as td:
            tdir = Path(td)
            ts_out = tdir / "TS_OUT"
            stl_out = tdir / "STL_OUT"
            ts_out.mkdir(parents=True, exist_ok=True)
            stl_out.mkdir(parents=True, exist_ok=True)

            input_key = s3_key(INPUT_ZIP)
            input_zip = tdir / "input.zip"
            input_zip.write_bytes(s3.download_bytes(input_key))
            wlog("ZIP de entrada listo.")

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

            try:
                wlog("Convirtiendo NIfTI a STL ...")
                conv = NiftiToSTLConverter(progress_cb=lambda s: wlog(str(s)))
                nifti_root = _collect_nifti_root(ts_out)
                conv.convert_folder(nifti_root, stl_out, recursive=True)
                wlog("Conversión a STL finalizada.")
                write_manifest_patch({"phase": "zipping", "percent": 90.0})
            except Exception as e:
                save_worker_error("convert_to_stl", e)
                return

            try:
                zip_bytes = _zip_dir_to_bytes(stl_out)
                s3.upload_bytes(zip_bytes, s3_key(RESULT_ZIP), content_type="application/zip")
                wlog("ZIP subido a S3.")
            except Exception as e:
                save_worker_error("zip_upload", e)
                return

            keys = s3.list(s3_key(RESULT_SUBDIR)) or []
            write_manifest_patch({
                "status": "done",
                "phase": "finished",
                "percent": 100.0,
                "error": None,
                "metrics": {"stl_count": int(len([k for k in keys if k.lower().endswith('.stl')]))},
                "s3_keys_results": keys,
            })
            wlog("Job terminado.")
            _update_jobconv("DONE", set_finished=True)
    except Exception as e:
        save_worker_error("fatal", e)


@router.post("/jobs")
async def create_job(
    file: UploadFile = File(..., description=".zip con DICOMs"),
    enable_ortopedia: bool = Form(True),
    enable_appendicular: bool = Form(False),
    enable_muscles: bool = Form(False),
    enable_hip_implant: bool = Form(False),
    teeth: bool = Form(False),
    cranio: bool = Form(False),
    extra_tasks: Optional[str] = Form(None),
    id_paciente: Optional[int] = Form(None),
    db: Session = Depends(get_db),
    user = Depends(get_current_user),
):
    # Validación rápida de variables S3 para errores claros
    _require_s3_env()
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Se espera un .zip con DICOMs")

    s3 = _ensure_s3()
    job_id = uuid.uuid4().hex
    s3_prefix = _s3_job_base(job_id)

    input_key = _s3_key(job_id, INPUT_ZIP)
    tmp = Path(tempfile.mkdtemp(prefix=f"upload_{job_id}_")) / file.filename
    try:
        with tmp.open("wb") as f:
            while True:
                chunk = await file.read(2 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        s3.upload_file(str(tmp), input_key)
    finally:
        try:
            shutil.rmtree(tmp.parent, ignore_errors=True)
        except Exception:
            pass

    extra_list: List[str] = []
    if extra_tasks:
        extra_list += [t.strip() for t in extra_tasks.split(",") if t.strip()]
    if teeth:
        extra_list.append("teeth")
    if cranio:
        extra_list.append("craniofacial_structures")
    seen = set()
    extra_list = [t for t in extra_list if not (t in seen or seen.add(t))]

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
        bucket=s3.cfg.bucket,
        s3_prefix=s3_prefix,
        s3_keys_results=[],
    )
    _save_manifest(manifest)
    _append_log(job_id, "Job creado. Lanzando proceso...", manifest)

    s3_cfg = dict(
        endpoint=s3.cfg.endpoint,
        insecure=s3.cfg.insecure,
        bucket=s3.cfg.bucket,
        prefix=s3.cfg.prefix,
        region=s3.cfg.region,
        access_key=s3.cfg.access_key,
        secret_key=s3.cfg.secret_key,
    )
    db_url = None
    try:
        db_url = mysql_url()
    except Exception:
        db_url = None
    proc = Process(target=_worker_entry, args=(job_id, s3_cfg, manifest.params, db_url), daemon=True)
    proc.start()
    _append_log(job_id, f"PID worker: {proc.pid}")

    # Persistencia opcional en BD (JobConv) si tenemos usuario autenticado
    try:
        jc = JobConv(
            job_id=job_id,
            id_usuario=user.id_usuario,
            status="RUNNING",
            started_at=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            enable_ortopedia=bool(enable_ortopedia),
            enable_appendicular=bool(enable_appendicular),
            enable_muscles=bool(enable_muscles),
            enable_skull=bool(cranio),
            enable_teeth=bool(teeth),
            enable_hip_implant=bool(enable_hip_implant),
            extra_tasks_json=(json.dumps(extra_list, ensure_ascii=False) if extra_list else None),
        )
        db.add(jc)
        db.commit()
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass

    return JSONResponse(content=manifest.to_dict(), status_code=201)


@router.get("/jobs")
async def list_jobs():
    s3 = _ensure_s3()
    try:
        keys = s3.list(DEFAULT_PREFIX)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"List S3 error: {e}")

    manifests: List[Dict[str, Any]] = []
    for k in keys:
        if not k.endswith("/" + MANIFEST_KEY):
            continue
        try:
            parts = k.strip("/").split("/")
            job_id = parts[-2]
            m = _load_manifest(job_id)
            if m:
                manifests.append(m.to_dict())
        except Exception:
            continue

    return {"jobs": manifests}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    m = _load_manifest(job_id)
    if not m:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    return m.to_dict()


@router.get("/jobs/{job_id}/progress")
async def get_progress(job_id: str):
    m = _load_manifest(job_id)
    if not m:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    return {"status": m.status, "phase": m.phase, "percent": m.percent, "error": m.error}


@router.get("/jobs/{job_id}/log")
async def get_log(job_id: str, tail: int = 200):
    s3 = _ensure_s3()
    log_key = _s3_key(job_id, LOG_KEY)
    try:
        data = s3.download_bytes(log_key)
        lines = data.decode("utf-8", errors="ignore").splitlines()
        if tail and tail > 0:
            lines = lines[-int(tail):]
        return {"job_id": job_id, "lines": lines}
    except Exception:
        return {"job_id": job_id, "lines": []}


@router.get("/jobs/{job_id}/stls")
async def get_stls(job_id: str):
    s3 = _ensure_s3()
    base1 = s3.join_key(s3.cfg.prefix or "", job_id)
    base2 = s3.join_key(base1, RESULT_SUBDIR)
    if not base1.endswith("/"):
        base1 += "/"
    if not base2.endswith("/"):
        base2 += "/"
    try:
        keys1 = s3.list(base1)
        keys2 = s3.list(base2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 list error: {e}")
    all_keys = (keys1 or []) + (keys2 or [])
    stl_keys = sorted([k for k in all_keys if k.lower().endswith(".stl")])
    if not stl_keys:
        raise HTTPException(status_code=404, detail={
            "reason": "stl_not_found",
            "checked_prefixes": [base1, base2],
            "message": "No se encontraron STL para este job"
        })
    items = [{"filename": k.rsplit("/", 1)[-1], "s3_key": k} for k in stl_keys]
    return {
        "job_id": job_id,
        "source": "s3",
        "bucket": s3.cfg.bucket,
        "s3_prefix": base1.rstrip("/"),
        "count": len(items),
        "items": items,
    }


@router.get("/jobs/{job_id}/result")
async def get_result(job_id: str, expires: int = 3600):
    s3 = _ensure_s3()
    m = _load_manifest(job_id)
    if not m:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    if m.status != "done":
        detail = m.error or "Resultado no disponible aún"
        raise HTTPException(status_code=409, detail=detail)
    key = _s3_key(job_id, RESULT_ZIP)
    if not s3.exists(key):
        raise HTTPException(status_code=404, detail="ZIP no encontrado en S3")
    try:
        url = s3.presign_get(key, expires=expires)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Presign error: {e}")
    return {"job_id": job_id, "url": url, "expires_in": expires}


@router.post("/jobs/{job_id}/finalize", response_model=JobSTLOut)
async def finalize_job(job_id: str, payload: FinalizeJobIn, db: Session = Depends(get_db), user = Depends(get_current_user)):
    # Optionally mark JobConv as DONE and create JobSTL linked to a Paciente
    # Validate paciente ownership if medico
    med = db.query(Medico).filter(Medico.id_usuario == user.id_usuario).first()
    if med is not None:
        p = db.query(Paciente).filter(Paciente.id_paciente == payload.id_paciente).first()
        if not p or p.id_medico != med.id_medico:
            raise HTTPException(status_code=403, detail="Paciente no pertenece al medico")

    # Update JobConv status if exists
    try:
        jc = db.query(JobConv).filter(JobConv.job_id == job_id).first()
        if jc:
            jc.status = "DONE"
            jc.finished_at = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            db.add(jc)
    except Exception:
        pass

    js = JobSTL(
        job_id=job_id,
        id_paciente=payload.id_paciente,
        stl_size=payload.stl_size,
        num_stl_archivos=payload.num_stl_archivos,
    )
    db.add(js)
    db.commit()
    db.refresh(js)
    return js


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    import psutil
    s3 = _ensure_s3()
    m = _load_manifest(job_id)
    if not m:
        return {"status": "not_found"}

    pid_to_kill = None
    try:
        log_key = _s3_key(job_id, LOG_KEY)
        data = s3.download_bytes(log_key)
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

    m.status = "canceled"
    m.phase = "canceled"
    m.percent = min(m.percent or 0.0, 99.0)
    m.error = "Cancelado por el usuario"
    (m.log_tail or []).append("Job cancelado por el usuario.")
    m.updated_at = _now_ts()
    _save_manifest(m)
    _append_log(job_id, "Job cancelado.", m)

    return m.to_dict()
def _require_s3_env():
    ak = os.getenv("AWS_ACCESS_KEY_ID")
    sk = os.getenv("AWS_SECRET_ACCESS_KEY")
    endpoint = os.getenv("S3_ENDPOINT")
    bucket = os.getenv("S3_BUCKET")
    # Permitir cadena de credenciales de AWS (perfiles) si está configurada
    profile = os.getenv("AWS_PROFILE") or os.getenv("AWS_DEFAULT_PROFILE") or os.getenv("AWS_SHARED_CREDENTIALS_FILE")
    if (not ak or not sk) and not profile:
        raise HTTPException(status_code=500, detail={
            "reason": "s3_credentials_missing",
            "message": "Faltan credenciales S3. Define AWS_ACCESS_KEY_ID y AWS_SECRET_ACCESS_KEY en tu entorno o .env",
        })
    if not bucket:
        raise HTTPException(status_code=500, detail={
            "reason": "s3_bucket_missing",
            "message": "Define S3_BUCKET en tu entorno o .env",
        })
    # endpoint puede omitirse si usas AWS S3 estándar; si usas MinIO, debe estar
    insecure = os.getenv("S3_INSECURE")
    if endpoint is None and os.getenv("AWS_ENDPOINT_URL_S3") is None:
        # Sin endpoint explícito: asumimos AWS S3; no forzamos error
        return
    # endpoint presente, ok
    return
