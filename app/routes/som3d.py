from __future__ import annotations
from stl import mesh

import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, asdict
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Optional

import json

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse, Response

from app.services.s3_manager import S3Manager, S3Config
from app.services.som3d.totalsegmentator import TotalSegmentatorRunner
from app.services.som3d.generator import NiftiToSTLConverter
from sqlalchemy.orm import Session
from ..db import get_db
from ..core.security import get_current_user, get_current_user_optional, require_admin
from ..models import JobConv, JobSTL, Paciente, Medico, Usuario, VisorEstado
from ..schemas import FinalizeJobIn, JobSTLOut, PatientJobSTLOut, JobSTLNoteUpdateIn
from ..core.config import mysql_url


router = APIRouter(prefix="/som3d", tags=["som3d"])


def _is_admin(user: Any) -> bool:
    return str(getattr(user, "rol", "")).upper() == "ADMINISTRADOR"


def _is_medico(user: Any) -> bool:
    return str(getattr(user, "rol", "")).upper() == "MEDICO"


def _ensure_job_access(db: Session, user: Any, job_id: str) -> None:
    if _is_admin(user):
        return
    owner = (
        db.query(JobConv.job_id)
        .filter(JobConv.job_id == job_id, JobConv.id_usuario == user.id_usuario)
        .first()
    )
    if not owner:
        raise HTTPException(status_code=404, detail="Job no encontrado")


def _get_jobstl_owned(db: Session, user: Any, id_jobstl: int) -> JobSTL:
    js = db.query(JobSTL).filter(JobSTL.id_jobstl == id_jobstl).first()
    if not js:
        raise HTTPException(status_code=404, detail="Caso 3D no encontrado")
    if _is_admin(user):
        return js
    if js.id_paciente is None:
        raise HTTPException(status_code=403, detail="No autorizado")
    p = db.query(Paciente).filter(Paciente.id_paciente == js.id_paciente).first()
    med = db.query(Medico).filter(Medico.id_usuario == user.id_usuario).first()
    if not p or not med or p.id_medico != med.id_medico:
        raise HTTPException(status_code=403, detail="No autorizado")
    return js


def _delete_job_artifacts_s3(job_id: str) -> int:
    """Elimina todos los objetos del job en S3/MinIO."""
    s3 = _ensure_s3()
    base_key = _s3_job_base(job_id)
    keys = s3.list(base_key)
    for key in keys:
        s3.delete(key)
    try:
        s3.delete(base_key)
    except Exception:
        pass
    return len(keys)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


S3: Optional[S3Manager] = None
DEFAULT_PREFIX = os.getenv("S3_PREFIX", "jobs/")
RESULT_SUBDIR = "stls"
LOG_KEY = "job.log"
MANIFEST_KEY = "manifest.json"
INPUT_ZIP = "input.zip"
RESULT_ZIP = "stl_result.zip"
MAX_UPLOAD_MB = max(1, int(os.getenv("SOM3D_MAX_UPLOAD_MB", "1024")))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
_RE_PROCESS_NAME = re.compile("^[A-Za-z\\u00C0-\\u00FF0-9()_., -]{3,80}$")


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


def _safe_upload_filename(filename: str) -> str:
    raw = str(filename or "").replace("\\", "/")
    base = raw.split("/")[-1].strip().replace("\x00", "")
    if not base or base in {".", ".."}:
        return "upload.zip"
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    if not safe.lower().endswith(".zip"):
        safe += ".zip"
    return safe


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


def _to_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or v == "":
            return default
        return int(float(v))
    except Exception:
        return default


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def _summarize_conversion(conversion_out: Dict[str, Any] | None) -> Dict[str, Any]:
    rows = (conversion_out or {}).get("results") or []
    files_total = len(rows)
    ok_rows = [r for r in rows if bool((r or {}).get("success"))]
    files_ok = len(ok_rows)

    verts_before_total = sum(_to_int((r or {}).get("verts_before")) for r in ok_rows)
    verts_after_total = sum(_to_int((r or {}).get("verts_after")) for r in ok_rows)
    faces_before_total = sum(_to_int((r or {}).get("faces_before")) for r in ok_rows)
    faces_after_total = sum(_to_int((r or {}).get("faces_after")) for r in ok_rows)

    vertices_reduced_total = max(0, verts_before_total - verts_after_total)
    faces_reduced_total = max(0, faces_before_total - faces_after_total)
    vertices_reduction_pct = (
        (vertices_reduced_total / float(verts_before_total)) * 100.0 if verts_before_total > 0 else 0.0
    )
    faces_reduction_pct = (
        (faces_reduced_total / float(faces_before_total)) * 100.0 if faces_before_total > 0 else 0.0
    )

    total_mesh_seconds = sum(_to_float((r or {}).get("seconds")) for r in ok_rows)
    avg_mesh_seconds = (total_mesh_seconds / float(files_ok)) if files_ok > 0 else 0.0

    return {
        "files_total": int(files_total),
        "files_ok": int(files_ok),
        "vertices_before_total": int(verts_before_total),
        "vertices_after_total": int(verts_after_total),
        "vertices_reduced_total": int(vertices_reduced_total),
        "vertices_reduction_pct": round(float(vertices_reduction_pct), 2),
        "faces_before_total": int(faces_before_total),
        "faces_after_total": int(faces_after_total),
        "faces_reduced_total": int(faces_reduced_total),
        "faces_reduction_pct": round(float(faces_reduction_pct), 2),
        "mesh_seconds_total": round(float(total_mesh_seconds), 3),
        "mesh_seconds_avg": round(float(avg_mesh_seconds), 3),
    }


def _probe_gpu_runtime() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "gpu_ready": False,
        "gpu_detected": False,
        "backend": "cpu",
        "post_restart_check": {
            "title": "Reiniciar backend y verificar",
            "command": 'python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"',
            "output": "",
            "ok": False,
            "error": None,
        },
        "python": sys.version.split(" ")[0],
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_smi": {
            "available": False,
            "error": None,
            "devices": [],
        },
        "torch": {
            "installed": False,
            "cuda_available": False,
            "cuda_version": None,
            "device_count": 0,
            "devices": [],
            "error": None,
        },
        "cupy": {
            "installed": False,
            "cuda_available": False,
            "device_count": 0,
            "error": None,
        },
    }

    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=6,
        )
        if smi.returncode == 0:
            rows = [ln.strip() for ln in (smi.stdout or "").splitlines() if ln.strip()]
            devices: list[dict[str, Any]] = []
            for row in rows:
                parts = [p.strip() for p in row.split(",")]
                if len(parts) >= 3:
                    devices.append(
                        {
                            "name": parts[0],
                            "memory_mb": _to_int(parts[1], 0),
                            "driver_version": parts[2],
                        }
                    )
                else:
                    devices.append({"raw": row})
            out["nvidia_smi"]["available"] = bool(devices)
            out["nvidia_smi"]["devices"] = devices
        else:
            err = (smi.stderr or smi.stdout or "").strip()
            out["nvidia_smi"]["error"] = err or f"return_code={smi.returncode}"
    except Exception as ex:
        out["nvidia_smi"]["error"] = str(ex)

    try:
        import torch  # type: ignore

        out["torch"]["installed"] = True
        out["torch"]["cuda_version"] = getattr(torch.version, "cuda", None)
        cuda_ok = bool(torch.cuda.is_available())
        out["torch"]["cuda_available"] = cuda_ok
        if cuda_ok:
            cnt = int(torch.cuda.device_count())
            out["torch"]["device_count"] = cnt
            out["torch"]["devices"] = [torch.cuda.get_device_name(i) for i in range(cnt)]
    except Exception as ex:
        out["torch"]["error"] = str(ex)

    try:
        import cupy as cp  # type: ignore

        out["cupy"]["installed"] = True
        cnt = int(cp.cuda.runtime.getDeviceCount())
        out["cupy"]["device_count"] = cnt
        out["cupy"]["cuda_available"] = cnt > 0
    except Exception as ex:
        out["cupy"]["error"] = str(ex)

    smi_devices = out["nvidia_smi"]["devices"] or []
    torch_cuda = bool(out["torch"]["cuda_available"])
    cupy_cuda = bool(out["cupy"]["cuda_available"])
    out["gpu_detected"] = bool(smi_devices) or torch_cuda or cupy_cuda
    out["gpu_ready"] = torch_cuda or cupy_cuda
    out["backend"] = "cuda" if out["gpu_ready"] else "cpu"

    try:
        chk = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())",
            ],
            capture_output=True,
            text=True,
            timeout=8,
        )
        cmd_out = (chk.stdout or "").strip()
        cmd_err = (chk.stderr or "").strip()
        out["post_restart_check"]["output"] = cmd_out
        out["post_restart_check"]["ok"] = chk.returncode == 0
        out["post_restart_check"]["error"] = cmd_err or (None if chk.returncode == 0 else f"return_code={chk.returncode}")
    except Exception as ex:
        out["post_restart_check"]["ok"] = False
        out["post_restart_check"]["error"] = str(ex)
    return out


@dataclass
class JobManifest:
    job_id: str
    created_at: float
    updated_at: float
    status: str                                      
    phase: str                                                                       
    percent: float
    params: Dict[str, Any]
    metrics: Dict[str, Any]
    error: Optional[str] = None
    log_tail: List[str] = None                
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
    """Agrega una línea de log al manifest (tail pequeño en S3).
    El log completo se guarda por el worker como `job.log` al finalizar o ante error.
    """
    m = manifest or _load_manifest(job_id)
    if not m:
        return
    tail = (m.log_tail or [])
    tail.append(line.rstrip())
    if len(tail) > 200:
        tail = tail[-200:]
    m.log_tail = tail
    m.updated_at = _now_ts()
    _save_manifest(m)


def _worker_entry(job_id: str, s3_cfg: Dict[str, Any], params: Dict[str, Any], db_url: Optional[str] = None, local_input_zip: Optional[str] = None):
    from pathlib import Path
    import tempfile
    from sqlalchemy import create_engine, text as _sql_text

    s3 = S3Manager(S3Config(**s3_cfg))

    log_buffer: list[str] = []

    t_all_start = time.time()
    t_seg_start: Optional[float] = None
    t_conv_start: Optional[float] = None
    t_zip_start: Optional[float] = None
    seg_seconds: Optional[float] = None
    conv_seconds: Optional[float] = None
    zip_seconds: Optional[float] = None
    conversion_summary: Dict[str, Any] = {}

    def _fmt_dur(seconds: Optional[float]) -> str:
        try:
            if seconds is None:
                return "N/A"
            total = float(seconds)
            s = int(round(total))
            m, sec = divmod(s, 60)
            h, m = divmod(m, 60)
            return f"{h:02d}:{m:02d}:{sec:02d} ({total:.2f}s)"
        except Exception:
            return str(seconds)

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
        try:
            log_buffer.append(line)
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
            full = ("\n".join(log_buffer) + "\n").encode("utf-8") if log_buffer else b""
            s3.upload_bytes(full, s3_key(LOG_KEY), content_type="text/plain; charset=utf-8")
        except Exception:
            pass

    try:
        write_manifest_patch({"status": "running", "phase": "totalsegmentator"})
        with tempfile.TemporaryDirectory(prefix=f"som3d_job_{job_id}_") as td:
            tdir = Path(td)
            ts_out = tdir / "TS_OUT"
            stl_out = tdir / "STL_OUT"
            ts_out.mkdir(parents=True, exist_ok=True)
            stl_out.mkdir(parents=True, exist_ok=True)

            input_zip = tdir / "input.zip"
            if local_input_zip and Path(local_input_zip).exists():
                shutil.copy2(local_input_zip, input_zip)
                try:
                    src = Path(local_input_zip)
                    base = src.parent
                    src.unlink(missing_ok=True)
                    try:
                        base.rmdir()
                    except Exception:
                        pass
                except Exception:
                    pass
                wlog("ZIP de entrada copiado desde temporal local.")
            else:
                input_key = s3_key(INPUT_ZIP)
                input_zip.write_bytes(s3.download_bytes(input_key))
                wlog("ZIP de entrada descargado desde S3.")

            try:
                wlog("Iniciando TotalSegmentator ...")
                t_seg_start = time.time()
                runner = TotalSegmentatorRunner(
                    robust_import=True,
                    enable_ortopedia=bool(params.get("enable_ortopedia", False)),
                    enable_appendicular=bool(params.get("enable_appendicular", False)),
                    enable_muscles=bool(params.get("enable_muscles", False)),
                    enable_hip_implant=bool(params.get("enable_hip_implant", False)),
                    extra_tasks=list(params.get("extra_tasks", [])),
                    on_log=lambda s: wlog(str(s)),
                )
                runner.run(input_zip, ts_out)
                seg_seconds = time.time() - (t_seg_start or time.time())
                wlog(f"TotalSegmentator finalizado. Tiempo segmentación: {_fmt_dur(seg_seconds)}")
                write_manifest_patch({"phase": "converting", "percent": 70.0})
            except Exception as e:
                save_worker_error("totalsegmentator", e)
                return

            try:
                wlog("Convirtiendo NIfTI a STL ...")
                t_conv_start = time.time()
                conv = NiftiToSTLConverter(progress_cb=lambda s: wlog(str(s)))
                nifti_root = _collect_nifti_root(ts_out)
                conv_out = conv.convert_folder(nifti_root, stl_out, recursive=True)
                conversion_summary = _summarize_conversion(conv_out)
                conv_seconds = time.time() - (t_conv_start or time.time())
                wlog(f"Conversión a STL finalizada. Tiempo conversión: {_fmt_dur(conv_seconds)}")
                if conversion_summary.get("files_ok", 0) > 0:
                    wlog(
                        "Reducción de vértices (agregado): "
                        f"{conversion_summary.get('vertices_before_total', 0):,} -> "
                        f"{conversion_summary.get('vertices_after_total', 0):,} "
                        f"({conversion_summary.get('vertices_reduction_pct', 0.0):.2f}%)"
                    )
                write_manifest_patch({"phase": "zipping", "percent": 90.0})
            except Exception as e:
                save_worker_error("convert_to_stl", e)
                return


            try:
                wlog("Calculando métricas de vértices STL ...")
                hq_dir = stl_out / "hq_suavizado_decimado"
                original_dir = stl_out

                target = hq_dir if (hq_dir.exists() and any(hq_dir.rglob('*.stl'))) else original_dir

                def count_vertices(stl_path: Path) -> int:
                    try:
                        m = mesh.Mesh.from_file(str(stl_path))
                        unique_vertices = len(set(map(tuple, m.vectors.reshape(-1, 3).round(5))))
                        return unique_vertices
                    except Exception:
                        return 0

                total_vertices_original = 0
                total_vertices_hq = 0
                count_files = 0

                for f in original_dir.rglob("*.stl"):
                    v = count_vertices(f)
                    total_vertices_original += v
                    count_files += 1
                    wlog(f"Vertices (original) {f.name}: {v}")

                if hq_dir.exists():
                    for f in hq_dir.rglob("*.stl"):
                        v = count_vertices(f)
                        total_vertices_hq += v
                        wlog(f"Vertices (HQ) {f.name}: {v}")

                wlog(f"Total vértices originales: {total_vertices_original}")
                if total_vertices_hq > 0:
                    wlog(f"Total vértices HQ: {total_vertices_hq}")
                    if total_vertices_original > 0:
                        reduction = 100 * (1 - total_vertices_hq / total_vertices_original)
                        wlog(f"Reducción promedio de vértices: {reduction:.2f}%")
                else:
                    wlog("No se encontraron modelos HQ; se usan originales como referencia.")

                write_manifest_patch({"phase": "zipping", "percent": 90.0})
                t_zip_start = time.time()
                zip_bytes = _zip_dir_to_bytes(target)
                s3.upload_bytes(zip_bytes, s3_key(RESULT_ZIP), content_type="application/zip")
                zip_seconds = time.time() - (t_zip_start or time.time())
                wlog(f"ZIP subido a S3. Tiempo empaquetado/subida: {_fmt_dur(zip_seconds)}")

            except Exception as e:
                save_worker_error("zip_upload", e)
                return

            keys = []
            try:
                hq_dir = stl_out / "hq_suavizado_decimado"
                base = hq_dir if hq_dir.exists() else stl_out
                stl_count = int(len([p for p in base.rglob('*.stl')]))
            except Exception:
                stl_count = 0
            stl_zip_size = int(len(zip_bytes or b""))
            total_seconds = time.time() - t_all_start
            write_manifest_patch({
                "status": "done",
                "phase": "finished",
                "percent": 100.0,
                "error": None,
                "metrics": {
                    "stl_count": stl_count,
                    "stl_zip_size": stl_zip_size,
                    "mesh": conversion_summary,
                    "durations": {
                        "segment_seconds": float(seg_seconds or 0.0),
                        "convert_seconds": float(conv_seconds or 0.0),
                        "zip_seconds": float(zip_seconds or 0.0),
                        "total_seconds": float(total_seconds or 0.0),
                    },
                    "vertices": {
                        "total_original": int(total_vertices_original),
                        "total_hq": int(total_vertices_hq),
                        "reduction_percent": float(
                            100 * (1 - total_vertices_hq / total_vertices_original)
                        ) if total_vertices_original > 0 and total_vertices_hq > 0 else None,
                    },
                },
                "s3_keys_results": keys,
            })
            wlog(
                "Resumen tiempos — "
                f"Segmentación: {_fmt_dur(seg_seconds)} | "
                f"Conversión: {_fmt_dur(conv_seconds)} | "
                f"Zip/Subida: {_fmt_dur(zip_seconds)} | "
                f"Total: {_fmt_dur(total_seconds)}"
            )
            wlog("Job terminado.")
            _update_jobconv("DONE", set_finished=True)
            try:
                full = ("\n".join(log_buffer) + "\n").encode("utf-8") if log_buffer else b""
                s3.upload_bytes(full, s3_key(LOG_KEY), content_type="text/plain; charset=utf-8")
            except Exception:
                pass

            try:
                pid = params.get("id_paciente") if isinstance(params, dict) else None
                if db_url:
                    engine = create_engine(db_url, pool_pre_ping=True, future=True)
                    with engine.begin() as conn:
                        conn.execute(
                            _sql_text(
                                "INSERT INTO JobSTL (job_id, id_paciente, stl_size, num_stl_archivos) "
                                "VALUES (:job_id, :id_paciente, :stl_size, :num)"
                            ),
                            {
                                "job_id": job_id,
                                "id_paciente": (int(pid) if pid is not None else None),
                                "stl_size": int(stl_zip_size),
                                "num": int(stl_count),
                            },
                        )
                    wlog("Registro JobSTL guardado en BD.")
            except Exception as e:
                wlog(f"Error guardando JobSTL en BD: {e}")
    except Exception as e:
        save_worker_error("fatal", e)


@router.post("/jobs")
async def create_job(
    file: UploadFile = File(..., description=".zip con DICOMs"),
    nombre_proceso: Optional[str] = Form(None),
    enable_ortopedia: bool = Form(False),
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
    _require_s3_env()
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Se espera un .zip con DICOMs")
    process_name = (nombre_proceso or "").strip()
    if process_name and len(process_name) > 80:
        raise HTTPException(status_code=400, detail="nombre_proceso no puede superar 80 caracteres")
    if process_name:
        process_name = re.sub(r"\s{2,}", " ", process_name).replace("<", "").replace(">", "")
        if not _RE_PROCESS_NAME.fullmatch(process_name):
            raise HTTPException(status_code=422, detail="nombre_proceso invalido")

    s3 = _ensure_s3()
    job_id = uuid.uuid4().hex
    s3_prefix = _s3_job_base(job_id)

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"upload_{job_id}_"))
    safe_upload_name = _safe_upload_filename(file.filename or "")
    local_tmp = tmp_dir / safe_upload_name
    bytes_written = 0
    try:
        with local_tmp.open("wb") as f:
            while True:
                chunk = await file.read(2 * 1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"El ZIP supera el limite permitido de {MAX_UPLOAD_MB}MB",
                    )
                f.write(chunk)
    except HTTPException:
        try:
            local_tmp.unlink(missing_ok=True)
            tmp_dir.rmdir()
        except Exception:
            pass
        raise

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
            "nombre_proceso": (process_name or None),
            "enable_ortopedia": enable_ortopedia,
            "enable_appendicular": enable_appendicular,
            "enable_muscles": enable_muscles,
            "enable_hip_implant": enable_hip_implant,
            "id_paciente": (int(id_paciente) if id_paciente is not None else None),
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
    proc = mp.Process(target=_worker_entry, args=(job_id, s3_cfg, manifest.params, db_url, str(local_tmp)), daemon=True)
    proc.start()
    _append_log(job_id, f"PID worker: {proc.pid}")

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
            queue_name=(process_name or None),
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
async def list_jobs(user = Depends(require_admin)):
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


@router.get("/admin/gpu-check")
async def admin_gpu_check(user=Depends(require_admin)):
    """Diagnostico de reconocimiento de GPU para procesamiento 3D.
    Solo disponible para ADMINISTRADOR.
    """
    return _probe_gpu_runtime()


@router.get("/jobs/mine")
async def list_my_jobs(db: Session = Depends(get_db), user = Depends(get_current_user)):
    """Lista solo los jobs creados por el usuario autenticado (JobConv.id_usuario).
    Enriquecido con datos del manifiesto en S3 si existe (status/phase/percent).
    """
    entries = db.query(JobConv).filter(JobConv.id_usuario == user.id_usuario).order_by(JobConv.updated_at.desc()).all()
    items: List[Dict[str, Any]] = []
    for jc in entries:
        m = _load_manifest(jc.job_id)
        status = (m.status if m else jc.status)
        items.append({
            "job_id": jc.job_id,
            "nombre_proceso": getattr(jc, "queue_name", None),
            "status": status,
            "phase": (m.phase if m else None),
            "percent": (m.percent if m else (100.0 if status == "DONE" else 0.0)),
            "metrics": (m.metrics if m else {}),
            "updated_at": getattr(jc, "updated_at", None),
        })
    return {"jobs": items}


@router.get("/jobs/tracking/mine")
async def list_my_jobs_tracking(db: Session = Depends(get_db), user = Depends(get_current_user)):
    """Seguimiento de procesamiento del usuario autenticado.
    Fuente principal: JobConv (BD). Si existe manifest en S3, enriquece phase/percent.
    """
    rows = (
        db.query(JobConv, Usuario)
        .join(Usuario, Usuario.id_usuario == JobConv.id_usuario)
        .filter(JobConv.id_usuario == user.id_usuario)
        .order_by(JobConv.updated_at.desc())
        .all()
    )

    items: List[Dict[str, Any]] = []
    for jc, u in rows:
        m = _load_manifest(jc.job_id)
        status = (m.status if m else jc.status)
        items.append({
            "job_id": jc.job_id,
            "nombre_proceso": getattr(jc, "queue_name", None),
            "status": status,
            "phase": (m.phase if m else None),
            "percent": (m.percent if m else (100.0 if status == "DONE" else 0.0)),
            "metrics": (m.metrics if m else {}),
            "is_processing": str(status).upper() in ("QUEUED", "RUNNING"),
            "owner": {
                "id_usuario": u.id_usuario,
                "nombre": u.nombre,
                "apellido": u.apellido,
                "correo": u.correo,
            },
            "started_at": getattr(jc, "started_at", None),
            "finished_at": getattr(jc, "finished_at", None),
            "updated_at": getattr(jc, "updated_at", None),
        })
    return {"jobs": items}


@router.get("/jobs/tracking")
async def list_jobs_tracking(db: Session = Depends(get_db), user=Depends(require_admin)):
    """Seguimiento de procesamiento global (solo admin)."""
    rows = (
        db.query(JobConv, Usuario)
        .join(Usuario, Usuario.id_usuario == JobConv.id_usuario)
        .order_by(JobConv.updated_at.desc())
        .all()
    )

    items: List[Dict[str, Any]] = []
    for jc, u in rows:
        m = _load_manifest(jc.job_id)
        status = (m.status if m else jc.status)
        items.append({
            "job_id": jc.job_id,
            "nombre_proceso": getattr(jc, "queue_name", None),
            "status": status,
            "phase": (m.phase if m else None),
            "percent": (m.percent if m else (100.0 if status == "DONE" else 0.0)),
            "metrics": (m.metrics if m else {}),
            "is_processing": str(status).upper() in ("QUEUED", "RUNNING"),
            "owner": {
                "id_usuario": u.id_usuario,
                "nombre": u.nombre,
                "apellido": u.apellido,
                "correo": u.correo,
            },
            "started_at": getattr(jc, "started_at", None),
            "finished_at": getattr(jc, "finished_at", None),
            "updated_at": getattr(jc, "updated_at", None),
        })
    return {"jobs": items}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, db: Session = Depends(get_db), user = Depends(get_current_user)):
    _ensure_job_access(db, user, job_id)
    m = _load_manifest(job_id)
    if not m:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    return m.to_dict()


@router.get("/jobs/{job_id}/progress")
async def get_progress(job_id: str, db: Session = Depends(get_db), user = Depends(get_current_user)):
    _ensure_job_access(db, user, job_id)
    m = _load_manifest(job_id)
    if not m:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    return {"status": m.status, "phase": m.phase, "percent": m.percent, "error": m.error}


@router.get("/jobs/{job_id}/log")
async def get_log(
    job_id: str,
    tail: int = 200,
    full: bool = False,
    db: Session = Depends(get_db),
    user = Depends(get_current_user),
):
    _ensure_job_access(db, user, job_id)
    s3 = _ensure_s3()
    log_key = _s3_key(job_id, LOG_KEY)
    lines: list[str] = []
    used_fallback = True

    if full:
        try:
            data = s3.download_bytes(log_key)
            lines = data.decode("utf-8", errors="ignore").splitlines()
            used_fallback = False
        except Exception:
            try:
                old_key = _s3_key(job_id, "logs/job.log")
                data = s3.download_bytes(old_key)
                lines = data.decode("utf-8", errors="ignore").splitlines()
                used_fallback = False
            except Exception:
                pass

    if not lines:
        m = _load_manifest(job_id)
        if m and m.log_tail:
            lines = list(m.log_tail)
        else:
            try:
                data = s3.download_bytes(log_key)
                lines = data.decode("utf-8", errors="ignore").splitlines()
                used_fallback = False
            except Exception:
                try:
                    old_key = _s3_key(job_id, "logs/job.log")
                    data = s3.download_bytes(old_key)
                    lines = data.decode("utf-8", errors="ignore").splitlines()
                    used_fallback = False
                except Exception:
                    lines = []

    if (not full) and tail and tail > 0:
        try:
            t = int(tail)
            if t > 0:
                lines = lines[-t:]
        except Exception:
            pass

    status = None
    phase = None
    try:
        m2 = _load_manifest(job_id)
        if m2:
            status = m2.status
            phase = m2.phase
    except Exception:
        pass

    return {
        "job_id": job_id,
        "lines": lines,
        "fallback": used_fallback,
        "status": status,
        "phase": phase,
        "full_available": (not used_fallback),
    }


@router.get("/jobs/{job_id}/log/raw")
async def get_log_raw(
    job_id: str,
    download: bool = False,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """Devuelve el log completo en texto plano (sin limite de lineas)."""
    _ensure_job_access(db, user, job_id)
    s3 = _ensure_s3()
    log_key = _s3_key(job_id, LOG_KEY)
    source = "manifest_tail"
    text_data = ""

    try:
        data = s3.download_bytes(log_key)
        text_data = data.decode("utf-8", errors="ignore")
        source = "job.log"
    except Exception:
        try:
            old_key = _s3_key(job_id, "logs/job.log")
            data = s3.download_bytes(old_key)
            text_data = data.decode("utf-8", errors="ignore")
            source = "logs/job.log"
        except Exception:
            m = _load_manifest(job_id)
            text_data = "\n".join((m.log_tail or [])) if m else ""
            source = "manifest_tail"

    headers = {
        "X-Log-Source": source,
    }
    if download:
        headers["Content-Disposition"] = f'attachment; filename="som3d_{job_id}.log"'
    return Response(content=text_data, media_type="text/plain; charset=utf-8", headers=headers)


@router.get("/jobs/{job_id}/stls")
async def get_stls(job_id: str, db: Session = Depends(get_db), user = Depends(get_current_user)):
    _ensure_job_access(db, user, job_id)
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
async def get_result(
    job_id: str,
    expires: int = 3600,
    db: Session = Depends(get_db),
    user = Depends(get_current_user),
):
    _ensure_job_access(db, user, job_id)
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
    _ensure_job_access(db, user, job_id)
    med = db.query(Medico).filter(Medico.id_usuario == user.id_usuario).first()
    if med is not None:
        if payload.id_paciente is None:
            raise HTTPException(status_code=400, detail="id_paciente es obligatorio para medico")
        p = db.query(Paciente).filter(Paciente.id_paciente == payload.id_paciente).first()
        if not p or p.id_medico != med.id_medico:
            raise HTTPException(status_code=403, detail="Paciente no pertenece al medico")

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
        notas=payload.notas,
    )
    db.add(js)
    db.commit()
    db.refresh(js)
    return js


@router.patch("/jobstl/{id_jobstl}/notes", response_model=JobSTLOut)
async def update_jobstl_notes(
    id_jobstl: int,
    payload: JobSTLNoteUpdateIn,
    db: Session = Depends(get_db),
    user = Depends(get_current_user),
):
    js = _get_jobstl_owned(db, user, id_jobstl)
    js.notas = payload.notas
    db.add(js)
    db.commit()
    db.refresh(js)
    return js


@router.delete("/jobstl/{id_jobstl}", status_code=204)
async def delete_jobstl(
    id_jobstl: int,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    if not (_is_admin(user) or _is_medico(user)):
        raise HTTPException(status_code=403, detail="No autorizado")

    js = _get_jobstl_owned(db, user, id_jobstl)
    _ensure_job_access(db, user, js.job_id)

    try:
        _delete_job_artifacts_s3(js.job_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo borrar en MinIO/S3: {e}")

    jobstl_ids = [int(r[0]) for r in db.query(JobSTL.id_jobstl).filter(JobSTL.job_id == js.job_id).all()]
    try:
        if jobstl_ids:
            db.query(VisorEstado).filter(VisorEstado.id_jobstl.in_(jobstl_ids)).delete(synchronize_session=False)
        db.query(JobSTL).filter(JobSTL.job_id == js.job_id).delete(synchronize_session=False)
        db.query(JobConv).filter(JobConv.job_id == js.job_id).delete(synchronize_session=False)
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="No se pudo borrar el registro 3D en base de datos")

    return Response(status_code=204)


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str, user = Depends(require_admin)):
    import psutil
    s3 = _ensure_s3()
    m = _load_manifest(job_id)
    if not m:
        return {"status": "not_found"}

    pid_to_kill = None
    if m.log_tail:
        for line in m.log_tail:
            if "PID worker:" in line:
                try:
                    pid_to_kill = int(line.split("PID worker:")[1].strip())
                except Exception:
                    pass
    if not pid_to_kill:
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
    if not pid_to_kill:
        try:
            old_key = _s3_key(job_id, "logs/job.log")
            data = s3.download_bytes(old_key)
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
    insecure = os.getenv("S3_INSECURE")
    if endpoint is None and os.getenv("AWS_ENDPOINT_URL_S3") is None:
        return
    return


@router.get("/patients-with-stl", response_model=list[PatientJobSTLOut])
def patients_with_stl(db: Session = Depends(get_db), user=Depends(get_current_user)):
    """Lista pacientes que tienen registros en JobSTL.
    - Admin: ve todos
    - Médico: solo sus pacientes
    Devuelve id_jobstl, job_id, id_paciente y datos básicos del paciente.
    """
    q = db.query(JobSTL, Paciente).outerjoin(Paciente, JobSTL.id_paciente == Paciente.id_paciente)
    if getattr(user, "rol", None) != "ADMINISTRADOR":
        med = db.query(Medico).filter(Medico.id_usuario == user.id_usuario).first()
        if not med:
            return []
        q = q.filter(JobSTL.id_paciente.isnot(None), Paciente.id_medico == med.id_medico)
    rows = q.order_by(JobSTL.created_at.desc()).all()
    out: list[PatientJobSTLOut] = []
    for js, p in rows:
        out.append(PatientJobSTLOut(
            id_jobstl=js.id_jobstl,
            job_id=js.job_id,
            id_paciente=js.id_paciente,
            nombres=getattr(p, "nombres", None),
            apellidos=getattr(p, "apellidos", None),
            doc_numero=getattr(p, "doc_numero", None),
            notas=getattr(js, "notas", None),
            created_at=str(getattr(js, "created_at", "")) if getattr(js, "created_at", None) else None,
        ))
    return out


@router.get("/jobs/{job_id}/state-context")
def job_state_context(
    job_id: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """Devuelve contexto minimo para guardar/cargar estado de visor por job.

    Respuesta:
    - job_id
    - id_jobstl (ultimo registro de JobSTL para ese job)
    - id_paciente (si existe vinculacion)
    """
    _ensure_job_access(db, user, job_id)
    js = (
        db.query(JobSTL)
        .filter(JobSTL.job_id == job_id)
        .order_by(JobSTL.created_at.desc(), JobSTL.id_jobstl.desc())
        .first()
    )
    if not js:
        raise HTTPException(status_code=404, detail="Aun no existe contexto STL para este job")
    return {
        "job_id": job_id,
        "id_jobstl": int(js.id_jobstl),
        "id_paciente": (int(js.id_paciente) if js.id_paciente is not None else None),
    }


@router.get("/jobs/{job_id}/result/bytes")
async def download_result_zip_bytes(
    job_id: str,
    db: Session = Depends(get_db),
    user = Depends(get_current_user),
):
    """Descarga el ZIP de resultados desde S3 y lo devuelve en la misma
    respuesta, para evitar problemas de CORS con URLs presignadas.
    """
    _ensure_job_access(db, user, job_id)
    s3 = _ensure_s3()
    key = _s3_key(job_id, RESULT_ZIP)
    if not s3.exists(key):
        raise HTTPException(status_code=404, detail="ZIP no encontrado en S3")
    try:
        data = s3.download_bytes(key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 download error: {e}")
    headers = {
        "Content-Disposition": f"inline; filename=stl_result_{job_id}.zip",
        "Cache-Control": "private, max-age=60",
    }
    return Response(content=data, media_type="application/zip", headers=headers)
