#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from fastapi.responses import FileResponse

import asyncio
import io
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from multiprocessing import Process
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import deque

import psutil
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, File, Form, UploadFile, HTTPException

# NEW: para mensajes de error detallados
import json, traceback

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
from s3_manager import S3Manager, S3Config  # asegÃºrate del nombre de archivo


# ==========================
# Carga .env (si existe)
# ==========================
dotenv_path = find_dotenv(usecwd=True)
if dotenv_path:
    load_dotenv(dotenv_path, override=False)

# ==========================
# App
# ==========================
app = FastAPI(
    title="SOM3D Backend (TS â†’ STL) con Jobs + Progreso + CancelaciÃ³n",
    version="3.1.0"
)

# Instancia de S3 (una sola, para toda la app)
S3: Optional[S3Manager] = None


@app.on_event("startup")
async def _on_startup():
    global S3
    # Permite inicializar de env o inyectar config si quieres cambiarla en tests
    S3 = S3Manager(S3Config(
        endpoint=os.getenv("S3_ENDPOINT"),
        insecure=bool(os.getenv("S3_INSECURE")),
        bucket=os.getenv("S3_BUCKET", "som3d"),
        prefix=os.getenv("S3_PREFIX", "jobs/"),
        region=os.getenv("AWS_REGION", "us-east-1"),
        access_key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    ))
    # No forzamos create bucket aquÃ­ por si no hay permisos; se harÃ¡ al subir.
    # Pero puedes hacerlo si quieres:
    # try:
    #     S3.ensure_bucket()
    # except Exception as e:
    #     print(f"[WARN] No se pudo verificar/crear bucket en startup: {e}")


# ==========================
# Helpers internos
# ==========================
def _zip_dir_to_bytes(dir_path: Path) -> bytes:
    mem_zip = io.BytesIO()
    with tempfile.TemporaryDirectory(prefix="stl_zip_") as td:
        temp_zip_path = Path(td) / "result_stl.zip"
        shutil.make_archive(str(temp_zip_path.with_suffix('')), 'zip', root_dir=dir_path)
        data = temp_zip_path.read_bytes()
        mem_zip.write(data)
    mem_zip.seek(0)
    return mem_zip.getvalue()


def _collect_nifti_root(ts_out_dir: Path) -> Path:
    if ts_out_dir.exists() and any(p.suffixes[-2:] == ['.nii', '.gz'] for p in ts_out_dir.rglob('*.nii.gz')):
        return ts_out_dir
    for sub in ts_out_dir.iterdir():
        if sub.is_dir() and any(p.suffixes[-2:] == ['.nii', '.gz'] for p in sub.rglob('*.nii.gz')):
            return sub
    return ts_out_dir


# ==========================
# Modelo Job (igual que antes)
# ==========================
@dataclass
class Job:
    job_id: str
    created_at: float
    updated_at: float
    status: str  # queued|running|done|error|canceled
    params: Dict[str, Any]
    work_dir: Path
    zip_in: Path
    ts_out: Path
    stl_out: Path
    result_zip: Optional[Path] = None
    error: Optional[str] = None
    # Info de S3
    s3_bucket: Optional[str] = None
    s3_key_result: Optional[str] = None
    s3_keys_results: list[str] = field(default_factory=list)
    # runtime (no serializable)
    proc: Optional[Process] = field(default=None, repr=False)
    watcher_task: Optional[asyncio.Task] = field(default=None, repr=False)
    monitor_task: Optional[asyncio.Task] = field(default=None, repr=False)
    cancel_flag: bool = False
    # progreso
    phase: str = "queued"  # queued, totalsegmentator, converting, zipping, finished, canceled, error
    percent: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    # log ligero
    log_path: Path = field(default_factory=lambda: Path(tempfile.mkdtemp(prefix="som3d_log_")) / "job.log")
    _tail: deque[str] = field(default_factory=lambda: deque(maxlen=200), repr=False)

    def log(self, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}".rstrip()
        self._tail.append(line)
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass
        self.updated_at = time.time()

    def serialize(self, include_tail: bool = False) -> Dict[str, Any]:
        data = {
            "job_id": self.job_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "phase": self.phase,
            "percent": round(float(self.percent), 1),
            "params": self.params,
            "work_dir": str(self.work_dir),
            "ts_out": str(self.ts_out),
            "stl_out": str(self.stl_out),
            "result_zip": (str(self.result_zip) if self.result_zip else None),
            "error": self.error,
            "metrics": self.metrics,
            "s3_bucket": self.s3_bucket,
            "s3_key_result": self.s3_key_result,
            "s3_keys_results": self.s3_keys_results,
        }
        if include_tail:
            data["log_tail"] = list(self._tail)
        return data


JOBS: Dict[str, Job] = {}


# ==========================
# Worker (proceso separado)
# ==========================
def _worker_entry(
    job_dir: str,
    zip_in: str,
    ts_out: str,
    stl_out: str,
    enable_ortopedia: bool,
    enable_appendicular: bool,
    enable_muscles: bool,
    enable_hip_implant: bool,
    extra_tasks: Optional[List[str]],
    log_path: str,
):
    import time
    from pathlib import Path

    def wlog(msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}\n"
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass

    def _save_worker_error(job_dir_p: Path, phase_name: str, exc: BaseException):
        err = {
            "phase": phase_name,
            "exc_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            (job_dir_p / "worker_error.json").write_text(
                json.dumps(err, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            try:
                (job_dir_p / "worker_error.txt").write_text(
                    f"{phase_name}: {type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                    encoding="utf-8"
                )
            except Exception:
                pass

    job_dir_p = Path(job_dir)
    ts_out_p = Path(ts_out)
    stl_out_p = Path(stl_out)
    zip_in_p = Path(zip_in)

    phase = "start"
    try:
        # 1) TotalSegmentator
        phase = "totalsegmentator"
        wlog("Iniciando TotalSegmentator â€¦")
        runner = TotalSegmentatorRunner(
            robust_import=True,
            enable_ortopedia=bool(enable_ortopedia),
            enable_appendicular=bool(enable_appendicular),
            enable_muscles=bool(enable_muscles),
            enable_hip_implant=bool(enable_hip_implant),
            extra_tasks=extra_tasks or [],
            on_log=lambda s: wlog(s)
        )
        runner.run(zip_in_p, ts_out_p)
        wlog("TotalSegmentator finalizado .")

        # 2) NIfTI â†’ STL
        phase = "convert_to_stl"
        wlog("Convirtiendo NIfTI a STL â€¦")

        def _log_to_job(msg: str):
            try:
                if not msg.endswith("\n"):
                    msg += "\n"
                wlog(msg.rstrip("\n"))
            except Exception:
                pass

        converter = NiftiToSTLConverter(progress_cb=_log_to_job)
        nifti_root = _collect_nifti_root(ts_out_p)
        try:
            converter.convert_folder(
                input_dir=nifti_root,
                output_dir=stl_out_p,
                recursive=True,
                clip_min=None,
                clip_max=None,
                exclude_zeros=True,
                min_voxels=10,
                log_name=None,
            )
        except TypeError:
            converter.convert_folder(
                nifti_root,
                stl_out_p,
                True,
                None,
                None,
                True,
                10,
                None,
            )
        wlog("ConversiÃ³n a STL finalizada .")

    except Exception as e:
        wlog(f"WORKER ERROR en fase '{phase}': {type(e).__name__}: {e}")
        _save_worker_error(job_dir_p, phase, e)

    finally:
        # 3) ZIP del resultado (siempre)
        phase = "zip_result"
        try:
            wlog("Empaquetando STL en ZIP â€¦")
            stl_out_p.mkdir(parents=True, exist_ok=True)
            data = _zip_dir_to_bytes(stl_out_p)
            out_zip = job_dir_p / "stl_result.zip"
            with out_zip.open("wb") as f:
                f.write(data)
            wlog("ZIP finalizado .")
        except Exception as z:
            wlog(f"ZIP ERROR: {type(z).__name__}: {z}")
            _save_worker_error(job_dir_p, phase, z)


# ==========================
# Watcher progreso
# ==========================
async def _watch_progress(job: Job):
    try:
        while job.status in ("queued", "running") and not job.cancel_flag:
            nifti = list(job.ts_out.rglob("*.nii.gz")) if job.ts_out.exists() else []
            stls = list(job.stl_out.rglob("*.stl")) if job.stl_out.exists() else []
            stats_json = list(job.ts_out.rglob("statistics_all.json"))
            log_csv = list(job.stl_out.rglob("log.csv"))

            pct = job.percent
            if job.phase == "totalsegmentator":
                nn = min(len(nifti), 63)
                pct = max(pct, min(70.0, 10.0 + nn * (60.0 / 63.0)))
            elif job.phase == "converting":
                ns = len(stls)
                pct = max(pct, min(95.0, 70.0 + min(ns, 40) * (25.0 / 40.0)))
            elif job.phase == "zipping":
                pct = max(pct, 97.0)

            job.percent = pct
            job.metrics = {
                "nifti_found": len(nifti),
                "stl_found": len(stls),
                "has_ts_stats": bool(stats_json),
                "has_convert_log": bool(log_csv),
            }
            await asyncio.sleep(2.0)
    except asyncio.CancelledError:
        return
    except Exception as e:
        job.log(f"Watcher error: {e}")


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
    extra_tasks: Optional[str] = Form(None),
):
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Se espera un .zip con DICOMs")

    job_id = uuid.uuid4().hex
    work_dir = Path(tempfile.mkdtemp(prefix=f"som3d_job_{job_id}_"))
    ts_out = work_dir / "TS_OUT"
    stl_out = work_dir / "STL_OUT"
    ts_out.mkdir(parents=True, exist_ok=True)
    stl_out.mkdir(parents=True, exist_ok=True)

    zip_path = work_dir / file.filename
    with zip_path.open("wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    extra_list = None
    if extra_tasks:
        extra_list = [t.strip() for t in extra_tasks.split(",") if t.strip()]

    job = Job(
        job_id=job_id,
        created_at=time.time(),
        updated_at=time.time(),
        status="running",
        params={
            "enable_ortopedia": enable_ortopedia,
            "enable_appendicular": enable_appendicular,
            "enable_muscles": enable_muscles,
            "enable_hip_implant": enable_hip_implant,
            "extra_tasks": extra_list,
        },
        work_dir=work_dir,
        zip_in=zip_path,
        ts_out=ts_out,
        stl_out=stl_out,
        phase="totalsegmentator",
        percent=0.0,
    )
    job.log("Job creado. Lanzando procesoâ€¦")

    job.watcher_task = asyncio.create_task(_watch_progress(job))

    proc = Process(
        target=_worker_entry,
        args=(
            str(work_dir),
            str(zip_path),
            str(ts_out),
            str(stl_out),
            enable_ortopedia,
            enable_appendicular,
            enable_muscles,
            enable_hip_implant,
            extra_list,
            str(job.log_path),
        ),
        daemon=True,
    )
    proc.start()
    job.proc = proc
    JOBS[job_id] = job

    async def _monitor():
        try:
            while proc.is_alive():
                if job.phase == "totalsegmentator" and any(ts_out.rglob("*.nii.gz")):
                    job.phase = "converting"
                if job.phase in ("totalsegmentator", "converting") and any(stl_out.rglob("*.stl")):
                    job.phase = "converting"
                await asyncio.sleep(2)
        finally:
            if job.cancel_flag:
                job.status = "canceled"
                job.phase = "canceled"
                job.percent = min(job.percent, 99.0)
                job.error = "Cancelado por el usuario"
                job.log("Job cancelado (monitor).")
            else:
                out_zip = work_dir / "stl_result.zip"
                if out_zip.exists():
                    job.result_zip = out_zip
                    job.status = "done"
                    job.phase = "finished"
                    job.percent = 100.0
                    job.error = None
                    job.log("Job finalizado correctamente (monitor).")

                    # === Subida a S3/MinIO usando S3Manager ===
                    if S3 is not None:
                        try:
                            S3.ensure_bucket()
                            s3_prefix = os.getenv("S3_PREFIX", "jobs/")
                            job_base_key = S3.join_key(s3_prefix, job.job_id)
                            hq_dir = job.stl_out / "hq_suavizado_decimado"
                            uploaded_keys = []
                            if hq_dir.exists():
                                for stl_path in sorted(hq_dir.rglob("*.stl")):
                                    s3_key = S3.join_key(job_base_key, stl_path.name)
                                    S3.upload_file(str(stl_path), s3_key)
                                    uploaded_keys.append(s3_key)
                                job.log(f"STLs HQ subidos a s3://{S3.cfg.bucket}/{job_base_key} ({len(uploaded_keys)} archivos).")
                            else:
                                job.log("Carpeta HQ no encontrada; no se subieron STLs.")
                            job.s3_bucket = S3.cfg.bucket
                            job.s3_key_result = job_base_key
                            job.s3_keys_results = uploaded_keys
                            job.metrics = {**job.metrics, "hq_stl_uploaded": len(uploaded_keys)}
                        except (S3Manager.ClientError, S3Manager.EndpointError, Exception) as e:
                            job.log(f"S3 upload ERROR: {type(e).__name__}: {e}")
                    else:
                        job.log("S3Manager no inicializado: no se subió a S3.")
                else:
                    specific = None
                    err_json = work_dir / "worker_error.json"
                    err_txt  = work_dir / "worker_error.txt"
                    if err_json.exists():
                        try:
                            data = json.loads(err_json.read_text(encoding="utf-8"))
                            phase = data.get("phase")
                            etype = data.get("exc_type")
                            msg   = data.get("message")
                            specific = f"{etype}: {msg} (fase: {phase})"
                        except Exception:
                            pass
                    if not specific and err_txt.exists():
                        try:
                            first = err_txt.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
                            specific = first.strip()
                        except Exception:
                            pass
                    if not specific:
                        specific = "No se generÃ³ ZIP y no hay detalle en worker_error.* (revisa el log)."

                    job.status = "error"
                    job.phase = "error"
                    job.error = specific
                    job.log(f"Job terminÃ³ con ERROR: {specific}")

    job.monitor_task = asyncio.create_task(_monitor())
    return job.serialize(include_tail=True)


@app.get("/jobs")
async def list_jobs():
    return {"jobs": [j.serialize() for j in JOBS.values()]}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    return job.serialize(include_tail=True)



@app.get("/jobs/{job_id}/stls")
async def job_stls(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job no encontrado")

    if job.s3_keys_results:
        items = [
            {
                "filename": Path(key).name,
                "s3_key": key,
            }
            for key in job.s3_keys_results
        ]
        source = "s3"
    else:
        hq_dir = job.stl_out / "hq_suavizado_decimado"
        stl_paths = []
        if hq_dir.exists():
            stl_paths = sorted(hq_dir.glob("*.stl"))
        elif job.stl_out.exists():
            stl_paths = sorted(job.stl_out.glob("*.stl"))
        items = [
            {
                "filename": p.name,
                "path": str(p),
                "size_bytes": p.stat().st_size,
            }
            for p in stl_paths
        ]
        source = "local"

    return {
        "job_id": job.job_id,
        "source": source,
        "bucket": job.s3_bucket,
        "s3_prefix": job.s3_key_result,
        "items": items,
    }

@app.get("/jobs/{job_id}/progress")
async def job_progress(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    return {
        "job_id": job.job_id,
        "status": job.status,
        "phase": job.phase,
        "percent": round(float(job.percent), 1),
        "metrics": job.metrics,
        "message": (job._tail[-1] if len(job._tail) else None),
        "error": job.error,
        "log_tail": list(job._tail)[-20:],
        "s3_bucket": job.s3_bucket,
        "s3_key_result": job.s3_key_result,
        "s3_keys_results": job.s3_keys_results,
    }


@app.get("/jobs/{job_id}/log")
async def job_log(job_id: str, tail: int = 200):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    tail = max(1, min(int(tail), 2000))
    try:
        if job.log_path.exists():
            with job.log_path.open("r", encoding="utf-8", errors="ignore") as f:
                lines = f.read().splitlines()[-tail:]
        else:
            lines = list(job._tail)[-tail:]
    except Exception:
        lines = list(job._tail)[-tail:]
    return {"job_id": job.job_id, "lines": lines}


@app.get("/jobs/{job_id}/result")
async def get_result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    if job.status != "done" or not job.result_zip or not job.result_zip.exists():
        detail = job.error or "Resultado no disponible aÃºn"
        raise HTTPException(status_code=409, detail=detail)
    return FileResponse(
        job.result_zip,
        media_type="application/zip",
        filename=job.result_zip.name,
    )


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return {"status": "not_found"}

    job.cancel_flag = True
    job.log("Cancelando job por peticiÃ³n del usuarioâ€¦")

    for t in (job.watcher_task, job.monitor_task):
        if t and not t.done():
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

    if job.proc and job.proc.is_alive():
        try:
            parent = psutil.Process(job.proc.pid)
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
            try:
                job.proc.terminate()
            except Exception:
                pass
        finally:
            try:
                job.proc.join(timeout=5)
            except Exception:
                pass

    job.status = "canceled"
    job.phase = "canceled"
    job.percent = min(job.percent, 99.0)
    job.error = "Cancelado por el usuario"
    job.log("Job cancelado.")

    try:
        shutil.rmtree(job.work_dir, ignore_errors=True)
    except Exception:
        pass

    return job.serialize(include_tail=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("som3d_api:app", host="0.0.0.0", port=8000, reload=False)

