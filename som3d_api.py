#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from fastapi.responses import StreamingResponse, FileResponse  # añade FileResponse

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

import psutil  # <- IMPORTANTE para matar árbol de procesos
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, File, Form, UploadFile, HTTPException

# ==== Importa tus clases (deben existir en el mismo directorio o en el PYTHONPATH)
try:
    from ClaseTotalsegmentor import TotalSegmentatorRunner  # type: ignore
except Exception as e:
    raise RuntimeError("No se pudo importar TotalSegmentatorRunner desde ClaseTotalsegmentor.py") from e

try:
    from ClaseGenerator import NiftiToSTLConverter  # type: ignore
except Exception as e:
    raise RuntimeError("No se pudo importar NiftiToSTLConverter desde ClaseGenerator.py") from e


dotenv_path = find_dotenv(usecwd=True)
if dotenv_path:
    load_dotenv(dotenv_path, override=False)

app = FastAPI(
    title="SOM3D Backend (TS → STL) con Jobs + Progreso + Cancelación",
    version="3.1.0"
)


def _zip_dir_to_bytes(dir_path: Path) -> bytes:
    """Comprime un directorio a ZIP y lo devuelve como bytes (se apoya en un zip temporal)."""
    mem_zip = io.BytesIO()
    with tempfile.TemporaryDirectory(prefix="stl_zip_") as td:
        temp_zip_path = Path(td) / "result_stl.zip"
        shutil.make_archive(str(temp_zip_path.with_suffix('')), 'zip', root_dir=dir_path)
        data = temp_zip_path.read_bytes()
        mem_zip.write(data)
    mem_zip.seek(0)
    return mem_zip.getvalue()


def _iter_file_bytes(data: bytes, chunk_size: int = 1024 * 1024):
    """Iterador para hacer streaming de bytes en chunks."""
    view = memoryview(data)
    for i in range(0, len(view), chunk_size):
        yield view[i:i + chunk_size]


def _collect_nifti_root(ts_out_dir: Path) -> Path:
    """Intenta deducir la carpeta donde quedaron los .nii.gz tras TotalSegmentator."""
    if ts_out_dir.exists() and any(p.suffixes[-2:] == ['.nii', '.gz'] for p in ts_out_dir.rglob('*.nii.gz')):
        return ts_out_dir
    for sub in ts_out_dir.iterdir():
        if sub.is_dir() and any(p.suffixes[-2:] == ['.nii', '.gz'] for p in sub.rglob('*.nii.gz')):
            return sub
    return ts_out_dir


# ==========================
# Modelo de Job + memoria
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
    # runtime (no serializable)
    proc: Optional[Process] = field(default=None, repr=False)       # <- proceso worker
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
        }
        if include_tail:
            data["log_tail"] = list(self._tail)
        return data


# memoria de jobs en proceso (en vivo)
JOBS: Dict[str, Job] = {}


# ==========================
# Worker en proceso separado
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
    """
    Código síncrono que se ejecuta en un proceso separado.
    Hace: TotalSegmentator -> NIfTI->STL -> ZIP
    Escribe logs en log_path (texto plano).
    """
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

    job_dir_p = Path(job_dir)
    ts_out_p = Path(ts_out)
    stl_out_p = Path(stl_out)
    zip_in_p = Path(zip_in)

    # 1) TotalSegmentator (sin --fast; GPU→CPU lo maneja el runner)
    wlog("Iniciando TotalSegmentator (worker)…")
    runner = TotalSegmentatorRunner(
        robust_import=True,
        enable_ortopedia=bool(enable_ortopedia),
        enable_appendicular=bool(enable_appendicular),
        enable_muscles=bool(enable_muscles),
        enable_hip_implant=bool(enable_hip_implant),
        extra_tasks=extra_tasks or [],
        on_log=lambda s: wlog(s)  # duplica también a run.log del runner
    )

    runner.run(zip_in_p, ts_out_p)
    wlog("TotalSegmentator finalizado (worker).")

    # 2) Conversión NIfTI→STL
    wlog("Convirtiendo NIfTI a STL (worker)…")
    converter = NiftiToSTLConverter()
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
        # compatibilidad con firma antigua
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
    wlog("Conversión a STL finalizada (worker).")

    # 3) ZIP resultado
    wlog("Empaquetando STL en ZIP (worker)…")
    data = _zip_dir_to_bytes(stl_out_p)
    out_zip = job_dir_p / "stl_result.zip"
    with out_zip.open("wb") as f:
        f.write(data)
    wlog("ZIP finalizado (worker).")


# ==========================
# Watcher de progreso
# ==========================

async def _watch_progress(job: Job):
    """Inspecciona el filesystem para exponer progreso aproximado y contadores."""
    try:
        while job.status in ("queued", "running") and not job.cancel_flag:
            nifti = list(job.ts_out.rglob("*.nii.gz")) if job.ts_out.exists() else []
            stls = list(job.stl_out.rglob("*.stl")) if job.stl_out.exists() else []
            stats_json = list(job.ts_out.rglob("statistics_all.json"))
            log_csv = list(job.stl_out.rglob("log.csv"))

            # Heurística de porcentaje por fases (aprox):
            # TS 0–70%, Convert 70–95%, Zip 95–100%
            pct = job.percent
            if job.phase == "totalsegmentator":
                nn = min(len(nifti), 63)  # número esperado aprox. de máscaras
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
    # --- Nuevos toggles: por defecto solo ORTOPEDIA=ON; resto OFF ---
    enable_ortopedia: bool = Form(True),
    enable_appendicular: bool = Form(False),
    enable_muscles: bool = Form(False),
    enable_hip_implant: bool = Form(False),
    # Lista de tasks extra (separados por coma)
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
    job.log("Job creado. Lanzando proceso…")

    # Lanzar watcher de progreso (event loop principal)
    job.watcher_task = asyncio.create_task(_watch_progress(job))

    # Lanzar proceso worker
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

    # Monitor para saber cuándo termina el proceso y fijar estado final
    async def _monitor():
        try:
            while proc.is_alive():
                # cambio de fase heurístico
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
                job.log("Job cancelado (monitor).")
            else:
                out_zip = work_dir / "stl_result.zip"
                if out_zip.exists():
                    job.result_zip = out_zip
                    job.status = "done"
                    job.phase = "finished"
                    job.percent = 100.0
                    job.log("Job finalizado correctamente (monitor).")
                else:
                    job.status = "error"
                    job.phase = "error"
                    job.log("Job terminó sin ZIP final. Posible error en worker.")

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
        "log_tail": list(job._tail)[-20:],
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
        raise HTTPException(status_code=409, detail="Resultado no disponible aún")

    return FileResponse(
        job.result_zip,
        media_type="application/zip",
        filename=job.result_zip.name,  # Content-Disposition correcto
    )


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return {"status": "not_found"}

    job.cancel_flag = True
    job.log("Cancelando job por petición del usuario…")

    # 1) Parar watcher/monitor
    for t in (job.watcher_task, job.monitor_task):
        if t and not t.done():
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

    # 2) Matar proceso worker (y su árbol) si sigue vivo
    if job.proc and job.proc.is_alive():
        try:
            parent = psutil.Process(job.proc.pid)
            # matar hijos primero
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except Exception:
                    pass
            # matar al padre
            try:
                parent.kill()
            except Exception:
                pass
        except Exception:
            # Fallback sin psutil (no ideal; psutil ya está instalado)
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
    job.log("Job cancelado.")

    # 3) Limpieza de disco (opcional; comenta si quieres conservar para debug)
    try:
        shutil.rmtree(job.work_dir, ignore_errors=True)
    except Exception:
        pass

    return job.serialize(include_tail=True)


# ==========================
# Arranque opcional
# ==========================

if __name__ == "__main__":
    import uvicorn
    # IMPORTANTE: no usar reload en Windows con multiprocessing
    uvicorn.run("som3d_api:app", host="0.0.0.0", port=8000, reload=False)
