# segmentor.py
# -*- coding: utf-8 -*-
"""
Segmentación de estudios DICOM usando TotalSegmentator.

- Si hay GPU (según cuda.py) -> usa modo completo en GPU.
- Si NO hay GPU -> fuerza CPU y usa --fast.
- Puedes correr SIN argumentos usando DEFAULT_INPUT/DEFAULT_OUTPUT (abajo).
- O bien pasar -i/-o para sobreescribir en tiempo de ejecución.

Requisitos:
  - torch instalado (para que cuda.py pueda detectar GPU)
  - TotalSegmentator instalado y en PATH (CLI "TotalSegmentator")
  - cuda.py en el PYTHONPATH del proyecto
"""

from __future__ import annotations
import os
import sys
import time
import shutil
import zipfile
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List

# ========= EDITA ESTAS RUTAS POR DEFECTO =========
DEFAULT_INPUT  = r"Z:/DICOM/ct-headnii.zip"
DEFAULT_OUTPUT = r"C:/Dicoms - SOM3D"
DEFAULT_TASK   = "total"   # p.ej. "total", "total_mr", "liver", etc.
DEFAULT_FORCE_CPU = False  # True para forzar CPU/--fast
# ================================================

# Importa tu detector
try:
    from cuda import cuda  # tu clase 'cuda' definida en cuda.py
except Exception as e:
    cuda = None
    print(f"[WARN] No se pudo importar cuda.py ({e}). Se asumirá CPU.", file=sys.stderr)


def _which(cmd: str) -> Optional[str]:
    from shutil import which
    return which(cmd)


def _is_dicom_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    for ext in (".dcm", ".DCM"):
        if any(p.glob(f"**/*{ext}")):
            return True
    cnt = 0
    for f in p.rglob("*"):
        if f.is_file():
            try:
                with open(f, "rb") as fh:
                    fh.seek(128)
                    magic = fh.read(4)
                if magic == b"DICM":
                    return True
            except Exception:
                pass
            cnt += 1
            if cnt > 50:
                break
    return False


def _extract_zip_if_needed(input_path: Path) -> Tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    if input_path.suffix.lower() == ".zip":
        tmpdir = tempfile.TemporaryDirectory(prefix="ts_dicom_")
        out_dir = Path(tmpdir.name)
        print(f"[INFO] Descomprimiendo ZIP en: {out_dir}")
        with zipfile.ZipFile(str(input_path), 'r') as zf:
            zf.extractall(out_dir)

        candidate = out_dir
        subs = [p for p in out_dir.iterdir() if p.is_dir()]
        if len(subs) == 1 and _is_dicom_dir(subs[0]):
            candidate = subs[0]
        elif _is_dicom_dir(out_dir):
            candidate = out_dir
        else:
            for p in out_dir.rglob("*"):
                if p.is_dir() and _is_dicom_dir(p):
                    candidate = p
                    break

        if not _is_dicom_dir(candidate):
            print("[WARN] No se detectaron DICOMs claros en el ZIP.", file=sys.stderr)

        return candidate, tmpdir
    else:
        return input_path, None


def _build_totalseg_cmd(
    dicom_dir: Path,
    output_dir: Path,
    task: str,
    use_gpu: bool,
    extra_args: Optional[List[str]] = None
) -> List[str]:
    cmd = [
        "TotalSegmentator",
        "-i", str(dicom_dir),
        "-o", str(output_dir),
        "--task", task
    ]
    if not use_gpu:
        cmd.append("--fast")
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def _run_subprocess(cmd: List[str], env: Optional[dict], log_path: Path) -> int:
    print(f"[INFO] Ejecutando: {' '.join(cmd)}")
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        proc.wait()
        return proc.returncode


def run_segmentation(
    input_path: Path,
    output_path: Path,
    task: str = "total",
    force_cpu: bool = False,
    keep_temp: bool = False,
    extra_args: Optional[List[str]] = None
) -> None:
    start_total = time.time()

    if _which("TotalSegmentator") is None:
        raise RuntimeError(
            "No se encontró el ejecutable 'TotalSegmentator' en el PATH.\n"
            "Instálalo con: pip install TotalSegmentator"
        )

    input_path = input_path.resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    use_gpu = False
    if not force_cpu and cuda is not None:
        try:
            dev = cuda.device()
            use_gpu = (
                getattr(dev, "type", None) == "cuda"
                or (isinstance(dev, str) and dev.startswith("cuda"))
                or str(dev).startswith("cuda")
            )
            print(f"[INFO] Dispositivo detectado por cuda.py: {dev}  -> use_gpu={use_gpu}")

        except Exception as e:
            print(f"[WARN] No se pudo consultar cuda.device(): {e}. Se asumirá CPU.", file=sys.stderr)
            use_gpu = False

    print(f"[INFO] Modo seleccionado: {'GPU' if use_gpu else 'CPU/FAST'}")

    dicom_dir, tmpdir = _extract_zip_if_needed(input_path)
    print(f"[INFO] Directorio usado como entrada DICOM: {dicom_dir}")

    cmd = _build_totalseg_cmd(dicom_dir, output_path, task=task, use_gpu=use_gpu, extra_args=extra_args)

    env = os.environ.copy()
    if use_gpu:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = "-1"

    log_path = output_path / "run.log"

    start_seg = time.time()
    code = _run_subprocess(cmd, env=env, log_path=log_path)
    end_seg = time.time()

    if code != 0:
        if use_gpu:
            print("[ERROR] Falló la ejecución en GPU. Intentando reintentar en CPU con --fast...", file=sys.stderr)
            cmd_cpu = _build_totalseg_cmd(dicom_dir, output_path, task=task, use_gpu=False, extra_args=extra_args)
            env_cpu = env.copy()
            env_cpu["CUDA_VISIBLE_DEVICES"] = "-1"
            code2 = _run_subprocess(cmd_cpu, env=env_cpu, log_path=log_path)
            if code2 != 0:
                raise RuntimeError(f"TotalSegmentator falló (GPU y CPU). Revisa el log: {log_path}")
        else:
            raise RuntimeError(f"TotalSegmentator falló en CPU. Revisa el log: {log_path}")

    if tmpdir and not keep_temp:
        tmpdir.cleanup()

    end_total = time.time()
    print(f"[INFO] Tiempo segmentación: {end_seg - start_seg:0.2f}s")
    print(f"[INFO] Tiempo total:        {end_total - start_total:0.2f}s")
    print(f"[INFO] Resultados en:       {output_path}")
    print(f"[INFO] Log del proceso:     {log_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Ejecuta TotalSegmentator sobre un ZIP/carpeta DICOM usando GPU si está disponible."
    )
    # AHORA NO SON REQUERIDOS: si faltan, usaremos DEFAULT_*
    parser.add_argument("-i", "--input", help="Ruta a .zip con DICOMs o carpeta DICOM")
    parser.add_argument("-o", "--output", help="Carpeta de salida para NIfTI segmentados")
    parser.add_argument("--task", default=None, help="Tarea de TotalSegmentator (p.ej. total, total_mr, liver, etc.)")
    parser.add_argument("--force-cpu", action="store_true", help="Forzar ejecución en CPU (--fast)")
    parser.add_argument("--keep-temp", action="store_true", help="No borrar carpeta temporal al finalizar")
    parser.add_argument("--extra", nargs="*", default=None, help="Argumentos extra para pasar al CLI de TotalSegmentator")
    args = parser.parse_args()

    # Toma defaults si no pasaron -i/-o/--task/--force-cpu
    input_path = Path(args.input) if args.input else Path(DEFAULT_INPUT)
    output_path = Path(args.output) if args.output else Path(DEFAULT_OUTPUT)
    task = args.task if args.task else DEFAULT_TASK
    force_cpu = args.force_cpu or DEFAULT_FORCE_CPU

    print("[INFO] Config efectiva:")
    print(f"   INPUT : {input_path}")
    print(f"   OUTPUT: {output_path}")
    print(f"   TASK  : {task}")
    print(f"   FORCE_CPU: {force_cpu}")

    run_segmentation(
        input_path=input_path,
        output_path=output_path,
        task=task,
        force_cpu=force_cpu,
        keep_temp=args.keep_temp,
        extra_args=args.extra,
    )


if __name__ == "__main__":
    main()
