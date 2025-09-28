#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TotalSegmentator - GUI de un solo archivo con importación DICOM robusta
- Entrada: .zip / carpeta DICOM / .nii/.nii.gz
- Salida: carpeta con NIfTI de clases
- Task seleccionable (total, total_mr, etc.)
- Extra args (p.ej. --ml --statistics --preview --roi_subset ...)
- Modos de dispositivo: Auto (GPU si hay), Solo CPU, CPU+--fast
- Fallback robusto: detecta series, usa dcm2niix; si no, dicom2nifti relajado;
  si hay JPEG 12-bit, intenta descomprimir con pydicom+pylibjpeg y reintenta.
- Parches de estabilidad: límites de hilos BLAS/NumPy, TMP/TEMP dedicado, logger.
- Preset: SOLO ORTOPEDIA (inyecta --roi_subset con estructuras osteo-articulares).
- Auto Threads: calcula --nr_thr_resamp / --nr_thr_saving según VRAM y CPU.

🆕 DUAL PASS: si hay preset Ortopedia o mezcla de ROIs de 'total' y 'body',
   corre dos pasadas reutilizando el MISMO NIfTI:
   1) task=total  → out_total/
   2) task=body   → out_body/
"""

import os
import sys
import shlex
import zipfile
import subprocess
import threading
import shutil
import stat
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Tuple, List

# ------------------- Config ------------------- #
TASKS = [
    "total", "total_mr",
    "lung_vessels", "body", "body_mr",
    "vertebrae_mr", "cerebral_bleed", "hip_implant",
    "pleural_pericard_effusion", "head_glands_cavities",
    "head_muscles", "headneck_bones_vessels", "headneck_muscles",
    "liver_vessels", "oculomotor_muscles", "lung_nodules",
    "kidney_cysts", "breasts", "liver_segments", "liver_segments_mr",
    "craniofacial_structures", "abdominal_muscles", "teeth"
]

# === Preferencias de trabajo sin TEMP del sistema ===
USE_LOCAL_WORK = True               # True => usar carpeta de trabajo dentro de la salida
DEFAULT_WORK_SUBDIR = "_work"       # nombre del subdirectorio de trabajo

# ---------------- Preset: SOLO ORTOPEDIA (sin músculos) ---------------- #
ORTHO_ROI = [
    # Cráneo / pelvis
    "skull", "sacrum",
    # Columna C1–C7, T1–T12, L1–L5, S1
    "vertebrae_C1","vertebrae_C2","vertebrae_C3","vertebrae_C4","vertebrae_C5","vertebrae_C6","vertebrae_C7",
    "vertebrae_T1","vertebrae_T2","vertebrae_T3","vertebrae_T4","vertebrae_T5","vertebrae_T6","vertebrae_T7","vertebrae_T8","vertebrae_T9","vertebrae_T10","vertebrae_T11","vertebrae_T12",
    "vertebrae_L1","vertebrae_L2","vertebrae_L3","vertebrae_L4","vertebrae_L5","vertebrae_S1",
    # Costillas 1–12 izquierda y derecha
    "rib_left_1","rib_left_2","rib_left_3","rib_left_4","rib_left_5","rib_left_6","rib_left_7","rib_left_8","rib_left_9","rib_left_10","rib_left_11","rib_left_12",
    "rib_right_1","rib_right_2","rib_right_3","rib_right_4","rib_right_5","rib_right_6","rib_right_7","rib_right_8","rib_right_9","rib_right_10","rib_right_11","rib_right_12",
    # Esternón / cartílagos costales
    "sternum","costal_cartilages",
    # Cintura escapular
    "clavicula_left","clavicula_right","scapula_left","scapula_right",
    # Huesos largos y cadera
    "humerus_left","humerus_right","femur_left","femur_right","hip_left","hip_right",
]

# ---------------- Conjuntos válidos por task ---------------- #
VALID_TOTAL = {
    "skull","sacrum",
    "vertebrae_C1","vertebrae_C2","vertebrae_C3","vertebrae_C4","vertebrae_C5","vertebrae_C6","vertebrae_C7",
    "vertebrae_T1","vertebrae_T2","vertebrae_T3","vertebrae_T4","vertebrae_T5","vertebrae_T6","vertebrae_T7","vertebrae_T8","vertebrae_T9","vertebrae_T10","vertebrae_T11","vertebrae_T12",
    "vertebrae_L1","vertebrae_L2","vertebrae_L3","vertebrae_L4","vertebrae_L5","vertebrae_S1",
    "rib_left_1","rib_left_2","rib_left_3","rib_left_4","rib_left_5","rib_left_6","rib_left_7","rib_left_8","rib_left_9","rib_left_10","rib_left_11","rib_left_12",
    "rib_right_1","rib_right_2","rib_right_3","rib_right_4","rib_right_5","rib_right_6","rib_right_7","rib_right_8","rib_right_9","rib_right_10","rib_right_11","rib_right_12",
    "sternum","costal_cartilages","clavicula_left","clavicula_right","scapula_left","scapula_right",
    "humerus_left","humerus_right","femur_left","femur_right","hip_left","hip_right",
}
VALID_BODY = {"body","body_trunc","body_extremities","skin"}

# ---------------- Helpers de paths/seguridad ---------------- #
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _on_rm_error(func, path, exc_info):
    # Quita solo-lectura y reintenta (útil en Windows)
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass

def _is_subpath(parent: Path, child: Path) -> bool:
    try:
        parent = parent.resolve()
        child = child.resolve()
        return parent in child.parents or parent == child
    except Exception:
        return False

def safe_rmtree(p: Path, retries: int = 3, delay: float = 0.25):
    for _ in range(retries):
        try:
            shutil.rmtree(p, ignore_errors=False, onerror=_on_rm_error)
            return
        except Exception:
            time.sleep(delay)
    shutil.rmtree(p, ignore_errors=True)

def get_work_dir(output_dir: Path, custom: Optional[Path] = None) -> Path:
    """
    Devuelve la carpeta de trabajo local. Si custom es None,
    usa <output_dir>/_work.
    """
    if custom:
        wd = Path(custom)
    else:
        wd = Path(output_dir) / DEFAULT_WORK_SUBDIR
    ensure_dir(wd)
    return wd

def setup_env_workdir(env: dict, work_dir: Path):
    """
    Redirige TODOS los temporales a la carpeta de trabajo local.
    Incluye joblib, numpy, etc.
    """
    joblib_dir = work_dir / "joblib"
    ensure_dir(joblib_dir)
    env["JOBLIB_TEMP_FOLDER"] = str(joblib_dir)
    env["TMPDIR"] = str(work_dir)
    env["TMP"] = str(work_dir)
    env["TEMP"] = str(work_dir)

# ---------------- Detección de GPU ---------------- #
def detect_device():
    info = {
        "backend": "cpu", "available": False, "device": "cpu",
        "torch_version": None, "cuda_compiled": False, "cuda_available": False,
        "cuda_version": None, "device_count": 0, "device_name": None,
        "vram_free_gb": None, "vram_total_gb": None,
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_compiled"] = torch.version.cuda is not None
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        info["device_count"] = torch.cuda.device_count() if info["cuda_available"] else 0
        if info["cuda_available"] and info["device_count"] > 0:
            idx = torch.cuda.current_device()
            prop = torch.cuda.get_device_properties(idx)
            info["backend"] = "cuda"; info["available"] = True
            info["device"] = f"cuda:{idx}"; info["device_name"] = prop.name
            try:
                free, total = torch.cuda.mem_get_info(idx)
                info["vram_free_gb"] = round(free/(1024**3), 2)
                info["vram_total_gb"] = round(total/(1024**3), 2)
            except Exception:
                pass
    except Exception:
        pass
    return info

# --------------- Utilidades path/zip --------------- #
def is_nii(path: Path) -> bool:
    n = path.name.lower()
    return n.endswith(".nii") or n.endswith(".nii.gz")

def unzip_to_work(zip_path: Path, work_dir: Path) -> Path:
    zip_root = work_dir / "zip"
    shutil.rmtree(zip_root, ignore_errors=True)
    ensure_dir(zip_root)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(zip_root)
    subs = [p for p in zip_root.iterdir() if p.is_dir()]
    if len(subs) == 1:
        return subs[0]
    return zip_root

def pick_largest_nii(folder: Path) -> Path | None:
    niis = list(folder.rglob("*.nii")) + list(folder.rglob("*.nii.gz"))
    if not niis:
        return None
    return max(niis, key=lambda p: p.stat().st_size)

# ----------- Agrupar por serie y elegir la mejor ----------- #
def split_series(input_dir: Path, on_log=print) -> dict[str, list[Path]]:
    """Agrupa archivos por SeriesInstanceUID. Requiere pydicom."""
    try:
        import pydicom as dcm
    except Exception:
        on_log("[WARN] pydicom no disponible; no se puede separar por serie.")
        return {}
    series: dict[str, list[Path]] = {}
    files = [p for p in input_dir.rglob("*") if p.is_file()]
    for f in files:
        try:
            ds = dcm.dcmread(str(f), stop_before_pixels=True, force=True)
            uid = str(getattr(ds, 'SeriesInstanceUID', 'UNKNOWN'))
            series.setdefault(uid, []).append(f)
        except Exception:
            continue
    for uid, lst in series.items():
        on_log(f"[INFO] Serie {uid[:8]}… con {len(lst)} archivos")
    return series

def select_best_series(input_dir: Path, on_log=print, work_dir: Optional[Path]=None) -> Path | None:
    """Copia la *mejor* serie (más imágenes y orientación consistente) a <work_dir>/series."""
    series = split_series(input_dir, on_log)
    if not series:
        return None
    uid, files = max(series.items(), key=lambda kv: len(kv[1]))
    try:
        import pydicom as dcm, numpy as np
        def orient_key(fp: Path):
            try:
                ds = dcm.dcmread(str(fp), stop_before_pixels=True, force=True)
                iop = getattr(ds, 'ImageOrientationPatient', None)
                if iop is not None:
                    arr = np.array(iop, dtype=float)
                    return tuple((arr/np.linalg.norm(arr)).round(3))
            except Exception:
                pass
            return None
        keys = {orient_key(f) for f in files}
        on_log(f"[INFO] Orientaciones detectadas en serie elegida: {len(keys)}")
    except Exception:
        pass
    target = (work_dir or input_dir) / "series"
    shutil.rmtree(target, ignore_errors=True)
    ensure_dir(target)
    for i, f in enumerate(sorted(files)):
        dst = target / f"img_{i:05d}.dcm"
        try:
            shutil.copy2(f, dst)
        except Exception:
            pass
    on_log(f"[INFO] Serie seleccionada → {target} ({len(files)} DICOMs)")
    return target

# ----------- Conversión robusta DICOM -> NIfTI ----------- #
def convert_with_dcm2niix(input_dir: Path, out_dir: Path, on_log=print) -> Path | None:
    exe = shutil.which("dcm2niix")
    if not exe:
        return None
    shutil.rmtree(out_dir, ignore_errors=True)
    ensure_dir(out_dir)
    cmd = [exe, "-z", "y", "-m", "y", "-i", "y", "-f", "converted_%p_%s", "-o", str(out_dir), str(input_dir)]
    on_log("dcm2niix: " + " ".join(shlex.quote(x) for x in cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        on_log(f"dcm2niix rc={rc} (no se generó NIfTI)")
        return None
    nii = pick_largest_nii(out_dir)
    if nii:
        on_log(f"dcm2niix OK → {nii.name}")
    return nii

def decompress_series_with_pydicom(input_dir: Path, out_dir: Path, on_log=print) -> Path | None:
    """Reescribe DICOMs comprimidos como ExplicitVRLittleEndian (sin compresión)."""
    try:
        import pydicom as dcm
        from pydicom.uid import ExplicitVRLittleEndian
        shutil.rmtree(out_dir, ignore_errors=True)
        ensure_dir(out_dir)
        count = 0
        for f in sorted([p for p in input_dir.rglob("*") if p.is_file()]):
            try:
                ds = dcm.dcmread(str(f), force=True)
                ts = ds.file_meta.TransferSyntaxUID
                compressed = getattr(ts, 'is_compressed', False)
                if compressed:
                    _ = ds.pixel_array  # fuerza la decodificación
                    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
                    ds.is_implicit_VR = False
                    ds.is_little_endian = True
                dst = out_dir / f"{count:05d}.dcm"
                ds.save_as(str(dst), write_like_original=False)
                count += 1
            except Exception as e:
                on_log(f"[WARN] No se pudo reescribir {f.name}: {e}")
        on_log(f"[INFO] Reescritos {count} DICOMs sin compresión en {out_dir}")
        return out_dir if count > 0 else None
    except Exception as e:
        on_log(f"[WARN] pydicom/pylibjpeg no disponibles para descomprimir: {e}")
        return None

def convert_with_dicom2nifti_relaxed(input_dir: Path, out_dir: Path, on_log=print) -> Path | None:
    try:
        import dicom2nifti
        from dicom2nifti import settings
        shutil.rmtree(out_dir, ignore_errors=True)
        ensure_dir(out_dir)
        for fn in (
            'disable_validate_orientation',
            'disable_validate_orthogonal',
            'disable_validate_slice_increment',
            'disable_validate_instance_number',
            'disable_validate_woodpecker',
        ):
            if hasattr(settings, fn):
                getattr(settings, fn)()
        on_log("dicom2nifti (relajado): convirtiendo directorio…")
        dicom2nifti.convert_directory(str(input_dir), str(out_dir), compression=True, reorient=True)
        nii = pick_largest_nii(out_dir)
        if nii:
            on_log(f"dicom2nifti OK → {nii.name}")
        return nii
    except Exception as e:
        on_log(f"dicom2nifti error: {e}")
        return None

def robust_dicom_to_nifti(input_dir: Path, work_dir: Path, on_log=print) -> Path:
    """Pipeline robusto DICOM→NIfTI usando solo carpetas locales de trabajo."""
    series_dir = select_best_series(input_dir, on_log=on_log, work_dir=work_dir) or input_dir
    conv_dir   = work_dir / "conv"
    dec_dir    = work_dir / "dec"   # para reintentos descomprimidos

    nii = convert_with_dcm2niix(series_dir, conv_dir, on_log=on_log)
    if not nii:
        nii = convert_with_dicom2nifti_relaxed(series_dir, conv_dir, on_log=on_log)
    if not nii:
        if decompress_series_with_pydicom(series_dir, dec_dir, on_log=on_log):
            nii = convert_with_dcm2niix(dec_dir, conv_dir, on_log=on_log) or \
                  convert_with_dicom2nifti_relaxed(dec_dir, conv_dir, on_log=on_log)
    if not nii:
        raise RuntimeError("Conversión robusta DICOM→NIfTI falló (ni dcm2niix ni dicom2nifti).")
    return nii

# --------------- Flags helpers --------------- #
def _ensure_flag(args: list[str], flag: str, value: str | None = None):
    """Añade --flag [value] si aún no está presente en args."""
    if flag in args:
        return args
    if value is None:
        return args + [flag]
    return args + [flag, value]

def remove_flag_and_value(args: list[str], flag: str) -> list[str]:
    """Elimina --flag y su valor inmediato (si existe) de la lista."""
    if not args:
        return args
    out = []
    skip = 0
    for i, a in enumerate(args):
        if skip:
            skip -= 1
            continue
        if a == flag:
            if i + 1 < len(args) and not args[i+1].startswith("--"):
                skip = 1
            continue
        out.append(a)
    return out

def normalize_roi_subset_args(args: list[str]) -> list[str]:
    """
    Convierte '--roi_subset a,b,c' en '--roi_subset a b c'
    y mantiene ya-correctos '--roi_subset a b c'.
    """
    out = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--roi_subset":
            out.append(a)
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                token = args[i]
                if "," in token:
                    out.extend([t for t in token.split(",") if t])
                else:
                    out.append(token)
                i += 1
            continue
        else:
            out.append(a)
            i += 1
    return out

def extract_roi_list(args: list[str]) -> list[str]:
    rois = []
    if "--roi_subset" in args:
        i = args.index("--roi_subset") + 1
        while i < len(args) and not args[i].startswith("--"):
            rois.append(args[i]); i += 1
    return rois

def replace_roi_subset(args: list[str], new_rois: list[str]) -> list[str]:
    args = remove_flag_and_value(args, "--roi_subset")
    if new_rois:
        args = args + ["--roi_subset"] + new_rois
    return args

# --------------- Auto threads según VRAM/CPU --------------- #
def suggest_threads(device_mode: str, dev_info: dict) -> tuple[int, int]:
    """
    Devuelve (nr_thr_resamp, nr_thr_saving) sugeridos según:
    - GPU: VRAM total.
    - CPU: núcleos disponibles.
    """
    cpu_count = max(os.cpu_count() or 2, 2)

    if device_mode == "auto" and dev_info.get("available"):
        vram = dev_info.get("vram_total_gb") or 0
        if vram >= 12:
            return (6, 6)
        elif vram >= 8:
            return (4, 4)
        elif vram >= 6:
            return (3, 3)
        elif vram >= 4:
            return (2, 2)
        else:
            return (1, 1)

    if device_mode in ("cpu", "cpu_fast") or not dev_info.get("available"):
        n = min(max(cpu_count // 2, 2), 8)
        return (n, n)

    return (1, 1)

# --------------- Ejecutor TotalSegmentator --------------- #
def build_command(input_path: Path, output_dir: Path, task: str, device_mode: str, extra_args: list[str] | None):
    """device_mode ∈ {auto, cpu, cpu_fast}"""
    cmd = ["TotalSegmentator", "-i", str(input_path), "-o", str(output_dir)]
    if task:
        cmd += ["--task", task]

    if device_mode == "cpu":
        cmd += ["--device", "cpu"]
    elif device_mode == "cpu_fast":
        cmd += ["--device", "cpu", "--fast"]
    else:  # auto
        dev = detect_device()
        if dev["available"]:
            cmd += ["--device", "gpu"]

    if extra_args:
        cmd += extra_args

    return cmd

def run_with_streaming(cmd, env, on_line, cwd=None) -> int:
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, bufsize=1, env=env, cwd=cwd
    )
    for line in proc.stdout:
        on_line(line.rstrip("\n"))
    proc.wait()
    return proc.returncode

def run_totalsegmentator(input_path: Path, output_path: Path, task: str, device_mode: str, extra_args, on_log, work_dir: Path | None = None):
    env = os.environ.copy()

    # Limita hilos BLAS para bajar picos de RAM/fragmentación
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    # Redirigir temporales SOLO a local work_dir
    if work_dir is not None:
        setup_env_workdir(env, work_dir)

    if device_mode in ("cpu", "cpu_fast"):
        env["CUDA_VISIBLE_DEVICES"] = "-1"

    cmd = build_command(input_path, output_path, task, device_mode, extra_args)
    on_log("Comando: " + " ".join(shlex.quote(x) for x in cmd))
    return run_with_streaming(cmd, env, on_line=lambda s: on_log(s))

def run_segmentation(
    input_path: Path,
    output_path: Path,
    task: str = "total",
    device_mode: str = "auto",  # auto | cpu | cpu_fast
    keep_temp: bool = False,
    extra_args: list[str] | None = None,
    robust_import: bool = True,
    on_log=lambda s: print(s),
    use_local_work: bool = USE_LOCAL_WORK,
    custom_work_dir: Optional[Path] = None
):
    """Pipeline de alto nivel con fallback robusto para DICOMs problemáticos (sin usar TEMP del sistema)."""
    ensure_dir(output_path)

    log_file = output_path / "run.log"
    def log(msg: str):
        on_log(msg)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("=== Inicio TotalSegmentator ===")
    dev = detect_device()
    log(
        f"Dispositivo detectado: backend={dev['backend']} avail={dev['available']} "
        f"device={dev['device']} torch={dev['torch_version']} cuda={dev['cuda_version']} "
        f"name={dev['device_name']} VRAM={dev['vram_free_gb']}/{dev['vram_total_gb']} GB"
    )

    work_dir = get_work_dir(output_path, custom_work_dir) if use_local_work else None
    if work_dir:
        log(f"[WORK] Carpeta de trabajo local: {work_dir}")

    true_input = input_path

    # unzip si es zip → extraer en work_dir
    if input_path.suffix.lower() == ".zip":
        if not work_dir:
            raise RuntimeError("Se requiere work_dir local para extraer ZIP sin usar TEMP del sistema.")
        log(f"Descomprimiendo ZIP en carpeta local…")
        temp_dir = unzip_to_work(input_path, work_dir)
        log(f"ZIP extraído en: {temp_dir}")
        true_input = temp_dir

    if not Path(true_input).exists():
        raise FileNotFoundError(f"Entrada no encontrada: {true_input}")

    # 1) Importación robusta si DICOM; si ya es NIfTI, usar directo
    if is_nii(true_input):
        nii_input = true_input
        log(f"Entrada ya es NIfTI: {Path(nii_input).name}")
    else:
        if robust_import:
            if not work_dir:
                raise RuntimeError("Se requiere work_dir local para conversión DICOM→NIfTI sin TEMP.")
            log("Iniciando conversión robusta DICOM→NIfTI (series + dcm2niix/dicom2nifti)…")
            nii_input = robust_dicom_to_nifti(Path(true_input), work_dir, on_log=log)
            log(f"NIfTI intermedio: {nii_input}")
        else:
            nii_input = None

    # 2) Ejecutar TotalSegmentator apuntando env a work_dir
    input_for_ts = nii_input if nii_input else true_input
    rc = run_totalsegmentator(Path(input_for_ts), output_path, task, device_mode, extra_args, log, work_dir=work_dir)

    if rc != 0 and device_mode == "auto":
        log(f"⚠️ Falló rc={rc} en modo auto. Reintento en CPU/--fast…")
        rc = run_totalsegmentator(Path(input_for_ts), output_path, task, "cpu_fast", extra_args, log, work_dir=work_dir)

    if rc != 0:
        raise RuntimeError("TotalSegmentator falló incluso tras pipeline robusto. Revisa run.log.")

    # 3) Limpieza: eliminar por completo la carpeta de trabajo
    if work_dir and not keep_temp:
        try:
            if _is_subpath(output_path, work_dir) and work_dir.name == DEFAULT_WORK_SUBDIR:
                safe_rmtree(work_dir)
                log(f"Work eliminado: {work_dir}")
            else:
                log(f"[SKIP] Work no eliminado por seguridad: {work_dir}")
        except Exception as e:
            log(f"No se pudo eliminar work: {e}")

    log("=== Fin ===")

# --------------- Pipeline dual (total + body) --------------- #
def run_dual_pass(
    input_path: Path,
    output_root: Path,
    total_rois: list[str],
    body_rois: list[str],
    device_mode: str,
    extra_common: list[str],
    robust_import: bool,
    on_log=lambda s: print(s)
):
    """
    Prepara UNA sola vez el NIfTI y ejecuta:
      out_total/ con task=total y total_rois
      out_body/  con task=body  y body_rois
    Comparte hilos/flags comunes y respeta work_dir local.
    """
    ensure_dir(output_root)
    # Logging
    log_file = output_root / "run_dual.log"
    def log(msg: str):
        on_log(msg)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("=== DUAL PASS (total + body) ===")
    dev = detect_device()
    log(
        f"Dispositivo detectado: backend={dev['backend']} avail={dev['available']} "
        f"device={dev['device']} torch={dev['torch_version']} cuda={dev['cuda_version']} "
        f"name={dev['device_name']} VRAM={dev['vram_free_gb']}/{dev['vram_total_gb']} GB"
    )

    # Preparar work_dir local
    work_dir = get_work_dir(output_root)
    log(f"[WORK] Carpeta de trabajo local: {work_dir}")

    # Preparar true_input (zip/carpeta/NIfTI)
    true_input = input_path
    if input_path.suffix.lower() == ".zip":
        log(f"Descomprimiendo ZIP en carpeta local…")
        temp_dir = unzip_to_work(input_path, work_dir)
        log(f"ZIP extraído en: {temp_dir}")
        true_input = temp_dir
    if not Path(true_input).exists():
        raise FileNotFoundError(f"Entrada no encontrada: {true_input}")

    # Convertir solo una vez si es DICOM
    if is_nii(true_input):
        nii_input = true_input
        log(f"Entrada ya es NIfTI: {Path(nii_input).name}")
    else:
        if robust_import:
            log("Iniciando conversión robusta DICOM→NIfTI (series + dcm2niix/dicom2nifti)…")
            nii_input = robust_dicom_to_nifti(Path(true_input), work_dir, on_log=log)
            log(f"NIfTI intermedio: {nii_input}")
        else:
            raise RuntimeError("Para dual pass desde DICOM, activa importación robusta.")

    # --- PASADA 1: total ---
    if total_rois:
        out_total = output_root / "out_total"
        ensure_dir(out_total)
        args_total = replace_roi_subset(extra_common, total_rois)
        log(f"[TOTAL] {len(total_rois)} clases: {', '.join(total_rois)}")
        rc = run_totalsegmentator(nii_input, out_total, "total", device_mode, args_total, log, work_dir=work_dir)
        if rc != 0 and device_mode == "auto":
            log("⚠️ TOTAL falló en auto. Reintento CPU/--fast…")
            rc = run_totalsegmentator(nii_input, out_total, "total", "cpu_fast", args_total, log, work_dir=work_dir)
        if rc != 0:
            raise RuntimeError("Fallo en pasada TOTAL incluso tras retry.")

    # --- PASADA 2: body ---
    if body_rois:
        out_body = output_root / "out_body"
        ensure_dir(out_body)
        args_body = replace_roi_subset(extra_common, body_rois)
        log(f"[BODY] {len(body_rois)} clases: {', '.join(body_rois)}")
        rc = run_totalsegmentator(nii_input, out_body, "body", device_mode, args_body, log, work_dir=work_dir)
        if rc != 0 and device_mode == "auto":
            log("⚠️ BODY falló en auto. Reintento CPU/--fast…")
            rc = run_totalsegmentator(nii_input, out_body, "body", "cpu_fast", args_body, log, work_dir=work_dir)
        if rc != 0:
            raise RuntimeError("Fallo en pasada BODY incluso tras retry.")

    # Limpieza del work
    try:
        if _is_subpath(output_root, work_dir) and work_dir.name == DEFAULT_WORK_SUBDIR:
            safe_rmtree(work_dir)
            log(f"Work eliminado: {work_dir}")
        else:
            log(f"[SKIP] Work no eliminado por seguridad: {work_dir}")
    except Exception as e:
        log(f"No se pudo eliminar work: {e}")

    log("=== DUAL PASS FIN ===")

# ---------------------- GUI ---------------------- #
class TextRedirector:
    def __init__(self, widget: tk.Text): self.widget = widget
    def write(self, s): self.widget.after(0, self._append, s)
    def flush(self): pass
    def _append(self, s): self.widget.insert(tk.END, s); self.widget.see(tk.END)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TotalSegmentator GUI (One-File, Robust) - SOM3D")
        self.geometry("1000x760")

        self.input_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.task = tk.StringVar(value="total")
        # Modo de dispositivo: auto | cpu | cpu_fast
        self.device_mode = tk.StringVar(value="auto")
        self.extra_args = tk.StringVar()
        self.use_robust = tk.BooleanVar(value=True)  # por defecto activado

        # NUEVO: Toggles
        self.only_ortho = tk.BooleanVar(value=False)      # preset SOLO ORTOPEDIA (dual pass)
        self.auto_threads = tk.BooleanVar(value=True)     # auto-equilibrar hilos

        # Cabecera: device
        dev_frame = ttk.LabelFrame(self, text="Aceleración detectada")
        dev_frame.pack(fill="x", padx=10, pady=8)
        self.dev_label = ttk.Label(dev_frame, text="Detectando dispositivo…")
        self.dev_label.pack(anchor="w", padx=10, pady=6)
        self._show_device()

        # Entradas/salidas
        io = ttk.LabelFrame(self, text="Entradas y salidas")
        io.pack(fill="x", padx=10, pady=8)

        r = ttk.Frame(io); r.pack(fill="x", padx=10, pady=4)
        ttk.Label(r, text="ZIP / Carpeta DICOM / NIfTI:").pack(side="left")
        ttk.Entry(r, textvariable=self.input_path).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(r, text="Elegir…", command=self.browse_input).pack(side="left")

        r = ttk.Frame(io); r.pack(fill="x", padx=10, pady=4)
        ttk.Label(r, text="Carpeta de salida (.nii):").pack(side="left")
        ttk.Entry(r, textvariable=self.output_dir).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(r, text="Elegir…", command=self.browse_output).pack(side="left")

        # Opciones
        opts = ttk.LabelFrame(self, text="Opciones")
        opts.pack(fill="x", padx=10, pady=8)

        r = ttk.Frame(opts); r.pack(fill="x", padx=10, pady=6)
        ttk.Label(r, text="Task:").pack(side="left")
        self.task_cb = ttk.Combobox(r, textvariable=self.task, values=TASKS, width=30, state="readonly")
        self.task_cb.pack(side="left", padx=8)
        ttk.Label(r, text="(doble click para editar manual)").pack(side="left")
        self.task_cb.bind("<Double-Button-1>", lambda e: self.task_cb.configure(state="normal"))

        # Modos de dispositivo
        r = ttk.LabelFrame(opts, text="Modo de dispositivo")
        r.pack(fill="x", padx=10, pady=6)
        ttk.Radiobutton(r, text="Auto (GPU si hay)", value="auto", variable=self.device_mode).pack(anchor="w")
        ttk.Radiobutton(r, text="Solo CPU", value="cpu", variable=self.device_mode).pack(anchor="w")
        ttk.Radiobutton(r, text="CPU + --fast", value="cpu_fast", variable=self.device_mode).pack(anchor="w")

        r = ttk.Frame(opts); r.pack(fill="x", padx=10, pady=6)
        ttk.Checkbutton(r, text="Importación robusta (recomendada)", variable=self.use_robust).pack(side="left")

        r = ttk.Frame(opts); r.pack(fill="x", padx=10, pady=6)
        ttk.Checkbutton(r, text="Modo SOLO ORTOPEDIA (hueso + envolventes = dual pass)", variable=self.only_ortho).pack(side="left")

        r = ttk.Frame(opts); r.pack(fill="x", padx=10, pady=6)
        ttk.Checkbutton(r, text="Auto-equilibrar hilos (resample/guardado) según GPU/CPU", variable=self.auto_threads).pack(side="left")

        r = ttk.Frame(opts); r.pack(fill="x", padx=10, pady=6)
        ttk.Label(r, text="Extra args:").pack(side="left")
        ttk.Entry(r, textvariable=self.extra_args).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Label(r, text='Ej: --ml --statistics --preview').pack(side="left")

        # Botones
        run_frame = ttk.Frame(self); run_frame.pack(fill="x", padx=10, pady=8)
        ttk.Button(run_frame, text="Ejecutar", command=self.run_clicked).pack(side="left")
        ttk.Button(run_frame, text="Limpiar log", command=self.clear_log).pack(side="left", padx=8)

        # Consola
        logf = ttk.LabelFrame(self, text="Consola / Log")
        logf.pack(fill="both", expand=True, padx=10, pady=8)
        self.text = tk.Text(logf, height=20, wrap="word")
        self.text.pack(fill="both", expand=True, padx=6, pady=6)

        self._stdout_bak = sys.stdout
        sys.stdout = TextRedirector(self.text)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _show_device(self):
        dev = detect_device()
        devtxt = (
            f"Backend: {dev['backend']} | Disponible: {dev['available']} | "
            f"Dispositivo: {dev['device']} | Torch: {dev['torch_version']} | "
            f"CUDA: {dev['cuda_version']} | GPUs: {dev['device_count']} | "
            f"Nombre: {dev['device_name']} | VRAM: {dev['vram_free_gb']}/{dev['vram_total_gb']} GB"
        )
        self.dev_label.config(text=devtxt)

    def browse_input(self):
        path = filedialog.askopenfilename(
            title="Selecciona ZIP o NIfTI (o Cancela para elegir carpeta DICOM)",
            filetypes=[("ZIP", "*.zip"), ("NIfTI", "*.nii *.nii.gz"), ("Todos", "*.*")]
        )
        if path:
            self.input_path.set(path)
        else:
            folder = filedialog.askdirectory(title="Selecciona carpeta DICOM")
            if folder:
                self.input_path.set(folder)

    def browse_output(self):
        folder = filedialog.askdirectory(title="Selecciona carpeta de salida")
        if folder:
            self.output_dir.set(folder)

    def clear_log(self): self.text.delete("1.0", tk.END)

    def on_close(self):
        try: sys.stdout = self._stdout_bak
        except Exception: pass
        self.destroy()

    def run_clicked(self):
        in_path = self.input_path.get().strip()
        out_dir = self.output_dir.get().strip()
        task = (self.task.get().strip() or "total")
        device_mode = self.device_mode.get().strip() or "auto"
        extra = self.extra_args.get().strip()
        robust = self.use_robust.get()

        if not in_path:
            messagebox.showerror("Falta entrada", "Selecciona un archivo .zip/.nii o carpeta DICOM.")
            return
        if not out_dir:
            messagebox.showerror("Falta salida", "Selecciona una carpeta de salida.")
            return

        in_path = Path(in_path)
        out_dir = Path(out_dir)

        # Construye extra_list editable
        extra_list = shlex.split(extra) if extra else []

        # Normaliza si el usuario escribió '--roi_subset a,b,c'
        if extra_list:
            extra_list = normalize_roi_subset_args(extra_list)

        # --- PRESET ORTOPEDIA: preparar listas para dual pass ---
        ortho_total = []
        ortho_body  = []
        if self.only_ortho.get():
            # NO inyectamos aquí --roi_subset; lo haremos por pasada
            extra_list = remove_flag_and_value(extra_list, "--roi_subset")
            ortho_total = [r for r in ORTHO_ROI if r in VALID_TOTAL]
            ortho_body  = [r for r in ORTHO_ROI if r in VALID_BODY]

        # Auto threads (según VRAM/CPU)
        if self.auto_threads.get():
            dev = detect_device()
            n_resamp, n_saving = suggest_threads(device_mode, dev)
            extra_list = remove_flag_and_value(extra_list, "--nr_thr_resamp")
            extra_list = remove_flag_and_value(extra_list, "--nr_thr_saving")
            extra_list += ["--nr_thr_resamp", str(n_resamp), "--nr_thr_saving", str(n_saving)]
            print(f"[AUTO] Hilos sugeridos → resample={n_resamp} saving={n_saving}")

        t = threading.Thread(
            target=self._worker,
            args=(in_path, out_dir, task, device_mode, extra_list, robust, ortho_total, ortho_body),
            daemon=True
        )
        t.start()

    def _worker(self, in_path: Path, out_dir: Path, task: str, device_mode: str, extra_list, robust: bool,
                ortho_total: list[str], ortho_body: list[str]):
        try:
            print("=== TotalSegmentator (One-File GUI, Robust) ===")
            print(f"Entrada : {in_path}")
            print(f"Salida  : {out_dir}")
            print(f"Task    : {task}")
            print(f"Modo    : {device_mode}")
            print(f"Robusto : {robust}")
            print(f"Extra   : {extra_list}")

            # --- PRESET ORTOPEDIA → DUAL PASS ---
            if (ortho_total or ortho_body):
                print("[PRESET ORTOPEDIA] Activado → DUAL PASS con ORTHO_ROI dividido por task")
                if ortho_total:
                    print(f"[PRESET ORTOPEDIA] TOTAL ({len(ortho_total)}): {', '.join(ortho_total)}")
                if ortho_body:
                    print(f"[PRESET ORTOPEDIA] BODY  ({len(ortho_body)}): {', '.join(ortho_body)}")

                extra_common = replace_roi_subset(extra_list, [])
                run_dual_pass(
                    input_path=in_path,
                    output_root=out_dir,
                    total_rois=ortho_total,
                    body_rois=ortho_body,
                    device_mode=device_mode,
                    extra_common=extra_common,
                    robust_import=robust,
                    on_log=lambda s: print(s)
                )
                print("✔️ Listo (dual pass ORTOPEDIA). Revisa 'run_dual.log'.")
                return

            # --- Si usuario mezcló ROIs de ambos mundos, también dual pass ---
            rois = extract_roi_list(extra_list)
            if rois:
                print(f"[ROI_SUBSET] {len(rois)} solicitadas: {', '.join(rois)}")
            total_rois = [r for r in rois if r in VALID_TOTAL]
            body_rois  = [r for r in rois if r in VALID_BODY]
            dropped    = sorted(set(rois) - set(total_rois) - set(body_rois))
            if "hip_implant" in rois:
                print("[WARN] 'hip_implant' no existe en tasks 'total' ni 'body'; se descarta.")
            if dropped:
                print(f"[WARN] ROI no válidas para 'total'/'body': {', '.join(dropped)}")

            if (total_rois and body_rois):
                print("→ Mezcla de ROIs total/body detectada. Ejecutando DUAL PASS…")
                extra_common = replace_roi_subset(extra_list, [])
                run_dual_pass(
                    input_path=in_path,
                    output_root=out_dir,
                    total_rois=total_rois,
                    body_rois=body_rois,
                    device_mode=device_mode,
                    extra_common=extra_common,
                    robust_import=robust,
                    on_log=lambda s: print(s)
                )
                print("✔️ Listo (dual pass). Revisa 'run_dual.log'.")
                return

            # --- Pasada única (normal) ---
            # Si solo hay body_rois y task != body, forzamos body
            task_local = task
            extra_local = extra_list
            if body_rois and not total_rois:
                if task != "body":
                    print("[INFO] Solo hay ROIs de 'body'. Forzando task='body'.")
                    task_local = "body"
                extra_local = replace_roi_subset(extra_list, body_rois)
            elif total_rois:
                extra_local = replace_roi_subset(extra_list, total_rois)

            eff = extract_roi_list(extra_local)
            if eff:
                print(f"[ROI_EFECTIVAS] {len(eff)}: {', '.join(eff)}")

            run_segmentation(
                input_path=in_path,
                output_path=out_dir,
                task=task_local,
                device_mode=device_mode,
                keep_temp=False,                        # ← asegura eliminación del work
                extra_args=extra_local,
                robust_import=robust,
                on_log=lambda s: print(s),
                use_local_work=True,                    # ← usar _work local
                custom_work_dir=out_dir / "_work"       # ← ruta del work
            )
            print("✔️ Listo. Revisa también 'run.log' en la salida.")
        except Exception as e:
            print(f"❌ Error: {e}")
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    App().mainloop()
