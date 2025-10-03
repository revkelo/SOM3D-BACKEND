#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SOM3D TotalSegmentator GUI (GPU→CPU fallback, Ortopedia→Body)
- Siempre intenta GPU; si falla, cae a CPU (sin --fast).
- Única opción visible: GPU --fast (opcional).
- Siempre genera estadísticas (--statistics).
- Solo preset ORTOPEDIA (inyecta --roi_subset osteo-articular).
- Auto-hilos según VRAM/CPU; importación DICOM robusta.
- TMP en mismo disco que la salida para reducir I/O cruzado.
- Banderas de estado por etapa + tiempos por paso.
- NOTA: Forzado a NO usar dcm2niix (solo dicom2nifti).
"""

import os, sys, shlex, zipfile, tempfile, subprocess, threading, shutil, time, queue, re, json
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from contextlib import contextmanager  # (solo una vez)

# ---------- Constantes ----------
ORTHO_ROI = [
    "skull","sacrum",
    "vertebrae_C1","vertebrae_C2","vertebrae_C3","vertebrae_C4","vertebrae_C5","vertebrae_C6","vertebrae_C7",
    "vertebrae_T1","vertebrae_T2","vertebrae_T3","vertebrae_T4","vertebrae_T5","vertebrae_T6","vertebrae_T7","vertebrae_T8","vertebrae_T9","vertebrae_T10","vertebrae_T11","vertebrae_T12",
    "vertebrae_L1","vertebrae_L2","vertebrae_L3","vertebrae_L4","vertebrae_L5","vertebrae_S1",
    "rib_left_1","rib_left_2","rib_left_3","rib_left_4","rib_left_5","rib_left_6","rib_left_7","rib_left_8","rib_left_9","rib_left_10","rib_left_11","rib_left_12",
    "rib_right_1","rib_right_2","rib_right_3","rib_right_4","rib_right_5","rib_right_6","rib_right_7","rib_right_8","rib_right_9","rib_right_10","rib_right_11","rib_right_12",
    "sternum","costal_cartilages",
    "clavicula_left","clavicula_right","scapula_left","scapula_right",
    "humerus_left","humerus_right","femur_left","femur_right","hip_left","hip_right",
]

os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS","1")
os.environ.setdefault("PYTHONUTF8","1")

# ---------- Utilidades ----------
def format_duration(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_sec = total_ms // 1000
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def detect_device_once():
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
        if info["cuda_available"]:
            idx = torch.cuda.current_device()
            prop = torch.cuda.get_device_properties(idx)
            info.update({
                "backend":"cuda","available":True,"device":f"cuda:{idx}",
                "device_count": torch.cuda.device_count(),
                "device_name": prop.name
            })
            try:
                free, total = torch.cuda.mem_get_info(idx)
                info["vram_free_gb"]  = round(free /(1024**3), 2)
                info["vram_total_gb"] = round(total/(1024**3), 2)
            except Exception:
                pass
    except Exception:
        pass
    return info

def device_summary_text(dev: dict) -> str:
    return (
        f"Backend: {dev['backend']} | Disponible: {dev['available']} | "
        f"Dispositivo: {dev['device']} | Torch: {dev['torch_version']} | "
        f"CUDA: {dev['cuda_version']} | GPUs: {dev['device_count']} | "
        f"Nombre: {dev['device_name']} | VRAM: {dev['vram_free_gb']}/{dev['vram_total_gb']} GB"
    )

def is_nii(path: Path) -> bool:
    n = path.name.lower()
    return n.endswith(".nii") or n.endswith(".nii.gz")

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def _safe_extractall(zf: zipfile.ZipFile, dest: Path):
    dest = dest.resolve()
    for member in zf.infolist():
        out_path = (dest / member.filename).resolve()
        if not str(out_path).startswith(str(dest)):
            raise RuntimeError(f"Entrada ZIP inválida: {member.filename}")
    zf.extractall(dest)

def unzip_to_temp(zip_path: Path, base_tmp: Path) -> Path:
    tmpdir = base_tmp / f"ts_zip_{int(time.time()*1000)}"
    tmpdir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        _safe_extractall(zf, tmpdir)
    subs = [p for p in tmpdir.iterdir() if p.is_dir()]
    return subs[0] if len(subs) == 1 else tmpdir

def pick_largest_nii(folder: Path) -> Path | None:
    largest = None
    largest_size = -1
    for ext in ("*.nii", "*.nii.gz"):
        for p in folder.rglob(ext):
            s = p.stat().st_size
            if s > largest_size:
                largest_size = s
                largest = p
    return largest

# --- util: merge stats en un solo archivo ---
def merge_and_cleanup_stats(out_dir: Path, on_log=print):
    out_dir = Path(out_dir)
    stats = out_dir / "statistics.json"
    stats_all = out_dir / "statistics_all.json"
    if not stats.exists():
        return
    try:
        new = json.loads(stats.read_text(encoding="utf-8"))
    except Exception as e:
        on_log(f"[STATS] ERROR leyendo statistics.json: {e}")
        return
    base = {}
    if stats_all.exists():
        try:
            base = json.loads(stats_all.read_text(encoding="utf-8"))
        except Exception:
            base = {}
    # mezcla (lo nuevo sobreescribe claves iguales)
    base.update(new)
    stats_all.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        stats.unlink()
        on_log("[STATS] Consolidado en statistics_all.json y eliminado statistics.json")
    except Exception:
        pass

# ---------- Banderas / timing ----------
def flag(on_log, label: str, status: str, extra: str = ""):
    line = f"[{label}] {status}{(' | ' + extra) if extra else ''}"
    on_log(line)

@contextmanager
def step(on_log, label: str, status: str):
    t0 = time.perf_counter()
    flag(on_log, label, f"{status}…")
    try:
        yield
        dt = time.perf_counter() - t0
        flag(on_log, label, f"OK", f"dur={format_duration(dt)}")
    except Exception as e:
        dt = time.perf_counter() - t0
        flag(on_log, label, f"ERROR", f"{e} | dur={format_duration(dt)}")
        raise

# ---------- DICOM series ----------
def split_series(input_dir: Path, on_log=print) -> dict[str, list[Path]]:
    try:
        import pydicom as dcm
    except Exception:
        flag(on_log, "DICOM", "pydicom no disponible; no se separa por serie")
        return {}
    series = {}
    for root, _, files in os.walk(input_dir):
        root_p = Path(root)
        for name in files:
            f = root_p / name
            try:
                ds = dcm.dcmread(str(f), stop_before_pixels=True, force=True)
                uid = str(getattr(ds, 'SeriesInstanceUID', 'UNKNOWN'))
                series.setdefault(uid, []).append(f)
            except Exception:
                continue
    for uid, lst in series.items():
        flag(on_log, "DICOM", f"Serie {uid[:8]}…", f"{len(lst)} archivos")
    return series

def select_best_series(input_dir: Path, on_log=print) -> Path | None:
    with step(on_log, "DICOM", "Analizando series"):
        series = split_series(input_dir, on_log)
        if not series:
            return None
        uid, files = max(series.items(), key=lambda kv: len(kv[1]))
        tmp = input_dir.parent / f"ts_series_{int(time.time()*1000)}"
        tmp.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(sorted(files)):
            dst = tmp / f"img_{i:05d}.dcm"
            try: shutil.copy2(f, dst)
            except Exception: pass
        flag(on_log, "DICOM", "Serie seleccionada", f"{tmp} | {len(files)} DICOMs")
        return tmp

# ---------- Conversión DICOM -> NIfTI ----------
def decompress_series_with_pydicom(input_dir: Path, out_dir: Path, on_log=print) -> Path | None:
    try:
        import pydicom as dcm
        from pydicom.uid import ExplicitVRLittleEndian
        ensure_dir(out_dir); count = 0
        for root, _, files in os.walk(input_dir):
            root_p = Path(root)
            for name in sorted(files):
                f = root_p / name
                try:
                    ds = dcm.dcmread(str(f), force=True)
                    ts = ds.file_meta.TransferSyntaxUID
                    if getattr(ts, 'is_compressed', False):
                        _ = ds.pixel_array  # fuerza decodificación
                        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
                        ds.is_implicit_VR = False; ds.is_little_endian = True
                    dst = out_dir / f"{count:05d}.dcm"
                    ds.save_as(str(dst), write_like_original=False); count += 1
                except Exception as e:
                    flag(on_log, "DICOM", f"No reescrito {f.name}", str(e))
        flag(on_log, "DICOM", "Reescritura", f"{count} DICOMs en {out_dir}")
        return out_dir if count > 0 else None
    except Exception as e:
        flag(on_log, "DICOM", "pydicom/pylibjpeg no disponibles", str(e))
        return None

def convert_with_dicom2nifti_relaxed(input_dir: Path, out_dir: Path, on_log=print) -> Path | None:
    try:
        import dicom2nifti
        from dicom2nifti import settings
        ensure_dir(out_dir)
        for fn in ('disable_validate_orientation','disable_validate_orthogonal',
                   'disable_validate_slice_increment','disable_validate_instance_number',
                   'disable_validate_woodpecker'):
            if hasattr(settings, fn): getattr(settings, fn)()
        flag(on_log, "NIFTI", "dicom2nifti (relajado)", "convirtiendo…")
        dicom2nifti.convert_directory(str(input_dir), str(out_dir),
                                      compression=True, reorient=True)
        nii = pick_largest_nii(out_dir)
        if nii: flag(on_log, "NIFTI", "dicom2nifti OK", nii.name)
        return nii
    except Exception as e:
        flag(on_log, "NIFTI", "dicom2nifti error", str(e))
        try:
            tmp_dec = out_dir.parent / f"ts_dec_{int(time.time()*1000)}"
            if decompress_series_with_pydicom(input_dir, tmp_dec, on_log=on_log):
                flag(on_log, "NIFTI", "Reintento", "sobre serie descomprimida…")
                return convert_with_dicom2nifti_relaxed(tmp_dec, out_dir, on_log)
        except Exception:
            pass
        return None

def robust_dicom_to_nifti(input_dir: Path, work_tmp: Path, on_log=print) -> Path:
    with step(on_log, "NIFTI", "Conversión robusta DICOM→NIfTI"):
        best_dir = select_best_series(input_dir, on_log) or input_dir
        tmp_out = work_tmp / f"ts_conv_{int(time.time()*1000)}"
        flag(on_log, "NIFTI", "Salida intermedia", str(tmp_out))
        nii = convert_with_dicom2nifti_relaxed(best_dir, tmp_out, on_log=on_log)
        if not nii: raise RuntimeError("Conversión DICOM→NIfTI falló.")
        return nii

# ---------- Hilos ----------
def suggest_threads(dev_info: dict) -> tuple[int, int]:
    cpu_count = max(os.cpu_count() or 2, 2)
    if dev_info.get("available"):
        vram = dev_info.get("vram_total_gb") or 0
        if vram >= 12: return (6,6)
        if vram >= 8:  return (4,4)
        if vram >= 6:  return (3,3)
        if vram >= 4:  return (2,2)
        return (1,1)
    n = min(max(cpu_count // 2, 2), 8)
    return (n, n)

# ---------- Runner ----------
_ansi_re = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

def _sanitize_line(s: str) -> str:
    s = _ansi_re.sub("", s)
    return s.replace("\r", "").rstrip("\n")

def run_with_streaming(cmd, env, on_line) -> int:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    if proc.stdout is None:
        return proc.wait()
    for line in proc.stdout:
        on_line(_sanitize_line(line))
    return proc.wait()

# ==== builders: aceptan task ====
def build_cmd_gpu(input_nii: Path, output_dir: Path, task: str, extra_flags: list[str], fast_gpu: bool) -> list[str]:
    cmd = ["TotalSegmentator", "-i", str(input_nii), "-o", str(output_dir), "--task", task, "--device", "gpu"]
    if fast_gpu:
        cmd.append("--fast")
    cmd += extra_flags
    return cmd

def build_cmd_cpu(input_nii: Path, output_dir: Path, task: str, extra_flags: list[str]) -> list[str]:
    return ["TotalSegmentator","-i",str(input_nii),"-o",str(output_dir),
            "--task",task,"--device","cpu", *extra_flags]

# ==== ejecución GPU→CPU por task ====
def run_totalsegmentator_gpu_then_cpu(input_nii: Path, output_dir: Path,
                                      task: str, extra_flags: list[str], fast_gpu: bool, on_log):
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("TQDM_DISABLE", "1")

    cmd_gpu = build_cmd_gpu(input_nii, output_dir, task, extra_flags, fast_gpu)
    with step(on_log, "TS|GPU", f"Ejecución {task}"):
        flag(on_log, "TS|GPU", "cmd", " ".join(shlex.quote(x) for x in cmd_gpu))
        rc = run_with_streaming(cmd_gpu, env, on_line=on_log)
        flag(on_log, "TS|GPU", "rc", str(rc))
    if rc == 0:
        return rc

    env["CUDA_VISIBLE_DEVICES"] = "-1"
    flag(on_log, "TS|GPU", f"rc={rc}", f"Fallback a CPU sin --fast ({task})")
    cmd_cpu = build_cmd_cpu(input_nii, output_dir, task, extra_flags)
    with step(on_log, "TS|CPU", f"Ejecución {task}"):
        flag(on_log, "TS|CPU", "cmd", " ".join(shlex.quote(x) for x in cmd_cpu))
        return run_with_streaming(cmd_cpu, env, on_line=on_log)

def normalize_roi_subset_args(args: list[str]) -> list[str]:
    out = []; i = 0
    while i < len(args):
        a = args[i]
        if a == "--roi_subset":
            out.append(a); i += 1
            while i < len(args) and not args[i].startswith("--"):
                token = args[i]
                out.extend([t for t in token.split(",") if t]) if "," in token else out.append(token)
                i += 1
            continue
        out.append(a); i += 1
    return out

def preflight_or_fail(on_log):
    with step(on_log, "START", "Preflight"):
        if not shutil.which("TotalSegmentator"):
            raise RuntimeError("No se encontró 'TotalSegmentator' en PATH.")

def best_tmp_for(output_path: Path) -> Path:
    try:
        out = output_path.resolve()
    except Exception:
        out = output_path
    if os.name == "nt":
        drive = out.drive or Path(tempfile.gettempdir()).drive or "C:"
        base = Path(drive + "\\") / "_ts_tmp"
    else:
        anchor = out.anchor or "/"
        base = Path(anchor) / "_ts_tmp"
        try:
            base.mkdir(parents=True, exist_ok=True)
        except Exception:
            base = Path(tempfile.gettempdir()) / "_ts_tmp"
    base.mkdir(parents=True, exist_ok=True)
    return base

def run_pipeline(
    input_path: Path,
    output_path: Path,
    fast_gpu: bool,
    robust_import: bool,
    dev_info: dict,
    on_log=print
) -> float:
    t0_total = time.perf_counter()
    start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ensure_dir(output_path)
    log_file = output_path / "run.log"

    def log(msg: str):
        on_log(msg)
        try:
            with log_file.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass

    preflight_or_fail(log)
    flag(log, "START", "TotalSegmentator (GPU→CPU, Ortopedia→Body)")
    flag(log, "START", "Inicio", start_ts)
    flag(log, "START", "Dispositivo", device_summary_text(dev_info))

    with step(log, "TMP", "Preparando área temporal"):
        tmp_root = best_tmp_for(output_path)
        flag(log, "TMP", "Base", str(tmp_root))

    true_input = Path(input_path)
    temp_mount = None
    try:
        with step(log, "IMPORT", "Resolviendo entrada"):
            if true_input.suffix.lower() == ".zip":
                flag(log, "IMPORT", "ZIP", "Descomprimiendo…")
                temp_mount = unzip_to_temp(true_input, tmp_root)
                flag(log, "IMPORT", "ZIP extraído", str(temp_mount))
                true_input = temp_mount
            if not true_input.exists():
                raise FileNotFoundError(f"Entrada no encontrada: {true_input}")

        if is_nii(true_input):
            flag(log, "NIFTI", "Entrada NIfTI", Path(true_input).name)
            nii_input = true_input
        else:
            if not robust_import:
                raise RuntimeError("Se requiere NIfTI o robust_import=True para DICOM.")
            nii_input = robust_dicom_to_nifti(true_input, tmp_root, on_log=log)
            flag(log, "NIFTI", "Intermedio", str(nii_input))

        # ===== Hilos y flags base =====
        with step(log, "THREADS", "Calculando hilos"):
            n_resamp, n_saving = suggest_threads(dev_info)
            flags_common = ["--statistics", "--nr_thr_resamp", str(n_resamp), "--nr_thr_saving", str(n_saving)]
            flag(log, "THREADS", "resample/saving", f"{n_resamp}/{n_saving}")

        # ===== Paso 1: ORTOPEDIA (total + roi_subset + statistics) =====
        orto_flags = normalize_roi_subset_args(["--roi_subset", *ORTHO_ROI] + flags_common)
        flag(log, "TS", "Resumen ORTO", f"task=total | fast_gpu={fast_gpu} | stats=ON")
        t0_orto = time.perf_counter()
        rc1 = run_totalsegmentator_gpu_then_cpu(Path(nii_input), output_path, "total", orto_flags, fast_gpu, on_log=log)
        if rc1 != 0:
            raise RuntimeError("TotalSegmentator (ORTOPEDIA) falló incluso tras fallback CPU.")
        t_orto = time.perf_counter() - t0_orto
        flag(log, "END|ORTO", "Duración", format_duration(t_orto))
        # Consolidar estadísticas del paso ORTO
        merge_and_cleanup_stats(output_path, on_log=log)

        # ===== Paso 3: HIP_IMPLANT (con estadísticas) =====
        hip_flags = list(flags_common)  # incluye --statistics y hilos
        flag(log, "TS", "Resumen HIP_IMPLANT",
            f"task=hip_implant | out={output_path} | fast_gpu={fast_gpu} | stats=ON")

        t0_hip = time.perf_counter()
        rc_hip = run_totalsegmentator_gpu_then_cpu(
            Path(nii_input), output_path, "hip_implant", hip_flags, fast_gpu, on_log=log
        )
        if rc_hip != 0:
            raise RuntimeError("TotalSegmentator (HIP_IMPLANT) falló incluso tras fallback CPU.")
        t_hip = time.perf_counter() - t0_hip
        flag(log, "END|HIP_IMPLANT", "Duración", format_duration(t_hip))

        # Consolidar estadísticas acumuladas (BODY + HIP_IMPLANT)
        merge_and_cleanup_stats(output_path, on_log=log)

    finally:
        with step(log, "CLEANUP", "Eliminando temporales"):
            if temp_mount:
                try:
                    shutil.rmtree(temp_mount, ignore_errors=True)
                    flag(log, "CLEANUP", "Temp eliminado", str(temp_mount))
                except Exception as e:
                    flag(log, "CLEANUP", "WARN", f"No se pudo borrar temp: {e}")

    elapsed_total = time.perf_counter() - t0_total
    end_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    flag(log, "END", "Fin", end_ts)
    flag(log, "END", "Duración total (ORTO → BODY)", format_duration(elapsed_total))
    flag(log, "END", "Listo", "Ver también run.log")
    return elapsed_total

# ---------- UI ----------
class BufferedLogger:
    """Reduce llamadas a Tk: acumula y drena en lotes."""
    def __init__(self, text_widget: tk.Text, flush_ms: int = 40):
        self.text = text_widget
        self.q = queue.Queue()
        self.flush_ms = flush_ms
        self.running = False

    def start(self, root: tk.Tk):
        if self.running: return
        self.running = True
        def _drain():
            if not self.running: return
            try:
                chunks = []
                while True:
                    chunks.append(self.q.get_nowait())
            except queue.Empty:
                pass
            if chunks:
                self.text.insert(tk.END, "".join(chunks))
                self.text.see(tk.END)
            root.after(self.flush_ms, _drain)
        root.after(self.flush_ms, _drain)

    def stop(self): self.running = False
    def write(self, s: str): self.q.put(s if s.endswith("\n") else s + "\n")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SOM3D - TotalSegmentator (GPU→CPU, Ortopedia→Body)")
        self.geometry("960x780")

        self.input_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.fast_gpu = tk.BooleanVar(value=False)

        self.dev_info = detect_device_once()

        dev_frame = ttk.LabelFrame(self, text="Aceleración detectada")
        dev_frame.pack(fill="x", padx=10, pady=8)
        self.dev_label = ttk.Label(dev_frame, text=device_summary_text(self.dev_info))
        self.dev_label.pack(anchor="w", padx=10, pady=6)

        io = ttk.LabelFrame(self, text="Entradas y salidas")
        io.pack(fill="x", padx=10, pady=8)

        r = ttk.Frame(io); r.pack(fill="x", padx=10, pady=4)
        ttk.Label(r, text="ZIP / Carpeta DICOM / NIfTI:").pack(side="left")
        ttk.Entry(r, textvariable=self.input_path).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(r, text="Elegir…", command=self.browse_input).pack(side="left")

        r = ttk.Frame(io); r.pack(fill="x", padx=10, pady=4)
        ttk.Label(r, text="Carpeta de salida:").pack(side="left")
        ttk.Entry(r, textvariable=self.output_dir).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(r, text="Elegir…", command=self.browse_output).pack(side="left")

        opts = ttk.LabelFrame(self, text="Opciones")
        opts.pack(fill="x", padx=10, pady=8)
        ttk.Checkbutton(opts, text="GPU --fast", variable=self.fast_gpu).pack(anchor="w")
        ttk.Label(opts, text="Secuencia fija: Ortopedia → Body (ambas con --statistics)").pack(anchor="w", pady=(6,0))

        logf = ttk.LabelFrame(self, text="Consola / Log")
        logf.pack(fill="both", expand=True, padx=10, pady=8)
        self.text = tk.Text(logf, height=22, wrap="word")
        self.text.pack(fill="both", expand=True, padx=6, pady=6)

        self.status = ttk.Label(self, text=device_summary_text(self.dev_info), relief="sunken", anchor="w")
        self.status.pack(fill="x", side="bottom")

        self.logger = BufferedLogger(self.text)
        self.logger.start(self)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        toolbar = ttk.Frame(self); toolbar.pack(fill="x", padx=10, pady=8)
        ttk.Button(toolbar, text="Ejecutar", command=self.pick_and_run).pack(side="left")
        ttk.Button(toolbar, text="Limpiar log", command=self.clear_log).pack(side="left", padx=8)

    def browse_input(self):
        path = filedialog.askopenfilename(
            title="Selecciona ZIP o NIfTI (o Cancela para carpeta DICOM)",
            filetypes=[("ZIP","*.zip"),("NIfTI","*.nii *.nii.gz"),("Todos","*.*")]
        )
        if path: self.input_path.set(path)
        else:
            folder = filedialog.askdirectory(title="Selecciona carpeta DICOM")
            if folder: self.input_path.set(folder)

    def browse_output(self):
        folder = filedialog.askdirectory(title="Selecciona carpeta de salida")
        if folder: self.output_dir.set(folder)

    def clear_log(self): self.text.delete("1.0", tk.END)

    def on_close(self):
        self.logger.stop()
        self.destroy()

    def pick_and_run(self):
        in_path = self.input_path.get().strip()
        out_dir = self.output_dir.get().strip()
        if not in_path:
            messagebox.showerror("Falta entrada", "Selecciona .zip/.nii o carpeta DICOM.")
            return
        if not out_dir:
            messagebox.showerror("Falta salida", "Selecciona una carpeta de salida.")
            return
        self._start_worker(Path(in_path), Path(out_dir))

    def _start_worker(self, in_path: Path, out_dir: Path):
        t = threading.Thread(
            target=self._worker,
            args=(in_path, out_dir, self.fast_gpu.get()),
            daemon=True
        )
        t.start()

    def _worker(self, in_path: Path, out_dir: Path, fast_gpu: bool):
        try:
            self.logger.write("=== TotalSegmentator (GPU→CPU, Ortopedia→Body) ===")
            self.logger.write(f"Entrada : {in_path}")
            self.logger.write(f"Salida  : {out_dir}")
            self.logger.write(f"GPU --fast        : {fast_gpu}")
            self.logger.write(f"Flujo             : ORTO → BODY (ambas con --statistics)")

            def on_log(s: str): self.logger.write(s)

            elapsed = run_pipeline(
                input_path=in_path,
                output_path=out_dir,
                fast_gpu=fast_gpu,
                robust_import=True,
                dev_info=self.dev_info,
                on_log=on_log
            )
            dur = format_duration(elapsed)
            self.logger.write(f"[END] ⏱ Duración total: {dur}")
            messagebox.showinfo("Completado", f"Segmentación terminada.\nDuración total: {dur}")
            self.logger.write("[END] ✔️ Listo. Revisa también 'run.log'.")
        except Exception as e:
            self.logger.write(f"[ERROR] {e}")
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    App().mainloop()
