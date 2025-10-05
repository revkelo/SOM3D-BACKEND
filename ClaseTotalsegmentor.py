#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SOM3D TotalSegmentator Runner (clase, sin GUI)
- Siempre ejecuta ORTOPEDIA (task=total con --roi_subset óseo-articular + --statistics)
- HIP_IMPLANT habilitable/deshabilitable
- --fast SOLO aplica al paso ORTOPEDIA (nunca a hip_implant ni a otros tasks)
- Permite tasks extra (p. ej. body, total_mr, lung_vessels, etc.) sin --fast
- GPU→CPU fallback
- Importación robusta: ZIP (DICOM), carpeta DICOM o NIfTI directo
- Hilos sugeridos según VRAM/CPU
- Logs tipo servidor a consola y a run.log en la carpeta de salida
"""
from __future__ import annotations
import os, shlex, zipfile, tempfile, subprocess, shutil, time, re, json
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from typing import Callable, Optional, List, Dict, Tuple

# ---------- Constantes ----------
ORTHO_ROI: List[str] = [
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

LICENSE_ENV_VARS: Tuple[str, ...] = ("TOTALSEG_LICENSE_KEY", "TOTALSEG_LICENSE")

# Afinar BLAS para no sobrecargar
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS","1")
os.environ.setdefault("PYTHONUTF8","1")

# ---------- Utils ----------
def format_duration(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_sec = total_ms // 1000
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def is_nii(path: Path) -> bool:
    n = path.name.lower()
    return n.endswith(".nii") or n.endswith(".nii.gz")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _safe_extractall(zf: zipfile.ZipFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in zf.infolist():
        out_path = (dest / member.filename).resolve()
        if not str(out_path).startswith(str(dest)):
            raise RuntimeError(f"Entrada ZIP inválida (path traversal): {member.filename}")
    zf.extractall(dest)

def pick_largest_nii(folder: Path) -> Optional[Path]:
    largest = None
    largest_size = -1
    for ext in ("*.nii","*.nii.gz"):
        for p in folder.rglob(ext):
            s = p.stat().st_size
            if s > largest_size:
                largest = p
                largest_size = s
    return largest

# ---------- Clase principal ----------
class TotalSegmentatorRunner:
    def __init__(
        self,
        fast_gpu: bool = False,                 # solo se aplicará al paso ORTOPEDIA
        robust_import: bool = True,
        enable_hip_implant: bool = True,        # ON/OFF para hip_implant
        extra_tasks: Optional[List[str]] = None,# p.ej. ["body","total_mr"]
        on_log: Optional[Callable[[str], None]] = None
    ):
        self.fast_gpu = fast_gpu
        self.robust_import = robust_import
        self.enable_hip_implant = enable_hip_implant
        self.extra_tasks = list(extra_tasks) if extra_tasks else []
        self.on_log = on_log or (lambda s: print(s, flush=True))
        self.dev_info = self._detect_device_once()
        self._license_checked = False

    # ---------- Logging ----------
    def _log(self, msg: str) -> None:
        self.on_log(msg)

    @staticmethod
    def _flag(on_log: Callable[[str], None], label: str, status: str, extra: str = "") -> None:
        line = f"[{label}] {status}{(' | ' + extra) if extra else ''}"
        on_log(line)

    @contextmanager
    def _step(self, label: str, status: str):
        t0 = time.perf_counter()
        self._flag(self.on_log, label, f"{status}…")
        try:
            yield
            dt = time.perf_counter() - t0
            self._flag(self.on_log, label, "OK", f"dur={format_duration(dt)}")
        except Exception as e:
            dt = time.perf_counter() - t0
            self._flag(self.on_log, label, "ERROR", f"{e} | dur={format_duration(dt)}")
            raise

    # ---------- Device ----------
    @staticmethod
    def _detect_device_once() -> Dict[str, object]:
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

    @staticmethod
    def _device_summary_text(dev: dict) -> str:
        return (
            f"Backend: {dev['backend']} | Disponible: {dev['available']} | "
            f"Dispositivo: {dev['device']} | Torch: {dev['torch_version']} | "
            f"CUDA: {dev['cuda_version']} | GPUs: {dev['device_count']} | "
            f"Nombre: {dev['device_name']} | VRAM: {dev['vram_free_gb']}/{dev['vram_total_gb']} GB"
        )

    @staticmethod
    def _mask_license(value: str) -> str:
        value = value.strip()
        if len(value) <= 8:
            return "*" * len(value)
        return f"{value[:4]}{'*' * (len(value) - 8)}{value[-4:]}"

    def _ensure_license(self) -> None:
        if getattr(self, "_license_checked", False):
            return
        cmd_path = shutil.which("totalseg_set_license")
        if not cmd_path:
            raise RuntimeError("No se encontró 'totalseg_set_license' en PATH.")
        license_key = None
        env_used = None
        for name in LICENSE_ENV_VARS:
            candidate = os.environ.get(name)
            if candidate:
                license_key = candidate.strip()
                env_used = name
                break
        if not license_key:
            joined = ", ".join(LICENSE_ENV_VARS)
            raise RuntimeError(f"No se definió la licencia en las variables de entorno: {joined}")
        masked = self._mask_license(license_key)
        self._flag(self.on_log, "LICENSE", "Aplicando", f"env={env_used} | valor={masked}")
        try:
            proc = subprocess.run([cmd_path, "-l", license_key], capture_output=True, text=True, check=False)
        except Exception as exc:
            raise RuntimeError(f"totalseg_set_license no pudo ejecutarse: {exc}") from exc
        if proc.returncode != 0:
            output = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"totalseg_set_license rc={proc.returncode}: {output}")
        message = (proc.stdout or proc.stderr or "").strip()
        if message:
            self._flag(self.on_log, "LICENSE", "Respuesta", message)
        self._license_checked = True

    # ---------- DICOM helpers ----------
    @staticmethod
    def _split_series(input_dir: Path, on_log: Callable[[str], None]) -> Dict[str, List[Path]]:
        try:
            import pydicom as dcm
        except Exception:
            TotalSegmentatorRunner._flag(on_log, "DICOM", "pydicom no disponible; no se separa por serie")
            return {}
        series: Dict[str, List[Path]] = {}
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
            TotalSegmentatorRunner._flag(on_log, "DICOM", f"Serie {uid[:8]}…", f"{len(lst)} archivos")
        return series

    def _select_best_series(self, input_dir: Path) -> Optional[Path]:
        with self._step("DICOM", "Analizando series"):
            series = self._split_series(input_dir, self.on_log)
            if not series:
                return None
            uid, files = max(series.items(), key=lambda kv: len(kv[1]))
            tmp = input_dir.parent / f"ts_series_{int(time.time()*1000)}"
            tmp.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(sorted(files)):
                dst = tmp / f"img_{i:05d}.dcm"
                try: shutil.copy2(f, dst)
                except Exception: pass
            self._flag(self.on_log, "DICOM", "Serie seleccionada", f"{tmp} | {len(files)} DICOMs")
            return tmp

    def _decompress_series_with_pydicom(self, input_dir: Path, out_dir: Path) -> Optional[Path]:
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
                        self._flag(self.on_log, "DICOM", f"No reescrito {f.name}", str(e))
            self._flag(self.on_log, "DICOM", "Reescritura", f"{count} DICOMs en {out_dir}")
            return out_dir if count > 0 else None
        except Exception as e:
            self._flag(self.on_log, "DICOM", "pydicom/pylibjpeg no disponibles", str(e))
            return None

    def _convert_with_dicom2nifti_relaxed(self, input_dir: Path, out_dir: Path) -> Optional[Path]:
        try:
            import dicom2nifti
            from dicom2nifti import settings
            ensure_dir(out_dir)
            for fn in ('disable_validate_orientation','disable_validate_orthogonal',
                       'disable_validate_slice_increment','disable_validate_instance_number',
                       'disable_validate_woodpecker'):
                if hasattr(settings, fn): getattr(settings, fn)()
            self._flag(self.on_log, "NIFTI", "dicom2nifti (relajado)", "convirtiendo…")
            dicom2nifti.convert_directory(str(input_dir), str(out_dir),
                                          compression=True, reorient=True)
            nii = pick_largest_nii(out_dir)
            if nii: self._flag(self.on_log, "NIFTI", "dicom2nifti OK", nii.name)
            return nii
        except Exception as e:
            self._flag(self.on_log, "NIFTI", "dicom2nifti error", str(e))
            try:
                tmp_dec = out_dir.parent / f"ts_dec_{int(time.time()*1000)}"
                if self._decompress_series_with_pydicom(input_dir, tmp_dec):
                    self._flag(self.on_log, "NIFTI", "Reintento", "sobre serie descomprimida…")
                    return self._convert_with_dicom2nifti_relaxed(tmp_dec, out_dir)
            except Exception:
                pass
            return None

    def _robust_dicom_to_nifti(self, input_dir: Path, work_tmp: Path) -> Path:
        with self._step("NIFTI", "Conversión robusta DICOM→NIfTI"):
            best_dir = self._select_best_series(input_dir) or input_dir
            tmp_out = work_tmp / f"ts_conv_{int(time.time()*1000)}"
            self._flag(self.on_log, "NIFTI", "Salida intermedia", str(tmp_out))
            nii = self._convert_with_dicom2nifti_relaxed(best_dir, tmp_out)
            if not nii: raise RuntimeError("Conversión DICOM→NIfTI falló.")
            return nii

    # ---------- Threads ----------
    @staticmethod
    def _suggest_threads(dev_info: dict) -> Tuple[int,int]:
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

    # ---------- TotalSegmentator runner ----------
    _ansi_re = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

    @classmethod
    def _sanitize_line(cls, s: str) -> str:
        s = cls._ansi_re.sub("", s)
        return s.replace("\r", "").rstrip("\n")

    @staticmethod
    def _run_with_streaming(cmd: List[str], env: Dict[str,str], on_line: Callable[[str], None]) -> int:
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
            on_line(TotalSegmentatorRunner._sanitize_line(line))
        return proc.wait()

    @staticmethod
    def _normalize_roi_subset_args(args: List[str]) -> List[str]:
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

    @staticmethod
    def _build_cmd_gpu(input_nii: Path, output_dir: Path, task: str, extra_flags: List[str], fast_gpu: bool) -> List[str]:
        cmd = ["TotalSegmentator","-i",str(input_nii),"-o",str(output_dir),"--task",task,"--device","gpu"]
        if fast_gpu: cmd.append("--fast")
        cmd += extra_flags
        return cmd

    @staticmethod
    def _build_cmd_cpu(input_nii: Path, output_dir: Path, task: str, extra_flags: List[str]) -> List[str]:
        return ["TotalSegmentator","-i",str(input_nii),"-o",str(output_dir),"--task",task,"--device","cpu", *extra_flags]

    def _run_totalseg_gpu_then_cpu(self, input_nii: Path, output_dir: Path,
                                   task: str, extra_flags: List[str], fast_gpu: bool) -> int:
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING","utf-8")
        env.setdefault("PYTHONUTF8","1")
        env.setdefault("TQDM_DISABLE","1")

        cmd_gpu = self._build_cmd_gpu(input_nii, output_dir, task, extra_flags, fast_gpu)
        with self._step("TS|GPU", f"Ejecución {task}"):
            self._flag(self.on_log, "TS|GPU", "cmd", " ".join(shlex.quote(x) for x in cmd_gpu))
            rc = self._run_with_streaming(cmd_gpu, env, on_line=self.on_log)
            self._flag(self.on_log, "TS|GPU", "rc", str(rc))
        if rc == 0:
            return rc

        env["CUDA_VISIBLE_DEVICES"] = "-1"
        self._flag(self.on_log, "TS|GPU", f"rc={rc}", f"Fallback a CPU sin --fast ({task})")
        cmd_cpu = self._build_cmd_cpu(input_nii, output_dir, task, extra_flags)
        with self._step("TS|CPU", f"Ejecución {task}"):
            self._flag(self.on_log, "TS|CPU", "cmd", " ".join(shlex.quote(x) for x in cmd_cpu))
            return self._run_with_streaming(cmd_cpu, env, on_line=self.on_log)

    # ---------- Stats ----------
    def _merge_and_cleanup_stats(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        stats = out_dir / "statistics.json"
        stats_all = out_dir / "statistics_all.json"
        if not stats.exists():
            return
        try:
            new = json.loads(stats.read_text(encoding="utf-8"))
        except Exception as e:
            self._flag(self.on_log, "STATS", "ERROR leyendo statistics.json", str(e))
            return
        base = {}
        if stats_all.exists():
            try:
                base = json.loads(stats_all.read_text(encoding="utf-8"))
            except Exception:
                base = {}
        base.update(new)
        stats_all.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            stats.unlink()
            self._flag(self.on_log, "STATS", "Consolidado", "statistics_all.json (borrado statistics.json)")
        except Exception:
            pass

    # ---------- Preflight & tmp ----------
    def _preflight_or_fail(self) -> None:
        with self._step("START", "Preflight"):
            if not shutil.which("TotalSegmentator"):
                raise RuntimeError("No se encontró 'TotalSegmentator' en PATH.")
            self._ensure_license()

    @staticmethod
    def _best_tmp_for(output_path: Path) -> Path:
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

    # ---------- API principal ----------
    def run(self, input_path: Path, output_path: Path) -> float:
        """
        Ejecuta el pipeline:
          1) ORTOPEDIA (task=total + --roi_subset ORTHO_ROI) [--fast si hay GPU]
          2) HIP_IMPLANT (si está habilitado), sin --fast
          3) Tasks extra (si se definieron), sin --fast
        """
        t0_total = time.perf_counter()
        start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ensure_dir(output_path)
        log_file = output_path / "run.log"

        # logger que duplica a archivo
        base_logger = self.on_log
        def tee_log(msg: str):
            try:
                with log_file.open("a", encoding="utf-8") as f:
                    f.write(msg + "\n")
            except Exception:
                pass
            try:
                base_logger(msg)
            except Exception:
                pass

        # sustituir temporalmente el logger
        original_logger = self.on_log
        self.on_log = tee_log

        temp_mount = None
        try:
            self._preflight_or_fail()
            self._flag(self.on_log, "START", "TotalSegmentator (GPU→CPU, Ortopedia obligatoria)")
            self._flag(self.on_log, "START", "Inicio", start_ts)
            self._flag(self.on_log, "START", "Dispositivo", self._device_summary_text(self.dev_info))

            with self._step("TMP", "Preparando área temporal"):
                tmp_root = self._best_tmp_for(output_path)
                self._flag(self.on_log, "TMP", "Base", str(tmp_root))

            true_input = Path(input_path)
            with self._step("IMPORT", "Resolviendo entrada"):
                if true_input.suffix.lower() == ".zip":
                    self._flag(self.on_log, "IMPORT", "ZIP", "Descomprimiendo…")
                    tmpdir = tmp_root / f"ts_zip_{int(time.time()*1000)}"
                    tmpdir.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(true_input, 'r') as zf:
                        _safe_extractall(zf, tmpdir)
                    subs = [p for p in tmpdir.iterdir() if p.is_dir()]
                    temp_mount = subs[0] if len(subs) == 1 else tmpdir
                    self._flag(self.on_log, "IMPORT", "ZIP extraído", str(temp_mount))
                    true_input = temp_mount
                if not true_input.exists():
                    raise FileNotFoundError(f"Entrada no encontrada: {true_input}")

            if is_nii(true_input):
                self._flag(self.on_log, "NIFTI", "Entrada NIfTI", Path(true_input).name)
                nii_input = true_input
            else:
                if not self.robust_import:
                    raise RuntimeError("Se requiere NIfTI o robust_import=True para DICOM.")
                nii_input = self._robust_dicom_to_nifti(true_input, tmp_root)
                self._flag(self.on_log, "NIFTI", "Intermedio", str(nii_input))

            # Hilos y flags comunes
            with self._step("THREADS", "Calculando hilos"):
                n_resamp, n_saving = self._suggest_threads(self.dev_info)
                flags_common = ["--statistics","--nr_thr_resamp",str(n_resamp),"--nr_thr_saving",str(n_saving)]
                self._flag(self.on_log, "THREADS", "resample/saving", f"{n_resamp}/{n_saving}")

            # 1) ORTOPEDIA (siempre)  --fast solo aquí
            orto_flags = self._normalize_roi_subset_args(["--roi_subset", *ORTHO_ROI] + flags_common)
            self._flag(self.on_log, "TS", "Resumen ORTO", f"task=total | fast_gpu={self.fast_gpu} | stats=ON")
            t0_orto = time.perf_counter()
            rc1 = self._run_totalseg_gpu_then_cpu(Path(nii_input), output_path, "total", orto_flags, self.fast_gpu)
            if rc1 != 0:
                raise RuntimeError("TotalSegmentator (ORTOPEDIA) falló incluso tras fallback CPU.")
            t_orto = time.perf_counter() - t0_orto
            self._flag(self.on_log, "END|ORTO", "Duración", format_duration(t_orto))
            self._merge_and_cleanup_stats(output_path)

            # 2) HIP_IMPLANT (opcional), sin --fast
            if self.enable_hip_implant:
                hip_flags = list(flags_common)
                self._flag(self.on_log, "TS", "Resumen HIP_IMPLANT",
                           f"task=hip_implant | out={output_path} | fast_gpu=False | stats=ON")
                t0_hip = time.perf_counter()
                rc_hip = self._run_totalseg_gpu_then_cpu(Path(nii_input), output_path, "hip_implant", hip_flags, fast_gpu=False)
                if rc_hip != 0:
                    raise RuntimeError("TotalSegmentator (HIP_IMPLANT) falló incluso tras fallback CPU.")
                t_hip = time.perf_counter() - t0_hip
                self._flag(self.on_log, "END|HIP_IMPLANT", "Duración", format_duration(t_hip))
                self._merge_and_cleanup_stats(output_path)
            else:
                self._flag(self.on_log, "TS", "HIP_IMPLANT", "Deshabilitado por configuración")

            # 3) Tasks extra (si los hay), sin --fast
            for tname in self.extra_tasks:
                tname = (tname or "").strip()
                if not tname:
                    continue
                extra_flags = list(flags_common)
                self._flag(self.on_log, "TS", "Resumen EXTRA",
                           f"task={tname} | out={output_path} | fast_gpu=False | stats=ON")
                t0_ex = time.perf_counter()
                rc_ex = self._run_totalseg_gpu_then_cpu(Path(nii_input), output_path, tname, extra_flags, fast_gpu=False)
                if rc_ex != 0:
                    raise RuntimeError(f"TotalSegmentator (task='{tname}') falló incluso tras fallback CPU.")
                t_ex = time.perf_counter() - t0_ex
                self._flag(self.on_log, f"END|{tname}", "Duración", format_duration(t_ex))
                self._merge_and_cleanup_stats(output_path)

        finally:
            # Limpieza de temporales
            with self._step("CLEANUP", "Eliminando temporales"):
                if temp_mount:
                    try:
                        shutil.rmtree(temp_mount, ignore_errors=True)
                        self._flag(self.on_log, "CLEANUP", "Temp eliminado", str(temp_mount))
                    except Exception as e:
                        self._flag(self.on_log, "CLEANUP", "WARN", f"No se pudo borrar temp: {e}")

            elapsed_total = time.perf_counter() - t0_total
            end_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._flag(self.on_log, "END", "Fin", end_ts)
            self._flag(self.on_log, "END", "Duración total", format_duration(elapsed_total))
            self._flag(self.on_log, "END", "Listo", "Ver también run.log")

            # restaurar logger original
            self.on_log = original_logger

        return elapsed_total


# ---------- Main de ejemplo ----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SOM3D TotalSegmentator Runner (sin GUI)")
    parser.add_argument("input", type=str, help="Ruta de entrada: .zip (DICOM), carpeta DICOM o .nii/.nii.gz")
    parser.add_argument("output", type=str, help="Carpeta de salida")
    # fast SOLO para ORTOPEDIA:
    parser.add_argument("--fast-gpu", action="store_true", help="Usar --fast SOLO en ORTOPEDIA cuando haya GPU")
    parser.add_argument("--no-robust", action="store_true", help="Desactivar importación robusta (requiere NIfTI)")
    # HIP_IMPLANT toggle:
    hip_group = parser.add_mutually_exclusive_group()
    hip_group.add_argument("--hip-implant", dest="hip_implant", action="store_true", help="Habilitar hip_implant (por defecto ON)")
    hip_group.add_argument("--no-hip-implant", dest="hip_implant", action="store_false", help="Deshabilitar hip_implant")
    parser.set_defaults(hip_implant=True)
    # Tasks extra (puede repetirse):
    parser.add_argument("--task", dest="tasks", action="append", default=[], help="Task extra (repetible), sin --fast")

    args = parser.parse_args()

    def server_log(line: str):
        print(line, flush=True)

    runner = TotalSegmentatorRunner(
        fast_gpu=args.fast_gpu,
        robust_import=not args.no_robust,
        enable_hip_implant=args.hip_implant,
        extra_tasks=args.tasks,
        on_log=server_log
    )

    in_path = Path(args.input)
    out_dir = Path(args.output)
    elapsed = runner.run(in_path, out_dir)
    print(f"[END] ⏱ Duración total: {format_duration(elapsed)}", flush=True)
