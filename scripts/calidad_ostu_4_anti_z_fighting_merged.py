#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import threading
from pathlib import Path
import csv
import numpy as np

# ---- Librerías científicas ----
import nibabel as nib
from skimage.filters import threshold_otsu
from skimage.measure import marching_cubes
from scipy import ndimage  # ← NEW: suavizado volumétrico

# ---- Decimación / Suavizado de malla (MeshLab) ----
try:
    import pymeshlab as ml
    _HAS_PYMESHLAB = True
except Exception:
    _HAS_PYMESHLAB = False

# ---- STL fallback (si no hay MeshLab) ----
from stl import mesh as stlmesh

# ---- GUI ----
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ================= Utilidades core ================= #

def find_nii_files(root: Path, recursive: bool) -> list[Path]:
    patterns = ["*.nii", "*.nii.gz"]
    files = []
    if recursive:
        for pat in patterns:
            files.extend(root.rglob(pat))
    else:
        for pat in patterns:
            files.extend(root.glob(pat))
    return sorted(files)

def voxel_spacing_from_affine(affine: np.ndarray) -> tuple[float, float, float]:
    A = affine[:3, :3]
    sx = np.linalg.norm(A[:, 0])
    sy = np.linalg.norm(A[:, 1])
    sz = np.linalg.norm(A[:, 2])
    return float(sx), float(sy), float(sz)

def load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    return data, affine

def compute_otsu_threshold(vals: np.ndarray) -> float:
    return float(threshold_otsu(vals))

def mask_from_otsu(vol: np.ndarray,
                   clip_min: float|None,
                   clip_max: float|None,
                   exclude_zeros: bool) -> tuple[np.ndarray, float]:
    data = vol
    if clip_min is not None or clip_max is not None:
        lo = -np.inf if clip_min is None else clip_min
        hi =  np.inf if clip_max is None else clip_max
        data = np.clip(data, lo, hi).astype(np.float32, copy=False)
    vec = data.reshape(-1)
    vec = vec[np.isfinite(vec)]
    if exclude_zeros:
        vec = vec[vec != 0]
    if vec.size == 0:
        raise RuntimeError("Sin vóxeles válidos para Otsu (revisa clip/exclude_zeros).")
    t = compute_otsu_threshold(vec)
    mask = (data >= t)
    return mask, t

def _save_stl_numpy(vertices: np.ndarray, faces: np.ndarray, out_path: Path):
    tris = np.zeros((faces.shape[0], 3, 3), dtype=np.float32)
    tris[:, 0, :] = vertices[faces[:, 0], :]
    tris[:, 1, :] = vertices[faces[:, 1], :]
    tris[:, 2, :] = vertices[faces[:, 2], :]
    m = stlmesh.Mesh(np.zeros(tris.shape[0], dtype=stlmesh.Mesh.dtype))
    m.vectors[:] = tris
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))

def postprocess_with_meshlab(vertices: np.ndarray,
                             faces: np.ndarray,
                             out_path: Path,
                             *,
                             targetperc: float = 0.65,    # ↑ más caras = apariencia más lisa
                             taubin_iters: int = 60,      # ↑ más iteraciones
                             taubin_lambda: float = 0.6,  # ↑ un poco más fuerte
                             taubin_mu: float = -0.63,    # emparejado con lambda
                             close_holes_maxsize: int = 200) -> dict:
    """
    Pipeline HQ: limpieza -> suavizado Taubin -> decimación cuadrática -> pulido -> normales -> guardar STL.
    """
    ms = ml.MeshSet()
    mesh = ml.Mesh(vertices.astype(np.float32), faces.astype(np.int32))
    ms.add_mesh(mesh, "raw")

    # Limpiezas y preparación
    for f, kw in [
        ('meshing_remove_duplicate_vertices', {}),
        ('meshing_remove_duplicate_faces', {}),
        ('meshing_remove_null_faces', {}),
        ('meshing_remove_unreferenced_vertices', {}),
        ('meshing_close_holes', {'maxholesize': close_holes_maxsize}),
    ]:
        try:
            ms.apply_filter(f, **kw)
        except Exception:
            pass

    # ---- Suavizado Taubin (fuerte) ----
    def _try_taubin(steps, lam, mu):
        try:
            ms.apply_filter('apply_coord_taubin_smoothing',
                            stepsmoothnum=steps,
                            lambda_=lam,
                            mu=mu)
            return True
        except Exception:
            try:
                ms.apply_filter('apply_coord_laplacian_smoothing_surface_preserving',
                                stepsmoothnum=max(10, steps),
                                cotangentweight=True)
                return True
            except Exception:
                return False

    if not _try_taubin(taubin_iters, taubin_lambda, taubin_mu):
        raise RuntimeError("No encontré ni Taubin ni el suavizado alterno en tu PyMeshLab.")

    # ---- Decimación cuadrática (alta calidad) ----
    ms.apply_filter('meshing_decimation_quadric_edge_collapse',
                    targetperc=targetperc,
                    preservenormal=True,
                    planarquadric=True,
                    optimalplacement=True,
                    qualitythr=0.3,
                    boundaryweight=1.0,
                    preservetopology=True)

    # Pulido ligero final (un poco más fuerte)
    try:
        _try_taubin(15, 0.58, -0.62)
    except Exception:
        pass

    # Recalcular normales
    for f in ('compute_normal_per_vertex', 'compute_normal_per_face'):
        try:
            ms.apply_filter(f)
        except Exception:
            pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ms.save_current_mesh(str(out_path), binary=True)
    curr = ms.current_mesh()
    return {"faces_after": int(curr.face_number()), "verts_after": int(curr.vertex_number())}

# ================= Worker de conversión ================= #

def convert_folder(
    input_dir: Path,
    output_dir: Path,
    recursive: bool,
    clip_min: float|None,
    clip_max: float|None,
    exclude_zeros: bool,
    min_voxels: int,
    log_name: str,
    progress_cb,
    done_cb,
):
    start_total = time.perf_counter()
    files = find_nii_files(input_dir, recursive)
    total = len(files)
    successes = 0
    results = []
    originals_dir = output_dir / "originales"
    hq_dir = output_dir / "hq_suavizado_decimado"
    log_path = output_dir / (log_name or "log.csv")

    progress_cb(f"Encontrados {total} archivos NIfTI.\n")
    if total == 0:
        done_cb(False, "No se encontraron archivos.")
        return

    if not _HAS_PYMESHLAB:
        progress_cb("[WARN] PyMeshLab no está disponible. Se guardará solo el STL original sin HQ.\n")

    for idx, f in enumerate(files, start=1):
        per_start = time.perf_counter()
        progress_cb(f"\n[{idx}/{total}] {f.name}\n")
        try:
            vol, affine = load_nifti(f)
            mask, t = mask_from_otsu(vol, clip_min, clip_max, exclude_zeros)
            voxels = int(mask.sum())
            if voxels < int(min_voxels):
                raise RuntimeError(f"Máscara muy pequeña: {voxels} < {min_voxels}")

            # ---- Marching cubes con suavizado volumétrico previo ----
            spacing = voxel_spacing_from_affine(affine)

            # Anti-escalonado de bordes (ajustable 1.0–1.6)
            smoothed = ndimage.gaussian_filter(mask.astype(np.float32), sigma=1.3)
            level = 0.5

            verts, faces, _, _ = marching_cubes(
                smoothed, level=level, spacing=spacing
            )
            verts = verts.astype(np.float32)
            faces = faces.astype(np.int32)

            # Guardar STL original (trazabilidad)
            out_stl_orig = originals_dir / (f.stem.replace(".nii", "") + ".stl")
            _save_stl_numpy(verts, faces, out_stl_orig)

            faces_after = faces.shape[0]
            out_hq = ""
            method = "auto-otsu + VolSmooth + HQ(Taubin+Quadric)"
            if _HAS_PYMESHLAB:
                out_stl_hq = hq_dir / (f.stem.replace(".nii", "") + "_HQ.stl")
                metrics = postprocess_with_meshlab(
                    verts, faces, out_stl_hq,
                    targetperc=0.65,       # más caras = apariencia más lisa
                    taubin_iters=60,
                    taubin_lambda=0.6,
                    taubin_mu=-0.63,
                    close_holes_maxsize=200
                )
                faces_after = metrics["faces_after"]
                out_hq = str(out_stl_hq)
            else:
                method = "auto-otsu + VolSmooth (sin HQ: falta PyMeshLab)"

            elapsed = time.perf_counter() - per_start
            progress_cb(f"  OK | Otsu t={t:.3f} | voxels={voxels} | caras={faces_after} | {elapsed:.2f}s\n")
            successes += 1
            results.append({
                "file": str(f),
                "name": f.name,
                "success": True,
                "method": method,
                "auto_t": f"{t:.3f}",
                "faces_before": faces.shape[0],
                "faces_after": faces_after,
                "voxels_mask": voxels,
                "stl": str(out_stl_orig),
                "stl_reduced": out_hq,
                "message": "",
                "seconds": f"{elapsed:.3f}",
            })
        except Exception as exc:
            elapsed = time.perf_counter() - per_start
            progress_cb(f"  FAIL | {elapsed:.2f}s | reason={exc}\n")
            results.append({
                "file": str(f),
                "name": f.name,
                "success": False,
                "method": "auto-otsu",
                "auto_t": "",
                "faces_before": "",
                "faces_after": "",
                "voxels_mask": "",
                "stl": "",
                "stl_reduced": "",
                "message": str(exc),
                "seconds": f"{elapsed:.3f}",
            })

    # CSV
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "file","name","success","method","auto_t",
                "faces_before","faces_after","voxels_mask",
                "stl","stl_reduced","message","seconds"
            ])
            for r in results:
                writer.writerow([r[k] for k in [
                    "file","name","success","method","auto_t",
                    "faces_before","faces_after","voxels_mask",
                    "stl","stl_reduced","message","seconds"
                ]])
        progress_cb(f"\nLog guardado en: {log_path}\n")
    except Exception as exc:
        progress_cb(f"\n[WARN] No se pudo escribir log: {exc}\n")

    total_elapsed = time.perf_counter() - start_total
    progress_cb(f"\nResumen: {successes}/{total} OK | Tiempo total: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)\n")
    done_cb(True, f"Completado: {successes}/{total} OK")

# ================= GUI (Tkinter) ================= #

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NIfTI ➜ STL (Otsu + VolSmooth + Taubin + Quadric) — by José 💪")
        self.geometry("840x640")
        self.resizable(True, True)

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.recursive_var = tk.BooleanVar(value=True)
        self.clip_min_var = tk.StringVar()
        self.clip_max_var = tk.StringVar()
        self.exclude_zeros_var = tk.BooleanVar(value=True)
        self.min_voxels_var = tk.StringVar(value="50")
        self.log_name_var = tk.StringVar(value="log.csv")

        self._build_ui()
        self.worker = None

    def _build_ui(self):
        pad = {'padx': 8, 'pady': 6}

        frm = ttk.Frame(self)
        frm.pack(fill="x", **pad)

        ttk.Label(frm, text="Carpeta de entrada (NIfTI):").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.input_var, width=70).grid(row=0, column=1, sticky="we")
        ttk.Button(frm, text="Seleccionar…", command=self.browse_input).grid(row=0, column=2)

        ttk.Label(frm, text="Carpeta de salida (STL):").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.output_var, width=70).grid(row=1, column=1, sticky="we")
        ttk.Button(frm, text="Seleccionar…", command=self.browse_output).grid(row=1, column=2)

        opt = ttk.LabelFrame(self, text="Opciones de Otsu y filtrado")
        opt.pack(fill="x", **pad)

        ttk.Checkbutton(opt, text="Buscar recursivo", variable=self.recursive_var).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(opt, text="Excluir ceros", variable=self.exclude_zeros_var).grid(row=0, column=1, sticky="w")

        ttk.Label(opt, text="Clip min (HU):").grid(row=1, column=0, sticky="e")
        ttk.Entry(opt, textvariable=self.clip_min_var, width=12).grid(row=1, column=1, sticky="w")
        ttk.Label(opt, text="Clip max (HU):").grid(row=1, column=2, sticky="e")
        ttk.Entry(opt, textvariable=self.clip_max_var, width=12).grid(row=1, column=3, sticky="w")

        ttk.Label(opt, text="Min voxels:").grid(row=2, column=0, sticky="e")
        ttk.Entry(opt, textvariable=self.min_voxels_var, width=12).grid(row=2, column=1, sticky="w")

        ttk.Label(opt, text="Log CSV:").grid(row=2, column=2, sticky="e")
        ttk.Entry(opt, textvariable=self.log_name_var, width=20).grid(row=2, column=3, sticky="w")

        btns = ttk.Frame(self)
        btns.pack(fill="x", **pad)
        self.run_btn = ttk.Button(btns, text="▶ Ejecutar", command=self.run)
        self.run_btn.pack(side="left")
        ttk.Button(btns, text="Limpiar consola", command=self.clear_console).pack(side="left", padx=6)

        self.pbar = ttk.Progressbar(self, mode="indeterminate")
        self.pbar.pack(fill="x", **pad)

        self.console = tk.Text(self, height=18, wrap="word")
        self.console.pack(fill="both", expand=True, **pad)
        self.console.configure(state="disabled")

    def browse_input(self):
        d = filedialog.askdirectory(title="Selecciona carpeta con NIfTI")
        if d:
            self.input_var.set(d)

    def browse_output(self):
        d = filedialog.askdirectory(title="Selecciona carpeta de salida")
        if d:
            self.output_var.set(d)

    def clear_console(self):
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")

    def log(self, text: str):
        self.console.configure(state="normal")
        self.console.insert("end", text)
        self.console.see("end")
        self.console.configure(state="disabled")

    def run(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("En ejecución", "Ya hay un proceso corriendo.")
            return
        inp = Path(self.input_var.get().strip())
        out = Path(self.output_var.get().strip())
        if not inp.exists():
            messagebox.showerror("Error", "La carpeta de entrada no existe.")
            return
        if not out.exists():
            try:
                out.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                messagebox.showerror("Error", f"No se pudo crear la carpeta de salida:\n{exc}")
                return

        def parse_float(s):
            s = s.strip()
            return None if s == "" else float(s)
        def parse_int(s):
            return int(s.strip())

        clip_min = parse_float(self.clip_min_var.get())
        clip_max = parse_float(self.clip_max_var.get())
        min_voxels = parse_int(self.min_voxels_var.get())
        recursive = bool(self.recursive_var.get())
        exclude_zeros = bool(self.exclude_zeros_var.get())
        log_name = self.log_name_var.get().strip() or "log.csv"

        self.run_btn.config(state="disabled")
        self.pbar.start(10)
        self.log("\n=== Inicio de conversión (Otsu + VolSmooth + HQ Taubin + Quadric) ===\n")
        if not _HAS_PYMESHLAB:
            self.log("[AVISO] Instala PyMeshLab para suavizado/decimación HQ: pip install pymeshlab\n")

        def progress_cb(msg):
            self.after(0, lambda: self.log(msg))

        def done_cb(ok, msg):
            def _done():
                self.pbar.stop()
                self.run_btn.config(state="normal")
                if ok:
                    messagebox.showinfo("Listo", msg)
                else:
                    messagebox.showwarning("Atención", msg)
            self.after(0, _done)

        self.worker = threading.Thread(
            target=convert_folder,
            args=(inp, out, recursive, clip_min, clip_max, exclude_zeros, min_voxels, log_name, progress_cb, done_cb),
            daemon=True
        )
        self.worker.start()

if __name__ == "__main__":
    try:
        App().mainloop()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)
