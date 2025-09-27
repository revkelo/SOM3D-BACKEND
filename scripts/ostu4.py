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
from scipy import ndimage  # NEW

# ---- Malla / STL (trimesh) ----
import trimesh  # NEW
from trimesh.smoothing import filter_laplacian  # NEW

# ---- GUI ----
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ------------- Utilidades core ------------- #

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
    # tamaño de voxel aproximado en mm a partir de la matriz affine
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

def mask_from_otsu(vol: np.ndarray, clip_min: float|None, clip_max: float|None, exclude_zeros: bool) -> tuple[np.ndarray, float]:
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

def mesh_to_stl(vertices: np.ndarray, faces: np.ndarray, out_path: Path):
    """
    Exporta STL 'sólido' con limpieza, normales y suavizado (trimesh).
    """
    tm = trimesh.Trimesh(vertices=vertices.astype(np.float32),
                         faces=faces.astype(np.int32),
                         process=True)
    tm.remove_degenerate_faces()
    tm.remove_unreferenced_vertices()
    tm.fix_normals()
    if not tm.is_watertight:
        tm.fill_holes()

    # Suavizado tipo código 2
    filter_laplacian(tm, lamb=0.5, iterations=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tm.export(str(out_path))

# ------------- Worker de conversión ------------- #

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
    log_path = output_dir / (log_name or "log.csv")

    progress_cb(f"Encontrados {total} archivos NIfTI.\n")
    if total == 0:
        done_cb(False, "No se encontraron archivos.")
        return

    for idx, f in enumerate(files, start=1):
        per_start = time.perf_counter()
        progress_cb(f"\n[{idx}/{total}] {f.name}\n")
        try:
            vol, affine = load_nifti(f)
            mask, t = mask_from_otsu(vol, clip_min, clip_max, exclude_zeros)
            voxels = int(mask.sum())
            if voxels < int(min_voxels):
                raise RuntimeError(f"Máscara muy pequeña: {voxels} < {min_voxels}")

            # --- LIMPIEZA Y SUAVIZADO (para evitar 'voxelado' y apariencia traslúcida) ---
            mask_bool = mask.astype(bool)
            # Morfología ligera + rellenado de huecos
            mask_bool = ndimage.binary_closing(
                mask_bool,
                structure=ndimage.generate_binary_structure(3, 1),
                iterations=1
            )
            mask_bool = ndimage.binary_fill_holes(mask_bool)

            # Suavizado gaussiano en float
            vol_smooth = ndimage.gaussian_filter(mask_bool.astype(np.float32), sigma=1.0)

            # Nivel de isosuperficie robusto
            vmin, vmax = float(vol_smooth.min()), float(vol_smooth.max())
            level = 0.5 if (vmin <= 0.5 <= vmax) else (vmin + 0.5 * (vmax - vmin))

            # Marching Cubes con spacing real
            spacing = voxel_spacing_from_affine(affine)
            verts, faces, _, _ = marching_cubes(vol_smooth, level=level, spacing=spacing)

            # guardar STL (solo “original”), con limpieza/solidificación/auto-smooth
            out_stl = originals_dir / (f.stem.replace(".nii", "") + ".stl")
            mesh_to_stl(verts, faces, out_stl)

            elapsed = time.perf_counter() - per_start
            progress_cb(f"  OK | Otsu t={t:.3f} | voxels={voxels} | caras={faces.shape[0]} | {elapsed:.2f}s\n")
            successes += 1
            results.append({
                "file": str(f),
                "name": f.name,
                "success": True,
                "method": "auto-otsu+smooth",
                "auto_t": f"{t:.3f}",
                "faces_before": faces.shape[0],
                "faces_after": faces.shape[0],
                "voxels_mask": voxels,
                "stl": str(out_stl),
                "stl_reduced": "",
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
                "method": "auto-otsu+smooth",
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
            writer.writerow(["file","name","success","method","auto_t","faces_before","faces_after",
                             "voxels_mask","stl","stl_reduced","message","seconds"])
            for r in results:
                writer.writerow([r[k] for k in ["file","name","success","method","auto_t","faces_before",
                                                "faces_after","voxels_mask","stl","stl_reduced","message","seconds"]])
        progress_cb(f"\nLog guardado en: {log_path}\n")
    except Exception as exc:
        progress_cb(f"\n[WARN] No se pudo escribir log: {exc}\n")

    total_elapsed = time.perf_counter() - start_total
    progress_cb(f"\nResumen: {successes}/{total} OK | Tiempo total: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)\n")
    done_cb(True, f"Completado: {successes}/{total} OK")

# ------------- GUI (Tkinter) ------------- #

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NIfTI ➜ STL (Otsu) — by José 💪")
        self.geometry("820x620")
        self.resizable(True, True)

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.recursive_var = tk.BooleanVar(value=True)
        self.clip_min_var = tk.StringVar()  # vacío = None
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

        # Input
        ttk.Label(frm, text="Carpeta de entrada (NIfTI):").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.input_var, width=70).grid(row=0, column=1, sticky="we")
        ttk.Button(frm, text="Seleccionar…", command=self.browse_input).grid(row=0, column=2)

        # Output
        ttk.Label(frm, text="Carpeta de salida (STL):").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.output_var, width=70).grid(row=1, column=1, sticky="we")
        ttk.Button(frm, text="Seleccionar…", command=self.browse_output).grid(row=1, column=2)

        # Opciones
        opt = ttk.LabelFrame(self, text="Opciones de umbral (Otsu) y filtrado")
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

        # Botones
        btns = ttk.Frame(self)
        btns.pack(fill="x", **pad)
        self.run_btn = ttk.Button(btns, text="▶ Ejecutar", command=self.run)
        self.run_btn.pack(side="left")
        ttk.Button(btns, text="Limpiar consola", command=self.clear_console).pack(side="left", padx=6)

        # Progreso
        self.pbar = ttk.Progressbar(self, mode="indeterminate")
        self.pbar.pack(fill="x", **pad)

        # Consola
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

        # parseo de numéricos
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

        # lanzar hilo
        self.run_btn.config(state="disabled")
        self.pbar.start(10)
        self.log("\n=== Inicio de conversión (Otsu) ===\n")

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
