#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NIfTI ➜ STL (Otsu) — versión optimizada

Modos simples para el usuario final:
- **Normal**: máxima calidad (MC step_size=1, downsample=1).
- **Equilibrado**: buen rendimiento con alta fidelidad (MC step_size=2, downsample=1).

El usuario solo elige el **Modo** y presiona **Ejecutar**.

Autor: José + ChatGPT
"""

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

# ---- STL ----
from stl import mesh as stlmesh

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
    # mmap para ahorrar RAM; caching='unchanged' evita copias extra
    img = nib.load(str(path), mmap=True)
    data = img.get_fdata(dtype=np.float32, caching='unchanged')
    affine = img.affine
    return data, affine


def otsu_fast_sample(vec: np.ndarray, max_samples: int = 2_000_000) -> float:
    """Calcula umbral de Otsu de forma más rápida: si hay demasiados voxeles,
    usa un muestreo estratificado (stride o aleatorio) para acelerar.
    """
    n = vec.size
    if n == 0:
        raise RuntimeError("Sin vóxeles válidos para Otsu.")
    if n <= max_samples:
        return float(threshold_otsu(vec))
    # Estrategia: stride para evitar sobrecoste de RNG en arrays gigantes
    stride = max(1, n // max_samples)
    sample = vec[::stride]
    return float(threshold_otsu(sample))


def mask_from_otsu(
    vol: np.ndarray,
    clip_min: float | None,
    clip_max: float | None,
    exclude_zeros: bool,
    max_samples: int,
) -> tuple[np.ndarray, float]:
    # clipping in-place compatible (sin crear copia si es posible)
    data = vol
    if clip_min is not None or clip_max is not None:
        lo = -np.inf if clip_min is None else clip_min
        hi = np.inf if clip_max is None else clip_max
        # np.clip con copy=False puede no garantizar totalmente in-place según alineación, pero reduce copias
        data = np.clip(data, lo, hi, out=np.empty_like(data))

    vec = data.reshape(-1)
    # Filtro de finitos
    # (más rápido que np.isfinite dos veces cuando también excluimos ceros)
    finite_mask = np.isfinite(vec)
    vec = vec[finite_mask]
    if exclude_zeros:
        vec = vec[vec != 0]
    if vec.size == 0:
        raise RuntimeError("Sin vóxeles válidos para Otsu (revisa clip/exclude_zeros).")

    t = otsu_fast_sample(vec, max_samples=max_samples)
    mask = (data >= t)
    return mask, t


def mesh_to_stl(vertices: np.ndarray, faces: np.ndarray, out_path: Path):
    # vertices: (N, 3) en mm. faces: (M, 3) índices
    tris = np.zeros((faces.shape[0], 3, 3), dtype=np.float32)
    tris[:, 0, :] = vertices[faces[:, 0], :]
    tris[:, 1, :] = vertices[faces[:, 1], :]
    tris[:, 2, :] = vertices[faces[:, 2], :]
    m = stlmesh.Mesh(np.zeros(tris.shape[0], dtype=stlmesh.Mesh.dtype))
    m.vectors[:] = tris
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))


# ------------- Worker de conversión ------------- #

def _convert_one(
    f: Path,
    output_dir: Path,
    clip_min: float | None,
    clip_max: float | None,
    exclude_zeros: bool,
    min_voxels: int,
    step_size: int,
    max_samples: int,
    downsample: int,
):
    per_start = time.perf_counter()

    vol, affine = load_nifti(f)

    # Downsample entero opcional para acelerar (recomendado 1 o 2)
    if downsample > 1:
        vol = vol[::downsample, ::downsample, ::downsample]
        # El espaciado efectivo se multiplica por el factor
        spacing = tuple(s * downsample for s in voxel_spacing_from_affine(affine))
    else:
        spacing = voxel_spacing_from_affine(affine)

    mask, t = mask_from_otsu(vol, clip_min, clip_max, exclude_zeros, max_samples)
    voxels = int(np.count_nonzero(mask))
    if voxels < int(min_voxels):
        raise RuntimeError(f"Máscara muy pequeña: {voxels} < {min_voxels}")

    # marching cubes sobre máscara binaria (uint8) con step_size>1 para velocidad
    m_uint8 = mask.astype(np.uint8, copy=False)
    verts, faces, _, _ = marching_cubes(
        m_uint8,
        level=0.5,
        spacing=spacing,
        step_size=max(1, int(step_size)),
        allow_degenerate=False,
    )

    # guardar STL (solo “original”)
    originals_dir = output_dir / "originales"
    out_stl = originals_dir / (f.stem.replace(".nii", "") + ".stl")
    mesh_to_stl(verts.astype(np.float32, copy=False), faces.astype(np.int32, copy=False), out_stl)

    elapsed = time.perf_counter() - per_start
    return {
        "file": str(f),
        "name": f.name,
        "success": True,
        "method": "auto-otsu",
        "auto_t": f"{t:.3f}",
        "faces_before": faces.shape[0],
        "faces_after": faces.shape[0],
        "voxels_mask": voxels,
        "stl": str(out_stl),
        "stl_reduced": "",
        "message": "",
        "seconds": f"{elapsed:.3f}",
    }, elapsed, t, voxels, faces.shape[0]


def convert_folder(
    input_dir: Path,
    output_dir: Path,
    recursive: bool,
    clip_min: float | None,
    clip_max: float | None,
    exclude_zeros: bool,
    min_voxels: int,
    log_name: str,
    step_size: int,
    max_samples: int,
    downsample: int,
    progress_cb,
    done_cb,
):
    start_total = time.perf_counter()
    files = find_nii_files(input_dir, recursive)
    total = len(files)
    successes = 0
    results = []
    log_path = output_dir / (log_name or "log.csv")

    progress_cb(f"Encontrados {total} archivos NIfTI.\n")
    if total == 0:
        done_cb(False, "No se encontraron archivos.")
        return

    for idx, f in enumerate(files, start=1):
        progress_cb(f"\n[{idx}/{total}] {f.name}\n")
        try:
            record, elapsed, t, voxels, nfaces = _convert_one(
                f,
                output_dir,
                clip_min,
                clip_max,
                exclude_zeros,
                min_voxels,
                step_size,
                max_samples,
                downsample,
            )
            progress_cb(
                f"  OK | Otsu t={t:.3f} | voxels={voxels} | caras={nfaces} | {elapsed:.2f}s\n"
            )
            results.append(record)
            successes += 1
        except Exception as exc:
            elapsed = time.perf_counter() - start_total
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
                "file","name","success","method","auto_t","faces_before","faces_after",
                "voxels_mask","stl","stl_reduced","message","seconds"
            ])
            for r in results:
                writer.writerow([
                    r[k] for k in [
                        "file","name","success","method","auto_t","faces_before",
                        "faces_after","voxels_mask","stl","stl_reduced","message","seconds"
                    ]
                ])
        progress_cb(f"\nLog guardado en: {log_path}\n")
    except Exception as exc:
        progress_cb(f"\n[WARN] No se pudo escribir log: {exc}\n")

    total_elapsed = time.perf_counter() - start_total
    progress_cb(
        f"\nResumen: {successes}/{total} OK | Tiempo total: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)\n"
    )
    done_cb(True, f"Completado: {successes}/{total} OK")


# ------------- GUI (Tkinter) ------------- #

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NIfTI ➜ STL (Otsu) — by José 💪 (simple)")
        self.geometry("820x560")
        self.resizable(True, True)

        # Variables mínimas para usuario final
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="Equilibrado")  # Normal | Equilibrado

        # Parámetros fijos (ocultos al usuario)
        self._exclude_zeros = True
        self._min_voxels = 50
        self._log_name = "log.csv"
        self._clip_min = None
        self._clip_max = None
        self._recursive = True

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

        # Selector de Modo
        opt = ttk.LabelFrame(self, text="Modo")
        opt.pack(fill="x", **pad)
        ttk.Label(opt, text="Selecciona velocidad/calidad:").grid(row=0, column=0, sticky="e")
        cb = ttk.Combobox(opt, textvariable=self.mode_var, width=12, state="readonly",
                          values=("Normal", "Equilibrado"))
        cb.grid(row=0, column=1, sticky="w")

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

    @staticmethod
    def _preset_for(mode: str) -> tuple[int, int, int]:
        """Devuelve (step_size, max_samples, downsample) para el modo."""
        m = mode.lower()
        if m.startswith("equilibr"):
            # Equilibrado: compromiso calidad/tiempo
            return (2, 1_500_000, 1)
        # Normal: máxima calidad
        return (1, 2_000_000, 1)

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
                messagebox.showerror("Error", f"No se pudo crear la carpeta de salida:{exc}")
                return

        step_size, max_samples, downsample = self._preset_for(self.mode_var.get())

        # lanzar hilo
        self.run_btn.config(state="disabled")
        self.pbar.start(10)
        self.log("=== Inicio de conversión (Modo: %s) ===" % self.mode_var.get())

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
            args=(
                inp, out, self._recursive, self._clip_min, self._clip_max,
                self._exclude_zeros, self._min_voxels, self._log_name,
                step_size, max_samples, downsample,
                progress_cb, done_cb,
            ),
            daemon=True
        )
        self.worker.start()


if __name__ == "__main__":
    try:
        App().mainloop()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)

    try:
        App().mainloop()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)
