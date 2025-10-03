#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, time, threading, shutil
from pathlib import Path
import csv
import numpy as np

_HAS_CUPY = False
_cupy_ver = "unknown"
GPU_ONLY_OTSU = False
try:
    os.environ.setdefault("CUPY_DONT_WARN_ON_CUDA_PATH", "1")
    import cupy as cp
    _HAS_CUPY = True
    GPU_ONLY_OTSU = True
    _cupy_ver = getattr(cp, "__version__", "unknown")
except Exception:
    pass

import nibabel as nib
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import marching_cubes as skimage_marching_cubes

_HAS_VTK = False
try:
    import vtk
    from vtk.util import numpy_support as vtk_np  # type: ignore
    _HAS_VTK = True
except Exception:
    pass

_HAS_PYMESHLAB = False
try:
    import pymeshlab as ml
    _HAS_PYMESHLAB = True
except Exception:
    pass

from stl import mesh as stlmesh

import tkinter as tk
from tkinter import ttk, filedialog, messagebox


def find_nii_files(root: Path, recursive: bool) -> list[Path]:
    pats = ["*.nii", "*.nii.gz"]
    files = []
    if recursive:
        for p in pats: files.extend(root.rglob(p))
    else:
        for p in pats: files.extend(root.glob(p))
    return sorted(files)


def voxel_spacing_from_affine(affine: np.ndarray) -> tuple[float, float, float]:
    A = affine[:3, :3]
    return float(np.linalg.norm(A[:, 0])), float(np.linalg.norm(A[:, 1])), float(np.linalg.norm(A[:, 2]))


def load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32), img.affine


def _is_vertebra(name: str) -> bool:
    return "vertebrae_" in name.lower()


# --- Blur gaussiano previo ---
def _pre_smooth_volume(vol: np.ndarray, spacing: tuple[float, float, float], name: str) -> np.ndarray:
    sx, sy, sz = [float(s) for s in spacing]
    mean_mm = (sx + sy + sz) / 3.0
    sigma = 0.9 if _is_vertebra(name) else 1.3
    if mean_mm >= 1.2: sigma *= 1.15
    if mean_mm >= 1.6: sigma *= 1.25
    sigma = float(np.clip(sigma, 0.6, 2.2))
    return gaussian(vol, sigma=sigma, preserve_range=True)


def _compute_otsu_cpu(vec_np: np.ndarray) -> float:
    return float(threshold_otsu(vec_np))


def _compute_otsu_gpu_cupy(vec_cp: "cp.ndarray", bins: int = 4096) -> float:
    vec_cp = vec_cp.astype(cp.float32, copy=False)
    vmin, vmax = cp.min(vec_cp), cp.max(vec_cp)
    if not cp.isfinite(vmin) or not cp.isfinite(vmax) or vmin == vmax:
        return float(vmin)
    hist, bin_edges = cp.histogram(vec_cp, bins=bins, range=(float(vmin), float(vmax)))
    p = hist.astype(cp.float64); n = p.sum()
    if n == 0: return float(vmin)
    p /= n
    centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    omega = cp.cumsum(p)
    mu = cp.cumsum(p * centers)
    mu_t = mu[-1]
    eps = 1e-12
    sigma_b2 = (mu_t * omega - mu)**2 / (omega * (1.0 - omega) + eps)
    k_star = int(cp.argmax(sigma_b2).get())
    return float(centers[k_star])


def mask_from_otsu_cpu(vol: np.ndarray, clip_min: float|None, clip_max: float|None, exclude_zeros: bool) -> tuple[np.ndarray, float]:
    data = vol
    if clip_min is not None or clip_max is not None:
        lo = -np.inf if clip_min is None else clip_min
        hi =  np.inf if clip_max is None else clip_max
        data = np.clip(data, lo, hi).astype(np.float32, copy=False)
    vec = data.reshape(-1)
    vec = vec[np.isfinite(vec)]
    if exclude_zeros: vec = vec[vec != 0]
    if vec.size == 0: raise RuntimeError("Sin vóxeles válidos para Otsu (CPU).")
    t = _compute_otsu_cpu(vec)
    return (data >= t), float(t)


def mask_from_otsu_gpu_only(vol_np: np.ndarray, clip_min: float|None, clip_max: float|None, exclude_zeros: bool) -> tuple[np.ndarray, float]:
    if not _HAS_CUPY: raise RuntimeError("GPU Otsu requiere CuPy.")
    data = cp.asarray(vol_np, dtype=cp.float32)
    if clip_min is not None or clip_max is not None:
        lo = -cp.inf if clip_min is None else clip_min
        hi =  cp.inf if clip_max is None else clip_max
        data = cp.clip(data, lo, hi).astype(cp.float32, copy=False)
    vec = data.ravel()
    vec = vec[cp.isfinite(vec)]
    if exclude_zeros: vec = vec[vec != 0]
    if int(vec.size) == 0: raise RuntimeError("Sin vóxeles válidos para Otsu (GPU).")
    t = _compute_otsu_gpu_cupy(vec)
    mask_np = cp.asnumpy(data >= t)
    return mask_np, float(t)


def _numpy_to_vtk_image(mask: np.ndarray,
                        spacing: tuple[float, float, float],
                        origin: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> "vtk.vtkImageData":
    x, y, z = mask.shape
    img = vtk.vtkImageData()
    img.SetDimensions(int(x), int(y), int(z))
    img.SetSpacing(float(spacing[0]), float(spacing[1]), float(spacing[2]))
    img.SetOrigin(float(origin[0]), float(origin[1]), float(origin[2]))
    arr_np = mask.astype(np.uint8, copy=False).ravel(order="F")
    arr_vtk = vtk_np.numpy_to_vtk(arr_np, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    img.GetPointData().SetScalars(arr_vtk)
    return img


def marching_cubes_cpu_vtk(mask: np.ndarray, spacing: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    if not _HAS_VTK: raise RuntimeError("VTK no disponible.")
    mc = vtk.vtkFlyingEdges3D()
    mc.SetInputData(_numpy_to_vtk_image(mask, spacing))
    mc.SetValue(0, 0.5)
    mc.Update()
    poly = mc.GetOutput()
    pts = poly.GetPoints()
    verts = vtk_np.vtk_to_numpy(pts.GetData()).astype(np.float32, copy=False)
    cells = poly.GetPolys().GetData()
    faces = vtk_np.vtk_to_numpy(cells).reshape(-1, 4)[:, 1:].astype(np.int32, copy=False)
    return verts, faces


def marching_cubes_cpu_skimage(mask: np.ndarray, spacing: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    verts, faces, _, _ = skimage_marching_cubes(mask.astype(np.uint8), level=0.5, spacing=spacing)
    return verts.astype(np.float32), faces.astype(np.int32)


def _save_stl_numpy(vertices: np.ndarray, faces: np.ndarray, out_path: Path):
    tris = np.zeros((faces.shape[0], 3, 3), dtype=np.float32)
    tris[:, 0, :] = vertices[faces[:, 0], :]
    tris[:, 1, :] = vertices[faces[:, 1], :]
    tris[:, 2, :] = vertices[faces[:, 2], :]
    m = stlmesh.Mesh(np.zeros(tris.shape[0], dtype=stlmesh.Mesh.dtype))
    m.vectors[:] = tris
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))


# ------ Helper: tamaño legible ------
def _fmt_size(nbytes: int) -> str:
    try:
        u = ["B","KiB","MiB","GiB","TiB"]
        size = float(nbytes); i = 0
        while size >= 1024.0 and i < len(u)-1:
            size /= 1024.0; i += 1
        return f"{size:.2f} {u[i]}"
    except Exception:
        return f"{nbytes} B"
# ------------------------------------


# --- Postproceso con control por objetivos (ajustado) ---
def postprocess_with_meshlab(
    vertices: np.ndarray,
    faces: np.ndarray,
    out_path: Path,
    *,
    target_faces: int | None = None,
    max_faces: int | None = None,
    targetperc: float | None = None,
    taubin_iters: int = 25,
    taubin_lambda: float = 0.5,
    taubin_mu: float = -0.53,
    close_holes_maxsize: int = 200,
    tol_ratio: float = 0.05,
    max_decim_rounds: int = 4,
    force_extra_round: bool = True,
    preserve_topology: bool = True
) -> dict:
    if not _HAS_PYMESHLAB:
        raise RuntimeError("PyMeshLab no disponible.")

    ms = ml.MeshSet()
    ms.add_mesh(ml.Mesh(vertices.astype(np.float32), faces.astype(np.int32)), "raw")

    # Limpiezas iniciales (SIN close_holes aquí para no inflar antes)
    for f, kw in [
        ('meshing_remove_duplicate_vertices', {}),
        ('meshing_remove_duplicate_faces', {}),
        ('meshing_remove_null_faces', {}),
        ('meshing_remove_unreferenced_vertices', {}),
    ]:
        try: ms.apply_filter(f, **kw)
        except Exception: pass

    # Suavizado inicial
    base_iters = max(taubin_iters, 90)
    ok_smooth = False
    try:
        ms.apply_filter('apply_coord_taubin_smoothing',
                        stepsmoothnum=base_iters,
                        lambda_=max(taubin_lambda, 0.6),
                        mu=min(taubin_mu, -0.63))
        ok_smooth = True
    except Exception:
        try:
            ms.apply_filter('apply_coord_laplacian_smoothing_surface_preserving',
                            stepsmoothnum=max(20, base_iters // 2),
                            cotangentweight=True)
            ok_smooth = True
        except Exception:
            pass
    if not ok_smooth:
        raise RuntimeError("No se pudo suavizar (Taubin/alterno).")

    curr = ms.current_mesh()
    faces_before = int(curr.face_number())

    # Determinar objetivo
    desired = None
    if target_faces is not None:
        desired = max(200, int(target_faces))
    elif max_faces is not None:
        desired = max(200, min(faces_before, int(max_faces)))

    # Helpers de decimación (más agresivos)
    def _decimate_to_faces(target_faces: int, keep_topology: bool = True):
        ms.apply_filter(
            'meshing_decimation_quadric_edge_collapse',
            targetfacenum=int(max(200, target_faces)),
            preservenormal=True,
            planarquadric=True,
            optimalplacement=True,
            qualitythr=0.5,      # <-- subido
            boundaryweight=0.6,  # <-- menos rigidez de borde
            preservetopology=bool(keep_topology)
        )

    def _decimate_to_ratio(ratio: float, keep_topology: bool = True):
        ratio = float(np.clip(ratio, 0.02, 0.98))
        ms.apply_filter(
            'meshing_decimation_quadric_edge_collapse',
            targetperc=ratio,
            preservenormal=True,
            planarquadric=True,
            optimalplacement=True,
            qualitythr=0.5,      # <-- subido
            boundaryweight=0.6,  # <-- menos rigidez de borde
            preservetopology=bool(keep_topology)
        )

    # Estrategia
    if desired is not None and desired < faces_before:
        rounds = 0
        stalled = 0
        last_faces = faces_before
        while rounds < max_decim_rounds:
            curr_faces = int(ms.current_mesh().face_number())
            if curr_faces <= max(desired, int(desired * (1 + tol_ratio))):
                break

            _decimate_to_faces(desired, keep_topology=preserve_topology)

            curr_faces2 = int(ms.current_mesh().face_number())
            achieved_ratio = curr_faces2 / max(1, curr_faces)
            over = 1.0 - achieved_ratio
            inter_iters = max(16, int((base_iters // 2) * (1.0 + 1.4 * over)))
            try:
                ms.apply_filter('apply_coord_taubin_smoothing',
                                stepsmoothnum=inter_iters, lambda_=0.55, mu=-0.58)
            except Exception:
                pass

            new_faces = int(ms.current_mesh().face_number())
            if new_faces >= last_faces - max(800, int(0.05 * last_faces)):
                stalled += 1
                for hard_ratio in (0.75, 0.60, 0.45, 0.35):
                    if int(ms.current_mesh().face_number()) <= desired: break
                    rel_ratio = desired / max(1, int(ms.current_mesh().face_number()))
                    _decimate_to_ratio(rel_ratio, keep_topology=preserve_topology)
                    _decimate_to_ratio(hard_ratio, keep_topology=preserve_topology)
            else:
                stalled = 0

            last_faces = int(ms.current_mesh().face_number())
            rounds += 1

        if force_extra_round and int(ms.current_mesh().face_number()) > desired:
            _decimate_to_faces(desired, keep_topology=preserve_topology)
            if int(ms.current_mesh().face_number()) > int(desired * (1 + tol_ratio)):
                _decimate_to_ratio(0.35, keep_topology=preserve_topology)

    else:
        if targetperc is None:
            targetperc = 0.40 if faces_before >= 120_000 else 0.50
        _decimate_to_ratio(float(targetperc), keep_topology=preserve_topology)
        try:
            ms.apply_filter('apply_coord_taubin_smoothing',
                            stepsmoothnum=max(18, base_iters // 3),
                            lambda_=0.55, mu=-0.58)
        except Exception:
            pass

    # Empuje de emergencia si casi no redujo
    try:
        curr_faces = int(ms.current_mesh().face_number())
        if desired is not None and curr_faces > int(0.90 * faces_before):
            _decimate_to_ratio(0.50, keep_topology=preserve_topology)
            _decimate_to_ratio(0.35, keep_topology=preserve_topology)
    except Exception:
        pass

    # Suavizado final
    try:
        ms.apply_filter('apply_coord_taubin_smoothing',
                        stepsmoothnum=max(28, base_iters // 2),
                        lambda_=0.50, mu=-0.55)
    except Exception:
        pass
    try:
        ms.apply_filter('apply_coord_laplacian_smoothing_surface_preserving',
                        stepsmoothnum=18, cotangentweight=True)
    except Exception:
        pass

    # Cerrar huecos pequeños AL FINAL (evita inflado previo)
    try:
        ms.apply_filter('meshing_close_holes', maxholesize=min(80, close_holes_maxsize))
    except Exception:
        pass

    # Normales
    for f in ('compute_normal_per_vertex', 'compute_normal_per_face'):
        try: ms.apply_filter(f)
        except Exception: pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ms.save_current_mesh(str(out_path), binary=True)
    curr = ms.current_mesh()
    return {
        "faces_before": faces_before,
        "faces_after": int(curr.face_number()),
        "verts_after": int(curr.vertex_number()),
        "target_faces": desired
    }


# --- Objetivo de caras por nombre (más agresivo en costillas/esternón/escápula) ---
def _suggest_target_faces(name: str, faces_before: int) -> int:
    n = name.lower()
    if "vertebrae_" in n:
        if faces_before >= 200_000: return 18_000
        elif faces_before >= 150_000: return 16_000
        elif faces_before >= 100_000: return 14_000
        else: return max(10_000, faces_before // 6)
    else:
        if any(k in n for k in ["rib_", "sternum", "scapula"]):
            if faces_before >= 80_000: return 6_000
            elif faces_before >= 40_000: return 5_000
            else: return 4_000
        if faces_before >= 200_000: return 10_000
        elif faces_before >= 150_000: return 8_000
        elif faces_before >= 100_000: return 6_500
        else: return max(4_000, faces_before // 10)


def convert_folder(input_dir: Path, output_dir: Path, recursive: bool, clip_min: float | None, clip_max: float | None,
                   exclude_zeros: bool, min_voxels: int, log_name: str, progress_cb, done_cb):
    start_total = time.perf_counter()
    files = find_nii_files(input_dir, recursive)
    total, successes = len(files), 0
    results = []
    originals_dir = output_dir / "originales"
    hq_dir = output_dir / "hq_suavizado_decimado"
    log_path = output_dir / (log_name or "log.csv")
    progress_cb(f"Archivos NIfTI: {total}\n")
    if total == 0:
        done_cb(False, "No se encontraron archivos.")
        return
    if not _HAS_PYMESHLAB:
        progress_cb("Aviso: PyMeshLab no disponible. Se guardará solo STL original.\n")

    for idx, f in enumerate(files, start=1):
        t0 = time.perf_counter()
        progress_cb(f"[{idx}/{total}] {f.name}\n")
        try:
            vol, affine = load_nifti(f)
            spacing = voxel_spacing_from_affine(affine)


            vol = _pre_smooth_volume(vol, spacing, f.name)

           
            if GPU_ONLY_OTSU:
                if not _HAS_CUPY:
                    raise RuntimeError("Modo solo-GPU activo pero CuPy no está disponible.")
                try:
                    mask_np, t = mask_from_otsu_gpu_only(vol, clip_min, clip_max, exclude_zeros)
                    used_otsu = "gpu(cupy)"
                except Exception as e:

                    raise RuntimeError(f"Otsu(GPU) falló en modo solo-GPU: {e}")
            else:

                try:
                    if _HAS_CUPY:
                        mask_np, t = mask_from_otsu_gpu_only(vol, clip_min, clip_max, exclude_zeros)
                        used_otsu = "gpu(cupy)"
                    else:
                        raise RuntimeError("GPU no disponible")
                except Exception:
                    mask_np, t = mask_from_otsu_cpu(vol, clip_min, clip_max, exclude_zeros)
                    used_otsu = "cpu"
   


            voxels = int(mask_np.sum())
            if voxels < int(min_voxels):
                raise RuntimeError(f"Máscara muy pequeña: {voxels} < {min_voxels}")

            try:
                if _HAS_VTK:
                    verts, faces = marching_cubes_cpu_vtk(mask_np, spacing)
                    used_backend = "cpu(vtk)"
                else:
                    raise RuntimeError("VTK no disponible")
            except Exception:
                verts, faces = marching_cubes_cpu_skimage(mask_np, spacing)
                used_backend = "cpu(skimage)"

            out_stl_orig = originals_dir / (f.stem.replace(".nii", "") + ".stl")
            _save_stl_numpy(verts, faces, out_stl_orig)

            # Tamaño STL original
            stl_orig_bytes = out_stl_orig.stat().st_size if out_stl_orig.exists() else 0
            stl_orig_hr = _fmt_size(stl_orig_bytes)

            faces_after = faces.shape[0]
            out_hq = ""
            stl_hq_bytes = 0
            stl_hq_hr = ""
            size_saving_pct = 0.0
            method = f"otsu-{used_otsu} + MC({used_backend})"

            # Regla: si ya es pequeño -> generar SIEMPRE _HQ con SUAVIZADO (sin decimation)
            skip_hq = (faces.shape[0] < 5000) or (stl_orig_bytes < 300 * 1024)

            out_stl_hq = None
            if _HAS_PYMESHLAB:
                out_stl_hq = hq_dir / (f.stem.replace(".nii", "") + "_HQ.stl")

            if _HAS_PYMESHLAB and skip_hq:
                # === SUAVIZADO-ONLY (sin decimation) ===
                try:
                    ms = ml.MeshSet()
                    ms.add_mesh(ml.Mesh(verts.astype(np.float32), faces.astype(np.int32)), "small")

                    # Taubin: más iteraciones si la malla es un poco más grande
                    nfaces = faces.shape[0]
                    base_steps = 50 if nfaces < 2000 else 70
                    ms.apply_filter('apply_coord_taubin_smoothing',
                                    stepsmoothnum=base_steps,
                                    lambda_=0.60, mu=-0.63)

                    # Toca laplaciano suave para redondear aristas duras
                    try:
                        ms.apply_filter('apply_coord_laplacian_smoothing_surface_preserving',
                                        stepsmoothnum=15, cotangentweight=True)
                    except Exception:
                        pass

                    # Normales y guardar
                    for flt in ('compute_normal_per_vertex', 'compute_normal_per_face'):
                        try: ms.apply_filter(flt)
                        except Exception: pass

                    out_stl_hq.parent.mkdir(parents=True, exist_ok=True)
                    ms.save_current_mesh(str(out_stl_hq), binary=True)

                    # Actualizar métricas/log como HQ generado
                    out_hq = str(out_stl_hq)
                    stl_hq_bytes = out_stl_hq.stat().st_size if out_stl_hq.exists() else 0
                    stl_hq_hr = _fmt_size(stl_hq_bytes)
                    size_saving_pct = (1.0 - (stl_hq_bytes / stl_orig_bytes)) * 100.0 if stl_orig_bytes > 0 else 0.0
                    method = f"{method} | HQ=suavizado-only(taubin/laplaciano)"
                except Exception as e:
                    # Fallback: si algo falla, al menos duplica el original como HQ
                    try:
                        shutil.copy2(out_stl_orig, out_stl_hq)
                        out_hq = str(out_stl_hq)
                        stl_hq_bytes = out_stl_hq.stat().st_size
                        stl_hq_hr = _fmt_size(stl_hq_bytes)
                        size_saving_pct = (1.0 - (stl_hq_bytes / stl_orig_bytes)) * 100.0 if stl_orig_bytes > 0 else 0.0
                        method = f"{method} | HQ=copia(original) (suavizado falló: {e})"
                    except Exception:
                        method = f"{method} | HQ=fallback(failed: {e})"

            elif _HAS_PYMESHLAB and not skip_hq:
                # === Ruta normal: decimation + suavizado (como ya la tenías) ===
                try:
                    is_vert = _is_vertebra(f.name)
                    target_faces = _suggest_target_faces(f.name, faces.shape[0])

                    if is_vert:
                        tb_iters = 70
                        max_rounds = 3
                        tol = 0.08
                        preserve_topo = True
                    else:
                        big_mesh = faces.shape[0] >= 150_000
                        tb_iters = 110 if big_mesh else 90
                        max_rounds = 4
                        tol = 0.06
                        preserve_topo = False

                    metrics = postprocess_with_meshlab(
                        verts, faces, out_stl_hq,
                        target_faces=target_faces,
                        taubin_iters=tb_iters,
                        taubin_lambda=0.6, taubin_mu=-0.63,
                        close_holes_maxsize=220,
                        max_decim_rounds=max_rounds,
                        tol_ratio=tol,
                        force_extra_round=True,
                        preserve_topology=preserve_topo
                    )
                    faces_after = metrics["faces_after"]
                    out_hq = str(out_stl_hq)

                    # Tamaño STL HQ y % ahorro
                    stl_hq_bytes = out_stl_hq.stat().st_size if out_stl_hq.exists() else 0
                    stl_hq_hr = _fmt_size(stl_hq_bytes)
                    size_saving_pct = (1.0 - (stl_hq_bytes / stl_orig_bytes)) * 100.0 if stl_orig_bytes > 0 else 0.0
                    method += f" + HQ(targetfacenum={metrics.get('target_faces')})"

                    # Reintento agresivo si casi no redujo (igual que antes)
                    if stl_orig_bytes > 0 and stl_hq_bytes >= int(0.98 * stl_orig_bytes):
                        try:
                            ms2 = ml.MeshSet()
                            ms2.add_mesh(ml.Mesh(verts.astype(np.float32), faces.astype(np.int32)), "retry")
                            def _decim(ms, r):
                                ms.apply_filter('meshing_decimation_quadric_edge_collapse',
                                                targetperc=float(r), preservenormal=True,
                                                planarquadric=True, optimalplacement=True,
                                                qualitythr=0.5, boundaryweight=0.6,
                                                preservetopology=False)
                            _decim(ms2, 0.50); _decim(ms2, 0.35)
                            ms2.save_current_mesh(str(out_stl_hq), binary=True)
                            stl_hq_bytes = out_stl_hq.stat().st_size if out_stl_hq.exists() else stl_hq_bytes
                            stl_hq_hr = _fmt_size(stl_hq_bytes)
                            size_saving_pct = (1.0 - (stl_hq_bytes / stl_orig_bytes)) * 100.0 if stl_orig_bytes > 0 else 0.0
                        except Exception:
                            pass
                        if stl_hq_bytes >= int(0.98 * stl_orig_bytes):
                            try:
                                shutil.copy2(out_stl_orig, out_stl_hq)
                                stl_hq_bytes = out_stl_hq.stat().st_size
                                stl_hq_hr = _fmt_size(stl_hq_bytes)
                                size_saving_pct = (1.0 - (stl_hq_bytes / stl_orig_bytes)) * 100.0 if stl_orig_bytes > 0 else 0.0
                                method += " | HQ=fallback(original)"
                            except Exception:
                                method += " | HQ=fallback(failed)"
                except Exception as e:
                    method += f" (HQ fail: {e})"

            else:
                # Sin PyMeshLab: no se puede suavizar ni decimar -> solo original
                method += " (sin HQ)"

            elapsed = time.perf_counter() - t0

            # Mensaje con signo correcto
            if _HAS_PYMESHLAB and out_hq:
                signo = "menos" if size_saving_pct >= 0 else "más"
                progress_cb(
                    f"OK | Otsu({used_otsu}) t={t:.3f} | voxels={voxels} | caras={faces_after} | "
                    f"STL: {stl_orig_hr} → HQ: {stl_hq_hr} ({abs(size_saving_pct):.1f}% {signo}) | {elapsed:.2f}s\n"
                )
            else:
                progress_cb(
                    f"OK | Otsu({used_otsu}) t={t:.3f} | voxels={voxels} | caras={faces_after} | "
                    f"STL: {stl_orig_hr} | {elapsed:.2f}s\n"
                )

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
                "stl_size_bytes": stl_orig_bytes,
                "stl_size_hr": stl_orig_hr,
                "stl_reduced": out_hq,
                "stl_reduced_size_bytes": (stl_hq_bytes if _HAS_PYMESHLAB and out_hq else ""),
                "stl_reduced_size_hr": (stl_hq_hr if _HAS_PYMESHLAB and out_hq else ""),
                "size_saving_pct": (f"{size_saving_pct:.2f}" if _HAS_PYMESHLAB and out_hq else ""),
                "message": "",
                "seconds": f"{elapsed:.3f}",
            })
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            progress_cb(f"FAIL | {elapsed:.2f}s | {exc}\n")
            results.append({
                "file": str(f), "name": f.name, "success": False, "method": "auto",
                "auto_t": "", "faces_before": "", "faces_after": "", "voxels_mask": "",
                "stl": "", "stl_size_bytes": "", "stl_size_hr": "",
                "stl_reduced": "", "stl_reduced_size_bytes": "", "stl_reduced_size_hr": "",
                "size_saving_pct": "",
                "message": str(exc), "seconds": f"{elapsed:.3f}",
            })

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow([
                "file","name","success","method","auto_t","faces_before","faces_after","voxels_mask",
                "stl","stl_size_bytes","stl_size_hr",
                "stl_reduced","stl_reduced_size_bytes","stl_reduced_size_hr","size_saving_pct",
                "message","seconds"
            ])
            for r in results:
                w.writerow([
                    r["file"], r["name"], r["success"], r["method"], r["auto_t"],
                    r["faces_before"], r["faces_after"], r["voxels_mask"],
                    r["stl"], r.get("stl_size_bytes",""), r.get("stl_size_hr",""),
                    r["stl_reduced"], r.get("stl_reduced_size_bytes",""), r.get("stl_reduced_size_hr",""),
                    r.get("size_saving_pct",""),
                    r["message"], r["seconds"]
                ])
        progress_cb(f"\nLog: {log_path}\n")
    except Exception as exc:
        progress_cb(f"\nNo se pudo escribir log: {exc}\n")

    # Resumen de tamaño total (opcional pero útil)
    try:
        tot_orig = sum(int(r["stl_size_bytes"]) for r in results if r.get("stl_size_bytes"))
        tot_hq   = sum(int(r["stl_reduced_size_bytes"]) for r in results if r.get("stl_reduced_size_bytes"))
        if tot_orig > 0 and tot_hq > 0:
            pct = (1.0 - (tot_hq / tot_orig)) * 100.0
            progress_cb(f"\nTamaño total original: {_fmt_size(tot_orig)}\n")
            progress_cb(f"Tamaño total HQ:       {_fmt_size(tot_hq)}\n")
            progress_cb(f"Ahorro total:          {pct:.1f}%\n")
    except Exception:
        pass

    total_elapsed = time.perf_counter() - start_total
    progress_cb(f"\nResumen: {successes}/{total} OK | {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)\n")
    done_cb(True, f"Completado: {successes}/{total} OK")


def _gpu_info_text() -> str:
    info = []
    if _HAS_CUPY:
        try:
            n = cp.cuda.runtime.getDeviceCount()
            info.append(f"CuPy: {_cupy_ver} | GPUs: {n}")
            for i in range(n):
                p = cp.cuda.runtime.getDeviceProperties(i)
                name = p.get("name", b"").decode() if isinstance(p.get("name", ""), (bytes, bytearray)) else p.get("name", "unknown")
                mem = int(p.get("totalGlobalMem", 0)) // (1024*1024)
                cc = f"{p.get('major', '?')}.{p.get('minor', '?')}"
                info.append(f"GPU {i}: {name} | CC {cc} | {mem} MiB")
        except Exception as e:
            info.append(f"Error CuPy: {e}")
    else:
        info.append("CuPy: NO")
        try:
            import subprocess
            if shutil.which("nvidia-smi"):
                r = subprocess.run(
                    ["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5
                )
                out = r.stdout.strip()
                if out:
                    for idx, line in enumerate(out.splitlines()):
                        parts = [x.strip() for x in line.split(",")]
                        if len(parts) >= 3:
                            info.append(f"GPU {idx}: {parts[0]} | Driver {parts[1]} | {parts[2]}")
                        else:
                            info.append(f"GPU {idx}: {line}")
                else:
                    info.append("nvidia-smi sin salida.")
            else:
                info.append("nvidia-smi no encontrado.")
        except Exception as e:
            info.append(f"Error nvidia-smi: {e}")
    return "\n".join(info)


def _libs_info_text() -> str:
    vals = []
    try:
        import numpy as _np; vals.append(f"numpy: {_np.__version__}")
    except Exception: vals.append("numpy: ?")
    try:
        import nibabel as _nib; vals.append(f"nibabel: {_nib.__version__}")
    except Exception: vals.append("nibabel: ?")
    try:
        import skimage as _ski; vals.append(f"scikit-image: {_ski.__version__}")
    except Exception: vals.append("scikit-image: ?")
    try:
        import vtk as _vtk; vals.append(f"vtk: {_vtk.vtkVersion.GetVTKVersion()}")
    except Exception: vals.append("vtk: NO")
    try:
        import pymeshlab as _ml; vals.append(f"pymeshlab: {_ml.__version__}")
    except Exception: vals.append("pymeshlab: NO")
    vals.append(f"cupy: {_cupy_ver if _HAS_CUPY else 'NO'}")
    try:
        import stl as _stl; vals.append(f"numpy-stl: {getattr(_stl,'__version__','OK')}")
    except Exception: vals.append("numpy-stl: ?")
    return "\n".join(vals)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NIfTI ➜ STL (Otsu GPU/CPU + MC CPU) — by José")
        self.geometry("840x640")
        self.resizable(True, True)
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.recursive_var = tk.BooleanVar(value=True)
        self.clip_min_var = tk.StringVar()
        self.clip_max_var = tk.StringVar()
        self.exclude_zeros_var = tk.BooleanVar(value=True)
        self.min_voxels_var = tk.StringVar(value="10")
        self.log_name_var = tk.StringVar(value="log.csv")
        self._build_ui()
        self.worker = None

    def _build_ui(self):
        pad = {'padx': 8, 'pady': 6}
        menubar = tk.Menu(self)
        sistema = tk.Menu(menubar, tearoff=0)
        sistema.add_command(label="Detectar GPU", command=self.show_gpu_info)
        sistema.add_command(label="Versiones de librerías", command=self.show_libs_info)
        sistema.add_separator()
        sistema.add_command(label="Salir", command=self.destroy)
        menubar.add_cascade(label="Sistema", menu=sistema)
        self.config(menu=menubar)

        frm = ttk.Frame(self); frm.pack(fill="x", **pad)
        ttk.Label(frm, text="Carpeta de entrada (NIfTI):").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.input_var, width=70).grid(row=0, column=1, sticky="we")
        ttk.Button(frm, text="Seleccionar…", command=self.browse_input).grid(row=0, column=2)
        ttk.Label(frm, text="Carpeta de salida (STL):").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.output_var, width=70).grid(row=1, column=1, sticky="we")
        ttk.Button(frm, text="Seleccionar…", command=self.browse_output).grid(row=1, column=2)

        opt = ttk.LabelFrame(self, text="Opciones")
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

        btns = ttk.Frame(self); btns.pack(fill="x", **pad)
        self.run_btn = ttk.Button(btns, text="▶ Ejecutar", command=self.run); self.run_btn.pack(side="left")
        ttk.Button(btns, text="Limpiar consola", command=self.clear_console).pack(side="left", padx=6)

        self.pbar = ttk.Progressbar(self, mode="indeterminate"); self.pbar.pack(fill="x", **pad)
        self.console = tk.Text(self, height=18, wrap="word"); self.console.pack(fill="both", expand=True, **pad)
        self.console.configure(state="disabled")

    def show_gpu_info(self):
        messagebox.showinfo("Detección de GPU", _gpu_info_text())

    def show_libs_info(self):
        messagebox.showinfo("Versiones de librerías", _libs_info_text())

    def browse_input(self):
        d = filedialog.askdirectory(title="Selecciona carpeta con NIfTI")
        if d: self.input_var.set(d)

    def browse_output(self):
        d = filedialog.askdirectory(title="Selecciona carpeta de salida")
        if d: self.output_var.set(d)

    def clear_console(self):
        self.console.configure(state="normal"); self.console.delete("1.0", "end"); self.console.configure(state="disabled")

    def log(self, text: str):
        self.console.configure(state="normal"); self.console.insert("end", text); self.console.see("end"); self.console.configure(state="disabled")

    def run(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("En ejecución", "Ya hay un proceso corriendo."); return
        inp, out = Path(self.input_var.get().strip()), Path(self.output_var.get().strip())
        if not inp.exists():
            messagebox.showerror("Error", "La carpeta de entrada no existe."); return
        if not out.exists():
            try: out.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                messagebox.showerror("Error", f"No se pudo crear la carpeta de salida:\n{exc}"); return

        def parse_float(s): s=s.strip(); return None if s=="" else float(s)
        def parse_int(s): return int(s.strip())

        clip_min = parse_float(self.clip_min_var.get())
        clip_max = parse_float(self.clip_max_var.get())
        min_voxels = parse_int(self.min_voxels_var.get())
        recursive = bool(self.recursive_var.get())
        exclude_zeros = bool(self.exclude_zeros_var.get())
        log_name = self.log_name_var.get().strip() or "log.csv"

        self.run_btn.config(state="disabled"); self.pbar.start(10)
        self.log("\n=== Conversión iniciada ===\n")
        self.log(f"GPU CuPy: {'Sí' if _HAS_CUPY else 'No'} | MC: {'VTK' if _HAS_VTK else 'skimage'}\n")
        if not _HAS_PYMESHLAB: self.log("HQ: PyMeshLab no instalado (se omitirá)\n")

        def progress_cb(msg): self.after(0, lambda: self.log(msg))
        def done_cb(ok, msg):
            def _done():
                self.pbar.stop(); self.run_btn.config(state="normal")
                (messagebox.showinfo if ok else messagebox.showwarning)("Finalizado", msg)
            self.after(0, _done)

        self.worker = threading.Thread(
            target=convert_folder,
            args=(inp, out, recursive, clip_min, clip_max, exclude_zeros, min_voxels, log_name, progress_cb, done_cb),
            daemon=True
        ); self.worker.start()


if __name__ == "__main__":
    try:
        App().mainloop()
    except Exception as e:
        print(f"FATAL {e}", file=sys.stderr); sys.exit(1)
