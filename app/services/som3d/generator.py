#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, time, shutil
from pathlib import Path
import numpy as np

_HAS_CUPY = False
_cupy_ver = "unknown"
_CUPY = None
_CUPY_TRIED = False

GPU_ONLY_OTSU = str(os.getenv("SOM3D_GPU_ONLY_OTSU", "false")).strip().lower() in ("1", "true", "yes", "y", "on")


def _get_cupy():
    global _HAS_CUPY, _cupy_ver, _CUPY, _CUPY_TRIED
    if _CUPY_TRIED:
        return _CUPY
    _CUPY_TRIED = True
    try:
        os.environ.setdefault("CUPY_DONT_WARN_ON_CUDA_PATH", "1")
        import cupy as cp  # type: ignore
        _CUPY = cp
        _HAS_CUPY = True
        _cupy_ver = getattr(cp, "__version__", "unknown")
    except Exception:
        _CUPY = None
        _HAS_CUPY = False
        _cupy_ver = "unavailable"
    return _CUPY

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



def find_nii_files(root: Path, recursive: bool) -> list[Path]:
    pats = ["*.nii", "*.nii.gz"]
    files: list[Path] = []
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
    cp = _get_cupy()
    if cp is None:
        raise RuntimeError("GPU Otsu requiere CuPy.")
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
    cp = _get_cupy()
    if cp is None:
        raise RuntimeError("GPU Otsu requiere CuPy.")
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

def _fmt_size(nbytes: int) -> str:
    try:
        u = ["B","KiB","MiB","GiB","TiB"]
        size = float(nbytes); i = 0
        while size >= 1024.0 and i < len(u)-1:
            size /= 1024.0; i += 1
        return f"{size:.2f} {u[i]}"
    except Exception:
        return f"{nbytes} B"



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

    for f, kw in [
        ('meshing_remove_duplicate_vertices', {}),
        ('meshing_remove_duplicate_faces', {}),
        ('meshing_remove_null_faces', {}),
        ('meshing_remove_unreferenced_vertices', {}),
    ]:
        try: ms.apply_filter(f, **kw)
        except Exception: pass

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

    desired = None
    if target_faces is not None:
        desired = max(200, int(target_faces))
    elif max_faces is not None:
        desired = max(200, min(faces_before, int(max_faces)))

    def _decimate_to_faces(target_faces: int, keep_topology: bool = True):
        ms.apply_filter(
            'meshing_decimation_quadric_edge_collapse',
            targetfacenum=int(max(200, target_faces)),
            preservenormal=True,
            planarquadric=True,
            optimalplacement=True,
            qualitythr=0.5,
            boundaryweight=0.6,
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
            qualitythr=0.5,
            boundaryweight=0.6,
            preservetopology=bool(keep_topology)
        )

    if desired is not None and desired < faces_before:
        rounds = 0
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
                for hard_ratio in (0.75, 0.60, 0.45, 0.35):
                    if int(ms.current_mesh().face_number()) <= desired: break
                    rel_ratio = desired / max(1, int(ms.current_mesh().face_number()))
                    _decimate_to_ratio(rel_ratio, keep_topology=preserve_topology)
                    _decimate_to_ratio(hard_ratio, keep_topology=preserve_topology)
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

    try:
        curr_faces = int(ms.current_mesh().face_number())
        if desired is not None and curr_faces > int(0.90 * faces_before):
            _decimate_to_ratio(0.50, keep_topology=preserve_topology)
            _decimate_to_ratio(0.35, keep_topology=preserve_topology)
    except Exception:
        pass

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

    try:
        ms.apply_filter('meshing_close_holes', maxholesize=min(80, close_holes_maxsize))
    except Exception:
        pass

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



class NiftiToSTLConverter:
    """
    Conversor de carpetas NIfTI -> STL con:
      - Otsu GPU (CuPy) o CPU (fallback o modo solo-GPU)
      - Marching Cubes vía VTK (preferido) o scikit-image (fallback)
      - STL original + STL HQ (suavizado-only si pequeño; si no, decimation agresivo + suavizado)
    """

    def __init__(self, *, progress_cb=None):
        self.progress_cb = progress_cb or (lambda msg: print(msg, end=""))

    def convert_folder(
        self,
        input_dir: Path | str,
        output_dir: Path | str,
        *,
        recursive: bool = True,
        clip_min: float | None = None,
        clip_max: float | None = None,
        exclude_zeros: bool = True,
        min_voxels: int = 10,
    ) -> dict:
        """
        Procesa todos los .nii/.nii.gz en input_dir y guarda STL en output_dir.
        Retorna un dict(resumen).
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        start_total = time.perf_counter()
        files = find_nii_files(input_dir, recursive)
        total, successes = len(files), 0
        results: list[dict] = []

        _get_cupy()

        originals_dir = output_dir / "originales"
        hq_dir = output_dir / "hq_suavizado_decimado"

        self._log(f"\n=== Conversión iniciada ===\n")
        self._log(f"GPU CuPy: {'Sí' if _HAS_CUPY else 'No'} | MC: {'VTK' if _HAS_VTK else 'skimage'}\n")
        if not _HAS_PYMESHLAB: self._log("HQ: PyMeshLab no instalado (se omitirá)\n")
        self._log(f"Archivos NIfTI: {total}\n")
        if total == 0:
            return {"ok": False, "message": "No se encontraron archivos.", "results": []}

        for idx, f in enumerate(files, start=1):
            t0 = time.perf_counter()
            self._log(f"[{idx}/{total}] {f.name}\n")
            try:
                vol, affine = load_nifti(f)
                spacing = voxel_spacing_from_affine(affine)
                vol = _pre_smooth_volume(vol, spacing, f.name)

                if GPU_ONLY_OTSU:
                    if not _HAS_CUPY:
                        raise RuntimeError("Modo solo-GPU activo pero CuPy no está disponible.")
                    mask_np, t = mask_from_otsu_gpu_only(vol, clip_min, clip_max, exclude_zeros)
                    used_otsu = "gpu(cupy)"
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

                stl_orig_bytes = out_stl_orig.stat().st_size if out_stl_orig.exists() else 0
                stl_orig_hr = _fmt_size(stl_orig_bytes)

                verts_before = int(verts.shape[0])
                verts_after = int(verts.shape[0])
                faces_after = faces.shape[0]
                out_hq = ""
                stl_hq_bytes = 0
                stl_hq_hr = ""
                size_saving_pct = 0.0
                method = f"otsu-{used_otsu} + MC({used_backend})"

                skip_hq = (faces.shape[0] < 5000) or (stl_orig_bytes < 300 * 1024)
                out_stl_hq = None
                if _HAS_PYMESHLAB:
                    out_stl_hq = hq_dir / (f.stem.replace(".nii", "") + "_HQ.stl")

                if _HAS_PYMESHLAB and skip_hq:
                    try:
                        ms = ml.MeshSet()
                        ms.add_mesh(ml.Mesh(verts.astype(np.float32), faces.astype(np.int32)), "small")
                        nfaces = faces.shape[0]
                        base_steps = 50 if nfaces < 2000 else 70
                        ms.apply_filter('apply_coord_taubin_smoothing',
                                        stepsmoothnum=base_steps,
                                        lambda_=0.60, mu=-0.63)
                        try:
                            ms.apply_filter('apply_coord_laplacian_smoothing_surface_preserving',
                                            stepsmoothnum=15, cotangentweight=True)
                        except Exception:
                            pass
                        for flt in ('compute_normal_per_vertex', 'compute_normal_per_face'):
                            try: ms.apply_filter(flt)
                            except Exception: pass
                        out_stl_hq.parent.mkdir(parents=True, exist_ok=True)
                        ms.save_current_mesh(str(out_stl_hq), binary=True)
                        out_hq = str(out_stl_hq)
                        stl_hq_bytes = out_stl_hq.stat().st_size if out_stl_hq.exists() else 0
                        stl_hq_hr = _fmt_size(stl_hq_bytes)
                        size_saving_pct = (1.0 - (stl_hq_bytes / stl_orig_bytes)) * 100.0 if stl_orig_bytes > 0 else 0.0
                        method = f"{method} | HQ=suavizado-only(taubin/laplaciano)"
                    except Exception as e:
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
                        verts_after = int(metrics.get("verts_after") or verts_after)
                        out_hq = str(out_stl_hq)

                        stl_hq_bytes = out_stl_hq.stat().st_size if out_stl_hq.exists() else 0
                        stl_hq_hr = _fmt_size(stl_hq_bytes)
                        size_saving_pct = (1.0 - (stl_hq_bytes / stl_orig_bytes)) * 100.0 if stl_orig_bytes > 0 else 0.0
                        method += f" + HQ(targetfacenum={metrics.get('target_faces')})"

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
                    method += " (sin HQ)"

                elapsed = time.perf_counter() - t0
                if _HAS_PYMESHLAB and out_hq:
                    signo = "menos" if size_saving_pct >= 0 else "más"
                    self._log(
                        f"OK | Otsu({used_otsu}) t={t:.3f} | voxels={voxels} | caras={faces_after} | "
                        f"STL: {stl_orig_hr} → HQ: {stl_hq_hr} ({abs(size_saving_pct):.1f}% {signo}) | {elapsed:.2f}s\n"
                    )
                else:
                    self._log(
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
                    "verts_before": verts_before,
                    "verts_after": verts_after,
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
                self._log(f"FAIL | {elapsed:.2f}s | {exc}\n")
                results.append({
                    "file": str(f), "name": f.name, "success": False, "method": "auto",
                    "auto_t": "", "verts_before": "", "verts_after": "", "faces_before": "", "faces_after": "", "voxels_mask": "",
                    "stl": "", "stl_size_bytes": "", "stl_size_hr": "",
                    "stl_reduced": "", "stl_reduced_size_bytes": "", "stl_reduced_size_hr": "",
                    "size_saving_pct": "",
                    "message": str(exc), "seconds": f"{elapsed:.3f}",
                })


        try:
            tot_orig = sum(int(r["stl_size_bytes"]) for r in results if r.get("stl_size_bytes"))
            tot_hq   = sum(int(r["stl_reduced_size_bytes"]) for r in results if r.get("stl_reduced_size_bytes"))
            if tot_orig > 0 and tot_hq > 0:
                pct = (1.0 - (tot_hq / tot_orig)) * 100.0
                self._log(f"\nTamaño total original: {_fmt_size(tot_orig)}\n")
                self._log(f"Tamaño total HQ:       {_fmt_size(tot_hq)}\n")
                self._log(f"Ahorro total:          {pct:.1f}%\n")
        except Exception:
            pass

        total_elapsed = time.perf_counter() - start_total
        self._log(f"\nResumen: {successes}/{total} OK | {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)\n")
        return {
            "ok": True,
            "message": f"Completado: {successes}/{total} OK",
            "results": results,
            "elapsed_sec": total_elapsed
        }

    def convert_file(
        self,
        nifti_path: Path | str,
        output_dir: Path | str,
        *,
        clip_min: float | None = None,
        clip_max: float | None = None,
        exclude_zeros: bool = True,
        min_voxels: int = 10,
    ) -> dict:
        """Convierte un único archivo NIfTI a STL(s) y retorna el registro de resultados."""
        input_dir = Path(nifti_path).parent
        out_dir = Path(output_dir)
        return self.convert_folder(
            input_dir, out_dir,
            recursive=False,
            clip_min=clip_min, clip_max=clip_max,
            exclude_zeros=exclude_zeros, min_voxels=min_voxels,
        )["results"][0]

    def _log(self, msg: str):
        try:
            self.progress_cb(msg)
        except Exception:
            print(msg, end="")


if __name__ == "__main__":
    """
    Ejemplo:
      python nifti_to_stl.py "Z:/DICOM/Casos" "C:/_out_stl" -- no args extra

    Ajusta las rutas abajo o invócalo desde otro script importando la clase:
        from nifti_to_stl import NiftiToSTLConverter
        conv = NiftiToSTLConverter()
        conv.convert_folder("Z:/DICOM/Casos", "C:/_out_stl", recursive=True)
    """
    import argparse
    ap = argparse.ArgumentParser(description="NIfTI -> STL (Otsu GPU/CPU + MC + HQ)")
    ap.add_argument("input_dir", type=str, help="Carpeta con .nii / .nii.gz")
    ap.add_argument("output_dir", type=str, help="Carpeta de salida")
    ap.add_argument("--no-recursive", action="store_true", help="No buscar recursivamente")
    ap.add_argument("--clip-min", type=float, default=None)
    ap.add_argument("--clip-max", type=float, default=None)
    ap.add_argument("--include-zeros", action="store_true", help="No excluir ceros antes de Otsu")
    ap.add_argument("--min-voxels", type=int, default=10)
    args = ap.parse_args()

    conv = NiftiToSTLConverter()
    conv.convert_folder(
        args.input_dir,
        args.output_dir,
        recursive=not args.no_recursive,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        exclude_zeros=not args.include_zeros,
        min_voxels=args.min_voxels,
        
    )
