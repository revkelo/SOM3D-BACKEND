# pyright: reportMissingImports=false
import os
import csv
import math
import importlib
import numpy as np
import nibabel as nib
from skimage import measure
from skimage.filters import threshold_otsu
from skimage.measure import label as cc_label
from scipy import ndimage
import trimesh
from trimesh.smoothing import filter_laplacian


class NiiToStlConverter:
    def __init__(self, input_dir, output_root,
                 label_value=None, threshold_min=150, threshold_max=3075,
                 keep_ratio=0.5, laplacian_lambda=0.5, laplacian_iters=10,
                 min_voxels=50, downsample_factor=2, recursive=False):
        self.input_dir = input_dir
        self.output_root = output_root
        self.label_value = label_value
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.keep_ratio = keep_ratio
        self.laplacian_lambda = laplacian_lambda
        self.laplacian_iters = laplacian_iters
        self.min_voxels = min_voxels
        self.downsample_factor = downsample_factor
        self.recursive = recursive

        # Crear subcarpetas
        self.original_dir = os.path.join(self.output_root, "originales")
        self.reducido_dir = os.path.join(self.output_root, "reducidos")
        os.makedirs(self.original_dir, exist_ok=True)
        os.makedirs(self.reducido_dir, exist_ok=True)

    def to_stl_paths(self, name):
        return (
            os.path.join(self.original_dir, f"{name}.stl"),
            os.path.join(self.reducido_dir, f"{name}-reducido.stl"),
        )

    # ===============================
    # UTILIDADES
    # ===============================
    def list_nii_files(self):
        exts = (".nii", ".nii.gz")
        files = []
        if self.recursive:
            for root, _, fns in os.walk(self.input_dir):
                for fn in fns:
                    if fn.lower().endswith(exts):
                        files.append(os.path.join(root, fn))
        else:
            for fn in os.listdir(self.input_dir):
                if fn.lower().endswith(exts):
                    files.append(os.path.join(self.input_dir, fn))
        files.sort()
        return files

    @staticmethod
    def safe_basename(path):
        base = os.path.basename(path)
        if base.lower().endswith(".nii.gz"):
            base = base[:-7]
        elif base.lower().endswith(".nii"):
            base = base[:-4]
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base)

    @staticmethod
    def is_binary_like(vol):
        sample = vol[::4, ::4, ::4]
        us = np.unique(sample)
        if us.size <= 6 and us.min() >= 0 and us.max() <= 1 and np.allclose(us, np.round(us)):
            return True
        return False

    @staticmethod
    def keep_largest_component(mask_bool):
        lbl = cc_label(mask_bool, connectivity=1)
        n = int(lbl.max())
        if n == 0:
            return mask_bool
        counts = np.bincount(lbl.ravel())
        counts[0] = 0
        largest = counts.argmax()
        return (lbl == largest)

    def segment_volume(self, vol):
        nz = int(np.count_nonzero(vol))
        if nz == 0:
            return None, "empty_all_zero"

        if self.label_value is not None:
            mask = (vol == float(self.label_value))
            return mask, f"label=={self.label_value}"

        if self.is_binary_like(vol):
            mask = (vol > 0.5)
            return mask, "binary_like>0.5"

        mask = (vol > self.threshold_min) & (vol < self.threshold_max)
        if mask.sum() >= self.min_voxels:
            return mask, "range_threshold"

        try:
            t = threshold_otsu(vol)
            mask = vol > t
            return mask, f"otsu(t={t:.2f})"
        except Exception:
            return None, "segmentation_failed"

    # ===============================
    # PIPELINE POR ARCHIVO
    # ===============================
    def process_one_file(self, nii_path):
        name = self.safe_basename(nii_path)
        out_stl, out_stl_reduced = self.to_stl_paths(name)

        result = {
            "file": nii_path, "name": name, "success": False, "method": None,
            "faces_before": None, "faces_after": None, "voxels_mask": None,
            "stl": out_stl, "stl_reduced": out_stl_reduced, "message": ""
        }

        try:
            # 1) Cargar NIfTI
            img = nib.load(nii_path)
            vol = img.get_fdata(dtype=np.float32)
            voxel_sizes = tuple(float(z) for z in img.header.get_zooms()[:3])
            vmin_all, vmax_all = float(np.nanmin(vol)), float(np.nanmax(vol))

            print(f"\n[{name}] Volumen: {vol.shape}, voxel_sizes(mm): {voxel_sizes}, range={vmin_all:.2f}..{vmax_all:.2f}")

            # 2) Segmentación
            mask, mode = self.segment_volume(vol)
            if mask is None:
                result["message"] = f"Skip: volumen vacío ({mode})"
                print(f"[{name}] ⏭️  {result['message']}")
                return result

            vox = int(mask.sum())
            result["voxels_mask"] = vox
            if vox < self.min_voxels:
                result["message"] = f"Skip: máscara demasiado pequeña ({vox} vox.)"
                print(f"[{name}] ⏭️  {result['message']}")
                return result

            # Componente mayor + cierre/relleno
            mask_orig = mask.astype(bool)
            mask = self.keep_largest_component(mask_orig)
            mask = ndimage.binary_closing(mask, structure=ndimage.generate_binary_structure(3, 1), iterations=1)
            mask = ndimage.binary_fill_holes(mask)

            if int(mask.sum()) < self.min_voxels:
                mask = self.keep_largest_component(mask_orig)

            # Suavizado
            smoothed = ndimage.gaussian_filter(mask.astype(np.float32), sigma=1)
            if float(smoothed.max()) == float(smoothed.min()):
                smoothed = mask.astype(np.float32)

            # 3) Marching Cubes
            vmin, vmax = float(smoothed.min()), float(smoothed.max())
            level = 0.5 if (vmin <= 0.5 <= vmax) else (vmin + 0.5 * (vmax - vmin))
            print(f"[{name}] Segmentación: {mode} | MC level={level:.3f} (range {vmin:.3f}-{vmax:.3f})")

            verts, faces, normals, _ = measure.marching_cubes(
                volume=smoothed, level=level, spacing=(1.0, 1.0, 1.0)
            )

            # Ahora aplica affine UNA sola vez
            verts = nib.affines.apply_affine(img.affine, verts)

            # STL base con trimesh
            tm = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
            tm.remove_unreferenced_vertices()
            tm.remove_degenerate_faces()
            tm.fix_normals()

            if not tm.is_watertight:
                tm.fill_holes()  # FIX: rellenar huecos
            tm.export(out_stl)
            print(f"[{name}] ✅ STL base: {out_stl}")


            # Decimación
            method, reduced_mesh = None, None
            try:
                fs = importlib.import_module("fast_simplification")
                target_faces = max(1000, int(tm.faces.shape[0] * self.keep_ratio))
                v_dec, f_dec = fs.simplify(tm.vertices.astype(np.float64), tm.faces.astype(np.int64), target_count=target_faces)
                reduced_mesh = trimesh.Trimesh(vertices=v_dec, faces=f_dec, process=True)
                method = "fast_simplification"
            except Exception as e:
                print(f"[{name}] [WARN] fast_simplification no disponible: {e}")

            if reduced_mesh is None:
                try:
                    o3d = importlib.import_module("open3d")
                    o3 = o3d.geometry.TriangleMesh(
                        o3d.utility.Vector3dVector(tm.vertices),
                        o3d.utility.Vector3iVector(tm.faces)
                    )
                    target_faces = max(1000, int(tm.faces.shape[0] * self.keep_ratio))
                    o3 = o3.simplify_quadric_decimation(target_faces)
                    reduced_mesh = trimesh.Trimesh(np.asarray(o3.vertices), np.asarray(o3.triangles), process=True)
                    method = "open3d"
                except Exception as e:
                    print(f"[{name}] [WARN] open3d no disponible: {e}")

            if reduced_mesh is None:
                try:
                    ml = importlib.import_module("pymeshlab")
                    ms = ml.MeshSet()
                    m = ml.Mesh(vertex_matrix=tm.vertices, face_matrix=tm.faces)
                    ms.add_mesh(m, "mesh")
                    target_faces = max(1000, int(tm.faces.shape[0] * self.keep_ratio))
                    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces, preservenormal=True)
                    m2 = ms.current_mesh()
                    reduced_mesh = trimesh.Trimesh(m2.vertex_matrix(), m2.face_matrix(), process=True)
                    method = "pymeshlab"
                except Exception as e:
                    print(f"[{name}] [WARN] pymeshlab no disponible: {e}")

            if reduced_mesh is None:
                print(f"[{name}] [Fallback] Downsampling volumétrico + MC")
                zf = 1.0 / float(self.downsample_factor)
                smoothed_ds = ndimage.zoom(smoothed, zoom=zf, order=0)
                spacing_ds = tuple(s * self.downsample_factor for s in voxel_sizes)
                vmin2, vmax2 = float(smoothed_ds.min()), float(smoothed_ds.max())
                level2 = 0.5 if (vmin2 <= 0.5 <= vmax2) else (vmin2 + 0.5 * (vmax2 - vmin2))
                verts2, faces2, _, _ = measure.marching_cubes(volume=smoothed_ds, level=level2, spacing=spacing_ds)
                reduced_mesh = trimesh.Trimesh(vertices=verts2, faces=faces2, process=True)
                method = f"downsample_volume(x{self.downsample_factor})"

            reduced_mesh.remove_unreferenced_vertices()
            reduced_mesh.remove_degenerate_faces()
            reduced_mesh.fix_normals()

            if not reduced_mesh.is_watertight:
                reduced_mesh.fill_holes()  # FIX

            filter_laplacian(reduced_mesh, lamb=self.laplacian_lambda, iterations=self.laplacian_iters)
            reduced_mesh.export(out_stl_reduced)

            result["faces_after"] = int(reduced_mesh.faces.shape[0])
            result["method"] = method
            result["success"] = True
            print(f"[{name}] ✅ STL reducido y suavizado ({method}): {out_stl_reduced}")

        except Exception as e:
            result["message"] = str(e)
            print(f"[{name}] ❌ Error: {e}")

        return result

    # ===============================
    # MAIN
    # ===============================
    def run(self):
        files = self.list_nii_files()
        if not files:
            print("No se encontraron .nii/.nii.gz en la carpeta indicada.")
            return

        log_path = os.path.join(self.output_root, "log.csv")
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "name", "success", "method", "faces_before", "faces_after",
                             "voxels_mask", "stl", "stl_reduced", "message"])
            for path in files:
                res = self.process_one_file(path)
                writer.writerow([res["file"], res["name"], res["success"], res["method"],
                                 res["faces_before"], res["faces_after"], res["voxels_mask"],
                                 os.path.relpath(res["stl"], self.output_root),
                                 os.path.relpath(res["stl_reduced"], self.output_root),
                                 res["message"]])
        print(f"\n📄 Log guardado en: {log_path}")


# Ejemplo de uso
if __name__ == "__main__":
    input_dir   = r"Y:/SALIDAS/Craneo/nii"
    output_root = r"C:/Users/K/Documents/GitHub/SOM3D-BACKEND/otros/STL_out"

    converter = NiiToStlConverter(input_dir, output_root)
    converter.run()
