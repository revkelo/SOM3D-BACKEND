from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from pathlib import Path
import shutil, uuid, tempfile, zipfile
import sys
import io

from app.pipelines.segmentor import run_segmentation

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

def _which(cmd: str):
    from shutil import which
    return which(cmd)

def _has_dicoms(folder: Path) -> bool:
    # ¿Hay .dcm en algún nivel?
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".dcm":
            return True
    # Heurística por magic "DICM"
    cnt = 0
    for p in folder.rglob("*"):
        if p.is_file():
            try:
                with open(p, "rb") as fh:
                    fh.seek(128)
                    if fh.read(4) == b"DICM":
                        return True
            except Exception:
                pass
            cnt += 1
            if cnt > 100:
                break
    return False

def _prepare_stl_dirs() -> Path:
    stl_root = Path("media") / "stl"
    if stl_root.exists() and not stl_root.is_dir():
        raise HTTPException(
            status_code=500,
            detail="Existe un archivo en 'media/stl'. Debe ser una carpeta. Borra o renómbralo y reintenta."
        )
    (stl_root / "originales").mkdir(parents=True, exist_ok=True)
    (stl_root / "reducidos").mkdir(parents=True, exist_ok=True)
    (stl_root / "packs").mkdir(parents=True, exist_ok=True)
    return stl_root

def _save_upload_to_temp(file: UploadFile, td_path: Path) -> tuple[Path, Path, Path]:
    case_dir = td_path / "case"; case_dir.mkdir(parents=True, exist_ok=True)
    out_case = td_path / "nifti"; out_case.mkdir(parents=True, exist_ok=True)
    upload_path = td_path / Path(file.filename).name
    with upload_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    # Normalizar a carpeta con serie
    suf = upload_path.suffix.lower()
    if suf == ".zip":
        try:
            with zipfile.ZipFile(upload_path, "r") as z:
                z.extractall(case_dir)
        except Exception as e:
            raise HTTPException(400, f"ZIP inválido o corrupto: {e}")
    elif suf == ".dcm":
        shutil.copy(upload_path, case_dir / upload_path.name)
    else:
        raise HTTPException(400, "Sube .zip (serie DICOM) o .dcm")
    if not _has_dicoms(case_dir):
        raise HTTPException(400, "El ZIP no contiene archivos DICOM (.dcm) válidos.")
    return case_dir, out_case, upload_path

def _iter_valid_niis(out_case: Path, min_voxels: int):
    import nibabel as nib
    import numpy as np
    nii_files = sorted(list(out_case.rglob("*.nii.gz")) + list(out_case.rglob("*.nii")))
    if not nii_files:
        raise HTTPException(404, "No se generó ningún NIfTI. Revisa la validez de la serie DICOM o la tarea.")
    for p in nii_files:
        nz = 0
        try:
            img = nib.load(str(p))
            vol = img.get_fdata(dtype=np.float32)
            nz = int(np.count_nonzero(vol))
        except Exception:
            nz = 0
        yield p, nz, (nz >= min_voxels)

# ============================================================
#  A) Endpoint: primer STL utilizable
# ============================================================
@router.post("/dicom-to-stl")
async def dicom_to_stl(
    file: UploadFile = File(..., description="ZIP con serie DICOM (o .dcm demo)"),
    task: str = Query("total", description="TotalSegmentator task"),
    threshold_min: int = 150,
    threshold_max: int = 3075,
    keep_ratio: float = 0.5,
    laplacian_lambda: float = 0.5,
    laplacian_iters: int = 10,
    min_voxels: int = 50,
    downsample_factor: int = 2,
):
    from app.pipelines.generadorSTL import NiiToStlConverter  # import diferido

    # (Opcional) si segmentor usa el PATH; si cambiaste a `python -m`, puedes quitarlo.
    if _which("TotalSegmentator") is None:
        raise HTTPException(
            status_code=500,
            detail="TotalSegmentator no está instalado/visible en este entorno. Instálalo en el mismo venv."
        )

    stl_root = _prepare_stl_dirs()

    try:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            case_dir, out_case, _ = _save_upload_to_temp(file, td_path)

            # 1) SEGMENTACIÓN → NIfTI
            try:
                run_segmentation(
                    input_path=case_dir,
                    output_path=out_case,
                    task=task,
                    force_cpu=False,
                    keep_temp=False,
                    extra_args=None,
                )
            except Exception as e:
                raise HTTPException(400, f"Fallo en segmentación (TotalSegmentator): {e}")

            # 2) iterar y saltar volúmenes vacíos
            revisados = []
            for p, nz, ok in _iter_valid_niis(out_case, min_voxels):
                if not ok:
                    revisados.append(f"{p.name}: nz={nz} (skip)")
                    continue
                # 3) conversión del NIfTI elegido
                try:
                    conv = NiiToStlConverter(
                        input_dir=str(out_case),
                        output_root=str(stl_root),
                        label_value=None,
                        threshold_min=threshold_min,
                        threshold_max=threshold_max,
                        keep_ratio=keep_ratio,
                        laplacian_lambda=laplacian_lambda,
                        laplacian_iters=laplacian_iters,
                        min_voxels=min_voxels,
                        downsample_factor=downsample_factor,
                        recursive=False,
                    )
                    res = conv.process_one_file(str(p))
                except Exception as e:
                    revisados.append(f"{p.name}: convert_fail={e}")
                    continue

                if not res.get("success"):
                    revisados.append(f"{p.name}: convert_fail={res.get('message','unknown')}")
                    continue

                stl_abs = Path(res["stl"])
                stl_red_abs = Path(res["stl_reduced"])
                print(f"[DEBUG] NIfTI elegido: {p.name} | nz={nz}", file=sys.stdout)
                return {
                    "ok": True,
                    "nifti": p.name,
                    "stl_original": stl_abs.name,
                    "stl_reducido": stl_red_abs.name,
                    "url_original": f"/pipeline/stl/originales/{stl_abs.name}",
                    "url_reducido": f"/pipeline/stl/reducidos/{stl_red_abs.name}",
                    "metodo_reduccion": res.get("method"),
                    "voxels_mask": res.get("voxels_mask"),
                    "revisados": revisados[:10],
                }

            # ninguno sirvió
            msg = "No se encontró un NIfTI utilizable (todos vacíos o conversión fallida)."
            if revisados:
                msg += " Revisados: " + "; ".join(revisados[:10])
            raise HTTPException(404, msg)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inesperado: {type(e).__name__}: {e}")

# ============================================================
#  B) Endpoint: TODOS los STL (batch)
# ============================================================
@router.post("/dicom-to-stl-all")
async def dicom_to_stl_all(
    file: UploadFile = File(..., description="ZIP con serie DICOM (o .dcm demo)"),
    task: str = Query("total", description="TotalSegmentator task"),
    threshold_min: int = 150,
    threshold_max: int = 3075,
    keep_ratio: float = 0.5,
    laplacian_lambda: float = 0.5,
    laplacian_iters: int = 10,
    min_voxels: int = 50,
    downsample_factor: int = 2,
    pack_zip: bool = Query(False, description="Si true, genera un ZIP con los STL reducidos"),
    include_originals_in_zip: bool = Query(False, description="Si true + pack_zip, incluye originales también"),
):
    from app.pipelines.generadorSTL import NiiToStlConverter  # import diferido

    if _which("TotalSegmentator") is None:
        raise HTTPException(
            status_code=500,
            detail="TotalSegmentator no está instalado/visible en este entorno. Instálalo en el mismo venv."
        )

    stl_root = _prepare_stl_dirs()

    try:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            case_dir, out_case, up = _save_upload_to_temp(file, td_path)

            # 1) SEGMENTACIÓN → NIfTI
            try:
                run_segmentation(
                    input_path=case_dir,
                    output_path=out_case,
                    task=task,
                    force_cpu=False,
                    keep_temp=False,
                    extra_args=None,
                )
            except Exception as e:
                raise HTTPException(400, f"Fallo en segmentación (TotalSegmentator): {e}")

            # 2) iterar todos los NIfTI no vacíos y convertirlos
            successes = []
            failures = []
            for p, nz, ok in _iter_valid_niis(out_case, min_voxels):
                if not ok:
                    failures.append({"nifti": p.name, "reason": f"nz={nz} < min_voxels({min_voxels})"})
                    continue
                try:
                    conv = NiiToStlConverter(
                        input_dir=str(out_case),
                        output_root=str(stl_root),
                        label_value=None,
                        threshold_min=threshold_min,
                        threshold_max=threshold_max,
                        keep_ratio=keep_ratio,
                        laplacian_lambda=laplacian_lambda,
                        laplacian_iters=laplacian_iters,
                        min_voxels=min_voxels,
                        downsample_factor=downsample_factor,
                        recursive=False,
                    )
                    res = conv.process_one_file(str(p))
                    if res.get("success"):
                        stl_abs = Path(res["stl"])
                        stl_red_abs = Path(res["stl_reduced"])
                        successes.append({
                            "nifti": p.name,
                            "nz": nz,
                            "stl_original": stl_abs.name,
                            "stl_reducido": stl_red_abs.name,
                            "url_original": f"/pipeline/stl/originales/{stl_abs.name}",
                            "url_reducido": f"/pipeline/stl/reducidos/{stl_red_abs.name}",
                            "metodo_reduccion": res.get("method"),
                            "voxels_mask": res.get("voxels_mask"),
                        })
                    else:
                        failures.append({"nifti": p.name, "reason": f"convert_fail: {res.get('message','unknown')}"})
                except Exception as e:
                    failures.append({"nifti": p.name, "reason": f"convert_exception: {e}"})

            if not successes:
                raise HTTPException(404, "No se pudo convertir ningún NIfTI a STL (todos vacíos o conversión fallida).")

            pack_url = None
            if pack_zip:
                uid = uuid.uuid4().hex[:8]
                pack_name = f"stls_{Path(up).stem}_{uid}.zip"
                pack_path = stl_root / "packs" / pack_name
                with zipfile.ZipFile(pack_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for s in successes:
                        if include_originals_in_zip:
                            zf.write(stl_root / "originales" / s["stl_original"], arcname=f"originales/{s['stl_original']}")
                        zf.write(stl_root / "reducidos" / s["stl_reducido"], arcname=f"reducidos/{s['stl_reducido']}")
                pack_url = f"/pipeline/stl/pack/{pack_name}"

            return {
                "ok": True,
                "count": len(successes),
                "results": successes,
                "failures": failures[:20],
                "zip_url": pack_url,
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inesperado: {type(e).__name__}: {e}")

# ============================================================
#  Descarga de STL y paquetes
# ============================================================
@router.get("/stl/{kind}/{filename}")
def download_stl(kind: str, filename: str):
    if kind not in {"originales", "reducidos"}:
        raise HTTPException(400, "kind debe ser 'originales' o 'reducidos'")
    path = Path("media/stl") / kind / filename
    if not path.exists():
        raise HTTPException(404, "STL no encontrado")
    return FileResponse(path, media_type="model/stl", filename=path.name)

@router.get("/stl/pack/{filename}")
def download_pack(filename: str):
    path = Path("media/stl/packs") / filename
    if not path.exists():
        raise HTTPException(404, "Paquete ZIP no encontrado")
    return FileResponse(path, media_type="application/zip", filename=path.name)
