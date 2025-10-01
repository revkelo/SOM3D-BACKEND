# radiomics_task.py
import argparse, json, pathlib, sys
from radiomics import featureextractor

def run_radiomics(image_path, mask_path):
    extractor = featureextractor.RadiomicsFeatureExtractor()
    res = extractor.execute(str(image_path), str(mask_path))
    out = {}
    for k, v in res.items():
        try:
            out[k] = float(v)
        except Exception:
            out[k] = str(v)
    return out

def main():
    p = argparse.ArgumentParser(description="PyRadiomics batch → statistics_radiomics.json")
    p.add_argument("--images", required=True, help="Carpeta con NIfTI de imagen (CT/MR)")
    p.add_argument("--masks", required=True, help="Carpeta con máscaras NIfTI")
    p.add_argument("--out", default="statistics_radiomics.json", help="Ruta de salida JSON")
    p.add_argument("--pattern-image", default="*.nii*", help="Patrón para imágenes")
    p.add_argument("--pattern-mask", default="*.nii*", help="Patrón para máscaras")
    args = p.parse_args()

    img_dir = pathlib.Path(args.images)
    msk_dir = pathlib.Path(args.masks)
    out_path = pathlib.Path(args.out)

    images = sorted(img_dir.glob(args.pattern-image))
    masks  = sorted(msk_dir.glob(args.pattern_mask))

    if not images or not masks:
        print("No se encontraron imágenes o máscaras. Revisa rutas/patrones.", file=sys.stderr)
        sys.exit(2)

    # Empareja simple por nombre base (ajústalo a tu lógica)
    # p.ej: imagen 'body.nii.gz' con máscara 'body_mask.nii.gz'
    index_masks = {m.name.lower().replace("mask_", "").replace("_mask", ""): m for m in masks}
    results = []

    for im in images:
        key = im.name.lower()
        # intenta variantes
        candidates = [
            f"mask_{im.name.lower()}",
            key.replace(".nii.gz", "_mask.nii.gz"),
            key.replace(".nii", "_mask.nii"),
            key
        ]
        m = None
        for c in candidates:
            m = index_masks.get(c)
            if m:
                break
        if not m:
            # usa la primera máscara si no hay emparejamiento (o sáltalo)
            # continue
            m = masks[0]

        try:
            feat = run_radiomics(im, m)
            results.append({
                "image": im.name,
                "mask": m.name,
                "features": feat
            })
            print(f"[OK] {im.name}  +  {m.name}")
        except Exception as e:
            print(f"[ERR] {im.name}: {e}", file=sys.stderr)

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nEscrito: {out_path.resolve()}")

if __name__ == "__main__":
    main()
