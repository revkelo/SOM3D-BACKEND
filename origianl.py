#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script minimalista: elegir carpeta de NIfTI y carpeta de salida y convertir a STL.
- Sin modos por lotes
- Sin decenas de flags
- Muestra un resumen al final
"""

from pathlib import Path
import time
import sys

# --- GUI (opcional) ---
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    GUI_OK = True
except Exception:
    GUI_OK = False

# --- Pipeline existente ---
try:
    from app.pipelines.generadorSTL import NiiToStlConverter
except Exception as e:
    print("ERROR: No se pudo importar NiiToStlConverter desde app.pipelines.generadorSTL")
    print(f"Detalle: {e}")
    sys.exit(1)

# --- Parámetros por defecto (ajusta si lo necesitas) ---
THRESH_MIN = 150
THRESH_MAX = 3075
KEEP_RATIO = 0.5           # fracción de caras a mantener en malla reducida
LAPL_LAMBDA = 0.5
LAPL_ITERS = 10
MIN_VOXELS = 50
DOWNSAMPLE = 2
RECURSIVE = True           # buscar .nii/.nii.gz en subcarpetas
LOG_NAME = "log.csv"       # deja "" para no escribir CSV


def pick_directory(title: str) -> Path:
    """Devuelve una carpeta usando Tkinter; si no hay GUI, pide por consola."""
    if GUI_OK:
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askdirectory(title=title)
        root.destroy()
        if not path:
            raise SystemExit("No se seleccionó carpeta.")
        return Path(path)
    else:
        raw = input(f"{title}: ").strip()
        if not raw:
            raise SystemExit("No se proporcionó ruta.")
        return Path(raw).expanduser().resolve()


def main():
    print("=== Conversión NIfTI -> STL (simple) ===")
    try:
        input_dir = pick_directory("Selecciona la carpeta con NIfTI (.nii / .nii.gz)")
        if not input_dir.exists():
            raise SystemExit(f"La carpeta de entrada no existe: {input_dir}")

        output_dir = pick_directory("Selecciona la carpeta de salida para los STL")
        output_dir.mkdir(parents=True, exist_ok=True)
    except KeyboardInterrupt:
        print("\nCancelado por el usuario.")
        sys.exit(1)

    # Instanciar el convertidor con solo lo esencial
    conv = NiiToStlConverter(
        input_dir=str(input_dir),
        output_root=str(output_dir),
        label_value=None,                 # umbral automático por defecto en tu pipeline
        threshold_min=THRESH_MIN,
        threshold_max=THRESH_MAX,
        keep_ratio=KEEP_RATIO,
        laplacian_lambda=LAPL_LAMBDA,
        laplacian_iters=LAPL_ITERS,
        min_voxels=MIN_VOXELS,
        downsample_factor=DOWNSAMPLE,
        recursive=RECURSIVE,
    )

    files = conv.list_nii_files()
    if not files:
        msg = f"No se encontraron .nii/.nii.gz en {input_dir}"
        print(msg)
        if GUI_OK:
            messagebox.showwarning("Sin archivos", msg)
        sys.exit(0)

    print(f"Entradas: {len(files)} archivos")
    print(f"Salida  : {output_dir}")
    t0 = time.perf_counter()

    results = []
    ok = 0
    for i, f in enumerate(files, 1):
        name = Path(f).name
        t1 = time.perf_counter()
        r = conv.process_one_file(f)
        elapsed = time.perf_counter() - t1
        r["seconds"] = elapsed
        results.append(r)
        if r.get("success"):
            ok += 1
            print(f"[{i}/{len(files)}] {name}  OK  {elapsed:.2f}s  faces={r.get('faces_after')}")
        else:
            print(f"[{i}/{len(files)}] {name}  FAIL  {elapsed:.2f}s  motivo={r.get('message','desconocido')}")

    total = time.perf_counter() - t0

    # CSV opcional
    if LOG_NAME:
        import csv
        log_path = Path(output_dir) / LOG_NAME
        try:
            with log_path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["file","name","success","method","faces_before","faces_after","voxels_mask","stl","stl_reduced","message","seconds"])
                for r in results:
                    w.writerow([
                        r.get("file"),
                        r.get("name"),
                        r.get("success"),
                        r.get("method"),
                        r.get("faces_before"),
                        r.get("faces_after"),
                        r.get("voxels_mask"),
                        r.get("stl",""),
                        r.get("stl_reduced",""),
                        r.get("message",""),
                        f"{r.get('seconds',0.0):.3f}",
                    ])
            print(f"\nLog guardado en: {log_path}")
        except Exception as e:
            print(f"[WARN] No se pudo escribir el log CSV: {e}")

    print("\n=== Resumen ===")
    print(f"Procesados : {len(files)}")
    print(f"Éxitos     : {ok}")
    print(f"Fallos     : {len(files)-ok}")
    print(f"Tiempo tot.: {total:.2f}s  (~{total/60:.2f} min)")
    print(f"STL orig.  : {Path(output_dir) / 'originales'}")
    print(f"STL red.   : {Path(output_dir) / 'reducidos'}")

    if GUI_OK:
        messagebox.showinfo("Listo", f"Éxitos: {ok}/{len(files)}\nSalida: {output_dir}\nTiempo: {total:.1f}s")


if __name__ == "__main__":
    main()
