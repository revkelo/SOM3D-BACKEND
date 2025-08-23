import zipfile
import os
import time
from totalsegmentator.python_api import totalsegmentator

# Rutas
zip_path = "D:/PC/SOM3D/Dicom-a-STL-main/Dicom-a-STL-main/Dev/ct-headnii.zip"
extract_folder = "D:/PC/SOM3D/Dicom-a-STL-main/Dicom-a-STL-main/Dev/ct-headnii_extracted"
output_path = "D:/PC/SOM3D/Dicom-a-STL-main/Dicom-a-STL-main/Dev/ct-headnii_segmented"

if __name__ == "__main__":
    # 🕒 Contador general
    start_total = time.time()

    print("⏳ Iniciando proceso...")

    # 1️⃣ Descomprimir ZIP
    os.makedirs(extract_folder, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    print(f"📂 DICOMs extraídos en: {extract_folder}")

    # 2️⃣ Crear carpeta de salida
    os.makedirs(output_path, exist_ok=True)

    # 3️⃣ Ejecutar segmentación en modo rápido
    print("🔎 Iniciando segmentación...")
    start_seg = time.time()

    totalsegmentator(extract_folder, output_path, fast=True)

    end_seg = time.time()
    print(f"✅ Segmentación completada en {end_seg - start_seg:.2f} segundos.")
    print(f"📁 Resultados en: {output_path}")

    # 🕒 Tiempo total
    end_total = time.time()
    print(f"🏁 Proceso total finalizado en {end_total - start_total:.2f} segundos.")
