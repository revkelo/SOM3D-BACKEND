import zipfile
import os
import time
import tempfile
import shutil
from totalsegmentator.python_api import totalsegmentator


class DicomSegmenter:
    def __init__(self, zip_path, output_path):
        self.zip_path = zip_path
        self.output_path = output_path

    def run_pipeline(self, fast=True):
        """Ejecuta todo el pipeline: unzip temporal + segmentación + limpiar"""
        start_total = time.time()
        print("⏳ Iniciando proceso...")

        # Crear carpeta temporal
        temp_dir = tempfile.mkdtemp()
        try:
            # 1️⃣ Extraer ZIP en carpeta temporal
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            print(f"📂 Archivos extraídos temporalmente en: {temp_dir}")

            # 2️⃣ Ejecutar segmentación
            os.makedirs(self.output_path, exist_ok=True)
            print("🔎 Iniciando segmentación...")
            start_seg = time.time()

            totalsegmentator(temp_dir, self.output_path, fast=fast)

            end_seg = time.time()
            print(f"✅ Segmentación completada en {end_seg - start_seg:.2f} segundos.")
            print(f"📁 Resultados en: {self.output_path}")

        finally:
            # 3️⃣ Eliminar carpeta temporal
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("🧹 Carpeta temporal eliminada.")

        end_total = time.time()
        print(f"🏁 Proceso total finalizado en {end_total - start_total:.2f} segundos.")


# Ejemplo de uso
if __name__ == "__main__":
    zip_path = "Y:/DICOM/ct-headnii.zip"
    output_path = "C:/Users/K/Documents/GitHub/SOM3D-BACKEND/otros/ct-headnii_segmented"

    segmenter = DicomSegmenter(zip_path, output_path)
    segmenter.run_pipeline(fast=True)
