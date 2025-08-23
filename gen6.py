import os
import pydicom
import numpy as np
from skimage import measure, morphology
from scipy import ndimage
from stl import mesh
import trimesh  # Nueva librería para reducción de malla

from trimesh.smoothing import filter_laplacian
# Ruta a la carpeta DICOM
folder_path = "D:/DESARROLLO/Dicom v1/ct-2"

# Definir el rango de umbral (Threshold Range)
threshold_min = 150  # Cambia este valor según sea necesario
threshold_max = 3075  # Valor máximo para ajustar rango

# Cargar imágenes DICOM
dicom_files = [pydicom.dcmread(os.path.join(folder_path, f)) for f in os.listdir(folder_path) if f.endswith('.dcm')]
dicom_files.sort(key=lambda x: float(x.ImagePositionPatient[2]) if 'ImagePositionPatient' in x else 0)

# Convertir imágenes en una matriz 3D
slices = np.stack([s.pixel_array for s in dicom_files])

# Escalado de la imagen con valores reales
rescale_slope = float(getattr(dicom_files[0], 'RescaleSlope', 1))
rescale_intercept = float(getattr(dicom_files[0], 'RescaleIntercept', 0))
slices = slices * rescale_slope + rescale_intercept

# Obtener el espaciado real (en mm)
pixel_spacing = dicom_files[0].PixelSpacing  # Espaciado en XY
slice_thickness = float(getattr(dicom_files[0], 'SliceThickness', 1))  # Espaciado en Z
spacing = (slice_thickness, pixel_spacing[0], pixel_spacing[1])

# Aplicar umbral para extraer estructuras dentro del rango
binary_volume = np.logical_and(slices > threshold_min, slices < threshold_max)

# Eliminar componentes pequeños
binary_volume = morphology.remove_small_objects(binary_volume, min_size=1000)

# Suavizado
smoothed_volume = ndimage.gaussian_filter(binary_volume.astype(float), sigma=1)

# Generar la malla con Marching Cubes considerando el espaciado real
verts, faces, normals, _ = measure.marching_cubes(smoothed_volume, level=0.5, spacing=spacing)

# Centrar la malla automáticamente
centroid = np.mean(verts, axis=0)
centered_verts = verts - centroid


rotation_matrix = np.array([
    [np.cos(np.radians(90)), 0, np.sin(np.radians(90))],
    [0, 1, 0],
    [-np.sin(np.radians(90)), 0, np.cos(np.radians(90))]
])

# Aplicar rotación a todos los vértices
rotated_verts = np.dot(verts, rotation_matrix)

# Crear el archivo STL rotado
stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        stl_mesh.vectors[i][j] = rotated_verts[f[j], :]

# Guardar STL con proporciones correctas y rotación aplicada
output_path = "D:/DESARROLLO/Dicom v1/modelo.stl"
stl_mesh.save(output_path)
print(f"Modelo STL rotado -90° en Y generado con Threshold [{threshold_min}, {threshold_max}] en: {output_path}")



# Reducir la malla usando trimesh
original_mesh = trimesh.load_mesh(output_path)
reduction_factor = 0.9  # Reducir al 90% de los vértices originales
reduced_mesh = original_mesh.simplify_quadric_decimation(reduction_factor)

# Aplicar suavizado Laplaciano
smoothed_reduced_mesh = reduced_mesh.copy()
filter_laplacian(smoothed_reduced_mesh, lamb=0.5, iterations=10)  # lamb controla la intensidad del suavizado

# Guardar STL reducido y suavizado
smoothed_reduced_output_path = "D:/DESARROLLO/Dicom v1/modelo-reducido.stl"
smoothed_reduced_mesh.export(smoothed_reduced_output_path)
print(f"Modelo STL reducido y suavizado generado en: {smoothed_reduced_output_path}")
