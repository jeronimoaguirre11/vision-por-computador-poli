import cv2
import os
import numpy as np
from scipy.ndimage import uniform_filter

# =========================
# RUTAS
# =========================
base_path = 'C:/Users/jeron/vision por computadora/practica3/Dataset-jero/'

raw_path = base_path + 'Imagen-Original/'
noisy_path = base_path + 'Imagen-Ruido/'
clean_path = base_path + 'Imagen-Limpia/'

os.makedirs(noisy_path, exist_ok=True)
os.makedirs(clean_path, exist_ok=True)

# =========================
# FILTRO LEE
# =========================
def lee_filter(img, size=7):
    img = img.astype(np.float32)

    mean = uniform_filter(img, size)
    mean_sq = uniform_filter(img**2, size)
    variance = mean_sq - mean**2

    overall_variance = np.var(img)

    weights = variance / (variance + overall_variance + 1e-8)
    img_filtered = mean + weights * (img - mean)

    return img_filtered

# =========================
# GENERAR PATCHES
# =========================
def generar_patches(noisy, clean, size=512, stride=256, start_index=0):

    h, w = noisy.shape
    count = start_index

    for i in range(0, h - size, stride):
        for j in range(0, w - size, stride):

            patch_noisy = noisy[i:i+size, j:j+size]
            patch_clean = clean[i:i+size, j:j+size]

            cv2.imwrite(noisy_path + f'img_{count}.png', patch_noisy)
            cv2.imwrite(clean_path + f'img_{count}.png', patch_clean)

            count += 1

    return count

# =========================
# LEER IMÁGENES
# =========================
image_files = [f for f in os.listdir(raw_path) if f.endswith('.png')]

total_patches = 0

for file in image_files:

    path = raw_path + file
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Error leyendo:", file)
        continue

    print(f"Procesando: {file}")

    # =========================
    # NORMALIZAR
    # =========================
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img_norm = img_norm.astype(np.uint8)

    # =========================
    # FILTRAR (LEE)
    # =========================
    img_clean = lee_filter(img_norm, size=7)

    # Normalizar salida del filtro
    img_clean = cv2.normalize(img_clean, None, 0, 255, cv2.NORM_MINMAX)
    img_clean = img_clean.astype(np.uint8)

    # =========================
    # GENERAR PATCHES
    # =========================
    total_patches = generar_patches(
        img_norm,
        img_clean,
        size=512,
        stride=256,
        start_index=total_patches
    )

print(f"Total patches generados: {total_patches}")