import cv2
import os
import numpy as np

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
# LEER IMÁGENES
# =========================
image_files = [f for f in os.listdir(raw_path) if f.endswith('.png')]

imagenes = []

for file in image_files:
    path = raw_path + file
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # REDUCIR TAMAÑO
    scale = 0.3   # puedes probar 0.5 también
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    if img is None:
        print("Error leyendo:", file)
        continue

    imagenes.append(img)

print(f"Total imágenes cargadas: {len(imagenes)}")

# =========================
# SELECCIONAR BASE
# =========================
base = imagenes[0]
base = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX)
base = base.astype(np.uint8)

# =========================
# REGISTRO
# =========================
def registrar(base, img):
    base = base.astype(np.float32)
    img = img.astype(np.float32)

    warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)

    try:
        _, warp_matrix = cv2.findTransformECC(
            base, img, warp_matrix, cv2.MOTION_TRANSLATION, criteria
        )

        aligned = cv2.warpAffine(
            img, warp_matrix, (base.shape[1], base.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )

        return aligned

    except:
        print("Error en registro, usando imagen original")
        return img


# =========================
# REGISTRAR TODAS
# =========================
registradas = []

for img in imagenes:
    aligned = registrar(base, img)
    registradas.append(aligned)

# =========================
# FUSIÓN (MEDIANA)
# =========================
fusion = np.median(np.array(registradas), axis=0)
#fusion = np.log1p(fusion)
fusion = cv2.normalize(fusion, None, 0, 255, cv2.NORM_MINMAX)
#fusion = cv2.convertScaleAbs(fusion, alpha=1.5, beta=20)
fusion = fusion.astype(np.uint8)

print("Fusión completada")

# =========================
# GENERAR PATCHES 512x512
# =========================
def generar_patches(noisy, clean, size=512, stride=512):

    h, w = noisy.shape
    count = 0

    for i in range(0, h - size, stride):
        for j in range(0, w - size, stride):

            patch_noisy = noisy[i:i+size, j:j+size]
            patch_clean = clean[i:i+size, j:j+size]

            cv2.imwrite(noisy_path + f'img_{count}.png', patch_noisy)
            cv2.imwrite(clean_path + f'img_{count}.png', patch_clean)

            count += 1

    print(f"Total patches generados: {count}")


# =========================
# EJECUCIÓN FINAL
# =========================
generar_patches(base, fusion)