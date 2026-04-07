import cv2
import os
import numpy as np
from sklearn.cluster import KMeans

# =========================
# FUNCIÓN KMEANS
# =========================

def aplicar_kmeans(img, k=3, sample_size=100000):
    pixels = img.reshape(-1, 1)

    # Muestreo para acelerar
    if len(pixels) > sample_size:
        idx = np.random.choice(len(pixels), sample_size, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels

    # Entrenar con muestra
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(sample)

    # Predecir toda la imagen
    labels = kmeans.predict(pixels)

    clustered = labels.reshape(img.shape)

    # Escalar a 0–255
    clustered = (clustered / (k - 1)) * 255
    clustered = clustered.astype(np.uint8)

    return clustered

# =========================
# RUTAS
# =========================
input_folder = 'C:/Users/jeron/vision por computadora/practica3/img-jero/'
output_folder = 'C:/Users/jeron/vision por computadora/practica3/kmeans-jero/'

os.makedirs(output_folder, exist_ok=True)

# =========================
# PROCESAMIENTO
# =========================
for file in os.listdir(input_folder):

    if file.endswith('.png'):

        path = input_folder + file
        print(f"Procesando: {file}")

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print("Error leyendo:", file)
            continue

        # =========================
        # KMeans con diferentes clases
        # =========================
        for k in [2, 3, 4]:

            clustered = aplicar_kmeans(img, k=k)

            name = file.replace('.png', '')
            output_path = output_folder + f"{name}_k{k}.png"

            cv2.imwrite(output_path, clustered)

            print(f"Guardado: {output_path}")