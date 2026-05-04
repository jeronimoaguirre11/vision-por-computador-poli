import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim

# ==============================
# 1. CONFIGURACIÓN
# ==============================
IMG_SIZE = 128
BATCH_SIZE = 4   # clave para memoria
EPOCHS = 25

noisy_path = "C:/Users/jeron/vision por computadora/practica4/Dataset-SAR/SAR-RUIDO"
clean_path = "C:/Users/jeron/vision por computadora/practica4/Dataset-SAR/GROUND-TRUTH"

# ==============================
# 2. CARGA DE DATOS
# ==============================
def load_pairs(noisy_path, clean_path):
    noisy_images = []
    clean_images = []

    files = sorted(os.listdir(noisy_path))

    for file in files:
        noisy_img = cv2.imread(os.path.join(noisy_path, file), cv2.IMREAD_GRAYSCALE)
        clean_img = cv2.imread(os.path.join(clean_path, file), cv2.IMREAD_GRAYSCALE)

        if noisy_img is None or clean_img is None:
            continue

        noisy_img = cv2.resize(noisy_img, (IMG_SIZE, IMG_SIZE)) / 255.0
        clean_img = cv2.resize(clean_img, (IMG_SIZE, IMG_SIZE)) / 255.0

        noisy_images.append(noisy_img)
        clean_images.append(clean_img)

    noisy_images = np.array(noisy_images)[..., np.newaxis]
    clean_images = np.array(clean_images)[..., np.newaxis]

    return noisy_images, clean_images

X_noisy, X_clean = load_pairs(noisy_path, clean_path)

print("Datos:", X_noisy.shape)

trainX, testX, trainY, testY = train_test_split(
    X_noisy, X_clean, test_size=0.2, random_state=42
)

# ==============================
# 3. DATASET (optimizado memoria)
# ==============================
train_dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY)) \
    .shuffle(100) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((testX, testY)) \
    .batch(BATCH_SIZE)

# ==============================
# 4. AUTOENCODER BASE
# ==============================
def build_autoencoder():
    input_img = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    # Encoder
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D(2, padding='same')(x)

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2, padding='same')(x)

    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2, padding='same')(x)

    # Decoder
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)

    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)

    output = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)

    return models.Model(input_img, output)

model = build_autoencoder()
model.compile(optimizer='adam', loss='mse')

model.summary()

# ==============================
# 5. ENTRENAMIENTO
# ==============================
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS
)

# ==============================
# 6. MÉTRICAS
# ==============================
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 20 * np.log10(1.0 / np.sqrt(mse + 1e-8))

def enl(image):
    mean = np.mean(image)
    std = np.std(image)
    return (mean**2) / (std**2 + 1e-8)

psnr_list, ssim_list, enl_list = [], [], []

preds = model.predict(testX)

for i in range(len(testX)):
    gt = testY[i].squeeze()
    pred = preds[i].squeeze()

    psnr_list.append(psnr(gt, pred))
    ssim_list.append(ssim(gt, pred, data_range=1.0))
    enl_list.append(enl(pred))

print("\n--- RESULTADOS ---")
print(f"PSNR promedio: {np.mean(psnr_list):.2f}")
print(f"SSIM promedio: {np.mean(ssim_list):.4f}")
print(f"ENL promedio: {np.mean(enl_list):.2f}")

# ==============================
# 7. VISUALIZACIÓN
# ==============================
idx = 0

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Imagen con ruido (SAR)")
plt.imshow(testX[idx].squeeze(), cmap='gray')

plt.subplot(1,3,2)
plt.title("Imagen filtrada (Autoencoder)")
plt.imshow(preds[idx].squeeze(), cmap='gray')

plt.subplot(1,3,3)
plt.title("Imagen original (Ground Truth)")
plt.imshow(testY[idx].squeeze(), cmap='gray')

plt.show()

# ==============================
# 8. GRÁFICA
# ==============================
plt.plot(history.history['loss'], label='Error entrenamiento')
plt.plot(history.history['val_loss'], label='Error validación')

plt.legend()
plt.title("Evolución del error del modelo")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()