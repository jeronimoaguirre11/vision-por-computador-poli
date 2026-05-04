import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# ==============================
# 1. CONFIGURACIÓN
# ==============================
IMG_SIZE = 128
base_path = "C:/Users/jeron/vision por computadora/practica4/Landuse"
subfolders = ['airplane', 'buildings', 'tenniscourt']

# ==============================
# 2. CARGA DE DATOS
# ==============================
def load_data(base_path, subfolders):
    data = []
    labels = []
    paths = []

    for label, subfolder in enumerate(subfolders):
        folder_path = os.path.join(base_path, subfolder)

        for file in os.listdir(folder_path):
            if file.endswith('.png'):
                img_path = os.path.join(folder_path, file)
                img = cv2.imread(img_path)

                if img is None:
                    continue

                paths.append(img_path)

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0

                data.append(img)
                labels.append(label)

    return np.array(data), np.array(labels), paths

X, y, paths = load_data(base_path, subfolders)

print("Datos cargados:", X.shape, y.shape)

trainX, testX, trainY, testY, trainPaths, testPaths = train_test_split(
    X, y, paths, test_size=0.25, random_state=42
)

# ==============================
# 3. VISUALIZACIÓN (comparación)
# ==============================
idx = random.randint(0, len(testX) - 1)

img_original = cv2.imread(testPaths[idx])
img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

img_resized = cv2.resize(img_original, (IMG_SIZE, IMG_SIZE))

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(img_original)
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(img_resized)
plt.title("Redimensionada")

plt.show()

# ==============================
# 4. MODELO CNN
# ==============================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Entrenando modelo...")
history = model.fit(trainX, trainY, epochs=10, validation_data=(testX, testY))

# ==============================
# 5. EVALUACIÓN
# ==============================
test_loss, test_acc = model.evaluate(testX, testY)
print(f"\nAccuracy final en test: {test_acc*100:.2f}%")

# Gráfica
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend()
plt.title("Evolución del accuracy")
plt.show()

# ==============================
# 6. PREDICCIÓN VISUAL
# ==============================

img = testX[idx]
true_label = testY[idx]

pred = model.predict(np.expand_dims(img, axis=0))
pred_label = np.argmax(pred)

class_names = subfolders

# Mostrar imagen ORIGINAL (más profesional)
img_original = cv2.imread(testPaths[idx])
img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

plt.imshow(img_original)
plt.title(f"Real: {class_names[true_label]} | Pred: {class_names[pred_label]}")
plt.axis('off')
plt.show()