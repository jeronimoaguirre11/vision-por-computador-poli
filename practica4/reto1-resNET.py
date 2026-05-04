import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# ==============================
# 1. CONFIGURACIÓN
# ==============================
IMG_SIZE = 224
base_path = "C:/Users/jeron/vision por computadora/practica2/DataEscenas"
subfolders = ['coast', 'forest', 'highway']

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
            if file.endswith('.jpg'):
                img_path = os.path.join(folder_path, file)
                img = cv2.imread(img_path)

                if img is None:
                    continue

                paths.append(img_path)

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                # IMPORTANTE: usar preprocess_input
                img = preprocess_input(img)

                data.append(img)
                labels.append(label)

    return np.array(data), np.array(labels), paths

X, y, paths = load_data(base_path, subfolders)

print("Datos cargados:", X.shape, y.shape)

trainX, testX, trainY, testY, trainPaths, testPaths = train_test_split(
    X, y, paths, test_size=0.25, random_state=42
)

# ==============================
# 3. MODELO RESNET
# ==============================
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False  # transfer learning

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
output = layers.Dense(3, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================
# 4. ENTRENAMIENTO
# ==============================
print("Entrenando ResNet...")
history = model.fit(trainX, trainY, epochs=10, validation_data=(testX, testY))

# ==============================
# 5. EVALUACIÓN
# ==============================
test_loss, test_acc = model.evaluate(testX, testY)
print(f"\nAccuracy final (ResNet): {test_acc*100:.2f}%")

# ==============================
# 6. GRÁFICA
# ==============================
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend()
plt.title("ResNet Accuracy")
plt.show()

# ==============================
# 7. PREDICCIÓN VISUAL
# ==============================
idx = random.randint(0, len(testX) - 1)

img_original = cv2.imread(testPaths[idx])
img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

img_resized = cv2.resize(img_original, (IMG_SIZE, IMG_SIZE))
img_input = preprocess_input(img_resized)

pred = model.predict(np.expand_dims(img_input, axis=0))
pred_label = np.argmax(pred)

class_names = subfolders

plt.imshow(img_original)
plt.title(f"Real: {class_names[testY[idx]]} | Pred: {class_names[pred_label]}")
plt.axis('off')
plt.show()