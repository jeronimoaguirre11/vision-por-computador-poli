from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import os

base_path = "C:/Users/jeron/vision por computadora/practica4/Landuse/"
subfolders = ['airplane', 'buildings', 'tenniscourt']

paths = []
data = []
labels = []

for subfolder in subfolders:
    current_folder_path = os.path.join(base_path, subfolder)
    files = [f for f in os.listdir(current_folder_path) if f.endswith('.png')]

    for file in files:

        path = os.path.join(current_folder_path, file)
        paths.append(path)

        img = cv2.imread(path)

        B, G, R = cv2.split(img)

        features = [
            np.mean(R), np.mean(G), np.mean(B),
            np.std(R), np.std(G), np.std(B)
        ]

        data.append(features)
        labels.append(subfolder)

le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(
    np.array(data),
    np.array(labels),
    test_size=0.25,
    random_state=42
)

models = {

    "KNN": KNeighborsClassifier(),

    "Naive Bayes": GaussianNB(),

    "Logistic Regression": LogisticRegression(max_iter=1000),

    "SVM": SVC(),

    "Decision Tree": DecisionTreeClassifier(),

    "Random Forest": RandomForestClassifier(),

    "MLP": MLPClassifier(max_iter=1000)

}

results = {}

for name, model in models.items():
    print("Entrenando:", name)
    model.fit(trainX, trainY)
    predictions = model.predict(testX)
    acc = accuracy_score(testY, predictions)
    results[name] = acc

    # Aseguramos que las etiquetas sean strings para evitar el TypeError
    target_names = [str(c) for c in le.classes_]
    print(classification_report(testY, predictions, target_names=target_names))

    print(f"Accuracy: {acc*100:.2f}%")
    print("--------------")

print("\nComparación de modelos\n")

for model, acc in results.items():
    print(f"{model}: {acc*100:.2f}%")

# 1. Seleccionar un índice aleatorio
rand_idx = random.randint(0, len(paths) - 1)
random_image_path = paths[rand_idx]

# 2. Obtener la etiqueta real (texto)
label_raw = labels[rand_idx]
if isinstance(label_raw, (str, np.str_)):
    true_label_text = str(label_raw)
else:
    true_label_text = str(le.inverse_transform([label_raw])[0])

# 3. Cargar imagen y extraer características
img = cv2.imread(random_image_path)
B, G, R = cv2.split(img)
img_features = [np.mean(R), np.mean(G), np.mean(B), np.std(R), np.std(G), np.std(B)]

# 4. Generar comparativa con todos los modelos
print(f"Ruta de imagen: {random_image_path}")
print(f"--- CLASE REAL: {true_label_text.upper()} ---\n")
print(f"{'Modelo':<20} | {'Predicción':<15} | {'Accuracy General'}")
print("-" * 55)

for name, model in models.items():
    # Predicción individual
    pred_num = model.predict([img_features])
    pred_text = str(le.inverse_transform(pred_num)[0])

    # Obtener accuracy del diccionario de resultados previo
    acc_val = results.get(name, 0) * 100

    # Mostrar fila de la tabla
    print(f"{name:<20} | {pred_text:<15} | {acc_val:>6.2f}%")

# 5. Mostrar la imagen

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.title("Imagen seleccionada")
plt.axis("off")
plt.show()
print('\nComparación completada con éxito.')
