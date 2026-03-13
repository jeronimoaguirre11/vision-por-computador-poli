import os
import cv2
import numpy as np

from skimage.feature import hog

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# CONFIGURACIÓN

dataset_path = "Dataset500/Dataset500"   # carpeta donde están las imágenes
image_size = (160, 160) # tamaño al que se redimensionarán las imágenes antes de extraer características HOG

X = []
y = []

# CARGAR IMÁGENES

for label in os.listdir(dataset_path):

    class_folder = os.path.join(dataset_path, label)

    if not os.path.isdir(class_folder):
        continue

    for img_name in os.listdir(class_folder):

        img_path = os.path.join(class_folder, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        # convertir a gris
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # redimensionar
        img_resized = cv2.resize(img_gray, image_size)

        # HOG características

        hog_features = hog(
            img_resized,
            orientations=12,
            pixels_per_cell=(4,4),
            cells_per_block=(2,2),
            visualize=False
        )

        X.append(hog_features)
        y.append(label)

# convertir a numpy
X = np.array(X)
y = np.array(y)

print("Número de muestras:", len(X))
print("Dimensión de características:", X.shape)

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# PIPELINE

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("svc", SVC())
])

# HIPERPARÁMETROS o combinaciones a probar en el GridSearchCV

param_grid = {

    "pca__n_components":[0.90,0.95,0.98],

    "svc__kernel":[
        "linear",
        "rbf",
        "poly"
    ],

    "svc__C":[
        0.1,
        1,
        10,
        100
    ],

    "svc__gamma":[
        "scale",
        "auto"
    ]
}

# GRID SEARCH

grid = GridSearchCV(

    pipeline,
    param_grid,
    cv=3, # modificar de 3 a 5, crea las folders
    n_jobs=-1,
    verbose=2
)

print("Buscando mejores parámetros...")

grid.fit(X_train, y_train)

print("\nMejores parámetros encontrados:")
print(grid.best_params_)

# EVALUACIÓN FINAL

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy final:", accuracy)

print("\nReporte:")
print(classification_report(y_test, y_pred))