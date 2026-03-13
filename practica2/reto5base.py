import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

datadir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Dataset500', 'Dataset500'))

Categories=['Cat','Dog']
flat_data_arr=[] #crea el vector unidimensional para almacenar las imágenes aplanadas
target_arr=[] #vector para almacenar las etiquetas de cada imagen

# ciclos para eliminar las imagenes que no esten acorde a las dimensiones requeridas, y para cargar las imagenes en el dataset
for i in Categories:
    print(f'loading... category: {i}')
    path=os.path.join(datadir, i)
    for img in os.listdir(path):
      img_array=imread(os.path.join(path, img))
      img_resized=resize(img_array, (150, 150, 3))
      if img_resized.flatten().shape == (67500, ):
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(Categories.index(i))
    print(f'loaded category: {i} successfully')

#convertir a numpy arrays
flat_data=np.array(flat_data_arr)
target=np.array(target_arr)

#dataframe para almacenar los datos aplanados y las etiquetas, para facilitar el manejo de los datos
df=pd.DataFrame(flat_data)
df['Target']=target
print(df.shape)

#Separamos los datos de entrada y salida, para entrenar el modelo 

#input data
x=df.iloc[:,:-1] #todas las columnas excepto la última, que es la columna de etiquetas (Target)
#output data
y=df.iloc[:,-1] #todas las filas pero solo la última columna (Target)

# dividimos los datos en entrenamiento y prueba, con un 20% de los datos para prueba, y estratificando por la variable de salida (y) para mantener la proporción de clases en ambos conjuntos
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.20, random_state=77, stratify=y)

# crear el support vector classifier
model = svm.SVC()

param_grid = {'kernel': ['rbf', 'linear', 'poly']} # distintos tipos de kernel para probar en el modelo SVM
grid = GridSearchCV(model, param_grid, cv=2, verbose=2, n_jobs=-1) 

# entrenar el modelo con los datos de entrenamiento utilizando la búsqueda en cuadrícula para encontrar los mejores hiperparámetros
grid.fit(x_train, y_train)

# Mostrar los mejores parámetros y la mejor precisión
print(f"Mejores parámetros encontrados: {grid.best_params_}")
print(f"Mejor precisión en validación cruzada: {grid.best_score_:.2f}")

# predecir las etiquetas de las imágenes de prueba utilizando el modelo entrenado
y_pred = grid.predict(x_test)

# Calcular la precisión comparando las etiquetas predichas con las etiquetas reales
accuracy = accuracy_score(y_pred, y_test)

# mostrar los resultados 
print(f"The model is {accuracy * 100}% accurate")
print(classification_report(y_test, y_pred, target_names=['cat', 'dog']))

rand_index = np.random.randint(0, len(y_test))
rand_img = x_test.iloc[rand_index].values.reshape(150, 150, 3)
rand_class = y_test.iloc[rand_index]
pred_class = y_pred[rand_index]

plt.imshow(rand_img)
plt.title(f"Predicted: {Categories[pred_class]}, Actual: {Categories[rand_class]}")
plt.show()