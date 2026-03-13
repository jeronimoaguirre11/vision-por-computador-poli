# Practica 2

Esta practica se enfoco en como leer imagenes, clasificarlas y en aplicar varios modelos de prediccion 

## Estructura y Definicion de los archivos

- Reto 5 base  →  Modelo base, para entender el ejercicio, comparar y aplicar nuevos modelos que mejoren el porcentaje de predicción. en este reto base contamos con un vector unidimensional que almacena los pixeles de cada imagen, contamos con un svc de 3 kernels y c = 2, (folds)

- Reto 5 pca → Es el primer cambio desde el modelo base, en este añadimos un nuevo modelo llamado PCA, que nos ayuda a  redimensionar o tomar trozos de ese vector unidimensional antes creado. también contamos con un pipeline, el cual encadena modelos en un orden (svc, pca , etc)

- Reto 5 hog → En este modelo ya no usamos el vector unidimensional como en los modelos previos, definimos un tamaño previo a las imágenes, lo interesante de este modelo es que al cargar las imagenes las transformamos en escala de grises, para no toparnos con irregularidades con los píxeles R,G,B u otros, si no que encasilla más fácil los contornos por las luces

- Reto 6 → Lee un Dataset con 3 tipos de imagenes distintas (costas, bosques, carreteras), e implementamos 7 modelos para ver cual de todos predice mejor y que tan acertado es con respecto a la imagen real. 

## tecnologias

- Python
- numpy
- os
- pandas
- matplotlib
- sklearn
- skimage
- cv2
