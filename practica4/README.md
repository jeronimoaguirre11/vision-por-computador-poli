# Práctica 4 – APRENDIZAJE PROFUNDO E IA GENERATIVA

## Descripción general

En esta práctica se desarrollaron diferentes enfoques para la **clasificación de imágenes y restauración de ruido**, utilizando técnicas de Machine Learning, Deep Learning y arquitecturas avanzadas.

El objetivo fue comprender cómo evolucionan los modelos desde métodos clásicos hasta enfoques más complejos como redes profundas y modelos basados en atención.

---

## Reto 1 – Clasificación de escenas

Se trabajó con un dataset de tres clases: **coast, forest y highway**.

Se implementaron modelos de Machine Learning tradicionales y posteriormente una red neuronal convolucional (CNN) para mejorar el desempeño. Finalmente, se utilizó una arquitectura no secuencial basada en **ResNet** mediante transferencia de aprendizaje.

---

## Reto 2 – Clasificación LandUse

Se utilizó un dataset con tres clases: **airplane, buildings y tenniscourt**.

Se reutilizaron los enfoques de Deep Learning y arquitecturas avanzadas, permitiendo analizar el comportamiento del modelo con un conjunto de datos más amplio y variado.

---

## Reto 3 – Eliminación de ruido speckle (SAR)

Se trabajó con pares de imágenes (ruido y ground truth) para aplicar técnicas de restauración.

Se implementó un **autoencoder** como modelo base y posteriormente una arquitectura **U-Net**, la cual mejora la reconstrucción al conservar mejor la información espacial mediante conexiones internas.

Se evaluaron los resultados utilizando métricas como PSNR, SSIM y ENL, además de inspección visual.

---

## Reto 4 – Restauración con SwinIR

Se utilizó un modelo preentrenado basado en Transformers (**SwinIR**) para restauración de imágenes.

A diferencia de los modelos anteriores, este no requirió entrenamiento, sino que se aplicó directamente sobre imágenes con ruido para obtener versiones restauradas.

Los resultados fueron evaluados mediante métricas y análisis visual.

---

## Conclusiones

- Los modelos de Machine Learning presentan limitaciones frente a problemas de visión más complejos.
- Las redes neuronales convolucionales permiten mejorar el desempeño al aprender características espaciales.
- Arquitecturas avanzadas como ResNet facilitan el aprendizaje mediante modelos preentrenados.
- U-Net mejora la restauración de imágenes al conservar mejor los detalles.
- Modelos basados en Transformers como SwinIR ofrecen resultados más robustos en tareas de reconstrucción sin necesidad de entrenamiento adicional.

---

## Nota

Los datasets utilizados no se incluyen en este repositorio debido a su tamaño.
Es necesario configurar las rutas de acceso a los datos de forma local.

---
