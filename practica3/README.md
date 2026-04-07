# Proyecto de Visión por Computador – Análisis de Imágenes SAR

Este proyecto aborda el procesamiento y análisis de imágenes satelitales SAR (Synthetic Aperture Radar), enfocado en la reducción de ruido speckle, clasificación no supervisada y generación de datasets para tareas de aprendizaje automático.

---

# Reto 1: Imágenes y Filtrado

Se seleccionó una región geográfica con presencia de cuerpos de agua y se descargaron múltiples imágenes SAR en diferentes fechas.

## Procesos realizados:
- Re-escalado de imágenes para visualización
- Aplicación de filtro de speckle (Filtro de Lee)
- Análisis visual de mejoras

## Conclusión:
El filtrado permitió reducir significativamente el ruido speckle, mejorando la identificación de estructuras en la imagen.

---

# Reto 2: Clasificación No Supervisada

Se aplicó clustering utilizando K-Means sobre:
- Imagen original
- Imagen filtrada

## Procesos realizados:
- Agrupamiento en 2, 3 y 4 clases
- Re-escalado a escala de grises (0–255)

## Conclusión:
La imagen filtrada permitió una mejor separación de clases, reduciendo la influencia del ruido en la segmentación.

---

# Reto 3: Clasificación Agua / No Agua

Se identificó la clase correspondiente al agua a partir del clustering.

## Procesos realizados:
- Selección de la clase de menor intensidad
- Aplicación de operaciones morfológicas
- Filtrado por tamaño de regiones
- Cálculo del porcentaje de agua

## Conclusión:
Se evidenció que el ruido puede generar falsas detecciones, por lo que fue necesario aplicar técnicas de postprocesamiento para mejorar la precisión.

---

# Reto 4: Creación del Dataset

Se generó un dataset para reducción de ruido speckle.

## Procesos realizados:
- Selección de imágenes SAR no filtradas
- Registro de imágenes respecto a una base
- Fusión mediante mediana
- Generación de patches de 512x512
- Creación de pares:
  - Imagen con ruido (noisy)
  - Imagen limpia (clean)

## Estructura del dataset:

- Imagen-Original/
- Imagen-Ruido/
- Imagen-Limpia/


## Conclusión:
La fusión de múltiples adquisiciones permitió reducir el ruido speckle y generar un dataset útil para tareas de aprendizaje automático.

---

# Tecnologías utilizadas

- Python
- OpenCV
- NumPy
- Scikit-learn

---

# Nota

Las imágenes no se incluyen en el repositorio debido a su gran tamaño. Solo se proporciona el código necesario para reproducir el proceso.


