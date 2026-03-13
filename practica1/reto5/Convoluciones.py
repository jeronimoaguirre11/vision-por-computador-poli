import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage

# Cargar imagen en color (PIL devuelve RGB)
imagen_color = np.array(Image.open('jordan-imagen.jpg'))
print(imagen_color.shape)  # (altura, ancho, 3)

# cargar directamente en escala de grises (modo 'L')
imagen_gris = np.array(Image.open('jordan-imagen.jpg').convert('L'))
print(imagen_gris.shape)  # (altura, ancho)

# Función para agregar ruido salt & pepper
def salt_pepper(imagen, probabilidad=0.10):

    imagen_ruidosa = imagen.copy()
    altura, ancho = imagen.shape
    
    # Número total de píxeles a afectar
    cantidad_ruido = int(altura * ancho * probabilidad)
    
    # Seleccionar píxeles aleatorios
    for _ in range(cantidad_ruido):
        y = np.random.randint(0, altura)
        x = np.random.randint(0, ancho)
        
        # Aleatoriamente blanco (salt=255) o negro (pepper=0)
        if np.random.random() < 0.5:
            imagen_ruidosa[y, x] = 255  # Blanco (salt)
        else:
            imagen_ruidosa[y, x] = 0    # Negro (pepper)
    
    return imagen_ruidosa

# Funciones de convolución para eliminar ruido
def filtro_media(imagen, tamaño=5):
    """Filtro de media (promedio). Suaviza pero puede desenfocar."""
    return ndimage.uniform_filter(imagen.astype(float), size=tamaño).astype(np.uint8)

def filtro_gaussiano(imagen, sigma=1.0):
    """Filtro Gaussiano. Suaviza de forma natural."""
    return ndimage.gaussian_filter(imagen.astype(float), sigma=sigma).astype(np.uint8)

def filtro_mediana(imagen, tamaño=5):
    """Filtro de Mediana. Excelente para salt & pepper."""
    return ndimage.median_filter(imagen, size=tamaño).astype(np.uint8)

# Aplicar ruido salt & pepper
imagen_ruidosa = salt_pepper(imagen_gris, probabilidad=0.10)

# Aplicar todos los filtros
imagen_media = filtro_media(imagen_ruidosa, tamaño=5)
imagen_gaussiana = filtro_gaussiano(imagen_ruidosa, sigma=1.5)
imagen_mediana = filtro_mediana(imagen_ruidosa, tamaño=5)

# Mostrar todas las imágenes en una cuadrícula
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Fila 1
axes[0, 0].imshow(imagen_gris, cmap='gray')
axes[0, 0].set_title('Original (limpia)')
axes[0, 0].axis('off')

axes[0, 1].imshow(imagen_ruidosa, cmap='gray')
axes[0, 1].set_title('Con ruido Salt & Pepper')
axes[0, 1].axis('off')

axes[0, 2].imshow(imagen_media, cmap='gray')
axes[0, 2].set_title('Filtro Media (5x5)')
axes[0, 2].axis('off')

# Fila 2
axes[1, 0].imshow(imagen_gaussiana, cmap='gray')
axes[1, 0].set_title('Filtro Gaussiano (σ=1.5)')
axes[1, 0].axis('off')

axes[1, 1].imshow(imagen_mediana, cmap='gray')
axes[1, 1].set_title('Filtro Mediana (5x5)')
axes[1, 1].axis('off')

axes[1, 2].imshow(imagen_color, cmap='gray')
axes[1, 2].set_title('(Imagen a color)')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()
