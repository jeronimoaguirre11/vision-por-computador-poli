import cv2
import os
import numpy as np

# =========================
# RUTAS
# =========================
input_folder = 'C:/Users/jeron/vision por computadora/practica3/kmeans-jero/'
output_folder = 'C:/Users/jeron/vision por computadora/practica3/agua-jero/'

os.makedirs(output_folder, exist_ok=True)

# =========================
# FUNCIÓN PARA EXTRAER AGUA
# =========================
def extraer_agua(img):

    # Reducir tamaño para acelerar procesamiento
    scale = 0.25  # puedes probar 0.5 también
    img = cv2.resize(img, (0,0), fx=scale, fy=scale)

    # 1. Obtener clases
    valores = np.unique(img)
    print("Clases detectadas:", valores)

    # 2. Elegir clase más oscura (agua)
    agua_valor = valores[0]

    # 3. Crear máscara inicial
    mask = np.where(img == agua_valor, 255, 0).astype(np.uint8)

    # =========================
    # 4. LIMPIEZA MORFOLÓGICA
    # =========================
    kernel = np.ones((5,5), np.uint8)

    # eliminar ruido pequeño
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # cerrar huecos
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # =========================
    # 5. FILTRAR POR TAMAÑO
    # =========================
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    min_area = 5000  # AJUSTABLE según tu imagen

    mask_filtrada = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area > min_area:
            mask_filtrada[labels == i] = 255

    mask = mask_filtrada

    # =========================
    # 6. CALCULAR PORCENTAJE
    # =========================
    total = mask.size
    agua = np.sum(mask == 255)

    porcentaje = (agua / total) * 100

    return mask, porcentaje


# =========================
# PROCESAMIENTO
# =========================
resultados = {}

for file in os.listdir(input_folder):

    if file.endswith('_k3.png'):

        path = input_folder + file
        print(f"\nProcesando: {file}")

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print("Error leyendo:", file)
            continue

        # EXTRAER AGUA
        mask, porcentaje = extraer_agua(img)

        print(f"Agua detectada: {porcentaje:.2f}%")

        # GUARDAR IMAGEN
        name = file.replace('.png', '')
        output_path = output_folder + name + '_agua.png'

        cv2.imwrite(output_path, mask)

        # GUARDAR RESULTADOS PARA COMPARACIÓN
        base = name.replace('_lee', '').replace('_original', '')

        if base not in resultados:
            resultados[base] = {}

        if 'lee' in name:
            resultados[base]['lee'] = porcentaje
        else:
            resultados[base]['original'] = porcentaje


# =========================
# COMPARACIÓN FINAL
# =========================
print("\n===== COMPARACIÓN FINAL =====")

for key in resultados:

    orig = resultados[key].get('original', None)
    lee = resultados[key].get('lee', None)

    print(f"\nImagen: {key}")

    if orig is not None:
        print(f"Original: {orig:.2f}%")

    if lee is not None:
        print(f"Lee: {lee:.2f}%")

    if orig is not None and lee is not None:
        diferencia = abs(orig - lee)
        print(f"Diferencia: {diferencia:.2f}%")