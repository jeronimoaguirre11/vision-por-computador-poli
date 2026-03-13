import random
import math
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


def crear_matriz_aleatoria(filas = 1000, columnas = 1000, min_val=0, max_val=255):

    matriz = []
    for i in range(filas):
        fila = []
        for j in range(columnas):
            fila.append(random.randint(min_val, max_val))
        matriz.append(fila)
    return matriz


def calcular_estadisticas(matriz):
   
    # Aplanar la matriz en una lista unidimensional
    valores = []
    for fila in matriz:
        for valor in fila:
            valores.append(valor)
    
    # Calcular mínimo y máximo
    minimo = valores[0]
    maximo = valores[0]
    for valor in valores:
        if valor < minimo:
            minimo = valor
        if valor > maximo:
            maximo = valor
    
    # Calcular media (promedio)
    suma = 0
    for valor in valores:
        suma += valor
    media = suma / len(valores)
    
    # Calcular desviación estándar
    suma_cuadrados_diferencias = 0
    for valor in valores:
        diferencia = valor - media
        suma_cuadrados_diferencias += diferencia * diferencia
    
    varianza = suma_cuadrados_diferencias / len(valores)
    desv_estandar = math.sqrt(varianza)
    
    return {
        'minimo': minimo,
        'maximo': maximo,
        'media': media,
        'desv_estandar': desv_estandar
    }


# Programa principal
if __name__ == "__main__":
    print("Creando matriz de 1000x1000 con números aleatorios...")
    matriz = crear_matriz_aleatoria()
    
    print("Calculando estadísticas...")
    stats = calcular_estadisticas(matriz)
    
    print("\n" + "="*50)
    print("ESTADÍSTICAS DE LA MATRIZ 1000x1000")
    print("="*50)
    print(f"Valor mínimo:      {stats['minimo']}")
    print(f"Valor máximo:      {stats['maximo']}")
    print(f"Media:             {stats['media']:.2f}")
    print(f"Desviación estándar: {stats['desv_estandar']:.2f}")
    print("="*50)
    
    # Crear interfaz gráfica con tkinter
    print("\nAbriendo interfaz gráfica...")
    ventana = tk.Tk()
    ventana.title("Visualizador de Matriz - Análisis de Imágenes")
    ventana.geometry("1200x750")
    
    # Frame para la imagen
    frame_imagen = ttk.Frame(ventana)
    frame_imagen.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
    
    label_titulo_imagen = ttk.Label(frame_imagen, text="Matriz Visualizada (1000x1000)", 
                                     font=("Arial", 12, "bold"))
    label_titulo_imagen.pack(pady=5)
    
    # Crear imagen a partir de la matriz (escala de grises)
    # Reducir tamaño para visualización (500x500)
    tamaño_visualizacion = 500
    tamaño_bloque = 1000 // tamaño_visualizacion  # 2 píxeles por bloque
    
    imagen_datos = []
    for i in range(0, 1000, tamaño_bloque):
        fila_imagen = []
        for j in range(0, 1000, tamaño_bloque):
            fila_imagen.append(matriz[i][j])
        imagen_datos.append(fila_imagen)
    
    # Convertir lista a imagen PIL
    imagen_pil = Image.new('L', (tamaño_visualizacion, tamaño_visualizacion))
    pixels = imagen_pil.load()
    for i in range(tamaño_visualizacion):
        for j in range(tamaño_visualizacion):
            pixels[j, i] = imagen_datos[i][j]
    
    # Escalar imagen para mejor visualización
    imagen_pil = imagen_pil.resize((500, 500), Image.Resampling.NEAREST)
    imagen_tk = ImageTk.PhotoImage(imagen_pil)
    
    label_imagen = ttk.Label(frame_imagen, image=imagen_tk)
    label_imagen.image = imagen_tk  # Mantener referencia
    label_imagen.pack()
    
    # Frame para las estadísticas
    frame_stats = ttk.Frame(ventana, width=300)
    frame_stats.pack(side=tk.RIGHT, padx=15, pady=15, fill=tk.BOTH)
    
    label_titulo_stats = ttk.Label(frame_stats, text="ESTADÍSTICAS", 
                                   font=("Arial", 14, "bold"))
    label_titulo_stats.pack(pady=10)
    
    # Crear tabla de estadísticas
    stats_frame = ttk.Frame(frame_stats)
    stats_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Estadísticas
    etiquetas = [
        ("Dimensiones:", f"1000 × 1000"),
        ("Total píxeles:", f"{1000 * 1000:,}"),
        ("", ""),  # Separador
        ("Valor Mínimo:", f"{stats['minimo']}"),
        ("Valor Máximo:", f"{stats['maximo']}"),
        ("", ""),  # Separador
        ("Media (μ):", f"{stats['media']:.4f}"),
        ("Desv. Estándar (σ):", f"{stats['desv_estandar']:.4f}"),
    ]
    
    for label_text, valor_text in etiquetas:
        if label_text == "":
            separator = ttk.Separator(stats_frame, orient=tk.HORIZONTAL)
            separator.pack(fill=tk.X, pady=5)
        else:
            frame_stat = ttk.Frame(stats_frame)
            frame_stat.pack(fill=tk.X, pady=8)
            
            label = ttk.Label(frame_stat, text=label_text, font=("Arial", 10, "bold"), width=20)
            label.pack(side=tk.LEFT, anchor=tk.W)
            
            valor = ttk.Label(frame_stat, text=valor_text, font=("Arial", 10), foreground="blue")
            valor.pack(side=tk.LEFT, anchor=tk.E, padx=5)
    
    ventana.mainloop()
