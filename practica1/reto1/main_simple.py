import time
from matricesImagenes import crear_matriz_aleatoria, calcular_estadisticas


def main():
    
    # Tiempo total del proceso
    tiempo_inicio_total = time.time()
    
    # Crear matriz
    tiempo_inicio_matriz = time.time()
    matriz = crear_matriz_aleatoria()
    tiempo_fin_matriz = time.time()
    duracion_matriz = tiempo_fin_matriz - tiempo_inicio_matriz
    
    # Calcular estadísticas
    tiempo_inicio_stats = time.time()
    stats = calcular_estadisticas(matriz)
    tiempo_fin_stats = time.time()
    duracion_stats = tiempo_fin_stats - tiempo_inicio_stats

    tiempo_fin_total = time.time()
    duracion_total = tiempo_fin_total - tiempo_inicio_total
    
    # Mostrar resultados
    print("\n" + "="*60)
    print("RESULTADOS ESTADÍSTICOS")
    print("="*60)
    print(f"\nDimensiones:           1000 × 1000")
    print(f"Total de píxeles:      {1000 * 1000:,}")
    print(f"\nValor mínimo:          {stats['minimo']}")
    print(f"Valor máximo:          {stats['maximo']}")
    print(f"Rango:                 {stats['maximo'] - stats['minimo']}")
    print(f"\nMedia (μ):             {stats['media']:.4f}")
    print(f"Desviación estándar (σ): {stats['desv_estandar']:.4f}")
    
    print("\n" + "="*60)
    print("TIEMPOS DE EJECUCIÓN")
    print("="*60)
    print(f"\nCreación de matriz:        {duracion_matriz:.6f} segundos")
    print(f"Cálculo de estadísticas:   {duracion_stats:.6f} segundos")
    print(f"─" * 60)
    print(f"Tiempo total:              {duracion_total:.6f} segundos")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
