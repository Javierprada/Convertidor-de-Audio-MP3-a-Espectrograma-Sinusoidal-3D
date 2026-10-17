# FUNCIONES PRINCIPALES:  crear_visualizador()  / actualizar_barras()
# RECIBE: Número de barras y estilos / Barras y magnitudes actuales
# DEVUELVE: Figuras ejes y barras  / Alturas actualizadas
import math
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def normalizar_superficie(superficie):
    """Normaliza las magnitudes de una superficie sin dividir por cero."""
    maxima =  np.max(superficie)
    
    if maxima <= 0:
        return np.zeros_like(superficie)
    
    return superficie / maxima


def color_degradado(posicion):
    """
    Degradado de cian a morado y magenta.
    posicion debe estar entre 0 y 1.
    """
    cian = np.array([0.0, 0.85, 1.0])
    morado = np.array([0.55, 0.15, 1.0])
    magenta = np.array([1.0, 0.0, 0.65])
    
    if posicion <= 0.5:
        factor = posicion * 2
        return cian * (1 - factor) + morado  * factor
    
    factor = (posicion - 0.5) * 2
    return morado * (1 - factor) + magenta * factor


def generar_fotogramas(
    matriz_magnitudes,
    frecuencias,
    frecuencia_muestreo,
    duracion_audio,
    longitud_de_salto=512,
    fps=30,
    cantidad_puntos_frecuencia=80,
    profundidad_malla=32
    
):
    """
    Genera fotogramas RGB de una onda espectral 3D de neón.

    La profundidad representa ventanas FFT recientes.
    El eje horizontal representa las frecuencias.
    La altura representa las magnitudes normalizadas.

    No modifica ni exporta el audio original.
    """
    
    if matriz_magnitudes.ndim != 2:
        raise ValueError(
            "La matriz de magnitudes debe ser bidimensional."
        )
        
    if len(matriz_magnitudes) == 0:
        raise ValueError(
            "No hay ventanas FFT para visualizar."
        )
        
    if len(frecuencias) != matriz_magnitudes.shape[1]:
        raise ValueError(
            "Las frecuencias no coinciden con las columnas "
            "de la matriz de magnitudes."
        )
        
    if frecuencia_muestreo <= 0 or longitud_de_salto <= 0:
        raise ValueError(
            "La frecuencia de muestreo y el salto deben ser positivos."
        )

    if fps <= 0 or duracion_audio < 0:
        raise ValueError(
            "FPS o duración no válidos."
        )
        
    numero_fotogramas = math.ceil(duracion_audio * fps)
    
    if numero_fotogramas == 0:
        return
    
    numero_ventanas = matriz_magnitudes.shape[0]
    
    # Reducir las columnas para dibujar una malla manejable.
    cantidad_puntos_frecuencia = min(
        cantidad_puntos_frecuencia,
        matriz_magnitudes.shape[1]
    )
    
    indices_frecuencia = np.linspace(
        0,
        matriz_magnitudes.shape[1] - 1,
        cantidad_puntos_frecuencia,
        dtype=int
    )
    
    eje_frecuencias = frecuencias[indices_frecuencia]
    
    # Escala logarítmica para distribuir visualmente las frecuencias.
    eje_x = np.log10(1 + eje_frecuencias)
    eje_x = (eje_x - eje_x.min()) / max(
        np.ptp(eje_x), 1e-12
    )
    
    # La malla conserva las ventanas recientes.
    eje_profundidad = np.linspace(
        -1.0, 0.0, profundidad_malla
    )
    # Definimos la malla
    x, y = np.meshgrid(eje_x, eje_profundidad)
    
    # Preparar la figura 3D.
    fig = plt.figure(figsize=(12, 6), dpi=100)
    fig.patch.set_facecolor("#05051a")
    
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#05051a")
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 0)
    ax.set_zlim(0, 1)
    
    ax.view_init(elev=28, azim=-65)
    ax.set_box_aspect((2.0, 1.3, 0.8))
    
    # Ocultar ejes y paneles para una estética limpia.
    ax.set_axis_off()
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1
    ) 
    
    # Crear las líneas de la malla una sola vez.
    lineas_malla = []
    
    # Líneas que recorren las frecuencias.
    for fila in range(profundidad_malla):
        color = color_degradado(
            fila / max(profundidad_malla -1, 1)
        )
        
        linea, = ax.plot(
            x[fila],
            y[fila],
            np.zeros(cantidad_puntos_frecuencia),
            color=color,
            linewidth=1.0,
            alpha=0.95
        )
        
        lineas_malla.append(("fila", fila, linea))
        
    # Líneas que recorren la profundidad.
    for columna in range(cantidad_puntos_frecuencia):
        color= color_degradado(
            columna / max(cantidad_puntos_frecuencia -1, 1)
        )
        
        linea, = ax.plot(
            x[:, columna],
            y[:, columna],
            np.zeros(profundidad_malla),
            color=color,
            linewidth=0.8,
            alpha=0.65
        )
        
        lineas_malla.append(("columna", columna, linea))
    
    # Matriz reducida: magnitudes por ventana y frecuencia.
    espectro_reducido = matriz_magnitudes[
        :, indices_frecuencia
    ].astype(float)
    
    # Comprimir el rango para evitar que unos pocos picos
    # oculten el resto del espectro.
    espectro_reducido = np.log1p(espectro_reducido)
    
    maximo_global = np.max(espectro_reducido)
    
    if maximo_global > 0:
        espectro_reducido /= maximo_global
        
    
    try:
        for numero_fotograma in range(numero_fotogramas):

            tiempo_actual = numero_fotograma / fps

            indice_ventana = int(
                tiempo_actual
                * frecuencia_muestreo
                / longitud_de_salto
            )

            indice_ventana = min(
                indice_ventana,
                numero_ventanas - 1,
            )

            # Extraer el historial disponible hasta este instante.
            inicio_historial = max(
                0, indice_ventana - profundidad_malla + 1
            )

            historial = espectro_reducido[
                inicio_historial:indice_ventana + 1
            ]

            # Completar el historial al comienzo del audio.
            superficie = np.zeros(
                (profundidad_malla, cantidad_puntos_frecuencia)
            )

            superficie[-len(historial):] = historial

            # Las magnitudes determinan el relieve vertical.
            Z = superficie

            # Actualizar las líneas sin reconstruir la figura.
            for tipo, indice, linea in lineas_malla:

                if tipo == "fila":
                    linea.set_data_3d(
                        x[indice],
                        y[indice],
                        Z[indice],
                    )
                else:
                    linea.set_data_3d(
                        x[:, indice],
                        y[:, indice],
                        Z[:, indice],
                    )

            fig.canvas.draw()

            # Entregar una imagen RGB independiente.
            fotograma = np.asarray(
                fig.canvas.buffer_rgba()
            )[:, :, :3].copy()

            yield fotograma

    finally:
        plt.close(fig)
    

