# FUNCIONES PRINCIPALES:  crear_visualizador()  / actualizar_barras()
# RECIBE: Número de barras y estilos / Barras y magnitudes actuales
# DEVUELVE: Figuras ejes y barras  / Alturas actualizadas
import math
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from datetime import date


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
    cantidad_puntos_frecuencia=50,
    profundidad_malla=50
    
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
    
    
    
    # Ignorar la columna 0 (DC) y limitar a un rango audible útil.
    frecuencia_minima = 30.0
    frecuencia_maxima = min(16000.0, frecuencias[-1])
    
    # ------------------------------------------------------------------
    # CORREGIDO (#3): en vez de tomar UNA columna FFT por punto,
    # se definen BANDAS logarítmicas y se promedia cada una.
    # Esto elimina los escalones planos en graves y estabiliza agudos.
    # ------------------------------------------------------------------
    bordes = np.logspace(
        np.log10(frecuencia_minima),
        np.log10(frecuencia_maxima),
        cantidad_puntos_frecuencia + 1
    )
    
    centros = np.sqrt(bordes[:-1] * bordes[1:]) # Centro geometrico

    
    magnitudes = matriz_magnitudes.astype(float)
    espectro_reducido = np.zeros(
        (numero_ventanas, cantidad_puntos_frecuencia)
    )
    posiciones_bins = np.arange(len(frecuencias))
    
    
    
    for i in range(cantidad_puntos_frecuencia):
        mascara = (frecuencias >= bordes[i]) & (frecuencias < bordes[i + 1])
        
        if np.any(mascara):
            # Banda con varias columnas FFT: promedio.
            espectro_reducido[:, i] = magnitudes[:, mascara].mean(axis=1)
            
        else:
            # Banda más estrecha que una columna FFT (típico en graves):
            # interpolar linealmente entre las dos columnas vecinas
            # en vez de repetir la misma columna.
            pos = np.interp(centros[i], frecuencias, posiciones_bins)
            bajo = int(np.floor(pos))
            alto = min(bajo + 1, len(frecuencias) -1)
            peso = pos - bajo
            espectro_reducido[:, i] = (
                magnitudes[:, bajo] * (1 - peso) + magnitudes[:, alto] * peso
            )
    
    print("Forma:", espectro_reducido.shape)  # esperado: (2471, 50)
    print("Columnas con datos:", np.count_nonzero(espectro_reducido.sum(axis=0)))  # esperado: 50
    print("Suma por columna:", np.round(espectro_reducido.sum(axis=0), 1))
    
    
    # Escala logarítmica para distribuir visualmente las frecuencias.
    eje_x = np.linspace(1, 0, cantidad_puntos_frecuencia)
   
    
    pendiente_db_por_octava = 4.0   # sube a 5-6 si los agudos siguen bajos
    rango_dinamico_db = 60.0        # baja a 50 para más contraste
    contraste = 1.3                 # >1 acentúa los picos
    
    # Magnitud -> dB (el 1e-10 evita log(0) en silencios).
    db = 20.0 * np.log10(espectro_reducido + 1e-10)
    
    # Compensar la caída natural de energía hacia agudos.
    octavas = np.log2(centros / centros[0])
    db = db + pendiente_db_por_octava * octavas
    
    # Techo global (percentil, para que un pico aislado no aplaste todo).
    techo_db = np.percentile(db, 99.5)
    piso_db = techo_db - rango_dinamico_db
    
    espectro_reducido = np.clip(
        (db - piso_db) / rango_dinamico_db, 0.0, 1.0
    ) ** contraste
    
    
    # La malla conserva las ventanas recientes.
    posiciones = np.linspace(0, 1, profundidad_malla)
    eje_profundidad = -1.0 + posiciones
    
    # Definimos la malla
    x, y = np.meshgrid(eje_x, eje_profundidad)
    
    # Preparar la figura 3D.
    fig = plt.figure(figsize=(12, 6.76), dpi=100)
    fig.patch.set_facecolor("#ffffff") # Color del margin
    
    ax = fig.add_axes(
        [0.02, 0.02, 0.96, 0.96],
        projection="3d"
        )
    ax.set_facecolor("#f9f9f9") # Color del padding
    
    # Nombre del desarrollador
    fig.text(
        0.5, 0.92,
        "Developer / Joseth Prada",
        color="#00D9FF",
        fontsize=25,
        fontweight="bold",
        ha="center",
        va="center",
        family="sans-serif"
    )
    
     # Fecha de géneración del video.
    fig.text(
        0.5, 0.86,
        date.today().strftime("%d/%m/%Y"),
        color="#00D9FF",
        fontsize=25,
        fontweight="bold",
        ha="center",
        va="center",
        family="sans-serif"
    )
    
    # Límites de la escena
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 0)
    ax.set_zlim(0, 1.5)
    
    # Perspectiva de la cámara
    ax.view_init(elev=25, azim=90)
    
    
    
    # Ajusta la proporción de la malla
    ax.set_box_aspect((60, 60, 50), zoom=1.4)
    
    # Ocultar ejes y paneles para una estética limpia.
    ax.set_axis_off()
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=0.88
    ) 
    
    # Crear las líneas de la malla una sola vez.
    lineas_malla = []
    
    # Líneas que recorren las frecuencias de color cyan eléctrico.
    for fila in range(profundidad_malla):
        color = color_degradado(
            fila / max(profundidad_malla -1, 1)
        )
        
        es_linea_principal = fila % 5 == 0
        
        linea, = ax.plot(
            x[fila],
            y[fila],
            np.zeros(cantidad_puntos_frecuencia),
            color="#35ff5d" if es_linea_principal else color,
            linewidth=1.5 if es_linea_principal else 0.45,
            alpha=1.0 if es_linea_principal else 0.45
        )
        
        lineas_malla.append(("fila", fila, linea))
        
    for indice_columna, posicion in enumerate(eje_x):
        
        linea, = ax.plot(
            [posicion] * profundidad_malla,
            eje_profundidad,
            np.zeros(profundidad_malla),
            color="#00FFF2" ,
            linewidth=0.55,
            alpha=0.5
        )
        
        lineas_malla.append(("columna", indice_columna, linea))
    
   
    
        
    
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
            
            if numero_fotograma % 30 == 0:
                print(
                    "Fotograma:", numero_fotograma,
                    "Tiempo:", tiempo_actual,
                    "Ventana:", indice_ventana
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
            
            if numero_fotograma % 30 == 0:
                print(
                    "Altura min:", Z.min(),
                    "Altura maxima:", Z.max(),
                    "Variación:", np.std(Z)
                )

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
    

