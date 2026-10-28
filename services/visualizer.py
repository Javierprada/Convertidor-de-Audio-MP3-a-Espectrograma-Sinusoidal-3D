"""
visualizer.py
==============

Responsabilidad:
    Coordinar la generación de fotogramas RGB a partir del análisis FFT.

Geometrías admitidas:
    - malla_sinusoidal_3D
    - esfera_3D

Flujo:
    audio_analysis.py
        -> matriz de magnitudes FFT
        -> preparar_espectro()
        -> seleccionar visualizador
        -> generar fotogramas RGB
        -> video_renderer.py

IMPORTANTE:
    Los generadores utilizan yield para entregar un fotograma a la vez.
    Así evitamos almacenar todos los fotogramas del vídeo en memoria RAM.
"""

import math
import time
import matplotlib
matplotlib.use("Agg")  # Backend sin interfaz gráfica, adecuado para renderizar video.

import matplotlib.pyplot as plt
import numpy as np

from figuras_geometricas_3D.malla_sinusoidal_3D import MallaSinusoidal3D

from figuras_geometricas_3D.esfera_3D import(
    generar_esfera_vertices,
    mapear_magnitudes_a_esfera,
    renderizar_a_frame_rgb
)

# ============================================================
# 1. PREPARACIÓN DEL ESPECTRO
# ============================================================

def preparar_espectro(
    matriz_magnitudes,
    frecuencias,
    cantidad_puntos_frecuencia=50,
    
):

    # np.asarray evita copiar los datos cuando ya son arrays compatibles.
    matriz_magnitudes = np.asarray(matriz_magnitudes)
    frecuencias = np.asarray(frecuencias, dtype=float)
    
    # ---------------- VALIDACIONES ----------------
    if matriz_magnitudes.ndim != 2: # Cumprueba el numero de dimensiones
        raise ValueError(
            "La matriz de magnitudes debe ser bidimensional"
        )
        
    if matriz_magnitudes.shape[0] == 0: # 
        raise ValueError("No hay ventanas FFT para visualizar")
    
    if matriz_magnitudes.shape[1] != len(frecuencias):
        raise ValueError("Las columnas de magnitudes no coinciden con las frecuencias.")
    
    if len(frecuencias) < 2:
        raise ValueError("Se necesitan al menos dos valores de frecuencia.")
    
    if not np.all(np.isfinite(frecuencias)):
        raise ValueError("Las frecuencias contienen valores no finitos.")
    
    if np.any(np.diff(frecuencias) < 0):
        raise ValueError("Las frecuencias deben estar ordenadas.")

    if not np.all(np.isfinite(matriz_magnitudes)):
        raise ValueError(
            "La matriz de magnitudes contiene valores no finitos."
        )

    if np.any(matriz_magnitudes < 0):
        raise ValueError(
            "Se esperaban magnitudes FFT no negativas."
        )

    if cantidad_puntos_frecuencia <= 0:
        raise ValueError(
            "cantidad_puntos_frecuencia debe ser positiva."
        )
        
    # ---------------- DEFINICIÓN DE BANDAS ----------------
     # No solicitamos más bandas que bins disponibles.
    cantidad_bandas = min(
        cantidad_puntos_frecuencia,
        matriz_magnitudes.shape[1],
    )

    frecuencia_minima = 30.0

    # Evitamos superar la frecuencia máxima presente en el análisis.
    frecuencia_maxima = min(
        16000.0,
        float(frecuencias[-1]),
    )

    if frecuencia_maxima <= frecuencia_minima:
        raise ValueError(
            "El rango de frecuencias debe superar los 30 Hz."
        )

    # Las bandas logarítmicas distribuyen mejor la resolución perceptual:
    # hay mayor separación relativa en frecuencias graves y agudas.
    bordes = np.logspace(
        np.log10(frecuencia_minima),
        np.log10(frecuencia_maxima),
        cantidad_bandas + 1,
    )

    # Cada centro representa una banda del eje de frecuencias.
    centros = np.sqrt(bordes[:-1] * bordes[1:])

    # Creamos una matriz nueva para almacenar el espectro reducido.
    numero_ventanas = matriz_magnitudes.shape[0]

    espectro_reducido = np.zeros(
        (numero_ventanas, cantidad_bandas),
        dtype=np.float32,
    )

    # Convertimos a float para poder calcular promedios con seguridad.
    # Esta conversión puede crear una copia completa de la matriz.
    magnitudes = matriz_magnitudes.astype(float, copy=False)

    # ---------------- REDUCCIÓN A BANDAS ----------------

    for indice_banda in range(cantidad_bandas):

        # Identificamos los bins FFT pertenecientes a esta banda.
        mascara = (
            (frecuencias >= bordes[indice_banda])
            & (frecuencias < bordes[indice_banda + 1])
        )

        # La última banda incluye también su borde superior.
        if indice_banda == cantidad_bandas - 1:
            mascara |= (
                frecuencias == bordes[indice_banda + 1]
            )

        if np.any(mascara):
            # Promedio de los bins de esta banda para TODAS las ventanas.
            #
            # magnitudes[:, mascara]:
            #   selección temporal de las columnas de la banda.
            #
            # mean(axis=1):
            #   un promedio por ventana FFT.
            #
            # El resultado tiene una dimensión temporal, no una imagen.
            espectro_reducido[:, indice_banda] = np.mean(
                magnitudes[:, mascara],
                axis=1,
            )

        else:
            # Si no existe ningún bin en la banda, buscamos el más cercano
            # al centro para no dejarla vacía.
            indice_cercano = np.argmin(
                np.abs(frecuencias - centros[indice_banda])
            )

            espectro_reducido[:, indice_banda] = magnitudes[
                :, indice_cercano
            ]

    # ---------------- CONVERSIÓN A DECIBELES ----------------

    # Evitamos log10(0), que produciría -inf.
    epsilon = 1e-10

    espectro_db = 20.0 * np.log10(
        np.maximum(espectro_reducido, epsilon)
    )

    # Referencia global: el valor más alto de toda la matriz.
    #
    # La normalización global conserva una escala relativa consistente
    # entre ventanas, en lugar de normalizar cada instante por separado.
    maximo_db = float(np.max(espectro_db))

    espectro_relativo_db = espectro_db - maximo_db

    # Limitamos el rango dinámico a 60 dB.
    # -60 dB -> 0.0
    #   0 dB -> 1.0
    espectro_normalizado = np.clip(
        (espectro_relativo_db + 60.0) / 60.0,
        0.0,
        1.0,
    ).astype(np.float32)

    return espectro_normalizado, centros


# ============================================================
# 2. VISUALIZADOR DE LA MALLA SINUSOIDAL 3D
# ============================================================

def generar_fotogramas_malla(
    espectro_normalizado,
    centros_frecuencia,
    duracion_audio,
    fps=30,
    profundidad_malla=50,
):
    """
    Genera fotogramas de una malla espectral 3D.

    Ejes:
        X: bandas de frecuencia.
        Y: ventanas FFT recientes.
        Z: magnitudes normalizadas.

    La geometría de la superficie se delega a MallaSinusoidal3D.

    NOTA:
        Este código presupone que el constructor de tu clase recibe
        (ax, centros_frecuencia, posiciones_profundidad) y que
        actualizar(superficie) dibuja o actualiza la superficie.
        Si la firma real de tu clase difiere, hay que ajustar esa llamada.
    """

    numero_ventanas, numero_bandas = espectro_normalizado.shape

    # El número de fotogramas depende de la duración del audio y los FPS.
    numero_fotogramas = math.ceil(duracion_audio * fps)

    if numero_fotogramas <= 0:
        return

    # ---------------- CREACIÓN DE LA ESCENA ----------------

    # 1200 x 676 píxeles a 100 DPI.
    # Esta dimensión debe coincidir con video_renderer.py.
    figura, ax = plt.subplots(
        figsize=(12.0, 6.76),
        dpi=100,
        subplot_kw={"projection": "3d"},
    )

    # Eje Y: posiciones de la ventana temporal que se verá en la malla.
    posiciones_profundidad = np.arange(
        profundidad_malla,
        dtype=float,
    )

    # Instanciamos la geometría una sola vez por vídeo.
    # La clase se encarga de dibujar/actualizar la superficie.
    malla = MallaSinusoidal3D(
        ax,
        centros_frecuencia,
        posiciones_profundidad,
    )

    # Configuración de la escena.
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Tiempo reciente")
    ax.set_zlabel("Magnitud normalizada")
    ax.set_title("Malla sinusoidal 3D reactiva al audio")

    ax.set_zlim(0.0, 1.0)
    ax.view_init(elev=28, azim=-60)

    # Una sola matriz temporal de trabajo para la superficie.
    # Se reutiliza en cada iteración para evitar asignaciones constantes.
    superficie = np.zeros(
        (profundidad_malla, numero_bandas),
        dtype=np.float32,
    )

    try:
        for indice_fotograma in range(numero_fotogramas):

            # Convertimos el tiempo del fotograma al índice FFT.
            tiempo = indice_fotograma / fps

            indice_ventana = min(
                int(tiempo * numero_ventanas / max(duracion_audio, 1e-9)),
                numero_ventanas - 1,
            )

            # Seleccionamos las ventanas anteriores a la actual.
            inicio = max(
                0,
                indice_ventana - profundidad_malla + 1,
            )

            historial = espectro_normalizado[
                inicio:indice_ventana + 1
            ]

            # Limpiamos la matriz reutilizable.
            superficie.fill(0.0)

            # Colocamos el historial al final para que las ventanas más
            # recientes aparezcan en el extremo más próximo del eje Y.
            cantidad_historial = historial.shape[0]

            superficie[
                -cantidad_historial:, :
            ] = historial

            # Delegamos el dibujo a la clase de geometría.
            malla.actualizar(superficie)

            # Matplotlib dibuja la escena en su buffer de memoria.
            figura.canvas.draw()

            # buffer_rgba contiene RGBA.
            # Tomamos RGB y hacemos una copia independiente del buffer.
            fotograma_rgb = np.asarray(
                figura.canvas.buffer_rgba()
            )[:, :, :3].copy()

            # El resultado tiene forma (676, 1200, 3), uint8.
            yield fotograma_rgb

    finally:
        # Liberamos la figura incluso si el consumidor interrumpe
        # la generación o se produce una excepción.
        plt.close(figura)


# ============================================================
# 3. VISUALIZADOR DE LA ESFERA 3D
# ============================================================

def generar_fotogramas_esfera(
    espectro_normalizado,
    duracion_audio,
    fps=30,
    n_lat=180,
    n_lon=160,
    factor_altura=0.3,
):
    """
    Genera fotogramas RGB de una esfera que responde al espectro.

    En cada fotograma:
        1. Selecciona las magnitudes del instante.
        2. Modifica los vértices de la esfera.
        3. Renderiza la geometría como imagen RGB.

    La geometría y el renderizado se delegan a esfera_3D.py.
    """

    numero_ventanas, _ = espectro_normalizado.shape

    numero_fotogramas = math.ceil(duracion_audio * fps)

    if numero_fotogramas <= 0:
        return
    
    # --------------------------------------------------------
    # 1. Generar la geometría base una sola vez
    # --------------------------------------------------------

    (
        vertices_base,
        anillo_por_vertice,
        n_lat,
        n_lon
    ) = generar_esfera_vertices(
        n_lat=n_lat,
        n_lon=n_lon,
        radio=1.0,
    )
    
    
    # --------------------------------------------------------
    # 2. Recorrer los fotogramas del vídeo
    # --------------------------------------------------------
    
    inicio_renderizado = time.perf_counter()
    ultimo_porcentaje = -1

    try:
        for indice_fotograma in range(numero_fotogramas):

            tiempo = indice_fotograma / fps

            # Seleccionamos la ventana FFT correspondiente al tiempo.
            indice_ventana = min(
                int(tiempo * numero_ventanas / max(duracion_audio, 1e-9)),
                numero_ventanas - 1,
            )

            # Este vector contiene una magnitud por banda.
            # No contiene todos los instantes del audio.
            magnitudes_instante = espectro_normalizado[
                indice_ventana
            ]

            # La función de la esfera crea los vértices modificados.
            # Si internamente copia vertices_base, esa copia pertenece
            # a este fotograma y no modifica la esfera original.
            vertices_modificados, colores = mapear_magnitudes_a_esfera(
                vertices=vertices_base,
                anillo_por_vertice=anillo_por_vertice,
                magnitudes=magnitudes_instante,
                n_lat=n_lat,
                factor_altura=factor_altura,
                suavizado=True,
            )

            # Renderiza los vértices como una imagen RGB.
            # Según esfera_3D.py, se espera una imagen de 1200 x 676.
            fotograma_rgb = renderizar_a_frame_rgb(
                vertices_modificados,
                colores=colores,
                ancho=1200,
                alto=676,
            )
            
            # Calcular progreso después de renderizar el fotograma.
            frames_completados = indice_fotograma + 1
            porcentaje = int(
                frames_completados * 100 / numero_fotogramas
            )
            
            # Actualizar solamente cuando cambia el porcentaje.
            if porcentaje != ultimo_porcentaje:
                transcurrido = time.perf_counter() - inicio_renderizado
                
                velocidad = frames_completados / transcurrido
                frames_restantes = numero_fotogramas - frames_completados
                
                segundos_restantes = (
                    frames_restantes / velocidad if velocidad > 0 else 0
                )
                
                def formatear_tiempo(segundos):
                    segundos = int(segundos)
                    minutos, segundos = divmod(segundos, 60)
                    horas, minutos = divmod(minutos, 60)
                    return f"{horas:02d}:{minutos:02d}:{segundos:02d}"
                
                print(
                    f"\rRenderizando esféra_3D: {porcentaje}%"
                    f" | {frames_completados}/{numero_fotogramas} frames"
                    f" | transcurrido: {formatear_tiempo(transcurrido)}"
                    f" | restante: {formatear_tiempo(segundos_restantes)}",
                    end="",
                    flush=True,
                )
                
                ultimo_porcentaje = porcentaje

            # Entregamos el fotograma inmediatamente al consumidor.
            # No acumulamos todos los fotogramas en una lista.
            yield fotograma_rgb
        
        print()

    finally:
        # No creamos una figura de Matplotlib en este visualizador.
        # El renderizado de la esfera se realiza en esfera_3D.py.
        pass


# ============================================================
# 4. REGISTRO DE VISUALIZADORES
# ============================================================

# Este diccionario relaciona el identificador recibido desde FastAPI
# con la función responsable de generar los fotogramas de esa figura.
#
# Los nombres de las funciones deben estar definidos antes de crear
# el diccionario.
#
# Para añadir otra figura en el futuro:
#   1. Crear su módulo geométrico.
#   2. Crear su función generar_fotogramas_<figura>().
#   3. Añadir una entrada a este registro.

VISUALIZADORES = {
    "malla_sinusoidal_3D": generar_fotogramas_malla,
    "esfera_3D": generar_fotogramas_esfera,
}


# ============================================================
# 5. PUNTO DE ENTRADA PÚBLICO
# ============================================================

def generar_fotogramas(
    matriz_magnitudes,
    frecuencias,
    frecuencia_muestreo,
    duracion_audio,
    longitud_de_salto=512,
    fps=30,
    cantidad_puntos_frecuencia=50,
    profundidad_malla=50,
    figura="malla_sinusoidal_3D",
):
    """
    Punto de entrada llamado desde main.py.

    Recibe los datos del análisis de audio, prepara el espectro y
    selecciona el generador según la figura elegida por el usuario.

    Devuelve un generador de fotogramas RGB.
    """

    # FastAPI puede entregar un miembro del Enum Figura.
    # Si tiene atributo .value, extraemos el identificador de texto.
    identificador_figura = getattr(figura, "value", figura)

    # Buscamos la función asociada al identificador recibido.
    visualizador = VISUALIZADORES.get(identificador_figura)

    if visualizador is None:
        figuras_disponibles = ", ".join(VISUALIZADORES.keys())

        raise ValueError(
            f"Figura no soportada: {identificador_figura!r}. "
            f"Opciones disponibles: {figuras_disponibles}"
        )

    # Validación básica de parámetros temporales.
    if frecuencia_muestreo <= 0:
        raise ValueError("La frecuencia de muestreo debe ser positiva.")

    if longitud_de_salto <= 0:
        raise ValueError("longitud_de_salto debe ser positiva.")

    if fps <= 0:
        raise ValueError("FPS debe ser positivo.")

    if duracion_audio < 0:
        raise ValueError("La duración no puede ser negativa.")

    # Convertimos la matriz FFT en una representación común a las figuras.
    #
    # Se calcula una vez por petición, no una vez por fotograma.
    espectro_normalizado, centros_frecuencia = preparar_espectro(
        matriz_magnitudes=matriz_magnitudes,
        frecuencias=frecuencias,
        cantidad_puntos_frecuencia=cantidad_puntos_frecuencia,
    )

    # Transmitimos los datos comunes al visualizador seleccionado.
    #
    # La malla necesita centros de frecuencia y profundidad.
    # La esfera necesita las magnitudes y la duración.
    if identificador_figura == "malla_sinusoidal_3D":
        return visualizador(
            espectro_normalizado=espectro_normalizado,
            centros_frecuencia=centros_frecuencia,
            duracion_audio=duracion_audio,
            fps=fps,
            profundidad_malla=profundidad_malla,
        )

    return visualizador(
        espectro_normalizado=espectro_normalizado,
        duracion_audio=duracion_audio,
        fps=fps,
    )
