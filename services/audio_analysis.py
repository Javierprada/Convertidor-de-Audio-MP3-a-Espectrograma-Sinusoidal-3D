# FUNCIÓN PRINCIPAL: analizar_audio()  
# RECIBE: Ruta del audio y parámetros FFT
# DEVUELVE: Matriz, magnitudes y datos del audio

import librosa
import numpy as np


def analizar_audio(
    ruta_audio: str,
    n_fft: int = 2048,
    longitud_de_salto: int = 512
):
    """
    Carga el audio y calcula las magnitudes de la FFT
    para cada ventana del canal izquierdo.
    """

    # 1. Cargar el audio conservando su frecuencia original
    # y los canales disponibles.
    muestras, frecuencia_muestreo = librosa.load(
        ruta_audio,
        sr=None,
        mono=False
    )

    # 2. Seleccionar el canal que vamos análizar.
    if muestras.ndim == 1:
        canal = muestras
        
    else:
        canal = np.mean(muestras, axis=0)
        
    # 3. Comprobamos que hayan suficientes muetras
    numero_muestras = len(canal)

    if numero_muestras < n_fft:
        raise ValueError(f"El audio debe tener al menos {n_fft} muetras")


    # 4. Calcular cuantas ventanas caben en el audio.
    numero_ventanas = (
        (numero_muestras - n_fft) // longitud_de_salto
    ) + 1

    # 5. Calcular la fft de cada ventana.
    todas_las_magnitudes = []

    for numero in range(numero_ventanas):
        inicio = numero * longitud_de_salto
        final =  inicio + n_fft
        
        ventana = canal[inicio:final]
        
        resultado_fft = np.fft.rfft(ventana)
        
        magnitudes = np.abs(resultado_fft)
        
        todas_las_magnitudes.append(magnitudes)
        
    # 6. Convertir los resultados de una matriz.
    matriz_magnitudes = np.array(todas_las_magnitudes)
    
    # 7. Obtener la frecuencia correspondiente a cada columna.
    frecuencias = np.fft.rfftfreq(
        n_fft,
        d=1 / frecuencia_muestreo
    )
    
    # 8. Calcular la duración del canal analizado.
    duracion_audio = numero_muestras / frecuencia_muestreo
    
    
    return (
        matriz_magnitudes,
        frecuencias,
        frecuencia_muestreo,
        duracion_audio
    )