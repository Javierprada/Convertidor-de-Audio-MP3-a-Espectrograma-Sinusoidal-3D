# FUNCIÓN PRINCIPAL: Renderizar el video.
# Controla FFmpeg, envía los fotogramas y combina el video con el audio original.
# RECIBE: Audio análisis y configuración visual.
# DEVUELVE: Ruta del mp4 génerado.

import subprocess
from pathlib import Path

import numpy as np

def crear_video(
    fotogramas,
    ruta_audio,
    ruta_salida,
    fps=30,
    codec_video="libx264",
    calidad=18
    
):
    """
    Codifica fotogramas RGB en un MP4 y añade el audio original.

    Parámetros:
        fotogramas: iterable que produce matrices RGB uint8.
        ruta_audio: ruta del archivo de audio original.
        ruta_salida: ruta del MP4 que se generará.
        fps: fotogramas por segundo.
        codec_video: códec de vídeo utilizado por FFmpeg.
        calidad: valor CRF de libx264; valores menores dan
                 mayor calidad y archivos generalmente más grandes.

    Retorna:
        La ruta del vídeo generado como objeto Path.
    """
    
    ruta_audio = Path(ruta_audio)
    ruta_salida = Path(ruta_salida)
    
    if not ruta_audio.is_file():
        raise ValueError(
            f"No se encontro el audio original: {ruta_audio}"
        )
        
    if fps <= 0:
        raise ValueError(
            "Los FPS deben ser mayores que cero."
        )
        
    if not 0 <= calidad <= 51:
        raise ValueError(
            "La calidad CRF debe estar entre 0 y 51."
        )
        
    ruta_salida.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    
    comando = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",

        # Entrada 0: fotogramas RGB por la entrada estándar.
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-video_size", "1200x600",
        "-framerate", str(fps),
        "-i", "pipe:0",

        # Entrada 1: archivo de audio original.
        "-i", str(ruta_audio),

        # Usar el vídeo generado y el audio original.
        "-map", "0:v:0",
        "-map", "1:a:0",

        # Codificación del vídeo.
        "-c:v", codec_video,
        "-preset", "medium",
        "-crf", str(calidad),
        "-pix_fmt", "yuv420p",

        # Codificación del audio para compatibilidad MP4.
        "-c:a", "aac",
        "-b:a", "192k",

        # El audio determina el final del vídeo.
        "-shortest",
        "-movflags", "+faststart",

        str(ruta_salida),
    ]
    
    proceso = subprocess.Popen(
        comando,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    
    cantidad_fotogramas = 0
    
    try:
        for fotograma in fotogramas:
            if not isinstance(fotograma, np.ndarray):
                raise TypeError(
                    "Cada fotograma debe ser un numpy.ndarray."
                )

            if fotograma.shape != (600, 1200, 3):
                raise ValueError(
                    "Cada fotograma debe tener dimensiones "
                    "(600, 1200, 3)."
                )

            if fotograma.dtype != np.uint8:
                raise TypeError(
                    "Los fotogramas deben tener tipo uint8."
                )

        try:
            proceso.stdin.write(
                np.ascontiguousarray(fotograma).tobytes()
            )
            
        except BrokenPipeError as error:
            
                errores = proceso.stderr.read()
                codigo_salida = proceso.wait()
                
                raise RuntimeError(
                    f"FFmpeg se cerró mientras recibía los fotogramas. "
                    f"Código de salida: {codigo_salida}\n"
                    f"Detalle de FFmpeg:\n"
                    f"{errores.decode('utf-8', errors='replace')}"
                ) from error

                

        cantidad_fotogramas += 1

        if cantidad_fotogramas == 0:
            raise ValueError(
                "El visualizador no produjo ningún fotograma."
            )

        proceso.stdin.close()
        errores = proceso.stderr.read()
        codigo_salida = proceso.wait()

        if codigo_salida != 0:
            raise RuntimeError(
                "FFmpeg no pudo generar el vídeo:\n"
                + errores.decode("utf-8", errors="replace")
            )

        if not ruta_salida.is_file():
            raise RuntimeError(
                "FFmpeg terminó, pero no se encontró el vídeo generado."
            )

        if ruta_salida.stat().st_size == 0:
            raise RuntimeError("El archivo de vídeo está vacío.")

        return ruta_salida

    except Exception:
        if proceso.poll() is None:
            proceso.kill()

        proceso.wait()
        raise

    finally:
        if proceso.stdin and not proceso.stdin.closed:
            proceso.stdin.close()

        if proceso.stderr:
            proceso.stderr.close()
