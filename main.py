# FUNCIÓN PRINCIPAL: recibir_audio()
# RECIBE: Archivo de audio enviado por el cliente.
# DEVUELVE: Video MP4 generado a partir del audio.

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse

from services.audio_analysis import analizar_audio
from services.visualizer import generar_fotogramas
from services.video_renderer import crear_video


app = FastAPI()




@app.get("/")
def raiz():
    return {"mensaje": "API funcionando correctamente"}


@app.post("/upload")
async def recibir_audio(
    background_tasks: BackgroundTasks,
    archivo: UploadFile = File(...)
):
    ruta_audio = None
    ruta_video = None
    
    # 1. Guardar temporalmente el audio recibido.
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(archivo.filename or "audio.wav").suffix or ".wav"
        ) as temporal_audio:
            ruta_audio = temporal_audio.name
            shutil.copyfileobj(archivo.file, temporal_audio)

        # 2. Analizar el audio.
        (
            matriz_magnitudes,
            frecuencias,
            frecuencia_muestreo,
            duracion_audio,
        ) = analizar_audio(ruta_audio)

        # 3. Generar los fotogramas de la visualización.
        fotogramas = generar_fotogramas(
            matriz_magnitudes=matriz_magnitudes,
            frecuencias=frecuencias,
            frecuencia_muestreo=frecuencia_muestreo,
            duracion_audio=duracion_audio,
            fps=30,
        )

        # 4. Crear el vídeo con el audio original.
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )as temporal_video:
            ruta_salida_video = temporal_video.name

        # 5. Devolver el vídeo y limpiar los temporales al finalizar.
        ruta_video = crear_video(
            fotogramas=fotogramas,
            ruta_audio=ruta_audio,
            ruta_salida=ruta_salida_video,
            fps=30,
        )

    except Exception:
        if ruta_audio and os.path.exists(ruta_audio):
            os.remove(ruta_audio)

        if ruta_video and os.path.exists(ruta_video):
            os.remove(ruta_video)

        raise

    finally:
        await archivo.close()

