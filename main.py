# FUNCIÓN PRINCIPAL: recibir_audio()
# RECIBE: Archivo de audio enviado por el cliente.
# DEVUELVE: Video MP4 generado a partir del audio.

import os
from enum import Enum
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from fastapi.responses import FileResponse

from services.audio_analysis import analizar_audio
from services.visualizer import generar_fotogramas
from services.video_renderer import crear_video


app = FastAPI()




@app.get("/")
def raiz():
    return {"mensaje": "API funcionando correctamente"}


class Figura(str, Enum):
    malla_sinusoidal_3D = "malla_sinusoidal_3D"
    esfera_3D = "esfera_3D"

@app.post("/upload")
async def recibir_audio(
    background_tasks: BackgroundTasks,
    archivo: UploadFile = File(...),
    figura: Figura = Form(Figura.malla_sinusoidal_3D)
):
    
    
    print(">>> ENTRO EN recibir_audio()", flush=True)
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
            figura=figura.value
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
        
        # 6. Programar la limpieza de los archivos temporales.
        background_tasks.add_task(os.remove, str(ruta_video))
        background_tasks.add_task(os.remove, str(ruta_audio))
        
        # 7. Devolver el vídeo MP4 al cliente.
        return FileResponse(
            path=str(ruta_video),
            media_type="video/mp4",
            filename="visualizacion_isunoidal3D.mp4",
            background=background_tasks,
        )

    except Exception:
        if ruta_audio and os.path.exists(ruta_audio):
            os.remove(ruta_audio)

        if ruta_video and os.path.exists(ruta_video):
            os.remove(ruta_video)

        raise

    finally:
        await archivo.close()

