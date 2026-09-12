from fastapi import FastAPI, UploadFile, File
import librosa
import numpy as np

app = FastAPI()


@app.get("/")
def raiz():
    return {"mensaje": "API funcionando correctamente"}


@app.post("/upload")
async def recibir_audio(archivo: UploadFile = File(...)):

    muestras, frecuencia_muestreo = librosa.load(
        archivo.file,
        sr=None,  # No hagas conversión de frecuencia de muestreo. Déjame la frecuencia original.
        mono=False,  # le decimos a librosa que no convierta automáticamente un audio estéreo a mono.
    )

    # Separar los canales.
    canal_izquierdo = muestras[0]  # Canal Derecho
    canal_derecho = muestras[1]  # Canal Izquierdo

    # Tamaño de cada ventana
    n_fft = 2048
    
    # Cuánto avanzamos para buscar la siguiente ventana
    longitud_de_salto = 512
    
    # Cantidad total de muestras del canal izquierdo
    numero_muestras = len(canal_izquierdo)

    # Cantidad de ventanas que caben en el audio
    numero_ventanas = ((numero_muestras - n_fft) // longitud_de_salto) + 1
    
    # Bucle 
    for numero in range (numero_ventanas):
        inicio = numero * longitud_de_salto
        final = inicio + n_fft
        
        # Aplicamos FFT a esta ventana
        ventana_izquierda = canal_izquierdo[inicio:final]
        
        print("Ventana:", numero)
        print("Inicio:", inicio)
        print("Final:", final)
        print("Cantidad de muestras", len(ventana_izquierda))
        print()
    
    

    

    
