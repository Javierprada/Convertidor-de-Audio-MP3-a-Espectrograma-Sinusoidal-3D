🎵 Convertidor de Audio MP3 a Espectrograma Sinusoidal 3D

📌 Descripción

Este proyecto es una API desarrollada en Python que permite cargar archivos de audio y analizar sus características mediante técnicas de procesamiento digital de señales (DSP).

A partir de un archivo de audio, el sistema procesa la señal para obtener información sobre sus componentes frecuenciales y generar una representación visual tridimensional del espectro sinusoidal del sonido.

El proyecto utiliza FastAPI para recibir los archivos de audio y organizar el procesamiento mediante módulos independientes.

🎯 Objetivos

Recibir archivos de audio mediante una API.

Procesar señales de audio digital.

Analizar las frecuencias presentes en el sonido mediante la Transformada Rápida de Fourier (FFT).

Obtener información frecuencial a lo largo del tiempo.

Generar una representación gráfica tridimensional del análisis del audio.

Organizar el código en módulos para facilitar su mantenimiento y ampliación.

⚙️ Tecnologías utilizadas

Python: lenguaje principal del proyecto.

FastAPI: creación de la API para recibir archivos de audio.

Librosa: análisis y procesamiento de señales de audio.

NumPy: operaciones numéricas y manipulación de matrices.

Matplotlib: generación de gráficos y visualizaciones.

FFmpeg: apoyo en la conversión y lectura de formatos de audio, cuando sea necesario.

Uvicorn: servidor ASGI para ejecutar la API.

🔄 Funcionamiento

El usuario carga un archivo de audio mediante la API.

FastAPI recibe el archivo y lo prepara para su procesamiento.

El módulo de análisis de audio procesa la señal.

Se calcula el contenido frecuencial mediante técnicas FFT y análisis espectral.

Se generan los datos necesarios para la visualización.

El sistema produce la representación gráfica tridimensional del análisis.
