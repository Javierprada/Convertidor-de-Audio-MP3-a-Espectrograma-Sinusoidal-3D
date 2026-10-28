import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image, ImageFilter, ImageEnhance


def generar_esfera_vertices(n_lat=130, n_lon=110, radio=1.0, limpiar_polos=True):
    """
    Genera puntos ordenados por latitud (anillos horizontales).
    n_lat: número de anillos (de polo a polo)
    n_lon: puntos por anillo
    """
    
    
    
    # Creamos el grid de latitudes y longitudes
    lat = np.linspace(-np.pi/2, np.pi/2, n_lat) # Angulo polar. De 0 a 180
    lon = np.linspace(0, 2*np.pi, n_lon, endpoint=False) # Angulo de giro alrededor del angulo horizontal.
    
    
    
    
    
    
    # Matriz de meshgrid
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
    
    
    
    # Coordenadas x,y,z
    x = radio * np.cos(lat_grid) * np.cos(lon_grid)   # X eje izquierda y derecha. 
    y = radio * np.cos(lat_grid) * np.sin(lon_grid)   # Y eje adelante o atras.
    z = radio * np.sin(lat_grid)                        # Z es la altura. o arriba abajo
    
    
    # el atributo shape[] consulta las dimensiones, filas y columnas.
    
    # Aplanamos la lista de vertices
    vertices = np.stack(
        [x.ravel(), # ravel() es un metodo que devuelve los elementos en una dimensión
        y.ravel(),
        z.ravel()],
        axis=1
        
    )
    
    # Guardamos también la información de anillos para mapear después
    anillo_por_vertice = np.repeat(np.arange(n_lat), n_lon)
    
    if limpiar_polos:
        # Elimina puntos demasiado cerca de los polos (donde se amontonan)
        # Mantén solo los anillos que no estén en los extremos extremos
        mascara = (anillo_por_vertice > 2) & (anillo_por_vertice < n_lat - 3)
        vertices = vertices[mascara]
        anillo_por_vertice = anillo_por_vertice[mascara]
        
    
    
    
    return vertices, anillo_por_vertice, n_lat, n_lon # Sahpe (resolucion * resolucion, 3)





def mapear_magnitudes_a_esfera(vertices, anillo_por_vertice, magnitudes, n_lat, factor_altura=0.35, suavizado=True):
    """
    Responsabilidad: modificar la geometría de la esfera en función de las magnitudes de audio.
    
    Copia los vértices originales para no modificarlos directamente.
    Divide los vértices en un máximo de 20 grupos.
    Normaliza cada magnitud respecto a la magnitud máxima de la ventana.
    Calcula un desplazamiento proporcional mediante factor_altura.
    Aplica ese desplazamiento exclusivamente a la coordenada Z de cada grupo.
    
    
    Retorna:
        vertices_modificados (numpy array): Copia con Z alterado según el audio
    """
    
    
    vertices_mod = vertices.copy()
    
    colores = np.zeros((len(vertices), 3))
    
    if len(magnitudes) == 0:
        # Color base si no hay audio
        colores[:] = [0.05, 0.9, 0.15]
        return vertices_mod, colores
    
    # Normalizar magnitudes (0-1)
    max_mag = np.max(magnitudes) + 1e-10
    magnitudes_norm = np.array(magnitudes) / max_mag
    
    
    # Cuántos anillos le tocan a cada banda
    n_bandas = min(len(magnitudes_norm), n_lat)
    anillos_por_banda = n_lat / n_bandas
    
    
    # Opcional: suavizar un poco las magnitudes para que no sea tan "escalonado"
    if suavizado and len(magnitudes_norm) > 3:
        from scipy.ndimage import gaussian_filter1d
        magnitudes_norm = gaussian_filter1d(magnitudes_norm, sigma=1.2)
        
    for i in range(n_bandas):
        inicio_anillo = int(i * anillos_por_banda)
        fin_anillo = int((i + 1) * anillos_por_banda)
        
        # Máscara de los vértices que pertenecen a estos anillos
        mascara = (anillo_por_vertice >= inicio_anillo) & (anillo_por_vertice < fin_anillo)
        
        if not np.any(mascara):
            continue
        
        mag = magnitudes_norm[i]
        empuje = factor_altura * mag
        
        # --- Deformación radial ---
        normales = vertices[mascara]
        normas = np.linalg.norm(normales, axis=1, keepdims=True) + 1e-10
        direcciones = normales / normas
        
        # Empujar hacia afuera
        vertices_mod[mascara] += direcciones * empuje
        
        # --- Color según magnitud ---
        # Base verde → se vuelve más cian/amarillo y brillante cuando hay más energía
        # Puedes cambiar esta fórmula a tu gusto
        r = 0.02 + mag * 0.60
        g = 0.85 + mag * 0.15
        b = 0.35 + mag * 0.65          
        
        colores[mascara] = [r, g, b]
    
    # Asegurar rango 0-1
    colores = np.clip(colores, 0, 1)
    
    return vertices_mod, colores
        
        
def renderizar_a_frame_rgb(
    vertices,
    colores=None,
    color_fallback=(0.05, 1.0, 0.12),
    ancho=1200, alto=676, 
    fondo=None,  # ruta a imagen de fondo o None
    glow=True
):

    """
    Responsabilidad: transformar los vértices 3D en un fotograma RGB que pueda recibir FFmpeg.

        Construye una figura 3D mediante go.Scatter3d
        Representa los vértices como puntos de color verde.
        Configura el fondo, oculta los ejes y fija la cámara.
        Renderiza la escena como PNG en memoria mediante to_image().
        Convierte la imagen a RGB utilizando Pillow y NumPy.
        Ajusta las dimensiones para devolver un array de forma (676, 1200, 3) con tipo uint8.

        Retorna:
            Esta función no genera el video ni analiza el audio. Produce una imagen estática de la geometría recibida.
    """
   
    
    
    dpi = 100
    fig = Figure(figsize=(ancho / dpi, alto / dpi), dpi=dpi, facecolor='#0a0a0c')
    
    canvas = FigureCanvasAgg(fig)
    
    
    # Crear el sistema de coordenadas 3D.
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor('#0a0a0c')
    
    # Si no nos pasan colores, usamos el color fijo
    if colores is None:
        colores = color_fallback
    
    # === Capa de glow (puntos grandes y transparentes) ===
    if glow:
        ax.scatter(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            s=7,            # ←--- más grande pero con poca opacidad
            c=colores,
            alpha=0.12,      # <--- Más transparente
            depthshade=False,
            linewidths=0,
            edgecolors='none'
        )

    
    
    # === Capa principal (puntos nítidos) ===
    ax.scatter(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        s=2.1,                  # ← CLAVE: puntos pequeños para nitidez
        c=colores,
        alpha=0.92,
        depthshade=False,
        linewidths=0,
        edgecolors='none'
    )
    
    
    
    
    # Límites y cámara
    lim = 1.20
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    
    ax.set_box_aspect((1, 1, 1))
    
    # Ángulo similar al de la foto (un poco más frontal)
    ax.view_init(elev=22, azim=28)
    
    ax.set_axis_off()
    ax.set_position([0, 0, 1, 1])
    
    canvas.draw()
    
    # Extraer RGB
    imagen_rgba = np.asarray(canvas.buffer_rgba())
    fotograma = imagen_rgba[:, :, :3].copy()
    fig.clear()
    
    # === Post-procesado: Bloom / Glow real ===
    if glow:
        img = Image.fromarray(fotograma)
        
        # Capa de glow
        glow_img = img.filter(ImageFilter.GaussianBlur(radius=4.5))
        glow_img = ImageEnhance.Brightness(glow_img).enhance(1.6)
        
        # Mezclar (additive-ish)
        fotograma = np.clip(
            np.array(img).astype(np.float32) * 0.94 + np.array(glow_img).astype(np.float32) * 0.28, 0, 255
        ).astype(np.uint8)
        
        # === Fondo (opcional) ===
        if fondo is not None:
            bg = Image.open(fondo).convert('RGB').resize((ancho, alto))
            pass
        
    return fotograma
    
    
    

        
    


