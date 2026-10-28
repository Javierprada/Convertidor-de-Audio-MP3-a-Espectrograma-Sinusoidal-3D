# Responsabilidad: diseñar y construir la malla.
"""
Generar las coordenadas de la malla 3D.
Definir las líneas, filas y columnas geométricas.
Gestionar el patrón sinusoidal y sus parámetros.
Proporcionar los datos geométricos necesarios para representar la superficie.
"""

"""Construcción y actualización geométrica de la malla espectral 3D.

Este módulo no analiza audio ni genera imágenes. Recibe un eje 3D de Matplotlib
para crear las líneas de la malla y actualizar sus coordenadas Z.
"""

import numpy as np
from datetime import date

def color_degradado(posicion: float) -> np.ndarray:
    """Devuelve un color RGB interpolado de cian a morado y magenta.

    Args:
        posicion: valor esperado entre 0 y 1.

    Returns:
        Vector NumPy RGB con valores entre 0 y 1.
    """
    posicion = float(np.clip(posicion, 0.0, 1.0))
    cian = np.array([0.0, 0.85, 1.0])
    morado = np.array([0.55, 0.15, 1.0])
    magenta = np.array([1.0, 0.0, 0.65])

    if posicion <= 0.5:
        factor = posicion * 2
        return cian * (1 - factor) + morado * factor

    factor = (posicion - 0.5) * 2
    return morado * (1 - factor) + magenta * factor


class MallaSinusoidal3D:
    """Crea las líneas de una malla temporal de frecuencias y actualiza su relieve.

    La geometría base usa:
      - X: posiciones de las bandas de frecuencia.
      - Y: posiciones de las ventanas FFT en el historial.
      - Z: magnitudes procesadas del audio.

    Nota: esta malla representa un espectrograma 3D en forma de rejilla; no es
    una esfera geométrica.
    """

    def __init__(self, ax, eje_x, eje_profundidad):
        self.ax = ax
        self.eje_x = np.asarray(eje_x, dtype=float)
        self.eje_profundidad = np.asarray(eje_profundidad, dtype=float)

        if self.eje_x.ndim != 1 or self.eje_x.size == 0:
            raise ValueError("eje_x debe ser un vector no vacío.")
        if self.eje_profundidad.ndim != 1 or self.eje_profundidad.size == 0:
            raise ValueError("eje_profundidad debe ser un vector no vacío.")

        self.profundidad = len(self.eje_profundidad)
        self.cantidad_frecuencias = len(self.eje_x)
        self.x, self.y = np.meshgrid(self.eje_x, self.eje_profundidad)
        self.lineas = []
        self._crear_lineas()

    def _crear_lineas(self):
        """Crea una sola vez las líneas horizontales y verticales de la malla."""
        for fila in range(self.profundidad):
            color = color_degradado(fila / max(self.profundidad - 1, 1))
            es_linea_principal = fila % 5 == 0

            linea, = self.ax.plot(
                self.x[fila],
                self.y[fila],
                np.zeros(self.cantidad_frecuencias),
                color="#35ff5d" if es_linea_principal else color,
                linewidth=1.5 if es_linea_principal else 0.45,
                alpha=1.0 if es_linea_principal else 0.45,
            )
            self.lineas.append(("fila", fila, linea))

        for indice_columna, posicion in enumerate(self.eje_x):
            linea, = self.ax.plot(
                [posicion] * self.profundidad,
                self.eje_profundidad,
                np.zeros(self.profundidad),
                color="#00FFF2",
                linewidth=0.55,
                alpha=0.5,
            )
            self.lineas.append(("columna", indice_columna, linea))

    def actualizar(self, superficie):
        """Actualiza las alturas Z de la malla con una superficie 2D.

        Args:
            superficie: matriz con forma (profundidad, cantidad_frecuencias).
        """
        superficie = np.asarray(superficie, dtype=float)
        forma_esperada = (self.profundidad, self.cantidad_frecuencias)
        if superficie.shape != forma_esperada:
            raise ValueError(
                f"La superficie debe tener forma {forma_esperada}; "
                f"se recibió {superficie.shape}."
            )

        for tipo, indice, linea in self.lineas:
            if tipo == "fila":
                linea.set_data_3d(
                    self.x[indice], self.y[indice], superficie[indice]
                )
            else:
                linea.set_data_3d(
                    self.x[:, indice], self.y[:, indice], superficie[:, indice]
                )
