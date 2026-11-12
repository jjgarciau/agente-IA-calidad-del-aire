"""Índice de Calidad del Aire (ICA) para PM2.5 de 24 horas.

Puntos de corte de la Resolución 2254 de 2017 del Ministerio de Ambiente y
Desarrollo Sostenible (art. 20), tal como los resume la hoja metodológica del
ICA del IDEAM. La norma define los rangos con enteros (0-12, 13-37, ...), por
lo que el promedio diario se redondea al entero más cercano antes de clasificar.
"""
from __future__ import annotations

import math

# (nombre, límite superior en µg/m³, color estándar del ICA)
CATEGORIAS = [
    ("Buena", 12, "#00B050"),
    ("Aceptable", 37, "#F2C80F"),
    ("Dañina para grupos sensibles", 55, "#FF7E00"),
    ("Dañina", 150, "#FF0000"),
    ("Muy dañina", 250, "#8F3F97"),
    ("Peligrosa", 500, "#7E0023"),
]
NOMBRES = [c[0] for c in CATEGORIAS]
COLORES = {c[0]: c[2] for c in CATEGORIAS}
LIMITES = [c[1] for c in CATEGORIAS]

# Clases que predice el modelo: las tres categorías más altas son muy raras en
# promedios diarios, así que se agrupan en una sola.
CLASES = ["Buena", "Aceptable", "Dañina para grupos sensibles", "Dañina o peor"]
COLORES_CLASE = {
    "Buena": COLORES["Buena"],
    "Aceptable": COLORES["Aceptable"],
    "Dañina para grupos sensibles": COLORES["Dañina para grupos sensibles"],
    "Dañina o peor": COLORES["Dañina"],
}
# Desde esta clase en adelante se considera "episodio" (lo que interesa alertar).
CLASE_EPISODIO = 2


def _redondear(pm25: float) -> int:
    """Redondeo aritmético (0.5 hacia arriba), no el bancario de round()."""
    return int(math.floor(pm25 + 0.5))


def categoria(pm25: float) -> str:
    """Categoría ICA (6 niveles) de un promedio de PM2.5 de 24 horas en µg/m³."""
    if pm25 is None or (isinstance(pm25, float) and math.isnan(pm25)) or pm25 < 0:
        raise ValueError(f"PM2.5 inválido: {pm25!r}")
    valor = _redondear(pm25)
    for nombre, limite, _ in CATEGORIAS:
        if valor <= limite:
            return nombre
    return CATEGORIAS[-1][0]


def clase(pm25: float) -> int:
    """Índice de la clase del modelo (0..3) para un promedio de PM2.5 de 24 horas."""
    return min(NOMBRES.index(categoria(pm25)), len(CLASES) - 1)


def nombre_clase(indice: int) -> str:
    return CLASES[int(indice)]
