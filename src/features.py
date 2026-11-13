"""Variables del modelo: todo lo que se conoce el día t para predecir la clase ICA del día t+1.

Todas las variables son categóricas, que es lo que necesita un Naive Bayes categórico:

  nivel_hoy     PM2.5 de hoy en la estación, en 14 niveles (más finos cerca de los cortes del ICA)
  nivel_semana  promedio de PM2.5 de la estación en los últimos 7 días (incluido hoy), en los mismos niveles
  dia_manana    tipo de día objetivo (laboral / sábado / domingo), por el tráfico
  cambio_valle  cambio del promedio de todas las estaciones entre ayer y hoy (5 niveles)

El conjunto se eligió en el año de validación (2024), nunca en el de prueba. Frente a la primera
versión se retiraron la clase de ayer, la tendencia de la estación, la temporada y el nivel del valle:
repetían la información de nivel_hoy y, como Naive Bayes cuenta cada variable como evidencia
independiente, empeoraban la predicción (ver la ablación en notebooks/02_modelo.ipynb).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import ica

# Bordes de los niveles de PM2.5. 12.5, 37.5 y 55.5 coinciden con los cortes del ICA
# (la norma usa enteros: 12.5 ya se redondea a 13, "Aceptable").
BORDES_NIVEL = [8.0, 10.5, 12.5, 14.5, 17.0, 20.0, 24.0, 28.0, 33.0, 37.5, 42.0, 48.0, 55.5]
ETIQUETAS_NIVEL = ["0-7", "8-10", "11-12", "13-14", "15-16", "17-19", "20-23", "24-27", "28-32", "33-37", "38-41", "42-47", "48-55", "56 o más"]
BORDES_CLASE = [12.5, 37.5, 55.5]
BORDES_CAMBIO = [-4.0, -1.5, 1.5, 4.0]
DIAS_SEMANA = 7

VARIABLES: dict[str, list[str]] = {
    "nivel_hoy": ETIQUETAS_NIVEL,
    "nivel_semana": ETIQUETAS_NIVEL,
    "dia_manana": ["laboral", "sábado", "domingo"],
    "cambio_valle": ["baja fuerte", "baja", "estable", "sube", "sube fuerte"],
}
NOMBRES = list(VARIABLES)
N_CATEGORIAS = [len(v) for v in VARIABLES.values()]


def clase_de(pm25) -> np.ndarray:
    """Clase del modelo (0..3) para valores de PM2.5; vectorizada y coherente con ica.clase."""
    return np.digitize(np.asarray(pm25, dtype=float), BORDES_CLASE)


def construir_tabla(diario: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (estación, día t) con las variables codificadas y el objetivo del día t+1.

    Solo se incluyen los días con dato válido en la estación. La columna `y` queda vacía
    cuando no hay dato del día siguiente (sirve para predecir, no para entrenar).
    """
    ancho = diario.pivot(index="fecha", columns="estacion", values="pm25").sort_index()
    ancho = ancho.reindex(pd.date_range(ancho.index.min(), ancho.index.max(), freq="D"))
    valle = ancho.mean(axis=1)                                   # promedio de las estaciones con dato ese día
    cambio_valle = (valle - valle.shift(1)).fillna(0.0)           # sin dato de ayer se asume estable

    objetivo = ancho.index + pd.Timedelta(days=1)
    dia = np.where(objetivo.day_of_week < 5, 0, np.where(objetivo.day_of_week == 5, 1, 2))

    partes = []
    for estacion in ancho.columns:
        hoy = ancho[estacion]
        parte = pd.DataFrame(
            {
                "estacion": estacion,
                "fecha": ancho.index,
                "fecha_objetivo": objetivo,
                "pm25_hoy": hoy.to_numpy(),
                "pm25_semana": hoy.rolling(DIAS_SEMANA, min_periods=1).mean().round(1).to_numpy(),   # solo días pasados y hoy
                "pm25_manana": hoy.shift(-1).to_numpy(),
                "pm25_valle": valle.round(1).to_numpy(),
                "cambio_valle_ug": cambio_valle.round(1).to_numpy(),
                "dia_manana": dia,
            }
        )
        partes.append(parte.dropna(subset=["pm25_hoy"]))
    t = pd.concat(partes, ignore_index=True)

    t["nivel_hoy"] = np.digitize(t["pm25_hoy"], BORDES_NIVEL)
    t["nivel_semana"] = np.digitize(t["pm25_semana"], BORDES_NIVEL)
    t["cambio_valle"] = np.digitize(t["cambio_valle_ug"], BORDES_CAMBIO)
    t["clase_hoy"] = clase_de(t["pm25_hoy"])
    t["y"] = np.where(t["pm25_manana"].notna(), clase_de(t["pm25_manana"].fillna(0)), np.nan)
    return t


def matriz_X(tabla: pd.DataFrame, variables: list[str] | None = None) -> np.ndarray:
    return tabla[variables or NOMBRES].to_numpy(dtype=int)


def describir(fila: pd.Series) -> dict[str, str]:
    """Valores de las variables en palabras, para explicar una predicción."""
    return {
        "nivel_hoy": f"{VARIABLES['nivel_hoy'][int(fila['nivel_hoy'])]} µg/m³",
        "nivel_semana": f"{VARIABLES['nivel_semana'][int(fila['nivel_semana'])]} µg/m³ (promedio de los últimos 7 días)",
        "dia_manana": VARIABLES["dia_manana"][int(fila["dia_manana"])],
        "cambio_valle": f"{VARIABLES['cambio_valle'][int(fila['cambio_valle'])]} ({fila['cambio_valle_ug']:+.1f} µg/m³ frente a ayer en el promedio del valle)",
    }
