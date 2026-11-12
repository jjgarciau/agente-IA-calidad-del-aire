"""Variables del modelo: todo lo que se conoce el día t para predecir la clase ICA del día t+1.

Todas las variables son categóricas, que es lo que necesita un Naive Bayes categórico:

  nivel_hoy    PM2.5 de hoy en la estación, en 7 niveles (más fino que la categoría ICA)
  clase_ayer   clase ICA de ayer en la estación
  tendencia    cambio de PM2.5 entre ayer y hoy (baja / estable / sube, umbral de 3 µg/m³)
  temporada    época del año del día objetivo (feb-abr y oct-nov son las de episodios en el valle)
  dia_manana   tipo de día objetivo (laboral / sábado / domingo), por el tráfico
  nivel_valle  promedio de PM2.5 de todas las estaciones hoy, en los mismos 7 niveles
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import ica

# Bordes de los niveles de PM2.5. 12.5, 37.5 y 55.5 coinciden con los cortes del ICA
# (la norma usa enteros: 12.5 ya se redondea a 13, "Aceptable").
BORDES_NIVEL = [12.5, 20.0, 28.0, 37.5, 45.0, 55.5]
ETIQUETAS_NIVEL = ["0-12", "13-19", "20-27", "28-37", "38-44", "45-55", "56 o más"]
BORDES_CLASE = [12.5, 37.5, 55.5]
UMBRAL_TENDENCIA = 3.0

VARIABLES: dict[str, list[str]] = {
    "nivel_hoy": ETIQUETAS_NIVEL,
    "clase_ayer": ica.CLASES,
    "tendencia": ["baja", "estable", "sube"],
    "temporada": ["resto del año", "feb-abr", "oct-nov"],
    "dia_manana": ["laboral", "sábado", "domingo"],
    "nivel_valle": ETIQUETAS_NIVEL,
}
NOMBRES = list(VARIABLES)
N_CATEGORIAS = [len(v) for v in VARIABLES.values()]


def clase_de(pm25) -> np.ndarray:
    """Clase del modelo (0..3) para valores de PM2.5; vectorizada y coherente con ica.clase."""
    return np.digitize(np.asarray(pm25, dtype=float), BORDES_CLASE)


def construir_tabla(diario: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (estación, día t) con las variables codificadas y el objetivo del día t+1.

    Solo se incluyen los días con dato válido hoy y ayer. La columna `y` queda vacía
    cuando no hay dato del día siguiente (sirve para predecir, no para entrenar).
    """
    ancho = diario.pivot(index="fecha", columns="estacion", values="pm25").sort_index()
    ancho = ancho.reindex(pd.date_range(ancho.index.min(), ancho.index.max(), freq="D"))
    valle = ancho.mean(axis=1)

    objetivo = ancho.index + pd.Timedelta(days=1)
    mes = objetivo.month
    temporada = np.where(np.isin(mes, [2, 3, 4]), 1, np.where(np.isin(mes, [10, 11]), 2, 0))
    dia = np.where(objetivo.day_of_week < 5, 0, np.where(objetivo.day_of_week == 5, 1, 2))

    partes = []
    for estacion in ancho.columns:
        hoy = ancho[estacion]
        ayer = hoy.shift(1)
        manana = hoy.shift(-1)
        diferencia = hoy - ayer
        parte = pd.DataFrame(
            {
                "estacion": estacion,
                "fecha": ancho.index,
                "fecha_objetivo": objetivo,
                "pm25_hoy": hoy.to_numpy(),
                "pm25_ayer": ayer.to_numpy(),
                "pm25_manana": manana.to_numpy(),
                "pm25_valle": valle.to_numpy(),
                "temporada": temporada,
                "dia_manana": dia,
                "diferencia": diferencia.to_numpy(),
            }
        )
        partes.append(parte.dropna(subset=["pm25_hoy", "pm25_ayer"]))
    t = pd.concat(partes, ignore_index=True)

    t["nivel_hoy"] = np.digitize(t["pm25_hoy"], BORDES_NIVEL)
    t["clase_hoy"] = clase_de(t["pm25_hoy"])
    t["clase_ayer"] = clase_de(t["pm25_ayer"])
    t["tendencia"] = np.where(t["diferencia"] < -UMBRAL_TENDENCIA, 0, np.where(t["diferencia"] > UMBRAL_TENDENCIA, 2, 1))
    t["nivel_valle"] = np.digitize(t["pm25_valle"], BORDES_NIVEL)
    t["y"] = np.where(t["pm25_manana"].notna(), clase_de(t["pm25_manana"].fillna(0)), np.nan)
    return t.drop(columns="diferencia")


def matriz_X(tabla: pd.DataFrame, variables: list[str] | None = None) -> np.ndarray:
    return tabla[variables or NOMBRES].to_numpy(dtype=int)


def describir(fila: pd.Series) -> dict[str, str]:
    """Valores de las variables en palabras, para explicar una predicción."""
    return {nombre: VARIABLES[nombre][int(fila[nombre])] for nombre in NOMBRES}
