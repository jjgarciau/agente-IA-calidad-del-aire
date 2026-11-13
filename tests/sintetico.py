"""Generador de datos SINTÉTICOS con el mismo esquema que la API de datos.gov.co.

Solo se usa en las pruebas automáticas (carpetas temporales). Los resultados del
proyecto se calculan siempre con los datos reales descargados por src.download_data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ESTACIONES = [
    ("101", "TRÁFICO CENTRO", "Medellín", 6.252709, -75.569656, 8.0),
    ("102", "VILLAHERMOSA", "Medellín", 6.258909, -75.548261, 2.0),
    ("103", "CASA JUSTICIA", "Itagüí", 6.185667, -75.597206, 4.0),
    ("104", "SANTA ELENA", "Medellín", 6.236365, -75.498347, -8.0),
]


def _sin_punto_decimal(df: pd.DataFrame, estacion: str, hasta: str) -> pd.DataFrame:
    """Reproduce el defecto real de la fuente: la estación pierde el punto decimal antes de `hasta`.

    Los dos primeros días de cada mes vienen con un decimal (39.9 -> "399") y el resto con dos (29.49 -> "2949").
    """
    df = df.copy()
    afectadas = (df["nombre_est"] == estacion) & (df["med_fecha_inicio"] < hasta)
    valores = df.loc[afectadas, "med_concentracion_estandar"].astype(float)
    primeros = df.loc[afectadas, "med_fecha_inicio"].str[8:10].isin(["01", "02"])
    df.loc[afectadas, "med_concentracion_estandar"] = [
        (f"{v:.1f}" if p else f"{v:.2f}").replace(".", "") for v, p in zip(valores, primeros)
    ]
    return df


def generar_crudo(desde: str = "2023-01-01", hasta: str = "2025-12-31", semilla: int = 7, corrupta_hasta: str | None = None) -> pd.DataFrame:
    """Mediciones horarias con estacionalidad (picos en feb-abr y oct-nov) y persistencia diaria.

    Con `corrupta_hasta`, la estación SANTA ELENA pierde el punto decimal antes de esa fecha.
    """
    rng = np.random.default_rng(semilla)
    dias = pd.date_range(desde, hasta, freq="D")
    estacional = np.where(dias.month.isin([2, 3, 4]), 14.0, np.where(dias.month.isin([10, 11]), 6.0, 0.0))
    finde = np.where(dias.day_of_week >= 5, -3.0, 0.0)
    # Componente regional autocorrelacionada (AR(1)) compartida por todas las estaciones.
    regional = np.zeros(len(dias))
    for i in range(1, len(dias)):
        regional[i] = 0.8 * regional[i - 1] + rng.normal(0, 4.0)

    filas = []
    for est_id, nombre, municipio, lat, lon, sesgo in ESTACIONES:
        base = 20.0 + sesgo + estacional + finde + regional + rng.normal(0, 2.0, len(dias))
        for dia, media in zip(dias, base):
            horas = np.arange(24)
            ciclo = 4.0 * np.sin((horas - 7) / 24 * 2 * np.pi)
            valores = np.clip(media + ciclo + rng.normal(0, 2.5, 24), 1.0, None)
            presentes = rng.random(24) > 0.04            # ~4 % de horas faltantes
            if rng.random() < 0.02:                      # ~2 % de días con muy pocas horas
                presentes[:] = False
                presentes[:10] = True
            for h in horas[presentes]:
                filas.append((est_id, nombre, municipio, lat, lon, "Fija",
                              f"{dia:%Y-%m-%d} {h:02d}:00:00", f"{valores[h]:.2f}"))
    df = pd.DataFrame(filas, columns=["estacion_id", "nombre_est", "municipio", "latitud", "longitud",
                                      "tipo_estacion", "med_fecha_inicio", "med_concentracion_estandar"])
    if corrupta_hasta:
        df = _sin_punto_decimal(df, "SANTA ELENA", corrupta_hasta)
    # Ruido realista: un valor centinela, un duplicado y una estación con un solo dato.
    df.loc[5, "med_concentracion_estandar"] = "2554"
    df = pd.concat([df, df.iloc[[10]]], ignore_index=True)
    extra = df.iloc[[0]].assign(nombre_est="POLITÉCNICO JIC (S)", estacion_id="999")
    return pd.concat([df, extra], ignore_index=True).astype(str)


def escribir_crudo(carpeta: Path, **kw) -> pd.DataFrame:
    """Escribe el crudo sintético repartido por mes, como lo hace la descarga real."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    df = generar_crudo(**kw)
    for mes, parte in df.groupby(df["med_fecha_inicio"].str[:7]):
        parte.to_csv(carpeta / f"pm25_{mes}.csv", index=False, encoding="utf-8")
    return df
