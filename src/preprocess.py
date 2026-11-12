"""Convierte las mediciones horarias en promedios diarios de PM2.5 con su categoría ICA.

Reglas de limpieza (todas quedan documentadas en el informe):
  1. Se descartan valores no numéricos, negativos o mayores a 500 µg/m³ (tope de la escala ICA;
     en los datos aparecen valores centinela como 2554).
  2. Se eliminan mediciones duplicadas de la misma estación y hora.
  3. Un día solo es válido si tiene al menos 18 de 24 horas medidas (75 %).
  4. Se conservan las estaciones con al menos 180 días válidos.

    python -m src.preprocess
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config, ica

PM25_MAX_HORARIO = 500.0


def cargar_crudos(carpeta: Path | None = None) -> pd.DataFrame:
    carpeta = Path(carpeta or config.DATA_RAW)
    archivos = sorted(carpeta.glob("pm25_*.csv"))
    if not archivos:
        raise FileNotFoundError(
            f"No hay archivos pm25_*.csv en {carpeta}. Ejecuta primero: python -m src.download_data"
        )
    return pd.concat((pd.read_csv(a, dtype=str) for a in archivos), ignore_index=True)


def limpiar_horario(crudo: pd.DataFrame) -> pd.DataFrame:
    """Tipos correctos, sin valores imposibles ni duplicados."""
    df = pd.DataFrame(
        {
            "estacion": crudo["nombre_est"].astype(str).str.strip(),
            "municipio": crudo["municipio"].astype(str).str.strip(),
            "latitud": pd.to_numeric(crudo["latitud"], errors="coerce"),
            "longitud": pd.to_numeric(crudo["longitud"], errors="coerce"),
            "tipo": crudo["tipo_estacion"].astype(str).str.strip(),
            "fecha_hora": pd.to_datetime(crudo["med_fecha_inicio"], errors="coerce"),
            "pm25": pd.to_numeric(crudo["med_concentracion_estandar"], errors="coerce"),
        }
    )
    df = df.dropna(subset=["fecha_hora", "pm25"])
    df = df[(df["pm25"] >= 0) & (df["pm25"] <= PM25_MAX_HORARIO)]
    df = df.drop_duplicates(subset=["estacion", "fecha_hora"], keep="first")
    return df.reset_index(drop=True)


def promedio_diario(
    horario: pd.DataFrame,
    horas_minimas: int = config.HORAS_MINIMAS,
    dias_minimos: int = config.DIAS_MINIMOS_ESTACION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (tabla diaria, tabla de estaciones)."""
    h = horario.assign(fecha=horario["fecha_hora"].dt.normalize())
    diario = (
        h.groupby(["estacion", "fecha"], as_index=False)
        .agg(pm25=("pm25", "mean"), horas_validas=("pm25", "size"))
    )
    diario = diario[diario["horas_validas"] >= horas_minimas].copy()

    conteo = diario.groupby("estacion")["fecha"].agg(dias_validos="size", primer_dia="min", ultimo_dia="max")
    conteo = conteo[conteo["dias_validos"] >= dias_minimos]
    diario = diario[diario["estacion"].isin(conteo.index)].copy()

    diario["pm25"] = diario["pm25"].round(1)
    diario["categoria"] = diario["pm25"].map(ica.categoria)

    # Metadatos de cada estación: se toma el valor más frecuente por si hay variaciones.
    def moda(s: pd.Series):
        m = s.dropna().mode()
        return m.iloc[0] if len(m) else None

    meta = (
        horario[horario["estacion"].isin(conteo.index)]
        .groupby("estacion")
        .agg(municipio=("municipio", moda), latitud=("latitud", moda), longitud=("longitud", moda), tipo=("tipo", moda))
    )
    estaciones = meta.join(conteo).reset_index().sort_values("estacion")
    estaciones["primer_dia"] = estaciones["primer_dia"].dt.strftime("%Y-%m-%d")
    estaciones["ultimo_dia"] = estaciones["ultimo_dia"].dt.strftime("%Y-%m-%d")

    diario = diario.merge(estaciones[["estacion", "municipio"]], on="estacion", how="left")
    diario = diario[["fecha", "estacion", "municipio", "pm25", "horas_validas", "categoria"]]
    diario = diario.sort_values(["estacion", "fecha"]).reset_index(drop=True)
    return diario, estaciones.reset_index(drop=True)


def cargar_diario(ruta: Path | None = None) -> pd.DataFrame:
    """Lee la tabla diaria ya procesada (la usan el modelo, las herramientas y el dashboard)."""
    ruta = Path(ruta or config.DAILY_CSV)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta: python -m src.download_data y luego python -m src.preprocess"
        )
    return pd.read_csv(ruta, parse_dates=["fecha"])


def cargar_estaciones(ruta: Path | None = None) -> pd.DataFrame:
    ruta = Path(ruta or config.STATIONS_CSV)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. Ejecuta: python -m src.preprocess")
    return pd.read_csv(ruta)


def main() -> None:
    crudo = cargar_crudos()
    print(f"Mediciones horarias leídas: {len(crudo):,}")
    horario = limpiar_horario(crudo)
    print(f"Tras la limpieza:           {len(horario):,} ({len(crudo) - len(horario):,} descartadas)")

    diario, estaciones = promedio_diario(horario)
    config.DAILY_CSV.parent.mkdir(parents=True, exist_ok=True)
    diario.to_csv(config.DAILY_CSV, index=False, date_format="%Y-%m-%d", encoding="utf-8")
    estaciones.to_csv(config.STATIONS_CSV, index=False, encoding="utf-8")

    print(f"\nEstaciones conservadas: {len(estaciones)}")
    print(f"Días-estación válidos:  {len(diario):,}  ({diario['fecha'].min():%Y-%m-%d} a {diario['fecha'].max():%Y-%m-%d})")
    print("\nDistribución de categorías ICA:")
    dist = diario["categoria"].value_counts().reindex(ica.NOMBRES, fill_value=0)
    for nombre, n in dist.items():
        print(f"  {nombre:<30} {n:>7,}  {n / len(diario):6.1%}")
    print(f"\nGuardado: {config.DAILY_CSV}\n          {config.STATIONS_CSV}")
    print("Siguiente paso: python -m src.train")


if __name__ == "__main__":
    main()
