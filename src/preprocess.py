"""Convierte las mediciones horarias en promedios diarios de PM2.5 con su categoría ICA.

Reglas de limpieza (todas quedan documentadas en el informe):
  1. Se descartan valores no numéricos, negativos o mayores a 500 µg/m³ (tope de la escala ICA;
     en los datos aparecen valores centinela como 2554).
  2. Se eliminan mediciones duplicadas de la misma estación y hora.
  3. Un día solo es válido si tiene al menos 18 de 24 horas medidas (75 %).
  4. Se conservan las estaciones con al menos 180 días válidos.
  5. Se descartan los meses-estación con la escala corrupta. En la fuente, 14 estaciones tienen
     en 2022 y 2023 el punto decimal perdido (39.9 aparece como 399 y 29.49 como 2949). La regla
     1 elimina los valores de dos decimales, pero los de un decimal sobreviven multiplicados por
     10. Como no se puede saber qué factor aplicar a cada valor, no se corrigen: se descartan.
     Para detectarlos se compara cada estación con la red de referencia del día (mediana de las
     estaciones con al menos 300 días válidos ese año, que no pueden estar afectadas porque el
     defecto destruye la cobertura). Un mes-estación se descarta si su nivel relativo a la red
     supera 2.5 veces el nivel relativo habitual de esa estación. Comparar contra la red del
     mismo día cancela la estacionalidad: en los datos sanos la mayor razón es 1.5 y en los
     corruptos la menor es 3.4. Lo descartado queda en data/processed/descartes_escala.csv.

    python -m src.preprocess
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config, ica

PM25_MAX_HORARIO = 500.0
RAZON_ESCALA = 2.5           # nivel relativo del mes / nivel relativo habitual a partir del cual el mes se descarta
DIAS_REFERENCIA = 300        # días válidos en un año para que una estación sirva de referencia ese año
MIN_REFERENCIAS = 3          # estaciones de referencia necesarias para comparar un día


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


def agregar_diario(horario: pd.DataFrame, horas_minimas: int = config.HORAS_MINIMAS) -> pd.DataFrame:
    """Promedio de PM2.5 por estación y día, solo para los días con suficientes horas medidas."""
    h = horario.assign(fecha=horario["fecha_hora"].dt.normalize())
    diario = (
        h.groupby(["estacion", "fecha"], as_index=False)
        .agg(pm25=("pm25", "mean"), horas_validas=("pm25", "size"))
    )
    diario = diario[diario["horas_validas"] >= horas_minimas].copy()
    diario["pm25"] = diario["pm25"].round(1)
    return diario


def _con_dias_suficientes(diario: pd.DataFrame, dias_minimos: int) -> pd.DataFrame:
    conteo = diario.groupby("estacion")["fecha"].transform("size")
    return diario[conteo >= dias_minimos].copy()


def filtrar_escala_corrupta(diario: pd.DataFrame, razon: float = RAZON_ESCALA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quita los meses-estación con la escala corrupta (punto decimal perdido en la fuente).

    Devuelve (tabla limpia, registro de meses descartados).
    """
    anio = diario["fecha"].dt.year
    mes = diario["fecha"].dt.to_period("M").astype(str)

    # Red de referencia: mediana diaria de las estaciones con cobertura casi completa ese año.
    dias_en_el_anio = diario.groupby(["estacion", anio])["fecha"].transform("size")
    referencia = diario[dias_en_el_anio >= DIAS_REFERENCIA].groupby("fecha")["pm25"].agg(["median", "size"])
    red = referencia.loc[referencia["size"] >= MIN_REFERENCIAS, "median"]

    relativo = diario["pm25"] / diario["fecha"].map(red)                    # nivel de la estación frente a la red ese día
    habitual = relativo.groupby(diario["estacion"]).transform("median")     # nivel relativo habitual de la estación
    del_mes = relativo.groupby([diario["estacion"], mes]).transform("median")
    corrupto = (del_mes / habitual) > razon                                 # sin referencia -> NaN -> no se descarta

    descartes = (
        diario[corrupto]
        .assign(mes=mes[corrupto], razon=(del_mes / habitual)[corrupto])
        .groupby(["estacion", "mes"], as_index=False)
        .agg(dias_descartados=("pm25", "size"), mediana_pm25=("pm25", "median"), razon=("razon", "first"))
    )
    descartes["razon"] = descartes["razon"].round(1)
    return diario[~corrupto].copy(), descartes


def depurar_diario(diario: pd.DataFrame, dias_minimos: int = config.DIAS_MINIMOS_ESTACION) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reglas 4 y 5 sobre la tabla diaria: estaciones con suficientes días y sin meses de escala corrupta."""
    diario = _con_dias_suficientes(diario, dias_minimos)
    diario, descartes = filtrar_escala_corrupta(diario)
    diario = _con_dias_suficientes(diario, dias_minimos)        # una estación pudo quedar corta tras el descarte
    diario = diario.assign(categoria=diario["pm25"].map(ica.categoria))
    return diario.sort_values(["estacion", "fecha"]).reset_index(drop=True), descartes


def resumen_estaciones(diario: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Tabla de estaciones: metadatos (municipio, coordenadas, tipo) más la cobertura final."""
    conteo = diario.groupby("estacion")["fecha"].agg(dias_validos="size", primer_dia="min", ultimo_dia="max")
    estaciones = meta.set_index("estacion")[["municipio", "latitud", "longitud", "tipo"]].join(conteo, how="inner")
    estaciones = estaciones.reset_index().sort_values("estacion")
    estaciones["primer_dia"] = pd.to_datetime(estaciones["primer_dia"]).dt.strftime("%Y-%m-%d")
    estaciones["ultimo_dia"] = pd.to_datetime(estaciones["ultimo_dia"]).dt.strftime("%Y-%m-%d")
    return estaciones.reset_index(drop=True)


def promedio_diario(
    horario: pd.DataFrame,
    horas_minimas: int = config.HORAS_MINIMAS,
    dias_minimos: int = config.DIAS_MINIMOS_ESTACION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Devuelve (tabla diaria, tabla de estaciones, meses descartados por escala corrupta)."""
    diario, descartes = depurar_diario(agregar_diario(horario, horas_minimas), dias_minimos)

    # Metadatos de cada estación: se toma el valor más frecuente por si hay variaciones.
    def moda(s: pd.Series):
        m = s.dropna().mode()
        return m.iloc[0] if len(m) else None

    meta = (
        horario[horario["estacion"].isin(diario["estacion"].unique())]
        .groupby("estacion", as_index=False)
        .agg(municipio=("municipio", moda), latitud=("latitud", moda), longitud=("longitud", moda), tipo=("tipo", moda))
    )
    estaciones = resumen_estaciones(diario, meta)
    diario = diario.merge(estaciones[["estacion", "municipio"]], on="estacion", how="left")
    diario = diario[["fecha", "estacion", "municipio", "pm25", "horas_validas", "categoria"]]
    return diario, estaciones, descartes


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

    diario, estaciones, descartes = promedio_diario(horario)
    config.DAILY_CSV.parent.mkdir(parents=True, exist_ok=True)
    diario.to_csv(config.DAILY_CSV, index=False, date_format="%Y-%m-%d", encoding="utf-8", lineterminator="\n")
    estaciones.to_csv(config.STATIONS_CSV, index=False, encoding="utf-8", lineterminator="\n")
    descartes.to_csv(config.DISCARDS_CSV, index=False, encoding="utf-8", lineterminator="\n")

    if len(descartes):
        print(f"\nEscala corrupta: se descartaron {len(descartes)} meses-estación "
              f"({int(descartes['dias_descartados'].sum()):,} días) de {descartes['estacion'].nunique()} estaciones.")

    print(f"\nEstaciones conservadas: {len(estaciones)}")
    print(f"Días-estación válidos:  {len(diario):,}  ({diario['fecha'].min():%Y-%m-%d} a {diario['fecha'].max():%Y-%m-%d})")
    print("\nDistribución de categorías ICA:")
    dist = diario["categoria"].value_counts().reindex(ica.NOMBRES, fill_value=0)
    for nombre, n in dist.items():
        print(f"  {nombre:<30} {n:>7,}  {n / len(diario):6.1%}")
    print(f"\nGuardado: {config.DAILY_CSV}\n          {config.STATIONS_CSV}\n          {config.DISCARDS_CSV}")
    print("Siguiente paso: python -m src.train")


if __name__ == "__main__":
    main()
