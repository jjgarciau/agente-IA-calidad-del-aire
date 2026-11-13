"""Descarga los datos horarios de PM2.5 del Valle de Aburrá desde datos.gov.co.

Fuente: conjunto "Calidad del Aire en Colombia" (SISAIRE / IDEAM), id g4t8-zkc3,
filtrado por autoridad ambiental AMVA y variable PM2.5. La API es Socrata (SoQL).

Se descarga un archivo por mes en data/raw/ para que la descarga sea reanudable:
si se corta, al volver a ejecutar solo se piden los meses que faltan.

    python -m src.download_data                       # 2022-01-01 a 2025-12-31
    python -m src.download_data --desde 2024-01-01    # rango más corto (plan B)
    python -m src.download_data --forzar              # vuelve a bajar todo
"""
from __future__ import annotations

import argparse
import io
import os
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from src import config

COLUMNAS = [
    "estacion_id",
    "nombre_est",
    "municipio",
    "latitud",
    "longitud",
    "tipo_estacion",
    "med_fecha_inicio",
    "med_concentracion_estandar",
]
LIMITE_PAGINA = 50_000


def meses(desde: date, hasta: date) -> list[tuple[date, date]]:
    """Lista de (primer día del mes, primer día del mes siguiente) que cubre el rango."""
    out = []
    actual = date(desde.year, desde.month, 1)
    while actual <= hasta:
        siguiente = date(actual.year + (actual.month == 12), actual.month % 12 + 1, 1)
        out.append((actual, siguiente))
        actual = siguiente
    return out


def _pedir(session: requests.Session, params: dict, intentos: int = 4) -> str:
    """GET con reintentos y espera exponencial (la API a veces responde 429/5xx)."""
    ultimo_error: Exception | None = None
    for i in range(intentos):
        try:
            r = session.get(config.SOCRATA_URL, params=params, timeout=120)
            if r.status_code == 200:
                return r.text
            ultimo_error = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except requests.RequestException as e:
            ultimo_error = e
        espera = 2 ** (i + 1)
        print(f"    reintento en {espera}s ({ultimo_error})")
        time.sleep(espera)
    raise RuntimeError(f"La API no respondió tras {intentos} intentos: {ultimo_error}")


def descargar_mes(session: requests.Session, ini: date, fin: date) -> pd.DataFrame:
    """Todas las mediciones horarias de PM2.5 del AMVA con inicio en [ini, fin)."""
    where = (
        f"nombre_fgda='{config.AUTORIDAD}' AND msfl_code='{config.VARIABLE}' "
        f"AND med_fecha_inicio >= '{ini.isoformat()}' AND med_fecha_inicio < '{fin.isoformat()}'"
    )
    partes, offset = [], 0
    while True:
        texto = _pedir(
            session,
            {
                "$select": ",".join(COLUMNAS),
                "$where": where,
                "$order": ":id",          # orden estable para paginar
                "$limit": LIMITE_PAGINA,
                "$offset": offset,
            },
        )
        if not texto.strip():
            break
        df = pd.read_csv(io.StringIO(texto), dtype=str)
        if df.empty:
            break
        partes.append(df)
        if len(df) < LIMITE_PAGINA:
            break
        offset += LIMITE_PAGINA
    if not partes:
        return pd.DataFrame(columns=COLUMNAS)
    return pd.concat(partes, ignore_index=True)


def descargar(desde: date, hasta: date, destino: Path | None = None, forzar: bool = False) -> int:
    destino = Path(destino or config.DATA_RAW)
    destino.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    token = os.getenv("SOCRATA_APP_TOKEN", "").strip()
    if token:
        session.headers["X-App-Token"] = token

    total = 0
    for ini, fin in meses(desde, hasta):
        archivo = destino / f"pm25_{ini:%Y-%m}.csv"
        if archivo.exists() and not forzar:
            with open(archivo, encoding="utf-8") as f:
                n = sum(1 for _ in f) - 1
            print(f"{ini:%Y-%m}: ya existe ({n:,} filas), se omite")
            total += max(n, 0)
            continue
        df = descargar_mes(session, ini, fin)
        if df.empty:
            print(f"{ini:%Y-%m}: sin datos")
            continue
        df.to_csv(archivo, index=False, encoding="utf-8", lineterminator="\n")
        total += len(df)
        print(f"{ini:%Y-%m}: {len(df):,} filas, {df['nombre_est'].nunique()} estaciones")
    return total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--desde", default=config.FECHA_INICIO, help="AAAA-MM-DD")
    p.add_argument("--hasta", default=config.FECHA_FIN, help="AAAA-MM-DD")
    p.add_argument("--forzar", action="store_true", help="vuelve a descargar meses existentes")
    a = p.parse_args()

    print(f"Fuente: {config.SOCRATA_URL}")
    print(f"Filtro: autoridad={config.AUTORIDAD}, variable={config.VARIABLE}, {a.desde} a {a.hasta}\n")
    total = descargar(date.fromisoformat(a.desde), date.fromisoformat(a.hasta), forzar=a.forzar)
    print(f"\nTotal: {total:,} mediciones horarias en {config.DATA_RAW}")
    if total == 0:
        raise SystemExit("No se descargó ningún dato. Revisa la conexión o el rango de fechas.")
    print("Siguiente paso: python -m src.preprocess")


if __name__ == "__main__":
    main()
