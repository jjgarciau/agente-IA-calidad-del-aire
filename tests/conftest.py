"""Fixtures compartidas: un proyecto temporal con datos sintéticos ya procesados."""
from __future__ import annotations

import pytest

from src import config, preprocess
from tests import sintetico


@pytest.fixture(scope="session")
def proyecto(tmp_path_factory):
    """Redirige las rutas de config a una carpeta temporal y deja datos diarios listos."""
    raiz = tmp_path_factory.mktemp("proyecto")
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "ROOT", raiz)
    mp.setattr(config, "DATA_RAW", raiz / "data" / "raw")
    mp.setattr(config, "DAILY_CSV", raiz / "data" / "processed" / "pm25_diario.csv")
    mp.setattr(config, "STATIONS_CSV", raiz / "data" / "stations.csv")
    mp.setattr(config, "DISCARDS_CSV", raiz / "data" / "processed" / "descartes_escala.csv")
    mp.setattr(config, "MODEL_PATH", raiz / "models" / "naive_bayes.json")
    mp.setattr(config, "METRICS_PATH", raiz / "models" / "metricas.json")
    mp.setattr(config, "DOCS_DIR", raiz / "docs")
    mp.setattr(config, "SOURCES_CSV", raiz / "docs" / "fuentes.csv")
    mp.setattr(config, "INDEX_DIR", raiz / "index")

    sintetico.escribir_crudo(config.DATA_RAW)
    horario = preprocess.limpiar_horario(preprocess.cargar_crudos())
    diario, estaciones, _ = preprocess.promedio_diario(horario)
    config.DAILY_CSV.parent.mkdir(parents=True, exist_ok=True)
    diario.to_csv(config.DAILY_CSV, index=False, date_format="%Y-%m-%d")
    estaciones.to_csv(config.STATIONS_CSV, index=False)
    yield raiz
    mp.undo()
