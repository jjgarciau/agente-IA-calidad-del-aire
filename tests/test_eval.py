import json

import pandas as pd
import pytest

from eval import run_eval
from src import config, preprocess, train
from src.agent import Resultado
from src.tools import Recursos
from tests.test_agente import BuscadorFalso, ClienteFalso, _respuesta


def test_criterio_de_herramienta():
    assert run_eval.herramienta_correcta("get_pm25", ["get_pm25"])
    assert not run_eval.herramienta_correcta("get_pm25", ["search_docs"])
    assert run_eval.herramienta_correcta("predict_tomorrow+search_docs", ["predict_tomorrow", "search_docs"])
    assert not run_eval.herramienta_correcta("predict_tomorrow+search_docs", ["predict_tomorrow"])
    assert run_eval.herramienta_correcta("get_pm25|list_stations", ["list_stations"])
    assert run_eval.herramienta_correcta("ninguna", [])
    assert not run_eval.herramienta_correcta("ninguna", ["get_pm25"])


def test_criterio_de_cita():
    assert run_eval.cito_fuente("2254|ICA", "El límite es 37 [Fuente: Resolución 2254 de 2017, p. 3].") is True
    assert run_eval.cito_fuente("POECA", "El límite es 37 [Fuente: Resolución 2254 de 2017, p. 3].") is False
    assert run_eval.cito_fuente("2254", "La Resolución 2254 dice 37, pero sin formato de cita.") is False
    assert run_eval.cito_fuente("", "cualquier cosa") is None
    assert run_eval.cito_fuente(float("nan"), "cualquier cosa") is None

    r = Resultado("x", traza=[{"herramienta": "search_docs", "error": False, "resultado": {"resultados": [
        {"titulo": "Plan de acción POECA", "archivo": "poeca_plan_accion.pdf", "pagina": 4}]}}])
    assert run_eval.recupero_fuente("POECA|2254", r) is True
    assert run_eval.recupero_fuente("OMS", r) is False


def test_las_preguntas_son_validas():
    preguntas = pd.read_csv(run_eval.PREGUNTAS)
    assert 10 <= len(preguntas) <= 15 and preguntas["id"].is_unique
    validas = {"get_pm25", "predict_tomorrow", "search_docs", "list_stations", "ninguna"}
    for esperada in preguntas["herramienta_esperada"]:
        assert set(esperada.replace("+", "|").split("|")) <= validas


def test_evaluacion_de_punta_a_punta(proyecto):
    modelo, metricas = train.entrenar(preprocess.cargar_diario())
    modelo.guardar(config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(metricas), encoding="utf-8")
    preguntas = pd.read_csv(run_eval.PREGUNTAS)
    preguntas = preguntas[preguntas["id"].isin([1, 7])]
    cliente = ClienteFalso([
        _respuesta(llamadas=[("get_pm25", '{"estacion": "TRÁFICO CENTRO", "fecha": "2025-03-12"}')]),
        _respuesta("Fue de 30 µg/m³."),
        _respuesta(llamadas=[("search_docs", '{"query": "nivel máximo permisible PM2.5 24 horas"}')]),
        _respuesta("Es 37 µg/m³ [Fuente: Resolución 2254 de 2017, p. 3]."),
    ])
    df, trazas = run_eval.evaluar(preguntas, Recursos(buscador=BuscadorFalso()), pausa=0, client=cliente)
    assert list(df["herramienta_correcta"]) == [True, True]
    assert pd.isna(df.iloc[0]["cito_fuente"]) and df.iloc[1]["cito_fuente"] == True  # noqa: E712
    assert len(trazas) == 2 and trazas[1]["traza"][0]["herramienta"] == "search_docs"
    texto = run_eval.resumen_markdown(df)
    assert "Herramienta correcta: **2/2**" in texto and "Citó un documento esperado: **1/1**" in texto
