import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.naive_bayes import CategoricalNB

from src import config, features, ica, preprocess, train
from src.model import (
    NaiveBayesCategorico,
    exactitud,
    f1_macro,
    matriz_confusion,
    precision_recall_binario,
    predecir_mayoritaria,
)


def _datos_aleatorios(n=3000, semilla=0):
    rng = np.random.default_rng(semilla)
    n_cat = [7, 4, 3, 3, 3, 7]
    y = rng.integers(0, 4, n)
    # Las variables dependen de la clase para que haya algo que aprender.
    X = np.column_stack([np.clip(y * (v // 4 + 1) + rng.integers(0, max(v // 2, 2), n), 0, v - 1) for v in n_cat])
    return X, y, n_cat


@pytest.mark.parametrize("alpha", [0.1, 1.0, 10.0])
def test_coincide_con_sklearn(alpha):
    X, y, n_cat = _datos_aleatorios()
    propio = NaiveBayesCategorico(n_cat, 4, alpha).fit(X[:2000], y[:2000])
    referencia = CategoricalNB(alpha=alpha, min_categories=n_cat).fit(X[:2000], y[:2000])
    np.testing.assert_allclose(propio.predict_proba(X[2000:]), referencia.predict_proba(X[2000:]), atol=1e-10)
    assert (propio.predict(X[2000:]) == referencia.predict(X[2000:])).all()


def test_probabilidades_validas_y_ejemplo_a_mano():
    # Dos clases, una variable binaria: P(x=1|y=0)=(0+1)/(2+2)=0.25, P(x=1|y=1)=(2+1)/(2+2)=0.75
    m = NaiveBayesCategorico([2], 2, alpha=1.0).fit(np.array([[0], [0], [1], [1]]), np.array([0, 0, 1, 1]))
    p = m.predict_proba(np.array([[1]]))
    assert p == pytest.approx(np.array([[0.25, 0.75]]))
    assert p.sum() == pytest.approx(1.0)


def test_guardar_y_cargar(tmp_path):
    X, y, n_cat = _datos_aleatorios(500)
    m = NaiveBayesCategorico(n_cat, 4, 1.0).fit(X, y)
    m.guardar(tmp_path / "m.json", extra={"clases": ica.CLASES})
    cargado, extra = NaiveBayesCategorico.cargar(tmp_path / "m.json")
    np.testing.assert_allclose(cargado.predict_proba(X), m.predict_proba(X))
    assert extra["clases"] == ica.CLASES


def test_clase_ausente_no_rompe():
    m = NaiveBayesCategorico([3], 4, 1.0).fit(np.array([[0], [1], [2], [1]]), np.array([0, 1, 1, 0]))
    p = m.predict_proba(np.array([[2]]))
    assert p[0, 2] == 0 and p[0, 3] == 0 and p.sum() == pytest.approx(1.0)


def test_metricas_coinciden_con_sklearn():
    rng = np.random.default_rng(1)
    y, pred = rng.integers(0, 4, 500), rng.integers(0, 4, 500)
    assert exactitud(y, pred) == pytest.approx(accuracy_score(y, pred))
    assert f1_macro(y, pred, 4) == pytest.approx(f1_score(y, pred, average="macro"))
    assert (matriz_confusion(y, pred, 4) == confusion_matrix(y, pred, labels=range(4))).all()
    precision, recall = precision_recall_binario(y, pred, 2)
    assert recall == pytest.approx(((y >= 2) & (pred >= 2)).sum() / (y >= 2).sum())
    assert precision == pytest.approx(((y >= 2) & (pred >= 2)).sum() / (pred >= 2).sum())
    assert (predecir_mayoritaria(np.array([1, 1, 0]), 4) == 1).all()


def test_variables_sin_fuga_del_futuro(proyecto):
    diario = preprocess.cargar_diario()
    tabla = features.construir_tabla(diario)
    assert (tabla[features.NOMBRES].to_numpy() >= 0).all()
    for nombre, n in zip(features.NOMBRES, features.N_CATEGORIAS):
        assert tabla[nombre].max() < n

    # Una fila concreta: sus variables solo dependen de hoy y de ayer, y el objetivo es mañana.
    fila = tabla[tabla["y"].notna()].iloc[200]
    serie = diario[diario["estacion"] == fila["estacion"]].set_index("fecha")["pm25"]
    assert fila["pm25_hoy"] == serie[fila["fecha"]]
    assert fila["pm25_ayer"] == serie[fila["fecha"] - pd.Timedelta(days=1)]
    assert fila["fecha_objetivo"] == fila["fecha"] + pd.Timedelta(days=1)
    assert int(fila["y"]) == ica.clase(serie[fila["fecha_objetivo"]])
    assert int(fila["clase_hoy"]) == ica.clase(fila["pm25_hoy"])


def test_clase_vectorizada_igual_a_ica():
    valores = np.arange(0, 80, 0.1)
    assert (features.clase_de(valores) == [ica.clase(v) for v in valores]).all()


def test_entrenamiento_completo(proyecto):
    modelo, metricas = train.entrenar(preprocess.cargar_diario())
    r = metricas["resultados"]
    assert set(r) == {"Clase mayoritaria", "Persistencia", "Naive Bayes (propio)"}
    assert metricas["particion"]["entrenamiento"]["hasta"] < config.TEST_DESDE <= metricas["particion"]["prueba"]["desde"]
    assert 0 <= r["Naive Bayes (propio)"]["exactitud"] <= 1
    assert r["Naive Bayes (propio)"]["exactitud"] > r["Clase mayoritaria"]["exactitud"]
    assert len(metricas["ablacion"]) == len(features.NOMBRES)
    assert np.array(r["Persistencia"]["matriz_confusion"]).sum() == metricas["particion"]["prueba"]["n"]
