"""Entrena el Naive Bayes y lo compara con dos líneas base en un periodo que no vio.

Protocolo experimental:
  1. Partición temporal: se entrena con los días objetivo anteriores a TEST_DESDE (2025-01-01)
     y se prueba con los posteriores. Nunca se mezcla al azar (se filtraría el futuro).
  2. El suavizado α se elige con una validación interna: el último año del entrenamiento.
  3. Con el α elegido se reentrena con todo el entrenamiento y se evalúa UNA vez en prueba.
  4. Ablación: se repite quitando una variable a la vez para ver cuánto aporta cada una.

    python -m src.train
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src import config, features, ica, preprocess
from src.model import NaiveBayesCategorico, evaluar, predecir_mayoritaria, predecir_persistencia

ALPHAS = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
K = len(ica.CLASES)


def _ajustar(datos: pd.DataFrame, alpha: float, variables: list[str] | None = None) -> NaiveBayesCategorico:
    variables = variables or features.NOMBRES
    n_cat = [len(features.VARIABLES[v]) for v in variables]
    return NaiveBayesCategorico(n_cat, K, alpha).fit(features.matriz_X(datos, variables), datos["y"].to_numpy(int))


def _evaluar(modelo: NaiveBayesCategorico, datos: pd.DataFrame, variables: list[str] | None = None) -> dict:
    pred = modelo.predict(features.matriz_X(datos, variables))
    return evaluar(datos["y"].to_numpy(int), pred, K, ica.CLASE_EPISODIO)


def elegir_alpha(entrenamiento: pd.DataFrame) -> tuple[float, list[dict]]:
    """Valida α en el último año del entrenamiento (criterio: F1-macro)."""
    corte = entrenamiento["fecha_objetivo"].max() - pd.Timedelta(days=365)
    ajuste, validacion = entrenamiento[entrenamiento["fecha_objetivo"] <= corte], entrenamiento[entrenamiento["fecha_objetivo"] > corte]
    if len(ajuste) < 100 or len(validacion) < 100:
        return 1.0, []
    resultados = []
    for alpha in ALPHAS:
        m = _evaluar(_ajustar(ajuste, alpha), validacion)
        resultados.append({"alpha": alpha, "f1_macro": m["f1_macro"], "exactitud": m["exactitud"]})
    mejor = max(resultados, key=lambda r: (r["f1_macro"], r["exactitud"]))
    return mejor["alpha"], resultados


def entrenar(diario: pd.DataFrame, test_desde: str = config.TEST_DESDE) -> tuple[NaiveBayesCategorico, dict]:
    tabla = features.construir_tabla(diario).dropna(subset=["y"])
    corte = pd.Timestamp(test_desde)
    entrenamiento, prueba = tabla[tabla["fecha_objetivo"] < corte], tabla[tabla["fecha_objetivo"] >= corte]
    if entrenamiento.empty or prueba.empty:
        raise ValueError(f"La partición en {test_desde} deja vacío el entrenamiento o la prueba.")

    alpha, validacion = elegir_alpha(entrenamiento)
    modelo = _ajustar(entrenamiento, alpha)
    y_prueba = prueba["y"].to_numpy(int)

    resultados = {
        "Clase mayoritaria": evaluar(y_prueba, predecir_mayoritaria(entrenamiento["y"], len(prueba)), K, ica.CLASE_EPISODIO),
        "Persistencia": evaluar(y_prueba, predecir_persistencia(prueba["clase_hoy"]), K, ica.CLASE_EPISODIO),
        "Naive Bayes (propio)": _evaluar(modelo, prueba),
    }

    ablacion = []
    for quitada in features.NOMBRES:
        resto = [v for v in features.NOMBRES if v != quitada]
        m = _evaluar(_ajustar(entrenamiento, alpha, resto), prueba, resto)
        ablacion.append({"sin_variable": quitada, "exactitud": m["exactitud"], "f1_macro": m["f1_macro"], "recall_episodio": m["recall_episodio"]})

    por_estacion = []
    pred = modelo.predict(features.matriz_X(prueba))
    for estacion, grupo in prueba.assign(pred=pred).groupby("estacion"):
        y = grupo["y"].to_numpy(int)
        por_estacion.append(
            {
                "estacion": estacion,
                "n": int(len(grupo)),
                "naive_bayes": round(float((grupo["pred"].to_numpy() == y).mean()), 4),
                "persistencia": round(float((grupo["clase_hoy"].to_numpy() == y).mean()), 4),
            }
        )

    def distribucion(d: pd.DataFrame) -> dict:
        conteo = np.bincount(d["y"].to_numpy(int), minlength=K)
        return {ica.CLASES[i]: int(conteo[i]) for i in range(K)}

    metricas = {
        "clases": ica.CLASES,
        "variables": features.VARIABLES,
        "particion": {
            "test_desde": test_desde,
            "entrenamiento": {"n": int(len(entrenamiento)), "desde": f"{entrenamiento['fecha_objetivo'].min():%Y-%m-%d}", "hasta": f"{entrenamiento['fecha_objetivo'].max():%Y-%m-%d}", "clases": distribucion(entrenamiento)},
            "prueba": {"n": int(len(prueba)), "desde": f"{prueba['fecha_objetivo'].min():%Y-%m-%d}", "hasta": f"{prueba['fecha_objetivo'].max():%Y-%m-%d}", "clases": distribucion(prueba)},
        },
        "alpha": alpha,
        "validacion_alpha": validacion,
        "resultados": resultados,
        "ablacion": ablacion,
        "por_estacion": por_estacion,
    }
    return modelo, metricas


def imprimir(metricas: dict) -> None:
    p = metricas["particion"]
    print(f"Entrenamiento: {p['entrenamiento']['n']:,} días-estación ({p['entrenamiento']['desde']} a {p['entrenamiento']['hasta']})")
    print(f"Prueba:        {p['prueba']['n']:,} días-estación ({p['prueba']['desde']} a {p['prueba']['hasta']})")
    print(f"Clases en prueba: {p['prueba']['clases']}")
    print(f"α elegido por validación: {metricas['alpha']}\n")
    print(f"{'Modelo':<24}{'Exactitud':>10}{'F1-macro':>10}{'Prec. epi.':>12}{'Recall epi.':>12}")
    for nombre, m in metricas["resultados"].items():
        print(f"{nombre:<24}{m['exactitud']:>10.3f}{m['f1_macro']:>10.3f}{m['precision_episodio']:>12.3f}{m['recall_episodio']:>12.3f}")
    print('\n("episodio" = Dañina para grupos sensibles o peor)')
    print("\nAblación (Naive Bayes sin una variable):")
    for a in metricas["ablacion"]:
        print(f"  sin {a['sin_variable']:<12} exactitud {a['exactitud']:.3f}  F1-macro {a['f1_macro']:.3f}  recall epi. {a['recall_episodio']:.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-desde", default=config.TEST_DESDE, help="AAAA-MM-DD: inicio del periodo de prueba")
    a = p.parse_args()

    modelo, metricas = entrenar(preprocess.cargar_diario(), a.test_desde)
    modelo.guardar(config.MODEL_PATH, extra={"clases": ica.CLASES, "variables": features.VARIABLES, "particion": metricas["particion"]})
    config.METRICS_PATH.write_text(json.dumps(metricas, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    imprimir(metricas)
    print(f"\nGuardado: {config.MODEL_PATH}\n          {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
