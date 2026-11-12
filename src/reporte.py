"""Genera RESULTADOS.md a partir de los archivos que producen el entrenamiento y la evaluación.

Así los números del README y del informe salen siempre de una corrida real y no se copian a mano.

    python -m src.reporte
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src import config, ica, preprocess


def _tabla(filas: list[list], encabezado: list[str]) -> list[str]:
    out = ["| " + " | ".join(encabezado) + " |", "|" + "---|" * len(encabezado)]
    out += ["| " + " | ".join(str(c) for c in fila) + " |" for fila in filas]
    return out


def seccion_datos() -> list[str]:
    diario, estaciones = preprocess.cargar_diario(), preprocess.cargar_estaciones()
    conteo = diario["categoria"].value_counts().reindex(ica.NOMBRES, fill_value=0)
    lineas = [
        "## Datos",
        "",
        f"- Periodo: **{diario['fecha'].min():%Y-%m-%d} a {diario['fecha'].max():%Y-%m-%d}**",
        f"- Estaciones: **{len(estaciones)}** · días-estación con promedio válido: **{len(diario):,}**",
        f"- PM2.5 diario: media {diario['pm25'].mean():.1f} µg/m³, mediana {diario['pm25'].median():.1f}, máximo {diario['pm25'].max():.1f}",
        "",
    ]
    lineas += _tabla([[n, f"{c:,}", f"{c / len(diario):.1%}"] for n, c in conteo.items()], ["Categoría ICA", "Días-estación", "Fracción"])
    return lineas + [""]


def seccion_modelo() -> list[str]:
    if not config.METRICS_PATH.exists():
        return ["## Modelo", "", "Aún no hay métricas. Ejecuta `python -m src.train`.", ""]
    m = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    p = m["particion"]
    lineas = [
        "## Modelo de la Parte 1: Naive Bayes frente a las líneas base",
        "",
        f"- Entrenamiento: {p['entrenamiento']['n']:,} días-estación ({p['entrenamiento']['desde']} a {p['entrenamiento']['hasta']})",
        f"- Prueba: {p['prueba']['n']:,} días-estación ({p['prueba']['desde']} a {p['prueba']['hasta']}), nunca vistos al entrenar",
        f"- Clases en prueba: " + ", ".join(f"{k} {v:,}" for k, v in p["prueba"]["clases"].items()),
        f"- Suavizado α = {m['alpha']} (elegido validando en el último año del entrenamiento)",
        "",
    ]
    lineas += _tabla(
        [[k, f"{v['exactitud']:.3f}", f"{v['f1_macro']:.3f}", f"{v['precision_episodio']:.3f}", f"{v['recall_episodio']:.3f}"] for k, v in m["resultados"].items()],
        ["Modelo", "Exactitud", "F1-macro", "Precisión de episodio", "Recall de episodio"],
    )
    lineas += ["", "«Episodio» = Dañina para grupos sensibles o peor.", "", "Matriz de confusión del Naive Bayes (filas = clase real, columnas = clase predicha):", ""]
    matriz = m["resultados"]["Naive Bayes (propio)"]["matriz_confusion"]
    lineas += _tabla([[m["clases"][i], *fila] for i, fila in enumerate(matriz)], ["Real \\ Predicha", *m["clases"]])
    lineas += ["", "Ablación (Naive Bayes sin una variable):", ""]
    lineas += _tabla([[a["sin_variable"], f"{a['exactitud']:.3f}", f"{a['f1_macro']:.3f}", f"{a['recall_episodio']:.3f}"] for a in m["ablacion"]],
                     ["Variable retirada", "Exactitud", "F1-macro", "Recall de episodio"])
    return lineas + [""]


def seccion_agente() -> list[str]:
    ruta = config.ROOT / "eval" / "resultados.csv"
    if not ruta.exists():
        return ["## Agente", "", "Aún no se ha corrido la evaluación. Ejecuta `python -m eval.run_eval`.", ""]
    df = pd.read_csv(ruta)
    con_fuente = df[df["cito_fuente"].notna()]
    lineas = [
        "## Agente: preguntas de prueba",
        "",
        f"- Preguntas: **{len(df)}**",
        f"- Llamó a la herramienta esperada: **{int(df['herramienta_correcta'].sum())}/{len(df)}**",
    ]
    if len(con_fuente):
        lineas += [
            f"- Recuperó un documento esperado: **{int(con_fuente['recupero_fuente'].sum())}/{len(con_fuente)}**",
            f"- Citó un documento esperado: **{int(con_fuente['cito_fuente'].sum())}/{len(con_fuente)}**",
        ]
    revisadas = df[df["respuesta_correcta"].notna() & (df["respuesta_correcta"].astype(str).str.strip() != "")]
    if len(revisadas):
        correctas = revisadas["respuesta_correcta"].astype(str).str.strip().str.lower().isin(["sí", "si", "s", "true", "1"]).sum()
        lineas.append(f"- Respuesta correcta según la revisión manual: **{int(correctas)}/{len(revisadas)}**")
    lineas += [f"- Iteraciones promedio: **{df['iteraciones'].mean():.1f}** · alcanzaron T_max: **{int(df['limite_alcanzado'].sum())}**", ""]
    lineas += _tabla(
        [[f["id"], f["tipo"], f["pregunta"], f["herramientas_llamadas"], "sí" if f["herramienta_correcta"] else "no",
          "—" if pd.isna(f["cito_fuente"]) else ("sí" if f["cito_fuente"] else "no")] for _, f in df.iterrows()],
        ["#", "Tipo", "Pregunta", "Herramientas llamadas", "Herramienta correcta", "Citó fuente"],
    )
    return lineas + ["", "El detalle de cada respuesta está en `eval/resultados.csv` y las trazas completas en `eval/trazas.json`.", ""]


def generar(destino: Path | None = None) -> Path:
    destino = Path(destino or config.ROOT / "RESULTADOS.md")
    lineas = ["# Resultados", "", f"Generado el {datetime.now():%Y-%m-%d %H:%M} con `python -m src.reporte` a partir de los archivos de la última corrida.", ""]
    lineas += seccion_datos() + seccion_modelo() + seccion_agente()
    destino.write_text("\n".join(lineas), encoding="utf-8")
    return destino


if __name__ == "__main__":
    print(f"Guardado: {generar()}")
