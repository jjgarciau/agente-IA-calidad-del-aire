"""Evaluación reproducible del agente con preguntas de prueba fijas.

Para cada pregunta de eval/preguntas.csv se ejecuta el agente y se registra de forma automática:

  herramienta_correcta  ¿llamó a la(s) herramienta(s) esperada(s)?
                        "a+b" exige todas, "a|b" acepta cualquiera, "ninguna" exige no llamar ninguna.
  recupero_fuente       ¿search_docs devolvió un documento esperado?
  cito_fuente           ¿la respuesta trae una cita [Fuente: ...] de un documento esperado?
  errores_herramienta   cuántas llamadas devolvieron error (casos límite)

La columna respuesta_correcta se deja vacía para la revisión manual de cada respuesta
(si ya estaba diligenciada en una corrida anterior, se conserva).

    python -m eval.run_eval
    python -m eval.run_eval --ids 7 8 9      # solo algunas preguntas
"""
from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd

from src import config
from src.agent import Resultado, run_agent
from src.tools import Recursos

CARPETA = Path(__file__).resolve().parent
PREGUNTAS = CARPETA / "preguntas.csv"
RESULTADOS = CARPETA / "resultados.csv"
RESUMEN = CARPETA / "resultados.md"
TRAZAS = CARPETA / "trazas.json"


def _plano(texto: str) -> str:
    sin_tildes = "".join(c for c in unicodedata.normalize("NFKD", str(texto)) if not unicodedata.combining(c))
    return sin_tildes.lower()


def herramienta_correcta(esperada: str, llamadas: list[str]) -> bool:
    esperada = str(esperada).strip()
    if esperada == "ninguna":
        return not llamadas
    if "+" in esperada:
        return all(h.strip() in llamadas for h in esperada.split("+"))
    return any(h.strip() in llamadas for h in esperada.split("|"))


def _claves(fuente_esperada) -> list[str]:
    if pd.isna(fuente_esperada) or not str(fuente_esperada).strip():
        return []
    return [_plano(c.strip()) for c in str(fuente_esperada).split("|") if c.strip()]


def recupero_fuente(fuente_esperada, resultado: Resultado) -> bool | None:
    claves = _claves(fuente_esperada)
    if not claves:
        return None
    recuperadas = _plano(" ".join(f"{f['titulo']} {f['archivo']}" for f in resultado.fuentes))
    return any(c in recuperadas for c in claves)


def cito_fuente(fuente_esperada, respuesta: str) -> bool | None:
    """True si alguna cita [Fuente: ...] de la respuesta menciona un documento esperado."""
    claves = _claves(fuente_esperada)
    if not claves:
        return None
    citas = _plano(" ".join(re.findall(r"\[Fuente:[^\]]*\]", respuesta)))
    return any(c in citas for c in claves)


def evaluar(preguntas: pd.DataFrame, recursos: Recursos | None = None, pausa: float = 2.0, **kwargs) -> tuple[pd.DataFrame, list[dict]]:
    recursos = recursos or Recursos()
    filas, trazas = [], []
    for _, q in preguntas.iterrows():
        estacion = None if pd.isna(q["estacion"]) or not str(q["estacion"]).strip() else str(q["estacion"]).strip()
        fecha = None if pd.isna(q["fecha"]) or not str(q["fecha"]).strip() else str(q["fecha"]).strip()
        inicio = time.time()
        r = run_agent(q["pregunta"], recursos, estacion=estacion, fecha=fecha, **kwargs)
        filas.append(
            {
                "id": q["id"], "tipo": q["tipo"], "pregunta": q["pregunta"], "estacion": estacion or "", "fecha": fecha or "",
                "herramienta_esperada": q["herramienta_esperada"],
                "herramientas_llamadas": " > ".join(r.herramientas) or "ninguna",
                "herramienta_correcta": herramienta_correcta(q["herramienta_esperada"], r.herramientas),
                "fuente_esperada": "" if pd.isna(q["fuente_esperada"]) else q["fuente_esperada"],
                "recupero_fuente": recupero_fuente(q["fuente_esperada"], r),
                "cito_fuente": cito_fuente(q["fuente_esperada"], r.respuesta),
                "errores_herramienta": sum(1 for t in r.traza if t["error"]),
                "iteraciones": r.iteraciones,
                "limite_alcanzado": r.limite_alcanzado,
                "error_llm": r.error or "",
                "segundos": round(time.time() - inicio, 1),
                "respuesta": r.respuesta,
                "respuesta_correcta": "",
                "notas": "",
            }
        )
        trazas.append({"id": int(q["id"]), "pregunta": q["pregunta"], "respuesta": r.respuesta, "traza": r.traza})
        marca = "ok " if filas[-1]["herramienta_correcta"] else "NO "
        print(f"[{q['id']:>2}] {marca} {filas[-1]['herramientas_llamadas']:<40} {q['pregunta'][:60]}")
        time.sleep(pausa)
    return pd.DataFrame(filas), trazas


def resumen_markdown(df: pd.DataFrame) -> str:
    con_fuente = df[df["cito_fuente"].notna()]
    lineas = [
        "# Evaluación del agente",
        "",
        f"Corrida del {datetime.now():%Y-%m-%d %H:%M} con el modelo `{config.LLM_MODEL}` y T_max = {config.T_MAX}.",
        "",
        f"- Preguntas: **{len(df)}**",
        f"- Herramienta correcta: **{int(df['herramienta_correcta'].sum())}/{len(df)}** ({df['herramienta_correcta'].mean():.0%})",
    ]
    if len(con_fuente):
        lineas += [
            f"- Recuperó un documento esperado: **{int(con_fuente['recupero_fuente'].sum())}/{len(con_fuente)}**",
            f"- Citó un documento esperado: **{int(con_fuente['cito_fuente'].sum())}/{len(con_fuente)}**",
        ]
    lineas += [
        f"- Iteraciones promedio: **{df['iteraciones'].mean():.1f}** · alcanzaron T_max: **{int(df['limite_alcanzado'].sum())}**",
        "",
        "| # | Tipo | Pregunta | Herramientas llamadas | Herramienta correcta | Citó fuente |",
        "|---|---|---|---|---|---|",
    ]
    for _, f in df.iterrows():
        cita = "—" if pd.isna(f["cito_fuente"]) else ("sí" if f["cito_fuente"] else "no")
        lineas.append(f"| {f['id']} | {f['tipo']} | {f['pregunta']} | {f['herramientas_llamadas']} | {'sí' if f['herramienta_correcta'] else 'no'} | {cita} |")
    return "\n".join(lineas) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids", nargs="*", type=int, help="evaluar solo estas preguntas")
    p.add_argument("--pausa", type=float, default=2.0, help="segundos entre preguntas (límite de peticiones)")
    a = p.parse_args()

    preguntas = pd.read_csv(PREGUNTAS)
    if a.ids:
        preguntas = preguntas[preguntas["id"].isin(a.ids)]
    df, trazas = evaluar(preguntas, pausa=a.pausa)

    if RESULTADOS.exists():                       # conservar la revisión manual de corridas anteriores
        previo = pd.read_csv(RESULTADOS, dtype=str).set_index("id")
        for columna in ("respuesta_correcta", "notas"):
            if columna in previo.columns:
                df[columna] = df["id"].astype(str).map(previo[columna]).fillna("")
        if a.ids:                                 # corrida parcial: se actualizan solo esas filas
            resto = pd.read_csv(RESULTADOS)
            df = pd.concat([resto[~resto["id"].isin(a.ids)], df], ignore_index=True).sort_values("id")
            trazas_previas = json.loads(TRAZAS.read_text(encoding="utf-8")) if TRAZAS.exists() else []
            trazas = sorted([t for t in trazas_previas if t["id"] not in a.ids] + trazas, key=lambda t: t["id"])

    df.to_csv(RESULTADOS, index=False, encoding="utf-8")
    TRAZAS.write_text(json.dumps(trazas, ensure_ascii=False, indent=1), encoding="utf-8")
    RESUMEN.write_text(resumen_markdown(df), encoding="utf-8")
    print("\n" + "\n".join(resumen_markdown(df).splitlines()[4:9]))
    print(f"\nGuardado: {RESULTADOS}\n          {RESUMEN}\n          {TRAZAS}")
    print("Revisa cada respuesta y completa la columna respuesta_correcta (sí/no) en resultados.csv.")


if __name__ == "__main__":
    main()
