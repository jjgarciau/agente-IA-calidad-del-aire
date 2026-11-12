"""Configuración central del proyecto: rutas, parámetros y cliente del LLM.

Todo lo que cambia entre máquinas (clave de API, modelo) se lee de variables de
entorno o del archivo `.env` en la raíz del repositorio. Nada sensible queda en el código.

Uso rápido para comprobar la conexión con el LLM:

    python -m src.config --ping
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Raíz del repositorio. AIRE_ROOT permite apuntar a otra carpeta (lo usan las pruebas).
ROOT = Path(os.getenv("AIRE_ROOT") or Path(__file__).resolve().parents[1])
load_dotenv(ROOT / ".env")

# --- Rutas -----------------------------------------------------------------
DATA_RAW = ROOT / "data" / "raw"                      # CSV horarios por mes (no se suben)
DAILY_CSV = ROOT / "data" / "processed" / "pm25_diario.csv"
STATIONS_CSV = ROOT / "data" / "stations.csv"
MODEL_PATH = ROOT / "models" / "naive_bayes.json"
METRICS_PATH = ROOT / "models" / "metricas.json"
DOCS_DIR = ROOT / "docs"                              # PDFs del corpus RAG
SOURCES_CSV = DOCS_DIR / "fuentes.csv"
INDEX_DIR = ROOT / "index"

# --- Datos -----------------------------------------------------------------
DATASET_ID = "g4t8-zkc3"   # "Calidad del Aire en Colombia" (SISAIRE) en datos.gov.co
SOCRATA_URL = f"https://www.datos.gov.co/resource/{DATASET_ID}.csv"
AUTORIDAD = "AMVA"         # Área Metropolitana del Valle de Aburrá
VARIABLE = "PM2.5"
FECHA_INICIO = "2022-01-01"
FECHA_FIN = "2025-12-31"
HORAS_MINIMAS = 18         # un día es válido con >= 75 % de las 24 horas
DIAS_MINIMOS_ESTACION = 180
TEST_DESDE = "2025-01-01"  # partición temporal: se entrena antes y se prueba desde aquí

# --- LLM y RAG -------------------------------------------------------------
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("GROQ_API_KEY", "") or os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
T_MAX = int(os.getenv("AGENT_T_MAX", "5"))            # máximo de iteraciones del loop agéntico

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
RAG_TOP_K = 4
RAG_UMBRAL = float(os.getenv("RAG_UMBRAL", "0.30"))   # similitud coseno mínima para aceptar un fragmento


def hay_api_key() -> bool:
    return bool(LLM_API_KEY.strip())


def get_client():
    """Cliente OpenAI-compatible apuntando al proveedor configurado (Groq por defecto)."""
    from openai import OpenAI

    if not hay_api_key():
        raise RuntimeError(
            "Falta GROQ_API_KEY. Copia .env.example como .env y pega tu clave "
            "(https://console.groq.com/keys)."
        )
    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, timeout=60, max_retries=0)


def _ping() -> int:
    """Hace una llamada mínima al LLM para verificar clave, endpoint y modelo."""
    print(f"Endpoint: {LLM_BASE_URL}")
    print(f"Modelo:   {LLM_MODEL}")
    try:
        client = get_client()
        r = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "Responde únicamente: listo"}],
            max_tokens=200,
        )
        print("Respuesta:", (r.choices[0].message.content or "").strip())
        print("OK: el LLM responde.")
        return 0
    except Exception as e:  # noqa: BLE001 - queremos mostrar cualquier causa
        print(f"ERROR: {type(e).__name__}: {e}")
        try:
            disponibles = sorted(m.id for m in get_client().models.list().data)
            print("Modelos disponibles con tu clave:")
            for m in disponibles:
                print("  -", m)
            print("Ajusta GROQ_MODEL en .env con uno que soporte tool calling.")
        except Exception:  # noqa: BLE001
            pass
        return 1


if __name__ == "__main__":
    if "--ping" in sys.argv:
        raise SystemExit(_ping())
    print(f"Raíz del proyecto: {ROOT}")
    print(f"API key configurada: {'sí' if hay_api_key() else 'no'}")
