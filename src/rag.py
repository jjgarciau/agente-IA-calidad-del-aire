"""RAG sobre el corpus de documentos oficiales (carpeta docs/).

Flujo, adaptado del notebook de la Clase 10 del curso:
  1. Extraer el texto de cada PDF página por página (así cada fragmento conserva su página).
  2. Dividir cada página en fragmentos de ~500 caracteres que respetan los límites de oración.
  3. Calcular embeddings normalizados con un modelo multilingüe local (sentence-transformers).
  4. Guardar un índice FAISS de producto interno (equivale a similitud coseno).
  5. Al consultar: embedding de la pregunta -> k vecinos -> se descartan los que no superan un umbral.

El índice se construye una sola vez con `python -m src.build_index` y queda en index/.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from src import config

# El modelo MiniLM multilingüe trunca en 128 tokens (~500 caracteres en español):
# fragmentos más largos se indexarían incompletos.
MAX_CARACTERES = 500
EXTENSIONES = {".pdf", ".txt", ".md"}
ARCHIVO_INDICE = "faiss.index"
ARCHIVO_FRAGMENTOS = "chunks.json"


@dataclass
class Fragmento:
    id: str
    archivo: str
    titulo: str
    pagina: int
    texto: str


# ---------------------------------------------------------------------- lectura
def leer_fuentes(ruta: Path | None = None) -> dict[str, dict]:
    """docs/fuentes.csv -> {archivo: {titulo, entidad, url, ...}} para citar con un nombre legible."""
    ruta = Path(ruta or config.SOURCES_CSV)
    if not ruta.exists():
        return {}
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        return {fila["archivo"].strip(): fila for fila in csv.DictReader(f) if fila.get("archivo")}


def limpiar(texto: str) -> str:
    texto = texto.replace("­", "")
    texto = re.sub(r"-\s*\n\s*(?=[a-záéíóúñ])", "", texto)   # une palabras cortadas al final de línea
    return re.sub(r"\s+", " ", texto).strip()


def extraer_paginas(ruta: Path) -> list[tuple[int, str]]:
    """Lista de (número de página, texto limpio). Los .txt/.md cuentan como una sola página."""
    ruta = Path(ruta)
    if ruta.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        paginas = []
        for i, pagina in enumerate(PdfReader(str(ruta)).pages, start=1):
            try:
                texto = limpiar(pagina.extract_text() or "")
            except Exception:  # noqa: BLE001 - una página dañada no debe tumbar todo el documento
                texto = ""
            if texto:
                paginas.append((i, texto))
        return paginas
    texto = limpiar(ruta.read_text(encoding="utf-8", errors="ignore"))
    return [(1, texto)] if texto else []


def trocear(texto: str, max_caracteres: int = MAX_CARACTERES) -> list[str]:
    """Agrupa oraciones completas hasta el tamaño máximo, repitiendo la última como solape."""
    oraciones = [o.strip() for o in re.split(r"(?<=[.;:?!])\s+", texto) if o.strip()]
    piezas = []
    for o in oraciones:                       # una oración enorme (p. ej. una tabla) se parte por palabras
        while len(o) > max_caracteres:
            corte = o.rfind(" ", 0, max_caracteres)
            corte = corte if corte > max_caracteres // 2 else max_caracteres
            piezas.append(o[:corte].strip())
            o = o[corte:].strip()
        if o:
            piezas.append(o)

    fragmentos, actual = [], []
    for pieza in piezas:
        if actual and len(" ".join(actual)) + 1 + len(pieza) > max_caracteres:
            fragmentos.append(" ".join(actual))
            solape = actual[-1]
            actual = [solape] if len(solape) + 1 + len(pieza) <= max_caracteres else []
        actual.append(pieza)
    if actual:
        fragmentos.append(" ".join(actual))
    return [f for f in fragmentos if len(f) >= 40]   # descarta restos sin contenido (números de página, etc.)


def fragmentar_corpus(docs_dir: Path | None = None) -> tuple[list[Fragmento], list[dict]]:
    """Recorre docs/ y devuelve (fragmentos, resumen por archivo)."""
    docs_dir = Path(docs_dir or config.DOCS_DIR)
    fuentes = leer_fuentes(docs_dir / "fuentes.csv")
    fragmentos, resumen = [], []
    for ruta in sorted(docs_dir.iterdir()):
        if ruta.suffix.lower() not in EXTENSIONES or ruta.name.lower() == "readme.md":
            continue
        titulo = (fuentes.get(ruta.name, {}).get("titulo") or ruta.stem.replace("_", " ")).strip()
        paginas = extraer_paginas(ruta)
        antes = len(fragmentos)
        for numero, texto in paginas:
            for j, trozo in enumerate(trocear(texto)):
                fragmentos.append(Fragmento(f"{ruta.stem}-p{numero}-{j}", ruta.name, titulo, numero, trozo))
        resumen.append(
            {"archivo": ruta.name, "titulo": titulo, "paginas_con_texto": len(paginas),
             "caracteres": sum(len(t) for _, t in paginas), "fragmentos": len(fragmentos) - antes}
        )
    return fragmentos, resumen


# ---------------------------------------------------------------------- embeddings
class Embedder:
    """Envoltura mínima de sentence-transformers; devuelve vectores float32 de norma 1."""

    def __init__(self, modelo: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.nombre = modelo or config.EMBEDDING_MODEL
        self._modelo = SentenceTransformer(self.nombre)

    def codificar(self, textos: list[str]) -> np.ndarray:
        vectores = self._modelo.encode(list(textos), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vectores, dtype="float32")


# ---------------------------------------------------------------------- índice
def construir_indice(docs_dir: Path | None = None, index_dir: Path | None = None, embedder=None) -> list[dict]:
    import faiss

    index_dir = Path(index_dir or config.INDEX_DIR)
    fragmentos, resumen = fragmentar_corpus(docs_dir)
    if not fragmentos:
        raise RuntimeError(
            "No se extrajo texto de ningún documento. Pon PDFs con texto (no escaneados) en docs/; "
            "la lista sugerida está en docs/fuentes.csv."
        )
    embedder = embedder or Embedder()
    vectores = embedder.codificar([f.texto for f in fragmentos])
    indice = faiss.IndexFlatIP(vectores.shape[1])      # producto interno sobre vectores normalizados = coseno
    indice.add(vectores)

    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / ARCHIVO_INDICE).write_bytes(faiss.serialize_index(indice).tobytes())
    (index_dir / ARCHIVO_FRAGMENTOS).write_text(
        json.dumps(
            {
                "modelo_embeddings": getattr(embedder, "nombre", "desconocido"),
                "dimension": int(vectores.shape[1]),
                "creado": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "documentos": resumen,
                "fragmentos": [asdict(f) for f in fragmentos],
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    return resumen


def indice_disponible(index_dir: Path | None = None) -> bool:
    index_dir = Path(index_dir or config.INDEX_DIR)
    return (index_dir / ARCHIVO_INDICE).exists() and (index_dir / ARCHIVO_FRAGMENTOS).exists()


class Buscador:
    """Carga el índice guardado y responde consultas por similitud."""

    def __init__(self, index_dir: Path | None = None, embedder=None):
        import faiss

        index_dir = Path(index_dir or config.INDEX_DIR)
        if not indice_disponible(index_dir):
            raise FileNotFoundError(f"No hay índice en {index_dir}. Ejecuta: python -m src.build_index")
        self.indice = faiss.deserialize_index(np.frombuffer((index_dir / ARCHIVO_INDICE).read_bytes(), dtype="uint8"))
        meta = json.loads((index_dir / ARCHIVO_FRAGMENTOS).read_text(encoding="utf-8"))
        self.fragmentos = meta["fragmentos"]
        self.documentos = meta.get("documentos", [])
        self.modelo_embeddings = meta.get("modelo_embeddings")
        self._embedder = embedder

    @property
    def embedder(self):
        if self._embedder is None:               # el modelo se carga la primera vez que se busca
            self._embedder = Embedder(self.modelo_embeddings if self.modelo_embeddings != "desconocido" else None)
        return self._embedder

    def buscar(self, consulta: str, k: int = config.RAG_TOP_K, umbral: float | None = None) -> list[dict]:
        """Hasta k fragmentos con similitud >= umbral, del más al menos parecido."""
        umbral = config.RAG_UMBRAL if umbral is None else umbral
        vector = self.embedder.codificar([consulta])
        similitudes, posiciones = self.indice.search(vector, min(k, len(self.fragmentos)))
        resultados = []
        for similitud, posicion in zip(similitudes[0], posiciones[0]):
            if posicion < 0 or similitud < umbral:
                continue
            f = self.fragmentos[posicion]
            resultados.append({**f, "similitud": round(float(similitud), 3)})
        return resultados
