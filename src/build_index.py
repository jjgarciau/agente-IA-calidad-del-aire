"""Construye el índice FAISS a partir de los documentos de docs/.

    python -m src.build_index

La primera vez descarga el modelo de embeddings (~470 MB) desde Hugging Face.
Al terminar muestra unas consultas de prueba con su similitud, útiles para ajustar
el umbral RAG_UMBRAL (por defecto 0.30) si el corpus cambia.
"""
from __future__ import annotations

from src import config, rag

CONSULTAS_DE_PRUEBA = [
    "¿Cuál es el nivel máximo permisible de PM2.5 para 24 horas?",
    "¿Qué medidas se toman en el nivel de alerta de un episodio de contaminación?",
    "¿Qué deben hacer los grupos sensibles cuando la calidad del aire es dañina?",
    "¿Cuál es la receta de la bandeja paisa?",   # fuera del corpus: debería quedar bajo el umbral
]


def main() -> None:
    print(f"Documentos en: {config.DOCS_DIR}")
    print(f"Modelo de embeddings: {config.EMBEDDING_MODEL}\n")
    resumen = rag.construir_indice()

    print(f"{'Archivo':<40}{'Páginas':>9}{'Caracteres':>12}{'Fragmentos':>12}")
    for r in resumen:
        print(f"{r['archivo'][:39]:<40}{r['paginas_con_texto']:>9}{r['caracteres']:>12,}{r['fragmentos']:>12}")
        if r["fragmentos"] == 0:
            print("   AVISO: no se extrajo texto (¿PDF escaneado?). Reemplázalo por una versión con texto.")
    print(f"\nÍndice guardado en {config.INDEX_DIR} ({sum(r['fragmentos'] for r in resumen)} fragmentos)\n")

    buscador = rag.Buscador()
    print(f"Consultas de prueba (umbral actual = {config.RAG_UMBRAL}):")
    for consulta in CONSULTAS_DE_PRUEBA:
        mejores = buscador.buscar(consulta, k=1, umbral=-1.0)
        if mejores:
            m = mejores[0]
            estado = "se acepta" if m["similitud"] >= config.RAG_UMBRAL else "se descarta"
            print(f"  [{m['similitud']:.2f} {estado}] {consulta}\n      -> {m['titulo']}, p. {m['pagina']}: {m['texto'][:110]}...")


if __name__ == "__main__":
    main()
