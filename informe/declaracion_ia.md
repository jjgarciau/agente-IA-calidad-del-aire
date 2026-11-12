# Declaración de uso de IA generativa

**Proyecto:** Aire Aburrá, agente de calidad del aire del Valle de Aburrá · SI3003 – Inteligencia Artificial · noviembre de 2026
**Autor:** «completar nombre»

> Borrador para revisar. Debe describir lo que realmente ocurrió: ajusta cualquier punto que no corresponda antes de entregar.

## Herramientas utilizadas

| Herramienta | Propósito |
|---|---|
| **Claude (Anthropic)**, como asistente de programación | Análisis del enunciado y comparación de los retos; plan de desarrollo por etapas; generación del código de los módulos (`src/`, `app.py`, `eval/`), de las pruebas automáticas y de los notebooks; verificación de las fuentes de datos y de los documentos del corpus; borradores del README, del informe y de esta declaración. |
| **Modelo `openai/gpt-oss-120b` servido por Groq** | Componente del producto: es el modelo de lenguaje del agente (decide qué herramienta llamar y redacta la respuesta). No participó en el desarrollo. |
| **`paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers)** | Componente del producto: modelo de embeddings del RAG. |

## Qué hice yo

- Elegí el reto, el alcance del MVP y el proveedor del LLM, y aprobé el plan de desarrollo.
- Configuré el entorno, la clave de API y el repositorio; ejecuté la descarga de datos, el entrenamiento, la construcción del índice y la evaluación.
- Descargué los documentos del corpus y verifiqué su procedencia.
- «Revisé el código de `src/model.py`, `src/features.py`, `src/agent.py` y `src/tools.py` hasta poder explicarlo.»
- «Revisé una por una las respuestas de la evaluación del agente y marqué cuáles eran correctas.»
- «Interpreté los resultados y redacté la versión final del informe.»
- «Cambios que hice sobre el código generado: …»

## Cómo se verificó el trabajo generado con IA

- La implementación del Naive Bayes se compara en una prueba automática con `CategoricalNB` de scikit-learn (mismas probabilidades).
- Las métricas propias (exactitud, F1-macro, matriz de confusión) se comparan con las de scikit-learn.
- Hay pruebas del preprocesamiento, del RAG, de las herramientas y del loop del agente (`python -m pytest -q`).
- Los puntos de corte del ICA se contrastaron con la hoja metodológica del IDEAM, que cita la Resolución 2254 de 2017.
- Todos los números del informe salen de `RESULTADOS.md`, generado a partir de corridas reales con los datos descargados.

## Uso ético

- No se usaron datos personales: los datos son abiertos y agregados por estación.
- El sistema declara en la interfaz y en sus respuestas que es una herramienta académica y no un consejo médico, legal o de seguridad.
- El agente está instruido para responder sobre normativa solo con lo que recuperan los documentos, citar la fuente y reconocer cuando no tiene la información.
