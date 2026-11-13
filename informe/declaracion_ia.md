# Declaración de uso de IA generativa

**Proyecto:** Aire Aburrá, agente de calidad del aire del Valle de Aburrá · SI3003 – Inteligencia Artificial · noviembre de 2026

**Autor:** Juan José García Urrego

## Herramientas utilizadas

| Herramienta | Propósito |
|---|---|
| **Claude (Anthropic)**, como asistente de programación | Análisis del enunciado y comparación de los retos; plan de desarrollo por etapas; generación del código de los módulos (`src/`, `app.py`, `eval/`), de las pruebas automáticas y de los notebooks; verificación de las fuentes de datos y de los documentos del corpus; revisión de los resultados (detectó el defecto del punto decimal en la fuente y los fallos de la primera evaluación del agente) y sus correcciones; transcripción de las tablas de la Resolución 2254 a partir de las imágenes del PDF; redacción del README, del informe y de esta declaración. |
| **Modelo `openai/gpt-oss-120b` servido por Groq** | Componente del producto: es el modelo de lenguaje del agente (decide qué herramienta llamar y redacta la respuesta). No participó en el desarrollo. |
| **`paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers)** | Componente del producto: modelo de embeddings del RAG. |

## Alcance del uso

La mayor parte del código y de la documentación fue generada con Claude a partir de las decisiones y revisiones del autor. El autor eligió el reto, el alcance del MVP y el proveedor del modelo de lenguaje; aprobó el plan; configuró el entorno, la clave de API y el repositorio; descargó los datos y los documentos del corpus; ejecutó el entrenamiento, la construcción del índice, la evaluación del agente y el dashboard; y revisó el código y los resultados entregados.

## Cómo se verificó el trabajo generado con IA

- La implementación del Naive Bayes se compara en una prueba automática con `CategoricalNB` de scikit-learn (mismas probabilidades), y las métricas propias con las de scikit-learn.
- Hay 42 pruebas automáticas del preprocesamiento, el modelo, el RAG, las herramientas, el loop del agente y la evaluación (`python -m pytest -q`).
- El defecto del punto decimal se confirmó consultando directamente la fuente (datos.gov.co) y se reproduce en una prueba con datos sintéticos.
- Los puntos de corte del ICA se contrastaron con la hoja metodológica del IDEAM y con la tabla 6 de la Resolución 2254 de 2017.
- Todos los números del informe salen de corridas reales: `RESULTADOS.md`, `models/metricas.json`, `eval/corrida_1/` y los notebooks ejecutados.
- La evaluación del modelo separa validación y prueba, y reporta el intervalo de confianza de la diferencia con la línea base en lugar de presentar la mejora como concluyente.

## Uso ético

- No se usaron datos personales: los datos son abiertos y agregados por estación.
- Los datos defectuosos de la fuente se descartaron y se documentaron; no se corrigieron con supuestos.
- El sistema declara en la interfaz y en sus respuestas que es una herramienta académica y no un consejo médico, legal o de seguridad.
- El agente está instruido para responder sobre normativa solo con lo que recuperan los documentos, citar la fuente y reconocer cuando no tiene la información. Los casos en que no lo cumplió están documentados en `eval/corrida_1/README.md`.
