# Aire Aburrá: agente de calidad del aire del Valle de Aburrá

**SI3003 – Inteligencia Artificial · Proyecto final · Universidad EAFIT · noviembre de 2026**
**Autor:** «completar nombre» · **Repositorio:** https://github.com/jjgarciau/agente-IA-calidad-del-aire

> Borrador. Los campos entre «comillas angulares» se completan con los números de `RESULTADOS.md` y con la interpretación propia.
> Extensión máxima: 4 páginas. Para entregar, exportar a PDF en esta misma carpeta.

## 1. Problema

El Valle de Aburrá es un valle estrecho en el que el material particulado fino (PM2.5) se acumula en ciertas épocas del año. Las autoridades
publican la concentración de cada estación, pero un número en µg/m³ no le dice a un ciudadano si puede salir a hacer ejercicio ni qué esperar mañana.
El proyecto construye un asistente que (1) muestra el PM2.5 medido y su categoría del Índice de Calidad del Aire (ICA), (2) predice la categoría
del día siguiente y (3) responde preguntas en lenguaje natural apoyándose en la normativa y citando la fuente.

## 2. Datos

- **Fuente:** conjunto «Calidad del Aire en Colombia» (SISAIRE, IDEAM) en datos.gov.co, id `g4t8-zkc3`, consultado por API el «fecha de descarga».
- **Subconjunto:** autoridad ambiental AMVA, variable PM2.5, mediciones horarias de 2022-01-01 a 2025-12-31: «N» mediciones de «N» estaciones.
- **Limpieza:** se descartan valores negativos o mayores a 500 µg/m³ (hay valores centinela) y duplicados; un día es válido con al menos 18 de 24 horas;
  se conservan las estaciones con al menos 180 días válidos. Resultado: «N» días-estación en «N» estaciones.
- **Categorías ICA** (Resolución 2254 de 2017, PM2.5 de 24 h): Buena 0–12, Aceptable 13–37, Dañina para grupos sensibles 38–55, Dañina 56–150 µg/m³.
  Distribución observada: «% por categoría».
- **Hallazgos del análisis exploratorio** (`notebooks/01_eda.ipynb`): «meses con más episodios»; «diferencia entre días laborales y fin de semana»;
  «correlación entre hoy y mañana y porcentaje de días que repiten clase».

## 3. Método de la Parte 1: Naive Bayes categórico

**Tarea.** Clasificación de la clase ICA del día *t+1* por estación, en cuatro clases (las tres categorías superiores se agrupan en «Dañina o peor» por ser muy escasas).

**Variables** (todas conocidas el día *t*): nivel de PM2.5 de hoy (7 niveles), clase de ayer, tendencia entre ayer y hoy, temporada del día objetivo,
tipo de día objetivo (laboral, sábado, domingo) y nivel promedio del valle.

**Modelo.** P(y | x) ∝ P(y) · ∏ P(xⱼ | y), con P(xⱼ = v | y = c) = (N_cjv + α) / (N_c + α·Vⱼ). La implementación es propia, en numpy:
conteos, suavizado de Laplace y suma de log-probabilidades (`src/model.py`). Una prueba automática verifica que las probabilidades coinciden con
`CategoricalNB` de scikit-learn, que no se usa en el modelo entregado.

**Protocolo experimental.** Partición temporal: entrenamiento con días objetivo hasta el 2024-12-31 y prueba con todo 2025. El suavizado α se elige
validando en el último año del entrenamiento; después se reentrena y se evalúa una sola vez en prueba. Líneas base: clase mayoritaria y persistencia
(«mañana igual que hoy»). Métricas: exactitud, F1-macro, y precisión y recall de «episodio» (Dañina para grupos sensibles o peor).

## 4. Diseño del agente y del RAG

**Agente.** Loop agéntico manual sobre un endpoint compatible con OpenAI (Groq, modelo `openai/gpt-oss-120b`): el LLM recibe la pregunta y los esquemas
de las herramientas; si pide una herramienta, se ejecuta y el resultado vuelve como observación; se repite hasta que responde. T_max = 5 iteraciones;
al agotarse se fuerza una respuesta final sin herramientas. Los fallos de las herramientas vuelven como `{"error": ...}` con una pista, nunca como excepción.

| Herramienta | Qué hace |
|---|---|
| `get_pm25(estacion, fecha, dias)` | PM2.5 diario medido y categoría ICA, de un día o de una ventana |
| `predict_tomorrow(estacion, fecha)` | Predicción del Naive Bayes para el día siguiente, con probabilidades y variables usadas |
| `search_docs(query)` | Fragmentos de los documentos oficiales con documento y página |
| `list_stations()` | Estaciones disponibles |

**Integración de las dos partes.** El modelo clásico es una herramienta del agente: ante «¿cómo estará mañana y qué debo hacer?», el agente encadena
`predict_tomorrow` y `search_docs`.

**RAG.** Corpus de «N» documentos oficiales (`docs/fuentes.csv`): «listar». El texto se extrae por página, se divide en fragmentos de ~500 caracteres
que respetan las oraciones (el modelo de embeddings trunca en 128 tokens), se codifica con `paraphrase-multilingual-MiniLM-L12-v2` y se indexa en FAISS
(producto interno sobre vectores normalizados). Se devuelven hasta 4 fragmentos con similitud ≥ 0.30; si ninguno la supera, la herramienta responde
«sin contexto suficiente» y el prompt obliga al agente a decirlo en lugar de inventar. Total: «N» fragmentos.

**Dashboard.** Streamlit: selección de estación, fecha de referencia y ventana; mapa de estaciones por categoría; serie de tiempo con bandas del ICA;
predicción con probabilidades y comparación con el valor real; chat que muestra las herramientas usadas en cada respuesta.

## 5. Resultados

**Modelo** (prueba: 2025, «N» días-estación):

| Modelo | Exactitud | F1-macro | Precisión de episodio | Recall de episodio |
|---|---|---|---|---|
| Clase mayoritaria | « » | « » | « » | « » |
| Persistencia | « » | « » | « » | « » |
| Naive Bayes (propio) | « » | « » | « » | « » |

«Interpretación en 3–4 frases: ¿supera el Naive Bayes a la persistencia?, ¿en qué métrica?, ¿qué errores comete según la matriz de confusión?,
¿qué variable aporta más según la ablación?»

**Agente** (14 preguntas fijas, `eval/resultados.csv`): herramienta correcta en «N»/14; citó un documento esperado en «N»/4;
respuesta correcta según revisión manual en «N»/14. «Comentar 1–2 fallos concretos».

## 6. Limitaciones

- El modelo no usa meteorología (lluvia, viento, altura de la capa de mezcla), que es lo que determina físicamente un episodio.
- Naive Bayes supone independencia entre variables correlacionadas; sus probabilidades tienden a ser extremas (ver calibración en `notebooks/02_modelo.ipynb`).
- Las clases altas son escasas y un solo año de prueba puede no ser representativo.
- Los datos son históricos (hasta 2025-12-31); no hay consulta en tiempo real.
- Varias tablas de la Resolución 2254 son imágenes y no llegan al índice; el umbral de similitud se fijó a mano.
- La evaluación del agente es pequeña y con un solo modelo de lenguaje; no mide la variabilidad entre corridas.
- «Limitaciones observadas en las pruebas propias».

El sistema es una herramienta analítica académica: no constituye consejo médico, legal ni de seguridad.

## Referencias

- IDEAM / SISAIRE. *Calidad del Aire en Colombia*. datos.gov.co, conjunto `g4t8-zkc3`.
- Ministerio de Ambiente y Desarrollo Sostenible. *Resolución 2254 de 2017*, norma de calidad del aire ambiente.
- IDEAM. *Hoja metodológica del Índice de Calidad del Aire (ICA)*.
- Área Metropolitana del Valle de Aburrá. *Acuerdo Metropolitano 04 de 2018* (POECA) y documentos relacionados listados en `docs/fuentes.csv`.
- Russell, S. y Norvig, P. *Artificial Intelligence: A Modern Approach*, 4.ª ed. (clasificadores bayesianos).
- Material del curso SI3003: Clase 10 (RAG) y Clase 12 (agentes).
