# Aire Aburrá: agente de calidad del aire del Valle de Aburrá

**SI3003 – Inteligencia Artificial · Proyecto final · Universidad EAFIT · 13 de noviembre de 2026**

**Autor:** Juan José García Urrego · **Repositorio:** https://github.com/jjgarciau/agente-IA-calidad-del-aire

## 1. Problema

El Valle de Aburrá es un valle estrecho en el que el material particulado fino (PM2.5) se acumula en ciertas épocas del año. Las autoridades publican la concentración de cada estación, pero un número en µg/m³ no le dice a un ciudadano si puede salir a hacer ejercicio ni qué esperar mañana. El proyecto construye un asistente que (1) muestra el PM2.5 medido y su categoría del Índice de Calidad del Aire (ICA), (2) predice la categoría del día siguiente con un modelo propio y (3) responde preguntas en lenguaje natural apoyándose en la normativa y citando la fuente.

## 2. Datos

**Fuente.** Conjunto «Calidad del Aire en Colombia» (SISAIRE, IDEAM) de datos.gov.co, id `g4t8-zkc3`, descargado por su API el 12 de noviembre de 2026: mediciones horarias de PM2.5 de la red del Área Metropolitana del Valle de Aburrá (AMVA) entre 2022-01-01 y 2025-12-31.

**Limpieza.** Se descartan valores negativos o mayores a 500 µg/m³ y duplicados; un día es válido con al menos 18 de 24 horas; se conservan las estaciones con al menos 180 días válidos.

**Hallazgo de calidad de datos.** La primera tabla diaria tenía promedios de 100 a 300 µg/m³, imposibles para esta red. La causa está en la fuente: en 2022 y 2023, 14 estaciones aparecen **sin punto decimal** (39.9 como 399 y 29.49 como 2949). Los valores de un decimal sobrevivían al filtro multiplicados por 10. Como no se puede saber qué factor aplicar a cada valor, esos meses se descartan en lugar de corregirse: cada estación se compara con la mediana diaria de las estaciones de referencia (al menos 300 días válidos ese año) y se elimina el mes-estación cuyo nivel relativo supera 2.5 veces el habitual de la estación. En los meses sanos la mayor razón es 1.5 y en los corruptos la menor es 3.4. Se descartaron 927 días-estación en 275 meses.

**Resultado.** 21.245 días-estación de 25 estaciones: 27.4 % en categoría Buena, 71.5 % Aceptable, 1.1 % Dañina para grupos sensibles y un solo día Dañina. En 2022 y 2023 quedan 9 estaciones; desde 2024, 23 o 24.

**Análisis exploratorio** (`notebooks/01_eda.ipynb`). Los episodios se concentran en febrero-abril (7.1 % de los días-estación de marzo) y casi todos son de 2024 (204 de 243); 2025 fue un año muy limpio. El domingo es el día con menos PM2.5 (14.6 µg/m³ frente a 17–18 de martes a viernes). La correlación entre el PM2.5 de hoy y el de mañana es 0.80, y el 80.4 % de los días repite la clase del día anterior: la persistencia es una línea base exigente.

## 3. Método de la Parte 1: Naive Bayes categórico

**Tarea.** Clasificar la clase ICA del día *t+1* por estación, en cuatro clases (las categorías superiores se agrupan en «Dañina o peor»).

**Variables**, todas categóricas y conocidas el día *t*: nivel de PM2.5 de hoy (14 niveles, más finos cerca de los cortes del ICA), promedio de los últimos 7 días en los mismos niveles, tipo de día objetivo (laboral, sábado, domingo) y cambio del promedio del valle entre ayer y hoy (5 niveles).

**Modelo.** P(y | x) ∝ P(y) · ∏ P(xⱼ | y), con P(xⱼ = v | y = c) = (N_cjv + α) / (N_c + α·Vⱼ). La implementación es propia, en numpy: conteos, suavizado de Laplace y suma de log-probabilidades (`src/model.py`). Una prueba automática verifica que las probabilidades coinciden con `CategoricalNB` de scikit-learn (diferencia máxima de 2·10⁻¹⁵), que no se usa en el modelo entregado.

**Protocolo.** Partición temporal: entrenamiento con días objetivo hasta el 2024-12-31 (12.883 días-estación) y prueba con 2025 (7.682), evaluada una sola vez. Dentro del entrenamiento, 2024 se reserva como validación para elegir α (0.1) y el conjunto de variables. Líneas base: clase mayoritaria y persistencia («mañana igual que hoy»). Métricas: exactitud, F1-macro, y precisión y recall de «episodio» (Dañina para grupos sensibles o peor).

## 4. Diseño del agente y del RAG

**Agente.** Loop agéntico manual sobre un endpoint compatible con OpenAI (Groq, modelo `openai/gpt-oss-120b`): el modelo recibe la pregunta y los esquemas de las herramientas; si pide una, se ejecuta y el resultado vuelve como observación; se repite hasta que responde. T_max = 5 iteraciones; al agotarse se fuerza una respuesta final sin herramientas. Los fallos de las herramientas vuelven como `{"error": ...}` con una pista (estaciones parecidas, fechas válidas cercanas), nunca como excepción.

| Herramienta | Qué hace |
|---|---|
| `get_pm25(estacion, fecha, dias)` | PM2.5 diario medido y categoría ICA, de un día o de una ventana |
| `predict_tomorrow(estacion, fecha)` | Predicción del Naive Bayes para el día siguiente, con probabilidades y variables usadas |
| `search_docs(query)` | Fragmentos de los documentos oficiales, con documento y página |
| `list_stations()` | Estaciones disponibles |

**Integración de las dos partes.** El modelo clásico es una herramienta del agente: ante «¿cómo estará mañana y qué se recomienda?», el agente encadena `predict_tomorrow` y `search_docs`.

**RAG.** Corpus (`docs/fuentes.csv`): Resolución 2254 de 2017, hoja metodológica del ICA (IDEAM), plan de acción del protocolo POECA, una resolución metropolitana de gestión de episodios y la transcripción de las tablas de la Resolución 2254, que en el PDF son imágenes. El texto se extrae por página, se divide en fragmentos de unos 500 caracteres que respetan las oraciones (el modelo de embeddings trunca en 128 tokens), se codifica con `paraphrase-multilingual-MiniLM-L12-v2` y se indexa en FAISS (producto interno sobre vectores normalizados). Cada fragmento conserva documento y página, y el prompt obliga a citar con el formato `[Fuente: título, p. N]` y a reconocer cuando los documentos no contienen la respuesta.

**Dashboard.** Streamlit: selección de estación, fecha de referencia y ventana; mapa de estaciones por categoría; serie de tiempo con bandas del ICA; predicción con probabilidades y comparación con el valor real; chat que muestra las herramientas usadas en cada respuesta.

## 5. Resultados

**Modelo, periodo de prueba (2025):**

| Modelo | Exactitud | F1-macro |
|---|---|---|
| Clase mayoritaria | 0.634 | 0.259 |
| Persistencia | 0.782 | 0.510 |
| Naive Bayes (propio) | **0.791** | **0.550** |

El Naive Bayes supera a la persistencia por un punto porcentual. Un *bootstrap* por días da un intervalo de confianza del 95 % de −0.001 a +0.020 para esa diferencia, y el modelo queda por encima en el 97 % de los remuestreos y en 17 de 23 estaciones: es al menos tan bueno como la persistencia y probablemente algo mejor, no claramente superior. Casi todos los errores están en la frontera Buena/Aceptable (612 y 969 días). La ablación muestra que la variable que más aporta es el promedio de 7 días: sin ella la exactitud cae a 0.778, por debajo de la persistencia.

**Episodios.** 2025 solo tiene 6 días de episodio entre 7.682, así que allí la precisión y el recall de episodio no son estimables (el modelo acierta 1 de 6 y la persistencia ninguno). En validación (2024, 200 días de episodio) el modelo obtiene un recall de 0.63 con precisión de 0.47, frente a 0.57 y 0.56 de la persistencia: anticipa más episodios a cambio de más falsas alarmas. Estas cifras son optimistas porque ese año se usó para elegir el modelo.

**Una lección sobre Naive Bayes.** La primera versión tenía seis variables y quedaba por debajo de la persistencia (0.769). Retirar las que repetían información (clase de ayer, tendencia, temporada, nivel del valle) mejoró el resultado: el modelo trata cada variable como evidencia independiente y cuenta dos veces la información correlacionada.

**Agente** (14 preguntas fijas; primera corrida en `eval/corrida_1/`). Llamó a la herramienta esperada en 13 de 14, recuperó un documento esperado en 4 de 4 y lo citó con el formato pedido en 2 de 4. Las tres respuestas sobre datos se verificaron contra la tabla y son correctas. El análisis de los fallos motivó tres cambios:

- **Respuesta incorrecta sobre el límite diario de PM2.5** (dijo 54 µg/m³; es 37). La página de la resolución que lo contiene es una imagen y no llegó al índice, y el modelo leyó la columna de PM10 de otra tabla. Se transcribieron las tablas a un documento de texto del corpus y el prompt prohíbe adivinar columnas.
- **Citas con otro formato.** En dos respuestas el modelo citó con símbolos propios; ahora las citas se normalizan.
- **Recomendaciones no respaldadas.** En la pregunta encadenada agregó medidas que no estaban en los fragmentos; el prompt ahora lo prohíbe.

El único «fallo» de herramienta fue una fecha fuera de rango que el agente respondió bien sin llamar a ninguna, porque el prompt ya le informa el rango de datos: es un defecto del criterio automático, no del agente. En la segunda corrida, con estos cambios, el agente respondió correctamente el límite diario (37 µg/m³) citando la transcripción de la tabla, y las preguntas de normativa pasaron a citar la fuente esperada en 3 de 3. La corrida vigente, con su fecha, está en `RESULTADOS.md`.

## 6. Limitaciones

- **Modelo.** La mejora sobre la persistencia es pequeña y no es concluyente al 95 %. No usa meteorología (lluvia, viento, capa de mezcla), que es lo que determina físicamente un episodio. Sus probabilidades son sobreconfiadas en la zona baja (error de calibración de 0.075).
- **Episodios.** Casi todos los días de episodio son de una sola temporada (febrero-abril de 2024) y el año de prueba casi no tiene, de modo que la utilidad del modelo para alertar no queda demostrada.
- **Datos.** Por el defecto de la fuente, 2022 y 2023 solo tienen 9 estaciones utilizables. Los datos son históricos, hasta 2025-12-31.
- **RAG.** El umbral de similitud no discrimina con este modelo de embeddings (los fragmentos pertinentes puntuaron entre 0.60 y 0.77 y los de una pregunta ajena al corpus entre 0.54 y 0.62): lo que evita respuestas inventadas es el prompt. Las tablas que son imágenes en los PDF solo llegan al índice si se transcriben.
- **Agente.** Catorce preguntas y un solo modelo de lenguaje no miden la variabilidad entre corridas.

El sistema es una herramienta analítica académica: no constituye consejo médico, legal ni de seguridad, y no reemplaza el pronóstico oficial del SIATA.

## Referencias

IDEAM / SISAIRE, *Calidad del Aire en Colombia*, datos.gov.co, conjunto `g4t8-zkc3`. · Ministerio de Ambiente y Desarrollo Sostenible, *Resolución 2254 de 2017*. · IDEAM, *Hoja metodológica del Índice de Calidad del Aire (ICA)*. · Municipio de La Estrella y AMVA, *Plan de acción para la implementación del protocolo POECA* (2019). · AMVA, resolución metropolitana que declara el periodo de gestión de episodios del segundo semestre de 2023. · Russell y Norvig, *Artificial Intelligence: A Modern Approach*, 4.ª ed. · Material del curso SI3003, clases 10 (RAG) y 12 (agentes).
