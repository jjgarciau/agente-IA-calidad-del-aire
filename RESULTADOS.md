# Resultados

Generado el 2026-11-13 14:17 con `python -m src.reporte` a partir de los archivos de la última corrida.

## Datos

- Periodo: **2022-01-01 a 2025-12-31**
- Estaciones: **25** · días-estación con promedio válido: **21,245**
- PM2.5 diario: media 17.0 µg/m³, mediana 15.8, máximo 59.5
- Descartados por escala corrupta en la fuente (punto decimal perdido): **927** días-estación en 275 meses de 14 estaciones (`data/processed/descartes_escala.csv`)

| Categoría ICA | Días-estación | Fracción |
|---|---|---|
| Buena | 5,816 | 27.4% |
| Aceptable | 15,186 | 71.5% |
| Dañina para grupos sensibles | 242 | 1.1% |
| Dañina | 1 | 0.0% |
| Muy dañina | 0 | 0.0% |
| Peligrosa | 0 | 0.0% |

## Modelo de la Parte 1: Naive Bayes frente a las líneas base

- Entrenamiento: 12,883 días-estación (2022-01-02 a 2024-12-31)
- Prueba: 7,682 días-estación (2025-01-01 a 2025-12-31), nunca vistos al entrenar
- Clases en prueba: Buena 2,809, Aceptable 4,867, Dañina para grupos sensibles 6, Dañina o peor 0
- Suavizado α = 0.1 (elegido validando en el último año del entrenamiento)

| Modelo | Exactitud | F1-macro | Precisión de episodio | Recall de episodio |
|---|---|---|---|---|
| Clase mayoritaria | 0.634 | 0.259 | 0.000 | 0.000 |
| Persistencia | 0.782 | 0.510 | 0.000 | 0.000 |
| Naive Bayes (propio) | 0.791 | 0.550 | 0.059 | 0.167 |

«Episodio» = Dañina para grupos sensibles o peor. El periodo de prueba solo tiene 6 días de episodio, así que allí la precisión y el recall de episodio no son estimables.

Validación (2024-01-02 a 2024-12-31, 7,458 días-estación, 200 de episodio; modelo ajustado con datos hasta 2024-01-01). Este año se usó para elegir α y las variables, así que sus cifras son optimistas:

| Modelo (validación) | Exactitud | F1-macro | Precisión de episodio | Recall de episodio |
|---|---|---|---|---|
| Clase mayoritaria | 0.738 | 0.212 | 0.000 | 0.000 |
| Persistencia | 0.804 | 0.514 | 0.559 | 0.565 |
| Naive Bayes (propio) | 0.811 | 0.519 | 0.472 | 0.625 |

Matriz de confusión del Naive Bayes en prueba (filas = clase real, columnas = clase predicha):

| Real \ Predicha | Buena | Aceptable | Dañina para grupos sensibles | Dañina o peor |
|---|---|---|---|---|
| Buena | 2197 | 612 | 0 | 0 |
| Aceptable | 969 | 3882 | 16 | 0 |
| Dañina para grupos sensibles | 0 | 5 | 1 | 0 |
| Dañina o peor | 0 | 0 | 0 | 0 |

Ablación (Naive Bayes sin una variable):

| Variable retirada | Exactitud | F1-macro | Recall de episodio |
|---|---|---|---|
| nivel_hoy | 0.790 | 0.514 | 0.000 |
| nivel_semana | 0.778 | 0.496 | 0.000 |
| dia_manana | 0.791 | 0.552 | 0.167 |
| cambio_valle | 0.797 | 0.524 | 0.000 |

## Agente: preguntas de prueba

Corrida del 2026-11-13 14:06 con el modelo `openai/gpt-oss-120b` y T_max = 5.

- Preguntas: **14**
- Llamó a la herramienta esperada: **13/14**
- Recuperó un documento esperado: **4/4**
- Citó un documento esperado: **3/4**
- Iteraciones promedio: **1.9** · alcanzaron T_max: **0**

| # | Tipo | Pregunta | Herramientas llamadas | Herramienta correcta | Citó fuente |
|---|---|---|---|---|---|
| 1 | dato | ¿Cuál fue el PM2.5 en Tráfico Centro el 2025-03-12? | get_pm25 | sí | — |
| 2 | dato | ¿Cómo estuvo el aire hoy? | get_pm25 | sí | — |
| 3 | dato | ¿Cuál fue el día con más PM2.5 en Casa Justicia durante la semana que terminó el 2025-03-16? | get_pm25 | sí | — |
| 4 | prediccion | ¿Qué categoría de calidad del aire se espera mañana? | predict_tomorrow | sí | — |
| 5 | prediccion | Dame el pronóstico para el día siguiente al 2025-10-20 en la estación Santa Elena. | predict_tomorrow | sí | — |
| 6 | prediccion | ¿Qué probabilidad hay de que mañana el aire sea dañino para grupos sensibles? | predict_tomorrow | sí | — |
| 7 | normativa | ¿Cuál es el nivel máximo permisible de PM2.5 para 24 horas en Colombia? | search_docs | sí | sí |
| 8 | normativa | ¿Qué rangos de PM2.5 corresponden a los niveles de prevención, alerta y emergencia? | search_docs | sí | sí |
| 9 | normativa | ¿Qué es el POECA y quién declara un episodio de contaminación? | search_docs | sí | sí |
| 10 | encadenada | ¿Cómo estará mañana el aire y qué se recomienda hacer en ese nivel? | predict_tomorrow > search_docs > search_docs | sí | no |
| 11 | limite | ¿Cuál fue el PM2.5 en la estación Marte el 2025-03-12? | get_pm25 | sí | — |
| 12 | limite | ¿Cuál fue el PM2.5 en Tráfico Centro el 2027-06-01? | ninguna | no | — |
| 13 | limite | ¿Qué multa se paga por incumplir el pico y placa ambiental? | search_docs | sí | — |
| 14 | fuera_de_dominio | ¿Quién ganó el mundial de fútbol de 2014? | ninguna | sí | — |

El detalle de cada respuesta está en `eval/resultados.csv` y las trazas completas en `eval/trazas.json`.
