# Resultados

Generado el 2026-11-12 15:46 con `python -m src.reporte` a partir de los archivos de la última corrida.

## Datos

- Periodo: **2022-01-01 a 2025-12-31**
- Estaciones: **25** · días-estación con promedio válido: **22,172**
- PM2.5 diario: media 22.2 µg/m³, mediana 16.1, máximo 316.9

| Categoría ICA | Días-estación | Fracción |
|---|---|---|
| Buena | 5,816 | 26.2% |
| Aceptable | 15,190 | 68.5% |
| Dañina para grupos sensibles | 250 | 1.1% |
| Dañina | 543 | 2.4% |
| Muy dañina | 359 | 1.6% |
| Peligrosa | 14 | 0.1% |

## Modelo de la Parte 1: Naive Bayes frente a las líneas base

- Entrenamiento: 12,830 días-estación (2022-01-03 a 2024-12-31)
- Prueba: 7,461 días-estación (2025-01-01 a 2025-12-31), nunca vistos al entrenar
- Clases en prueba: Buena 2,713, Aceptable 4,742, Dañina para grupos sensibles 6, Dañina o peor 0
- Suavizado α = 0.1 (elegido validando en el último año del entrenamiento)

| Modelo | Exactitud | F1-macro | Precisión de episodio | Recall de episodio |
|---|---|---|---|---|
| Clase mayoritaria | 0.636 | 0.259 | 0.000 | 0.000 |
| Persistencia | 0.782 | 0.510 | 0.000 | 0.000 |
| Naive Bayes (propio) | 0.769 | 0.504 | 0.000 | 0.000 |

«Episodio» = Dañina para grupos sensibles o peor.

Matriz de confusión del Naive Bayes (filas = clase real, columnas = clase predicha):

| Real \ Predicha | Buena | Aceptable | Dañina para grupos sensibles | Dañina o peor |
|---|---|---|---|---|
| Buena | 1972 | 741 | 0 | 0 |
| Aceptable | 965 | 3769 | 8 | 0 |
| Dañina para grupos sensibles | 0 | 6 | 0 | 0 |
| Dañina o peor | 0 | 0 | 0 | 0 |

Ablación (Naive Bayes sin una variable):

| Variable retirada | Exactitud | F1-macro | Recall de episodio |
|---|---|---|---|
| nivel_hoy | 0.749 | 0.478 | 0.000 |
| clase_ayer | 0.768 | 0.500 | 0.000 |
| tendencia | 0.770 | 0.502 | 0.000 |
| temporada | 0.780 | 0.510 | 0.000 |
| dia_manana | 0.768 | 0.503 | 0.000 |
| nivel_valle | 0.773 | 0.498 | 0.000 |

## Agente: preguntas de prueba

- Preguntas: **14**
- Llamó a la herramienta esperada: **13/14**
- Recuperó un documento esperado: **4/4**
- Citó un documento esperado: **2/4**
- Iteraciones promedio: **1.9** · alcanzaron T_max: **0**

| # | Tipo | Pregunta | Herramientas llamadas | Herramienta correcta | Citó fuente |
|---|---|---|---|---|---|
| 1 | dato | ¿Cuál fue el PM2.5 en Tráfico Centro el 2025-03-12? | get_pm25 | sí | — |
| 2 | dato | ¿Cómo estuvo el aire hoy? | get_pm25 | sí | — |
| 3 | dato | ¿Cuál fue el día con más PM2.5 en Casa Justicia durante la semana que terminó el 2025-03-16? | get_pm25 | sí | — |
| 4 | prediccion | ¿Qué categoría de calidad del aire se espera mañana? | predict_tomorrow | sí | — |
| 5 | prediccion | Dame el pronóstico para el día siguiente al 2025-10-20 en la estación Santa Elena. | predict_tomorrow | sí | — |
| 6 | prediccion | ¿Qué probabilidad hay de que mañana el aire sea dañino para grupos sensibles? | predict_tomorrow | sí | — |
| 7 | normativa | ¿Cuál es el nivel máximo permisible de PM2.5 para 24 horas en Colombia? | search_docs | sí | no |
| 8 | normativa | ¿Qué rangos de PM2.5 corresponden a los niveles de prevención, alerta y emergencia? | search_docs | sí | sí |
| 9 | normativa | ¿Qué es el POECA y quién declara un episodio de contaminación? | search_docs | sí | sí |
| 10 | encadenada | ¿Cómo estará mañana el aire y qué se recomienda hacer en ese nivel? | predict_tomorrow > search_docs | sí | no |
| 11 | limite | ¿Cuál fue el PM2.5 en la estación Marte el 2025-03-12? | get_pm25 | sí | — |
| 12 | limite | ¿Cuál fue el PM2.5 en Tráfico Centro el 2027-06-01? | ninguna | no | — |
| 13 | limite | ¿Qué multa se paga por incumplir el pico y placa ambiental? | search_docs | sí | — |
| 14 | fuera_de_dominio | ¿Quién ganó el mundial de fútbol de 2014? | ninguna | sí | — |

El detalle de cada respuesta está en `eval/resultados.csv` y las trazas completas en `eval/trazas.json`.
