# Evaluación del agente

Corrida del 2026-11-13 14:06 con el modelo `openai/gpt-oss-120b` y T_max = 5.

- Preguntas: **14**
- Herramienta correcta: **13/14** (93%)
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
