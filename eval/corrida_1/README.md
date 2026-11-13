# Primera corrida de la evaluación del agente (12 de noviembre de 2026)

Estos tres archivos son la evaluación original, antes de las correcciones que motivó:

- **Herramienta esperada:** 13 de 14.
- **Recuperó un documento esperado:** 4 de 4.
- **Citó un documento esperado:** 2 de 4.

Qué falló y qué se cambió:

| Pregunta | Qué pasó | Corrección |
|---|---|---|
| 7. Límite de PM2.5 en 24 horas | Respondió 54 µg/m³ (el valor correcto es 37). La página de la Resolución 2254 que lo contiene es una imagen y no llegó al índice; el modelo leyó la columna de PM10 de otra tabla. | Se transcribieron las tablas a `docs/res_2254_2017_tablas.md` y el prompt prohíbe adivinar columnas. |
| 7 y 10. Citas | El modelo escribió las citas con símbolos propios (`【1†Fuente: ...】`) y el evaluador no las reconoció. | `normalizar_citas` deja todas las citas como `[Fuente: título, p. N]`. |
| 10. Recomendaciones | Agregó medidas que no estaban en los fragmentos (tipo de mascarilla, cerrar ventanas). | El prompt exige recomendar solo lo que digan los fragmentos. |
| 12. Fecha fuera de rango | No llamó a ninguna herramienta, pero respondió bien: el prompt ya le informa el rango de datos. El criterio automático lo marca como fallo. | Ninguna: se deja como está y se explica en el informe. |

Esta corrida usó además la primera versión del modelo, entrenada antes de limpiar los datos con el punto decimal perdido,
por lo que las probabilidades de las preguntas 4, 5, 6 y 10 no coinciden con las del modelo actual.

La evaluación vigente está en `eval/resultados.csv`, `eval/resultados.md` y `eval/trazas.json`.
