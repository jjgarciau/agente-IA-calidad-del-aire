# Corpus del RAG

Esta carpeta contiene los documentos oficiales que el agente puede consultar y citar.
`fuentes.csv` es el registro de procedencia: archivo, título, entidad, URL y fecha de consulta.

## Cómo armar el corpus

1. Descarga cada documento desde la URL indicada en `fuentes.csv`.
2. Guárdalo en esta carpeta **con el nombre de la columna `archivo`**.
3. Construye el índice:

   ```bash
   python -m src.build_index
   ```

El script informa cuántas páginas, caracteres y fragmentos se obtuvieron de cada archivo.
Si un PDF es una imagen escaneada no se extrae texto: hay que reemplazarlo por una versión con texto seleccionable.

## Cómo agregar un documento

Copia el PDF (o un `.txt` / `.md`) en esta carpeta, agrega una fila a `fuentes.csv` y vuelve a ejecutar
`python -m src.build_index`. El título de `fuentes.csv` es el que aparece en las citas del agente.

## Nota sobre las tablas

En el PDF de la Resolución 2254 de 2017 la página 3 completa (niveles máximos permisibles) y las tablas de las páginas 6 y 10
(niveles de prevención, alerta y emergencia; puntos de corte del ICA) están incrustadas como imágenes, así que su contenido no
llega al índice. En la primera evaluación del agente esto produjo una respuesta incorrecta sobre el límite diario de PM2.5.

Por eso el corpus incluye `res_2254_2017_tablas.md`: una transcripción de esos valores hecha a partir de las imágenes del mismo
PDF, redactada en oraciones completas para que cada fragmento conserve su contexto (resolución, tabla, contaminante y tiempo de
exposición). Al ser un archivo de texto no tiene paginación propia: la página del PDF original se indica dentro de cada oración.
