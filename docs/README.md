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

En la Resolución 2254 de 2017 varias tablas (niveles máximos, puntos de corte del ICA) están incrustadas como
imágenes, así que su contenido no llega al índice. La hoja metodológica del IDEAM y el plan de acción del POECA
sí traen esos valores como texto, y por eso forman parte del corpus.
