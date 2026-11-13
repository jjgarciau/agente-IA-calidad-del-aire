"""Herramientas que el agente puede invocar (tool calling).

Cada herramienta devuelve SIEMPRE un texto JSON. Si algo sale mal devuelve
{"error": "..."} con una pista para corregir, en lugar de lanzar una excepción:
así el error llega al LLM como observación y puede reintentar o explicarlo.

  get_pm25          dato histórico de una estación (un día o una ventana de días)
  predict_tomorrow  predicción del Naive Bayes para el día siguiente a una fecha
  search_docs       RAG sobre los documentos oficiales, con documento y página
  list_stations     estaciones disponibles
"""
from __future__ import annotations

import difflib
import json
import re
import unicodedata
from functools import cached_property

import pandas as pd

from src import config, features, ica, preprocess, rag
from src.model import NaiveBayesCategorico

PALABRAS_VACIAS = {"estacion", "de", "del", "la", "el", "los", "las", "en", "y"}


def _normalizar(texto: str) -> str:
    sin_tildes = "".join(c for c in unicodedata.normalize("NFKD", str(texto)) if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", sin_tildes.lower()).strip()


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _error(mensaje: str, **extra) -> str:
    return _json({"error": mensaje, **extra})


class Recursos:
    """Datos, modelo e índice, cargados una sola vez y compartidos con el dashboard."""

    def __init__(self, buscador=None):
        self._buscador = buscador

    @cached_property
    def diario(self) -> pd.DataFrame:
        return preprocess.cargar_diario()

    @cached_property
    def estaciones(self) -> pd.DataFrame:
        return preprocess.cargar_estaciones()

    @cached_property
    def tabla(self) -> pd.DataFrame:
        """Variables del modelo para cada (estación, día)."""
        return features.construir_tabla(self.diario).set_index(["estacion", "fecha"]).sort_index()

    @cached_property
    def serie(self) -> pd.DataFrame:
        return self.diario.set_index(["estacion", "fecha"]).sort_index()

    @cached_property
    def modelo(self) -> NaiveBayesCategorico:
        if not config.MODEL_PATH.exists():
            raise FileNotFoundError("No hay modelo entrenado. Ejecuta: python -m src.train")
        return NaiveBayesCategorico.cargar(config.MODEL_PATH)[0]

    @cached_property
    def metricas(self) -> dict:
        if not config.METRICS_PATH.exists():
            return {}
        return json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))

    @property
    def buscador(self):
        if self._buscador is None:
            self._buscador = rag.Buscador()
        return self._buscador

    @property
    def rango(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.diario["fecha"].min(), self.diario["fecha"].max()

    # ------------------------------------------------------------------ utilidades
    def resolver_estacion(self, nombre: str) -> tuple[str | None, list[str]]:
        """Devuelve (estación exacta, sugerencias). Tolera tildes, mayúsculas y nombres parciales."""
        nombres = list(self.estaciones["estacion"])
        normalizados = {_normalizar(n): n for n in nombres}
        consulta = _normalizar(nombre)
        if consulta in normalizados:
            return normalizados[consulta], []

        tokens = [t for t in consulta.split() if t not in PALABRAS_VACIAS]
        if tokens:
            candidatas = []
            for _, fila in self.estaciones.iterrows():
                texto = set(_normalizar(f"{fila['estacion']} {fila['municipio']}").split())
                if all(t in texto for t in tokens):
                    candidatas.append(fila["estacion"])
            if len(candidatas) == 1:
                return candidatas[0], []
            if candidatas:
                return None, candidatas

        cercanas = difflib.get_close_matches(consulta, list(normalizados), n=3, cutoff=0.75)
        if len(cercanas) == 1:
            return normalizados[cercanas[0]], []
        sugerencias = cercanas or difflib.get_close_matches(consulta, list(normalizados), n=5, cutoff=0.4)
        return None, [normalizados[s] for s in sugerencias]

    def _estacion_o_error(self, nombre) -> tuple[str | None, str | None]:
        if not nombre or not str(nombre).strip():
            return None, _error("Falta el nombre de la estación.", estaciones_disponibles=list(self.estaciones["estacion"]))
        estacion, sugerencias = self.resolver_estacion(str(nombre))
        if estacion:
            return estacion, None
        if sugerencias:
            return None, _error(f"No hay una única estación que coincida con '{nombre}'.", quiso_decir=sugerencias)
        return None, _error(f"No existe la estación '{nombre}'.", estaciones_disponibles=list(self.estaciones["estacion"]))

    def _fecha_o_error(self, fecha) -> tuple[pd.Timestamp | None, str | None]:
        try:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(fecha).strip()):
                raise ValueError
            f = pd.Timestamp(str(fecha).strip())
        except (ValueError, TypeError):
            return None, _error(f"Fecha inválida: '{fecha}'. Usa el formato AAAA-MM-DD, por ejemplo 2025-03-15.")
        desde, hasta = self.rango
        if not (desde <= f <= hasta):
            return None, _error(
                f"La fecha {f:%Y-%m-%d} está fuera de los datos disponibles.",
                datos_disponibles=f"{desde:%Y-%m-%d} a {hasta:%Y-%m-%d}",
            )
        return f, None

    def _fechas_cercanas(self, indice: pd.MultiIndex, estacion: str, fecha: pd.Timestamp) -> dict:
        """Fecha válida anterior y posterior más próximas, para sugerir una alternativa."""
        try:
            fechas = indice.get_level_values("fecha")[indice.get_level_values("estacion") == estacion]
        except KeyError:
            return {}
        anteriores, posteriores = fechas[fechas < fecha], fechas[fechas > fecha]
        out = {}
        if len(anteriores):
            out["fecha_valida_anterior"] = f"{anteriores.max():%Y-%m-%d}"
        if len(posteriores):
            out["fecha_valida_posterior"] = f"{posteriores.min():%Y-%m-%d}"
        return out

    # ------------------------------------------------------------------ herramientas
    def get_pm25(self, estacion: str, fecha: str, dias: int = 1) -> str:
        est, err = self._estacion_o_error(estacion)
        if err:
            return err
        f, err = self._fecha_o_error(fecha)
        if err:
            return err
        try:
            dias = max(1, min(int(dias), 60))
        except (ValueError, TypeError):
            return _error("El parámetro 'dias' debe ser un número entero entre 1 y 60.")
        municipio = self.estaciones.set_index("estacion").loc[est, "municipio"]

        if dias == 1:
            if (est, f) not in self.serie.index:
                return _error(
                    f"No hay un promedio diario válido de PM2.5 en {est} el {f:%Y-%m-%d} "
                    f"(se exigen al menos {config.HORAS_MINIMAS} horas medidas).",
                    **self._fechas_cercanas(self.serie.index, est, f),
                )
            fila = self.serie.loc[(est, f)]
            return _json(
                {
                    "estacion": est, "municipio": municipio, "fecha": f"{f:%Y-%m-%d}",
                    "pm25_ug_m3": float(fila["pm25"]), "categoria_ica": fila["categoria"],
                    "horas_validas": int(fila["horas_validas"]), "fuente": "SISAIRE / AMVA (datos.gov.co)",
                }
            )

        ventana = self.serie.loc[est].loc[f - pd.Timedelta(days=dias - 1): f]
        if ventana.empty:
            return _error(f"No hay datos válidos en {est} entre {f - pd.Timedelta(days=dias - 1):%Y-%m-%d} y {f:%Y-%m-%d}.")
        return _json(
            {
                "estacion": est, "municipio": municipio,
                "desde": f"{ventana.index.min():%Y-%m-%d}", "hasta": f"{ventana.index.max():%Y-%m-%d}",
                "dias_con_dato": int(len(ventana)), "promedio_ug_m3": round(float(ventana["pm25"].mean()), 1),
                "maximo": {"fecha": f"{ventana['pm25'].idxmax():%Y-%m-%d}", "pm25_ug_m3": float(ventana["pm25"].max())},
                "minimo": {"fecha": f"{ventana['pm25'].idxmin():%Y-%m-%d}", "pm25_ug_m3": float(ventana["pm25"].min())},
                "dias_por_categoria": {k: int(v) for k, v in ventana["categoria"].value_counts().items()},
                "serie": [{"fecha": f"{i:%Y-%m-%d}", "pm25_ug_m3": float(r["pm25"]), "categoria_ica": r["categoria"]} for i, r in ventana.iterrows()],
                "fuente": "SISAIRE / AMVA (datos.gov.co)",
            }
        )

    def predecir(self, estacion: str, fecha: pd.Timestamp) -> dict | None:
        """Predicción para el día siguiente a `fecha`; None si la estación no tiene dato válido ese día."""
        if (estacion, fecha) not in self.tabla.index:
            return None
        fila = self.tabla.loc[(estacion, fecha)]
        probabilidades = self.modelo.predict_proba(fila[features.NOMBRES].to_numpy(dtype=int))[0]
        prevista = int(probabilidades.argmax())
        out = {
            "estacion": estacion,
            "fecha_referencia": f"{fecha:%Y-%m-%d}",
            "fecha_predicha": f"{fila['fecha_objetivo']:%Y-%m-%d}",
            "categoria_predicha": ica.CLASES[prevista],
            "probabilidades": {ica.CLASES[i]: round(float(p), 3) for i, p in enumerate(probabilidades)},
            "pm25_hoy_ug_m3": float(fila["pm25_hoy"]),
            "variables_usadas": features.describir(fila),
        }
        if pd.notna(fila["y"]):
            out["valor_real"] = {"pm25_ug_m3": float(fila["pm25_manana"]), "categoria": ica.CLASES[int(fila["y"])]}
            out["acierto"] = int(fila["y"]) == prevista
        return out

    def predict_tomorrow(self, estacion: str, fecha: str) -> str:
        est, err = self._estacion_o_error(estacion)
        if err:
            return err
        f, err = self._fecha_o_error(fecha)
        if err:
            return err
        prediccion = self.predecir(est, f)
        if prediccion is None:
            return _error(
                f"No se puede predecir desde {f:%Y-%m-%d} en {est}: el modelo necesita un promedio diario "
                "válido de ese día.",
                **self._fechas_cercanas(self.tabla.index, est, f),
            )
        exactitud = self.metricas.get("resultados", {}).get("Naive Bayes (propio)", {}).get("exactitud")
        prediccion["modelo"] = "Naive Bayes categórico (implementación propia)" + (
            f"; exactitud en el periodo de prueba: {exactitud:.0%}" if exactitud else ""
        )
        return _json(prediccion)

    def search_docs(self, query: str) -> str:
        if not query or not str(query).strip():
            return _error("La consulta está vacía.")
        try:
            resultados = self.buscador.buscar(str(query))
        except FileNotFoundError as e:
            return _error(f"El índice de documentos no está construido. {e}")
        if not resultados:
            return _json({"resultados": [], "mensaje": "Sin contexto suficiente: los documentos no contienen información relevante para esta consulta."})
        return _json(
            {
                "resultados": [
                    {"cita": f"[Fuente: {r['titulo']}, p. {r['pagina']}]", "titulo": r["titulo"], "archivo": r["archivo"],
                     "pagina": r["pagina"], "similitud": r["similitud"], "texto": r["texto"]}
                    for r in resultados
                ]
            }
        )

    def list_stations(self) -> str:
        columnas = ["estacion", "municipio", "primer_dia", "ultimo_dia"]
        return _json({"estaciones": self.estaciones[columnas].to_dict("records")})

    # ------------------------------------------------------------------ despacho
    def ejecutar(self, nombre: str, argumentos: str | dict | None) -> str:
        """Ejecuta una herramienta por nombre. Nunca lanza: cualquier fallo vuelve como {"error": ...}."""
        registro = {
            "get_pm25": self.get_pm25,
            "predict_tomorrow": self.predict_tomorrow,
            "search_docs": self.search_docs,
            "list_stations": self.list_stations,
        }
        if nombre not in registro:
            return _error(f"La herramienta '{nombre}' no existe.", herramientas_disponibles=list(registro))
        try:
            if isinstance(argumentos, str):
                argumentos = json.loads(argumentos) if argumentos.strip() else {}
            argumentos = argumentos or {}
            if not isinstance(argumentos, dict):
                raise ValueError("los argumentos deben ser un objeto JSON")
        except (json.JSONDecodeError, ValueError) as e:
            return _error(f"Argumentos inválidos para {nombre}: {e}. Envía un objeto JSON con los parámetros.")
        try:
            return registro[nombre](**argumentos)
        except TypeError as e:
            return _error(f"Parámetros incorrectos para {nombre}: {e}")
        except Exception as e:  # noqa: BLE001 - el agente debe recibir el fallo como observación
            return _error(f"{nombre} falló: {type(e).__name__}: {e}")


# Esquemas que ve el LLM (formato de tool calling de OpenAI, compatible con Groq).
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_pm25",
            "description": (
                "Consulta el PM2.5 medido (promedio diario en µg/m³) y su categoría ICA en una estación del "
                "Valle de Aburrá. Con dias=1 devuelve ese día; con dias>1 devuelve la ventana de días que termina "
                "en 'fecha' con promedio, máximo, mínimo y serie. Úsala para cualquier dato histórico."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "estacion": {"type": "string", "description": "Nombre de la estación, por ejemplo 'TRÁFICO CENTRO'."},
                    "fecha": {"type": "string", "description": "Fecha en formato AAAA-MM-DD."},
                    "dias": {"type": "integer", "description": "Número de días de la ventana (1 a 60). Por defecto 1."},
                },
                "required": ["estacion", "fecha"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_tomorrow",
            "description": (
                "Predice la categoría ICA de PM2.5 del día SIGUIENTE a 'fecha' en una estación, con un modelo "
                "Naive Bayes. Devuelve la categoría más probable, las probabilidades de cada categoría, las variables "
                "usadas y, si ya se conoce, el valor real de ese día. Úsala para preguntas sobre 'mañana' o pronósticos."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "estacion": {"type": "string", "description": "Nombre de la estación."},
                    "fecha": {"type": "string", "description": "Fecha de referencia ('hoy') en formato AAAA-MM-DD; se predice el día siguiente."},
                },
                "required": ["estacion", "fecha"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Busca en los documentos oficiales (norma de calidad del aire, ICA, protocolo de episodios POECA) "
                "y devuelve fragmentos con su documento y página. Úsala para normativa, límites, niveles de "
                "prevención/alerta/emergencia, medidas y recomendaciones de salud. Cita siempre la fuente devuelta."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Consulta en español, con las palabras clave del tema."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_stations",
            "description": "Lista las estaciones de monitoreo disponibles con su municipio y el rango de fechas con datos.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
