"""Agente LLM: loop agéntico manual con tool calling (Clase 12 del curso).

El loop es el de clase: llamar al LLM -> si pide herramientas, ejecutarlas y devolverle
el resultado como observación -> repetir. Sobre esa base se agrega:

  - T_max: como máximo `t_max` iteraciones con herramientas. Si se agota, se pide una
    respuesta final sin herramientas y se avisa que se alcanzó el límite.
  - Errores como observación: un fallo de una herramienta nunca rompe el loop; vuelve
    al LLM como {"error": ...} para que corrija los argumentos o lo explique.
  - Reintentos con espera exponencial ante límites de peticiones o fallos de red.
  - Traza: cada llamada (herramienta, argumentos, resultado) queda registrada para
    mostrarla en el dashboard y para la evaluación automática.

    python -m src.agent "¿Cómo estuvo el aire en Tráfico Centro el 2025-03-15?"
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field

from src import config
from src.tools import TOOLS, Recursos

SYSTEM_PROMPT = """Eres AireAburrá, un asistente sobre la calidad del aire (PM2.5) en el Valle de Aburrá, Colombia.

Contexto de esta conversación:
- Los datos disponibles van de {desde} a {hasta}. No hay datos en tiempo real.
- "Hoy" es la fecha de referencia {fecha}; "mañana" es el día siguiente a esa fecha.
- Estación seleccionada por el usuario: {estacion}. Úsala cuando la pregunta no nombre otra.

Reglas:
1. Nunca inventes cifras. Todo valor de PM2.5 debe salir de get_pm25 y todo pronóstico de predict_tomorrow.
2. Para normativa, límites, niveles de alerta, medidas o recomendaciones usa search_docs y responde solo con lo que digan los fragmentos devueltos. Cita cada afirmación tomada de ellos copiando el campo "cita" del fragmento, con corchetes simples: [Fuente: <título>, p. <página>].
3. Si search_docs no devuelve resultados, o los fragmentos no responden exactamente lo preguntado, di que los documentos disponibles no contienen esa información. No completes con conocimiento propio ni agregues medidas o recomendaciones que no estén en los fragmentos.
4. Usa un número de un fragmento solo si el texto deja claro a qué contaminante, tiempo de exposición y categoría corresponde. En tablas con varias columnas, si no es evidente cuál es la de PM2.5, no lo adivines.
5. Si una herramienta devuelve un error, corrige los argumentos y reintenta si tiene sentido; si no, explica el problema y propone una alternativa (otra fecha u otra estación).
6. Si la pregunta combina datos y normativa, encadena las herramientas necesarias antes de responder.
7. Responde en español, en menos de 150 palabras, indicando la estación, la fecha y la categoría ICA de cada dato. Al dar un pronóstico menciona la probabilidad de la categoría prevista.
8. Esta es una herramienta académica y no es consejo médico: cuando des recomendaciones de salud, recuérdalo en una frase.
9. Si la pregunta no trata de la calidad del aire en el Valle de Aburrá, dilo con amabilidad y no uses herramientas."""

AVISO_LIMITE = (
    "Se alcanzó el límite de pasos con herramientas. Responde ahora con la información que ya tienes, "
    "sin pedir más herramientas, e indica qué parte quedó sin resolver."
)


@dataclass
class Resultado:
    respuesta: str
    traza: list[dict] = field(default_factory=list)       # una entrada por llamada a herramienta
    iteraciones: int = 0                                   # llamadas al LLM dentro del loop
    limite_alcanzado: bool = False
    error: str | None = None

    @property
    def herramientas(self) -> list[str]:
        return [t["herramienta"] for t in self.traza]

    @property
    def fuentes(self) -> list[dict]:
        """Documentos y páginas que devolvió search_docs, sin repetir."""
        vistas, out = set(), []
        for t in self.traza:
            if t["herramienta"] != "search_docs" or not isinstance(t["resultado"], dict):
                continue
            for r in t["resultado"].get("resultados", []):
                clave = (r["archivo"], r["pagina"])
                if clave not in vistas:
                    vistas.add(clave)
                    out.append({"titulo": r["titulo"], "archivo": r["archivo"], "pagina": r["pagina"]})
        return out


def normalizar_citas(texto: str) -> str:
    """Deja todas las citas en el formato [Fuente: título, p. N].

    Algunos modelos escriben las citas con sus propios símbolos, por ejemplo
    【1†Fuente: título, p. 9】. Se convierten a corchetes simples y se quita el índice.
    """
    texto = texto.replace("【", "[").replace("】", "]")
    return re.sub(r"\[\s*\d+\s*†\s*(?=Fuente:)", "[", texto)


def _llamar_llm(client, intentos: int = 4, **kwargs):
    """Llama al LLM reintentando con espera exponencial los fallos transitorios (429, red, 5xx)."""
    import openai

    transitorios = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError)
    for i in range(intentos):
        try:
            return client.chat.completions.create(**kwargs)
        except transitorios:
            if i == intentos - 1:
                raise
            time.sleep(2 ** (i + 1))


def _a_dict(mensaje) -> dict:
    """Convierte el mensaje del asistente a un dict estándar (algunos proveedores agregan campos extra)."""
    d = {"role": "assistant", "content": mensaje.content or ""}
    if mensaje.tool_calls:
        d["tool_calls"] = [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
            for tc in mensaje.tool_calls
        ]
    return d


def run_agent(
    pregunta: str,
    recursos: Recursos | None = None,
    estacion: str | None = None,
    fecha: str | None = None,
    historial: list[dict] | None = None,
    client=None,
    model: str | None = None,
    t_max: int | None = None,
) -> Resultado:
    """Ejecuta el loop agéntico para una pregunta y devuelve la respuesta con su traza.

    estacion / fecha: selección actual del dashboard (contexto para "hoy" y "mañana").
    historial: turnos previos de la conversación [{"role": "user"|"assistant", "content": ...}].
    """
    recursos = recursos or Recursos()
    t_max = t_max or config.T_MAX
    model = model or config.LLM_MODEL
    try:
        client = client or config.get_client()
    except Exception as e:  # noqa: BLE001
        return Resultado(respuesta=f"No se pudo iniciar el asistente: {e}", error=str(e))

    desde, hasta = recursos.rango
    system = SYSTEM_PROMPT.format(
        desde=f"{desde:%Y-%m-%d}", hasta=f"{hasta:%Y-%m-%d}",
        fecha=fecha or f"{hasta:%Y-%m-%d}", estacion=estacion or "ninguna",
    )
    mensajes = [{"role": "system", "content": system}, *(historial or [])[-6:], {"role": "user", "content": pregunta}]
    resultado = Resultado(respuesta="")

    try:
        for _ in range(t_max):
            respuesta = _llamar_llm(client, model=model, messages=mensajes, tools=TOOLS, tool_choice="auto", temperature=0.2)
            resultado.iteraciones += 1
            mensaje = respuesta.choices[0].message

            if not mensaje.tool_calls:                      # el modelo ya tiene la respuesta final
                resultado.respuesta = normalizar_citas((mensaje.content or "").strip())
                return resultado

            mensajes.append(_a_dict(mensaje))
            for tc in mensaje.tool_calls:                   # ejecutar cada herramienta pedida
                observacion = recursos.ejecutar(tc.function.name, tc.function.arguments)
                try:
                    argumentos = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    argumentos = tc.function.arguments
                contenido = json.loads(observacion)
                resultado.traza.append(
                    {"iteracion": resultado.iteraciones, "herramienta": tc.function.name, "argumentos": argumentos,
                     "resultado": contenido, "error": isinstance(contenido, dict) and "error" in contenido}
                )
                mensajes.append({"role": "tool", "tool_call_id": tc.id, "content": observacion})

        # Se agotó T_max y el modelo seguía pidiendo herramientas: forzar una respuesta final.
        resultado.limite_alcanzado = True
        mensajes.append({"role": "user", "content": AVISO_LIMITE})
        respuesta = _llamar_llm(client, model=model, messages=mensajes, tools=TOOLS, tool_choice="none", temperature=0.2)
        texto = normalizar_citas((respuesta.choices[0].message.content or "").strip())
        resultado.respuesta = texto or "No logré completar la consulta dentro del límite de pasos. Intenta con una pregunta más concreta."
        return resultado

    except Exception as e:  # noqa: BLE001 - la interfaz nunca debe caerse por un fallo del proveedor
        resultado.error = f"{type(e).__name__}: {e}"
        resultado.respuesta = (
            "No pude completar la consulta por un problema con el servicio del modelo de lenguaje. "
            f"Detalle: {resultado.error}"
        )
        return resultado


def main() -> None:
    p = argparse.ArgumentParser(description="Hace una pregunta al agente e imprime la traza de herramientas.")
    p.add_argument("pregunta")
    p.add_argument("--estacion", default=None)
    p.add_argument("--fecha", default=None, help="fecha de referencia AAAA-MM-DD")
    a = p.parse_args()

    r = run_agent(a.pregunta, estacion=a.estacion, fecha=a.fecha)
    for t in r.traza:
        marca = "ERROR" if t["error"] else "ok"
        print(f"[{t['iteracion']}] {t['herramienta']}({json.dumps(t['argumentos'], ensure_ascii=False)}) -> {marca}")
        print("    " + json.dumps(t["resultado"], ensure_ascii=False)[:300])
    print(f"\nIteraciones: {r.iteraciones}  |  Límite alcanzado: {'sí' if r.limite_alcanzado else 'no'}")
    print("\n" + r.respuesta)


if __name__ == "__main__":
    main()
