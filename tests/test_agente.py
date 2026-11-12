import json
from types import SimpleNamespace

import openai
import pytest

from src import agent, config, preprocess, train
from src.tools import TOOLS, Recursos


class BuscadorFalso:
    def buscar(self, consulta, k=4, umbral=None):
        if "pm2.5" in consulta.lower() or "alerta" in consulta.lower():
            return [{"id": "x", "archivo": "res_2254_2017.pdf", "titulo": "Resolución 2254 de 2017", "pagina": 3,
                     "texto": "El nivel máximo permisible de PM2.5 para 24 horas es 37 µg/m3.", "similitud": 0.71}]
        return []


@pytest.fixture(scope="module")
def recursos(proyecto):
    modelo, metricas = train.entrenar(preprocess.cargar_diario())
    modelo.guardar(config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(metricas, ensure_ascii=False), encoding="utf-8")
    return Recursos(buscador=BuscadorFalso())


# ------------------------------------------------------------------ herramientas
def test_get_pm25_dia_y_ventana(recursos):
    fila = recursos.diario.iloc[500]
    r = json.loads(recursos.get_pm25(fila["estacion"], f"{fila['fecha']:%Y-%m-%d}"))
    assert r["pm25_ug_m3"] == fila["pm25"] and r["categoria_ica"] == fila["categoria"]

    r = json.loads(recursos.get_pm25("trafico centro", "2025-03-15", dias=7))       # sin tilde ni mayúsculas
    assert r["estacion"] == "TRÁFICO CENTRO" and 1 <= r["dias_con_dato"] <= 7
    assert len(r["serie"]) == r["dias_con_dato"] and r["hasta"] <= "2025-03-15"


def test_resolucion_de_estaciones(recursos):
    assert recursos.resolver_estacion("Villahermosa")[0] == "VILLAHERMOSA"
    assert recursos.resolver_estacion("estación de Itagüí")[0] == "CASA JUSTICIA"      # por municipio único
    assert recursos.resolver_estacion("villa hermosa")[0] == "VILLAHERMOSA"            # coincidencia aproximada
    estacion, sugerencias = recursos.resolver_estacion("Medellín")                     # ambiguo: varias en Medellín
    assert estacion is None and len(sugerencias) == 3


def test_errores_como_observacion(recursos):
    assert "No existe la estación" in json.loads(recursos.get_pm25("Marte", "2025-03-15"))["error"]
    assert "quiso_decir" in json.loads(recursos.get_pm25("Medellín", "2025-03-15"))
    assert "fuera de los datos" in json.loads(recursos.get_pm25("VILLAHERMOSA", "2027-01-01"))["error"]
    assert "Fecha inválida" in json.loads(recursos.predict_tomorrow("VILLAHERMOSA", "15 de marzo"))["error"]
    assert "no existe" in json.loads(recursos.ejecutar("borrar_todo", "{}"))["error"]
    assert "Argumentos inválidos" in json.loads(recursos.ejecutar("get_pm25", "{no es json"))["error"]
    assert "Parámetros incorrectos" in json.loads(recursos.ejecutar("get_pm25", '{"ciudad": "x"}'))["error"]


def test_dia_sin_dato_sugiere_fechas(recursos):
    fechas = set(recursos.serie.loc["SANTA ELENA"].index)
    faltante = next(d for d in recursos.diario["fecha"].sort_values().unique() if d not in fechas)
    r = json.loads(recursos.get_pm25("SANTA ELENA", f"{faltante:%Y-%m-%d}"))
    assert "error" in r and ("fecha_valida_anterior" in r or "fecha_valida_posterior" in r)


def test_predict_tomorrow(recursos):
    r = json.loads(recursos.predict_tomorrow("VILLAHERMOSA", "2025-03-15"))
    if "error" in r:                                         # el día sintético pudo quedar sin dato
        pytest.skip(r["error"])
    assert r["fecha_predicha"] == "2025-03-16"
    assert sum(r["probabilidades"].values()) == pytest.approx(1.0, abs=0.01)
    assert r["categoria_predicha"] == max(r["probabilidades"], key=r["probabilidades"].get)
    assert set(r["variables_usadas"]) == {"nivel_hoy", "clase_ayer", "tendencia", "temporada", "dia_manana", "nivel_valle"}
    assert "valor_real" in r and isinstance(r["acierto"], bool)


def test_search_docs(recursos):
    r = json.loads(recursos.search_docs("límite de PM2.5"))
    assert r["resultados"][0]["cita"] == "[Fuente: Resolución 2254 de 2017, p. 3]"
    assert json.loads(recursos.search_docs("receta de arepas"))["resultados"] == []
    sin_indice = Recursos()
    assert "índice" in json.loads(sin_indice.search_docs("pm2.5"))["error"]


def test_esquemas_coinciden_con_las_funciones(recursos):
    for t in TOOLS:
        nombre = t["function"]["name"]
        assert callable(getattr(recursos, nombre))
        assert set(t["function"]["parameters"].get("required", [])) <= set(t["function"]["parameters"]["properties"])


# ------------------------------------------------------------------ loop agéntico
def _respuesta(contenido=None, llamadas=None):
    tool_calls = [
        SimpleNamespace(id=f"call_{i}", type="function", function=SimpleNamespace(name=n, arguments=a))
        for i, (n, a) in enumerate(llamadas or [])
    ] or None
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=contenido, tool_calls=tool_calls))])


class ClienteFalso:
    """Simula el endpoint de chat: devuelve respuestas guionizadas y guarda lo que recibió."""

    def __init__(self, guion):
        self.guion, self.recibido = list(guion), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.recibido.append({**kwargs, "messages": list(kwargs["messages"])})   # copia: el loop sigue agregando
        paso = self.guion.pop(0)
        if isinstance(paso, Exception):
            raise paso
        return paso


def test_loop_encadena_herramientas_y_cita(recursos):
    cliente = ClienteFalso([
        _respuesta(llamadas=[("predict_tomorrow", '{"estacion": "VILLAHERMOSA", "fecha": "2025-03-15"}')]),
        _respuesta(llamadas=[("search_docs", '{"query": "nivel de alerta PM2.5"}')]),
        _respuesta("Mañana se espera Aceptable [Fuente: Resolución 2254 de 2017, p. 3]."),
    ])
    r = agent.run_agent("¿Cómo estará mañana y qué dice la norma?", recursos, "VILLAHERMOSA", "2025-03-15", client=cliente)
    assert r.herramientas == ["predict_tomorrow", "search_docs"]
    assert r.iteraciones == 3 and not r.limite_alcanzado and r.error is None
    assert r.fuentes == [{"titulo": "Resolución 2254 de 2017", "archivo": "res_2254_2017.pdf", "pagina": 3}]
    assert "[Fuente:" in r.respuesta

    # El contexto del dashboard llega al system prompt y la observación vuelve con rol "tool".
    assert "VILLAHERMOSA" in cliente.recibido[0]["messages"][0]["content"]
    assert "2025-03-15" in cliente.recibido[0]["messages"][0]["content"]
    ultimo = cliente.recibido[-1]["messages"]
    assert [m["role"] for m in ultimo[-4:]] == ["assistant", "tool", "assistant", "tool"]
    assert ultimo[-1]["tool_call_id"] == "call_0"


def test_error_de_herramienta_vuelve_como_observacion(recursos):
    cliente = ClienteFalso([
        _respuesta(llamadas=[("get_pm25", '{"estacion": "Marte", "fecha": "2025-03-15"}')]),
        _respuesta(llamadas=[("get_pm25", '{"estacion": "VILLAHERMOSA", "fecha": "2025-03-14"}')]),
        _respuesta("Listo."),
    ])
    r = agent.run_agent("PM2.5 en Marte", recursos, client=cliente)
    assert [t["error"] for t in r.traza][0] is True
    observacion = json.loads(cliente.recibido[1]["messages"][-1]["content"])
    assert "No existe la estación" in observacion["error"]      # el LLM vio el error y pudo corregir
    assert r.respuesta == "Listo."


def test_t_max_fuerza_respuesta_final(recursos):
    pide = _respuesta(llamadas=[("list_stations", "{}")])
    cliente = ClienteFalso([pide, pide, pide, _respuesta("Respuesta parcial.")])
    r = agent.run_agent("bucle", recursos, client=cliente, t_max=3)
    assert r.limite_alcanzado and r.iteraciones == 3 and len(r.traza) == 3
    assert cliente.recibido[-1]["tool_choice"] == "none"         # última llamada: sin herramientas
    assert r.respuesta == "Respuesta parcial."


def test_reintenta_ante_limite_de_peticiones(recursos, monkeypatch):
    esperas = []
    monkeypatch.setattr(agent.time, "sleep", esperas.append)
    limite = openai.RateLimitError("429", response=SimpleNamespace(status_code=429, headers={}, request=None), body=None)
    cliente = ClienteFalso([limite, limite, _respuesta("Hola.")])
    r = agent.run_agent("hola", recursos, client=cliente)
    assert r.respuesta == "Hola." and esperas == [2, 4]


def test_fallo_del_proveedor_no_rompe(recursos, monkeypatch):
    monkeypatch.setattr(agent.time, "sleep", lambda s: None)
    r = agent.run_agent("hola", recursos, client=ClienteFalso([RuntimeError("sin red")]))
    assert r.error and "No pude completar" in r.respuesta

    monkeypatch.setattr(config, "LLM_API_KEY", "")
    r = agent.run_agent("hola", recursos)
    assert "GROQ_API_KEY" in r.respuesta


def test_historial_se_recorta(recursos):
    historial = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turno {i}"} for i in range(20)]
    cliente = ClienteFalso([_respuesta("ok")])
    agent.run_agent("nueva", recursos, historial=historial, client=cliente)
    mensajes = cliente.recibido[0]["messages"]
    assert len(mensajes) == 1 + 6 + 1 and mensajes[-1]["content"] == "nueva" and mensajes[1]["content"] == "turno 14"
