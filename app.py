"""Dashboard interactivo: calidad del aire (PM2.5) en el Valle de Aburrá.

    streamlit run app.py

El usuario elige estación, fecha de referencia y ventana de tiempo; ve el mapa de
estaciones, la serie de tiempo y la predicción para el día siguiente, y conversa
con el agente, que usa las mismas herramientas y muestra qué llamó en cada respuesta.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config, features, ica
from src.agent import run_agent
from src.tools import Recursos

st.set_page_config(page_title="Aire Aburrá · PM2.5", page_icon="🌫️", layout="wide")

AVISO = "Herramienta analítica con fines académicos. No constituye consejo médico, legal ni de seguridad."
GRIS = "#9AA0A6"
EJEMPLOS = {   # etiqueta corta del botón -> pregunta que se envía
    "Aire de hoy": "¿Cómo estuvo el aire hoy en esta estación?",
    "Pronóstico": "¿Cómo estará mañana y qué se recomienda en ese nivel?",
    "Norma": "¿Cuál es el límite de PM2.5 en 24 horas según la norma?",
}


# ---------------------------------------------------------------------- carga
@st.cache_resource(show_spinner="Cargando datos y modelo…")
def cargar_recursos() -> Recursos:
    r = Recursos()
    _ = (r.diario, r.estaciones, r.tabla, r.serie)   # fuerza la carga una sola vez
    return r


try:
    R = cargar_recursos()
except FileNotFoundError as e:
    st.error(f"Faltan los datos procesados. {e}")
    st.code("python -m src.download_data\npython -m src.preprocess\npython -m src.train", language="bash")
    st.stop()

HAY_MODELO = config.MODEL_PATH.exists()
DESDE, HASTA = R.rango


# ---------------------------------------------------------------------- barra lateral
with st.sidebar:
    st.title("Aire Aburrá")
    st.caption("PM2.5 en el Valle de Aburrá · datos de SISAIRE / AMVA")

    nombres = list(R.estaciones["estacion"])
    etiquetas = {f["estacion"]: f"{f['estacion'].title()} · {f['municipio']}" for _, f in R.estaciones.iterrows()}
    por_defecto = nombres.index("TRÁFICO CENTRO") if "TRÁFICO CENTRO" in nombres else 0
    estacion = st.selectbox("Estación", nombres, index=por_defecto, format_func=etiquetas.get)

    # Fecha inicial: el último día de la estación con predicción y valor real para comparar.
    if "fecha" not in st.session_state:
        filas_est = R.tabla.loc[estacion]
        con_real = filas_est[filas_est["y"].notna()]
        st.session_state.fecha = (con_real.index.max() if len(con_real) else filas_est.index.max()).date()
    fecha = pd.Timestamp(
        st.date_input(
            "Fecha de referencia («hoy»)", min_value=DESDE.date(), max_value=HASTA.date(), format="YYYY-MM-DD", key="fecha",
            help="Los datos son históricos: elige el día que quieres analizar. «Mañana» es el día siguiente.",
        )
    )
    ventana = st.slider("Días a mostrar en la serie", 30, 365, 120, step=15)

    st.divider()
    st.caption(f"Datos: {DESDE:%Y-%m-%d} a {HASTA:%Y-%m-%d} · {len(nombres)} estaciones")
    st.caption(AVISO)

manana = fecha + pd.Timedelta(days=1)
hoy = R.serie.loc[(estacion, fecha)] if (estacion, fecha) in R.serie.index else None
prediccion = R.predecir(estacion, fecha) if HAY_MODELO else None
del_dia = R.diario[R.diario["fecha"] == fecha]


# ---------------------------------------------------------------------- encabezado
st.title("Calidad del aire en el Valle de Aburrá")
st.caption(f"{etiquetas[estacion]} · fecha de referencia {fecha:%Y-%m-%d}")

def tarjeta(columna, titulo: str, valor: str, detalle: str = "", color: str = GRIS) -> None:
    """Indicador con una franja del color de la categoría ICA."""
    columna.markdown(
        f"""<div style="border-left:6px solid {color};padding:0.35rem 0.8rem;margin-bottom:0.6rem">
        <div style="font-size:0.82rem;opacity:0.75">{titulo}</div>
        <div style="font-size:1.45rem;font-weight:600;line-height:1.3">{valor}</div>
        <div style="font-size:0.82rem;opacity:0.75;min-height:1.2em">{detalle}</div></div>""",
        unsafe_allow_html=True,
    )


c1, c2, c3, c4 = st.columns(4)
if hoy is not None:
    tarjeta(c1, "PM2.5 hoy (promedio 24 h)", f"{hoy['pm25']:.1f} µg/m³", hoy["categoria"], ica.COLORES[hoy["categoria"]])
else:
    tarjeta(c1, "PM2.5 hoy (promedio 24 h)", "Sin dato", "menos de 18 horas medidas")
if prediccion:
    prevista = prediccion["categoria_predicha"]
    tarjeta(c2, f"Predicción para el {manana:%Y-%m-%d}", prevista, f"probabilidad {prediccion['probabilidades'][prevista]:.0%}", ica.COLORES_CLASE[prevista])
    if "valor_real" in prediccion:
        real = prediccion["valor_real"]
        tarjeta(c3, f"Valor real del {manana:%Y-%m-%d}", f"{real['pm25_ug_m3']:.1f} µg/m³",
                f"{real['categoria']} · la predicción {'acertó' if prediccion['acierto'] else 'no acertó'}", ica.COLORES_CLASE[real["categoria"]])
    else:
        tarjeta(c3, f"Valor real del {manana:%Y-%m-%d}", "—", "aún sin dato")
else:
    tarjeta(c2, f"Predicción para el {manana:%Y-%m-%d}", "No disponible", "falta el dato de hoy o de ayer" if HAY_MODELO else "modelo sin entrenar")
    tarjeta(c3, f"Valor real del {manana:%Y-%m-%d}", "—")
if len(del_dia):
    promedio = float(del_dia["pm25"].mean())
    tarjeta(c4, "Promedio del valle hoy", f"{promedio:.1f} µg/m³", f"{len(del_dia)} estaciones con dato", ica.COLORES[ica.categoria(promedio)])
else:
    tarjeta(c4, "Promedio del valle hoy", "Sin dato")


# ---------------------------------------------------------------------- figuras
def figura_mapa() -> go.Figure:
    datos = R.estaciones.merge(del_dia[["estacion", "pm25", "categoria"]], on="estacion", how="left")
    datos["categoria"] = datos["categoria"].fillna("Sin dato")
    fig = go.Figure()
    for nombre in [*ica.NOMBRES, "Sin dato"]:
        grupo = datos[datos["categoria"] == nombre]
        if grupo.empty:
            continue
        valores = grupo["pm25"].map(lambda v: "sin dato" if pd.isna(v) else f"{v:.1f} µg/m³")
        fig.add_trace(
            go.Scattermap(
                lat=grupo["latitud"], lon=grupo["longitud"], mode="markers", name=nombre,
                marker={"size": [22 if e == estacion else 13 for e in grupo["estacion"]], "color": ica.COLORES.get(nombre, GRIS), "opacity": 0.9},
                text=[f"<b>{e.title()}</b><br>{m}<br>{v}" for e, m, v in zip(grupo["estacion"], grupo["municipio"], valores)],
                hovertemplate="%{text}<extra>" + nombre + "</extra>",
            )
        )
    fig.update_layout(
        map={"style": "open-street-map", "center": {"lat": float(datos["latitud"].mean()), "lon": float(datos["longitud"].mean())}, "zoom": 9.6},
        margin={"l": 0, "r": 0, "t": 0, "b": 0}, height=380, showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 0.01, "x": 0.01, "bgcolor": "rgba(255,255,255,0.85)", "font": {"color": "#222"}},
    )
    return fig


def figura_serie() -> go.Figure:
    inicio = fecha - pd.Timedelta(days=ventana - 1)
    propia = R.serie.loc[estacion].loc[inicio:fecha]
    valle = R.diario[(R.diario["fecha"] >= inicio) & (R.diario["fecha"] <= fecha)].groupby("fecha")["pm25"].mean()
    tope = max(60.0, float(propia["pm25"].max() if len(propia) else 0) * 1.15, float(valle.max() if len(valle) else 0) * 1.15)

    fig = go.Figure()
    bordes = [0, *features.BORDES_CLASE, tope]
    for i, nombre in enumerate(ica.CLASES):                      # bandas de color del ICA
        if bordes[i] < tope:
            fig.add_hrect(y0=bordes[i], y1=min(bordes[i + 1], tope), fillcolor=ica.COLORES_CLASE[nombre], opacity=0.13, line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=valle.index, y=valle.round(1), name="Promedio del valle", mode="lines", line={"color": GRIS, "width": 1.5, "dash": "dot"},
                             hovertemplate="%{x|%Y-%m-%d}: %{y} µg/m³<extra>Valle</extra>"))
    fig.add_trace(go.Scatter(x=propia.index, y=propia["pm25"], name=estacion.title(), mode="lines+markers", line={"color": "#1F4E79", "width": 2},
                             marker={"size": 5, "color": [ica.COLORES[c] for c in propia["categoria"]], "line": {"color": "#1F4E79", "width": 0.5}},
                             text=propia["categoria"], hovertemplate="%{x|%Y-%m-%d}: %{y} µg/m³<br>%{text}<extra></extra>"))
    if prediccion and "valor_real" in prediccion:
        fig.add_trace(go.Scatter(x=[manana], y=[prediccion["valor_real"]["pm25_ug_m3"]], name="Día siguiente (real)", mode="markers",
                                 marker={"size": 11, "symbol": "diamond-open", "color": "#1F4E79", "line": {"width": 2}},
                                 hovertemplate="%{x|%Y-%m-%d}: %{y} µg/m³<extra>Real del día predicho</extra>"))
    fig.add_vline(x=fecha, line_width=1, line_dash="dash", line_color="#555")
    fig.update_layout(
        height=330, margin={"l": 10, "r": 10, "t": 10, "b": 10}, hovermode="x unified",
        yaxis={"title": "PM2.5 (µg/m³)", "range": [0, tope]}, xaxis={"title": None},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0},
    )
    return fig


def figura_probabilidades() -> go.Figure:
    clases = ica.CLASES[::-1]
    probabilidades = [prediccion["probabilidades"][c] for c in clases]
    fig = go.Figure(go.Bar(x=probabilidades, y=clases, orientation="h", marker_color=[ica.COLORES_CLASE[c] for c in clases],
                           text=[f"{p:.0%}" for p in probabilidades], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_layout(height=190, margin={"l": 10, "r": 40, "t": 5, "b": 5}, xaxis={"range": [0, 1.08], "tickformat": ".0%", "title": None}, yaxis={"title": None})
    return fig


# ---------------------------------------------------------------------- chat
def mostrar_traza(mensaje: dict) -> None:
    traza = mensaje.get("traza") or []
    if traza:
        with st.expander(f"Herramientas usadas ({len(traza)}) · {mensaje.get('iteraciones', 0)} iteraciones"):
            for t in traza:
                estado = "⚠️ error" if t["error"] else "✓"
                st.markdown(f"**{t['herramienta']}** `{json.dumps(t['argumentos'], ensure_ascii=False)}` {estado}")
                st.json(t["resultado"], expanded=False)
    if mensaje.get("fuentes"):
        st.caption("Documentos consultados: " + " · ".join(f"{f['titulo']}, p. {f['pagina']}" for f in mensaje["fuentes"]))
    if mensaje.get("limite"):
        st.caption("Se alcanzó el límite de iteraciones del agente (T_max).")


def panel_chat() -> None:
    st.subheader("Asistente")
    st.session_state.setdefault("chat", [])
    if not config.hay_api_key():
        st.info("Para conversar con el asistente configura `GROQ_API_KEY` en el archivo `.env` (ver README) y reinicia la aplicación.")

    caja = st.container(height=330)
    with caja:
        if not st.session_state.chat:
            st.caption("Pregunta por un dato, un pronóstico o la normativa. El asistente usa la estación y la fecha seleccionadas.")
        for m in st.session_state.chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m["role"] == "assistant":
                    mostrar_traza(m)

    columnas = st.columns(len(EJEMPLOS))
    for columna, (etiqueta, ejemplo) in zip(columnas, EJEMPLOS.items()):
        if columna.button(etiqueta, width="stretch", disabled=not config.hay_api_key(), help=ejemplo):
            st.session_state.pendiente = ejemplo
    escrita = st.chat_input("Escribe tu pregunta…", disabled=not config.hay_api_key())
    pregunta = st.session_state.pop("pendiente", None) or escrita

    if pregunta:
        historial = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat]
        st.session_state.chat.append({"role": "user", "content": pregunta})
        with caja:
            with st.chat_message("user"):
                st.markdown(pregunta)
            with st.chat_message("assistant"), st.spinner("Consultando…"):
                r = run_agent(pregunta, R, estacion=estacion, fecha=f"{fecha:%Y-%m-%d}", historial=historial)
        st.session_state.chat.append(
            {"role": "assistant", "content": r.respuesta, "traza": r.traza, "fuentes": r.fuentes, "iteraciones": r.iteraciones, "limite": r.limite_alcanzado}
        )
        st.rerun()
    if st.session_state.chat and st.button("Borrar conversación"):
        st.session_state.chat = []
        st.rerun()


# ---------------------------------------------------------------------- pestañas
tab_panel, tab_modelo, tab_fuentes = st.tabs(["Panel", "Modelo y evaluación", "Datos y fuentes"])

with tab_panel:
    izquierda, derecha = st.columns([3, 2], gap="large")
    with izquierda:
        st.subheader(f"Estaciones el {fecha:%Y-%m-%d}")
        st.plotly_chart(figura_mapa(), width="stretch", config={"displayModeBar": False})
        st.subheader(f"PM2.5 diario · últimos {ventana} días")
        st.plotly_chart(figura_serie(), width="stretch", config={"displayModeBar": False})
        st.caption("Las bandas de color son las categorías del ICA (Resolución 2254 de 2017). La línea vertical marca la fecha de referencia.")
    with derecha:
        st.subheader(f"Predicción para el {manana:%Y-%m-%d}")
        if prediccion:
            st.plotly_chart(figura_probabilidades(), width="stretch", config={"displayModeBar": False})
            v = prediccion["variables_usadas"]
            st.caption(
                f"Naive Bayes con lo conocido el {fecha:%Y-%m-%d}: nivel de hoy {v['nivel_hoy']} µg/m³, ayer «{v['clase_ayer']}», "
                f"tendencia {v['tendencia']}, temporada {v['temporada']}, día objetivo {v['dia_manana']}, valle {v['nivel_valle']} µg/m³."
            )
        elif not HAY_MODELO:
            st.warning("El modelo aún no está entrenado. Ejecuta `python -m src.train`.")
        else:
            st.info("No hay predicción para esta fecha: el modelo necesita un dato válido de este día y del anterior.")
        panel_chat()

with tab_modelo:
    m = R.metricas
    if not m:
        st.info("Aún no hay métricas. Ejecuta `python -m src.train`.")
    else:
        p = m["particion"]
        st.subheader("Naive Bayes frente a las líneas base")
        st.caption(
            f"Entrenamiento: {p['entrenamiento']['n']:,} días-estación ({p['entrenamiento']['desde']} a {p['entrenamiento']['hasta']}). "
            f"Prueba: {p['prueba']['n']:,} días-estación ({p['prueba']['desde']} a {p['prueba']['hasta']}), nunca vistos al entrenar. "
            f"Suavizado α = {m['alpha']}, elegido por validación."
        )
        tabla = pd.DataFrame(
            [{"Modelo": k, "Exactitud": v["exactitud"], "F1-macro": v["f1_macro"], "Precisión de episodio": v["precision_episodio"], "Recall de episodio": v["recall_episodio"]}
             for k, v in m["resultados"].items()]
        )
        st.dataframe(tabla, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in tabla.columns[1:]})
        st.caption("«Episodio» = Dañina para grupos sensibles o peor, que es lo que interesa alertar.")

        a, b = st.columns(2, gap="large")
        with a:
            st.subheader("Matriz de confusión (Naive Bayes)")
            matriz = m["resultados"]["Naive Bayes (propio)"]["matriz_confusion"]
            cortas = ["Buena", "Aceptable", "Grupos sens.", "Dañina o peor"]
            fig = go.Figure(go.Heatmap(z=matriz, x=cortas, y=cortas, colorscale="Blues", showscale=False, text=matriz, texttemplate="%{text}",
                                       hovertemplate="Real: %{y}<br>Predicha: %{x}<br>Días: %{z}<extra></extra>"))
            fig.update_layout(height=330, margin={"l": 10, "r": 10, "t": 10, "b": 10}, xaxis={"title": "Clase predicha"}, yaxis={"title": "Clase real", "autorange": "reversed"})
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        with b:
            st.subheader("Exactitud por estación")
            pe = pd.DataFrame(m["por_estacion"]).sort_values("naive_bayes")
            fig = go.Figure()
            fig.add_trace(go.Bar(y=pe["estacion"].str.title(), x=pe["persistencia"], name="Persistencia", orientation="h", marker_color=GRIS))
            fig.add_trace(go.Bar(y=pe["estacion"].str.title(), x=pe["naive_bayes"], name="Naive Bayes", orientation="h", marker_color="#1F4E79"))
            fig.update_layout(height=max(330, 26 * len(pe)), margin={"l": 10, "r": 10, "t": 10, "b": 10}, barmode="group",
                              xaxis={"tickformat": ".0%", "range": [0, 1]}, legend={"orientation": "h", "y": 1.02, "yanchor": "bottom"})
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        st.subheader("Ablación: el modelo sin una variable")
        st.dataframe(pd.DataFrame(m["ablacion"]).rename(columns={"sin_variable": "Variable retirada", "exactitud": "Exactitud", "f1_macro": "F1-macro", "recall_episodio": "Recall de episodio"}),
                     hide_index=True, width="stretch")

    resultados_agente = config.ROOT / "eval" / "resultados.csv"
    st.subheader("Evaluación del agente")
    if resultados_agente.exists():
        ev = pd.read_csv(resultados_agente)
        e1, e2, e3 = st.columns(3)
        e1.metric("Preguntas de prueba", len(ev))
        e2.metric("Herramienta correcta", f"{ev['herramienta_correcta'].mean():.0%}")
        con_fuente = ev[ev["fuente_esperada"].notna() & (ev["fuente_esperada"].astype(str).str.strip() != "")]
        e3.metric("Citó la fuente esperada", f"{con_fuente['cito_fuente'].mean():.0%}" if len(con_fuente) else "—")
        st.dataframe(ev[["id", "pregunta", "herramienta_esperada", "herramientas_llamadas", "herramienta_correcta", "cito_fuente", "iteraciones"]], hide_index=True, width="stretch")
    else:
        st.info("Aún no se ha corrido la evaluación. Ejecuta `python -m eval.run_eval`.")

with tab_fuentes:
    st.subheader("Datos")
    st.markdown(
        "Mediciones horarias de PM2.5 de la red del Área Metropolitana del Valle de Aburrá, publicadas en el conjunto "
        "[Calidad del Aire en Colombia](https://www.datos.gov.co/Ambiente-y-Desarrollo-Sostenible/Calidad-del-Aire-en-Colombia/g4t8-zkc3) "
        "(SISAIRE, IDEAM) de datos.gov.co. Se usa el promedio diario de los días con al menos "
        f"{config.HORAS_MINIMAS} horas medidas."
    )
    st.dataframe(R.estaciones.rename(columns=str.capitalize), hide_index=True, width="stretch")
    st.subheader("Documentos del asistente")
    if Path(config.SOURCES_CSV).exists():
        with open(config.SOURCES_CSV, encoding="utf-8-sig", newline="") as f:
            for fila in csv.DictReader(f):
                st.markdown(f"- [{fila['titulo']}]({fila['url']}) — {fila['entidad']} (consultado el {fila['fecha_consulta']})")
    st.caption(AVISO)
