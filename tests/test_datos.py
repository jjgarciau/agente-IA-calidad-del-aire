from datetime import date

import pandas as pd
import pytest

from src import config, download_data, ica, preprocess
from tests import sintetico


def test_ica_puntos_de_corte():
    # Resolución 2254 de 2017: 0-12, 13-37, 38-55, 56-150, 151-250, 251-500
    assert ica.categoria(0) == "Buena"
    assert ica.categoria(12.4) == "Buena"
    assert ica.categoria(12.5) == "Aceptable"
    assert ica.categoria(37.4) == "Aceptable"
    assert ica.categoria(37.5) == "Dañina para grupos sensibles"
    assert ica.categoria(55.4) == "Dañina para grupos sensibles"
    assert ica.categoria(56) == "Dañina"
    assert ica.categoria(151) == "Muy dañina"
    assert ica.categoria(300) == "Peligrosa"
    assert [ica.clase(v) for v in (5, 20, 40, 60, 200)] == [0, 1, 2, 3, 3]
    with pytest.raises(ValueError):
        ica.categoria(-1)


def test_meses_cubre_el_rango():
    m = download_data.meses(date(2024, 11, 15), date(2025, 2, 1))
    assert [a.strftime("%Y-%m") for a, _ in m] == ["2024-11", "2024-12", "2025-01", "2025-02"]
    assert m[1] == (date(2024, 12, 1), date(2025, 1, 1))


def test_descarga_pagina_y_es_reanudable(tmp_path, monkeypatch):
    crudo = sintetico.generar_crudo("2024-01-01", "2024-01-10")
    llamadas = []

    def falso_pedir(session, params, intentos=4):
        llamadas.append(params)
        assert "nombre_fgda='AMVA'" in params["$where"] and "msfl_code='PM2.5'" in params["$where"]
        pagina = crudo.iloc[params["$offset"]: params["$offset"] + params["$limit"]]
        return pagina.to_csv(index=False)

    monkeypatch.setattr(download_data, "_pedir", falso_pedir)
    monkeypatch.setattr(download_data, "LIMITE_PAGINA", 400)
    total = download_data.descargar(date(2024, 1, 1), date(2024, 1, 31), destino=tmp_path)
    assert total == len(crudo)
    assert len(llamadas) == len(crudo) // 400 + 1          # paginó hasta agotar
    assert (tmp_path / "pm25_2024-01.csv").exists()

    llamadas.clear()
    download_data.descargar(date(2024, 1, 1), date(2024, 1, 31), destino=tmp_path)
    assert llamadas == []                                   # segunda vez: no vuelve a pedir


def test_limpieza_y_promedio_diario():
    crudo = sintetico.generar_crudo("2024-01-01", "2024-12-31")
    horario = preprocess.limpiar_horario(crudo)
    assert horario["pm25"].max() <= 500                     # se fue el centinela 2554
    assert not horario.duplicated(["estacion", "fecha_hora"]).any()

    diario, estaciones, _ = preprocess.promedio_diario(horario)
    assert (diario["horas_validas"] >= config.HORAS_MINIMAS).all()
    assert "POLITÉCNICO JIC (S)" not in set(estaciones["estacion"])   # estación sin datos suficientes
    assert set(estaciones["estacion"]) == {e[1] for e in sintetico.ESTACIONES}
    assert set(diario["categoria"]) <= set(ica.NOMBRES)
    assert estaciones[["latitud", "longitud"]].notna().all().all()

    # El promedio coincide con un cálculo directo para un día cualquiera.
    fila = diario.iloc[100]
    sel = horario[(horario["estacion"] == fila["estacion"]) & (horario["fecha_hora"].dt.normalize() == fila["fecha"])]
    assert fila["pm25"] == pytest.approx(sel["pm25"].mean(), abs=0.05)
    assert fila["horas_validas"] == len(sel)


def test_carga_del_proyecto(proyecto):
    diario = preprocess.cargar_diario()
    assert pd.api.types.is_datetime64_any_dtype(diario["fecha"])
    assert len(preprocess.cargar_estaciones()) == 4


def test_descarta_meses_con_escala_corrupta():
    """Defecto real de la fuente: una estación sin punto decimal durante parte del periodo."""
    crudo = sintetico.generar_crudo("2023-01-01", "2025-12-31", corrupta_hasta="2024-01-01")
    horario = preprocess.limpiar_horario(crudo)

    sucio = preprocess.agregar_diario(horario)
    santa_2023 = sucio[(sucio["estacion"] == "SANTA ELENA") & (sucio["fecha"] < "2024-01-01")]
    assert len(santa_2023) > 0 and santa_2023["pm25"].median() > 80        # sobreviven días con valores x10

    diario, estaciones, descartes = preprocess.promedio_diario(horario)
    limpio_2023 = diario[(diario["estacion"] == "SANTA ELENA") & (diario["fecha"] < "2024-01-01")]
    assert limpio_2023.empty                                               # todo el periodo corrupto se fue
    assert set(descartes["estacion"]) == {"SANTA ELENA"} and (descartes["razon"] > preprocess.RAZON_ESCALA).all()
    assert descartes["dias_descartados"].sum() == len(santa_2023)
    assert diario["pm25"].max() < 80
    assert set(estaciones["estacion"]) == {e[1] for e in sintetico.ESTACIONES}   # la estación sigue, con 2024-2025

    # Las demás estaciones no pierden ni un día, ni siquiera en la temporada alta.
    for otra in ("TRÁFICO CENTRO", "VILLAHERMOSA", "CASA JUSTICIA"):
        assert (diario["estacion"] == otra).sum() == (sucio["estacion"] == otra).sum()
    fila = estaciones.set_index("estacion").loc["SANTA ELENA"]
    assert fila["primer_dia"] >= "2024-01-01" and fila["dias_validos"] == (diario["estacion"] == "SANTA ELENA").sum()


def test_datos_sanos_no_pierden_meses():
    horario = preprocess.limpiar_horario(sintetico.generar_crudo("2023-01-01", "2024-12-31"))
    _, _, descartes = preprocess.promedio_diario(horario)
    assert descartes.empty


def test_depurar_la_tabla_diaria_equivale_al_flujo_completo():
    """La depuración solo depende de la tabla diaria: aplicarla a una tabla ya agregada da lo mismo."""
    horario = preprocess.limpiar_horario(sintetico.generar_crudo("2023-01-01", "2025-12-31", corrupta_hasta="2024-01-01"))
    completo, _, _ = preprocess.promedio_diario(horario)
    previa = preprocess._con_dias_suficientes(preprocess.agregar_diario(horario), config.DIAS_MINIMOS_ESTACION)
    desde_tabla, _ = preprocess.depurar_diario(previa)
    columnas = ["fecha", "estacion", "pm25", "horas_validas", "categoria"]
    pd.testing.assert_frame_equal(completo[columnas].reset_index(drop=True), desde_tabla[columnas].reset_index(drop=True))
