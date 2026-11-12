import hashlib
import re

import numpy as np
import pytest

from src import rag


class EmbedderFalso:
    """Bolsa de palabras con hashing: determinista y sin descargar ningún modelo."""

    nombre = "falso-para-pruebas"
    dim = 512

    def codificar(self, textos):
        vectores = np.zeros((len(textos), self.dim), dtype="float32")
        for i, texto in enumerate(textos):
            for palabra in re.findall(r"[a-záéíóúñ0-9.]+", texto.lower()):
                if len(palabra) > 3:
                    vectores[i, int(hashlib.md5(palabra.encode()).hexdigest(), 16) % self.dim] += 1.0
        normas = np.linalg.norm(vectores, axis=1, keepdims=True)
        return vectores / np.where(normas == 0, 1, normas)


def pdf_minimo(paginas: list[str]) -> bytes:
    """Arma a mano un PDF válido con una línea de texto por página."""
    objetos = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"]
    hijos = []
    for texto in paginas:
        contenido = f"BT /F1 12 Tf 72 720 Td ({texto}) Tj ET".encode("cp1252")
        n_contenido = len(objetos) + 1
        objetos.append(b"<< /Length %d >>\nstream\n" % len(contenido) + contenido + b"\nendstream")
        hijos.append(len(objetos) + 1)
        objetos.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {n_contenido} 0 R /Resources << /Font << /F1 3 0 R >> >> >>")
    objetos[1] = f"<< /Type /Pages /Kids [{' '.join(f'{h} 0 R' for h in hijos)}] /Count {len(hijos)} >>"

    salida, posiciones = bytearray(b"%PDF-1.4\n"), []
    for i, obj in enumerate(objetos, start=1):
        posiciones.append(len(salida))
        cuerpo = obj if isinstance(obj, bytes) else obj.encode("cp1252")
        salida += b"%d 0 obj\n" % i + cuerpo + b"\nendobj\n"
    inicio_xref = len(salida)
    salida += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objetos) + 1)
    for p in posiciones:
        salida += b"%010d 00000 n \n" % p
    salida += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objetos) + 1, inicio_xref)
    return bytes(salida)


@pytest.fixture()
def corpus(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "norma.pdf").write_bytes(
        pdf_minimo(
            [
                "El nivel maximo permisible de PM2.5 para un tiempo de exposicion de 24 horas es de 37 microgramos por metro cubico.",
                "El nivel de prevencion por PM2.5 se declara cuando la concentracion de 24 horas esta entre 38 y 55 microgramos.",
            ]
        )
    )
    (docs / "recomendaciones.txt").write_text(
        "Cuando la calidad del aire es dañina para grupos sensibles, los niños, adultos mayores y personas con "
        "enfermedades respiratorias deben reducir el ejercicio prolongado al aire libre.",
        encoding="utf-8",
    )
    (docs / "README.md").write_text("Este archivo de instrucciones no debe indexarse nunca en el corpus.", encoding="utf-8")
    (docs / "fuentes.csv").write_text(
        "archivo,titulo,entidad,url,fecha_consulta\nnorma.pdf,Resolución de prueba,MinAmbiente,http://x,2026-11-12\n", encoding="utf-8"
    )
    return docs


def test_trocear_respeta_tamano_y_no_pierde_texto():
    texto = " ".join(f"Esta es la oración número {i} sobre la calidad del aire en el valle." for i in range(40))
    trozos = rag.trocear(texto, 300)
    assert len(trozos) > 3 and all(len(t) <= 300 for t in trozos)
    for i in range(40):
        assert any(f"número {i} " in t for t in trozos)
    largo = rag.trocear("palabra " * 400, 300)            # sin puntuación: se parte por palabras
    assert all(len(t) <= 300 for t in largo)


def test_limpiar_une_palabras_cortadas():
    assert rag.limpiar("conta-\nminación   del\naire") == "contaminación del aire"


def test_pdf_conserva_la_pagina(corpus):
    paginas = rag.extraer_paginas(corpus / "norma.pdf")
    assert [n for n, _ in paginas] == [1, 2]
    assert "37 microgramos" in paginas[0][1] and "prevencion" in paginas[1][1]


def test_indice_y_busqueda(corpus, tmp_path):
    indice = tmp_path / "index"
    resumen = rag.construir_indice(corpus, indice, EmbedderFalso())
    assert {r["archivo"] for r in resumen} == {"norma.pdf", "recomendaciones.txt"}     # README.md y fuentes.csv fuera
    assert rag.indice_disponible(indice)

    buscador = rag.Buscador(indice, EmbedderFalso())
    r = buscador.buscar("nivel maximo permisible de PM2.5 para 24 horas", umbral=0.2)
    assert r and r[0]["archivo"] == "norma.pdf" and r[0]["pagina"] == 1
    assert r[0]["titulo"] == "Resolución de prueba"                                     # título tomado de fuentes.csv
    assert r == sorted(r, key=lambda x: -x["similitud"])

    r = buscador.buscar("nivel de prevencion concentracion entre 38 y 55", umbral=0.2)
    assert r[0]["pagina"] == 2
    r = buscador.buscar("ejercicio al aire libre para niños y adultos mayores", umbral=0.2)
    assert r[0]["archivo"] == "recomendaciones.txt"
    assert buscador.buscar("receta de bandeja paisa con chicharrón", umbral=0.2) == []  # fuera del corpus


def test_sin_indice_ni_documentos(tmp_path):
    with pytest.raises(FileNotFoundError):
        rag.Buscador(tmp_path / "no-existe")
    vacio = tmp_path / "docs"
    vacio.mkdir()
    with pytest.raises(RuntimeError):
        rag.construir_indice(vacio, tmp_path / "index", EmbedderFalso())
