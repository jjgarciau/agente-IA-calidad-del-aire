"""Naive Bayes categórico implementado desde cero con numpy (técnica de la Parte 1).

Modelo:   P(y | x) ∝ P(y) · ∏_j P(x_j | y)
Entrenar: contar. P(y) es la frecuencia de cada clase y P(x_j = v | y = c) se estima con
          suavizado de Laplace:  (N_cjv + α) / (N_c + α · V_j)
Predecir: sumar log-probabilidades (evita el desbordamiento de multiplicar números pequeños)
          y normalizar con log-sum-exp.

También incluye las líneas base y las métricas, igualmente implementadas a mano.
scikit-learn solo se usa en las pruebas para verificar que los resultados coinciden.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class NaiveBayesCategorico:
    def __init__(self, n_categorias: list[int], n_clases: int, alpha: float = 1.0):
        self.n_categorias = list(n_categorias)   # V_j: número de valores posibles de cada variable
        self.n_clases = int(n_clases)
        self.alpha = float(alpha)
        self.log_prior: np.ndarray | None = None          # [K]
        self.log_verosimilitud: list[np.ndarray] = []     # por variable: [K, V_j]

    # ------------------------------------------------------------------ entrenamiento
    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveBayesCategorico":
        X = np.asarray(X, dtype=int)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2 or X.shape[1] != len(self.n_categorias):
            raise ValueError("X debe tener una columna por variable")
        conteo_clase = np.bincount(y, minlength=self.n_clases).astype(float)   # N_c
        with np.errstate(divide="ignore"):                                     # log(0) = -inf si una clase no aparece
            self.log_prior = np.log(conteo_clase) - np.log(conteo_clase.sum())

        self.log_verosimilitud = []
        for j, V in enumerate(self.n_categorias):
            conteo = np.zeros((self.n_clases, V))                              # N_cjv
            np.add.at(conteo, (y, X[:, j]), 1.0)
            suavizado = (conteo + self.alpha) / (conteo_clase[:, None] + self.alpha * V)
            self.log_verosimilitud.append(np.log(suavizado))
        return self

    # ------------------------------------------------------------------ predicción
    def _log_conjunta(self, X: np.ndarray) -> np.ndarray:
        """log P(y) + Σ_j log P(x_j | y), una fila por ejemplo y una columna por clase."""
        X = np.atleast_2d(np.asarray(X, dtype=int))
        total = np.tile(self.log_prior, (X.shape[0], 1))
        for j, tabla in enumerate(self.log_verosimilitud):
            total = total + tabla[:, X[:, j]].T
        return total

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        conjunta = self._log_conjunta(X)
        maximo = conjunta.max(axis=1, keepdims=True)
        log_evidencia = maximo + np.log(np.exp(conjunta - maximo).sum(axis=1, keepdims=True))
        return np.exp(conjunta - log_evidencia)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._log_conjunta(X).argmax(axis=1)

    # ------------------------------------------------------------------ persistencia
    def a_dict(self) -> dict:
        return {
            "n_categorias": self.n_categorias,
            "n_clases": self.n_clases,
            "alpha": self.alpha,
            "log_prior": [None if np.isinf(v) else float(v) for v in self.log_prior],
            "log_verosimilitud": [t.tolist() for t in self.log_verosimilitud],
        }

    @classmethod
    def de_dict(cls, d: dict) -> "NaiveBayesCategorico":
        m = cls(d["n_categorias"], d["n_clases"], d["alpha"])
        m.log_prior = np.array([-np.inf if v is None else v for v in d["log_prior"]], dtype=float)
        m.log_verosimilitud = [np.array(t, dtype=float) for t in d["log_verosimilitud"]]
        return m

    def guardar(self, ruta: Path, extra: dict | None = None) -> None:
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(json.dumps({**(extra or {}), "modelo": self.a_dict()}, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")

    @classmethod
    def cargar(cls, ruta: Path) -> tuple["NaiveBayesCategorico", dict]:
        d = json.loads(Path(ruta).read_text(encoding="utf-8"))
        return cls.de_dict(d["modelo"]), {k: v for k, v in d.items() if k != "modelo"}


# ---------------------------------------------------------------------- líneas base
def predecir_mayoritaria(y_entrenamiento: np.ndarray, n: int) -> np.ndarray:
    """Línea base 1: siempre la clase más frecuente del entrenamiento."""
    return np.full(n, np.bincount(np.asarray(y_entrenamiento, dtype=int)).argmax())


def predecir_persistencia(clase_hoy: np.ndarray) -> np.ndarray:
    """Línea base 2: mañana será igual que hoy."""
    return np.asarray(clase_hoy, dtype=int)


# ---------------------------------------------------------------------- métricas
def matriz_confusion(y, pred, n_clases: int) -> np.ndarray:
    """Filas = clase real, columnas = clase predicha."""
    m = np.zeros((n_clases, n_clases), dtype=int)
    np.add.at(m, (np.asarray(y, dtype=int), np.asarray(pred, dtype=int)), 1)
    return m


def exactitud(y, pred) -> float:
    return float((np.asarray(y) == np.asarray(pred)).mean())


def f1_macro(y, pred, n_clases: int) -> float:
    """Promedio simple del F1 de cada clase presente (en los reales o en las predicciones)."""
    m = matriz_confusion(y, pred, n_clases)
    f1s = []
    for c in range(n_clases):
        vp, fp, fn = m[c, c], m[:, c].sum() - m[c, c], m[c, :].sum() - m[c, c]
        if vp + fp + fn == 0:
            continue                      # la clase no aparece: no entra al promedio
        f1s.append(2 * vp / (2 * vp + fp + fn))
    return float(np.mean(f1s))


def precision_recall_binario(y, pred, desde_clase: int) -> tuple[float, float]:
    """Precisión y recall de "episodio": clase >= desde_clase frente al resto."""
    real = np.asarray(y) >= desde_clase
    alerta = np.asarray(pred) >= desde_clase
    vp = int((real & alerta).sum())
    precision = vp / alerta.sum() if alerta.sum() else 0.0
    recall = vp / real.sum() if real.sum() else 0.0
    return float(precision), float(recall)


def evaluar(y, pred, n_clases: int, clase_episodio: int) -> dict:
    precision, recall = precision_recall_binario(y, pred, clase_episodio)
    return {
        "exactitud": round(exactitud(y, pred), 4),
        "f1_macro": round(f1_macro(y, pred, n_clases), 4),
        "precision_episodio": round(precision, 4),
        "recall_episodio": round(recall, 4),
        "matriz_confusion": matriz_confusion(y, pred, n_clases).tolist(),
    }
