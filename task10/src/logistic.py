"""
logistic.py - Logisticka mapa a generator bifurkacneho diagramu.

Logisticka mapa:
    x_{n+1} = a * x_n * (1 - x_n)

Pre rozne hodnoty parametra a (0..4) sa mapa sprava roznym sposobom:
    a < 1.0       -> x konverguje k 0
    1 <= a < 3    -> x konverguje k jedinemu fixnemu bodu
    3 <= a < ~3.45 -> 2-cyklus (period doubling)
    ~3.45..3.57   -> dalsie zdvojnasobovania periody
    a > 3.57      -> chaos (s "oknami" stability)

Bifurkacny diagram: pre kazde a vykreslime mnozinu hodnot, ku ktorym sa
mapa po dlhom case "usadi" (atraktor). Tranzient na zaciatku zahodime.
"""

from __future__ import annotations
import numpy as np


def logistic_step(x: np.ndarray | float, a: np.ndarray | float) -> np.ndarray | float:
    """Jeden krok logistickej mapy: x_{n+1} = a * x * (1 - x)."""
    return a * x * (1.0 - x)


def compute_orbit(
    a: float,
    x0: float = 0.5,
    n_iter: int = 1000,
    n_transient: int = 200,
) -> np.ndarray:
    """
    Vypocita orbitu logistickej mapy pre konkretne a.

    Spusti mapu na n_iter krokov, pricom prvych n_transient zahodi
    (necha system "usadit sa" na atraktor).

    Navratova hodnota: pole dlzky n_iter - n_transient (atraktor body).
    """
    x = x0
    # Tranzient – nezachovavame
    for _ in range(n_transient):
        x = a * x * (1.0 - x)
    # Atraktor – ulozime
    out = np.empty(n_iter - n_transient, dtype=float)
    for i in range(n_iter - n_transient):
        x = a * x * (1.0 - x)
        out[i] = x
    return out


def compute_bifurcation(
    a_values: np.ndarray,
    n_iter: int = 1000,
    n_transient: int = 200,
    x0: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vygeneruje data pre bifurkacny diagram (vektorizovane cez NumPy).

    Pre kazdu hodnotu a_values spusti logisticku mapu a po tranzientne
    n_transient krokoch zacne zaznamenavat dalsich (n_iter - n_transient)
    hodnot. Vsetky body splosti do dvoch poli (a_arr, x_arr) vhodnych
    pre matplotlib scatter.

    Parametre
    ---------
    a_values     : pole hodnot parametra a (napr. np.linspace(0, 4, 1000))
    n_iter       : celkovy pocet iteracii pre kazde a
    n_transient  : pocet uvodnych iteracii ktore zahodime (tranzient)
    x0           : pociatocna hodnota x_0

    Navratova hodnota
    -----------------
    a_arr : np.ndarray dlzky N*n_keep – x-suradnice (parameter a)
    x_arr : np.ndarray rovnakej dlzky – y-suradnice (atraktor)
    """
    a = np.asarray(a_values, dtype=float)
    n_keep = n_iter - n_transient
    if n_keep <= 0:
        raise ValueError("n_iter musi byt vacsie ako n_transient")

    x = np.full_like(a, x0, dtype=float)
    # Tranzient (vektorizovane cez vsetky a sucasne)
    for _ in range(n_transient):
        x = a * x * (1.0 - x)

    # Atraktor – zber bodov
    x_arr = np.empty((n_keep, a.size), dtype=float)
    for i in range(n_keep):
        x = a * x * (1.0 - x)
        x_arr[i] = x

    a_arr = np.broadcast_to(a, x_arr.shape).ravel()
    x_arr = x_arr.ravel()
    return a_arr, x_arr


def make_training_pairs(
    n_samples: int = 50_000,
    a_min: float = 0.0,
    a_max: float = 4.0,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vyrobi nahodne (a_norm, x) -> x' tréningové pary pre neuronovu siet.

    Vstup pre NN: stlpce [a / 4.0, x] z rovnomerneho rozdelenia na
    [a_min, a_max] x [0, 1]. Normalizacia a -> [0, 1] zlepsuje konvergenciu.
    Cielova hodnota: x' = a * x * (1 - x).

    Navratova hodnota
    -----------------
    X : np.ndarray tvaru (n_samples, 2) – stlpce [a_norm, x]
    y : np.ndarray tvaru (n_samples,)   – x' = logistic_step(x, a)
    """
    rng = np.random.default_rng(seed)
    a = rng.uniform(a_min, a_max, n_samples)
    x = rng.uniform(0.0, 1.0, n_samples)
    y = a * x * (1.0 - x)
    # Normalizacia a do [0, 1] – rovnaky predspracovane vstup ako v predict_*.
    X = np.column_stack([a / 4.0, x])
    return X, y
