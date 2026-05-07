from __future__ import annotations

import numpy as np
from scipy.integrate import odeint

# ── Fyzikalne konstanty / defaulty ─────────────────────────────────────────────
G_DEFAULT = 9.81  # tiazove zrychlenie (m/s^2)
L1_DEFAULT = 1.0
L2_DEFAULT = 1.0
M1_DEFAULT = 1.0
M2_DEFAULT = 1.0


def get_derivative(
    state: np.ndarray,
    t: float,
    l1: float,
    l2: float,
    m1: float,
    m2: float,
    g: float = G_DEFAULT,
) -> tuple[float, float, float, float]:
    """
    Vrati derivaciu stavoveho vektora pre odeint.

    Vstup : state = (theta1, omega1, theta2, omega2), cas t (nepouzity – system
            je autonomny, ale odeint signatura ho vyzaduje)
    Vystup: (omega1, alpha1, omega2, alpha2), kde alpha_i = theta_i''

    Format presne ako PDF (strana 7): "function get_derivative will return tuple:
        theta1', theta1'', theta2', theta2''".
    """
    theta1, omega1, theta2, omega2 = state
    d = theta1 - theta2
    sd = np.sin(d)
    cd = np.cos(d)
    s1 = np.sin(theta1)
    s2 = np.sin(theta2)

    denom = m1 + m2 * sd * sd

    alpha1 = (
        m2 * g * s2 * cd
        - m2 * sd * (l1 * omega1 * omega1 * cd + l2 * omega2 * omega2)
        - (m1 + m2) * g * s1
    ) / (l1 * denom)

    alpha2 = (
        (m1 + m2) * (l1 * omega1 * omega1 * sd - g * s2 + g * s1 * cd)
        + m2 * l2 * omega2 * omega2 * sd * cd
    ) / (l2 * denom)

    return omega1, alpha1, omega2, alpha2


def integrate(
    state0: np.ndarray,
    duration: float = 20.0,
    dt: float = 0.02,
    l1: float = L1_DEFAULT,
    l2: float = L2_DEFAULT,
    m1: float = M1_DEFAULT,
    m2: float = M2_DEFAULT,
    g: float = G_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integruje rovnice dvojiteho kyvadla od t=0 po t=duration pomocou odeint.

    PDF (strana 7):
        od = odeint(get_derivative, state_0, t, args=(l1, l2, m1, m2))
        theta1 = od[:, 0],  theta2 = od[:, 2]

    Navratova hodnota
    -----------------
    t_arr     : pole casov tvaru (n_steps,)
    state_arr : pole stavov tvaru (n_steps, 4) – stlpce (theta1, omega1, theta2, omega2)
    """
    n_steps = int(round(duration / dt)) + 1
    t_arr = np.linspace(0.0, (n_steps - 1) * dt, n_steps)
    state_arr = odeint(
        get_derivative,
        np.asarray(state0, dtype=float),
        t_arr,
        args=(l1, l2, m1, m2, g),
    )
    return t_arr, state_arr


def positions(
    state_arr: np.ndarray,
    l1: float = L1_DEFAULT,
    l2: float = L2_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Premeni pole stavov na karteziánske pozicie zavazi.

    Vystup: (x1, y1, x2, y2), kazde tvaru (n_steps,).
    """
    theta1 = state_arr[:, 0]
    theta2 = state_arr[:, 2]
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)
    return x1, y1, x2, y2


def total_energy(
    state_arr: np.ndarray,
    l1: float = L1_DEFAULT,
    l2: float = L2_DEFAULT,
    m1: float = M1_DEFAULT,
    m2: float = M2_DEFAULT,
    g: float = G_DEFAULT,
) -> np.ndarray:
    """
    Vrati celkovu energiu E = T + V pre kazdy stav v poli.

    Pri dobre integracii by mala byt prakticky konstantna – sluzi ako sanity check.
    """
    th1 = state_arr[:, 0]
    w1 = state_arr[:, 1]
    th2 = state_arr[:, 2]
    w2 = state_arr[:, 3]

    # Kineticka energia (z PDF)
    T = 0.5 * m1 * (l1 * w1) ** 2 + 0.5 * m2 * (
        (l1 * w1) ** 2 + (l2 * w2) ** 2 + 2 * l1 * l2 * w1 * w2 * np.cos(th1 - th2)
    )
    # Potencialna (referenca: zaves)
    V = -(m1 + m2) * g * l1 * np.cos(th1) - m2 * g * l2 * np.cos(th2)
    return T + V
