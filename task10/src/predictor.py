"""
predictor.py - Jednoducha plne prepojena neuronova siet (MLP) cisto v NumPy.

Architektura:
    vstup (2)  ->  Dense(32) tanh  ->  Dense(16) tanh  ->  Dense(1) sigmoid
    vstup  = (a_normalized, x)   kde  a_normalized = a / 4.0
    vystup = predikovane x_{n+1}  v rozsahu [0, 1]

Tréning: mini-batch SGD s Adam optimizerom, MSE loss.

Pouzitie:
    >>> mlp = MLP([2, 32, 16, 1])
    >>> mlp.train(X, y, epochs=80, lr=0.01)
    >>> y_pred = mlp.predict(X_test)
"""

from __future__ import annotations
import numpy as np


# ── Konstanta pre normalizaciu vstupu a do [0,1] ───────────────────────────────
A_MAX = 4.0


def _xavier_init(n_in: int, n_out: int, rng: np.random.Generator) -> np.ndarray:
    """Xavier/Glorot inicializacia – stabilna pre tanh aktivacie."""
    std = np.sqrt(2.0 / (n_in + n_out))
    return rng.standard_normal((n_in, n_out)) * std


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


class MLP:
    """
    Multi-layer perceptron implementovany od nuly v NumPy.

    Skryta vrstva: tanh (hladka aktivacia, dobra pre regresiu).
    Vystupna vrstva: sigmoid (vystup v [0,1] – zhoduje sa s rozsahom logistickej mapy).
    Optimizer: Adam (beta1=0.9, beta2=0.999) – rychla a stabilna konvergencia.
    """

    def __init__(self, layers: list[int], seed: int | None = 0) -> None:
        rng = np.random.default_rng(seed)
        self.layers = layers
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for n_in, n_out in zip(layers[:-1], layers[1:]):
            self.W.append(_xavier_init(n_in, n_out, rng))
            self.b.append(np.zeros(n_out))

        # Adam stavy
        self._m_W = [np.zeros_like(w) for w in self.W]
        self._v_W = [np.zeros_like(w) for w in self.W]
        self._m_b = [np.zeros_like(b) for b in self.b]
        self._v_b = [np.zeros_like(b) for b in self.b]
        self._t = 0
        self._rng = rng

    # ── Forward pass ───────────────────────────────────────────────────────────
    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Vrati (vystup, zoznam aktivacii vsetkych vrstiev vratane vstupu)."""
        activations = [X]
        a = X
        last = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            if i == last:
                a = _sigmoid(z)            # vystupna vrstva: sigmoid
            else:
                a = np.tanh(z)             # skryte vrstvy: tanh
            activations.append(a)
        return a, activations

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predikuje x_{n+1} pre kazdy riadok X = [a_norm, x]. Vrati 1D pole."""
        out, _ = self._forward(X)
        return out.ravel()

    # ── Backprop a Adam update ─────────────────────────────────────────────────
    def _backward(self, activations: list[np.ndarray], y: np.ndarray) -> tuple[list, list]:
        """Vypocita gradienty MSE lossu pre vsetky parametre."""
        N = activations[0].shape[0]
        last = len(self.W) - 1

        # MSE: L = mean((a_out - y)^2). Pri sigmoide:
        #   d a_out / d z = a_out * (1 - a_out)
        a_out = activations[-1]
        delta = (a_out - y.reshape(-1, 1)) * (2.0 / N) * a_out * (1.0 - a_out)

        grads_W: list[np.ndarray] = [None] * len(self.W)  # type: ignore[list-item]
        grads_b: list[np.ndarray] = [None] * len(self.b)  # type: ignore[list-item]

        for i in range(last, -1, -1):
            grads_W[i] = activations[i].T @ delta
            grads_b[i] = delta.sum(axis=0)
            if i > 0:
                # Backprop cez tanh predoslej vrstvy
                a_prev = activations[i]    # toto je tanh(z) z vrstvy i-1 (vstupom do W[i])
                delta = (delta @ self.W[i].T) * (1.0 - a_prev ** 2)

        return grads_W, grads_b

    def _adam_update(
        self,
        grads_W: list[np.ndarray],
        grads_b: list[np.ndarray],
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self._t += 1
        bc1 = 1.0 - beta1 ** self._t
        bc2 = 1.0 - beta2 ** self._t
        for i in range(len(self.W)):
            # W
            self._m_W[i] = beta1 * self._m_W[i] + (1 - beta1) * grads_W[i]
            self._v_W[i] = beta2 * self._v_W[i] + (1 - beta2) * (grads_W[i] ** 2)
            m_hat = self._m_W[i] / bc1
            v_hat = self._v_W[i] / bc2
            self.W[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
            # b
            self._m_b[i] = beta1 * self._m_b[i] + (1 - beta1) * grads_b[i]
            self._v_b[i] = beta2 * self._v_b[i] + (1 - beta2) * (grads_b[i] ** 2)
            m_hat_b = self._m_b[i] / bc1
            v_hat_b = self._v_b[i] / bc2
            self.b[i] -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

    # ── Tréning ────────────────────────────────────────────────────────────────
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 80,
        batch_size: int = 256,
        lr: float = 0.01,
        verbose: bool = False,
        on_epoch=None,
    ) -> list[float]:
        """
        Trenuje siet mini-batch Adam. Vracia historiu MSE per epoch.

        on_epoch(epoch_idx, mse) – volitelny callback (pre status / animaciu).
        """
        history: list[float] = []
        N = X.shape[0]
        for ep in range(epochs):
            idx = self._rng.permutation(N)
            X_sh = X[idx]
            y_sh = y[idx]
            losses = []
            for start in range(0, N, batch_size):
                end = start + batch_size
                X_b = X_sh[start:end]
                y_b = y_sh[start:end]
                out, acts = self._forward(X_b)
                gW, gb = self._backward(acts, y_b)
                self._adam_update(gW, gb, lr)
                losses.append(np.mean((out.ravel() - y_b) ** 2))
            mse = float(np.mean(losses))
            history.append(mse)
            if verbose:
                print(f"  Epoch {ep + 1:3d}/{epochs}  MSE={mse:.6f}")
            if on_epoch is not None:
                on_epoch(ep, mse)
        return history


# ── Pomocne funkcie pre normalizaciu a predikciu bifurkacie ────────────────────

def normalize_inputs(a: np.ndarray | float, x: np.ndarray | float) -> np.ndarray:
    """Vyrobi vstup pre MLP: stack([a/A_MAX, x]) tvaru (N, 2)."""
    a_arr = np.atleast_1d(np.asarray(a, dtype=float))
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    if a_arr.shape != x_arr.shape:
        a_arr = np.broadcast_to(a_arr, x_arr.shape)
    return np.column_stack([a_arr.ravel() / A_MAX, x_arr.ravel()])


def predict_next(mlp: MLP, a: np.ndarray | float, x: np.ndarray | float) -> np.ndarray:
    """Predikuje x_{n+1} z (a, x_n) pomocou natrenovanej siete."""
    inp = normalize_inputs(a, x)
    return mlp.predict(inp)


def predict_bifurcation_onestep(
    mlp: MLP,
    a_values: np.ndarray,
    n_iter: int = 400,
    n_transient: int = 100,
    x0: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bifurkacna predikcia "point by point" (Example 3 z PDF).

    Postup:
      1. Pre kazde a spustime SKUTOCNU logisticku mapu po n_iter krokov,
         pricom prvych n_transient zahodime (tranzient).
      2. Pre kazdy bod atraktora (a, x_n) si NN predikuje x_{n+1} = NN(a, x_n).
      3. Vratime splostene pole tychto predikcii (a, x_pred).

    V stabilnych oblastiach NN spravne predpovie, ze x_{n+1} je opat
    pevny bod. V chaotickych x_n je v ramci atraktora a NN trafi nieco
    podobne, ale nie presne -> rozptyl cervenych bodov vidno v Example 3.

    Toto sa lisi od cisto rekurzivneho iterovania NN (ktore by akumulovalo
    male chyby naucenia a zbiehalo k pevnemu bodu NN-funkcie).
    """
    from .logistic import compute_bifurcation
    a_arr = np.asarray(a_values, dtype=float)
    a_real, x_real = compute_bifurcation(
        a_arr, n_iter=n_iter, n_transient=n_transient, x0=x0
    )
    inp = np.column_stack([a_real / A_MAX, x_real])
    x_pred = mlp.predict(inp)
    return a_real, x_pred


def predict_bifurcation_iterative(
    mlp: MLP,
    a_values: np.ndarray,
    n_iter: int = 400,
    n_transient: int = 100,
    x0: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bifurkacna predikcia ciste z NN dynamiky (autoregresivne iterovanie).

    Pre kazde a startuje x_0 a opakovane aplikuje x_{n+1} = NN(a, x_n).
    NN tu nahradzuje skutocnu logisticku funkciu vo vsetkych krokoch.

    POZN.: Pri perfektnom MSE na trénovacích datach NN aj tak akumuluje
    drobne chyby, takze v chaotickych oblastiach moze trajektoria odbiehat
    od skutocnej. V stabilnych oblastiach kopiruje atraktor presne.
    """
    a_arr = np.asarray(a_values, dtype=float)
    x = np.full_like(a_arr, x0, dtype=float)
    a_norm = a_arr / A_MAX

    for _ in range(n_transient):
        inp = np.column_stack([a_norm, x])
        x = mlp.predict(inp)

    n_keep = n_iter - n_transient
    x_history = np.empty((n_keep, a_arr.size), dtype=float)
    for i in range(n_keep):
        inp = np.column_stack([a_norm, x])
        x = mlp.predict(inp)
        x_history[i] = x

    a_out = np.broadcast_to(a_arr, x_history.shape).ravel()
    x_out = x_history.ravel()
    return a_out, x_out
