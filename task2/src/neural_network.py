"""
neural_network.py – XOR neuronova siet: 2 vstupy → 2 skryte neurony → 1 vystup.

Architektura:
    Vstupna vrstva  : 2 neurony  (x, y)
    Skryta vrstva   : 2 neurony  (sigmoid)
    Vystupna vrstva : 1 neuron   (sigmoid)

Trenovanie: spetne sirenie chyby (backpropagation) + gradient descent, MSE strata.
"""

import numpy as np


# ── Aktivacna funkcia ──────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _sigmoid_d(x: np.ndarray) -> np.ndarray:
    s = _sigmoid(x)
    return s * (1.0 - s)


# ── Model ──────────────────────────────────────────────────────────────────────

class XORNet:
    """
    Jednoducha trojvrstvova siet na riesenie XOR problemu.

    Parametre
    ---------
    learning_rate : rychlost ucenia (default 0.1)
    n_epochs      : pocet treningovych epoch (default 10 000)
    seed          : seed pre reprodukovatelnost (default None = nahodne)
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_epochs: int = 10_000,
        seed: int | None = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.seed = seed

        # Zaznam vah a strat po kazdej zaznamenej epoche
        # Kazdy zaznam: {epoch, w_hidden, b_hidden, w_output, b_output, loss, predictions}
        self.history: list[dict] = []

        rng = np.random.default_rng(seed)

        # w_hidden: tvar (2, 2) – stlpce su skryte neurony
        #   w_hidden[:, 0] = vahy hidden1 od vstupov (x, y)
        #   w_hidden[:, 1] = vahy hidden2 od vstupov (x, y)
        self.w_hidden: np.ndarray = rng.random((2, 2))
        self.b_hidden: np.ndarray = rng.random(2)

        # w_output: tvar (2,) – vahy vystupneho neuronu od hidden1, hidden2
        self.w_output: np.ndarray = rng.random(2)
        self.b_output: float = float(rng.random())

    # ------------------------------------------------------------------
    # Vlastnosti pre priamy pristup k vahám (format zhodny s PDF)
    # ------------------------------------------------------------------

    @property
    def hidden1_weights(self) -> list[float]:
        return self.w_hidden[:, 0].tolist()

    @property
    def hidden2_weights(self) -> list[float]:
        return self.w_hidden[:, 1].tolist()

    @property
    def output_weights(self) -> list[float]:
        return self.w_output.tolist()

    @property
    def hidden1_bias(self) -> float:
        return float(self.b_hidden[0])

    @property
    def hidden2_bias(self) -> float:
        return float(self.b_hidden[1])

    # ------------------------------------------------------------------
    # Dopredny prechod
    # ------------------------------------------------------------------

    def _forward(self, X: np.ndarray) -> tuple:
        """
        Vypocita aktivacie pre vstup X (tvar N×2).

        Vracia (z_h, a_h, z_o, a_o):
            z_h : predaktivacia skrytej vrstvy  (N, 2)
            a_h : aktivacia skrytej vrstvy       (N, 2)
            z_o : predaktivacia vystupnej vrstvy (N,)
            a_o : aktivacia vystupnej vrstvy     (N,)
        """
        z_h = X @ self.w_hidden + self.b_hidden   # (N, 2)
        a_h = _sigmoid(z_h)                        # (N, 2)
        z_o = a_h @ self.w_output + self.b_output  # (N,)
        a_o = _sigmoid(z_o)                        # (N,)
        return z_h, a_h, z_o, a_o

    # ------------------------------------------------------------------
    # Predikcia
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vracia spojite vystupy siete (hodnoty 0–1) pre kazdy riadok X."""
        _, _, _, a_o = self._forward(X)
        return a_o

    def predict_binary(self, X: np.ndarray) -> np.ndarray:
        """Vracia binarny vysledok (0 alebo 1) pre kazdy riadok X."""
        return (self.predict(X) >= 0.5).astype(int)

    # ------------------------------------------------------------------
    # Trenovanie
    # ------------------------------------------------------------------

    def _record(self, X: np.ndarray, y: np.ndarray, epoch: int) -> None:
        _, _, _, a_o = self._forward(X)
        loss = float(np.mean((a_o - y) ** 2))
        self.history.append({
            "epoch":       epoch,
            "w_hidden":    self.w_hidden.copy(),
            "b_hidden":    self.b_hidden.copy(),
            "w_output":    self.w_output.copy(),
            "b_output":    float(self.b_output),
            "loss":        loss,
            "predictions": a_o.copy(),
        })

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        record_every: int = 50,
    ) -> "XORNet":
        """
        Trenuje siet na vstupoch X (tvar N×2) s navestiami y (tvar N,) v {0, 1}.

        Zaznamenava snimky kazdych *record_every* epoch do self.history.
        """
        self.history = []
        self._record(X, y, epoch=0)

        for epoch in range(1, self.n_epochs + 1):
            z_h, a_h, z_o, a_o = self._forward(X)

            # Delta vystupnej vrstvy: dL/dz_o
            delta_o = (a_o - y) * _sigmoid_d(z_o)              # (N,)

            # Delta skrytej vrstvy: dL/dz_h
            delta_h = np.outer(delta_o, self.w_output) * _sigmoid_d(z_h)  # (N, 2)

            # Aktualizacia vystupnej vrstvy
            self.w_output -= self.learning_rate * (a_h.T @ delta_o)
            self.b_output -= self.learning_rate * float(delta_o.sum())

            # Aktualizacia skrytej vrstvy
            self.w_hidden -= self.learning_rate * (X.T @ delta_h)
            self.b_hidden -= self.learning_rate * delta_h.sum(axis=0)

            if epoch % record_every == 0 or epoch == self.n_epochs:
                self._record(X, y, epoch=epoch)

        return self
