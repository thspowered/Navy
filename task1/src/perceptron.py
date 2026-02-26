import numpy as np


class Perceptron:
    """Klasicky Rosenblatov perceptron pre binarne klasifikovanie (+1 / -1)."""

    def __init__(self, learning_rate: float = 0.1, n_epochs: int = 50):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights: np.ndarray | None = None  # tvar (n_features,)
        self.bias: float = 0.0
        # Kazdy zaznam: {"weights": w_kopia, "bias": b, "accuracy": presnost}
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    # pomocne metody
    # ------------------------------------------------------------------

    def _activation(self, net: np.ndarray) -> np.ndarray:
        """Skokova funkcia: vracia +1 pre net >= 0, inak -1."""
        return np.where(net >= 0, 1, -1)

    # ------------------------------------------------------------------
    # Verejne rozhranie
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        """
        Trenovanie na X (tvar N×2) s navestiami y v {-1, +1}.
        Uklada snimku vah po kazdej epoche do self.history.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.history = []

        # Snimka epochy 0 (pred zacatim trenovania)
        acc0 = self.accuracy(X, y)
        self.history.append(
            {
                "weights": self.weights.copy(),
                "bias": self.bias,
                "accuracy": acc0,
            }
        )

        for _ in range(self.n_epochs):
            for xi, yi in zip(X, y):
                prediction = self._activation(np.dot(xi, self.weights) + self.bias)
                if prediction != yi:
                    self.weights += self.learning_rate * yi * xi
                    self.bias += self.learning_rate * yi

            acc = self.accuracy(X, y)
            self.history.append(
                {
                    "weights": self.weights.copy(),
                    "bias": self.bias,
                    "accuracy": acc,
                }
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vracia predikcie +1/-1 pre kazdy riadok v X."""
        if self.weights is None:
            raise RuntimeError("Call fit() before predict().")
        net = X @ self.weights + self.bias
        return self._activation(net)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Podiel spravne klasifikovanych vzoriek."""
        if self.weights is None:
            return 0.0
        return float(np.mean(self.predict(X) == y))

    def boundary_y(
        self, x_vals: np.ndarray, epoch: int = -1
    ) -> np.ndarray | None:
        """
        Vypocita y-suradnice rozhodovacej hranice pre x_vals.

        Rozhodovacia hranica: w0*x + w1*y + b = 0
          =>  y = -(w0*x + b) / w1

        Vracia None ak w1 priblizne 0 (vertikalna / degenerovana hranica).
        """
        snap = self.history[epoch]
        w = snap["weights"]
        b = snap["bias"]

        if abs(w[1]) < 1e-8:
            return None

        return -(w[0] * x_vals + b) / w[1]
