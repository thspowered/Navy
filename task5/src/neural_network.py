"""
Architektura: 4 -> 64 -> 32 -> 2 (ReLU skryte vrstvy, Softmax vystup)
Trenovanie:   supervised learning na (stav, akcia) paroch zo Q-learning agenta
Zavislosti:   iba numpy (bez pytorch / tensorflow)
"""

import numpy as np
import gymnasium as gym


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(float)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class NeuralNetwork:
    """
    Jednoducha 3-vrstvova MLP pre predikciu akcie v CartPole.

    Parametre
    ---------
    input_size   : pocet vstupnych priznakov (4 pre CartPole)
    hidden_sizes : tuple s pocetmi neuronov v skrytych vrstvach
    output_size  : pocet akcii (2 pre CartPole)
    lr           : rychlost ucenia
    """

    def __init__(
        self,
        input_size: int   = 4,
        hidden_sizes: tuple = (64, 32),
        output_size: int  = 2,
        lr: float         = 0.005,
    ) -> None:
        self.lr           = lr
        self.input_size   = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size  = output_size

        self.layers: list[dict] = []
        sizes = [input_size] + list(hidden_sizes) + [output_size]
        for i in range(len(sizes) - 1):
            W = np.random.randn(sizes[i], sizes[i + 1]) * np.sqrt(2.0 / sizes[i])
            b = np.zeros(sizes[i + 1])
            self.layers.append({"W": W, "b": b})

        self.trained:        bool  = False
        self.train_accuracy: float = 0.0
        self.n_epochs_trained: int = 0
        self._cache: dict          = {}

    # ── Dopredny prechod ────────────────────────────────────────────────────────

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Vypocita pravdepodobnosti akcii pre davku stavov X (N x 4)."""
        self._cache = {"a0": X}
        a = X
        for i, layer in enumerate(self.layers[:-1]):
            z = a @ layer["W"] + layer["b"]
            a = _relu(z)
            self._cache[f"z{i + 1}"] = z
            self._cache[f"a{i + 1}"] = a
        # Vystupna vrstva – softmax
        n = len(self.layers)
        z_out = a @ self.layers[-1]["W"] + self.layers[-1]["b"]
        probs = _softmax(z_out)
        self._cache[f"z{n}"] = z_out
        self._cache[f"a{n}"] = probs
        return probs

    # ── Predikcia ───────────────────────────────────────────────────────────────

    def predict(self, obs: np.ndarray) -> int:
        """Vrati akciu pre jeden stav (1D alebo 2D pole)."""
        X = obs[np.newaxis, :] if obs.ndim == 1 else obs
        probs = self.forward(X)
        return int(np.argmax(probs[0]))

    # ── Strata ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
        n = len(y)
        return float(-np.mean(np.log(probs[np.arange(n), y] + 1e-8)))

    # ── Spatny prechod ──────────────────────────────────────────────────────────

    def _backward(self, y: np.ndarray) -> None:
        n       = len(y)
        n_lay   = len(self.layers)
        probs   = self._cache[f"a{n_lay}"]

        d = probs.copy()
        d[np.arange(n), y] -= 1.0
        d /= n

        grads: list[dict] = []
        for i in range(n_lay - 1, -1, -1):
            a_prev = self._cache[f"a{i}"]
            dW     = a_prev.T @ d
            db     = d.sum(axis=0)
            grads.insert(0, {"dW": dW, "db": db})
            if i > 0:
                d  = d @ self.layers[i]["W"].T
                d *= _relu_grad(self._cache[f"z{i}"])

        for i, g in enumerate(grads):
            self.layers[i]["W"] -= self.lr * g["dW"]
            self.layers[i]["b"] -= self.lr * g["db"]

    # ── Trenovanie ──────────────────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int     = 300,
        batch_size: int = 64,
    ) -> list[float]:
        """
        Trenovanie siete mini-batch gradientnym zostupom.

        Parametre
        ---------
        X          : (N, 4) – spojite stavy zo Q-learning agenta
        y          : (N,)   – akcie zo Q-learning agenta (0 alebo 1)
        epochs     : pocet epoch
        batch_size : velkost mini-davky

        Navratova hodnota
        -----------------
        list[float] – strata po kazdom epoche
        """
        # Normalizacia vstupu
        self._X_mean = X.mean(axis=0)
        self._X_std  = X.std(axis=0) + 1e-8
        Xn           = (X - self._X_mean) / self._X_std

        losses: list[float] = []
        n = len(Xn)

        for _ in range(epochs):
            idx      = np.random.permutation(n)
            Xs, ys   = Xn[idx], y[idx]
            ep_loss  = 0.0
            for start in range(0, n, batch_size):
                Xb    = Xs[start : start + batch_size]
                yb    = ys[start : start + batch_size]
                probs = self.forward(Xb)
                ep_loss += self._cross_entropy(probs, yb) * len(Xb)
                self._backward(yb)
            losses.append(ep_loss / n)

        # Presnost na trenovacich datach
        probs              = self.forward(Xn)
        preds              = np.argmax(probs, axis=1)
        self.train_accuracy  = float(np.mean(preds == y))
        self.trained         = True
        self.n_epochs_trained += epochs
        return losses

    # ── Spustenie epizody (trajektoria pre vizualizaciu) ───────────────────────

    def run_episode(self, max_steps: int = 500) -> list[np.ndarray]:
        """Spusti epizodu riadenou neuronovou sietou a vrati trajektoriu stavov."""
        env        = gym.make("CartPole-v1")
        obs, _     = env.reset()
        trajectory = [obs.copy()]
        done       = False
        steps      = 0

        while not done and steps < max_steps:
            obs_norm           = (obs - self._X_mean) / self._X_std
            action             = self.predict(obs_norm)
            obs, _, term, trunc, _ = env.step(action)
            done               = term or trunc
            trajectory.append(obs.copy())
            steps             += 1

        env.close()
        return trajectory

    # ── Reset ───────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reinicializuje vahy siete."""
        self.__init__(
            input_size=self.input_size,
            hidden_sizes=self.hidden_sizes,
            output_size=self.output_size,
            lr=self.lr,
        )
