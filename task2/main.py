"""
main.py – Vstupny bod pre Ulohu 2: Jednoducha neuronova siet – XOR problem.

Spaja dohromady:
    1. Definicia XOR dat
    2. XOR neuronova siet   (src/neural_network.py)
    3. Vizualizacia         (src/visualization.py)
"""

import numpy as np

from src.neural_network import XORNet
from src.visualization import run_animation


# ── XOR data ──────────────────────────────────────────────────────────────────
XOR_X: np.ndarray = np.array(
    [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]],
    dtype=float,
)
XOR_Y: np.ndarray = np.array([0, 1, 1, 0], dtype=float)


# ── Pomocna funkcia pre tabulkovy vypis vah ────────────────────────────────────

def _print_weights(label: str, model: XORNet) -> None:
    sep = "-" * 62
    print(sep)
    print(f"Weight and bias {label} the learning phase:")
    print(f"  neuron_hidden1.weights   {model.hidden1_weights}")
    print(f"  neuron_hidden2.weights   {model.hidden2_weights}")
    print(f"  neuron_output.weights    {model.output_weights}")
    print(f"  neuron_hidden1.bias      {model.hidden1_bias}")
    print(f"  neuron_hidden2.bias      {model.hidden2_bias}")
    print(f"  neuron_output.bias       {model.b_output}")
    print(sep)


# ── Hlavna funkcia ─────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 62)
    print("  Task 2 – Simple Neural Network: XOR Problem")
    print("=" * 62)
    print()

    # ── 1. Inicializacia modelu ────────────────────────────────────────────
    model = XORNet(learning_rate=0.1, n_epochs=10_000)

    # ── 2. Vahy PRED trenovenim ────────────────────────────────────────────
    _print_weights("before", model)
    print()

    # ── 3. Trenovanie ─────────────────────────────────────────────────────
    print("Learning in progress..")
    model.fit(XOR_X, XOR_Y, record_every=50)
    print()

    # ── 4. Vahy PO trenovani ───────────────────────────────────────────────
    _print_weights("after", model)
    print()

    # ── 5. Testovanie ─────────────────────────────────────────────────────
    print("Testing in progress..")
    print()

    preds  = model.predict(XOR_X)
    binary = model.predict_binary(XOR_X)

    col_w = 26
    print(f"{'Guess':<{col_w}} {'Expected output':<20} {'Is it equal?'}")
    correct = 0
    for pred, b_pred, expected in zip(preds, binary, XOR_Y):
        is_equal = (b_pred == int(expected))
        correct += int(is_equal)
        print(f"{pred:<{col_w}} {int(expected):<20} {is_equal}")

    success = correct / len(XOR_Y) * 100
    print(f"success is {success:.1f} %")
    print("-" * 62)
    print()
    print("Launching interactive visualisation …")

    # ── 6. Spustenie vizualizacie ──────────────────────────────────────────
    run_animation(model, XOR_X, XOR_Y, interval_ms=120)


if __name__ == "__main__":
    main()
