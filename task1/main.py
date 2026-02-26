"""
main.py – Vstupny bod pre Ulohu 1: Perceptron – Bod na priamke.

Spaja dohromady:
    1. Generovanie dat  (src/data_generator.py)
    2. Perceptron       (src/perceptron.py)
    3. Vizualizacia     (src/visualization.py)
"""

import numpy as np

from src.data_generator import generate_points, to_binary
from src.perceptron import Perceptron
from src.visualization import run_animation


SLOPE     = 3
INTERCEPT = 2


def main() -> None:
    # ── 1. Generovanie dat ────────────────────────────────────────────────────
    xs, ys, labels = generate_points(
        n=100, x_range=(-3.0, 3.0), seed=42,
        slope=SLOPE, intercept=INTERCEPT,
    )
    binary_labels = to_binary(labels)

    # Vypis rozdelenia
    n_above = int(np.sum(labels == 1))
    n_below = int(np.sum(labels == -1))
    n_on    = int(np.sum(labels == 0))
    print("=" * 45)
    print("  Task 1 – Perceptron: Point on the Line")
    print("=" * 45)
    print(f"  Points above  (+1) : {n_above}")
    print(f"  Points on line (0) : {n_on}")
    print(f"  Points below  (-1) : {n_below}")
    print(f"  Total              : {n_above + n_on + n_below}")
    print("-" * 45)

    # ── 2. Trenovanie perceptronu ─────────────────────────────────────────────
    X = np.column_stack([xs, ys])  # tvar (100, 2)

    model = Perceptron(learning_rate=0.1, n_epochs=50)
    model.fit(X, binary_labels)

    final_acc = model.history[-1]["accuracy"]
    print(f"  Learning rate : {model.learning_rate}")
    print(f"  Epochs        : {model.n_epochs}")
    print(f"  Final accuracy: {final_acc:.1%}")
    print("=" * 45)
    print()
    print("Launching interactive visualisation …")

    # ── 3. Spustenie vizualizacie ─────────────────────────────────────────────
    run_animation(
        xs=xs,
        ys=ys,
        true_labels=labels,
        binary_labels=binary_labels,
        perceptron=model,
        slope=SLOPE,
        intercept=INTERCEPT,
        interval_ms=400,
    )


if __name__ == "__main__":
    main()
