import numpy as np
# ── Definicie modelov ────────────────────────────────────────────────────────
MODELS = {
    1: {
        "name": "First model",
        "transformations": [
            # [a,    b,     c,    d,    e,    f,    g,    h,    i,    j,    k,    l]
            [0.00,  0.00,  0.01, 0.00, 0.26, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.00],
            [0.20, -0.26, -0.01, 0.23, 0.22,-0.07, 0.07, 0.00, 0.24, 0.00, 0.80, 0.00],
            [-0.25, 0.28,  0.01, 0.26, 0.24,-0.07, 0.07, 0.00, 0.24, 0.00, 0.22, 0.00],
            [0.85,  0.04, -0.01,-0.04, 0.85, 0.09, 0.00, 0.08, 0.84, 0.00, 0.80, 0.00],
        ],
    },
    2: {
        "name": "Second model",
        "transformations": [
            [0.05,  0.00,  0.00, 0.00, 0.60, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.00],
            [0.45, -0.22,  0.22, 0.22, 0.45, 0.22,-0.22, 0.22,-0.45, 0.00, 1.00, 0.00],
            [-0.45, 0.22, -0.22, 0.22, 0.45, 0.22, 0.22,-0.22, 0.45, 0.00, 1.25, 0.00],
            [0.49, -0.08,  0.08, 0.08, 0.49, 0.08, 0.08,-0.08, 0.49, 0.00, 2.00, 0.00],
        ],
    },
}


def generate_ifs(model_id: int, n_iterations: int = 50000) -> np.ndarray:
    """
    Vygeneruje IFS fraktal iterativnym sposobom.

    Parametre
    ---------
    model_id     : cislo modelu (1 alebo 2)
    n_iterations : pocet iteracii (= pocet bodov)

    Navratova hodnota
    -----------------
    points : np.ndarray tvaru (n_iterations, 3) – x, y, z suradnice
    """
    transforms = MODELS[model_id]["transformations"]

    # Rozloz transformacie na matice a translacne vektory
    matrices = []
    translations = []
    for t in transforms:
        a, b, c, d, e, f, g, h, i, j, k, l = t
        matrices.append(np.array([[a, b, c],
                                   [d, e, f],
                                   [g, h, i]]))
        translations.append(np.array([j, k, l]))

    # Pociatocny bod
    x, y, z = 0.0, 0.0, 0.0

    # Uloz historiu bodov
    points = np.zeros((n_iterations, 3))

    # Nahodne indexy transformacii (p = 0.25 pre kazdu)
    indices = np.random.randint(0, 4, size=n_iterations)

    for step in range(n_iterations):
        idx = indices[step]
        point = np.array([x, y, z])
        new_point = matrices[idx] @ point + translations[idx]
        x, y, z = new_point
        points[step] = new_point

    return points
