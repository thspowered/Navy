import numpy as np

# ── Konstanty ─────────────────────────────────────────────────────────────────
SLOPE = 3
INTERCEPT = 2
ON_TOLERANCE = 0.15  # |y - line_y(x)| <= tolerancia  =>  "na priamke"



def line_y(x: np.ndarray | float,
           slope: float = SLOPE,
           intercept: float = INTERCEPT) -> np.ndarray | float:
    """Vracia y = slope * x + intercept."""
    return slope * x + intercept


def classify_point(x: float, y: float,
                   slope: float = SLOPE,
                   intercept: float = INTERCEPT) -> int:
    """
    Klasifikuje bod vzhladom k y = slope*x + intercept.

    Vracia:
        +1  – nad priamkou
        -1  – pod priamkou
         0  – na priamke (v rozsahu ON_TOLERANCE)
    """
    diff = y - line_y(x, slope, intercept)
    if abs(diff) <= ON_TOLERANCE:
        return 0
    return 1 if diff > 0 else -1


def generate_points(
    n: int = 100,
    x_range: tuple[float, float] = (-3.0, 3.0),
    seed: int | None = 42,
    slope: float = SLOPE,
    intercept: float = INTERCEPT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generuje *n* navestiami oznacenych bodov okolo y = slope*x + intercept.

    Zlozenie:
        * 90 nahodnych bodov rozlozenych v rozsahu x, y vzorkovane okolo priamky
        * 10 explicitnych bodov na priamke (y_on priblizne line_y(x_on) + maly sum)

    Vracia:
        xs     – tvar (n,)
        ys     – tvar (n,)
        labels – tvar (n,)  hodnoty v {-1, 0, +1}
    """
    rng = np.random.default_rng(seed)
    n_random = n - 10
    n_online = 10

    # --- Nahodne body ---------------------------------------------------
    xs_rand = rng.uniform(x_range[0], x_range[1], size=n_random)
    # y-hodnoty rozlozene +-4 jednotky okolo priamky, aby sme ziskali obe triedy
    spread = 4.0
    ys_rand = line_y(xs_rand, slope, intercept) + rng.uniform(-spread, spread, size=n_random)

    # --- Body na priamke --------------------------------------------------
    xs_on = rng.uniform(x_range[0], x_range[1], size=n_online)
    # maly sum, aby neboli dokonale kolinearni
    ys_on = line_y(xs_on, slope, intercept) + rng.uniform(
        -ON_TOLERANCE * 0.5, ON_TOLERANCE * 0.5, size=n_online
    )

    # --- Zluc a zamiesaj -----------------------------------------------
    xs = np.concatenate([xs_rand, xs_on])
    ys = np.concatenate([ys_rand, ys_on])

    labels = np.array(
        [classify_point(x, y, slope, intercept) for x, y in zip(xs, ys)],
        dtype=int,
    )

    idx = rng.permutation(len(xs))
    return xs[idx], ys[idx], labels[idx]


def to_binary(labels: np.ndarray) -> np.ndarray:
    """
    Mapuje trojtriedne navestia → binarne navestia pre trenovanie perceptronu.

    Mapovanie:  -1 → -1,   0 → +1,   +1 → +1
    (body na priamke sa pocitaju ako trieda "nad / pozitivna")
    """
    return np.where(labels >= 0, 1, -1)
