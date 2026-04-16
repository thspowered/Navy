import numpy as np


def compute_mandelbrot(
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    width: int, height: int,
    max_iter: int = 256,
) -> np.ndarray:
    """
    Vypocita Mandelbrotovu mnozinu pre danu oblast.

    Navratova hodnota
    -----------------
    iterations : np.ndarray tvaru (height, width) – pocet iteracii pre kazdy pixel
    """
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    C = x[np.newaxis, :] + 1j * y[:, np.newaxis]

    Z = np.zeros_like(C, dtype=complex)
    iterations = np.zeros(C.shape, dtype=int)
    mask = np.ones(C.shape, dtype=bool)

    for i in range(max_iter):
        Z[mask] = Z[mask] ** 2 + C[mask]
        escaped = mask & (np.abs(Z) > 2)
        iterations[escaped] = i + 1
        mask &= ~escaped

    return iterations


def compute_julia(
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    width: int, height: int,
    c_real: float = -0.7,
    c_imag: float = 0.27015,
    max_iter: int = 256,
) -> np.ndarray:
    """
    Vypocita Juliovu mnozinu pre danu oblast a konstantu c.

    Navratova hodnota
    -----------------
    iterations : np.ndarray tvaru (height, width) – pocet iteracii pre kazdy pixel
    """
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    Z = x[np.newaxis, :] + 1j * y[:, np.newaxis]

    c = complex(c_real, c_imag)
    iterations = np.zeros(Z.shape, dtype=int)
    mask = np.ones(Z.shape, dtype=bool)

    for i in range(max_iter):
        Z[mask] = Z[mask] ** 2 + c
        escaped = mask & (np.abs(Z) > 2)
        iterations[escaped] = i + 1
        mask &= ~escaped

    return iterations


def iterations_to_rgb(iterations: np.ndarray, max_iter: int = 256) -> np.ndarray:
    """
    Premeni maticu iteracii na RGB obrazok pomocou HSV farebneho modelu.

    Body patriace do mnoziny (iterations == 0) su cierne.
    Ostatne su zafarbene podla Hue (saturacia + jas).
    """
    h, w = iterations.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Bod v mnozine => cierny
    in_set = iterations == 0

    # Normalizacia iteracii do [0, 1]
    norm = iterations.astype(float) / max_iter

    # HSV: Hue = normalizovana iteracia, Saturation = 1, Value = 1 (ak nie je v mnozine)
    from matplotlib.colors import hsv_to_rgb as _hsv_to_rgb

    hsv = np.zeros((h, w, 3))
    hsv[..., 0] = norm          # Hue
    hsv[..., 1] = 1.0           # Saturation
    hsv[..., 2] = ~in_set       # Value (0 pre body v mnozine, 1 inak)

    rgb_float = _hsv_to_rgb(hsv)
    rgb = (rgb_float * 255).astype(np.uint8)

    return rgb
