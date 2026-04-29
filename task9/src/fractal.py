import numpy as np


def diamond_square(
    size_exp: int = 7,
    roughness: float = 0.55,
    initial_offset: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Vygeneruje fraktalovy heightmap pomocou Diamond-Square algoritmu
    (spatial subdivision na stvorcovej mriezke).

    Postup:
      1. Stvorec rozdelime na 2x2 mriezku.
      2. Vertikalne posunieme 5 novych vrcholov o nahodnu hodnotu.
      3. Opakujeme rekurzivne, posun (offset) sa kazdou iteraciou zmensuje
         podla parametra roughness.

    Parametre
    ---------
    size_exp        : exponent N => velkost mriezky = 2^N + 1 (napr. 7 => 129x129)
    roughness       : faktor zmensovania amplitudy (0..1) – vacsia = drsnejsi terren
    initial_offset  : pociatocna amplituda posunu vrcholov
    seed            : seed pre random generator (pre reprodukovatelnost)

    Navratova hodnota
    -----------------
    heights : np.ndarray tvaru (size, size) – vyskovy heightmap
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    size = (1 << size_exp) + 1  # 2^N + 1
    h = np.zeros((size, size), dtype=float)

    # Inicializacia rohov (mozno aj 0; nahodne body dodaju zaujimavejsi tvar)
    h[0, 0]              = rng.uniform(-initial_offset, initial_offset)
    h[0, size - 1]       = rng.uniform(-initial_offset, initial_offset)
    h[size - 1, 0]       = rng.uniform(-initial_offset, initial_offset)
    h[size - 1, size - 1] = rng.uniform(-initial_offset, initial_offset)

    step = size - 1
    offset = initial_offset

    # Hlavny cyklus – kazda iteracia zmensi step na polovicu a offset o roughness
    while step > 1:
        half = step // 2

        # ── Diamond krok ────────────────────────────────────────────────────────
        # Stred kazdeho stvorca = priemer 4 rohov + nahodny posun
        for y in range(half, size, step):
            for x in range(half, size, step):
                avg = (
                    h[y - half, x - half] +
                    h[y - half, x + half] +
                    h[y + half, x - half] +
                    h[y + half, x + half]
                ) * 0.25
                h[y, x] = avg + rng.uniform(-offset, offset)

        # ── Square krok ─────────────────────────────────────────────────────────
        # Stredy hran = priemer susedov + nahodny posun
        for y in range(0, size, half):
            x_start = half if (y // half) % 2 == 0 else 0
            for x in range(x_start, size, step):
                vals = []
                if x - half >= 0:
                    vals.append(h[y, x - half])
                if x + half < size:
                    vals.append(h[y, x + half])
                if y - half >= 0:
                    vals.append(h[y - half, x])
                if y + half < size:
                    vals.append(h[y + half, x])
                if vals:
                    h[y, x] = np.mean(vals) + rng.uniform(-offset, offset)

        step = half
        offset *= roughness

    return h


def midpoint_displacement_1d(
    n_iterations: int = 8,
    initial_offset: float = 1.0,
    roughness: float = 0.55,
    seed: int | None = None,
) -> np.ndarray:
    """
    1D variant midpoint displacement (povodny zaklad fraktal landscape).

    Z jednej priamky postupne vyrobi krivku tym, ze stred kazdeho segmentu
    posunie vertikalne o nahodnu hodnotu (s 50 % pravdepodobnostou hore/dole).
    Amplituda sa kazdou iteraciou zmensuje.

    Navratova hodnota
    -----------------
    y : np.ndarray dlzky 2^n_iterations + 1 – y-suradnice profilu
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    n = (1 << n_iterations) + 1
    y = np.zeros(n)

    step = n - 1
    offset = initial_offset
    while step > 1:
        half = step // 2
        for i in range(half, n, step):
            avg = 0.5 * (y[i - half] + y[i + half])
            y[i] = avg + rng.uniform(-offset, offset)
        step = half
        offset *= roughness

    return y


# ── Definicia farebnych vrstiev (multiple elevation, >= 3 colors) ──────────────
# Kazda vrstva = (relativna hranica vysky 0..1, RGB hex, popis)
TERRAIN_LAYERS = [
    (0.18, "#1f4e79", "voda"),         # tmava modra
    (0.28, "#3e8ed0", "plytka voda"),  # svetlejsia modra
    (0.34, "#e8d28a", "piesok"),       # piesok
    (0.55, "#3a9d3a", "trava"),        # zelena
    (0.72, "#6b4423", "hora"),         # hneda
    (0.88, "#8a8a8a", "skala"),        # seda
    (1.01, "#f5f5f5", "snih"),         # biela (snih na vrcholoch)
]


def heights_to_colors(heights: np.ndarray) -> np.ndarray:
    """
    Premeni heightmap na RGBA farebne pole podla vrstiev TERRAIN_LAYERS.

    Navratova hodnota
    -----------------
    rgba : np.ndarray tvaru (H, W, 4), hodnoty v [0, 1]
    """
    h_min = heights.min()
    h_max = heights.max()
    if h_max - h_min < 1e-9:
        norm = np.zeros_like(heights)
    else:
        norm = (heights - h_min) / (h_max - h_min)

    rgba = np.zeros((*heights.shape, 4), dtype=float)

    for threshold, hex_color, _name in TERRAIN_LAYERS:
        mask = norm <= threshold
        # Aplikuj farbu len na este nezafarbene pixely (kde alpha == 0)
        unset = rgba[..., 3] == 0
        target = mask & unset
        r = int(hex_color[1:3], 16) / 255.0
        g = int(hex_color[3:5], 16) / 255.0
        b = int(hex_color[5:7], 16) / 255.0
        rgba[target] = (r, g, b, 1.0)

    # Bezpecnostna poistka – ak nieco zostalo nezafarbene
    unset = rgba[..., 3] == 0
    if unset.any():
        rgba[unset] = (1.0, 1.0, 1.0, 1.0)

    return rgba
