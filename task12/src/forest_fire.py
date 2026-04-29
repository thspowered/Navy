"""
forest_fire.py - Cellular automaton: Forest fire algorithm (Drossel-Schwabl).

Stavy bunky:
    0 = empty    (hneda)
    1 = tree     (zelena)
    2 = burning  (oranzova)
    3 = burnt    (cierna)

Pravidla (jeden krok automatu, synchronne):
    1. Empty alebo Burnt sa s pravdepodobnostou p stane Tree, inak skoncí
       ako Empty (cize Burnt z minuleho kroku sa "vycisti" na empty).
    2. Tree, ktoreho aspon jeden sused je Burning, sa stane Burning.
    3. Tree, ktoreho ziadny sused nehori, sa s pravdepodobnostou f
       sam vznieti (blesk) – stane sa Burning.
    4. Burning sa v dalsom kroku stane Burnt.

Okolie:
    von Neumann (4 susedia: hore/dole/vlavo/vpravo) – default
    Moore       (8 susedov vratane diagonal)

Hranice mriezky su toroidalne (np.roll). Vypoctovo je vsetko vektorizovane
cez numpy, takze CA bezi rychlo aj pri 200x200 mriezke.
"""

from __future__ import annotations
import numpy as np

# Stavy
EMPTY   = 0
TREE    = 1
BURNING = 2
BURNT   = 3

# Default parametre podla zadania
DEFAULT_P       = 0.05    # pravdepodobnost vyrastu noveho stromu
DEFAULT_F       = 0.001   # pravdepodobnost spontanneho zapalenia
DEFAULT_DENSITY = 0.5     # pociatocna hustota lesa


# Offsety susedov pre dva typy okolia
_NEIGHBOR_OFFSETS = {
    "von_neumann": [(-1, 0), (1, 0), (0, -1), (0, 1)],
    "moore":       [(dy, dx) for dy in (-1, 0, 1)
                              for dx in (-1, 0, 1)
                              if (dy, dx) != (0, 0)],
}


def init_grid(
    size: int = 100,
    density: float = DEFAULT_DENSITY,
    seed: int | None = None,
) -> np.ndarray:
    """
    Vygeneruje pociatocnu mriezku NxN: kazda bunka je s pravdepodobnostou
    `density` strom, inak prazdne miesto.

    Parametre
    ---------
    size    : rozmer mriezky (N x N)
    density : pociatocna hustota stromov v [0, 1]
    seed    : seed pre random generator (alebo None pre nahodu)

    Navratova hodnota
    -----------------
    grid : np.ndarray (size, size), dtype=int8 – stavy buniek
    """
    rng = np.random.default_rng(seed)
    grid = np.zeros((size, size), dtype=np.int8)
    grid[rng.random((size, size)) < density] = TREE
    return grid


def burning_neighbor_mask(
    grid: np.ndarray,
    neighborhood: str = "von_neumann",
) -> np.ndarray:
    """
    Pre kazdu bunku vrati True ak ma aspon jedneho horiaceho suseda.

    Pouziva np.roll – mriezka je toroidalna (lava hrana susedi s pravou).
    Vektorizovane: vyzaduje len 4 (resp. 8) operacii na celom poli.
    """
    offsets = _NEIGHBOR_OFFSETS[neighborhood]
    burning = grid == BURNING
    has_burning_neighbor = np.zeros_like(burning, dtype=bool)
    for dy, dx in offsets:
        # Posun masky horiacich buniek o (dy, dx) – tak sa do bunky (y, x)
        # premietne hodnota burning[y - dy, x - dx], teda jej sused.
        has_burning_neighbor |= np.roll(burning, shift=(dy, dx), axis=(0, 1))
    return has_burning_neighbor


def step(
    grid: np.ndarray,
    p: float = DEFAULT_P,
    f: float = DEFAULT_F,
    neighborhood: str = "von_neumann",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Vykona jeden synchronizovany krok forest-fire automatu.

    Implementacia presne podla zadania:
      1. Empty/Burnt  -> Tree    s pravdepodobnostou p, inak Empty
      2. Tree, sused horí  -> Burning   (deterministicky)
      3. Tree, ziadny sused nehori -> Burning s pravdepodobnostou f
      4. Burning  -> Burnt   (vzdy, v dalsom kroku)

    Vsetky pravidla sa aplikuju synchronne – pouzivame `grid` (stary stav)
    pri rozhodovani a zapisujeme do `new` (novy stav). Cierna ("burnt")
    farba je tak viditelna len jeden krok po vyhasnuti, presne ako v PDF.
    """
    if rng is None:
        rng = np.random.default_rng()

    new = grid.copy()
    rand = rng.random(grid.shape)

    # Pravidlo 4: Burning -> Burnt
    new[grid == BURNING] = BURNT

    # Pravidlo 1: Empty/Burnt -> Tree s pravdep. p, inak Empty
    empty_or_burnt = (grid == EMPTY) | (grid == BURNT)
    becomes_tree = empty_or_burnt & (rand < p)
    new[becomes_tree] = TREE
    # Burnt z minuleho kroku, ktore sa nestali stromom, "vycistime" na empty
    new[(grid == BURNT) & ~becomes_tree] = EMPTY

    # Pravidla 2 a 3 sa tykaju len buniek so stavom Tree
    tree = grid == TREE
    burning_neighbor = burning_neighbor_mask(grid, neighborhood=neighborhood)

    # Pravidlo 2: Tree so susedom v plamenoch -> Burning
    new[tree & burning_neighbor] = BURNING

    # Pravidlo 3: Tree bez horiaceho suseda -> Burning s pravdepodobnostou f
    spontaneous = tree & ~burning_neighbor & (rand < f)
    new[spontaneous] = BURNING

    return new


def step_inplace(
    state: dict,
) -> None:
    """
    Pohodlny wrapper okolo `step` – aktualizuje slovnik `state` na mieste.
    Ocakava klice: 'grid', 'p', 'f', 'neighborhood', 'rng', 'iteration'.
    """
    state["grid"] = step(
        state["grid"],
        p=state.get("p", DEFAULT_P),
        f=state.get("f", DEFAULT_F),
        neighborhood=state.get("neighborhood", "von_neumann"),
        rng=state.get("rng"),
    )
    state["iteration"] = state.get("iteration", 0) + 1


def grid_stats(grid: np.ndarray) -> dict:
    """Vrati pocet buniek v kazdom stave (pre info panel / diagnostiku)."""
    total = grid.size
    return {
        "total":   total,
        "empty":   int((grid == EMPTY).sum()),
        "tree":    int((grid == TREE).sum()),
        "burning": int((grid == BURNING).sum()),
        "burnt":   int((grid == BURNT).sum()),
    }
