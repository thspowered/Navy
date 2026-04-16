import numpy as np
from enum import IntEnum


class Action(IntEnum):
    UP    = 0
    DOWN  = 1
    LEFT  = 2
    RIGHT = 3


class CellType(IntEnum):
    EMPTY  = 0
    MOUSE  = 1
    CHEESE = 2
    TRAP   = 3
    WALL   = 4


# Delta (dr, dc) pre kazdu akciu
ACTION_DELTAS: dict[int, tuple[int, int]] = {
    int(Action.UP):    (-1,  0),
    int(Action.DOWN):  ( 1,  0),
    int(Action.LEFT):  ( 0, -1),
    int(Action.RIGHT): ( 0,  1),
}


class QLearningAgent:
    """
    Q-learning agent pre gridovu hru 'Najdi syr'.

    Parametre
    ---------
    n_rows, n_cols : rozmer mriezky
    alpha          : rychlost ucenia (learning rate)
    gamma          : diskontny faktor
    epsilon        : pociatocna pravdepodobnost nahodnej akcie
    epsilon_decay  : mnozitel poklesu epsilon po kazdom epizode
    epsilon_min    : minimalna hodnota epsilon
    reward_cheese  : odmena za najdenie syra
    reward_trap    : trest za pasc
    reward_step    : odmena za kazdy krok (zaporny = minimalizuj kroky)
    """

    def __init__(
        self,
        n_rows: int,
        n_cols: int,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        reward_cheese: float = 100.0,
        reward_trap: float = -100.0,
        reward_step: float = -1.0,
    ) -> None:
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.reward_cheese = reward_cheese
        self.reward_trap = reward_trap
        self.reward_step = reward_step

        # Q-tabulka: (riadok, stlpec, akcia)
        self.Q: np.ndarray = np.zeros((n_rows, n_cols, 4), dtype=float)
        # Matica odmien: (riadok, stlpec)
        self.R: np.ndarray = np.zeros((n_rows, n_cols), dtype=float)

        self.n_episodes_trained: int = 0
        self.converged_at: int | None = None   # epizoda, kedy agent prvykrat nasiel syr

    # ── Pomocne metody ──────────────────────────────────────────────────────────

    def _build_reward_matrix(self, grid: np.ndarray) -> None:
        """Vyplni maticu odmien podla aktualnej konfiguracie mriezky."""
        self.R = np.full((self.n_rows, self.n_cols), self.reward_step)
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                ct = int(grid[r, c])
                if ct == CellType.CHEESE:
                    self.R[r, c] = self.reward_cheese
                elif ct == CellType.TRAP:
                    self.R[r, c] = self.reward_trap
                elif ct == CellType.WALL:
                    self.R[r, c] = 0.0

    def _next_state(
        self,
        state: tuple[int, int],
        action: int,
        grid: np.ndarray,
    ) -> tuple[int, int]:
        """Vrati nasledujuci stav po vykonani akcie (stena/okraj = zostatok)."""
        dr, dc = ACTION_DELTAS[action]
        nr, nc = state[0] + dr, state[1] + dc
        if not (0 <= nr < self.n_rows and 0 <= nc < self.n_cols):
            return state
        if int(grid[nr, nc]) == int(CellType.WALL):
            return state
        return (nr, nc)

    def _is_terminal(self, state: tuple[int, int], grid: np.ndarray) -> bool:
        ct = int(grid[state[0], state[1]])
        return ct in (int(CellType.CHEESE), int(CellType.TRAP))

    def _choose_action(self, state: tuple[int, int]) -> int:
        """Epsilon-greedy vyber akcie."""
        if np.random.random() < self.epsilon:
            return int(np.random.randint(4))
        return int(np.argmax(self.Q[state[0], state[1]]))

    # ── Trenovanie ──────────────────────────────────────────────────────────────

    def train(
        self,
        grid: np.ndarray,
        mouse_pos: tuple[int, int],
        n_episodes: int = 1000,
    ) -> list[int]:
        """
        Trenovanie Q-learning agenta.

        Parametre
        ---------
        grid        : np.ndarray tvaru (n_rows, n_cols) s hodnotami CellType
        mouse_pos   : pociatocna pozicia mysi (riadok, stlpec)
        n_episodes  : pocet epizod trenovania

        Navratova hodnota
        -----------------
        list[int] – pocet krokov v kazdom epizode
        """
        self._build_reward_matrix(grid)
        steps_per_ep: list[int] = []
        max_steps = self.n_rows * self.n_cols * 4

        for ep in range(n_episodes):
            state = mouse_pos
            steps = 0
            found_cheese = False

            while not self._is_terminal(state, grid) and steps < max_steps:
                action = self._choose_action(state)
                nxt = self._next_state(state, action, grid)
                reward = self.R[nxt[0], nxt[1]]

                best_next = float(np.max(self.Q[nxt[0], nxt[1]]))
                self.Q[state[0], state[1], action] += self.alpha * (
                    reward + self.gamma * best_next
                    - self.Q[state[0], state[1], action]
                )

                state = nxt
                steps += 1

            # Zisti, ci agent nasiel syr v tejto epizode
            if int(grid[state[0], state[1]]) == int(CellType.CHEESE):
                found_cheese = True
                if self.converged_at is None:
                    self.converged_at = self.n_episodes_trained + ep + 1

            steps_per_ep.append(steps)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.n_episodes_trained += n_episodes
        return steps_per_ep

    # ── Greedy cesta ────────────────────────────────────────────────────────────

    def find_path(
        self,
        grid: np.ndarray,
        mouse_pos: tuple[int, int],
        max_steps: int = 200,
    ) -> list[tuple[int, int]]:
        """
        Sleduje greedy politiku z mouse_pos.

        Navratova hodnota
        -----------------
        list[tuple[int,int]] – zoznam navstivenych pozicii
        """
        state = mouse_pos
        path = [state]
        visited: set[tuple[int, int]] = {state}

        for _ in range(max_steps):
            if self._is_terminal(state, grid):
                break
            action = int(np.argmax(self.Q[state[0], state[1]]))
            nxt = self._next_state(state, action, grid)
            path.append(nxt)
            if nxt in visited:
                break
            visited.add(nxt)
            state = nxt

        return path

    # ── Reset ───────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Resetuje Q-tabulku a stav trenovania."""
        self.Q = np.zeros((self.n_rows, self.n_cols, 4), dtype=float)
        self.R = np.zeros((self.n_rows, self.n_cols), dtype=float)
        self.epsilon = 1.0
        self.n_episodes_trained = 0
        self.converged_at = None
