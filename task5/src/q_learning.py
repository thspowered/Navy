"""
q_learning.py – Q-learning agent pre CartPole (pole-balancing).

Stav:      diskretizovany z [cart_pos, cart_vel, pole_angle, pole_angular_vel]
Akcie:     0 = tlac vlavo, 1 = tlac vpravo
Diskret.:  kazda dimenzia rozdelena do fixnych intervalov (bins)
"""

import numpy as np
import gymnasium as gym


# ── Diskretizacne hranice pre kazdu dimenziu stavu ─────────────────────────────
BINS = [
    np.linspace(-2.4,    2.4,    7)[1:-1],   # cart position     (6 intervalov)
    np.linspace(-3.5,    3.5,    7)[1:-1],   # cart velocity     (6 intervalov)
    np.linspace(-0.2618, 0.2618, 11)[1:-1],  # pole angle        (10 intervalov)
    np.linspace(-3.5,    3.5,    11)[1:-1],  # pole angular vel  (10 intervalov)
]
N_BINS = [len(b) + 1 for b in BINS]   # [6, 6, 10, 10]


def discretize(obs: np.ndarray) -> tuple[int, ...]:
    """Prevedi spojity stav na diskretny index."""
    return tuple(int(np.digitize(obs[i], BINS[i])) for i in range(4))


class CartPoleQLearning:
    """
    Q-learning agent pre CartPole-v1.

    Parametre
    ---------
    alpha          : rychlost ucenia (learning rate)
    gamma          : diskontny faktor
    epsilon        : pociatocna exploracna miera
    epsilon_decay  : pokles epsilon po kazdom epizode
    epsilon_min    : minimalna hodnota epsilon
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ) -> None:
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min

        # Q-tabulka: shape (6, 6, 10, 10, 2)
        self.Q: np.ndarray = np.zeros(N_BINS + [2], dtype=float)

        self.n_episodes_trained: int = 0
        self.best_score: float       = 0.0
        self.last_avg_score: float   = 0.0

    # ── Vyber akcie ─────────────────────────────────────────────────────────────

    def choose_action(self, state: tuple[int, ...], greedy: bool = False) -> int:
        """Epsilon-greedy vyber akcie; pri greedy=True vzdy najlepsia akcia."""
        if not greedy and np.random.random() < self.epsilon:
            return int(np.random.randint(2))
        return int(np.argmax(self.Q[state]))

    # ── Trenovanie ──────────────────────────────────────────────────────────────

    def train(self, n_episodes: int = 5000) -> list[float]:
        """
        Trenovanie Q-learning agenta na CartPole-v1.

        Navratova hodnota
        -----------------
        list[float] – celkove odmeny v kazdom epizode
        """
        env = gym.make("CartPole-v1")
        scores: list[float] = []

        for _ in range(n_episodes):
            obs, _ = env.reset()
            state  = discretize(obs)
            total  = 0.0
            done   = False

            while not done:
                action                        = self.choose_action(state)
                next_obs, reward, term, trunc, _ = env.step(action)
                done                          = term or trunc
                next_state                    = discretize(next_obs)

                best_next = float(np.max(self.Q[next_state]))
                self.Q[state][action] += self.alpha * (
                    reward
                    + self.gamma * best_next * (not done)
                    - self.Q[state][action]
                )
                state  = next_state
                total += reward

            scores.append(total)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        env.close()
        self.n_episodes_trained += n_episodes
        if scores:
            self.best_score     = max(self.best_score, max(scores))
            self.last_avg_score = float(np.mean(scores[-100:]))
        return scores

    # ── Zber dat pre neuronovu siet ─────────────────────────────────────────────

    def collect_experiences(
        self, n_episodes: int = 200
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Spusti greedy politiku a zbiera (stav, akcia) pary pre trenovanie NN.

        Navratova hodnota
        -----------------
        (states, actions) – numpy polia tvaru (N,4) a (N,)
        """
        env     = gym.make("CartPole-v1")
        states  = []
        actions = []

        for _ in range(n_episodes):
            obs, _ = env.reset()
            done   = False
            while not done:
                state  = discretize(obs)
                action = self.choose_action(state, greedy=True)
                states.append(obs.copy())
                actions.append(action)
                obs, _, term, trunc, _ = env.step(action)
                done = term or trunc

        env.close()
        return np.array(states, dtype=float), np.array(actions, dtype=int)

    # ── Spustenie epizody (trajektoria pre vizualizaciu) ───────────────────────

    def run_episode(self, max_steps: int = 500) -> list[np.ndarray]:
        """Spusti greedy epizodu a vrati trajektoriu stavov."""
        env        = gym.make("CartPole-v1")
        obs, _     = env.reset()
        trajectory = [obs.copy()]
        done       = False
        steps      = 0

        while not done and steps < max_steps:
            action             = self.choose_action(discretize(obs), greedy=True)
            obs, _, term, trunc, _ = env.step(action)
            done               = term or trunc
            trajectory.append(obs.copy())
            steps             += 1

        env.close()
        return trajectory

    # ── Reset ───────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Resetuje Q-tabulku a stav trenovania."""
        self.Q                  = np.zeros(N_BINS + [2], dtype=float)
        self.epsilon            = 1.0
        self.n_episodes_trained = 0
        self.best_score         = 0.0
        self.last_avg_score     = 0.0
