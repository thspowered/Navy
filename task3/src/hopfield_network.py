import numpy as np


class HopfieldNetwork:
    """
    Hopfieldova siet s n_neurons neuronmi.

    Parametre
    ---------
    n_neurons : int
        Pocet neuronovv sieti (pre mriezku 10x10 = 100).
    """

    def __init__(self, n_neurons: int) -> None:
        self.n_neurons: int = n_neurons
        # Matica vah: symetricka, nulova hlavna diagonal
        self.W: np.ndarray = np.zeros((n_neurons, n_neurons), dtype=float)
        # Zoznam ulozenych vzorov (pre zobrazenie)
        self.stored_patterns: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Trenovanie (Hebbovo pravidlo)
    # ------------------------------------------------------------------

    def train(self, pattern: np.ndarray) -> None:
        """
        Ulozi vzor do siete pomocou Hebbovho pravidla.

        Aktualizacia: W += outer(p, p), potom nastavenie hlavnej diagonaly na 0.
        Vzor musi byt bipolarny: hodnoty +1 alebo -1, tvar (n_neurons,).

        Parametre
        ---------
        pattern : np.ndarray
            Bipolarny vzor tvaru (n_neurons,).
        """
        p = pattern.astype(float).flatten()
        if len(p) != self.n_neurons:
            raise ValueError(
                f"Vzor ma {len(p)} prvkov, ocakavane {self.n_neurons}."
            )
        # Hebbovo pravidlo: pridaj vonkajsi sucin
        self.W += np.outer(p, p)
        # Nulova hlavna diagonal (neuron sa nenapaja sam na seba)
        np.fill_diagonal(self.W, 0.0)
        # Uloz kopiu vzoru pre neskor
        self.stored_patterns.append(p.copy())

    def forget_all(self) -> None:
        """Vymaze vsetky ulozene vzory a resetuje maticu vah."""
        self.W = np.zeros((self.n_neurons, self.n_neurons), dtype=float)
        self.stored_patterns = []

    # ------------------------------------------------------------------
    # Znamienkova funkcia
    # ------------------------------------------------------------------

    @staticmethod
    def _sign(x: np.ndarray) -> np.ndarray:
        """Bipolarny znak: >= 0 -> +1, inak -> -1."""
        return np.where(x >= 0, 1.0, -1.0)

    # ------------------------------------------------------------------
    # Synchronna obnova
    # ------------------------------------------------------------------

    def recover_sync(
        self, pattern: np.ndarray, max_iter: int = 50
    ) -> list[np.ndarray]:
        """
        Synchronna obnova vzoru: vsetky neurony sa aktualizuju naraz v kazdom kroku.

        Vracia zoznam stavov (vratan pociatocneho), kde kazdy prvok je
        np.ndarray tvaru (n_neurons,).

        Parametre
        ---------
        pattern  : np.ndarray  – poskodeny vstupny vzor (bipolarny)
        max_iter : int          – maximalny pocet iteracii

        Navratova hodnota
        -----------------
        list[np.ndarray] – historia stavov od pociatocneho po konvergentny
        """
        state = pattern.astype(float).flatten().copy()
        history = [state.copy()]

        for _ in range(max_iter):
            # Vsetky neurony naraz
            new_state = self._sign(self.W @ state)
            history.append(new_state.copy())
            # Konvergencia: stav sa nezmenil
            if np.array_equal(new_state, state):
                break
            state = new_state

        return history

    # ------------------------------------------------------------------
    # Asynchronna obnova
    # ------------------------------------------------------------------

    def recover_async(
        self, pattern: np.ndarray, max_iter: int = 50
    ) -> list[np.ndarray]:
        """
        Asynchronna obnova vzoru: v kazdom kroku sa aktualizuje jeden nahodny neuron.
        Kazda jednotliva aktualizacia neuronu je novy stav v zozname.

        Parametre
        ---------
        pattern  : np.ndarray  – poskodeny vstupny vzor (bipolarny)
        max_iter : int          – maximalny pocet cyklov (1 cyklus = n_neurons krokov)

        Navratova hodnota
        -----------------
        list[np.ndarray] – historia stavov (kazda zmena neuronu = novy zaznam)
        """
        state = pattern.astype(float).flatten().copy()
        history = [state.copy()]

        for cycle in range(max_iter):
            # Nahodne poradie neuronovv tomto cykle
            order = np.random.permutation(self.n_neurons)
            state_before_cycle = state.copy()

            for neuron_idx in order:
                # Aktualizuj jeden neuron
                net_input = self.W[neuron_idx] @ state
                new_val = self._sign(np.array([net_input]))[0]

                if new_val != state[neuron_idx]:
                    state = state.copy()
                    state[neuron_idx] = new_val
                    # Kazda zmena je novy stav v historii
                    history.append(state.copy())

            # Konvergencia po celom cykle: ziadna zmena
            if np.array_equal(state, state_before_cycle):
                # Pridaj posledny stav pre zobrazenie konvergencie
                history.append(state.copy())
                break

        return history

    # ------------------------------------------------------------------
    # Vlastnost: maximalny odporucany pocet vzorov
    # ------------------------------------------------------------------

    @property
    def max_recommended(self) -> int:
        """
        Maximalny odporucany pocet vzorov bez vyraznej strat kapacity.
        Podla pravidla: ~0.138 * n_neurons.
        """
        return int(0.138 * self.n_neurons)
