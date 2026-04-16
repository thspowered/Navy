"""
main.py – Vstupny bod pre Ulohu 5: Pole-balancing (CartPole).

Spaja dohromady:
    1. Q-learning agent     (src/q_learning.py)
    2. Neuronova siet       (src/neural_network.py)
    3. Interaktivna vizualizacia (src/visualization.py)

Pouzitie:
    cd task5
    python main.py

Postup prace v aplikacii:
    1. "Train Q-agent"    – natrenuuje Q-learning agenta (5 000 epizod)
    2. "Train Neural Net" – zbere skusenosti Q-agenta a natrenuuje NN
                           (supervised learning: stav -> akcia)
    3. "Run Neural Net"   – zobrazi animaciu nauceneho neuronoveho agenta
    4. "Reset"            – resetuje vsetko a zacni odznova
    - Slider / Play / Prev / Next / Restart – ovladanie animacie

Algoritmus:
    Q-learning (diskretizovany stav):
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

        Diskretizacia stavu (CartPole-v1):
            cart_pos  : 6 intervalov  [-2.4,  2.4]
            cart_vel  : 6 intervalov  [-3.5,  3.5]
            pole_angle: 10 intervalov [-0.26, 0.26]
            pole_vel  : 10 intervalov [-3.5,  3.5]

        Parametre:
            alpha         = 0.1
            gamma         = 0.99
            epsilon       = 1.0 -> klesa na 0.01
            epsilon_decay = 0.995

    Neuronova siet (supervised):
        Architektura: 4 -> 64 -> 32 -> 2
        Aktivacia:    ReLU (skryte), Softmax (vystup)
        Strata:       krizova entropia
        Optimizer:    mini-batch SGD
        Trenovanie:   na (stav, akcia) paroch zo Q-agenta (200 greedy epizod)
"""

from src.visualization import run_app


if __name__ == "__main__":
    print("=" * 58)
    print("  Task 5 – Pole-balancing: Q-learning → Neural Network")
    print("=" * 58)
    print()
    print("Q-learning parametre:")
    print("  alpha (learning rate) : 0.1")
    print("  gamma (discount)      : 0.99")
    print("  epsilon start         : 1.0  -> klesa na 0.01")
    print("  epsilon decay         : 0.995")
    print("  epizody               : 5 000")
    print()
    print("Diskretizacia stavu CartPole:")
    print("  cart_pos   : 6  intervalov  [-2.4,  2.4]")
    print("  cart_vel   : 6  intervalov  [-3.5,  3.5]")
    print("  pole_angle : 10 intervalov  [-0.26, 0.26]")
    print("  pole_vel   : 10 intervalov  [-3.5,  3.5]")
    print()
    print("Neuronova siet (supervised):")
    print("  architektura : 4 -> 64 -> 32 -> 2")
    print("  aktivacia    : ReLU + Softmax")
    print("  epochy       : 300")
    print("  batch size   : 64")
    print()
    print("Spustam interaktivnu vizualizaciu...")
    print()
    run_app()
