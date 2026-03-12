"""
main.py – Vstupny bod pre Ulohu 3: Hopfieldova siet – Asociativna pamat.

Spaja dohromady:
    1. Hopfieldova siet         (src/hopfield_network.py)
    2. Interaktivna vizualizacia (src/visualization.py)

Pouzitie:
    cd task3
    python main.py

Interakcia:
    - Kliknite na bunky mriezky 10x10 pre kreslenie vzoru (+1 = modra, -1 = tmava)
    - "Save Pattern"        – ulozi aktualny vzor do siete (Hebbovo pravidlo)
    - "Repair Sync"         – synchronna obnova: vsetky neurony naraz
    - "Repair Async"        – asynchronna obnova: jeden neuron po druhom
    - "Show Saved Patterns" – zobrazi vsetky ulozene vzory v novej okne
    - "Clear Grid"          – vymaze mriezku (nastavi vsetky bunky na -1)
    - Slider / Play / Prev / Next / Restart – ovladanie animacie

Priklady vzorov (kresli rucne na mriezku 10x10):

    Pismeno "T":
        Riadok 1: vsetky stlpce +1
        Stlpce 4-5: vsetky riadky +1

    Pismeno "L":
        Stlpec 0: vsetky riadky +1
        Riadok 9: vsetky stlpce +1

    Pismeno "X":
        Hlavna a vedlajsia diagonal +1

Odporucany postup:
    1. Nakreslite vzor (napr. pismeno).
    2. Kliknite "Save Pattern".
    3. Opakujte pre 2-3 vzory.
    4. Nakreslite castocne poskodeny vzor.
    5. Kliknite "Repair Sync" alebo "Repair Async" a sledujte obnovu.
"""

from src.visualization import run_app


if __name__ == "__main__":
    print("=" * 62)
    print("  Task 3 – Hopfield Network: Associative Memory")
    print("=" * 62)
    print()
    print("Network capacity (10x10 = 100 neurons):")
    from src.hopfield_network import HopfieldNetwork
    _demo_net = HopfieldNetwork(100)
    print(f"  Max recommended patterns: {_demo_net.max_recommended}")
    print()
    print("Launching interactive visualisation...")
    print()
    run_app()
