"""
main.py – Vstupny bod pre Ulohu 7: IFS (Iterated Function System).

Spaja dohromady:
    1. IFS generator         (src/ifs.py)
    2. Interaktivna vizualizacia (src/visualization.py)

Pouzitie:
    cd task7
    python main.py

Postup prace v aplikacii:
    1. Klikni na model (1 alebo 2) pre generovanie fraktalu
    2. "Clear canvas" – vymaze canvas
    - Slider / Play / Prev / Next / Restart – animacia generovania

Algoritmus:
    IFS generovanie:
        1. Zaciname v bode (0, 0, 0)
        2. Nahodne vyberieme jednu zo 4 transformacii (p = 0.25)
        3. Aplikujeme afinnu transformaciu na aktualny bod:
            [x']   [a b c]   [x]   [j]
            [y'] = [d e f] * [y] + [k]
            [z']   [g h i]   [z]   [l]
        4. Ulozime novy bod do historie
        5. Opakujeme krok 2-4 (50 000 iteracii)

Modely:
    1. First model  – 3D fern s 4 transformaciami
    2. Second model – 3D fern s 4 transformaciami
"""

from src.visualization import run_app


if __name__ == "__main__":
    print("=" * 58)
    print("  Task 7 – IFS: Iterated Function System Fractals")
    print("=" * 58)
    print()
    print("Modely:")
    print("  1. First model  – 3D fern (4 transformacie)")
    print("  2. Second model – 3D fern (4 transformacie)")
    print()
    print("Algoritmus:")
    print("  - Nahodny vyber transformacie (p = 0.25)")
    print("  - Afinna transformacia v 3D priestore")
    print("  - 50 000 iteracii")
    print()
    print("Spustam interaktivnu vizualizaciu...")
    print()
    run_app()
