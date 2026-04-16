"""
main.py – Vstupny bod pre Ulohu 6: L-systems (Lindenmayer systems).

Spaja dohromady:
    1. L-system generator   (src/lsystem.py)
    2. Interaktivna vizualizacia (src/visualization.py)

Pouzitie:
    cd task6
    python main.py

Postup prace v aplikacii:
    1. Klikni na preset (1-4) pre okamzite vykreslenie fraktalu
    2. Alebo zadaj vlastny axiom, pravidlo, uhol a nesting
    3. "Draw custom"  – vykresli vlastny L-system
    4. "Clear canvas" – vymaze canvas
    - Slider / Play / Prev / Next / Restart – animacia kreslenia

Algoritmus:
    L-system generovanie:
        1. Zaciname s axiomom (pociatocny retazec)
        2. Opakovane aplikujeme prepisovacie pravidla (nesting-krat)
        3. Vysledny retazec interpretujeme ako instrukcie pre kresliaci kurzor

    Symboly:
        F  = posun vpred (kresli ciaru)
        b  = posun vpred (nekresli)
        +  = otocenie vpravo o dany uhol
        -  = otocenie vlavo o dany uhol
        [  = uloz aktualnu poziciu a uhol (push na zasobnik)
        ]  = obnov poslednu ulozen poziciu a uhol (pop zo zasobnika)

Predefinovane L-systemy:
    1. Carpet       : axiom F+F+F+F, rule F->F+F-F-FF+F+F-F, angle 90°
    2. Koch Snowflake: axiom F++F++F, rule F->F+F--F+F, angle 60°
    3. Weed 1       : axiom F, rule F->F[+F]F[-F]F, angle pi/7
    4. Weed 2       : axiom F, rule F->FF+[+F-F-F]-[-F+F+F], angle pi/8
"""

from src.visualization import run_app


if __name__ == "__main__":
    print("=" * 58)
    print("  Task 6 – L-systems: Lindenmayer Fractals")
    print("=" * 58)
    print()
    print("Predefinovane L-systemy:")
    print("  1. Carpet        : F+F+F+F, F->F+F-F-FF+F+F-F, 90°")
    print("  2. Koch Snowflake: F++F++F, F->F+F--F+F, 60°")
    print("  3. Weed 1        : F, F->F[+F]F[-F]F, pi/7")
    print("  4. Weed 2        : F, F->FF+[+F-F-F]-[-F+F+F], pi/8")
    print()
    print("Symboly:")
    print("  F = kresli vpred    b = posun bez kresby")
    print("  + = otoc vpravo     - = otoc vlavo")
    print("  [ = uloz poziciu    ] = obnov poziciu")
    print()
    print("Spustam interaktivnu vizualizaciu...")
    print()
    run_app()
