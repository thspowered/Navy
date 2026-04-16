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
