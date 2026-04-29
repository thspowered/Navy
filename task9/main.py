from src.visualization import run_app


if __name__ == "__main__":
    print("=" * 58)
    print("  Task 9 – Fractal Landscape (3D)")
    print("=" * 58)
    print()
    print("Algoritmus:")
    print("  Diamond-Square (spatial subdivision)")
    print("  1. Stvorec rozdelime na 2x2 mriezku")
    print("  2. Vertikalne posunieme 5 novych vrcholov o nahodnu hodnotu")
    print("  3. Pre kazdy novy stvorec opakujeme s mensim posunom")
    print()
    print("Farebne urovne (3D, multiple elevation):")
    print("  - voda    (modra)")
    print("  - piesok  (svetlo zlta)")
    print("  - trava   (zelena)")
    print("  - hora    (hneda)")
    print("  - snih    (biela)")
    print()
    print("Spustam interaktivnu vizualizaciu...")
    print()
    run_app()
