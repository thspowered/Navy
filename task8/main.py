"""
main.py – Vstupny bod pre Ulohu 8: Mandelbrot & Julia set.

Spaja dohromady:
    1. Vypocet fraktalov     (src/fractal.py)
    2. Interaktivna vizualizacia (src/visualization.py)

Pouzitie:
    cd task8
    python main.py

Postup prace v aplikacii:
    1. Klikni na "Mandelbrot set" alebo "Julia set"
    2. Klikni lavym tlacidlom na fraktal pre zoom in
    3. Klikni pravym tlacidlom pre zoom out
    4. "Zoom out (reset)" – navrat na predchadzajuci zoom
    5. "Clear canvas" – vymaze canvas
    - Slider / Play / Prev / Next / Restart – animacia generovania

Algoritmus:
    Mandelbrot:
        z0 = 0, z_{n+1} = z_n^2 + c  (c = suradnice pixelu)
        Bod patri do mnoziny ak |z_n| <= 2 pre vsetky n.

    Julia:
        z0 = suradnice pixelu, z_{n+1} = z_n^2 + c  (c = konstanta)
        Rovnaka podmienka ako Mandelbrot.

    Farba sa urcuje podla poctu iteracii (HSV farebny model).
"""

from src.visualization import run_app


if __name__ == "__main__":
    print("=" * 58)
    print("  Task 8 – TEA: Mandelbrot & Julia Set")
    print("=" * 58)
    print()
    print("Typy fraktalov:")
    print("  1. Mandelbrot set  – z0=0, c=pixel")
    print("  2. Julia set       – z0=pixel, c=konstanta")
    print()
    print("Algoritmus:")
    print("  z_{n+1} = z_n^2 + c")
    print("  |z| > 2 => bod unikol (nie je v mnozine)")
    print("  Farba = HSV podla poctu iteracii")
    print()
    print("Zoom: lavy klik = pribliz, pravy klik = oddial")
    print()
    print("Spustam interaktivnu vizualizaciu...")
    print()
    run_app()
