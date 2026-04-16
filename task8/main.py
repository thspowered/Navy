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
