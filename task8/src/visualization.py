"""
visualization.py – Interaktivna vizualizacia Task 8: Mandelbrot & Julia set.

Rozlozenie figury (15x9 palcov, tmava tema):
  Vlavo  (60%) – imshow plot pre fraktal
  Vpravo (35%) – tlacidla typov + info panel
  Dolu         – animacne ovladacie prvky (slider, Play/Pause, Prev/Next, Restart)

Farebna schema (identicka s predchadzajucimi taskmi):
  BG_DARK  = "#0d1117"
  BG_PANEL = "#161b22"
  BG_AXES  = "#1c2333"
  TEXT     = "#f0f6fc"
  ACCENT   = "#58a6ff"

Zoom: kliknutim na fraktal sa priblizi na dane miesto (2x zoom).
      Prave tlacidlo mysi oddiali (0.5x zoom).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim_mod
from matplotlib.widgets import Button, Slider
from matplotlib.lines import Line2D

from .fractal import compute_mandelbrot, compute_julia, iterations_to_rgb

# ── Farby temy (identicky s predchadzajucimi taskmi) ───────────────────────────
BG_DARK    = "#0d1117"
BG_PANEL   = "#161b22"
BG_AXES    = "#1c2333"
TEXT_COLOR = "#f0f6fc"
ACCENT     = "#58a6ff"

COLOR_MANDEL = "#56d364"   # zelena – Mandelbrot
COLOR_JULIA  = "#bc8cff"   # fialova – Julia
COLOR_CLEAR  = "#8b949e"   # seda – clear

# ── Rozlisenie a parametre ─────────────────────────────────────────────────────
IMG_WIDTH  = 600
IMG_HEIGHT = 400
MAX_ITER   = 256
ANIM_FRAMES = 60  # pocet framov animacie pri zoom-e


def _apply_dark_theme() -> None:
    """Nastavi globalne matplotlib parametre pre tmavu temu."""
    plt.rcParams.update({
        "figure.facecolor": BG_DARK,
        "axes.facecolor":   BG_AXES,
        "axes.edgecolor":   TEXT_COLOR,
        "axes.labelcolor":  TEXT_COLOR,
        "xtick.color":      TEXT_COLOR,
        "ytick.color":      TEXT_COLOR,
        "text.color":       TEXT_COLOR,
        "grid.color":       BG_PANEL,
        "grid.alpha":       1.0,
    })


def run_app() -> None:
    """Spusti hlavnu interaktivnu aplikaciu Task 8 – Mandelbrot & Julia set."""
    _apply_dark_theme()

    # Stav aplikacie
    app_state: dict = {
        "mode":        None,     # "mandelbrot" alebo "julia"
        "x_min":       -2.0,
        "x_max":        1.0,
        "y_min":       -1.0,
        "y_max":        1.0,
        "iterations":  None,     # numpy array iteracii
        "frames":      [],       # list RGB framov pre animaciu
        "frame":       0,
        "total":       0,
        "playing":     False,
        "_upd_slider": False,
        "anim_obj":    None,
        "max_iter":    MAX_ITER,
        # Julia parametre
        "c_real":     -0.7,
        "c_imag":      0.27015,
        # Zoom historia
        "zoom_history": [],
    }

    # ── Figura ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 9), facecolor=BG_DARK)
    fig.suptitle(
        "Task 8 – TEA  ·  Mandelbrot & Julia Set",
        color=TEXT_COLOR, fontsize=14, fontweight="bold", y=0.97,
    )

    # Oddelovacia cara nad animacnymi ovladacmi
    fig.add_artist(
        Line2D([0.03, 0.97], [0.275, 0.275],
               transform=fig.transFigure,
               color=ACCENT, linewidth=0.8, alpha=0.5)
    )

    # ── Osa pre fraktal (vlavo) ────────────────────────────────────────────────
    ax_frac = fig.add_axes([0.03, 0.30, 0.56, 0.62])
    ax_frac.set_facecolor(BG_DARK)
    ax_frac.set_xticks([])
    ax_frac.set_yticks([])
    ax_frac.set_title("Fractal Visualization", color=TEXT_COLOR, fontsize=12, pad=8)

    img_display = [None]  # mutable holder pre imshow objekt

    # ── Tlacidla (vpravo) ──────────────────────────────────────────────────────
    btn_m_ax  = fig.add_axes([0.63, 0.86, 0.33, 0.055])
    btn_j_ax  = fig.add_axes([0.63, 0.80, 0.33, 0.055])
    btn_cl_ax = fig.add_axes([0.63, 0.74, 0.33, 0.055])
    btn_zi_ax = fig.add_axes([0.63, 0.68, 0.33, 0.055])

    btn_mandel = Button(btn_m_ax, "Mandelbrot set",
                        color=BG_PANEL, hovercolor="#1f3a1f")
    btn_julia  = Button(btn_j_ax, "Julia set  (c = -0.7 + 0.27i)",
                        color=BG_PANEL, hovercolor="#2a1f3a")
    btn_clear  = Button(btn_cl_ax, "Clear canvas",
                        color=BG_PANEL, hovercolor="#2a2a2a")
    btn_zout   = Button(btn_zi_ax, "Zoom out (reset)",
                        color=BG_PANEL, hovercolor="#2a2a3a")

    btn_mandel.label.set_color(COLOR_MANDEL)
    btn_julia.label.set_color(COLOR_JULIA)
    btn_clear.label.set_color(COLOR_CLEAR)
    btn_zout.label.set_color(ACCENT)

    for btn in (btn_mandel, btn_julia, btn_clear, btn_zout):
        btn.label.set_fontsize(10)
        btn.label.set_fontweight("bold")

    # Informacny panel (vpravo dolu)
    ax_info = fig.add_axes([0.63, 0.30, 0.33, 0.34])
    ax_info.set_facecolor(BG_PANEL)
    ax_info.set_xticks([])
    ax_info.set_yticks([])
    for spine in ax_info.spines.values():
        spine.set_edgecolor(ACCENT)
        spine.set_linewidth(1.2)

    info_text = ax_info.text(
        0.5, 0.5,
        "Zvolte typ fraktalu:\n"
        "  Mandelbrot alebo Julia\n\n"
        "Algoritmus:\n"
        "  z0 = 0 (Mandelbrot)\n"
        "  z0 = pixel (Julia)\n"
        "  z_{n+1} = z_n^2 + c\n"
        "  |z| > 2 => unik\n\n"
        "Zoom: klikni na fraktal\n"
        "      (lavy = pribliz,\n"
        "       pravy = oddial)",
        transform=ax_info.transAxes,
        ha="center", va="center",
        fontsize=8, family="monospace",
        color=TEXT_COLOR,
    )

    # ── Animacne ovladacie prvky (dolu) ─────────────────────────────────────────
    slider_ax   = fig.add_axes([0.10, 0.18, 0.80, 0.03], facecolor=BG_PANEL)
    btn_prev_ax = fig.add_axes([0.33, 0.09, 0.07, 0.05], facecolor=BG_PANEL)
    btn_play_ax = fig.add_axes([0.41, 0.09, 0.12, 0.05], facecolor=BG_PANEL)
    btn_next_ax = fig.add_axes([0.54, 0.09, 0.07, 0.05], facecolor=BG_PANEL)
    btn_rst_ax  = fig.add_axes([0.62, 0.09, 0.09, 0.05], facecolor=BG_PANEL)

    slider = Slider(
        slider_ax, "Iter", 0, 1,
        valinit=0, valstep=1, color=ACCENT,
    )
    slider.label.set_color(TEXT_COLOR)
    slider.valtext.set_color(TEXT_COLOR)

    btn_play    = Button(btn_play_ax, "▶ Play",  color=BG_PANEL, hovercolor="#2a2a4a")
    btn_prev    = Button(btn_prev_ax, "< Prev",  color=BG_PANEL, hovercolor="#2a2a4a")
    btn_next    = Button(btn_next_ax, "Next >",  color=BG_PANEL, hovercolor="#2a2a4a")
    btn_restart = Button(btn_rst_ax,  "Restart", color=BG_PANEL, hovercolor="#2a2a4a")

    for btn in (btn_play, btn_prev, btn_next, btn_restart):
        btn.label.set_color(TEXT_COLOR)

    # Stavovy riadok
    ax_status = fig.add_axes([0.03, 0.265, 0.56, 0.032])
    ax_status.set_facecolor(BG_PANEL)
    ax_status.set_xticks([])
    ax_status.set_yticks([])
    for spine in ax_status.spines.values():
        spine.set_edgecolor(ACCENT)
        spine.set_linewidth(0.8)

    status_text = ax_status.text(
        0.01, 0.5,
        "Kliknite na tlacidlo pre generovanie fraktalu.",
        transform=ax_status.transAxes,
        ha="left", va="center",
        fontsize=9, family="monospace",
        color=TEXT_COLOR,
    )

    # ── Pomocne funkcie ─────────────────────────────────────────────────────────

    def _set_status(msg: str, color: str = TEXT_COLOR) -> None:
        status_text.set_text(msg)
        status_text.set_color(color)
        fig.canvas.draw_idle()

    def _update_info(msg: str) -> None:
        info_text.set_text(msg)
        fig.canvas.draw_idle()

    def _compute_current() -> np.ndarray:
        """Vypocita iteracie pre aktualny stav (oblast + mod)."""
        xn, xx = app_state["x_min"], app_state["x_max"]
        yn, yx = app_state["y_min"], app_state["y_max"]
        mi = app_state["max_iter"]

        if app_state["mode"] == "mandelbrot":
            return compute_mandelbrot(xn, xx, yn, yx, IMG_WIDTH, IMG_HEIGHT, mi)
        else:
            return compute_julia(
                xn, xx, yn, yx, IMG_WIDTH, IMG_HEIGHT,
                app_state["c_real"], app_state["c_imag"], mi,
            )

    def _build_anim_frames(iterations: np.ndarray) -> list:
        """Vytvori framy animacie – postupne zvysovanie max_iter pre vizualizaciu."""
        frames = []
        mi = app_state["max_iter"]
        step = max(1, mi // ANIM_FRAMES)
        for cutoff in range(step, mi + 1, step):
            clipped = np.where(
                (iterations > 0) & (iterations <= cutoff),
                iterations, 0
            )
            rgb = iterations_to_rgb(clipped, mi)
            frames.append(rgb)
        # Posledny frame = plna vizualizacia
        full = iterations_to_rgb(iterations, mi)
        if len(frames) == 0 or not np.array_equal(frames[-1], full):
            frames.append(full)
        return frames

    def _draw_frame(frame_idx: int | None = None) -> None:
        """Vykresli dany frame animacie."""
        frames = app_state["frames"]
        if len(frames) == 0:
            ax_frac.clear()
            ax_frac.set_facecolor(BG_DARK)
            ax_frac.set_xticks([])
            ax_frac.set_yticks([])
            ax_frac.set_title("Fractal Visualization",
                              color=TEXT_COLOR, fontsize=12, pad=8)
            fig.canvas.draw_idle()
            return

        if frame_idx is None:
            frame_idx = app_state["frame"]
        frame_idx = int(np.clip(frame_idx, 0, len(frames) - 1))

        rgb = frames[frame_idx]

        if img_display[0] is None:
            ax_frac.clear()
            ax_frac.set_facecolor(BG_DARK)
            img_display[0] = ax_frac.imshow(
                rgb,
                extent=[app_state["x_min"], app_state["x_max"],
                        app_state["y_min"], app_state["y_max"]],
                origin="lower", aspect="auto",
            )
        else:
            img_display[0].set_data(rgb)
            img_display[0].set_extent([
                app_state["x_min"], app_state["x_max"],
                app_state["y_min"], app_state["y_max"],
            ])

        ax_frac.set_xticks([])
        ax_frac.set_yticks([])

        mode_name = "Mandelbrot" if app_state["mode"] == "mandelbrot" else "Julia"
        ax_frac.set_title(
            f"{mode_name} set  ·  "
            f"X=[{app_state['x_min']:.4f}, {app_state['x_max']:.4f}]  "
            f"Y=[{app_state['y_min']:.4f}, {app_state['y_max']:.4f}]",
            color=TEXT_COLOR, fontsize=10, pad=8,
        )

        fig.canvas.draw_idle()

    # ── Generovanie fraktalu ───────────────────────────────────────────────────

    def _run_fractal(mode: str, color: str) -> None:
        """Vygeneruje fraktal a pripravi animaciu."""
        if app_state["anim_obj"] is not None:
            try:
                app_state["anim_obj"].event_source.stop()
            except Exception:
                pass

        app_state["mode"] = mode
        img_display[0] = None

        if mode == "mandelbrot":
            app_state["x_min"], app_state["x_max"] = -2.0, 1.0
            app_state["y_min"], app_state["y_max"] = -1.0, 1.0
        else:
            app_state["x_min"], app_state["x_max"] = -1.5, 1.5
            app_state["y_min"], app_state["y_max"] = -1.5, 1.5

        app_state["zoom_history"] = []

        _set_status("Generujem fraktal...", color=ACCENT)
        fig.canvas.draw_idle()
        plt.pause(0.01)

        iterations = _compute_current()
        app_state["iterations"] = iterations

        frames = _build_anim_frames(iterations)
        app_state["frames"]  = frames
        app_state["frame"]   = len(frames) - 1
        app_state["total"]   = len(frames)
        app_state["playing"] = False

        n = len(frames)
        slider.valmin = 0
        slider.valmax = max(n - 1, 1)
        slider.set_val(n - 1)
        slider.ax.set_xlim(0, max(n - 1, 1))

        mode_name = "Mandelbrot" if mode == "mandelbrot" else "Julia"
        c_info = ""
        if mode == "julia":
            c_info = (f"\nc = {app_state['c_real']:.4f}"
                      f" + {app_state['c_imag']:.5f}i\n")

        _update_info(
            f"Typ: {mode_name} set\n"
            f"{c_info}\n"
            f"Algoritmus:\n"
            f"  z_{{n+1}} = z_n^2 + c\n"
            f"  |z| > 2 => unik\n"
            f"  max_iter = {app_state['max_iter']}\n\n"
            f"Rozlisenie: {IMG_WIDTH}x{IMG_HEIGHT}\n"
            f"Animacne framy: {n}\n\n"
            f"Zoom: klikni lavym na fraktal\n"
            f"Reset: tlacidlo 'Zoom out'"
        )

        _set_status(
            f"Hotovo! {mode_name} set vygenerovany. ({n} framov)",
            color=color,
        )
        _draw_frame(n - 1)

        anim_obj = anim_mod.FuncAnimation(
            fig, _anim_step, interval=30,
            blit=False, cache_frame_data=False,
        )
        app_state["anim_obj"] = anim_obj
        fig._anim = anim_obj

    # ── Zoom funkcia ───────────────────────────────────────────────────────────

    def _zoom_at(cx: float, cy: float, factor: float) -> None:
        """Pribliz/oddial na dane suradnice."""
        if app_state["mode"] is None:
            return

        # Uloz aktualny stav
        app_state["zoom_history"].append(
            (app_state["x_min"], app_state["x_max"],
             app_state["y_min"], app_state["y_max"])
        )

        dx = (app_state["x_max"] - app_state["x_min"]) / factor / 2
        dy = (app_state["y_max"] - app_state["y_min"]) / factor / 2

        app_state["x_min"] = cx - dx
        app_state["x_max"] = cx + dx
        app_state["y_min"] = cy - dy
        app_state["y_max"] = cy + dy

        if app_state["anim_obj"] is not None:
            try:
                app_state["anim_obj"].event_source.stop()
            except Exception:
                pass

        _set_status("Zoom – prepocitavam fraktal...", color=ACCENT)
        fig.canvas.draw_idle()
        plt.pause(0.01)

        img_display[0] = None
        iterations = _compute_current()
        app_state["iterations"] = iterations

        frames = _build_anim_frames(iterations)
        app_state["frames"]  = frames
        app_state["frame"]   = len(frames) - 1
        app_state["total"]   = len(frames)
        app_state["playing"] = False

        n = len(frames)
        slider.valmin = 0
        slider.valmax = max(n - 1, 1)
        slider.set_val(n - 1)
        slider.ax.set_xlim(0, max(n - 1, 1))

        mode_name = "Mandelbrot" if app_state["mode"] == "mandelbrot" else "Julia"
        color = COLOR_MANDEL if app_state["mode"] == "mandelbrot" else COLOR_JULIA
        _set_status(
            f"Zoom hotovy! {mode_name} – "
            f"X=[{app_state['x_min']:.6f}, {app_state['x_max']:.6f}]",
            color=color,
        )
        _draw_frame(n - 1)

        anim_obj = anim_mod.FuncAnimation(
            fig, _anim_step, interval=30,
            blit=False, cache_frame_data=False,
        )
        app_state["anim_obj"] = anim_obj
        fig._anim = anim_obj

    def on_click(event) -> None:
        """Kliknutie na fraktalovu plochu => zoom."""
        if event.inaxes != ax_frac:
            return
        if app_state["mode"] is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        if event.button == 1:  # lavy = zoom in
            _zoom_at(event.xdata, event.ydata, 2.0)
        elif event.button == 3:  # pravy = zoom out
            _zoom_at(event.xdata, event.ydata, 0.5)

    fig.canvas.mpl_connect("button_press_event", on_click)

    # ── Animacia ────────────────────────────────────────────────────────────────

    def _set_frame(fi: int) -> None:
        n = len(app_state["frames"])
        if n == 0:
            return
        fi = int(np.clip(fi, 0, n - 1))
        app_state["frame"]       = fi
        app_state["_upd_slider"] = True
        slider.set_val(fi)
        app_state["_upd_slider"] = False
        _draw_frame(fi)

        if fi == n - 1:
            mode_name = ("Mandelbrot" if app_state["mode"] == "mandelbrot"
                         else "Julia")
            color = (COLOR_MANDEL if app_state["mode"] == "mandelbrot"
                     else COLOR_JULIA)
            _set_status(f"Kompletne vykreslenie: {mode_name} set.", color=color)
        else:
            _set_status(f"Frame {fi + 1} / {n}")

    def _anim_step(_fn: int) -> list:
        if not app_state["playing"]:
            return []
        nxt = app_state["frame"] + 1
        n = len(app_state["frames"])
        if nxt >= n:
            nxt = n - 1
            app_state["playing"] = False
            btn_play.label.set_text("▶ Play")
        _set_frame(nxt)
        return []

    # ── Obsluha animacnych tlacidiel ────────────────────────────────────────────

    def on_slider(val: float) -> None:
        if app_state["_upd_slider"]:
            return
        fi = int(val)
        app_state["frame"] = fi
        _draw_frame(fi)

    def on_play(event) -> None:
        if len(app_state["frames"]) == 0:
            return
        if app_state["playing"]:
            app_state["playing"] = False
            btn_play.label.set_text("▶ Play")
        else:
            if app_state["frame"] >= len(app_state["frames"]) - 1:
                app_state["frame"] = 0
            app_state["playing"] = True
            btn_play.label.set_text("⏸ Pause")

    def on_prev(event) -> None:
        if len(app_state["frames"]) == 0:
            return
        app_state["playing"] = False
        btn_play.label.set_text("▶ Play")
        _set_frame(app_state["frame"] - 1)

    def on_next(event) -> None:
        if len(app_state["frames"]) == 0:
            return
        app_state["playing"] = False
        btn_play.label.set_text("▶ Play")
        _set_frame(app_state["frame"] + 1)

    def on_restart(event) -> None:
        if len(app_state["frames"]) == 0:
            return
        app_state["playing"] = False
        btn_play.label.set_text("▶ Play")
        _set_frame(0)

    slider.on_changed(on_slider)
    btn_play.on_clicked(on_play)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    btn_restart.on_clicked(on_restart)

    # ── Obsluha modelov ────────────────────────────────────────────────────────

    def on_mandel(event):
        _run_fractal("mandelbrot", COLOR_MANDEL)

    def on_julia(event):
        _run_fractal("julia", COLOR_JULIA)

    btn_mandel.on_clicked(on_mandel)
    btn_julia.on_clicked(on_julia)

    # ── Zoom out (reset) ───────────────────────────────────────────────────────

    def on_zout(event) -> None:
        if app_state["mode"] is None:
            return
        if app_state["zoom_history"]:
            xn, xx, yn, yx = app_state["zoom_history"].pop()
            app_state["x_min"], app_state["x_max"] = xn, xx
            app_state["y_min"], app_state["y_max"] = yn, yx
        else:
            if app_state["mode"] == "mandelbrot":
                app_state["x_min"], app_state["x_max"] = -2.0, 1.0
                app_state["y_min"], app_state["y_max"] = -1.0, 1.0
            else:
                app_state["x_min"], app_state["x_max"] = -1.5, 1.5
                app_state["y_min"], app_state["y_max"] = -1.5, 1.5

        if app_state["anim_obj"] is not None:
            try:
                app_state["anim_obj"].event_source.stop()
            except Exception:
                pass

        _set_status("Zoom out – prepocitavam...", color=ACCENT)
        fig.canvas.draw_idle()
        plt.pause(0.01)

        img_display[0] = None
        iterations = _compute_current()
        app_state["iterations"] = iterations

        frames = _build_anim_frames(iterations)
        app_state["frames"]  = frames
        app_state["frame"]   = len(frames) - 1
        app_state["total"]   = len(frames)
        app_state["playing"] = False

        n = len(frames)
        slider.valmin = 0
        slider.valmax = max(n - 1, 1)
        slider.set_val(n - 1)
        slider.ax.set_xlim(0, max(n - 1, 1))

        mode_name = "Mandelbrot" if app_state["mode"] == "mandelbrot" else "Julia"
        color = COLOR_MANDEL if app_state["mode"] == "mandelbrot" else COLOR_JULIA
        _set_status(f"Zoom reset – {mode_name} set.", color=color)
        _draw_frame(n - 1)

        anim_obj = anim_mod.FuncAnimation(
            fig, _anim_step, interval=30,
            blit=False, cache_frame_data=False,
        )
        app_state["anim_obj"] = anim_obj
        fig._anim = anim_obj

    btn_zout.on_clicked(on_zout)

    # ── Obsluha clear ───────────────────────────────────────────────────────────

    def on_clear(event) -> None:
        if app_state["anim_obj"] is not None:
            try:
                app_state["anim_obj"].event_source.stop()
            except Exception:
                pass

        app_state["mode"]       = None
        app_state["iterations"] = None
        app_state["frames"]     = []
        app_state["frame"]      = 0
        app_state["total"]      = 0
        app_state["playing"]    = False
        app_state["anim_obj"]   = None
        app_state["zoom_history"] = []
        img_display[0] = None
        btn_play.label.set_text("▶ Play")

        _update_info(
            "Zvolte typ fraktalu:\n"
            "  Mandelbrot alebo Julia\n\n"
            "Algoritmus:\n"
            "  z0 = 0 (Mandelbrot)\n"
            "  z0 = pixel (Julia)\n"
            "  z_{n+1} = z_n^2 + c\n"
            "  |z| > 2 => unik\n\n"
            "Zoom: klikni na fraktal\n"
            "      (lavy = pribliz,\n"
            "       pravy = oddial)"
        )
        _set_status("Canvas vymazany.")
        _draw_frame()

    btn_clear.on_clicked(on_clear)

    # ── Pociatocny stav ─────────────────────────────────────────────────────────
    _draw_frame()
    plt.show()
