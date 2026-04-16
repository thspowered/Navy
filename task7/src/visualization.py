"""
visualization.py – Interaktivna vizualizacia Task 7: IFS fraktaly.

Rozlozenie figury (15x9 palcov, tmava tema):
  Vlavo  (60%) – 3D scatter plot pre IFS fraktal
  Vpravo (35%) – tlacidla modelov + info panel
  Dolu         – animacne ovladacie prvky (slider, Play/Pause, Prev/Next, Restart)

Farebna schema (identicka s predchadzajucimi taskmi):
  BG_DARK  = "#0d1117"
  BG_PANEL = "#161b22"
  BG_AXES  = "#1c2333"
  TEXT     = "#f0f6fc"
  ACCENT   = "#58a6ff"
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim_mod
from matplotlib.widgets import Button, Slider
from matplotlib.lines import Line2D

from .ifs import generate_ifs, MODELS

# ── Farby temy (identicky s predchadzajucimi taskmi) ───────────────────────────
BG_DARK    = "#0d1117"
BG_PANEL   = "#161b22"
BG_AXES    = "#1c2333"
TEXT_COLOR = "#f0f6fc"
ACCENT     = "#58a6ff"

COLOR_M1   = "#56d364"   # zelena – prvy model
COLOR_M2   = "#bc8cff"   # fialova – druhy model
COLOR_CLEAR = "#8b949e"  # seda – clear


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
    """Spusti hlavnu interaktivnu aplikaciu Task 7 – IFS fraktaly."""
    _apply_dark_theme()

    # Stav aplikacie
    app_state: dict = {
        "points":      np.empty((0, 3)),
        "frame":       0,
        "total":       0,
        "playing":     False,
        "_upd_slider": False,
        "anim_obj":    None,
        "color":       ACCENT,
        "model_id":    None,
    }

    # ── Figura ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 9), facecolor=BG_DARK)
    fig.suptitle(
        "Task 7 – IFS  ·  Iterated Function System Fractals",
        color=TEXT_COLOR, fontsize=14, fontweight="bold", y=0.97,
    )

    # Oddelovacia cara nad animacnymi ovladacmi
    fig.add_artist(
        Line2D([0.03, 0.97], [0.275, 0.275],
               transform=fig.transFigure,
               color=ACCENT, linewidth=0.8, alpha=0.5)
    )

    # ── 3D osa pre fraktal (vlavo) ─────────────────────────────────────────────
    ax_frac = fig.add_axes([0.03, 0.30, 0.56, 0.62], projection="3d")
    ax_frac.set_facecolor(BG_DARK)
    ax_frac.set_xlabel("X", color=TEXT_COLOR)
    ax_frac.set_ylabel("Y", color=TEXT_COLOR)
    ax_frac.set_zlabel("Z", color=TEXT_COLOR)
    ax_frac.xaxis.pane.fill = False
    ax_frac.yaxis.pane.fill = False
    ax_frac.zaxis.pane.fill = False
    ax_frac.xaxis.pane.set_edgecolor(BG_PANEL)
    ax_frac.yaxis.pane.set_edgecolor(BG_PANEL)
    ax_frac.zaxis.pane.set_edgecolor(BG_PANEL)
    ax_frac.grid(True, alpha=0.3, color=BG_PANEL)
    ax_frac.set_title("IFS Fractal", color=TEXT_COLOR, fontsize=12, pad=8)

    # ── Tlacidla modelov (vpravo) ──────────────────────────────────────────────
    btn_m1_ax = fig.add_axes([0.63, 0.86, 0.33, 0.055])
    btn_m2_ax = fig.add_axes([0.63, 0.80, 0.33, 0.055])
    btn_clear_ax = fig.add_axes([0.63, 0.74, 0.33, 0.055])

    btn_m1 = Button(btn_m1_ax, "1: First model (3D Fern)",
                    color=BG_PANEL, hovercolor="#1f3a1f")
    btn_m2 = Button(btn_m2_ax, "2: Second model (3D Fern)",
                    color=BG_PANEL, hovercolor="#2a1f3a")
    btn_clear = Button(btn_clear_ax, "Clear canvas",
                       color=BG_PANEL, hovercolor="#2a2a2a")

    btn_m1.label.set_color(COLOR_M1)
    btn_m2.label.set_color(COLOR_M2)
    btn_clear.label.set_color(COLOR_CLEAR)

    for btn in (btn_m1, btn_m2, btn_clear):
        btn.label.set_fontsize(10)
        btn.label.set_fontweight("bold")

    # Informacny panel (vpravo dolu)
    ax_info = fig.add_axes([0.63, 0.30, 0.33, 0.40])
    ax_info.set_facecolor(BG_PANEL)
    ax_info.set_xticks([])
    ax_info.set_yticks([])
    for spine in ax_info.spines.values():
        spine.set_edgecolor(ACCENT)
        spine.set_linewidth(1.2)

    info_text = ax_info.text(
        0.5, 0.5,
        "Zvolte model (1 alebo 2)\n"
        "pre generovanie IFS fraktalu.\n\n"
        "Algoritmus:\n"
        "  1. Zaciname v bode (0, 0, 0)\n"
        "  2. Nahodne vyberieme\n"
        "     jednu zo 4 transformacii\n"
        "     (p = 0.25 pre kazdu)\n"
        "  3. Aplikujeme afinnu\n"
        "     transformaciu na bod\n"
        "  4. Ulozime novy bod\n"
        "  5. Opakujeme 50 000x",
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
        slider_ax, "Pts", 0, 1,
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
        "Kliknite na tlacidlo modelu pre generovanie fraktalu.",
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

    def _draw_fractal(frame: int | None = None) -> None:
        """Vykreslí IFS body az po dany frame."""
        ax_frac.clear()
        ax_frac.set_facecolor(BG_DARK)
        ax_frac.set_xlabel("X", color=TEXT_COLOR)
        ax_frac.set_ylabel("Y", color=TEXT_COLOR)
        ax_frac.set_zlabel("Z", color=TEXT_COLOR)
        ax_frac.xaxis.pane.fill = False
        ax_frac.yaxis.pane.fill = False
        ax_frac.zaxis.pane.fill = False
        ax_frac.xaxis.pane.set_edgecolor(BG_PANEL)
        ax_frac.yaxis.pane.set_edgecolor(BG_PANEL)
        ax_frac.zaxis.pane.set_edgecolor(BG_PANEL)
        ax_frac.grid(True, alpha=0.3, color=BG_PANEL)

        pts = app_state["points"]
        if len(pts) == 0:
            ax_frac.set_title("IFS Fractal", color=TEXT_COLOR, fontsize=12, pad=8)
            fig.canvas.draw_idle()
            return

        if frame is None:
            frame = app_state["frame"]
        frame = int(np.clip(frame, 0, len(pts) - 1))

        visible = pts[:frame + 1]
        color = app_state["color"]

        ax_frac.scatter(
            visible[:, 0], visible[:, 1], visible[:, 2],
            c=color, s=0.3, alpha=0.6, depthshade=True,
        )

        # Nastav limity podla vsetkych bodov
        ax_frac.set_xlim(pts[:, 0].min() - 0.1, pts[:, 0].max() + 0.1)
        ax_frac.set_ylim(pts[:, 1].min() - 0.1, pts[:, 1].max() + 0.1)
        ax_frac.set_zlim(pts[:, 2].min() - 0.1, pts[:, 2].max() + 0.1)

        name = ""
        mid = app_state["model_id"]
        if mid and mid in MODELS:
            name = MODELS[mid]["name"]
        title = f"IFS  –  {name}" if name else "IFS Fractal"
        ax_frac.set_title(title, color=TEXT_COLOR, fontsize=12, pad=8)

        fig.canvas.draw_idle()

    # ── Generovanie a spustenie animacie ────────────────────────────────────────

    def _run_ifs(model_id: int, color: str) -> None:
        """Vygeneruje IFS fraktal a spusti animaciu."""
        if app_state["anim_obj"] is not None:
            try:
                app_state["anim_obj"].event_source.stop()
            except Exception:
                pass

        _set_status("Generujem IFS fraktal...", color=ACCENT)
        fig.canvas.draw_idle()
        plt.pause(0.01)

        points = generate_ifs(model_id, n_iterations=50000)

        app_state["points"]   = points
        app_state["frame"]    = len(points) - 1
        app_state["total"]    = len(points)
        app_state["playing"]  = False
        app_state["color"]    = color
        app_state["model_id"] = model_id

        n = len(points)
        slider.valmin = 0
        slider.valmax = max(n - 1, 1)
        slider.set_val(n - 1)
        slider.ax.set_xlim(0, max(n - 1, 1))

        model_name = MODELS[model_id]["name"]
        _update_info(
            f"Model: {model_name}\n\n"
            f"Transformacie: 4\n"
            f"Pravdepodobnost: p = 0.25\n"
            f"Iteracie: {n}\n\n"
            f"Afinne transformacie:\n"
            f"  [x']   [a b c]   [x]   [j]\n"
            f"  [y'] = [d e f] * [y] + [k]\n"
            f"  [z']   [g h i]   [z]   [l]\n\n"
            f"Pouzite animacne ovladace\n"
            f"pre postupne vykreslovanie."
        )

        _set_status(
            f"Hotovo! {n} bodov vygenerovanych.",
            color=color,
        )
        _draw_fractal(n - 1)

        anim_obj = anim_mod.FuncAnimation(
            fig, _anim_step, interval=1,
            blit=False, cache_frame_data=False,
        )
        app_state["anim_obj"] = anim_obj
        fig._anim = anim_obj

    # ── Animacia ────────────────────────────────────────────────────────────────

    def _set_frame(fi: int) -> None:
        n = len(app_state["points"])
        if n == 0:
            return
        fi = int(np.clip(fi, 0, n - 1))
        app_state["frame"]       = fi
        app_state["_upd_slider"] = True
        slider.set_val(fi)
        app_state["_upd_slider"] = False
        _draw_fractal(fi)

        if fi == n - 1:
            _set_status(f"Kompletne vykreslenie: {n} bodov.",
                        color=app_state["color"])
        else:
            _set_status(f"Bod {fi + 1} / {n}")

    def _get_anim_step_size() -> int:
        n = app_state["total"]
        if n < 500:
            return 1
        elif n < 2000:
            return 10
        elif n < 10000:
            return 50
        else:
            return 200

    def _anim_step(_fn: int) -> list:
        if not app_state["playing"]:
            return []
        step = _get_anim_step_size()
        nxt = app_state["frame"] + step
        n = len(app_state["points"])
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
        _draw_fractal(fi)

    def on_play(event) -> None:
        if len(app_state["points"]) == 0:
            return
        if app_state["playing"]:
            app_state["playing"] = False
            btn_play.label.set_text("▶ Play")
        else:
            if app_state["frame"] >= len(app_state["points"]) - 1:
                app_state["frame"] = 0
            app_state["playing"] = True
            btn_play.label.set_text("⏸ Pause")

    def on_prev(event) -> None:
        if len(app_state["points"]) == 0:
            return
        app_state["playing"] = False
        btn_play.label.set_text("▶ Play")
        step = _get_anim_step_size()
        _set_frame(app_state["frame"] - step)

    def on_next(event) -> None:
        if len(app_state["points"]) == 0:
            return
        app_state["playing"] = False
        btn_play.label.set_text("▶ Play")
        step = _get_anim_step_size()
        _set_frame(app_state["frame"] + step)

    def on_restart(event) -> None:
        if len(app_state["points"]) == 0:
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

    def on_m1(event):
        _run_ifs(1, COLOR_M1)

    def on_m2(event):
        _run_ifs(2, COLOR_M2)

    btn_m1.on_clicked(on_m1)
    btn_m2.on_clicked(on_m2)

    # ── Obsluha clear ───────────────────────────────────────────────────────────

    def on_clear(event) -> None:
        if app_state["anim_obj"] is not None:
            try:
                app_state["anim_obj"].event_source.stop()
            except Exception:
                pass

        app_state["points"]   = np.empty((0, 3))
        app_state["frame"]    = 0
        app_state["total"]    = 0
        app_state["playing"]  = False
        app_state["anim_obj"] = None
        app_state["model_id"] = None
        btn_play.label.set_text("▶ Play")

        _update_info(
            "Zvolte model (1 alebo 2)\n"
            "pre generovanie IFS fraktalu.\n\n"
            "Algoritmus:\n"
            "  1. Zaciname v bode (0, 0, 0)\n"
            "  2. Nahodne vyberieme\n"
            "     jednu zo 4 transformacii\n"
            "     (p = 0.25 pre kazdu)\n"
            "  3. Aplikujeme afinnu\n"
            "     transformaciu na bod\n"
            "  4. Ulozime novy bod\n"
            "  5. Opakujeme 50 000x"
        )
        _set_status("Canvas vymazany.")
        _draw_fractal()

    btn_clear.on_clicked(on_clear)

    # ── Pociatocny stav ─────────────────────────────────────────────────────────
    _draw_fractal()
    plt.show()
