"""
visualization.py – Interaktivna vizualizacia Hopfieldovej siete.

Rozlozenie figury (15x9 palcov, tmava tema):
  Vlavo  (60 %) – klikatelna mriezka 10x10 zobrazujuca aktualny vzor
  Vpravo (35 %) – tlacidla na ovladanie (uloz, oprav, zobrazi, vymaz)
  Dolu         – animacne ovladacie prvky (slider, Play/Pause, Prev/Next, Restart)

Farebna schema zodpoveda Task 2:
  BG_DARK  = "#0d1117"
  BG_PANEL = "#161b22"
  BG_AXES  = "#1c2333"
  TEXT     = "#f0f6fc"
  ACCENT   = "#58a6ff"
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider, Button
from matplotlib.lines import Line2D

from .hopfield_network import HopfieldNetwork

# ── Farby temy (identicky s Task 2) ────────────────────────────────────────────
BG_DARK    = "#0d1117"
BG_PANEL   = "#161b22"
BG_AXES    = "#1c2333"
TEXT_COLOR = "#f0f6fc"
ACCENT     = "#58a6ff"

# Farby tlacidiel
COLOR_SAVE    = "#56d364"   # zelena  – Uloz vzor
COLOR_SYNC    = "#58a6ff"   # modra   – Oprav Sync
COLOR_ASYNC   = "#e3b341"   # zlta    – Oprav Async
COLOR_SHOW    = "#bc8cff"   # fialova – Zobraz ulozene
COLOR_CLEAR   = "#ff7b72"   # cervena – Vymaz mriezku

# Rozmery mriezky
GRID_ROWS = 10
GRID_COLS = 10
N_NEURONS = GRID_ROWS * GRID_COLS  # 100


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


def _pattern_to_grid(pattern: np.ndarray) -> np.ndarray:
    """Prevedie ploch vzor (100,) na mriezku (10, 10)."""
    return pattern.reshape(GRID_ROWS, GRID_COLS)


def _grid_to_pattern(grid: np.ndarray) -> np.ndarray:
    """Prevedie mriezku (10, 10) na ploch vzor (100,)."""
    return grid.flatten()


def _draw_grid_on_ax(
    ax,
    grid: np.ndarray,
    highlight_neuron: int | None = None,
    highlight_color: str = "#ffa657",
) -> None:
    """
    Vykreslí mriezku 10x10 na danú os.

    Parametre
    ---------
    ax               : matplotlib Axes
    grid             : np.ndarray tvaru (10, 10), hodnoty +1 / -1
    highlight_neuron : index neuronu (0-99) na zvyraznenie (pre async animaciu)
    highlight_color  : farba zvyraznenia
    """
    ax.clear()
    ax.set_facecolor(BG_AXES)
    ax.set_xlim(-0.5, GRID_COLS - 0.5)
    ax.set_ylim(-0.5, GRID_ROWS - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            neuron_idx = r * GRID_COLS + c
            val = grid[r, c]

            # Farba bunky
            if highlight_neuron is not None and neuron_idx == highlight_neuron:
                fc = highlight_color
            elif val >= 0:
                fc = ACCENT
            else:
                fc = BG_AXES

            rect = mpatches.FancyBboxPatch(
                (c - 0.45, r - 0.45), 0.9, 0.9,
                boxstyle="round,pad=0.05",
                linewidth=0.8,
                edgecolor=BG_PANEL,
                facecolor=fc,
            )
            ax.add_patch(rect)


def run_app() -> None:
    """
    Spusti hlavnu interaktivnu aplikaciu Hopfieldovej siete.
    Vytvori matplotlib figuru a nastavuje vsetky ovladacie prvky.
    """
    _apply_dark_theme()

    # ── Hopfieldova siet ────────────────────────────────────────────────────────
    net = HopfieldNetwork(N_NEURONS)

    # ── Stav aplikacie ──────────────────────────────────────────────────────────
    # Aktualna mriezka (+1 / -1)
    current_grid = np.full((GRID_ROWS, GRID_COLS), -1.0)

    # Animacny stav
    anim_state: dict = {
        "steps":            [],      # zoznam np.ndarray krokov
        "frame":            0,
        "playing":          False,
        "_updating_slider": False,
        "mode":             None,    # "sync" | "async" | None
        "converged":        False,
        "anim_obj":         None,    # FuncAnimation objekt
    }

    # ── Figura ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 9), facecolor=BG_DARK)
    fig.suptitle(
        "Task 3 – Hopfield Network  ·  Associative Memory",
        color=TEXT_COLOR, fontsize=14, fontweight="bold", y=0.97,
    )

    # Rozlozenie: [vlavo mriezka | vpravo ovladanie]
    # Tenka cara oddelujuca animacne ovladacie prvky
    plt.subplots_adjust(left=0.04, right=0.97, top=0.91, bottom=0.30, wspace=0.05)

    # Osi pre mriezku (vlavo ~60 %)
    ax_grid = fig.add_axes([0.04, 0.30, 0.54, 0.60])

    # Oddelovacia cara (ako v task2)
    fig.add_artist(
        Line2D([0.03, 0.97], [0.275, 0.275],
               transform=fig.transFigure,
               color=ACCENT, linewidth=0.8, alpha=0.5)
    )

    # ── Tlacidla (vpravo) ───────────────────────────────────────────────────────
    # Pozicie: [x, y, sirka, vyska] v suradniciach figury
    btn_save_ax  = fig.add_axes([0.62, 0.78, 0.32, 0.065])
    btn_sync_ax  = fig.add_axes([0.62, 0.70, 0.32, 0.065])
    btn_async_ax = fig.add_axes([0.62, 0.62, 0.32, 0.065])
    btn_show_ax  = fig.add_axes([0.62, 0.54, 0.32, 0.065])
    btn_clear_ax = fig.add_axes([0.62, 0.46, 0.32, 0.065])

    btn_save  = Button(btn_save_ax,  "Save Pattern",        color=BG_PANEL, hovercolor="#1f3a1f")
    btn_sync  = Button(btn_sync_ax,  "Repair Sync",         color=BG_PANEL, hovercolor="#1a2a3a")
    btn_async = Button(btn_async_ax, "Repair Async",        color=BG_PANEL, hovercolor="#3a2e10")
    btn_show  = Button(btn_show_ax,  "Show Saved Patterns", color=BG_PANEL, hovercolor="#2a1f3a")
    btn_clear = Button(btn_clear_ax, "Clear Grid",          color=BG_PANEL, hovercolor="#3a1515")

    # Farby popiskov tlacidiel
    btn_save.label.set_color(COLOR_SAVE)
    btn_sync.label.set_color(COLOR_SYNC)
    btn_async.label.set_color(COLOR_ASYNC)
    btn_show.label.set_color(COLOR_SHOW)
    btn_clear.label.set_color(COLOR_CLEAR)

    for btn in (btn_save, btn_sync, btn_async, btn_show, btn_clear):
        btn.label.set_fontsize(11)
        btn.label.set_fontweight("bold")

    # Informacny text (pocet vzorov, maximum)
    ax_info = fig.add_axes([0.62, 0.30, 0.32, 0.13])
    ax_info.set_facecolor(BG_PANEL)
    ax_info.set_xticks([])
    ax_info.set_yticks([])
    for spine in ax_info.spines.values():
        spine.set_edgecolor(ACCENT)
        spine.set_linewidth(1.2)

    info_text = ax_info.text(
        0.5, 0.5, "",
        transform=ax_info.transAxes,
        ha="center", va="center",
        fontsize=10, family="monospace",
        color=TEXT_COLOR,
    )

    # ── Animacne ovladacie prvky (dolu, rovnaky styl ako task2) ────────────────
    slider_ax   = fig.add_axes([0.10, 0.18, 0.80, 0.03], facecolor=BG_PANEL)
    btn_prev_ax = fig.add_axes([0.33, 0.09, 0.07, 0.05], facecolor=BG_PANEL)
    btn_play_ax = fig.add_axes([0.41, 0.09, 0.12, 0.05], facecolor=BG_PANEL)
    btn_next_ax = fig.add_axes([0.54, 0.09, 0.07, 0.05], facecolor=BG_PANEL)
    btn_rst_ax  = fig.add_axes([0.62, 0.09, 0.09, 0.05], facecolor=BG_PANEL)

    slider = Slider(
        slider_ax, "Step", 0, 1,
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

    # Stavovy riadok (pod mriezkou)
    ax_status = fig.add_axes([0.04, 0.26, 0.54, 0.035])
    ax_status.set_facecolor(BG_PANEL)
    ax_status.set_xticks([])
    ax_status.set_yticks([])
    for spine in ax_status.spines.values():
        spine.set_edgecolor(ACCENT)
        spine.set_linewidth(0.8)

    status_text = ax_status.text(
        0.01, 0.5, "Draw a pattern and click Save Pattern.",
        transform=ax_status.transAxes,
        ha="left", va="center",
        fontsize=9, family="monospace",
        color=TEXT_COLOR,
    )

    # Info box pre animaciu (krok / celkovo)
    anim_info = ax_grid.text(
        0.99, 0.01, "",
        transform=ax_grid.transAxes,
        ha="right", va="bottom",
        fontsize=9, family="monospace",
        color=TEXT_COLOR,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor=BG_PANEL,
            edgecolor=ACCENT,
            alpha=0.88,
        ),
    )

    # ── Pomocne funkcie ─────────────────────────────────────────────────────────

    def _update_info_panel() -> None:
        """Aktualizuje informacny panel s poctom vzorov."""
        n_stored = len(net.stored_patterns)
        n_max    = net.max_recommended
        warn = " !" if n_stored > n_max else "  "
        info_text.set_text(
            f"Max recommended:\n"
            f"  {n_max} patterns\n\n"
            f"Saved:{warn}{n_stored} patterns"
        )
        fig.canvas.draw_idle()

    def _set_status(msg: str, color: str = TEXT_COLOR) -> None:
        """Nastavi text stavoveho riadku."""
        status_text.set_text(msg)
        status_text.set_color(color)
        fig.canvas.draw_idle()

    def _redraw_grid(
        highlight_neuron: int | None = None,
        show_anim_info: bool = False,
        frame_idx: int = 0,
        total_frames: int = 1,
    ) -> None:
        """Prekreslí mriezku podla aktuálneho current_grid."""
        _draw_grid_on_ax(ax_grid, current_grid, highlight_neuron=highlight_neuron)
        ax_grid.set_title("Current Pattern", color=TEXT_COLOR, fontsize=12, pad=8)

        # Obnov anim_info text (po clear os sa znici)
        if show_anim_info and anim_state["steps"]:
            converged = anim_state["converged"] and (frame_idx >= total_frames - 1)
            conv_str = "  Converged!" if converged else ""
            ax_grid.text(
                0.99, 0.01,
                f"Step: {frame_idx} / {total_frames - 1}{conv_str}",
                transform=ax_grid.transAxes,
                ha="right", va="bottom",
                fontsize=9, family="monospace",
                color=TEXT_COLOR,
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor=BG_PANEL,
                    edgecolor=ACCENT,
                    alpha=0.88,
                ),
            )
        fig.canvas.draw_idle()

    def _start_animation(steps: list[np.ndarray], mode: str) -> None:
        """
        Nastavi animacny stav a spusti FuncAnimation.

        Parametre
        ---------
        steps : list[np.ndarray]  – predvypocitane kroky obnovy
        mode  : "sync" | "async"
        """
        nonlocal current_grid

        # Zastav predchadzajucu animaciu
        if anim_state["anim_obj"] is not None:
            try:
                anim_state["anim_obj"].event_source.stop()
            except Exception:
                pass

        anim_state["steps"]   = steps
        anim_state["frame"]   = 0
        anim_state["playing"] = False
        anim_state["mode"]    = mode
        anim_state["converged"] = _check_convergence(steps)

        n_frames = len(steps)

        # Nastav slider
        slider.ax.set_visible(True)
        slider.valmin = 0
        slider.valmax = max(n_frames - 1, 1)
        slider.set_val(0)
        slider.ax.set_xlim(0, max(n_frames - 1, 1))

        # Zobraz prvy krok
        current_grid = _pattern_to_grid(steps[0])
        _redraw_grid(show_anim_info=True, frame_idx=0, total_frames=n_frames)

        # Spusti animaciu
        anim_obj = animation.FuncAnimation(
            fig, _anim_step, interval=180,
            blit=False, cache_frame_data=False,
        )
        anim_state["anim_obj"] = anim_obj
        fig._anim = anim_obj  # udrzujeme referenciu

    def _check_convergence(steps: list[np.ndarray]) -> bool:
        """Skontroluje, ci posledne dva kroky su rovnake (konvergencia)."""
        if len(steps) < 2:
            return True
        return bool(np.array_equal(steps[-1], steps[-2]))

    def _draw_frame_at(fi: int) -> None:
        """Vykreslí stav siete pre krok fi."""
        nonlocal current_grid
        steps = anim_state["steps"]
        if not steps:
            return

        fi = int(np.clip(fi, 0, len(steps) - 1))
        current_grid = _pattern_to_grid(steps[fi])

        # Pre async rezim: zvyrazni posledny zmeneny neuron
        highlight = None
        if anim_state["mode"] == "async" and fi > 0:
            diff = np.where(steps[fi] != steps[fi - 1])[0]
            if len(diff) > 0:
                highlight = int(diff[0])

        _redraw_grid(
            highlight_neuron=highlight,
            show_anim_info=True,
            frame_idx=fi,
            total_frames=len(steps),
        )

        # Stav obnovy
        mode_str = "synchronously" if anim_state["mode"] == "sync" else "asynchronously"
        if anim_state["converged"] and fi == len(steps) - 1:
            _set_status(
                f"Converged after {fi} step(s) ({mode_str}).",
                color=COLOR_SAVE,
            )
        else:
            _set_status(
                f"Repairing {mode_str}... step {fi} / {len(steps) - 1}",
            )

    def _set_frame(fi: int) -> None:
        """Skoci na krok fi a vykreslí stav."""
        fi = int(np.clip(fi, 0, len(anim_state["steps"]) - 1))
        anim_state["frame"] = fi
        anim_state["_updating_slider"] = True
        slider.set_val(fi)
        anim_state["_updating_slider"] = False
        _draw_frame_at(fi)

    # ── Animacna funkcia (volana FuncAnimation) ─────────────────────────────────

    def _anim_step(_frame_num: int) -> list:
        """Jeden krok automatickeho prehravania."""
        if not anim_state["playing"]:
            return []
        steps = anim_state["steps"]
        if not steps:
            return []
        nxt = anim_state["frame"] + 1
        if nxt >= len(steps):
            anim_state["playing"] = False
            btn_play.label.set_text("▶ Play")
            return []
        _set_frame(nxt)
        return []

    # ── Obsluha animacnych tlacidiel ────────────────────────────────────────────

    def on_slider(val: float) -> None:
        if anim_state["_updating_slider"]:
            return
        fi = int(val)
        anim_state["frame"] = fi
        _draw_frame_at(fi)

    def on_play(event) -> None:
        if not anim_state["steps"]:
            return
        if anim_state["playing"]:
            anim_state["playing"] = False
            btn_play.label.set_text("▶ Play")
        else:
            if anim_state["frame"] >= len(anim_state["steps"]) - 1:
                anim_state["frame"] = 0
            anim_state["playing"] = True
            btn_play.label.set_text("⏸ Pause")

    def on_prev(event) -> None:
        if not anim_state["steps"]:
            return
        anim_state["playing"] = False
        btn_play.label.set_text("▶ Play")
        _set_frame(anim_state["frame"] - 1)

    def on_next(event) -> None:
        if not anim_state["steps"]:
            return
        anim_state["playing"] = False
        btn_play.label.set_text("▶ Play")
        _set_frame(anim_state["frame"] + 1)

    def on_restart(event) -> None:
        if not anim_state["steps"]:
            return
        anim_state["playing"] = False
        btn_play.label.set_text("▶ Play")
        _set_frame(0)

    slider.on_changed(on_slider)
    btn_play.on_clicked(on_play)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    btn_restart.on_clicked(on_restart)

    # ── Klik na mriezku (prepinanie buniek) ─────────────────────────────────────

    def on_grid_click(event) -> None:
        """Obsluha kliknutia na mriezku – prepne bunku +1 / -1."""
        nonlocal current_grid
        if event.inaxes != ax_grid:
            return
        # Suradnice v mriezke
        c = int(np.floor(event.xdata + 0.5))
        r = int(np.floor(event.ydata + 0.5))
        if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
            current_grid[r, c] *= -1  # prepni +1 <-> -1
            _redraw_grid()

    fig.canvas.mpl_connect("button_press_event", on_grid_click)

    # ── Obsluha tlacidla: Uloz vzor ─────────────────────────────────────────────

    def on_save(event) -> None:
        pattern = _grid_to_pattern(current_grid)
        net.train(pattern)
        n_stored = len(net.stored_patterns)
        n_max    = net.max_recommended
        _update_info_panel()

        if n_stored > n_max:
            _set_status(
                f"Pattern saved! ({n_stored}/{n_max} stored) – Warning: exceeds recommended capacity!",
                color=COLOR_ASYNC,
            )
        else:
            _set_status(
                f"Pattern saved! ({n_stored}/{n_max} stored)",
                color=COLOR_SAVE,
            )

    btn_save.on_clicked(on_save)

    # ── Obsluha tlacidla: Oprav Sync ────────────────────────────────────────────

    def on_repair_sync(event) -> None:
        if len(net.stored_patterns) == 0:
            _set_status("No patterns saved! Save at least one pattern first.", color=COLOR_CLEAR)
            return
        pattern = _grid_to_pattern(current_grid)
        _set_status("Computing synchronous recovery...", color=ACCENT)
        fig.canvas.draw_idle()
        steps = net.recover_sync(pattern, max_iter=50)
        _set_status(f"Repair Sync: {len(steps) - 1} step(s) precomputed. Use controls to play.")
        _start_animation(steps, mode="sync")

    btn_sync.on_clicked(on_repair_sync)

    # ── Obsluha tlacidla: Oprav Async ───────────────────────────────────────────

    def on_repair_async(event) -> None:
        if len(net.stored_patterns) == 0:
            _set_status("No patterns saved! Save at least one pattern first.", color=COLOR_CLEAR)
            return
        pattern = _grid_to_pattern(current_grid)
        _set_status("Computing asynchronous recovery...", color=ACCENT)
        fig.canvas.draw_idle()
        steps = net.recover_async(pattern, max_iter=50)
        _set_status(f"Repair Async: {len(steps) - 1} individual neuron update(s). Use controls.")
        _start_animation(steps, mode="async")

    btn_async.on_clicked(on_repair_async)

    # ── Obsluha tlacidla: Vymaz mriezku ─────────────────────────────────────────

    def on_clear(event) -> None:
        nonlocal current_grid
        current_grid = np.full((GRID_ROWS, GRID_COLS), -1.0)
        _redraw_grid()
        _set_status("Grid cleared.")

    btn_clear.on_clicked(on_clear)

    # ── Obsluha tlacidla: Zobraz ulozene vzory ───────────────────────────────────

    def on_show_patterns(event) -> None:
        """Otvorí novu figuru so vsetkymi ulozenymo vzormi."""
        patterns = net.stored_patterns
        if not patterns:
            _set_status("No patterns saved yet.", color=COLOR_CLEAR)
            return

        n = len(patterns)
        cols = min(n, 5)
        rows = (n + cols - 1) // cols

        fig2, axes = plt.subplots(
            rows, cols,
            figsize=(cols * 2.2 + 0.5, rows * 2.4 + 0.8),
            facecolor=BG_DARK,
        )
        fig2.suptitle(
            f"Saved Patterns ({n} total)",
            color=TEXT_COLOR, fontsize=13, fontweight="bold", y=0.98,
        )
        fig2.patch.set_facecolor(BG_DARK)

        # Zabezpec, ze axes je vzdy 2D zoznam
        if n == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        for idx in range(rows * cols):
            r, c = divmod(idx, cols)
            ax = axes[r, c]
            if idx < n:
                grid = _pattern_to_grid(patterns[idx])
                _draw_grid_on_ax(ax, grid)
                ax.set_title(f"Pattern {idx + 1}", color=TEXT_COLOR, fontsize=9, pad=4)
            else:
                # Prazdna os
                ax.set_facecolor(BG_DARK)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    btn_show.on_clicked(on_show_patterns)

    # ── Pociatocny stav ─────────────────────────────────────────────────────────
    _update_info_panel()
    _redraw_grid()

    plt.show()
