"""
visualization.py - Interaktivna vizualizacia Task 12: Forest fire CA.

Rozlozenie figury (15x9 palcov, tmava tema):
  Vlavo  (~57 %) – mriezka lesa (imshow), animuje sa nekonecne
  Vpravo (~35 %) – tlacidla presetov + info panel
  Dolu           – animacne ovladacie prvky (speed slider, Play/Pause, Step, Restart)

Farebna schema (identicka s predchadzajucimi taskmi):
  BG_DARK  = "#0d1117"
  BG_PANEL = "#161b22"
  BG_AXES  = "#1c2333"
  TEXT     = "#f0f6fc"
  ACCENT   = "#58a6ff"

Presety:
  1. "Standardny les"  – p=0.05, f=0.001, density=0.5, von Neumann
  2. "Husty les"       – p=0.10, f=0.001, density=0.65, von Neumann
  3. "Moore okolie"    – p=0.05, f=0.001, density=0.5,  Moore (8 susedov)
  4. "Clear"           – vymazat plochu

Animacia: matplotlib.animation.FuncAnimation bezi nekonecne. Slider riadi
rychlost (FPS, 1..60). Tlacidlo "Step" posunie simulaciu o jeden krok aj
v pauznutom stave. "Restart" znovu nainicializuje mriezku rovnakym presetom.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim_mod
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.widgets import Button, Slider
from matplotlib.lines import Line2D

from .forest_fire import (
    EMPTY, TREE, BURNING, BURNT,
    DEFAULT_P, DEFAULT_F, DEFAULT_DENSITY,
    init_grid, step, grid_stats,
)

# ── Farby temy (identicky s predchadzajucimi taskmi) ───────────────────────────
BG_DARK    = "#0d1117"
BG_PANEL   = "#161b22"
BG_AXES    = "#1c2333"
TEXT_COLOR = "#f0f6fc"
ACCENT     = "#58a6ff"

# Farby tlacidiel modov
COLOR_STANDARD = "#56d364"   # zelena – standardny les
COLOR_DENSE    = "#bc8cff"   # fialova – husty les
COLOR_MOORE    = "#f0b429"   # zlta   – Moore okolie
COLOR_CLEAR    = "#8b949e"   # seda   – clear

# Farby buniek mriezky (presne ladene podla obrazkov v PDF)
COLOR_EMPTY    = "#8b6914"   # hneda – prazdne miesto / suchá zem
COLOR_TREE     = "#2d8a2d"   # zelena – zivy strom
COLOR_BURNING  = "#ff8c1a"   # oranzova – horiaci strom
COLOR_BURNT    = "#1a1a1a"   # cierna – vyhoreny strom

# ── Parametre simulacie (default) ──────────────────────────────────────────────
GRID_SIZE     = 120     # rozmer mriezky (NxN)
DEFAULT_FPS   = 12      # default rychlost animacie (snimky/s)
MIN_FPS       = 1
MAX_FPS       = 60


# ── Predefinovane konfiguracie simulacie ──────────────────────────────────────
PRESETS = {
    "standard": {
        "name":          "Standardny les",
        "p":             DEFAULT_P,
        "f":             DEFAULT_F,
        "density":       DEFAULT_DENSITY,
        "neighborhood":  "von_neumann",
    },
    "dense": {
        "name":          "Husty les",
        "p":             0.10,
        "f":             0.001,
        "density":       0.65,
        "neighborhood":  "von_neumann",
    },
    "moore": {
        "name":          "Moore okolie",
        "p":             DEFAULT_P,
        "f":             DEFAULT_F,
        "density":       DEFAULT_DENSITY,
        "neighborhood":  "moore",
    },
}


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


def _build_cmap() -> tuple[ListedColormap, BoundaryNorm]:
    """Vytvori cmap a normalizaciu pre 4 stavy bunky (0..3)."""
    cmap = ListedColormap([COLOR_EMPTY, COLOR_TREE, COLOR_BURNING, COLOR_BURNT])
    # Hranice tak aby kazda celociselna hodnota dostala spravnu farbu
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    return cmap, norm


def run_app() -> None:
    """Spusti hlavnu interaktivnu aplikaciu Task 12 – Forest fire CA."""
    _apply_dark_theme()
    cmap, norm = _build_cmap()

    # ── Stav aplikacie ─────────────────────────────────────────────────────────
    app_state: dict = {
        "preset_id":   None,    # "standard" / "dense" / "moore" / None
        "preset":      None,    # aktualny preset dict
        "grid":        None,    # aktualna mriezka
        "iteration":   0,
        "playing":     False,
        "fps":         DEFAULT_FPS,
        "anim_obj":    None,
        "rng":         np.random.default_rng(),
        "color":       ACCENT,
        "img":         None,    # AxesImage (imshow handle)
    }

    # ── Figura ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 9), facecolor=BG_DARK)
    fig.suptitle(
        "Task 12 – Cellular automata  ·  Forest fire algorithm",
        color=TEXT_COLOR, fontsize=14, fontweight="bold", y=0.97,
    )

    # Oddelovacia cara nad animacnymi ovladacmi
    fig.add_artist(
        Line2D([0.03, 0.97], [0.275, 0.275],
               transform=fig.transFigure,
               color=ACCENT, linewidth=0.8, alpha=0.5)
    )

    # ── Hlavna os pre mriezku ──────────────────────────────────────────────────
    ax_grid = fig.add_axes([0.04, 0.31, 0.55, 0.61])

    def _style_grid_axes(title: str) -> None:
        ax_grid.set_facecolor(BG_AXES)
        ax_grid.tick_params(colors=TEXT_COLOR, labelsize=9)
        for spine in ax_grid.spines.values():
            spine.set_edgecolor(BG_PANEL)
        ax_grid.set_title(title, color=TEXT_COLOR, fontsize=11, pad=6)
        ax_grid.set_xlabel("x", color=TEXT_COLOR, fontsize=10)
        ax_grid.set_ylabel("y", color=TEXT_COLOR, fontsize=10)

    _style_grid_axes("Forest Fire")

    # Inicialna prazdna mriezka aby imshow malo co vykreslit
    blank = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    img = ax_grid.imshow(
        blank, cmap=cmap, norm=norm,
        interpolation="nearest", origin="lower",
        extent=(0, GRID_SIZE, 0, GRID_SIZE),
    )
    app_state["img"] = img

    # ── Tlacidla presetov (vpravo) ─────────────────────────────────────────────
    btn_st_ax = fig.add_axes([0.63, 0.86, 0.33, 0.055])
    btn_dn_ax = fig.add_axes([0.63, 0.80, 0.33, 0.055])
    btn_mr_ax = fig.add_axes([0.63, 0.74, 0.33, 0.055])
    btn_cl_ax = fig.add_axes([0.63, 0.68, 0.33, 0.055])

    btn_standard = Button(btn_st_ax, "Standardny les",
                          color=BG_PANEL, hovercolor="#1f3a1f")
    btn_dense    = Button(btn_dn_ax, "Husty les",
                          color=BG_PANEL, hovercolor="#2a1f3a")
    btn_moore    = Button(btn_mr_ax, "Moore okolie",
                          color=BG_PANEL, hovercolor="#3a2f1f")
    btn_clear    = Button(btn_cl_ax, "Clear canvas",
                          color=BG_PANEL, hovercolor="#2a2a2a")

    btn_standard.label.set_color(COLOR_STANDARD)
    btn_dense.label.set_color(COLOR_DENSE)
    btn_moore.label.set_color(COLOR_MOORE)
    btn_clear.label.set_color(COLOR_CLEAR)
    for btn in (btn_standard, btn_dense, btn_moore, btn_clear):
        btn.label.set_fontsize(10)
        btn.label.set_fontweight("bold")

    # ── Info panel (vpravo dolu) ───────────────────────────────────────────────
    ax_info = fig.add_axes([0.63, 0.31, 0.33, 0.36])
    ax_info.set_facecolor(BG_PANEL)
    ax_info.set_xticks([])
    ax_info.set_yticks([])
    for spine in ax_info.spines.values():
        spine.set_edgecolor(ACCENT)
        spine.set_linewidth(1.2)

    info_default = (
        "Cellular automaton:\n"
        "  Forest Fire (Drossel-Schwabl)\n\n"
        "Stavy bunky:\n"
        "  empty   (hneda)\n"
        "  tree    (zelena)\n"
        "  burning (oranzova)\n"
        "  burnt   (cierna)\n\n"
        "Pravidla:\n"
        "  1. empty/burnt -> tree\n"
        "     s pravdep. p\n"
        "  2. tree so susedom v plameni\n"
        "     -> burning\n"
        "  3. tree bez horiaceho suseda\n"
        "     -> burning s pravdep. f\n"
        "  4. burning -> burnt\n\n"
        "Default parametre:\n"
        f"  p = {DEFAULT_P}\n"
        f"  f = {DEFAULT_F}\n"
        f"  density = {DEFAULT_DENSITY}\n\n"
        "Vyber preset a stlac Play."
    )
    info_text = ax_info.text(
        0.5, 0.5, info_default,
        transform=ax_info.transAxes,
        ha="center", va="center",
        fontsize=8, family="monospace",
        color=TEXT_COLOR,
    )

    # ── Animacne ovladacie prvky (dolu) ─────────────────────────────────────────
    slider_ax    = fig.add_axes([0.10, 0.18, 0.80, 0.03], facecolor=BG_PANEL)
    btn_step_ax  = fig.add_axes([0.33, 0.09, 0.07, 0.05], facecolor=BG_PANEL)
    btn_play_ax  = fig.add_axes([0.41, 0.09, 0.12, 0.05], facecolor=BG_PANEL)
    btn_rst_ax   = fig.add_axes([0.54, 0.09, 0.09, 0.05], facecolor=BG_PANEL)

    slider = Slider(
        slider_ax, "Speed (FPS)", MIN_FPS, MAX_FPS,
        valinit=DEFAULT_FPS, valstep=1, color=ACCENT,
    )
    slider.label.set_color(TEXT_COLOR)
    slider.valtext.set_color(TEXT_COLOR)

    btn_play    = Button(btn_play_ax, "▶ Play",  color=BG_PANEL, hovercolor="#2a2a4a")
    btn_step    = Button(btn_step_ax, "Step",    color=BG_PANEL, hovercolor="#2a2a4a")
    btn_restart = Button(btn_rst_ax,  "Restart", color=BG_PANEL, hovercolor="#2a2a4a")
    for btn in (btn_play, btn_step, btn_restart):
        btn.label.set_color(TEXT_COLOR)

    # Stavovy riadok
    ax_status = fig.add_axes([0.04, 0.265, 0.55, 0.032])
    ax_status.set_facecolor(BG_PANEL)
    ax_status.set_xticks([])
    ax_status.set_yticks([])
    for spine in ax_status.spines.values():
        spine.set_edgecolor(ACCENT)
        spine.set_linewidth(0.8)
    status_text = ax_status.text(
        0.01, 0.5,
        "Vyber preset (Standardny / Husty / Moore) a stlac Play.",
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

    def _format_info_for_preset(preset: dict) -> str:
        stats = grid_stats(app_state["grid"]) if app_state["grid"] is not None else None
        nbh   = "Moore (8 susedov)" if preset["neighborhood"] == "moore" else "von Neumann (4)"
        info  = (
            f"Preset: {preset['name']}\n\n"
            f"Mriezka: {GRID_SIZE} x {GRID_SIZE}\n"
            f"Okolie:  {nbh}\n"
            f"  p (rast stromu)   = {preset['p']}\n"
            f"  f (blesk)         = {preset['f']}\n"
            f"  density (init)    = {preset['density']}\n\n"
        )
        if stats is not None:
            total = stats["total"]
            info += (
                f"Iteracia: {app_state['iteration']}\n\n"
                f"Stav mriezky:\n"
                f"  empty   {stats['empty']:>6}  "
                f"({100*stats['empty']/total:5.1f} %)\n"
                f"  tree    {stats['tree']:>6}  "
                f"({100*stats['tree']/total:5.1f} %)\n"
                f"  burning {stats['burning']:>6}  "
                f"({100*stats['burning']/total:5.1f} %)\n"
                f"  burnt   {stats['burnt']:>6}  "
                f"({100*stats['burnt']/total:5.1f} %)"
            )
        return info

    def _render() -> None:
        """Prekresli mriezku (predpoklada ze app_state['grid'] je platny)."""
        grid = app_state["grid"]
        if grid is None:
            return
        app_state["img"].set_data(grid)
        title = "Forest Fire"
        if app_state["preset"] is not None:
            title = (f"Forest Fire  –  {app_state['preset']['name']}  ·  "
                     f"iter {app_state['iteration']}")
        ax_grid.set_title(title, color=TEXT_COLOR, fontsize=11, pad=6)
        if app_state["preset"] is not None:
            _update_info(_format_info_for_preset(app_state["preset"]))
        fig.canvas.draw_idle()

    def _stop_anim() -> None:
        if app_state["anim_obj"] is not None:
            try:
                app_state["anim_obj"].event_source.stop()
            except Exception:
                pass
        app_state["anim_obj"] = None

    def _restart_anim_with_fps(fps: int) -> None:
        """Znovu naplanuje animaciu s danym FPS (interval = 1000/fps)."""
        _stop_anim()
        interval = max(1, int(1000 / fps))
        anim_obj = anim_mod.FuncAnimation(
            fig, _anim_step, interval=interval,
            blit=False, cache_frame_data=False,
        )
        app_state["anim_obj"] = anim_obj
        # drz referenciu aby ju GC nezabil
        fig._anim = anim_obj

    # ── Spustenie presetu ───────────────────────────────────────────────────────

    def _start_preset(preset_id: str, color: str) -> None:
        preset = PRESETS[preset_id]

        _set_status("Inicializujem mriezku...", color=ACCENT)
        plt.pause(0.01)

        seed = int(np.random.default_rng().integers(0, 2**31))
        app_state["preset_id"] = preset_id
        app_state["preset"]    = preset
        app_state["grid"]      = init_grid(GRID_SIZE, preset["density"], seed=seed)
        app_state["iteration"] = 0
        app_state["rng"]       = np.random.default_rng(seed + 1)
        app_state["color"]     = color
        app_state["playing"]   = True
        btn_play.label.set_text("⏸ Pause")

        _render()
        _set_status(
            f"Bezi: {preset['name']}  (FPS = {app_state['fps']}). "
            "Stlac Pause na zastavenie.",
            color=color,
        )
        _restart_anim_with_fps(app_state["fps"])

    # ── Animacny krok ───────────────────────────────────────────────────────────

    def _do_step() -> None:
        """Vykona jeden krok CA a prekresli."""
        if app_state["grid"] is None or app_state["preset"] is None:
            return
        preset = app_state["preset"]
        app_state["grid"] = step(
            app_state["grid"],
            p=preset["p"], f=preset["f"],
            neighborhood=preset["neighborhood"],
            rng=app_state["rng"],
        )
        app_state["iteration"] += 1
        _render()

    def _anim_step(_fn: int) -> list:
        if not app_state["playing"]:
            return []
        _do_step()
        return []

    # ── Obsluha animacnych tlacidiel ────────────────────────────────────────────

    def on_slider(val: float) -> None:
        fps = int(val)
        app_state["fps"] = fps
        # Reschedule animaciu len ak prave bezi (inak ju spustime az s Play)
        if app_state["playing"] and app_state["anim_obj"] is not None:
            _restart_anim_with_fps(fps)
        if app_state["preset"] is not None:
            color = app_state["color"]
            running = "Bezi" if app_state["playing"] else "Pauznute"
            _set_status(
                f"{running}: {app_state['preset']['name']}  (FPS = {fps}).",
                color=color,
            )

    def on_play(event) -> None:
        if app_state["grid"] is None:
            _set_status("Najprv vyber preset (Standardny / Husty / Moore).",
                        color=COLOR_DENSE)
            return
        if app_state["playing"]:
            app_state["playing"] = False
            btn_play.label.set_text("▶ Play")
            _stop_anim()
            _set_status(
                f"Pauznute: {app_state['preset']['name']}  "
                f"(iter {app_state['iteration']}).",
                color=app_state["color"],
            )
        else:
            app_state["playing"] = True
            btn_play.label.set_text("⏸ Pause")
            _restart_anim_with_fps(app_state["fps"])
            _set_status(
                f"Bezi: {app_state['preset']['name']}  "
                f"(FPS = {app_state['fps']}).",
                color=app_state["color"],
            )

    def on_step(event) -> None:
        if app_state["grid"] is None:
            _set_status("Najprv vyber preset.", color=COLOR_DENSE)
            return
        # Pri Step animaciu pauzneme (aby nas nepretiahla) a posunieme o 1
        if app_state["playing"]:
            app_state["playing"] = False
            btn_play.label.set_text("▶ Play")
            _stop_anim()
        _do_step()
        _set_status(
            f"Krok: {app_state['preset']['name']}  "
            f"(iter {app_state['iteration']}).",
            color=app_state["color"],
        )

    def on_restart(event) -> None:
        if app_state["preset_id"] is None:
            _set_status("Najprv vyber preset.", color=COLOR_DENSE)
            return
        _start_preset(app_state["preset_id"], app_state["color"])

    slider.on_changed(on_slider)
    btn_play.on_clicked(on_play)
    btn_step.on_clicked(on_step)
    btn_restart.on_clicked(on_restart)

    # ── Obsluha presetov ────────────────────────────────────────────────────────

    def on_standard(event):
        _start_preset("standard", COLOR_STANDARD)

    def on_dense(event):
        _start_preset("dense", COLOR_DENSE)

    def on_moore(event):
        _start_preset("moore", COLOR_MOORE)

    def on_clear(event) -> None:
        _stop_anim()
        app_state["preset_id"] = None
        app_state["preset"]    = None
        app_state["grid"]      = None
        app_state["iteration"] = 0
        app_state["playing"]   = False
        btn_play.label.set_text("▶ Play")

        # Reset zobrazenia
        app_state["img"].set_data(np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8))
        ax_grid.set_title("Forest Fire", color=TEXT_COLOR, fontsize=11, pad=6)
        _update_info(info_default)
        _set_status("Canvas vymazany.")
        fig.canvas.draw_idle()

    btn_standard.on_clicked(on_standard)
    btn_dense.on_clicked(on_dense)
    btn_moore.on_clicked(on_moore)
    btn_clear.on_clicked(on_clear)

    # ── Pociatocny stav ─────────────────────────────────────────────────────────
    plt.show()
