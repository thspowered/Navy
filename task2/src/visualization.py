"""
visualization.py – Interaktivna animovana vizualizacia trenovania XOR siete.

Dvojpanelova figura s tmavou temou (rovnaky dizajn ako Task 1):
  Vlavo  – animovana krivka strat (MSE) po epochach
  Vpravo – tepelna mapa rozhodovacich oblasti + XOR body

Interaktivne ovladanie:
  Slider      – preskocit na lubovolnu snimku trenovania
  Play/Pause  – automaticke prehravanie animacie
  < Prev / Next > – krok o jednu snimku
  Restart     – skok spat na snimku 0
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider, Button

# ── Farby temy (zhodne s Task 1) ───────────────────────────────────────────────
BG_DARK    = "#0d1117"
BG_PANEL   = "#161b22"
BG_AXES    = "#1c2333"
TEXT_COLOR = "#f0f6fc"
ACCENT     = "#58a6ff"

COLOR_ZERO = "#ff7b72"   # XOR = 0  (cervena)
COLOR_ONE  = "#56d364"   # XOR = 1  (zelena)

_C_HIGH = "#ff7b72"   # strata vysoka / presnost nizka
_C_MID  = "#e3b341"   # strata stredna
_C_LOW  = "#56d364"   # strata nizka / presnost vysoka

# Rozlisenie meshgridu pre tepelnu mapu
_HEATMAP_RES = 300


def _apply_dark_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG_DARK,
        "axes.facecolor":   BG_AXES,
        "axes.edgecolor":   TEXT_COLOR,
        "axes.labelcolor":  TEXT_COLOR,
        "xtick.color":      TEXT_COLOR,
        "ytick.color":      TEXT_COLOR,
        "text.color":       TEXT_COLOR,
        "grid.color":       "#2a2a4a",
        "grid.alpha":       0.5,
    })


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _net_eval(snap: dict, X: np.ndarray) -> np.ndarray:
    """Vyhodnoti siet zo snimky historii na vstupe X."""
    a_h = _sigmoid(X @ snap["w_hidden"] + snap["b_hidden"])
    return _sigmoid(a_h @ snap["w_output"] + snap["b_output"])


# ── Hlavny vstupny bod ─────────────────────────────────────────────────────────

def run_animation(
    model,
    X_xor: np.ndarray,
    y_xor: np.ndarray,
    interval_ms: int = 120,
) -> None:
    """
    Zobrazi interaktivnu animovanu vizualizaciu trenovania XOR siete.

    Parametre
    ----------
    model      : uz natrenovana instancia XORNet
    X_xor      : vstupne body XOR (tvar 4×2)
    y_xor      : ocakavane vystupy {0, 1}  (tvar 4,)
    interval_ms: cas medzi snimkami animacie v ms
    """
    _apply_dark_theme()

    history  = model.history
    n_frames = len(history)
    losses   = [s["loss"] for s in history]
    epochs   = [s["epoch"] for s in history]

    # ── Pre-vypocitaj tepelne mapy ─────────────────────────────────────────
    res  = _HEATMAP_RES
    lins = np.linspace(-0.2, 1.2, res)
    xx, yy = np.meshgrid(lins, lins)
    grid   = np.column_stack([xx.ravel(), yy.ravel()])

    heatmaps: list[np.ndarray] = [
        _net_eval(snap, grid).reshape(res, res)
        for snap in history
    ]

    # ── Rozlozenie figury ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8), facecolor=BG_DARK)
    fig.suptitle(
        "Task 2 – XOR Neural Network  ·  2→2→1  Sigmoid",
        color=TEXT_COLOR, fontsize=14, fontweight="bold", y=0.97,
    )

    plt.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.30, wspace=0.32)

    ax_loss = fig.add_subplot(1, 2, 1)
    ax_heat = fig.add_subplot(1, 2, 2)

    # Tenky oddelovac
    fig.add_artist(
        Line2D([0.05, 0.95], [0.275, 0.275],
               transform=fig.transFigure,
               color=ACCENT, linewidth=0.8, alpha=0.5)
    )

    # ── Stylovanie osi ──────────────────────────────────────────────────────
    for ax in (ax_loss, ax_heat):
        ax.set_facecolor(BG_AXES)
        ax.grid(True, alpha=0.3)

    ax_loss.set_title("Training Loss (MSE)",   color=TEXT_COLOR, pad=8)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss (MSE)")
    ax_loss.set_xlim(0, epochs[-1])
    ax_loss.set_ylim(0, max(losses) * 1.12)

    ax_heat.set_title("Decision Boundary",     color=TEXT_COLOR, pad=8)
    ax_heat.set_xlabel("x")
    ax_heat.set_ylabel("y")
    ax_heat.set_xlim(-0.2, 1.2)
    ax_heat.set_ylim(-0.2, 1.2)

    # ── Lavy panel: krivka strat ────────────────────────────────────────────
    loss_line, = ax_loss.plot([], [], color=ACCENT, linewidth=2, zorder=3)
    loss_dot,  = ax_loss.plot([], [], "o", color=ACCENT, markersize=8, zorder=5)

    info_loss = ax_loss.text(
        0.98, 0.97, "",
        transform=ax_loss.transAxes,
        ha="right", va="top", fontsize=10,
        family="monospace", color=TEXT_COLOR,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_PANEL,
                  edgecolor=ACCENT, alpha=0.85),
    )

    # Legenda pre krivku strat
    ax_loss.legend(
        [loss_line], ["MSE Loss"],
        loc="upper right", fontsize=8,
        facecolor=BG_PANEL, labelcolor=TEXT_COLOR, edgecolor="none",
    )

    # ── Pravy panel: tepelna mapa ───────────────────────────────────────────
    im = ax_heat.imshow(
        heatmaps[0],
        extent=[-0.2, 1.2, -0.2, 1.2],
        origin="lower",
        cmap="RdYlGn",
        vmin=0, vmax=1,
        alpha=0.80,
        aspect="auto",
        zorder=1,
    )

    # Farebna stupnica
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label("Network output", color=TEXT_COLOR, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    cbar.outline.set_edgecolor(TEXT_COLOR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)

    # XOR body – farba podla spravnosti predikcie
    xor_colors_initial = [COLOR_ZERO if int(lbl) == 0 else COLOR_ONE for lbl in y_xor]
    scatter = ax_heat.scatter(
        X_xor[:, 0], X_xor[:, 1],
        c=xor_colors_initial, s=220, zorder=5,
        edgecolors=TEXT_COLOR, linewidths=1.8,
    )

    # Popisky XOR bodov
    label_texts = []
    for xi, yi_coord, lbl in zip(X_xor[:, 0], X_xor[:, 1], y_xor):
        t = ax_heat.text(
            xi + 0.04, yi_coord + 0.04,
            f"XOR={int(lbl)}",
            color=TEXT_COLOR, fontsize=8, fontweight="bold",
            zorder=6,
        )
        label_texts.append(t)

    # Kontura rozhodovacieho prahu 0.5
    contour_store: list = [None]

    info_heat = ax_heat.text(
        0.98, 0.97, "",
        transform=ax_heat.transAxes,
        ha="right", va="top", fontsize=10,
        family="monospace", color=TEXT_COLOR,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_PANEL,
                  edgecolor=ACCENT, alpha=0.85),
    )

    # Legenda pre body
    legend_handles = [
        plt.scatter([], [], c=COLOR_ONE,  s=80, edgecolors=TEXT_COLOR, lw=1.2, label="XOR = 1"),
        plt.scatter([], [], c=COLOR_ZERO, s=80, edgecolors=TEXT_COLOR, lw=1.2, label="XOR = 0"),
    ]
    ax_heat.legend(
        handles=legend_handles, loc="lower right", fontsize=8,
        facecolor=BG_PANEL, labelcolor=TEXT_COLOR, edgecolor="none",
    )

    # ── Stavovy slovnik ─────────────────────────────────────────────────────
    state: dict = {
        "frame":              0,
        "playing":            False,
        "_updating_slider":   False,
    }

    # ── Aktualizacna funkcia ────────────────────────────────────────────────

    def _draw_frame(fi: int) -> None:
        snap  = history[fi]
        loss  = snap["loss"]
        epoch = snap["epoch"]
        preds = snap["predictions"]
        binary = (preds >= 0.5).astype(int)
        correct = int(np.sum(binary == y_xor.astype(int)))

        # — Lavy panel: krivka strat —
        loss_line.set_data(epochs[: fi + 1], losses[: fi + 1])
        loss_dot.set_data([epoch], [loss])

        loss_clr = _C_LOW if loss < 0.05 else (_C_MID if loss < 0.15 else _C_HIGH)
        info_loss.set_text(
            f"Epoch  : {epoch:>6}\n"
            f"Loss   : {loss:.5f}"
        )
        info_loss.get_bbox_patch().set_edgecolor(loss_clr)

        # — Pravy panel: tepelna mapa —
        im.set_data(heatmaps[fi])

        # Odstran staru konturu
        if contour_store[0] is not None:
            try:
                for coll in contour_store[0].collections:
                    coll.remove()
            except Exception:
                pass
            contour_store[0] = None

        # Nakresli novu konturu pri 0.5
        try:
            cs = ax_heat.contour(
                lins, lins, heatmaps[fi],
                levels=[0.5],
                colors=[TEXT_COLOR],
                linewidths=[2.0],
                alpha=0.9,
                zorder=4,
            )
            contour_store[0] = cs
        except Exception:
            pass

        # Farba bodov: zelena = spravne, cervena = zle
        dot_colors = [
            COLOR_ONE if binary[i] == int(y_xor[i]) else COLOR_ZERO
            for i in range(len(y_xor))
        ]
        scatter.set_facecolors(dot_colors)

        # Info box pre pravy panel
        acc_clr = _C_LOW if correct == 4 else (_C_MID if correct >= 3 else _C_HIGH)
        info_heat.set_text(
            f"Correct: {correct}/4\n"
            f"Acc    : {correct / 4:.0%}"
        )
        info_heat.get_bbox_patch().set_edgecolor(acc_clr)

        fig.canvas.draw_idle()

    # ── Ovladacie prvky ─────────────────────────────────────────────────────
    slider_ax   = fig.add_axes([0.10, 0.18, 0.80, 0.03], facecolor=BG_PANEL)
    btn_prev_ax = fig.add_axes([0.35, 0.09, 0.07, 0.05], facecolor=BG_PANEL)
    btn_play_ax = fig.add_axes([0.43, 0.09, 0.10, 0.05], facecolor=BG_PANEL)
    btn_next_ax = fig.add_axes([0.54, 0.09, 0.07, 0.05], facecolor=BG_PANEL)
    btn_rst_ax  = fig.add_axes([0.62, 0.09, 0.09, 0.05], facecolor=BG_PANEL)

    slider = Slider(
        slider_ax, "Epoch", 0, n_frames - 1,
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

    # ── Interne pomocne funkcie ─────────────────────────────────────────────

    def _set_frame(fi: int) -> None:
        fi = int(np.clip(fi, 0, n_frames - 1))
        state["frame"] = fi
        state["_updating_slider"] = True
        slider.set_val(fi)
        state["_updating_slider"] = False
        _draw_frame(fi)

    def _anim_step(_frame: int) -> list:
        if not state["playing"]:
            return []
        nxt = state["frame"] + 1
        if nxt >= n_frames:
            state["playing"] = False
            btn_play.label.set_text("▶ Play")
            return []
        _set_frame(nxt)
        return []

    def on_slider(val: float) -> None:
        if state["_updating_slider"]:
            return
        fi = int(val)
        state["frame"] = fi
        _draw_frame(fi)

    def on_play(event) -> None:
        if state["playing"]:
            state["playing"] = False
            btn_play.label.set_text("▶ Play")
        else:
            if state["frame"] >= n_frames - 1:
                state["frame"] = 0
            state["playing"] = True
            btn_play.label.set_text("⏸ Pause")

    def on_prev(event) -> None:
        state["playing"] = False
        btn_play.label.set_text("▶ Play")
        _set_frame(state["frame"] - 1)

    def on_next(event) -> None:
        state["playing"] = False
        btn_play.label.set_text("▶ Play")
        _set_frame(state["frame"] + 1)

    def on_restart(event) -> None:
        state["playing"] = False
        btn_play.label.set_text("▶ Play")
        _set_frame(0)

    slider.on_changed(on_slider)
    btn_play.on_clicked(on_play)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    btn_restart.on_clicked(on_restart)

    # Pociatocna snimka
    _draw_frame(0)

    anim = animation.FuncAnimation(
        fig, _anim_step, interval=interval_ms, blit=False, cache_frame_data=False
    )
    fig._anim = anim  # type: ignore[attr-defined]

    plt.show()
