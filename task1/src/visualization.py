"""
visualization.py – Interaktivna animovana vizualizacia trenovania pre ulohu Perceptron.

Dvojpanelova figura s tmavou temou:
  Vlavo  – animovane trenovanie (rozhodovacia hranica sa vyvija po epochach)
  Vpravo – finalny vysledok (predikcie vs. skutocne navestia)

Interaktivne ovladanie:
  Slider      – preskocit na lubovolnu epochu
  Play/Pause  – automaticke prehravanie animacie
  < Prev / Next > – krok o jednu epochu
  Restart     – skok spat na epochu 0
  TextBox (a, b) + Retrain – zmena deliace priamky a opatovne trenovanie
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider, Button, TextBox

from src.data_generator import generate_points, to_binary

# ── Farby temy ─────────────────────────────────────────────────────────────────
BG_DARK    = "#0d1117"   # GitHub-dark temer cierna
BG_PANEL   = "#161b22"
BG_AXES    = "#1c2333"
TEXT_COLOR = "#f0f6fc"
ACCENT     = "#58a6ff"   # GitHub modra

COLOR_ABOVE = "#ff7b72"  # jemna cervena  – skutocne navestie +1
COLOR_BELOW = "#79c0ff"  # jemna modra    – skutocne navestie -1
COLOR_ON    = "#56d364"  # jemna zelena   – skutocne navestie 0 (na priamke)
COLOR_BOUND = "#e3b341"  # jantarova      – rozhodovacia hranica
COLOR_TRUE  = "#f0f6fc"  # temer biela    – skutocna deliaca priamka

_ACC_HIGH   = "#56d364"  # zelena  (> 95 %)
_ACC_MID    = "#e3b341"  # oranzova (> 80 %)
_ACC_LOW    = "#ff7b72"  # cervena  (<= 80 %)


def _apply_dark_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG_DARK,
            "axes.facecolor":   BG_AXES,
            "axes.edgecolor":   TEXT_COLOR,
            "axes.labelcolor":  TEXT_COLOR,
            "xtick.color":      TEXT_COLOR,
            "ytick.color":      TEXT_COLOR,
            "text.color":       TEXT_COLOR,
            "grid.color":       "#2a2a4a",
            "grid.alpha":       0.5,
        }
    )


def _scatter_by_label(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    labels: np.ndarray,
    size: float = 40,
) -> None:
    """Bodovy graf vyfarbeny podla SKUTOCNEHO ternarneho navestia (+1 / 0 / -1)."""
    mask_above = labels == 1
    mask_below = labels == -1
    mask_on    = labels == 0

    if mask_above.any():
        ax.scatter(xs[mask_above], ys[mask_above], c=COLOR_ABOVE,
                   s=size, label="Above (+1)", zorder=3, alpha=0.85)
    if mask_below.any():
        ax.scatter(xs[mask_below], ys[mask_below], c=COLOR_BELOW,
                   s=size, label="Below (−1)", zorder=3, alpha=0.85)
    if mask_on.any():
        ax.scatter(xs[mask_on], ys[mask_on], c=COLOR_ON,
                   s=size * 1.4, marker="*", label="On line (0)", zorder=4, alpha=0.95)


def _scatter_by_pred(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    preds: np.ndarray,
    size: float = 40,
) -> None:
    """Bodovy graf vyfarbeny podla PREDIKCIE PERCEPTRONU (+1 / -1)."""
    mask_pos = preds == 1
    mask_neg = preds == -1

    if mask_pos.any():
        ax.scatter(xs[mask_pos], ys[mask_pos], c=COLOR_ABOVE,
                   s=size, label="Pred +1", zorder=3, alpha=0.85)
    if mask_neg.any():
        ax.scatter(xs[mask_neg], ys[mask_neg], c=COLOR_BELOW,
                   s=size, label="Pred −1", zorder=3, alpha=0.85)


def _populate_axes(
    ax_train: plt.Axes,
    ax_final: plt.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    true_labels: np.ndarray,
    binary_labels: np.ndarray,
    perceptron,
    x_line: np.ndarray,
    slope: float,
    intercept: float,
    state: dict,
) -> None:
    """
    Vycisti a znovu naplni obe osi bodovym grafom, skutocnou deliacou priamkou,
    finalne naucenou hranicou a informacnymi boxmi.

    Aktualizuje *state* na mieste:
      - state["boundary_line"]  – animovany Line2D artist v ax_train
      - state["info_text"]      – Text artist v ax_train
      - state["boundary_ys"]    – predpocitane y-hodnoty hranice pre kazdu epochu
    """
    ax_train.cla()
    ax_final.cla()

    # Znovu nastav stylovanie (cla() resetuje facecolor / grid / popisky)
    for ax in (ax_train, ax_final):
        ax.set_facecolor(BG_AXES)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    ax_train.set_title("Training Animation", color=TEXT_COLOR, pad=8)
    ax_final.set_title("Final Result",        color=TEXT_COLOR, pad=8)

    # ── LAVY panel: bodovy graf + skutocna priamka ──────────────────────────
    _scatter_by_label(ax_train, xs, ys, true_labels)
    ax_train.plot(x_line, slope * x_line + intercept,
                  color=COLOR_TRUE, linewidth=1.5, linestyle="-",
                  label="True line", zorder=2)
    ax_train.legend(loc="upper left", fontsize=7,
                    facecolor=BG_PANEL, labelcolor=TEXT_COLOR, edgecolor="none")

    # Animovana rozhodovacia hranica (zacina prazdna)
    (boundary_line,) = ax_train.plot(
        [], [], color=COLOR_BOUND, linewidth=2, linestyle="--",
        label="Decision boundary", zorder=5,
    )

    # Informacny textovy box
    info_text = ax_train.text(
        0.98, 0.97, "",
        transform=ax_train.transAxes,
        ha="right", va="top", fontsize=10,
        family="monospace",
        color=TEXT_COLOR,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_PANEL,
                  edgecolor=ACCENT, alpha=0.85),
    )

    # ── PRAVY panel: finalne predikcie ─────────────────────────────────────
    final_preds = perceptron.predict(np.column_stack([xs, ys]))
    _scatter_by_pred(ax_final, xs, ys, final_preds)

    ax_final.plot(x_line, slope * x_line + intercept,
                  color=COLOR_TRUE, linewidth=1.5, linestyle="-",
                  label="True line", zorder=2)

    final_by = perceptron.boundary_y(x_line, epoch=-1)
    if final_by is not None:
        ax_final.plot(x_line, final_by, color=COLOR_BOUND,
                      linewidth=2, linestyle="--",
                      label="Learned boundary", zorder=5)

    final_acc  = perceptron.history[-1]["accuracy"]
    n_epochs   = len(perceptron.history) - 1
    acc_border = _ACC_HIGH if final_acc > 0.95 else (_ACC_MID if final_acc > 0.80 else _ACC_LOW)

    ax_final.text(
        0.98, 0.97,
        f"Accuracy : {final_acc:.1%}\n"
        f"Epochs   : {n_epochs}\n"
        f"LR       : {perceptron.learning_rate}",
        transform=ax_final.transAxes,
        ha="right", va="top", fontsize=10,
        family="monospace",
        color=TEXT_COLOR,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_PANEL,
                  edgecolor=acc_border, alpha=0.85),
    )
    ax_final.legend(loc="upper left", fontsize=7,
                    facecolor=BG_PANEL, labelcolor=TEXT_COLOR, edgecolor="none")

    # ── Synchronizacia y-limitov ───────────────────────────────────────────
    all_y = np.concatenate([ys, slope * x_line + intercept])
    y_margin = (all_y.max() - all_y.min()) * 0.12
    ylim = (all_y.min() - y_margin, all_y.max() + y_margin)
    x_min, x_max = float(x_line[0]), float(x_line[-1])
    for ax in (ax_train, ax_final):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(ylim)

    # Predpocitaj boundary_ys pre vsetky epochy
    boundary_ys: list[np.ndarray | None] = [
        perceptron.boundary_y(x_line, epoch=i)
        for i in range(len(perceptron.history))
    ]

    # Aktualizuj menitelny stavovy slovnik
    state["boundary_line"] = boundary_line
    state["info_text"]     = info_text
    state["boundary_ys"]   = boundary_ys


# ── Hlavny vstupny bod ─────────────────────────────────────────────────────────

def run_animation(
    xs: np.ndarray,
    ys: np.ndarray,
    true_labels: np.ndarray,
    binary_labels: np.ndarray,
    perceptron,
    slope: float = 3,
    intercept: float = 2,
    interval_ms: int = 400,
) -> None:
    """
    Sestavi a zobrazi interaktivnu animovanu figuru.

    Parametre
    ----------
    xs, ys          : suradnice bodov (tvar N,)
    true_labels     : ternarne navestia {-1, 0, +1}
    binary_labels   : binarne navestia  {-1, +1}  pouzivane pri trenovani
    perceptron      : uz natrenovana instancia Perceptronu
    slope           : smernica deliace priamky y = slope*x + intercept
    intercept       : y-suradnica priesecnika deliace priamky s osou y
    interval_ms     : milisekundy medzi snimkami automatickeho prehravania
    """
    _apply_dark_theme()

    x_min  = float(xs.min()) - 0.5
    x_max  = float(xs.max()) + 0.5
    x_line = np.linspace(x_min, x_max, 300)

    # ── Rozlozenie figury ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8), facecolor=BG_DARK)
    fig.suptitle(
        f"Perceptron  ·  y = {slope}x + {intercept}",
        color=TEXT_COLOR, fontsize=14, fontweight="bold", y=0.97,
    )

    # Vyhrad dolny pruh pre ovladacie prvky
    plt.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.30, wspace=0.30)

    ax_train = fig.add_subplot(1, 2, 1)
    ax_final = fig.add_subplot(1, 2, 2)

    # Tenky oddelovac medzi oblastou grafu a pruhom ovladacich prvkov
    fig.add_artist(
        Line2D([0.05, 0.95], [0.275, 0.275],
               transform=fig.transFigure,
               color=ACCENT, linewidth=0.8, alpha=0.5)
    )

    # ── Stavovy slovnik ────────────────────────────────────────────────────
    state: dict = {
        "epoch":             0,
        "playing":           False,
        "_updating_slider":  False,
        "boundary_line":     None,
        "info_text":         None,
        "boundary_ys":       [],
        "x_line":            x_line,
        "slope":             slope,
        "intercept":         intercept,
        "perceptron":        perceptron,
    }

    # Pociatocne naplnenie oboch osi
    _populate_axes(ax_train, ax_final, xs, ys, true_labels, binary_labels,
                   perceptron, x_line, slope, intercept, state)

    n_epochs = len(perceptron.history) - 1

    # ── Osi pre ovladacie prvky ────────────────────────────────────────────
    # Siroki posuvnik epochy
    slider_ax = fig.add_axes([0.10, 0.18, 0.80, 0.03], facecolor=BG_PANEL)

    # Oblast pretrenovania (lava strana rady tlacidiel)
    tb_slope_ax    = fig.add_axes([0.14, 0.09, 0.06, 0.05], facecolor=BG_PANEL)
    tb_icept_ax    = fig.add_axes([0.28, 0.09, 0.06, 0.05], facecolor=BG_PANEL)
    btn_retrain_ax = fig.add_axes([0.37, 0.09, 0.07, 0.05], facecolor=BG_PANEL)

    # Navigacne tlacidla (prava strana rady tlacidiel)
    btn_prev_ax = fig.add_axes([0.50, 0.09, 0.07, 0.05], facecolor=BG_PANEL)
    btn_play_ax = fig.add_axes([0.58, 0.09, 0.10, 0.05], facecolor=BG_PANEL)
    btn_next_ax = fig.add_axes([0.69, 0.09, 0.07, 0.05], facecolor=BG_PANEL)
    btn_rst_ax  = fig.add_axes([0.77, 0.09, 0.09, 0.05], facecolor=BG_PANEL)

    # Vertikalny oddelovac medzi skupinami pretrenovania a navigacie
    fig.text(0.465, 0.115, "│", ha="center", va="center",
             color=TEXT_COLOR, fontsize=18, alpha=0.45,
             transform=fig.transFigure)

    slider = Slider(
        slider_ax, "Epoch", 0, n_epochs, valinit=0, valstep=1, color=ACCENT,
    )
    slider.label.set_color(TEXT_COLOR)
    slider.valtext.set_color(TEXT_COLOR)

    btn_play    = Button(btn_play_ax, "▶ Play",     color=BG_PANEL, hovercolor="#2a2a4a")
    btn_prev    = Button(btn_prev_ax, "< Prev",     color=BG_PANEL, hovercolor="#2a2a4a")
    btn_next    = Button(btn_next_ax, "Next >",     color=BG_PANEL, hovercolor="#2a2a4a")
    btn_restart = Button(btn_rst_ax,  "Restart",    color=BG_PANEL, hovercolor="#2a2a4a")
    btn_retrain = Button(btn_retrain_ax, "⟳ Retrain", color=BG_PANEL, hovercolor="#2a2a4a")

    for btn in (btn_play, btn_prev, btn_next, btn_restart, btn_retrain):
        btn.label.set_color(TEXT_COLOR)

    tb_slope = TextBox(tb_slope_ax, "a = ", initial=str(slope))
    tb_icept = TextBox(tb_icept_ax, "b = ", initial=str(intercept))

    tb_slope.label.set_color(TEXT_COLOR)
    tb_icept.label.set_color(TEXT_COLOR)
    tb_slope.text_disp.set_color(BG_DARK)
    tb_icept.text_disp.set_color(BG_DARK)

    # ── Interne pomocne funkcie ────────────────────────────────────────────

    def _draw_epoch(epoch: int) -> None:
        """Aktualizuj animovane prvky laveho panela pre *epoch*."""
        boundary_ys = state["boundary_ys"]
        x_ln        = state["x_line"]
        n_ep        = len(boundary_ys) - 1

        by = boundary_ys[epoch]
        if by is not None:
            state["boundary_line"].set_data(x_ln, by)
        else:
            state["boundary_line"].set_data([], [])

        snap = state["perceptron"].history[epoch]
        state["info_text"].set_text(
            f"Epoch  : {epoch}/{n_ep}\n"
            f"Acc    : {snap['accuracy']:.1%}"
        )
        fig.canvas.draw_idle()

    def _set_epoch(epoch: int) -> None:
        n_ep  = len(state["boundary_ys"]) - 1
        epoch = int(np.clip(epoch, 0, n_ep))
        state["epoch"] = epoch
        state["_updating_slider"] = True
        slider.set_val(epoch)
        state["_updating_slider"] = False
        _draw_epoch(epoch)

    # Pociatocne vykreslenie
    _draw_epoch(0)

    # ── Animacia ──────────────────────────────────────────────────────────
    def _anim_step(frame: int) -> list:
        if not state["playing"]:
            return []
        n_ep       = len(state["boundary_ys"]) - 1
        next_epoch = state["epoch"] + 1
        if next_epoch > n_ep:
            state["playing"] = False
            btn_play.label.set_text("▶ Play")
            return []
        _set_epoch(next_epoch)
        return []

    anim = animation.FuncAnimation(
        fig, _anim_step, interval=interval_ms, blit=False, cache_frame_data=False
    )

    # ── Spetne volania ─────────────────────────────────────────────────────

    def on_retrain(event) -> None:
        # Parsuj smernica a intercept z textovych poli; pri chybe zachovaj predchadzajuci
        try:
            a = float(tb_slope.text)
        except ValueError:
            a = state["slope"]
            tb_slope.set_val(str(a))

        try:
            b = float(tb_icept.text)
        except ValueError:
            b = state["intercept"]
            tb_icept.set_val(str(b))

        state["slope"]     = a
        state["intercept"] = b
        state["playing"]   = False
        btn_play.label.set_text("▶ Play")

        # Generuj nove data a trenuj znovu s rovnakymi hyperparametrami
        new_xs, new_ys, new_labels = generate_points(slope=a, intercept=b)
        new_binary = to_binary(new_labels)

        new_X    = np.column_stack([new_xs, new_ys])
        old_perc = state["perceptron"]
        new_perc = type(old_perc)(
            learning_rate=old_perc.learning_rate,
            n_epochs=old_perc.n_epochs,
        )
        new_perc.fit(new_X, new_binary)

        # Prepocitaj x_line pre novy rozsah dat
        new_x_min  = float(new_xs.min()) - 0.5
        new_x_max  = float(new_xs.max()) + 0.5
        new_x_line = np.linspace(new_x_min, new_x_max, 300)

        state["perceptron"] = new_perc
        state["x_line"]     = new_x_line

        _populate_axes(ax_train, ax_final,
                       new_xs, new_ys, new_labels, new_binary,
                       new_perc, new_x_line, a, b, state)

        # Aktualizuj rozsah posuvnika
        new_n_epochs      = len(new_perc.history) - 1
        slider.valmax     = new_n_epochs
        slider.ax.set_xlim(0, new_n_epochs)

        # Reset na epochu 0
        state["epoch"]               = 0
        state["_updating_slider"]    = True
        slider.set_val(0)
        state["_updating_slider"]    = False
        _draw_epoch(0)

        # Dynamicky nadpis
        fig.suptitle(
            f"Perceptron  ·  y = {a}x + {b}",
            color=TEXT_COLOR, fontsize=14, fontweight="bold", y=0.97,
        )
        fig.canvas.draw_idle()

    def on_slider(val: float) -> None:
        if state["_updating_slider"]:
            return
        _draw_epoch(int(val))
        state["epoch"] = int(val)

    def on_play(event) -> None:
        if state["playing"]:
            state["playing"] = False
            btn_play.label.set_text("▶ Play")
        else:
            n_ep = len(state["boundary_ys"]) - 1
            if state["epoch"] >= n_ep:
                state["epoch"] = 0
            state["playing"] = True
            btn_play.label.set_text("⏸ Pause")

    def on_prev(event) -> None:
        state["playing"] = False
        btn_play.label.set_text("▶ Play")
        _set_epoch(state["epoch"] - 1)

    def on_next(event) -> None:
        state["playing"] = False
        btn_play.label.set_text("▶ Play")
        _set_epoch(state["epoch"] + 1)

    def on_restart(event) -> None:
        state["playing"] = False
        btn_play.label.set_text("▶ Play")
        _set_epoch(0)

    slider.on_changed(on_slider)
    btn_play.on_clicked(on_play)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    btn_restart.on_clicked(on_restart)
    btn_retrain.on_clicked(on_retrain)

    # Uchov referenciu na zabranenie garbage-collection animacie
    fig._anim = anim  # type: ignore[attr-defined]

    plt.show()
