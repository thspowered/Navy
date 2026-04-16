"""
visualization.py – Interaktivna vizualizacia Task 5: Pole-balancing.

Rozlozenie figury (15x9 palcov, tmava tema):
  Vlavo  (58%) – animovane prostredie CartPole (vozik + tycka)
  Vpravo (35%) – tlacidla na ovladanie + informacny panel
  Dolu         – animacne ovladacie prvky (slider, Play/Pause, Prev/Next, Restart)

Farebna schema (identicka s predchadzajucimi taskmi):
  BG_DARK  = "#0d1117"
  BG_PANEL = "#161b22"
  BG_AXES  = "#1c2333"
  TEXT     = "#f0f6fc"
  ACCENT   = "#58a6ff"

Postup:
  1. "Train Q-agent"   – trenovanie Q-learning agenta (5000 epizod)
  2. "Train Neural Net"– zbere skusenosti a natrenuoje NN (supervised)
  3. "Run Neural Net"  – zobrazi animaciu nauceneho agenta
  4. "Reset"           – resetuje vsetko
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim_mod
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, Slider
from matplotlib.lines import Line2D

from .q_learning   import CartPoleQLearning
from .neural_network import NeuralNetwork

# ── Farby temy (identicky s predchadzajucimi taskmi) ───────────────────────────
BG_DARK    = "#0d1117"
BG_PANEL   = "#161b22"
BG_AXES    = "#1c2333"
TEXT_COLOR = "#f0f6fc"
ACCENT     = "#58a6ff"

# Farby prvkov
COLOR_CART   = "#58a6ff"   # modra  – vozik
COLOR_POLE   = "#e3b341"   # zlta   – tycka
COLOR_TRACK  = "#484f58"   # seda   – kolataj
COLOR_WHEEL  = "#8b949e"   # svetla seda – kolesa

# Farby tlacidiel
COLOR_TRAIN_Q  = "#56d364"   # zelena  – trenovanie Q-agenta
COLOR_TRAIN_NN = "#bc8cff"   # fialova – trenovanie NN
COLOR_RUN_NN   = "#e3b341"   # zlta    – spustenie NN
COLOR_RESET    = "#ff7b72"   # cervena – reset

# Vizualizacne konstanty CartPole
CART_W      = 0.5    # sirka vozika (display jednotky)
CART_H      = 0.3    # vyska vozika
POLE_LENGTH = 1.0    # dlzka tycky (display jednotky)
WHEEL_R     = 0.08   # polomer kolesa
TRACK_Y     = 0.0    # y-suradnica voznice
TRACK_HALF  = 2.6    # polovina dlzky voznice

# Trenovacie parametre
N_EPISODES_Q  = 5000
N_COLLECT     = 200
N_EPOCHS_NN   = 300


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


def _draw_cartpole(
    ax,
    obs: np.ndarray,
    step: int = 0,
    total_steps: int = 0,
    label: str = "",
) -> None:
    """
    Vykreslí CartPole prostredie na danu os pre dany stav.

    Parametre
    ---------
    ax          : matplotlib Axes
    obs         : [cart_pos, cart_vel, pole_angle, pole_vel]
    step        : aktualny krok animacie
    total_steps : celkovy pocet krokov trajektorie
    label       : popis aktualne beziacej politiky
    """
    ax.clear()
    ax.set_facecolor(BG_DARK)
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-0.5, 2.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(BG_PANEL)

    cart_pos   = float(obs[0])
    pole_angle = float(obs[2])   # radians; 0 = kolmy, + = vpravo

    # Voznica (track)
    ax.plot(
        [-TRACK_HALF, TRACK_HALF],
        [TRACK_Y - CART_H / 2, TRACK_Y - CART_H / 2],
        color=COLOR_TRACK, linewidth=3, solid_capstyle="round", zorder=1,
    )

    # Vozik
    cart_x = cart_pos - CART_W / 2
    cart_y = TRACK_Y - CART_H / 2
    cart_rect = mpatches.FancyBboxPatch(
        (cart_x, cart_y), CART_W, CART_H,
        boxstyle="round,pad=0.03",
        linewidth=1.2,
        edgecolor=BG_PANEL,
        facecolor=COLOR_CART,
        zorder=3,
    )
    ax.add_patch(cart_rect)

    # Kolesa
    for dx in (-CART_W * 0.28, CART_W * 0.28):
        wheel = plt.Circle(
            (cart_pos + dx, TRACK_Y - CART_H / 2),
            WHEEL_R,
            color=COLOR_WHEEL,
            zorder=4,
        )
        ax.add_patch(wheel)
        # Hub
        ax.plot(
            cart_pos + dx, TRACK_Y - CART_H / 2,
            "o", color=BG_DARK, markersize=3, zorder=5,
        )

    # Tycka (pole)
    pivot_x = cart_pos
    pivot_y = TRACK_Y + CART_H / 2
    tip_x   = pivot_x + np.sin(pole_angle) * POLE_LENGTH
    tip_y   = pivot_y + np.cos(pole_angle) * POLE_LENGTH

    ax.plot(
        [pivot_x, tip_x], [pivot_y, tip_y],
        color=COLOR_POLE, linewidth=6, solid_capstyle="round", zorder=2,
    )
    # Cep (pivot)
    ax.plot(pivot_x, pivot_y, "o", color=BG_PANEL, markersize=6, zorder=5)

    # Tienova celu pri uhle
    shadow_alpha = min(0.25, abs(pole_angle) * 2)
    if shadow_alpha > 0.02:
        ax.plot(
            [pivot_x, tip_x], [pivot_y, tip_y],
            color=COLOR_POLE, linewidth=10,
            solid_capstyle="round", alpha=shadow_alpha, zorder=1,
        )

    # Informacia o kroku
    ax.text(
        0.99, 0.02,
        f"Step: {step} / {total_steps - 1}",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9, family="monospace", color=TEXT_COLOR,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_PANEL,
                  edgecolor=ACCENT, alpha=0.88),
    )

    # Uhol tycky
    ax.text(
        0.01, 0.98,
        f"angle: {np.degrees(pole_angle):+.1f}°",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9, family="monospace", color=COLOR_POLE,
    )
    # Pozicia vozika
    ax.text(
        0.01, 0.88,
        f"cart:  {cart_pos:+.2f} m",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9, family="monospace", color=COLOR_CART,
    )

    title = f"CartPole  –  {label}" if label else "CartPole"
    ax.set_title(title, color=TEXT_COLOR, fontsize=12, pad=8)


def run_app() -> None:
    """
    Spusti hlavnu interaktivnu aplikaciu Task 5 – Pole-balancing.
    """
    _apply_dark_theme()

    # ── Agenty ──────────────────────────────────────────────────────────────────
    q_agent = CartPoleQLearning()
    nn      = NeuralNetwork()

    # Stav aplikacie
    app_state: dict = {
        "q_trained":    False,
        "nn_trained":   False,
        "trajectory":   [],    # list[np.ndarray] – aktualna trajektoria
        "frame":        0,
        "playing":      False,
        "_upd_slider":  False,
        "anim_obj":     None,
        "mode":         "",    # "Q" alebo "NN" – kto generoval trajektoriu
    }

    # Pociatocna idle trajektoria (tycka stoji kolmo)
    idle_obs = np.zeros(4)

    # ── Figura ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 9), facecolor=BG_DARK)
    fig.suptitle(
        "Task 5 – Pole-balancing  ·  Q-learning  →  Neural Network",
        color=TEXT_COLOR, fontsize=14, fontweight="bold", y=0.97,
    )

    # Oddelovacia cara nad animacnymi ovladacmi
    fig.add_artist(
        Line2D([0.03, 0.97], [0.275, 0.275],
               transform=fig.transFigure,
               color=ACCENT, linewidth=0.8, alpha=0.5)
    )

    # ── Osa pre CartPole (vlavo) ────────────────────────────────────────────────
    ax_cart = fig.add_axes([0.03, 0.30, 0.56, 0.62])

    # ── Tlacidla (vpravo) ───────────────────────────────────────────────────────
    btn_train_q_ax  = fig.add_axes([0.63, 0.83,  0.33, 0.065])
    btn_train_nn_ax = fig.add_axes([0.63, 0.755, 0.33, 0.065])
    btn_run_nn_ax   = fig.add_axes([0.63, 0.655, 0.33, 0.065])
    btn_reset_ax    = fig.add_axes([0.63, 0.575, 0.33, 0.065])

    btn_train_q  = Button(btn_train_q_ax,  "Train Q-agent",   color=BG_PANEL, hovercolor="#1f3a1f")
    btn_train_nn = Button(btn_train_nn_ax, "Train Neural Net", color=BG_PANEL, hovercolor="#2a1f3a")
    btn_run_nn   = Button(btn_run_nn_ax,   "Run Neural Net",   color=BG_PANEL, hovercolor="#3a2e10")
    btn_reset    = Button(btn_reset_ax,    "Reset",            color=BG_PANEL, hovercolor="#3a1515")

    btn_train_q.label.set_color(COLOR_TRAIN_Q)
    btn_train_nn.label.set_color(COLOR_TRAIN_NN)
    btn_run_nn.label.set_color(COLOR_RUN_NN)
    btn_reset.label.set_color(COLOR_RESET)

    for btn in (btn_train_q, btn_train_nn, btn_run_nn, btn_reset):
        btn.label.set_fontsize(10)
        btn.label.set_fontweight("bold")

    # Informacny panel (vpravo dolu)
    ax_info = fig.add_axes([0.63, 0.30, 0.33, 0.250])
    ax_info.set_facecolor(BG_PANEL)
    ax_info.set_xticks([])
    ax_info.set_yticks([])
    for spine in ax_info.spines.values():
        spine.set_edgecolor(ACCENT)
        spine.set_linewidth(1.2)

    info_text = ax_info.text(
        0.5, 0.5,
        "Kroky:\n"
        "  1. 'Train Q-agent'\n"
        "     (5 000 epizod)\n\n"
        "  2. 'Train Neural Net'\n"
        "     (supervised learning\n"
        "      zo Q-agenta)\n\n"
        "  3. 'Run Neural Net'\n"
        "     (zobrazi animaciu)",
        transform=ax_info.transAxes,
        ha="center", va="center",
        fontsize=8.5, family="monospace",
        color=TEXT_COLOR,
    )

    # ── Animacne ovladacie prvky (dolu) ─────────────────────────────────────────
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
        "Kliknite 'Train Q-agent' pre spustenie trenovania.",
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

    def _refresh_cartpole(fi: int | None = None) -> None:
        traj = app_state["trajectory"]
        if not traj:
            _draw_cartpole(ax_cart, idle_obs, label="waiting…")
        else:
            fi = fi if fi is not None else app_state["frame"]
            fi = int(np.clip(fi, 0, len(traj) - 1))
            mode_lbl = (
                "Q-agent (greedy)" if app_state["mode"] == "Q"
                else "Neural Net"  if app_state["mode"] == "NN"
                else ""
            )
            _draw_cartpole(ax_cart, traj[fi], step=fi,
                           total_steps=len(traj), label=mode_lbl)
        fig.canvas.draw_idle()

    # ── Animacia ────────────────────────────────────────────────────────────────

    def _start_animation(trajectory: list) -> None:
        """Nastavi novu trajektoriu a spusti FuncAnimation."""
        if app_state["anim_obj"] is not None:
            try:
                app_state["anim_obj"].event_source.stop()
            except Exception:
                pass

        app_state["trajectory"] = trajectory
        app_state["frame"]      = 0
        app_state["playing"]    = False
        app_state["anim_obj"]   = None

        n = len(trajectory)
        slider.valmin = 0
        slider.valmax = max(n - 1, 1)
        slider.set_val(0)
        slider.ax.set_xlim(0, max(n - 1, 1))

        _refresh_cartpole(0)

        anim_obj = anim_mod.FuncAnimation(
            fig, _anim_step, interval=40,
            blit=False, cache_frame_data=False,
        )
        app_state["anim_obj"] = anim_obj
        fig._anim = anim_obj

    def _set_frame(fi: int) -> None:
        fi = int(np.clip(fi, 0, len(app_state["trajectory"]) - 1))
        app_state["frame"]       = fi
        app_state["_upd_slider"] = True
        slider.set_val(fi)
        app_state["_upd_slider"] = False
        _refresh_cartpole(fi)

        traj = app_state["trajectory"]
        n    = len(traj)
        if fi == n - 1:
            result = f"Survived {n - 1} steps!"
            color  = COLOR_RUN_NN if app_state["mode"] == "NN" else COLOR_TRAIN_Q
            _set_status(result, color=color)
        else:
            _set_status(f"Step {fi} / {n - 1}")

    def _anim_step(_fn: int) -> list:
        if not app_state["playing"]:
            return []
        nxt = app_state["frame"] + 1
        if nxt >= len(app_state["trajectory"]):
            app_state["playing"] = False
            btn_play.label.set_text("▶ Play")
            return []
        _set_frame(nxt)
        return []

    # ── Obsluha animacnych tlacidiel ────────────────────────────────────────────

    def on_slider(val: float) -> None:
        if app_state["_upd_slider"]:
            return
        fi = int(val)
        app_state["frame"] = fi
        _refresh_cartpole(fi)

    def on_play(event) -> None:
        if not app_state["trajectory"]:
            return
        if app_state["playing"]:
            app_state["playing"] = False
            btn_play.label.set_text("▶ Play")
        else:
            if app_state["frame"] >= len(app_state["trajectory"]) - 1:
                app_state["frame"] = 0
            app_state["playing"] = True
            btn_play.label.set_text("⏸ Pause")

    def on_prev(event) -> None:
        if not app_state["trajectory"]:
            return
        app_state["playing"] = False
        btn_play.label.set_text("▶ Play")
        _set_frame(app_state["frame"] - 1)

    def on_next(event) -> None:
        if not app_state["trajectory"]:
            return
        app_state["playing"] = False
        btn_play.label.set_text("▶ Play")
        _set_frame(app_state["frame"] + 1)

    def on_restart(event) -> None:
        if not app_state["trajectory"]:
            return
        app_state["playing"] = False
        btn_play.label.set_text("▶ Play")
        _set_frame(0)

    slider.on_changed(on_slider)
    btn_play.on_clicked(on_play)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    btn_restart.on_clicked(on_restart)

    # ── Obsluha: Train Q-agent ───────────────────────────────────────────────────

    def on_train_q(event) -> None:
        _set_status("Trenujem Q-agenta… prosim cakajte.", color=ACCENT)
        _update_info(f"Q-learning beží…\n{N_EPISODES_Q} epizod")
        fig.canvas.draw_idle()
        plt.pause(0.01)

        q_agent.reset()
        nn.reset()
        app_state["q_trained"]  = False
        app_state["nn_trained"] = False

        scores = q_agent.train(n_episodes=N_EPISODES_Q)
        app_state["q_trained"] = True

        avg100 = float(np.mean(scores[-100:]))
        best   = max(scores)
        _update_info(
            f"Q-agent natrenovany\n"
            f"  epizody : {N_EPISODES_Q}\n"
            f"  ε final : {q_agent.epsilon:.3f}\n\n"
            f"  best    : {best:.0f}\n"
            f"  avg(100): {avg100:.1f}\n\n"
            f"α={q_agent.alpha}  γ={q_agent.gamma}\n\n"
            f"Dalej: 'Train Neural Net'"
        )
        _set_status(
            f"Q-agent hotovy! Best={best:.0f}, avg(100)={avg100:.1f}",
            color=COLOR_TRAIN_Q,
        )

    btn_train_q.on_clicked(on_train_q)

    # ── Obsluha: Train Neural Net ────────────────────────────────────────────────

    def on_train_nn(event) -> None:
        if not app_state["q_trained"]:
            _set_status("Najprv spusti 'Train Q-agent'!", color=COLOR_RESET)
            return

        _set_status("Zbieram skusenosti a trenujem NN… prosim cakajte.", color=ACCENT)
        _update_info(
            f"Trenujem Neural Net…\n"
            f"  zber: {N_COLLECT} epizod\n"
            f"  epochy: {N_EPOCHS_NN}"
        )
        fig.canvas.draw_idle()
        plt.pause(0.01)

        nn.reset()
        X, y = q_agent.collect_experiences(n_episodes=N_COLLECT)
        losses = nn.train(X, y, epochs=N_EPOCHS_NN)

        app_state["nn_trained"] = True
        final_loss = losses[-1]
        _update_info(
            f"Neural Net natrenovana\n"
            f"  vzorky  : {len(X)}\n"
            f"  epochy  : {N_EPOCHS_NN}\n"
            f"  presnost: {nn.train_accuracy * 100:.1f}%\n"
            f"  loss    : {final_loss:.4f}\n\n"
            f"  arch.   : 4→64→32→2\n"
            f"  aktivac.: ReLU+Softmax\n\n"
            f"Dalej: 'Run Neural Net'"
        )
        _set_status(
            f"NN hotova! Presnost={nn.train_accuracy*100:.1f}%, loss={final_loss:.4f}",
            color=COLOR_TRAIN_NN,
        )

    btn_train_nn.on_clicked(on_train_nn)

    # ── Obsluha: Run Neural Net ─────────────────────────────────────────────────

    def on_run_nn(event) -> None:
        if not app_state["nn_trained"]:
            _set_status("Najprv spusti 'Train Neural Net'!", color=COLOR_RESET)
            return

        _set_status("Spustam NN epizodu…", color=ACCENT)
        fig.canvas.draw_idle()
        plt.pause(0.01)

        traj = nn.run_episode()
        app_state["mode"] = "NN"
        _set_status(
            f"NN epizoda: {len(traj) - 1} krokov.",
            color=COLOR_RUN_NN,
        )
        _start_animation(traj)
        on_play(None)

    btn_run_nn.on_clicked(on_run_nn)

    # ── Obsluha: Reset ───────────────────────────────────────────────────────────

    def on_reset(event) -> None:
        q_agent.reset()
        nn.reset()
        app_state["q_trained"]  = False
        app_state["nn_trained"] = False
        app_state["trajectory"] = []
        app_state["frame"]      = 0
        app_state["playing"]    = False
        app_state["mode"]       = ""
        btn_play.label.set_text("▶ Play")

        _update_info(
            "Kroky:\n"
            "  1. 'Train Q-agent'\n"
            "     (5 000 epizod)\n\n"
            "  2. 'Train Neural Net'\n"
            "     (supervised learning\n"
            "      zo Q-agenta)\n\n"
            "  3. 'Run Neural Net'\n"
            "     (zobrazi animaciu)"
        )
        _set_status("Resetovane. Kliknite 'Train Q-agent'.")
        _refresh_cartpole()

    btn_reset.on_clicked(on_reset)

    # ── Pociatocny stav ─────────────────────────────────────────────────────────
    _refresh_cartpole()
    plt.show()
