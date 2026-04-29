"""
visualization.py - Interaktivna vizualizacia Task 10: Logisticka mapa,
bifurkacny diagram a predikcia neuronovou sietou.

Rozlozenie figury (15x9 palcov, tmava tema):
  Vlavo (~57%) – bifurkacny diagram (jeden alebo dva subgrafy podla modu)
  Vpravo (~35%) – tlacidla modov + info panel
  Dolu          – animacne ovladacie prvky (slider, Play/Pause, Prev/Next, Restart)

Farebna schema (identicka s predchadzajucimi taskmi):
  BG_DARK  = "#0d1117"
  BG_PANEL = "#161b22"
  BG_AXES  = "#1c2333"
  TEXT     = "#f0f6fc"
  ACCENT   = "#58a6ff"

Mody:
  1. "Bifurkacny diagram"   – cista logisticka mapa
  2. "Trening + Predikcia"  – split-screen: vlavo trening, vpravo NN predikcia
  3. "Predikcia (overlay)"  – bifurkacia + cervene NN body (Example 3)
  4. "Clear"                – vymazat plochu

Animacia: progresivne dopisanie diagramu po stlpcoch parametra a (zlava doprava).
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim_mod
from matplotlib.widgets import Button, Slider
from matplotlib.lines import Line2D

from .logistic import compute_bifurcation, make_training_pairs
from .predictor import MLP, predict_bifurcation_onestep, A_MAX

# ── Farby temy (identicky s predchadzajucimi taskmi) ───────────────────────────
BG_DARK    = "#0d1117"
BG_PANEL   = "#161b22"
BG_AXES    = "#1c2333"
TEXT_COLOR = "#f0f6fc"
ACCENT     = "#58a6ff"

COLOR_BIF     = "#56d364"   # zelena – cista bifurkacia
COLOR_NN      = "#bc8cff"   # fialova – split-screen (NN predikcia)
COLOR_OVERLAY = "#f0b429"   # zlta   – overlay rezim
COLOR_CLEAR   = "#8b949e"   # seda   – clear

# Body bifurkacneho diagramu (cierne v Example 1, biele na tmavom pozadi)
COLOR_POINTS    = "#e6edf3"   # takmer biela – atraktor
COLOR_NN_DOTS   = "#ff5c5c"   # cervena – NN predikcia (Example 3)

# ── Parametre vypoctu (default) ────────────────────────────────────────────────
N_SAMPLES   = 800     # pocet hodnot parametra a (np.linspace(0, 4, N_SAMPLES))
N_ITER      = 600     # celkovy pocet iteracii pre kazde a
N_TRANSIENT = 150     # pocet uvodnych iteracii ktore zahodime

# Parametre NN trenovania
NN_LAYERS     = [2, 32, 16, 1]
NN_TRAIN_N    = 80_000
NN_EPOCHS     = 40
NN_LR         = 0.005
NN_BATCH      = 512
NN_SEED       = 0

ANIM_FRAMES = 30      # pocet animacnych framov (progresivne stlpce diagramu)


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


def _split_into_frames(
    a_arr: np.ndarray,
    x_arr: np.ndarray,
    n_frames: int,
    a_min: float = 0.0,
    a_max: float = 4.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Rozseka data podla a-suradnice na 'n_frames' kumulatívnych podmnozin
    (zlava doprava). Pre frame k vrati vsetky body s a <= a_min + (k+1)/n * (a_max-a_min).

    Vyuziva to fakt ze vstupne pole je vytvorene meshgridom z linspace,
    takze body s rovnakou hodnotou parametra a su pohromade.
    """
    edges = np.linspace(a_min, a_max, n_frames + 1)
    # Trieden chod – stacia masky podla a hodnoty
    frames: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_frames):
        upper = edges[k + 1]
        mask = a_arr <= upper + 1e-12
        frames.append((a_arr[mask], x_arr[mask]))
    return frames


# ──────────────────────────────────────────────────────────────────────────────
# Hlavna aplikacia
# ──────────────────────────────────────────────────────────────────────────────
def run_app() -> None:
    """Spusti hlavnu interaktivnu aplikaciu Task 10 – Logistic map + NN prediction."""
    _apply_dark_theme()

    # ── Stav aplikacie ─────────────────────────────────────────────────────────
    app_state: dict = {
        "mode":          None,    # "bifurc" | "split" | "overlay" | None
        # Bifurkacne data
        "a_full":        None,
        "x_full":        None,
        # NN predikcia (one-step na kazdom bode atraktora)
        "a_pred":        None,
        "x_pred":        None,
        # Animacne framy (kumulativne – pribudaju body zlava doprava)
        "frames_real":   [],
        "frames_pred":   [],
        "frame":         0,
        "total":         0,
        "playing":       False,
        "_upd_slider":   False,
        "anim_obj":      None,
        "color":         ACCENT,
        # Natrenovany model (cache – neuci sa znova)
        "mlp":           None,
        "train_hist":    None,
    }

    # ── Figura ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 9), facecolor=BG_DARK)
    fig.suptitle(
        "Task 10 – Theory of Chaos  ·  Logisticka mapa a NN predikcia",
        color=TEXT_COLOR, fontsize=14, fontweight="bold", y=0.97,
    )

    # Oddelovacia cara nad animacnymi ovladacmi
    fig.add_artist(
        Line2D([0.03, 0.97], [0.275, 0.275],
               transform=fig.transFigure,
               color=ACCENT, linewidth=0.8, alpha=0.5)
    )

    # ── Hlavne osi pre diagram ─────────────────────────────────────────────────
    # ax_left  – pouziva sa pre vsetky mody
    # ax_right – pouziva sa len v split-screen mode (Trening + Predikcia)
    ax_left  = fig.add_axes([0.04, 0.31, 0.55, 0.61])
    ax_right = fig.add_axes([0.32, 0.31, 0.27, 0.61])
    ax_right.set_visible(False)

    def _style_axes(ax, title: str = "") -> None:
        ax.set_facecolor(BG_AXES)
        for spine in ax.spines.values():
            spine.set_edgecolor(BG_PANEL)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax.set_xlabel("Parameter a", color=TEXT_COLOR, fontsize=10)
        ax.set_ylabel("x (atraktor)", color=TEXT_COLOR, fontsize=10)
        ax.grid(True, alpha=0.15, color=BG_PANEL)
        ax.set_xlim(0.0, 4.0)
        ax.set_ylim(-0.02, 1.02)
        if title:
            ax.set_title(title, color=TEXT_COLOR, fontsize=11, pad=6)

    _style_axes(ax_left, "Bifurkacny diagram")

    # ── Tlacidla modov (vpravo) ────────────────────────────────────────────────
    btn_b_ax  = fig.add_axes([0.63, 0.86, 0.33, 0.055])
    btn_s_ax  = fig.add_axes([0.63, 0.80, 0.33, 0.055])
    btn_o_ax  = fig.add_axes([0.63, 0.74, 0.33, 0.055])
    btn_cl_ax = fig.add_axes([0.63, 0.68, 0.33, 0.055])

    btn_bifurc  = Button(btn_b_ax, "Bifurkacny diagram",
                         color=BG_PANEL, hovercolor="#1f3a1f")
    btn_split   = Button(btn_s_ax, "Trening + Predikcia",
                         color=BG_PANEL, hovercolor="#2a1f3a")
    btn_overlay = Button(btn_o_ax, "Predikcia (overlay)",
                         color=BG_PANEL, hovercolor="#3a2f1f")
    btn_clear   = Button(btn_cl_ax, "Clear canvas",
                         color=BG_PANEL, hovercolor="#2a2a2a")

    btn_bifurc.label.set_color(COLOR_BIF)
    btn_split.label.set_color(COLOR_NN)
    btn_overlay.label.set_color(COLOR_OVERLAY)
    btn_clear.label.set_color(COLOR_CLEAR)
    for btn in (btn_bifurc, btn_split, btn_overlay, btn_clear):
        btn.label.set_fontsize(10)
        btn.label.set_fontweight("bold")

    # ── Info panel (vpravo) ─────────────────────────────────────────────────────
    ax_info = fig.add_axes([0.63, 0.31, 0.33, 0.36])
    ax_info.set_facecolor(BG_PANEL)
    ax_info.set_xticks([])
    ax_info.set_yticks([])
    for spine in ax_info.spines.values():
        spine.set_edgecolor(ACCENT)
        spine.set_linewidth(1.2)

    info_default = (
        "Logisticka mapa:\n"
        "  x_{n+1} = a * x_n * (1 - x_n)\n\n"
        "Mody:\n"
        "  Bifurkacny diagram\n"
        "    – atraktor pre kazde a\n"
        "  Trening + Predikcia\n"
        "    – realne data vlavo,\n"
        "      NN predikcia vpravo\n"
        "  Predikcia (overlay)\n"
        "    – bifurkacia + NN body\n"
        "      cervenou (point-by-point)\n\n"
        "Parametre:\n"
        f"  Samples (a):   {N_SAMPLES}\n"
        f"  Iteracii:      {N_ITER}\n"
        f"  Tranzient:     {N_TRANSIENT}\n\n"
        "NN: 2-32-16-1 (tanh+sigmoid)\n"
        f"  Trening: {NN_TRAIN_N//1000}k vzoriek,\n"
        f"  {NN_EPOCHS} epoch, Adam lr={NN_LR}"
    )
    info_text = ax_info.text(
        0.5, 0.5, info_default,
        transform=ax_info.transAxes,
        ha="center", va="center",
        fontsize=8, family="monospace",
        color=TEXT_COLOR,
    )

    # ── Animacne ovladacie prvky (dolu) ────────────────────────────────────────
    slider_ax   = fig.add_axes([0.10, 0.18, 0.80, 0.03], facecolor=BG_PANEL)
    btn_prev_ax = fig.add_axes([0.33, 0.09, 0.07, 0.05], facecolor=BG_PANEL)
    btn_play_ax = fig.add_axes([0.41, 0.09, 0.12, 0.05], facecolor=BG_PANEL)
    btn_next_ax = fig.add_axes([0.54, 0.09, 0.07, 0.05], facecolor=BG_PANEL)
    btn_rst_ax  = fig.add_axes([0.62, 0.09, 0.09, 0.05], facecolor=BG_PANEL)

    slider = Slider(slider_ax, "Frame", 0, 1, valinit=0, valstep=1, color=ACCENT)
    slider.label.set_color(TEXT_COLOR)
    slider.valtext.set_color(TEXT_COLOR)

    btn_play    = Button(btn_play_ax, "▶ Play",  color=BG_PANEL, hovercolor="#2a2a4a")
    btn_prev    = Button(btn_prev_ax, "< Prev",  color=BG_PANEL, hovercolor="#2a2a4a")
    btn_next    = Button(btn_next_ax, "Next >",  color=BG_PANEL, hovercolor="#2a2a4a")
    btn_restart = Button(btn_rst_ax,  "Restart", color=BG_PANEL, hovercolor="#2a2a4a")
    for btn in (btn_play, btn_prev, btn_next, btn_restart):
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
        "Vyber mod (Bifurkacny diagram / Trening + Predikcia / Overlay).",
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

    def _layout_for_mode(mode: str | None) -> None:
        """Prepne rozlozenie osi medzi 'split' a 'single'."""
        if mode == "split":
            ax_left.set_position([0.04, 0.31, 0.27, 0.61])
            ax_right.set_position([0.32, 0.31, 0.27, 0.61])
            ax_right.set_visible(True)
        else:
            ax_left.set_position([0.04, 0.31, 0.55, 0.61])
            ax_right.set_visible(False)

    def _draw_clear() -> None:
        """Vyprazdni obe osi do startovacieho stavu."""
        ax_left.clear()
        ax_right.clear()
        _style_axes(ax_left, "Bifurkacny diagram")
        _style_axes(ax_right, "Network prediction")

    def _draw_frame(frame_idx: int) -> None:
        """Vykresli frame podla aktualneho modu (a indexu)."""
        mode = app_state["mode"]
        frames_real = app_state["frames_real"]
        if not frames_real:
            return
        n = len(frames_real)
        frame_idx = int(np.clip(frame_idx, 0, n - 1))

        a_r, x_r = frames_real[frame_idx]

        # Vlavo – realne data (vzdy)
        ax_left.clear()
        ax_left.set_facecolor(BG_AXES)
        ax_left.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax_left.set_xlabel("Parameter a", color=TEXT_COLOR, fontsize=10)
        ax_left.set_ylabel("x (atraktor)", color=TEXT_COLOR, fontsize=10)
        ax_left.grid(True, alpha=0.15, color=BG_PANEL)
        ax_left.set_xlim(0.0, 4.0)
        ax_left.set_ylim(-0.02, 1.02)
        for spine in ax_left.spines.values():
            spine.set_edgecolor(BG_PANEL)

        if mode == "overlay":
            ax_left.set_title(
                f"Bifurkacia + NN predikcia  ·  frame {frame_idx + 1}/{n}",
                color=TEXT_COLOR, fontsize=11, pad=6,
            )
            ax_left.plot(a_r, x_r, ",", color=COLOR_POINTS, alpha=0.85, rasterized=True)
            a_p, x_p = app_state["frames_pred"][frame_idx]
            ax_left.plot(a_p, x_p, ",", color=COLOR_NN_DOTS, alpha=0.85, rasterized=True)
            # Mala legenda
            ax_left.scatter([], [], s=8, c=COLOR_POINTS,  label="Skutocnost")
            ax_left.scatter([], [], s=8, c=COLOR_NN_DOTS, label="NN predikcia")
            leg = ax_left.legend(
                loc="upper left", fontsize=8, framealpha=0.85,
                facecolor=BG_PANEL, edgecolor=ACCENT, labelcolor=TEXT_COLOR,
            )
            for txt in leg.get_texts():
                txt.set_color(TEXT_COLOR)
        elif mode == "split":
            ax_left.set_title(
                f"Training data  ·  frame {frame_idx + 1}/{n}",
                color=TEXT_COLOR, fontsize=11, pad=6,
            )
            ax_left.plot(a_r, x_r, ",", color="#79c0ff", alpha=0.9, rasterized=True)

            ax_right.clear()
            ax_right.set_facecolor(BG_AXES)
            ax_right.tick_params(colors=TEXT_COLOR, labelsize=9)
            ax_right.set_xlabel("Parameter a", color=TEXT_COLOR, fontsize=10)
            ax_right.set_ylabel("x (atraktor)", color=TEXT_COLOR, fontsize=10)
            ax_right.grid(True, alpha=0.15, color=BG_PANEL)
            ax_right.set_xlim(0.0, 4.0)
            ax_right.set_ylim(-0.02, 1.02)
            for spine in ax_right.spines.values():
                spine.set_edgecolor(BG_PANEL)
            ax_right.set_title(
                f"Network prediction  ·  frame {frame_idx + 1}/{n}",
                color=TEXT_COLOR, fontsize=11, pad=6,
            )
            a_p, x_p = app_state["frames_pred"][frame_idx]
            ax_right.plot(a_p, x_p, ",", color="#79c0ff", alpha=0.9, rasterized=True)
        else:  # bifurc
            ax_left.set_title(
                f"Bifurkacny diagram  ·  frame {frame_idx + 1}/{n}",
                color=TEXT_COLOR, fontsize=11, pad=6,
            )
            ax_left.plot(a_r, x_r, ",", color=COLOR_POINTS, alpha=0.9, rasterized=True)

        fig.canvas.draw_idle()

    # ── Generovanie dat a animacie ──────────────────────────────────────────────

    def _ensure_mlp() -> None:
        """Natrenuje NN ak este nie je v cache."""
        if app_state["mlp"] is not None:
            return
        _set_status("Generujem trening data...", color=ACCENT)
        plt.pause(0.01)
        X, y = make_training_pairs(n_samples=NN_TRAIN_N, seed=NN_SEED)

        mlp = MLP(NN_LAYERS, seed=NN_SEED)

        # Tréning po davkach – status sa prepisuje
        _set_status(f"Trenujem NN ({NN_EPOCHS} epoch)...", color=ACCENT)
        plt.pause(0.01)

        def _on_epoch(ep: int, mse: float) -> None:
            if (ep + 1) % 5 == 0 or ep == 0:
                _set_status(
                    f"Trenujem NN  epoch {ep + 1}/{NN_EPOCHS}  MSE={mse:.5f}",
                    color=ACCENT,
                )
                plt.pause(0.001)

        hist = mlp.train(
            X, y,
            epochs=NN_EPOCHS, batch_size=NN_BATCH, lr=NN_LR,
            on_epoch=_on_epoch,
        )
        app_state["mlp"]        = mlp
        app_state["train_hist"] = hist

    def _build_real_data() -> None:
        if app_state["a_full"] is None:
            _set_status("Pocitam bifurkacny diagram...", color=ACCENT)
            plt.pause(0.01)
            a_vals = np.linspace(0.0, 4.0, N_SAMPLES)
            a_arr, x_arr = compute_bifurcation(
                a_vals, n_iter=N_ITER, n_transient=N_TRANSIENT,
            )
            app_state["a_full"] = a_arr
            app_state["x_full"] = x_arr

    def _build_pred_data() -> None:
        a_vals = np.linspace(0.0, 4.0, N_SAMPLES)
        _set_status("Pocitam NN predikciu...", color=ACCENT)
        plt.pause(0.01)
        a_p, x_p = predict_bifurcation_onestep(
            app_state["mlp"], a_vals,
            n_iter=N_ITER, n_transient=N_TRANSIENT,
        )
        app_state["a_pred"] = a_p
        app_state["x_pred"] = x_p

    def _start_animation(mode: str, color: str) -> None:
        """Pripravi framy podla modu a spusti animaciu."""
        if app_state["anim_obj"] is not None:
            try:
                app_state["anim_obj"].event_source.stop()
            except Exception:
                pass

        # Pripravit data podla modu
        _build_real_data()
        if mode in ("split", "overlay"):
            _ensure_mlp()
            _build_pred_data()

        # Vyrobit kumulativne framy (zlava doprava)
        frames_real = _split_into_frames(
            app_state["a_full"], app_state["x_full"], ANIM_FRAMES,
        )
        frames_pred: list = []
        if mode in ("split", "overlay"):
            frames_pred = _split_into_frames(
                app_state["a_pred"], app_state["x_pred"], ANIM_FRAMES,
            )

        app_state["mode"]        = mode
        app_state["frames_real"] = frames_real
        app_state["frames_pred"] = frames_pred
        app_state["frame"]       = len(frames_real) - 1
        app_state["total"]       = len(frames_real)
        app_state["playing"]     = False
        app_state["color"]       = color

        # Slider
        n = len(frames_real)
        slider.valmin = 0
        slider.valmax = max(n - 1, 1)
        slider.set_val(n - 1)
        slider.ax.set_xlim(0, max(n - 1, 1))

        _layout_for_mode(mode)
        _draw_frame(n - 1)

        # Info update
        if mode == "bifurc":
            _update_info(
                "Mod: Bifurkacny diagram\n\n"
                "Vykreslujeme atraktor\n"
                "logistickej mapy pre\n"
                "vsetky a v [0, 4].\n\n"
                f"Samples a:   {N_SAMPLES}\n"
                f"Iteracii:    {N_ITER}\n"
                f"Tranzient:   {N_TRANSIENT}\n"
                f"Bodov spolu: {len(app_state['a_full']):,}\n\n"
                "Period-doubling start:\n"
                "  a ~ 3.0\n"
                "Chaos zacina priblizne:\n"
                "  a ~ 3.57\n\n"
                "Animaciou pridavame\n"
                "stlpce zlava doprava."
            )
            msg = f"Bifurkacny diagram hotovy ({len(app_state['a_full']):,} bodov)."
        elif mode == "split":
            mse = app_state["train_hist"][-1] if app_state["train_hist"] else 0.0
            _update_info(
                "Mod: Trening + Predikcia\n\n"
                "Vlavo:  realny atraktor\n"
                "        (training data)\n"
                "Vpravo: NN predikcia\n"
                "        x_{n+1} = NN(a, x_n)\n"
                "        z bodov atraktora.\n\n"
                f"NN: {NN_LAYERS}\n"
                f"  Adam lr={NN_LR}\n"
                f"  {NN_EPOCHS} epoch\n"
                f"  Final MSE: {mse:.6f}\n\n"
                "V stabilnych oblastiach\n"
                "NN replikuje atraktor.\n"
                "V chaose vznika rozptyl."
            )
            msg = (f"Hotovo. NN MSE={mse:.5f}, "
                   f"bodov: real {len(app_state['a_full']):,}, "
                   f"pred {len(app_state['a_pred']):,}.")
        else:  # overlay
            mse = app_state["train_hist"][-1] if app_state["train_hist"] else 0.0
            _update_info(
                "Mod: Predikcia (overlay)\n\n"
                "Sive body: skutocna\n"
                "  bifurkacia (training).\n"
                "Cervene body: NN predikcia\n"
                "  x_{n+1} pre kazdy realny\n"
                "  bod atraktora.\n\n"
                f"NN: {NN_LAYERS}\n"
                f"  Final MSE: {mse:.6f}\n\n"
                "V stabilnych oblastiach\n"
                "cervena prekryva sivu.\n"
                "V chaose vidno odchylky\n"
                "(Example 3 z PDF)."
            )
            msg = f"Overlay hotovy. NN MSE={mse:.5f}."

        _set_status(msg, color=color)

        anim_obj = anim_mod.FuncAnimation(
            fig, _anim_step, interval=120,
            blit=False, cache_frame_data=False,
        )
        app_state["anim_obj"] = anim_obj
        fig._anim = anim_obj

    # ── Obsluha animacie ────────────────────────────────────────────────────────

    def _set_frame(fi: int) -> None:
        n = len(app_state["frames_real"])
        if n == 0:
            return
        fi = int(np.clip(fi, 0, n - 1))
        app_state["frame"]       = fi
        app_state["_upd_slider"] = True
        slider.set_val(fi)
        app_state["_upd_slider"] = False
        _draw_frame(fi)

        if fi == n - 1:
            _set_status(
                f"Kompletny diagram ({n}/{n} framov).",
                color=app_state["color"],
            )
        else:
            _set_status(f"Frame {fi + 1} / {n}")

    def _anim_step(_fn: int) -> list:
        if not app_state["playing"]:
            return []
        nxt = app_state["frame"] + 1
        n = len(app_state["frames_real"])
        if nxt >= n:
            nxt = n - 1
            app_state["playing"] = False
            btn_play.label.set_text("▶ Play")
        _set_frame(nxt)
        return []

    def on_slider(val: float) -> None:
        if app_state["_upd_slider"]:
            return
        fi = int(val)
        app_state["frame"] = fi
        _draw_frame(fi)

    def on_play(event) -> None:
        if not app_state["frames_real"]:
            return
        if app_state["playing"]:
            app_state["playing"] = False
            btn_play.label.set_text("▶ Play")
        else:
            if app_state["frame"] >= len(app_state["frames_real"]) - 1:
                app_state["frame"] = 0
            app_state["playing"] = True
            btn_play.label.set_text("⏸ Pause")

    def on_prev(event) -> None:
        if not app_state["frames_real"]:
            return
        app_state["playing"] = False
        btn_play.label.set_text("▶ Play")
        _set_frame(app_state["frame"] - 1)

    def on_next(event) -> None:
        if not app_state["frames_real"]:
            return
        app_state["playing"] = False
        btn_play.label.set_text("▶ Play")
        _set_frame(app_state["frame"] + 1)

    def on_restart(event) -> None:
        if not app_state["frames_real"]:
            return
        app_state["playing"] = False
        btn_play.label.set_text("▶ Play")
        _set_frame(0)

    slider.on_changed(on_slider)
    btn_play.on_clicked(on_play)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    btn_restart.on_clicked(on_restart)

    # ── Obsluha modov ───────────────────────────────────────────────────────────

    def on_bifurc(event):
        _start_animation("bifurc", COLOR_BIF)

    def on_split(event):
        _start_animation("split", COLOR_NN)

    def on_overlay(event):
        _start_animation("overlay", COLOR_OVERLAY)

    def on_clear(event) -> None:
        if app_state["anim_obj"] is not None:
            try:
                app_state["anim_obj"].event_source.stop()
            except Exception:
                pass
        app_state["mode"]        = None
        app_state["frames_real"] = []
        app_state["frames_pred"] = []
        app_state["frame"]       = 0
        app_state["total"]       = 0
        app_state["playing"]     = False
        app_state["anim_obj"]    = None
        btn_play.label.set_text("▶ Play")

        _layout_for_mode(None)
        _draw_clear()
        _update_info(info_default)
        _set_status("Canvas vymazany.")

    btn_bifurc.on_clicked(on_bifurc)
    btn_split.on_clicked(on_split)
    btn_overlay.on_clicked(on_overlay)
    btn_clear.on_clicked(on_clear)

    # ── Pociatocny stav ─────────────────────────────────────────────────────────
    _draw_clear()
    plt.show()
