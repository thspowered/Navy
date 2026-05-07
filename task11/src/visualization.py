"""
visualization.py - Interaktivna vizualizacia Task 11: Dvojite kyvadlo
a chaoticky pohyb.

Rozlozenie figury (15x9 palcov, tmava tema):
  Vlavo (~57%) – kreslo s aktualnou polohou kyvadla(/iel) + stopa zavazia 2
  Vpravo (~35%) – tlacidla modov + info panel
  Dolu          – animacne ovladacie prvky (slider, Play/Pause, Prev/Next, Restart)

Farebna schema (identicka s predchadzajucimi taskmi):
  BG_DARK  = "#0d1117"
  BG_PANEL = "#161b22"
  BG_AXES  = "#1c2333"
  TEXT     = "#f0f6fc"
  ACCENT   = "#58a6ff"

Mody:
  1. "Standardne kyvadlo"   – jedno dvojite kyvadlo + stopa zavazia 2
  2. "Chaos (3 kyvadla)"    – tri kyvadla s drobne odlisnymi pociatocnymi uhlami,
                              demonstracia citlivosti na pociatocne podmienky
  3. "Bez stopy"            – jedno kyvadlo bez stopy (clean swing)
  4. "Clear"                – vymazat plochu

Animacia: pre-kalkulovana trajektoria, slider posuva cas. Play prehrava
realny pohyb krok po kroku.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim_mod
from matplotlib.widgets import Button, Slider
from matplotlib.lines import Line2D

from .pendulum import (
    integrate, positions, total_energy,
    L1_DEFAULT, L2_DEFAULT, M1_DEFAULT, M2_DEFAULT, G_DEFAULT,
)

# ── Farby temy (identicky s predchadzajucimi taskmi) ───────────────────────────
BG_DARK    = "#0d1117"
BG_PANEL   = "#161b22"
BG_AXES    = "#1c2333"
TEXT_COLOR = "#f0f6fc"
ACCENT     = "#58a6ff"

COLOR_SINGLE   = "#56d364"   # zelena – standardne kyvadlo
COLOR_CHAOS    = "#bc8cff"   # fialova – chaos (3 kyvadla)
COLOR_CLEAN    = "#f0b429"   # zlta   – bez stopy
COLOR_CLEAR    = "#8b949e"   # seda   – clear

# Farby kyvadiel pre chaos rezim (3 mierne odlisne pociatocne uhly)
CHAOS_COLORS = ["#ff5c5c", "#56d364", "#79c0ff"]

# Farby pre standardne kyvadlo
ROD_COLOR    = "#e6edf3"   # takmer biela – tyc
MASS1_COLOR  = "#f0b429"   # zlta – m1
MASS2_COLOR  = "#ff5c5c"   # cervena – m2
TRAIL_COLOR  = "#58a6ff"   # modra – stopa m2

# ── Parametre simulacie (default) ──────────────────────────────────────────────
SIM_DURATION = 20.0     # celkovy cas simulacie (s)
SIM_DT       = 0.02     # casovy krok (s)  -> 1001 framov pre 20s
TRAIL_LEN    = 250      # pocet poslednych poloh m2 v stope

# Pociatocne uhly z PDF: theta1 = 2*pi/6, theta2 = 5*pi/8
THETA1_INIT = 2.0 * np.pi / 6.0
THETA2_INIT = 5.0 * np.pi / 8.0

# Mala odchylka pre chaos demonstraciu (radiany, ~0.057 stupna)
CHAOS_DELTA = 0.001


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


def _simulate_set(
    initial_angles: list[tuple[float, float]],
    duration: float = SIM_DURATION,
    dt: float = SIM_DT,
    l1: float = L1_DEFAULT, l2: float = L2_DEFAULT,
    m1: float = M1_DEFAULT, m2: float = M2_DEFAULT,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
           list[np.ndarray]]:
    """
    Simuluje N dvojitych kyvadiel s roznymi pociatocnymi uhlami.

    Vystup:
        t_arr             – pole casov (n_steps,)
        positions_list    – pre kazde kyvadlo (x1, y1, x2, y2)
        energies_list     – pre kazde kyvadlo pole celkovej energie
    """
    t_arr = None
    positions_list = []
    energies_list  = []

    for (th1_0, th2_0) in initial_angles:
        state0 = np.array([th1_0, 0.0, th2_0, 0.0], dtype=float)
        t, state = integrate(state0, duration, dt, l1, l2, m1, m2)
        if t_arr is None:
            t_arr = t
        x1, y1, x2, y2 = positions(state, l1, l2)
        positions_list.append((x1, y1, x2, y2))
        energies_list.append(total_energy(state, l1, l2, m1, m2))

    return t_arr, positions_list, energies_list


# ──────────────────────────────────────────────────────────────────────────────
# Hlavna aplikacia
# ──────────────────────────────────────────────────────────────────────────────
def run_app() -> None:
    """Spusti hlavnu interaktivnu aplikaciu Task 11 – Double pendulum."""
    _apply_dark_theme()

    # ── Stav aplikacie ─────────────────────────────────────────────────────────
    app_state: dict = {
        "mode":          None,        # "single" | "chaos" | "clean" | None
        "t_arr":         None,        # pole casov
        "positions":     [],          # zoznam (x1, y1, x2, y2) pre kazde kyvadlo
        "energies":      [],          # celkova energia kazdeho kyvadla
        "n_steps":       0,
        "frame":         0,
        "playing":       False,
        "_upd_slider":   False,
        "anim_obj":      None,
        "color":         ACCENT,
        "show_trail":    True,
        "l_total":       L1_DEFAULT + L2_DEFAULT,
    }

    # ── Figura ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 9), facecolor=BG_DARK)
    fig.suptitle(
        "Task 11 – Double Pendulum  ·  Chaoticky pohyb",
        color=TEXT_COLOR, fontsize=14, fontweight="bold", y=0.97,
    )

    # Oddelovacia cara nad animacnymi ovladacmi
    fig.add_artist(
        Line2D([0.03, 0.97], [0.275, 0.275],
               transform=fig.transFigure,
               color=ACCENT, linewidth=0.8, alpha=0.5)
    )

    # ── Hlavne osi pre kreslo (vlavo) ──────────────────────────────────────────
    ax_main = fig.add_axes([0.04, 0.31, 0.55, 0.61])

    def _style_main_axes(title: str = "") -> None:
        ax_main.set_facecolor(BG_AXES)
        for spine in ax_main.spines.values():
            spine.set_edgecolor(BG_PANEL)
        ax_main.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax_main.set_xlabel("x [m]", color=TEXT_COLOR, fontsize=10)
        ax_main.set_ylabel("y [m]", color=TEXT_COLOR, fontsize=10)
        ax_main.grid(True, alpha=0.15, color=BG_PANEL)
        L = app_state["l_total"]
        ax_main.set_xlim(-L * 1.15, L * 1.15)
        ax_main.set_ylim(-L * 1.15, L * 1.15)
        ax_main.set_aspect("equal")
        if title:
            ax_main.set_title(title, color=TEXT_COLOR, fontsize=11, pad=6)

    _style_main_axes("Dvojite kyvadlo")

    # ── Tlacidla modov (vpravo) ────────────────────────────────────────────────
    btn_s_ax  = fig.add_axes([0.63, 0.86, 0.33, 0.055])
    btn_c_ax  = fig.add_axes([0.63, 0.80, 0.33, 0.055])
    btn_n_ax  = fig.add_axes([0.63, 0.74, 0.33, 0.055])
    btn_cl_ax = fig.add_axes([0.63, 0.68, 0.33, 0.055])

    btn_single = Button(btn_s_ax,  "Standardne kyvadlo",
                        color=BG_PANEL, hovercolor="#1f3a1f")
    btn_chaos  = Button(btn_c_ax,  "Chaos (3 kyvadla)",
                        color=BG_PANEL, hovercolor="#2a1f3a")
    btn_clean  = Button(btn_n_ax,  "Bez stopy",
                        color=BG_PANEL, hovercolor="#3a2f1f")
    btn_clear  = Button(btn_cl_ax, "Clear canvas",
                        color=BG_PANEL, hovercolor="#2a2a2a")

    btn_single.label.set_color(COLOR_SINGLE)
    btn_chaos.label.set_color(COLOR_CHAOS)
    btn_clean.label.set_color(COLOR_CLEAN)
    btn_clear.label.set_color(COLOR_CLEAR)
    for btn in (btn_single, btn_chaos, btn_clean, btn_clear):
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
        "Dvojite kyvadlo:\n"
        "  m1, m2  – hmotnosti zavazi\n"
        "  l1, l2  – dlzky vlakien\n"
        "  th1,th2 – uhly od vertikaly\n\n"
        "Pohybove rovnice (Lagrange):\n"
        "  Nelinearne ODR 2. radu\n"
        "  Riesime RK4 metodou\n\n"
        "Mody:\n"
        "  Standardne kyvadlo\n"
        "    – jedno + stopa zavazia 2\n"
        "  Chaos (3 kyvadla)\n"
        "    – mala zmena th1 (0.001 rad)\n"
        "      vyvolava velku divergenciu\n"
        "  Bez stopy\n"
        "    – jedno kyvadlo, ciste\n\n"
        "Pociatocne uhly:\n"
        f"  th1 = 2*pi/6 ({np.degrees(THETA1_INIT):5.1f} stupnov)\n"
        f"  th2 = 5*pi/8 ({np.degrees(THETA2_INIT):5.1f} stupnov)\n"
        f"  th1' = th2' = 0\n\n"
        f"Parametre: l1={L1_DEFAULT}, l2={L2_DEFAULT}\n"
        f"           m1={M1_DEFAULT}, m2={M2_DEFAULT}\n"
        f"           g={G_DEFAULT}, dt={SIM_DT}"
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

    slider = Slider(slider_ax, "Cas", 0, 1, valinit=0, valstep=1, color=ACCENT)
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
        "Vyber mod (Standardne kyvadlo / Chaos / Bez stopy).",
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

    def _draw_clear() -> None:
        """Vyprazdni hlavnu osu do startovacieho stavu."""
        ax_main.clear()
        _style_main_axes("Dvojite kyvadlo")
        # Maly bod v zavese
        ax_main.plot(0, 0, "o", color=TEXT_COLOR, markersize=4)

    def _draw_frame(frame_idx: int) -> None:
        """Vykresli aktualny frame – polohy kyvadiel + (volitelne) stopu."""
        positions_list = app_state["positions"]
        if not positions_list:
            return
        n = app_state["n_steps"]
        frame_idx = int(np.clip(frame_idx, 0, n - 1))
        mode = app_state["mode"]

        ax_main.clear()
        _style_main_axes()

        # Bod zavesu
        ax_main.plot(0, 0, "o", color=TEXT_COLOR, markersize=5,
                     markeredgecolor=BG_PANEL, zorder=5)

        if mode == "chaos":
            # Tri kyvadla, len telo (bez stop) – stopy by zhustili obraz
            for i, (x1, y1, x2, y2) in enumerate(positions_list):
                col = CHAOS_COLORS[i % len(CHAOS_COLORS)]
                xa = x1[frame_idx]; ya = y1[frame_idx]
                xb = x2[frame_idx]; yb = y2[frame_idx]
                # Stopa zavazia 2 (kratka, alpha gradient)
                start = max(0, frame_idx - TRAIL_LEN)
                if frame_idx > start:
                    ax_main.plot(x2[start:frame_idx + 1], y2[start:frame_idx + 1],
                                 "-", color=col, alpha=0.35, linewidth=1.0, zorder=2)
                # Tyc 1
                ax_main.plot([0, xa], [0, ya],
                             "-", color=col, linewidth=1.6, alpha=0.9, zorder=3)
                # Tyc 2
                ax_main.plot([xa, xb], [ya, yb],
                             "-", color=col, linewidth=1.6, alpha=0.9, zorder=3)
                # Hmoty
                ax_main.plot(xa, ya, "o", color=col, markersize=8,
                             markeredgecolor=BG_PANEL, zorder=4)
                ax_main.plot(xb, yb, "o", color=col, markersize=10,
                             markeredgecolor=BG_PANEL, zorder=4)

            t = app_state["t_arr"][frame_idx]
            ax_main.set_title(
                f"Chaos: 3 kyvadla, dth1 = {CHAOS_DELTA} rad  ·  "
                f"t = {t:5.2f} s  ·  frame {frame_idx + 1}/{n}",
                color=TEXT_COLOR, fontsize=11, pad=6,
            )
            # Mala legenda
            for i, col in enumerate(CHAOS_COLORS[:len(positions_list)]):
                ax_main.plot([], [], "o", color=col,
                             label=f"th1 + {i * CHAOS_DELTA:.3f} rad")
            leg = ax_main.legend(
                loc="upper left", fontsize=8, framealpha=0.85,
                facecolor=BG_PANEL, edgecolor=ACCENT,
            )
            for txt in leg.get_texts():
                txt.set_color(TEXT_COLOR)
        else:
            # Jedno kyvadlo – standardne / clean
            x1, y1, x2, y2 = positions_list[0]
            xa = x1[frame_idx]; ya = y1[frame_idx]
            xb = x2[frame_idx]; yb = y2[frame_idx]

            # Stopa zavazia 2 (len v "single" mode, nie v "clean")
            if app_state["show_trail"]:
                start = max(0, frame_idx - TRAIL_LEN)
                if frame_idx > start:
                    # Gradient alpha – starsie body bledsie
                    seg_x = x2[start:frame_idx + 1]
                    seg_y = y2[start:frame_idx + 1]
                    ax_main.plot(seg_x, seg_y, "-",
                                 color=TRAIL_COLOR, alpha=0.55, linewidth=1.2,
                                 zorder=2)

            # Tyce
            ax_main.plot([0, xa], [0, ya], "-",
                         color=ROD_COLOR, linewidth=2.0, zorder=3)
            ax_main.plot([xa, xb], [ya, yb], "-",
                         color=ROD_COLOR, linewidth=2.0, zorder=3)
            # Hmoty
            ax_main.plot(xa, ya, "o",
                         color=MASS1_COLOR, markersize=11,
                         markeredgecolor=BG_PANEL, zorder=4)
            ax_main.plot(xb, yb, "o",
                         color=MASS2_COLOR, markersize=13,
                         markeredgecolor=BG_PANEL, zorder=4)

            t = app_state["t_arr"][frame_idx]
            mode_lbl = "Standardne kyvadlo" if mode == "single" else "Bez stopy"
            ax_main.set_title(
                f"{mode_lbl}  ·  t = {t:5.2f} s  ·  frame {frame_idx + 1}/{n}",
                color=TEXT_COLOR, fontsize=11, pad=6,
            )

        fig.canvas.draw_idle()

    # ── Generovanie dat a animacie ──────────────────────────────────────────────

    def _start_animation(mode: str, color: str) -> None:
        """Pripravi simulaciu podla modu a spusti animaciu."""
        if app_state["anim_obj"] is not None:
            try:
                app_state["anim_obj"].event_source.stop()
            except Exception:
                pass

        _set_status("Pocitam simulaciu (RK4)...", color=ACCENT)
        plt.pause(0.01)

        if mode == "chaos":
            initials = [
                (THETA1_INIT + i * CHAOS_DELTA, THETA2_INIT)
                for i in range(3)
            ]
            show_trail = True
        else:  # "single" alebo "clean"
            initials = [(THETA1_INIT, THETA2_INIT)]
            show_trail = (mode == "single")

        t_arr, pos_list, energies = _simulate_set(initials)

        app_state["mode"]        = mode
        app_state["t_arr"]       = t_arr
        app_state["positions"]   = pos_list
        app_state["energies"]    = energies
        app_state["n_steps"]     = len(t_arr)
        app_state["frame"]       = 0
        app_state["playing"]     = False
        app_state["color"]       = color
        app_state["show_trail"]  = show_trail

        # Slider rozsah
        n = len(t_arr)
        slider.valmin = 0
        slider.valmax = max(n - 1, 1)
        slider.set_val(0)
        slider.ax.set_xlim(0, max(n - 1, 1))

        _draw_frame(0)

        # Info update + status
        e = energies[0]
        e_drift = float(np.max(np.abs(e - e[0])))
        if mode == "single":
            _update_info(
                "Mod: Standardne kyvadlo\n\n"
                "Jedno dvojite kyvadlo,\n"
                "stopa zavazia 2 (modra)\n"
                f"poslednych {TRAIL_LEN} bodov.\n\n"
                f"Pociatok:\n"
                f"  th1 = 2*pi/6\n"
                f"  th2 = 5*pi/8\n"
                f"  th1' = th2' = 0\n\n"
                f"Cas simulacie: {SIM_DURATION:.0f} s\n"
                f"dt = {SIM_DT}, framov: {n}\n\n"
                f"Energia (sanity check):\n"
                f"  E0    = {e[0]:7.3f}\n"
                f"  drift = {e_drift:7.5f}\n"
                f"  (RK4 zachovava E)"
            )
            msg = f"Hotovo. {n} framov, energy drift = {e_drift:.5f} J."
        elif mode == "chaos":
            _update_info(
                "Mod: Chaos (3 kyvadla)\n\n"
                "Tri kyvadla zacinaju\n"
                "skoro identicky:\n"
                f"  th1, th1+{CHAOS_DELTA},\n"
                f"  th1+{2*CHAOS_DELTA}\n"
                f"  (rozdiel ~{np.degrees(CHAOS_DELTA):.3f} stupna)\n\n"
                "Po par sekundach\n"
                "trajektorie divergencne\n"
                "rozisli sa - typicky\n"
                "znak chaotickeho systemu\n"
                "(citlivost na pociatocne\n"
                "podmienky).\n\n"
                f"Cas: {SIM_DURATION:.0f} s,  framov: {n}"
            )
            msg = f"Hotovo. 3 kyvadla, {n} framov, dth1 = {CHAOS_DELTA} rad."
        else:  # clean
            _update_info(
                "Mod: Bez stopy\n\n"
                "Jedno dvojite kyvadlo,\n"
                "len ramena a hmoty.\n"
                "Cisty pohlad na pohyb.\n\n"
                f"Pociatok:\n"
                f"  th1 = 2*pi/6\n"
                f"  th2 = 5*pi/8\n\n"
                f"Cas simulacie: {SIM_DURATION:.0f} s\n"
                f"dt = {SIM_DT}, framov: {n}\n\n"
                f"Energia (sanity check):\n"
                f"  E0    = {e[0]:7.3f}\n"
                f"  drift = {e_drift:7.5f}"
            )
            msg = f"Hotovo. {n} framov bez stopy."

        _set_status(msg, color=color)

        anim_obj = anim_mod.FuncAnimation(
            fig, _anim_step, interval=30,
            blit=False, cache_frame_data=False,
        )
        app_state["anim_obj"] = anim_obj
        fig._anim = anim_obj

    # ── Obsluha animacie ────────────────────────────────────────────────────────

    def _set_frame(fi: int) -> None:
        n = app_state["n_steps"]
        if n == 0:
            return
        fi = int(np.clip(fi, 0, n - 1))
        app_state["frame"]       = fi
        app_state["_upd_slider"] = True
        slider.set_val(fi)
        app_state["_upd_slider"] = False
        _draw_frame(fi)

        t = app_state["t_arr"][fi]
        if fi == n - 1:
            _set_status(
                f"Koniec simulacie  ·  t = {t:5.2f} s  ·  frame {fi + 1}/{n}",
                color=app_state["color"],
            )
        else:
            _set_status(
                f"t = {t:5.2f} s  ·  frame {fi + 1} / {n}"
            )

    def _anim_step(_fn: int) -> list:
        if not app_state["playing"]:
            return []
        nxt = app_state["frame"] + 1
        n = app_state["n_steps"]
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
        if app_state["n_steps"] == 0:
            return
        if app_state["playing"]:
            app_state["playing"] = False
            btn_play.label.set_text("▶ Play")
        else:
            if app_state["frame"] >= app_state["n_steps"] - 1:
                app_state["frame"] = 0
            app_state["playing"] = True
            btn_play.label.set_text("⏸ Pause")

    def on_prev(event) -> None:
        if app_state["n_steps"] == 0:
            return
        app_state["playing"] = False
        btn_play.label.set_text("▶ Play")
        _set_frame(app_state["frame"] - 1)

    def on_next(event) -> None:
        if app_state["n_steps"] == 0:
            return
        app_state["playing"] = False
        btn_play.label.set_text("▶ Play")
        _set_frame(app_state["frame"] + 1)

    def on_restart(event) -> None:
        if app_state["n_steps"] == 0:
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

    def on_single(event):
        _start_animation("single", COLOR_SINGLE)

    def on_chaos(event):
        _start_animation("chaos", COLOR_CHAOS)

    def on_clean(event):
        _start_animation("clean", COLOR_CLEAN)

    def on_clear(event) -> None:
        if app_state["anim_obj"] is not None:
            try:
                app_state["anim_obj"].event_source.stop()
            except Exception:
                pass
        app_state["mode"]      = None
        app_state["t_arr"]     = None
        app_state["positions"] = []
        app_state["energies"]  = []
        app_state["n_steps"]   = 0
        app_state["frame"]     = 0
        app_state["playing"]   = False
        app_state["anim_obj"]  = None
        btn_play.label.set_text("▶ Play")

        _draw_clear()
        _update_info(info_default)
        _set_status("Canvas vymazany.")

    btn_single.on_clicked(on_single)
    btn_chaos.on_clicked(on_chaos)
    btn_clean.on_clicked(on_clean)
    btn_clear.on_clicked(on_clear)

    # ── Pociatocny stav ─────────────────────────────────────────────────────────
    _draw_clear()
    plt.show()
