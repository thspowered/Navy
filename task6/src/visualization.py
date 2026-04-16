"""
visualization.py – Interaktivna vizualizacia Task 6: L-systems.

Rozlozenie figury (15x9 palcov, tmava tema):
  Vlavo  (60%) – canvas pre vykreslenie L-systemu
  Vpravo (35%) – tlacidla (4 presety + Draw custom + Clear) + vstupne polia
  Dolu         – animacne ovladacie prvky (slider, Play/Pause, Prev/Next, Restart)

Farebna schema (identicka s predchadzajucimi taskmi):
  BG_DARK  = "#0d1117"
  BG_PANEL = "#161b22"
  BG_AXES  = "#1c2333"
  TEXT     = "#f0f6fc"
  ACCENT   = "#58a6ff"
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim_mod
from matplotlib.widgets import Button, Slider, TextBox
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

from .lsystem import generate_string, compute_segments, PRESETS

# ── Farby temy (identicky s predchadzajucimi taskmi) ───────────────────────────
BG_DARK    = "#0d1117"
BG_PANEL   = "#161b22"
BG_AXES    = "#1c2333"
TEXT_COLOR = "#f0f6fc"
ACCENT     = "#58a6ff"

# Farby tlacidiel presetov
COLOR_P1     = "#56d364"   # zelena
COLOR_P2     = "#bc8cff"   # fialova
COLOR_P3     = "#e3b341"   # zlta
COLOR_P4     = "#ff7b72"   # cervena
COLOR_CUSTOM = "#58a6ff"   # modra – custom
COLOR_CLEAR  = "#8b949e"   # seda – clear

# Farba ciar fraktalu
FRACTAL_COLORS = {
    1: "#56d364",
    2: "#bc8cff",
    3: "#e3b341",
    4: "#ff7b72",
    "custom": "#58a6ff",
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


def run_app() -> None:
    """Spusti hlavnu interaktivnu aplikaciu Task 6 – L-systems."""
    _apply_dark_theme()

    # Stav aplikacie
    app_state: dict = {
        "segments":    [],       # list[(x0,y0,x1,y1)]
        "frame":       0,        # aktualny frame animacie
        "total":       0,        # celkovy pocet segmentov
        "playing":     False,
        "_upd_slider": False,
        "anim_obj":    None,
        "color":       ACCENT,   # farba aktualneho fraktalu
        "lsys_string": "",       # vygenerovany retazec
        "preset_id":   None,
    }

    # ── Figura ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 9), facecolor=BG_DARK)
    fig.suptitle(
        "Task 6 – L-systems  ·  Lindenmayer Fractals",
        color=TEXT_COLOR, fontsize=14, fontweight="bold", y=0.97,
    )

    # Oddelovacia cara nad animacnymi ovladacmi
    fig.add_artist(
        Line2D([0.03, 0.97], [0.275, 0.275],
               transform=fig.transFigure,
               color=ACCENT, linewidth=0.8, alpha=0.5)
    )

    # ── Osa pre fraktal (vlavo) ─────────────────────────────────────────────────
    ax_frac = fig.add_axes([0.03, 0.30, 0.56, 0.62])
    ax_frac.set_facecolor(BG_DARK)
    ax_frac.set_xticks([])
    ax_frac.set_yticks([])
    for spine in ax_frac.spines.values():
        spine.set_edgecolor(BG_PANEL)

    # ── Tlacidla presetov (vpravo) ──────────────────────────────────────────────
    btn_p1_ax = fig.add_axes([0.63, 0.86, 0.33, 0.055])
    btn_p2_ax = fig.add_axes([0.63, 0.80, 0.33, 0.055])
    btn_p3_ax = fig.add_axes([0.63, 0.74, 0.33, 0.055])
    btn_p4_ax = fig.add_axes([0.63, 0.68, 0.33, 0.055])

    btn_p1 = Button(btn_p1_ax, "1: Carpet (90°)",          color=BG_PANEL, hovercolor="#1f3a1f")
    btn_p2 = Button(btn_p2_ax, "2: Koch Snowflake (60°)",  color=BG_PANEL, hovercolor="#2a1f3a")
    btn_p3 = Button(btn_p3_ax, "3: Weed 1 (pi/7)",         color=BG_PANEL, hovercolor="#3a2e10")
    btn_p4 = Button(btn_p4_ax, "4: Weed 2 (pi/8)",         color=BG_PANEL, hovercolor="#3a1515")

    btn_p1.label.set_color(COLOR_P1)
    btn_p2.label.set_color(COLOR_P2)
    btn_p3.label.set_color(COLOR_P3)
    btn_p4.label.set_color(COLOR_P4)

    for btn in (btn_p1, btn_p2, btn_p3, btn_p4):
        btn.label.set_fontsize(10)
        btn.label.set_fontweight("bold")

    # ── Custom vstupy ──────────────────────────────────────────────────────────
    lbl_style = dict(fontsize=8, family="monospace", color=TEXT_COLOR,
                     ha="left", va="bottom")

    fig.text(0.63, 0.645, "Custom", color=ACCENT, fontsize=9,
             fontweight="bold", family="monospace")

    fig.text(0.63, 0.620, "Axiom (F,+,-,[,]):", **lbl_style)
    tb_axiom_ax = fig.add_axes([0.63, 0.595, 0.33, 0.025])
    tb_axiom = TextBox(tb_axiom_ax, "", initial="F+F+F+F",
                       color=BG_PANEL, hovercolor="#1c2333")
    tb_axiom.label.set_color(TEXT_COLOR)
    tb_axiom_ax.texts[0].set_color(TEXT_COLOR) if tb_axiom_ax.texts else None

    fig.text(0.63, 0.575, "Rule  F -> ...:", **lbl_style)
    tb_rule_ax = fig.add_axes([0.63, 0.550, 0.33, 0.025])
    tb_rule = TextBox(tb_rule_ax, "", initial="F+F-F-FF+F+F-F",
                      color=BG_PANEL, hovercolor="#1c2333")

    fig.text(0.63, 0.530, "Angle (degrees):", **lbl_style)
    tb_angle_ax = fig.add_axes([0.63, 0.505, 0.15, 0.025])
    tb_angle = TextBox(tb_angle_ax, "", initial="90",
                       color=BG_PANEL, hovercolor="#1c2333")

    fig.text(0.80, 0.530, "Nesting:", **lbl_style)
    tb_nest_ax = fig.add_axes([0.80, 0.505, 0.16, 0.025])
    tb_nest = TextBox(tb_nest_ax, "", initial="3",
                      color=BG_PANEL, hovercolor="#1c2333")

    for tb_ax in (tb_axiom_ax, tb_rule_ax, tb_angle_ax, tb_nest_ax):
        for spine in tb_ax.spines.values():
            spine.set_edgecolor(ACCENT)
            spine.set_linewidth(0.8)

    # Nastav farbu textu v textboxoch
    for tb in (tb_axiom, tb_rule, tb_angle, tb_nest):
        tb.text_disp.set_color(TEXT_COLOR)

    btn_custom_ax = fig.add_axes([0.63, 0.455, 0.155, 0.040])
    btn_clear_ax  = fig.add_axes([0.80, 0.455, 0.16,  0.040])

    btn_custom = Button(btn_custom_ax, "Draw custom", color=BG_PANEL, hovercolor="#1a2a4a")
    btn_clear  = Button(btn_clear_ax,  "Clear canvas", color=BG_PANEL, hovercolor="#2a2a2a")

    btn_custom.label.set_color(COLOR_CUSTOM)
    btn_clear.label.set_color(COLOR_CLEAR)
    btn_custom.label.set_fontsize(9)
    btn_custom.label.set_fontweight("bold")
    btn_clear.label.set_fontsize(9)
    btn_clear.label.set_fontweight("bold")

    # Informacny panel (vpravo dolu)
    ax_info = fig.add_axes([0.63, 0.30, 0.33, 0.145])
    ax_info.set_facecolor(BG_PANEL)
    ax_info.set_xticks([])
    ax_info.set_yticks([])
    for spine in ax_info.spines.values():
        spine.set_edgecolor(ACCENT)
        spine.set_linewidth(1.2)

    info_text = ax_info.text(
        0.5, 0.5,
        "Zvolte L-system (1-4)\n"
        "alebo zadajte vlastny.\n\n"
        "Symboly:\n"
        "  F = kresli vpred\n"
        "  b = posun bez kresby\n"
        "  + = otoc vpravo\n"
        "  - = otoc vlavo\n"
        "  [ = uloz poziciu\n"
        "  ] = obnov poziciu",
        transform=ax_info.transAxes,
        ha="center", va="center",
        fontsize=7.5, family="monospace",
        color=TEXT_COLOR,
    )

    # ── Animacne ovladacie prvky (dolu) ─────────────────────────────────────────
    slider_ax   = fig.add_axes([0.10, 0.18, 0.80, 0.03], facecolor=BG_PANEL)
    btn_prev_ax = fig.add_axes([0.33, 0.09, 0.07, 0.05], facecolor=BG_PANEL)
    btn_play_ax = fig.add_axes([0.41, 0.09, 0.12, 0.05], facecolor=BG_PANEL)
    btn_next_ax = fig.add_axes([0.54, 0.09, 0.07, 0.05], facecolor=BG_PANEL)
    btn_rst_ax  = fig.add_axes([0.62, 0.09, 0.09, 0.05], facecolor=BG_PANEL)

    slider = Slider(
        slider_ax, "Seg", 0, 1,
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
        "Kliknite na tlacidlo L-systemu alebo zadajte vlastny.",
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
        """Vykreslí L-system segmenty az po dany frame."""
        ax_frac.clear()
        ax_frac.set_facecolor(BG_DARK)
        ax_frac.set_xticks([])
        ax_frac.set_yticks([])
        for spine in ax_frac.spines.values():
            spine.set_edgecolor(BG_PANEL)

        segs = app_state["segments"]
        if not segs:
            ax_frac.set_title("L-system", color=TEXT_COLOR, fontsize=12, pad=8)
            fig.canvas.draw_idle()
            return

        if frame is None:
            frame = app_state["frame"]
        frame = int(np.clip(frame, 0, len(segs) - 1))

        visible = segs[:frame + 1]
        lines = [[(s[0], s[1]), (s[2], s[3])] for s in visible]

        color = app_state["color"]
        # Gradient: staršie segmenty tmavšie, novšie jasnejšie
        n = len(lines)
        alphas = np.linspace(0.3, 1.0, n)
        colors = []
        import matplotlib.colors as mcolors
        rgb = mcolors.to_rgb(color)
        for a in alphas:
            colors.append((*rgb, a))

        lc = LineCollection(lines, colors=colors, linewidths=0.8)
        ax_frac.add_collection(lc)

        # Bounding box zo vsetkych segmentov (nie len viditelnych)
        all_xs = [s[0] for s in segs] + [s[2] for s in segs]
        all_ys = [s[1] for s in segs] + [s[3] for s in segs]
        xmin, xmax = min(all_xs), max(all_xs)
        ymin, ymax = min(all_ys), max(all_ys)
        dx = (xmax - xmin) * 0.05 + 1
        dy = (ymax - ymin) * 0.05 + 1
        ax_frac.set_xlim(xmin - dx, xmax + dx)
        ax_frac.set_ylim(ymin - dy, ymax + dy)
        ax_frac.set_aspect("equal")

        # Info v rohu
        ax_frac.text(
            0.99, 0.02,
            f"Segment: {frame + 1} / {len(segs)}",
            transform=ax_frac.transAxes,
            ha="right", va="bottom",
            fontsize=9, family="monospace", color=TEXT_COLOR,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_PANEL,
                      edgecolor=ACCENT, alpha=0.88),
        )

        name = ""
        pid = app_state["preset_id"]
        if pid and pid in PRESETS:
            name = PRESETS[pid]["name"]
        elif pid == "custom":
            name = "Custom"
        title = f"L-system  –  {name}" if name else "L-system"
        ax_frac.set_title(title, color=TEXT_COLOR, fontsize=12, pad=8)

        fig.canvas.draw_idle()

    # ── Generovanie a spustenie animacie ────────────────────────────────────────

    def _run_lsystem(
        axiom: str,
        rules: dict[str, str],
        angle: float,
        nesting: int,
        step: float,
        color: str,
        preset_id,
        initial_angle: float = 0.0,
    ) -> None:
        """Vygeneruje L-system a spusti animaciu."""
        # Zastavenie predchadzajucej animacie
        if app_state["anim_obj"] is not None:
            try:
                app_state["anim_obj"].event_source.stop()
            except Exception:
                pass

        _set_status("Generujem L-system retazec...", color=ACCENT)
        fig.canvas.draw_idle()
        plt.pause(0.01)

        lsys_str = generate_string(axiom, rules, nesting)
        app_state["lsys_string"] = lsys_str

        _set_status(f"Pocitam segmenty... (retazec: {len(lsys_str)} znakov)", color=ACCENT)
        fig.canvas.draw_idle()
        plt.pause(0.01)

        segments, xmin, ymin, xmax, ymax = compute_segments(
            lsys_str, angle, step, initial_angle=initial_angle,
        )

        if not segments:
            _set_status("Ziadne segmenty na vykreslenie!", color=COLOR_P4)
            return

        app_state["segments"]  = segments
        app_state["frame"]     = len(segments) - 1
        app_state["total"]     = len(segments)
        app_state["playing"]   = False
        app_state["color"]     = color
        app_state["preset_id"] = preset_id

        n = len(segments)
        slider.valmin = 0
        slider.valmax = max(n - 1, 1)
        slider.set_val(n - 1)
        slider.ax.set_xlim(0, max(n - 1, 1))

        angle_deg = math.degrees(angle)
        _update_info(
            f"Axiom: {axiom}\n"
            f"Rule:  F -> {rules.get('F', '?')}\n"
            f"Angle: {angle_deg:.1f}°\n"
            f"Nesting: {nesting}\n\n"
            f"Retazec: {len(lsys_str)} znakov\n"
            f"Segmenty: {n}\n\n"
            f"Pouzite animacne ovladace\n"
            f"pre postupne vykreslovanie."
        )

        _set_status(
            f"Hotovo! {n} segmentov, retazec {len(lsys_str)} znakov.",
            color=color,
        )
        _draw_fractal(n - 1)

        # Priprav animaciu
        anim_obj = anim_mod.FuncAnimation(
            fig, _anim_step, interval=1,
            blit=False, cache_frame_data=False,
        )
        app_state["anim_obj"] = anim_obj
        fig._anim = anim_obj

    # ── Animacia ────────────────────────────────────────────────────────────────

    def _set_frame(fi: int) -> None:
        n = len(app_state["segments"])
        if n == 0:
            return
        fi = int(np.clip(fi, 0, n - 1))
        app_state["frame"]       = fi
        app_state["_upd_slider"] = True
        slider.set_val(fi)
        app_state["_upd_slider"] = False
        _draw_fractal(fi)

        if fi == n - 1:
            _set_status(f"Kompletne vykreslenie: {n} segmentov.",
                        color=app_state["color"])
        else:
            _set_status(f"Segment {fi + 1} / {n}")

    # Adaptivny krok pre animaciu – pri veľkom pocte segmentov preskoci
    def _get_anim_step_size() -> int:
        n = app_state["total"]
        if n < 500:
            return 1
        elif n < 2000:
            return 5
        elif n < 10000:
            return 20
        else:
            return 50

    def _anim_step(_fn: int) -> list:
        if not app_state["playing"]:
            return []
        step = _get_anim_step_size()
        nxt = app_state["frame"] + step
        n = len(app_state["segments"])
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
        if not app_state["segments"]:
            return
        if app_state["playing"]:
            app_state["playing"] = False
            btn_play.label.set_text("▶ Play")
        else:
            if app_state["frame"] >= len(app_state["segments"]) - 1:
                app_state["frame"] = 0
            app_state["playing"] = True
            btn_play.label.set_text("⏸ Pause")

    def on_prev(event) -> None:
        if not app_state["segments"]:
            return
        app_state["playing"] = False
        btn_play.label.set_text("▶ Play")
        step = _get_anim_step_size()
        _set_frame(app_state["frame"] - step)

    def on_next(event) -> None:
        if not app_state["segments"]:
            return
        app_state["playing"] = False
        btn_play.label.set_text("▶ Play")
        step = _get_anim_step_size()
        _set_frame(app_state["frame"] + step)

    def on_restart(event) -> None:
        if not app_state["segments"]:
            return
        app_state["playing"] = False
        btn_play.label.set_text("▶ Play")
        _set_frame(0)

    slider.on_changed(on_slider)
    btn_play.on_clicked(on_play)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    btn_restart.on_clicked(on_restart)

    # ── Obsluha presetov ────────────────────────────────────────────────────────

    def _make_preset_handler(pid: int):
        def handler(event):
            p = PRESETS[pid]
            ia = p.get("initial_angle", 0.0)
            _run_lsystem(
                p["axiom"], p["rules"], p["angle"],
                p["nesting"], p["step"],
                FRACTAL_COLORS[pid], pid,
                initial_angle=ia,
            )
        return handler

    btn_p1.on_clicked(_make_preset_handler(1))
    btn_p2.on_clicked(_make_preset_handler(2))
    btn_p3.on_clicked(_make_preset_handler(3))
    btn_p4.on_clicked(_make_preset_handler(4))

    # ── Obsluha custom ──────────────────────────────────────────────────────────

    def on_custom(event) -> None:
        axiom = tb_axiom.text.strip()
        rule  = tb_rule.text.strip()
        angle_str = tb_angle.text.strip()
        nest_str  = tb_nest.text.strip()

        if not axiom or not rule:
            _set_status("Zadajte axiom a pravidlo!", color=COLOR_P4)
            return
        try:
            angle_deg = float(angle_str)
        except ValueError:
            _set_status("Neplatny uhol!", color=COLOR_P4)
            return
        try:
            nesting = int(nest_str)
        except ValueError:
            _set_status("Neplatny nesting!", color=COLOR_P4)
            return

        if nesting < 0 or nesting > 8:
            _set_status("Nesting musi byt 0-8!", color=COLOR_P4)
            return

        rules = {"F": rule}
        angle_rad = math.radians(angle_deg)

        # Ak su branches, predpokladame rast hore
        has_branches = "[" in rule
        ia = math.pi / 2 if has_branches else 0.0

        _run_lsystem(
            axiom, rules, angle_rad, nesting, 5,
            FRACTAL_COLORS["custom"], "custom",
            initial_angle=ia,
        )

    btn_custom.on_clicked(on_custom)

    # ── Obsluha clear ───────────────────────────────────────────────────────────

    def on_clear(event) -> None:
        if app_state["anim_obj"] is not None:
            try:
                app_state["anim_obj"].event_source.stop()
            except Exception:
                pass

        app_state["segments"]  = []
        app_state["frame"]     = 0
        app_state["total"]     = 0
        app_state["playing"]   = False
        app_state["anim_obj"]  = None
        app_state["preset_id"] = None
        btn_play.label.set_text("▶ Play")

        _update_info(
            "Zvolte L-system (1-4)\n"
            "alebo zadajte vlastny.\n\n"
            "Symboly:\n"
            "  F = kresli vpred\n"
            "  b = posun bez kresby\n"
            "  + = otoc vpravo\n"
            "  - = otoc vlavo\n"
            "  [ = uloz poziciu\n"
            "  ] = obnov poziciu"
        )
        _set_status("Canvas vymazany.")
        _draw_fractal()

    btn_clear.on_clicked(on_clear)

    # ── Pociatocny stav ─────────────────────────────────────────────────────────
    _draw_fractal()
    plt.show()
