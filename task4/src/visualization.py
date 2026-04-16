"""
visualization.py – Interaktivna vizualizacia Q-learning hry 'Najdi syr'.

Rozlozenie figury (15x9 palcov, tmava tema):
  Vlavo  (60%) – interaktivna mriezka 8x8
  Vpravo (35%) – tlacidla na ovladanie
  Dolu         – animacne ovladacie prvky (slider, Play/Pause, Prev/Next, Restart)

Farebna schema (identicky s predchadzajucimi taskmi):
  BG_DARK  = "#0d1117"
  BG_PANEL = "#161b22"
  BG_AXES  = "#1c2333"
  TEXT     = "#f0f6fc"
  ACCENT   = "#58a6ff"
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim_mod
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, Slider
from matplotlib.lines import Line2D

from .q_learning import QLearningAgent, CellType, Action, ACTION_DELTAS

# ── Farby temy (identicky s predchadzajucimi taskmi) ───────────────────────────
BG_DARK    = "#0d1117"
BG_PANEL   = "#161b22"
BG_AXES    = "#1c2333"
TEXT_COLOR = "#f0f6fc"
ACCENT     = "#58a6ff"

# Farby buniek
COLOR_EMPTY  = "#1c2333"
COLOR_MOUSE  = "#58a6ff"   # modra  – mys
COLOR_CHEESE = "#e3b341"   # zlta   – syr
COLOR_TRAP   = "#ff7b72"   # cervena – pasca
COLOR_WALL   = "#484f58"   # seda   – stena
COLOR_PATH   = "#2d5a27"   # tmavo zelena – navstivena cesta

# Farby tlacidiel
COLOR_LEARN      = "#56d364"   # zelena
COLOR_FIND       = "#e3b341"   # zlta
COLOR_SEL_MOUSE  = "#58a6ff"   # modra
COLOR_SEL_TRAP   = "#ff7b72"   # cervena
COLOR_SEL_WALL   = "#8b949e"   # seda
COLOR_SEL_CHEESE = "#e3b341"   # zlta
COLOR_MATRIX_BTN = "#bc8cff"   # fialova
COLOR_CLEAR      = "#ff7b72"   # cervena

# Rozmery mriezky
GRID_ROWS = 8
GRID_COLS = 8

# Pocet epizod trenovania
N_EPISODES = 1000


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


def _cell_color(ct: int) -> str:
    """Vrati farbu pre dany typ bunky."""
    if ct == int(CellType.MOUSE):   return COLOR_MOUSE
    if ct == int(CellType.CHEESE):  return COLOR_CHEESE
    if ct == int(CellType.TRAP):    return COLOR_TRAP
    if ct == int(CellType.WALL):    return COLOR_WALL
    return COLOR_EMPTY


def _cell_symbol(ct: int) -> str:
    """Vrati zobrazovany symbol pre dany typ bunky."""
    if ct == int(CellType.MOUSE):   return "*"
    if ct == int(CellType.CHEESE):  return "C"
    if ct == int(CellType.TRAP):    return "O"
    if ct == int(CellType.WALL):    return "#"
    return ""


def _draw_grid(
    ax,
    grid: np.ndarray,
    path_cells: set[tuple[int, int]] | None = None,
    current_pos: tuple[int, int] | None = None,
    overlay: str | None = None,
    agent: QLearningAgent | None = None,
) -> None:
    """
    Vykreslí mriezku GRID_ROWS x GRID_COLS na danú os.

    Parametre
    ---------
    ax          : matplotlib Axes
    grid        : np.ndarray tvaru (GRID_ROWS, GRID_COLS) s hodnotami CellType
    path_cells  : bunky navstivene pocas animacie (zobrazi sa inou farbou)
    current_pos : aktualna pozicia mysi v animacii
    overlay     : "Q" – zobrazi Q-hodnoty + sipky, "R" – zobrazi R-hodnoty, None
    agent       : QLearningAgent (potrebny pre overlay)
    """
    ax.clear()
    ax.set_facecolor(BG_DARK)
    # Invertovana os Y: riadok 0 je hore, riadok GRID_ROWS-1 je dolu
    ax.set_xlim(-0.5, GRID_COLS - 0.5)
    ax.set_ylim(GRID_ROWS - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(BG_PANEL)

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            ct = int(grid[r, c])
            is_path    = path_cells is not None and (r, c) in path_cells
            is_current = current_pos is not None and (r, c) == current_pos

            # Farba bunky
            if is_current:
                fc = COLOR_MOUSE
            elif is_path and ct == int(CellType.EMPTY):
                fc = COLOR_PATH
            else:
                fc = _cell_color(ct)

            rect = mpatches.FancyBboxPatch(
                (c - 0.46, r - 0.46), 0.92, 0.92,
                boxstyle="round,pad=0.04",
                linewidth=0.6,
                edgecolor=BG_PANEL,
                facecolor=fc,
            )
            ax.add_patch(rect)

            # Symbol bunky (pocas animacie: * = aktualna pozicia mysi)
            sym = "*" if is_current else _cell_symbol(ct)

            # Overlay: Q-tabulka (sipky + hodnoty)
            if overlay == "Q" and agent is not None:
                if ct not in (int(CellType.WALL), int(CellType.CHEESE), int(CellType.TRAP)):
                    q_vals = agent.Q[r, c]
                    best_a = int(np.argmax(q_vals))
                    best_q = float(np.max(q_vals))

                    # Sipka pre najlepsiu akciu
                    # S invertovanou osou Y: UP (dr=-1) => dy=-0.28 (vizualne hore)
                    arrow_dy_dx = {
                        int(Action.UP):    (0, -0.28),
                        int(Action.DOWN):  (0,  0.28),
                        int(Action.LEFT):  (-0.28, 0),
                        int(Action.RIGHT): ( 0.28, 0),
                    }
                    dx, dy = arrow_dy_dx[best_a]
                    ax.annotate(
                        "", xy=(c + dx, r + dy), xytext=(c - dx * 0.4, r - dy * 0.4),
                        arrowprops=dict(
                            arrowstyle="->",
                            color=TEXT_COLOR,
                            lw=1.0,
                        ),
                    )
                    ax.text(
                        c, r + 0.40, f"{best_q:.0f}",
                        ha="center", va="center",
                        fontsize=5, color=TEXT_COLOR,
                        family="monospace",
                    )
                elif sym:
                    ax.text(
                        c, r, sym,
                        ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color=BG_DARK,
                        family="monospace",
                    )

            # Overlay: Matica odmien (R-hodnoty)
            elif overlay == "R" and agent is not None:
                if ct != int(CellType.WALL):
                    rv = agent.R[r, c]
                    text_color = BG_DARK if fc != COLOR_EMPTY else TEXT_COLOR
                    ax.text(
                        c, r, f"{rv:.0f}",
                        ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color=text_color,
                        family="monospace",
                    )

            # Normalne zobrazenie: symbol bunky
            elif sym:
                text_color = BG_DARK if fc != COLOR_EMPTY else TEXT_COLOR
                ax.text(
                    c, r, sym,
                    ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color=text_color,
                    family="monospace",
                )


def run_app() -> None:
    """
    Spusti hlavnu interaktivnu aplikaciu Q-learning 'Najdi syr'.
    Vytvori matplotlib figuru a nastavuje vsetky ovladacie prvky.
    """
    _apply_dark_theme()

    # ── Agent a stav mriezky ────────────────────────────────────────────────────
    agent = QLearningAgent(GRID_ROWS, GRID_COLS)
    grid  = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)

    # Stav aplikacie
    app_state: dict = {
        "tool":          None,       # None | "mouse" | "trap" | "wall" | "cheese"
        "mouse_pos":     None,       # tuple[int,int] | None
        "trained":       False,
        "overlay":       None,       # None | "Q" | "R"
        "path":          [],         # list[tuple[int,int]]
        "frame":         0,
        "playing":       False,
        "_upd_slider":   False,
        "anim_obj":      None,
    }

    # ── Figura ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 9), facecolor=BG_DARK)
    fig.suptitle(
        "Task 4 – Q-learning  ·  Find the Cheese",
        color=TEXT_COLOR, fontsize=14, fontweight="bold", y=0.97,
    )

    # Oddelovacia cara nad animacnymi ovladacmi
    fig.add_artist(
        Line2D([0.03, 0.97], [0.275, 0.275],
               transform=fig.transFigure,
               color=ACCENT, linewidth=0.8, alpha=0.5)
    )

    # ── Osa pre mriezku (vlavo) ─────────────────────────────────────────────────
    ax_grid = fig.add_axes([0.03, 0.30, 0.56, 0.62])

    # ── Tlacidla (vpravo) ───────────────────────────────────────────────────────
    # [x, y, sirka, vyska] v suradniciach figury
    btn_learn_ax   = fig.add_axes([0.63, 0.83, 0.33, 0.065])
    btn_find_ax    = fig.add_axes([0.63, 0.755, 0.33, 0.065])

    btn_sel_mouse_ax  = fig.add_axes([0.63, 0.655, 0.33, 0.055])
    btn_sel_trap_ax   = fig.add_axes([0.63, 0.590, 0.33, 0.055])
    btn_sel_wall_ax   = fig.add_axes([0.63, 0.525, 0.33, 0.055])
    btn_sel_cheese_ax = fig.add_axes([0.63, 0.460, 0.33, 0.055])

    btn_show_q_ax  = fig.add_axes([0.63, 0.370, 0.33, 0.055])
    btn_show_r_ax  = fig.add_axes([0.63, 0.305, 0.33, 0.055])
    btn_clear_ax   = fig.add_axes([0.63, 0.305 - 0.065, 0.33, 0.055])

    btn_learn      = Button(btn_learn_ax,   "Start learning",          color=BG_PANEL, hovercolor="#1f3a1f")
    btn_find       = Button(btn_find_ax,    "Let's find the cheese!",  color=BG_PANEL, hovercolor="#3a2e10")
    btn_sel_mouse  = Button(btn_sel_mouse_ax,  "Select a mouse",       color=BG_PANEL, hovercolor="#1a2a3a")
    btn_sel_trap   = Button(btn_sel_trap_ax,   "Select a trap",        color=BG_PANEL, hovercolor="#3a1515")
    btn_sel_wall   = Button(btn_sel_wall_ax,   "Select a wall",        color=BG_PANEL, hovercolor="#2a2a2a")
    btn_sel_cheese = Button(btn_sel_cheese_ax, "Select a cheese",      color=BG_PANEL, hovercolor="#3a2e10")
    btn_show_q     = Button(btn_show_q_ax,  "Show values of matrix",        color=BG_PANEL, hovercolor="#2a1f3a")
    btn_show_r     = Button(btn_show_r_ax,  "Show values of memory matrix", color=BG_PANEL, hovercolor="#2a1f3a")
    btn_clear      = Button(btn_clear_ax,   "Clear grid",              color=BG_PANEL, hovercolor="#3a1515")

    # Farby popiskov
    btn_learn.label.set_color(COLOR_LEARN)
    btn_find.label.set_color(COLOR_FIND)
    btn_sel_mouse.label.set_color(COLOR_SEL_MOUSE)
    btn_sel_trap.label.set_color(COLOR_SEL_TRAP)
    btn_sel_wall.label.set_color(COLOR_SEL_WALL)
    btn_sel_cheese.label.set_color(COLOR_SEL_CHEESE)
    btn_show_q.label.set_color(COLOR_MATRIX_BTN)
    btn_show_r.label.set_color(COLOR_MATRIX_BTN)
    btn_clear.label.set_color(COLOR_CLEAR)

    for btn in (btn_learn, btn_find, btn_sel_mouse, btn_sel_trap,
                btn_sel_wall, btn_sel_cheese, btn_show_q, btn_show_r, btn_clear):
        btn.label.set_fontsize(10)
        btn.label.set_fontweight("bold")

    # Informacny panel (vpravo dolu)
    ax_info = fig.add_axes([0.63, 0.30, 0.33, 0.135])
    ax_info.set_facecolor(BG_PANEL)
    ax_info.set_xticks([])
    ax_info.set_yticks([])
    for spine in ax_info.spines.values():
        spine.set_edgecolor(ACCENT)
        spine.set_linewidth(1.2)

    info_text = ax_info.text(
        0.5, 0.5,
        "Only one mouse can be placed.\n\nSelect a tool and click\non the grid to place items.",
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

    # Stavovy riadok (pod mriezkou)
    ax_status = fig.add_axes([0.03, 0.265, 0.56, 0.032])
    ax_status.set_facecolor(BG_PANEL)
    ax_status.set_xticks([])
    ax_status.set_yticks([])
    for spine in ax_status.spines.values():
        spine.set_edgecolor(ACCENT)
        spine.set_linewidth(0.8)

    status_text = ax_status.text(
        0.01, 0.5, "Select a tool and click the grid to build your maze.",
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

    def _update_info(msg: str | None = None) -> None:
        """Aktualizuje informacny panel."""
        if msg is not None:
            info_text.set_text(msg)
        fig.canvas.draw_idle()

    def _refresh_grid(
        path_cells: set[tuple[int, int]] | None = None,
        current_pos: tuple[int, int] | None = None,
    ) -> None:
        """Prekreslí mriezku s aktualnym stavom."""
        _draw_grid(
            ax_grid, grid,
            path_cells=path_cells,
            current_pos=current_pos,
            overlay=app_state["overlay"],
            agent=agent if app_state["trained"] else None,
        )
        overlay_str = ""
        if app_state["overlay"] == "Q":
            overlay_str = "  [Q-matrix overlay ON]"
        elif app_state["overlay"] == "R":
            overlay_str = "  [R-matrix overlay ON]"
        tool_str = f"  Tool: {app_state['tool']}" if app_state["tool"] else ""
        ax_grid.set_title(
            f"Playing area{tool_str}{overlay_str}",
            color=TEXT_COLOR, fontsize=12, pad=8,
        )
        fig.canvas.draw_idle()

    def _highlight_selected_tool(tool: str | None) -> None:
        """Zvyrazni tlacidlo aktualneho nastroja."""
        btn_map = {
            "mouse":  (btn_sel_mouse,  COLOR_SEL_MOUSE),
            "trap":   (btn_sel_trap,   COLOR_SEL_TRAP),
            "wall":   (btn_sel_wall,   COLOR_SEL_WALL),
            "cheese": (btn_sel_cheese, COLOR_SEL_CHEESE),
        }
        for t, (b, col) in btn_map.items():
            if t == tool:
                b.ax.set_facecolor("#2a2a2a")
                b.label.set_color(col)
                b.label.set_fontsize(10)
            else:
                b.ax.set_facecolor(BG_PANEL)
                b.label.set_color({
                    "mouse": COLOR_SEL_MOUSE, "trap": COLOR_SEL_TRAP,
                    "wall": COLOR_SEL_WALL, "cheese": COLOR_SEL_CHEESE,
                }[t])
        fig.canvas.draw_idle()

    # ── Animacia ────────────────────────────────────────────────────────────────

    def _start_animation(path: list[tuple[int, int]]) -> None:
        """Nastavi animacny stav a spusti FuncAnimation."""
        if app_state["anim_obj"] is not None:
            try:
                app_state["anim_obj"].event_source.stop()
            except Exception:
                pass

        app_state["path"]    = path
        app_state["frame"]   = 0
        app_state["playing"] = False
        app_state["anim_obj"] = None

        n = len(path)
        slider.ax.set_visible(True)
        slider.valmin = 0
        slider.valmax = max(n - 1, 1)
        slider.set_val(0)
        slider.ax.set_xlim(0, max(n - 1, 1))

        _draw_frame(0)

        anim_obj = anim_mod.FuncAnimation(
            fig, _anim_step, interval=300,
            blit=False, cache_frame_data=False,
        )
        app_state["anim_obj"] = anim_obj
        fig._anim = anim_obj

    def _draw_frame(fi: int) -> None:
        """Vykreslí krok fi animacie."""
        path = app_state["path"]
        if not path:
            return
        fi = int(np.clip(fi, 0, len(path) - 1))
        current = path[fi]
        visited = set(path[:fi])

        _draw_grid(
            ax_grid, grid,
            path_cells=visited,
            current_pos=current,
            overlay=None,
            agent=None,
        )
        # Krok info
        n = len(path)
        end_ct = int(grid[path[-1][0], path[-1][1]])
        if fi == n - 1:
            if end_ct == int(CellType.CHEESE):
                result = "  Found the cheese!"
                result_color = COLOR_CHEESE
            elif end_ct == int(CellType.TRAP):
                result = "  Fell into a trap!"
                result_color = COLOR_TRAP
            else:
                result = "  Agent is lost (loop detected)."
                result_color = COLOR_SEL_WALL
            _set_status(f"Step {fi}/{n-1} –{result}", color=result_color)
        else:
            _set_status(f"Step {fi} / {n - 1}")

        ax_grid.text(
            0.99, 0.01, f"Step: {fi} / {n - 1}",
            transform=ax_grid.transAxes,
            ha="right", va="bottom",
            fontsize=9, family="monospace",
            color=TEXT_COLOR,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_PANEL,
                      edgecolor=ACCENT, alpha=0.88),
        )
        tool_str = ""
        ax_grid.set_title(
            "Playing area – path animation",
            color=TEXT_COLOR, fontsize=12, pad=8,
        )
        fig.canvas.draw_idle()

    def _set_frame(fi: int) -> None:
        fi = int(np.clip(fi, 0, len(app_state["path"]) - 1))
        app_state["frame"] = fi
        app_state["_upd_slider"] = True
        slider.set_val(fi)
        app_state["_upd_slider"] = False
        _draw_frame(fi)

    def _anim_step(_fn: int) -> list:
        if not app_state["playing"]:
            return []
        nxt = app_state["frame"] + 1
        if nxt >= len(app_state["path"]):
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
        _draw_frame(fi)

    def on_play(event) -> None:
        if not app_state["path"]:
            return
        if app_state["playing"]:
            app_state["playing"] = False
            btn_play.label.set_text("▶ Play")
        else:
            if app_state["frame"] >= len(app_state["path"]) - 1:
                app_state["frame"] = 0
            app_state["playing"] = True
            btn_play.label.set_text("⏸ Pause")

    def on_prev(event) -> None:
        if not app_state["path"]:
            return
        app_state["playing"] = False
        btn_play.label.set_text("▶ Play")
        _set_frame(app_state["frame"] - 1)

    def on_next(event) -> None:
        if not app_state["path"]:
            return
        app_state["playing"] = False
        btn_play.label.set_text("▶ Play")
        _set_frame(app_state["frame"] + 1)

    def on_restart(event) -> None:
        if not app_state["path"]:
            return
        app_state["playing"] = False
        btn_play.label.set_text("▶ Play")
        _set_frame(0)

    slider.on_changed(on_slider)
    btn_play.on_clicked(on_play)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    btn_restart.on_clicked(on_restart)

    # ── Klik na mriezku ─────────────────────────────────────────────────────────

    def on_grid_click(event) -> None:
        if event.inaxes != ax_grid:
            return
        tool = app_state["tool"]
        if tool is None:
            _set_status("No tool selected. Choose a tool from the right panel.", color=COLOR_SEL_WALL)
            return

        c = int(round(event.xdata))
        r = int(round(event.ydata))
        if not (0 <= r < GRID_ROWS and 0 <= c < GRID_COLS):
            return

        ct_map = {
            "mouse":  int(CellType.MOUSE),
            "trap":   int(CellType.TRAP),
            "wall":   int(CellType.WALL),
            "cheese": int(CellType.CHEESE),
        }
        new_ct = ct_map[tool]

        # Ak klikneme na uz obsadenu bunku rovnakym typom – vymazeme ju
        if int(grid[r, c]) == new_ct:
            if tool == "mouse":
                app_state["mouse_pos"] = None
            grid[r, c] = int(CellType.EMPTY)
            _refresh_grid()
            _set_status(f"Removed {tool} at ({r},{c}).")
            return

        # Mys mozno umiestnitjednu
        if tool == "mouse":
            if app_state["mouse_pos"] is not None:
                pr, pc = app_state["mouse_pos"]
                grid[pr, pc] = int(CellType.EMPTY)
            app_state["mouse_pos"] = (r, c)

        grid[r, c] = new_ct
        app_state["trained"] = False
        app_state["overlay"] = None
        agent.reset()
        _refresh_grid()
        _set_status(f"Placed {tool} at ({r},{c}).")

    fig.canvas.mpl_connect("button_press_event", on_grid_click)

    # ── Obsluha: Vyber nastroja ──────────────────────────────────────────────────

    def _select_tool(tool: str) -> None:
        if app_state["tool"] == tool:
            app_state["tool"] = None
            _set_status("Tool deselected.")
        else:
            app_state["tool"] = tool
            _set_status(f"Tool selected: {tool}. Click on grid to place.")
        _highlight_selected_tool(app_state["tool"])

    btn_sel_mouse.on_clicked(lambda e: _select_tool("mouse"))
    btn_sel_trap.on_clicked(lambda e: _select_tool("trap"))
    btn_sel_wall.on_clicked(lambda e: _select_tool("wall"))
    btn_sel_cheese.on_clicked(lambda e: _select_tool("cheese"))

    # ── Obsluha: Start learning ──────────────────────────────────────────────────

    def on_learn(event) -> None:
        mouse_pos = app_state["mouse_pos"]
        if mouse_pos is None:
            _set_status("Place a mouse (*) on the grid first!", color=COLOR_TRAP)
            return
        # Skontroluj, ci je aspon jeden syr
        cheese_cells = [(r, c) for r in range(GRID_ROWS) for c in range(GRID_COLS)
                        if int(grid[r, c]) == int(CellType.CHEESE)]
        if not cheese_cells:
            _set_status("Place at least one cheese (C) on the grid!", color=COLOR_TRAP)
            return

        agent.reset()
        app_state["overlay"] = None
        _set_status("Training... please wait.", color=ACCENT)
        fig.canvas.draw_idle()
        plt.pause(0.01)

        steps = agent.train(grid, mouse_pos, n_episodes=N_EPISODES)

        app_state["trained"] = True
        conv = agent.converged_at
        conv_str = f"first cheese at ep. {conv}" if conv else "cheese not found"
        _update_info(
            f"Trained: {N_EPISODES} episodes\n"
            f"ε = {agent.epsilon:.3f}\n\n"
            f"Convergence:\n  {conv_str}\n\n"
            f"α={agent.alpha}  γ={agent.gamma}"
        )
        _set_status(
            f"Training done! {N_EPISODES} episodes. {conv_str}.",
            color=COLOR_LEARN,
        )
        _refresh_grid()

    btn_learn.on_clicked(on_learn)

    # ── Obsluha: Let's find the cheese! ─────────────────────────────────────────

    def on_find(event) -> None:
        if not app_state["trained"]:
            _set_status("Run 'Start learning' first!", color=COLOR_TRAP)
            return
        mouse_pos = app_state["mouse_pos"]
        if mouse_pos is None:
            _set_status("No mouse on grid!", color=COLOR_TRAP)
            return

        app_state["overlay"] = None
        path = agent.find_path(grid, mouse_pos)
        app_state["path"] = path
        app_state["playing"] = False
        btn_play.label.set_text("▶ Play")

        _set_status(f"Path found: {len(path) - 1} step(s). Use controls to animate.", color=COLOR_FIND)
        _start_animation(path)

    btn_find.on_clicked(on_find)

    # ── Obsluha: Show values of matrix (Q-tabulka) ──────────────────────────────

    def on_show_q(event) -> None:
        if not app_state["trained"]:
            _set_status("Run 'Start learning' first!", color=COLOR_TRAP)
            return
        if app_state["overlay"] == "Q":
            app_state["overlay"] = None
            _set_status("Q-matrix overlay: OFF")
        else:
            app_state["overlay"] = "Q"
            _set_status("Q-matrix overlay: ON  (arrows = best action, number = max Q-value)")
        _refresh_grid()

    btn_show_q.on_clicked(on_show_q)

    # ── Obsluha: Show values of memory matrix (R-matica) ────────────────────────

    def on_show_r(event) -> None:
        if not app_state["trained"]:
            _set_status("Run 'Start learning' first!", color=COLOR_TRAP)
            return
        if app_state["overlay"] == "R":
            app_state["overlay"] = None
            _set_status("Reward matrix overlay: OFF")
        else:
            app_state["overlay"] = "R"
            _set_status("Reward matrix overlay: ON  (numbers = reward values)")
        _refresh_grid()

    btn_show_r.on_clicked(on_show_r)

    # ── Obsluha: Clear grid ──────────────────────────────────────────────────────

    def on_clear(event) -> None:
        grid[:] = int(CellType.EMPTY)
        app_state["mouse_pos"] = None
        app_state["trained"]   = False
        app_state["overlay"]   = None
        app_state["path"]      = []
        app_state["frame"]     = 0
        app_state["playing"]   = False
        btn_play.label.set_text("▶ Play")
        agent.reset()
        _update_info(
            "Only one mouse can be placed.\n\nSelect a tool and click\non the grid to place items."
        )
        _refresh_grid()
        _set_status("Grid cleared.")

    btn_clear.on_clicked(on_clear)

    # ── Pociatocny stav ─────────────────────────────────────────────────────────
    _refresh_grid()
    plt.show()
