"""
visualization.py – Interaktivna vizualizacia Task 9: Fractal Landscape (3D).

Rozlozenie figury (15x9 palcov, tmava tema):
  Vlavo  (60%) – 3D povrch terenu (plot_surface)
  Vpravo (35%) – tlacidla typov + info panel
  Dolu         – animacne ovladacie prvky (slider, Play/Pause, Prev/Next, Restart)

Farebna schema (identicka s predchadzajucimi taskmi):
  BG_DARK  = "#0d1117"
  BG_PANEL = "#161b22"
  BG_AXES  = "#1c2333"
  TEXT     = "#f0f6fc"
  ACCENT   = "#58a6ff"

Animacia: postupne sa pridavaju iteracie diamond-square algoritmu,
          zaciname plochou rovinou a koncime plnym fraktalovym terenom.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim_mod
from matplotlib.widgets import Button, Slider
from matplotlib.lines import Line2D

from .fractal import diamond_square, heights_to_colors, TERRAIN_LAYERS

# ── Farby temy (identicky s predchadzajucimi taskmi) ───────────────────────────
BG_DARK    = "#0d1117"
BG_PANEL   = "#161b22"
BG_AXES    = "#1c2333"
TEXT_COLOR = "#f0f6fc"
ACCENT     = "#58a6ff"

COLOR_SMOOTH = "#56d364"   # zelena – jemny teren
COLOR_ROUGH  = "#bc8cff"   # fialova – drsny teren
COLOR_ISLAND = "#f0b429"   # zlta – ostrov (vyssia hladina vody)
COLOR_CLEAR  = "#8b949e"   # seda – clear

# ── Parametre terenu ───────────────────────────────────────────────────────────
SIZE_EXP        = 7      # 2^7 + 1 = 129x129 mriezka
ANIM_FRAMES     = 8      # pocet iteracii zobrazenych v animacii (= size_exp)


# ── Predefinovane konfiguracie terenu ──────────────────────────────────────────
PRESETS = {
    "smooth": {
        "name":           "Jemny teren",
        "size_exp":       SIZE_EXP,
        "roughness":      0.45,
        "initial_offset": 1.0,
    },
    "rough": {
        "name":           "Drsny teren (skaly)",
        "size_exp":       SIZE_EXP,
        "roughness":      0.70,
        "initial_offset": 1.5,
    },
    "island": {
        "name":           "Ostrov",
        "size_exp":       SIZE_EXP,
        "roughness":      0.55,
        "initial_offset": 1.2,
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


def _build_progressive_heights(preset: dict, seed: int | None = None) -> list:
    """
    Vyrobi sekvenciu heightmap-ov s postupne rastucim poctom iteracii.

    Frame i obsahuje teren generovany pomocou size_exp = i+1, takze sledujeme
    ako sa terain postupne zjemnuje. Vsetky framy su upsampled na rovnaku
    velkost (2^size_exp + 1) aby sa daly zobrazit na rovnakej mriezke.
    """
    target_size = (1 << preset["size_exp"]) + 1
    frames = []
    base_seed = seed if seed is not None else int(np.random.default_rng().integers(0, 2**31))

    for level in range(1, preset["size_exp"] + 1):
        h_small = diamond_square(
            size_exp=level,
            roughness=preset["roughness"],
            initial_offset=preset["initial_offset"],
            seed=base_seed,
        )
        # Upsample na cielovu velkost (nearest-neighbor – zachovava "zubaty" charakter
        # nizsich uroni)
        small_n = h_small.shape[0]
        scale = (target_size - 1) // (small_n - 1)
        h_big = np.kron(h_small, np.ones((scale, scale)))
        # Orezat na presnu velkost
        h_big = h_big[:target_size, :target_size]
        # Doplnenie poslednych riadkov/stlpcov ak treba
        if h_big.shape[0] < target_size:
            pad = target_size - h_big.shape[0]
            h_big = np.pad(h_big, ((0, pad), (0, pad)), mode="edge")
        frames.append(h_big)

    return frames


def _shape_island(heights: np.ndarray) -> np.ndarray:
    """Pretransformuje heightmap na ostrov – posunie kraje pod uroven vody."""
    n = heights.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    cx = cy = (n - 1) / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_dist = np.sqrt(2) * cx
    # Plynula maska: 1 v strede, 0 na krajoch
    falloff = np.clip(1.0 - (dist / max_dist) ** 2, 0.0, 1.0)
    h_min = heights.min()
    return (heights - h_min) * falloff + h_min - (1.0 - falloff) * 1.5


def run_app() -> None:
    """Spusti hlavnu interaktivnu aplikaciu Task 9 – Fractal Landscape."""
    _apply_dark_theme()

    # Stav aplikacie
    app_state: dict = {
        "preset_id":     None,    # "smooth" / "rough" / "island"
        "heights":       None,    # finalny heightmap
        "frames":        [],      # progresivne heightmapy
        "frame":         0,
        "total":         0,
        "playing":       False,
        "_upd_slider":   False,
        "anim_obj":      None,
        "color":         ACCENT,
        "view_elev":     35,
        "view_azim":     -60,
    }

    # ── Figura ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 9), facecolor=BG_DARK)
    fig.suptitle(
        "Task 9 – Fractal Landscape  ·  Diamond-Square (3D)",
        color=TEXT_COLOR, fontsize=14, fontweight="bold", y=0.97,
    )

    # Oddelovacia cara nad animacnymi ovladacmi
    fig.add_artist(
        Line2D([0.03, 0.97], [0.275, 0.275],
               transform=fig.transFigure,
               color=ACCENT, linewidth=0.8, alpha=0.5)
    )

    # ── 3D osa pre teren (vlavo) ───────────────────────────────────────────────
    ax_terrain = fig.add_axes([0.03, 0.30, 0.56, 0.62], projection="3d")
    ax_terrain.set_facecolor(BG_DARK)
    ax_terrain.set_xlabel("X", color=TEXT_COLOR)
    ax_terrain.set_ylabel("Y", color=TEXT_COLOR)
    ax_terrain.set_zlabel("Z", color=TEXT_COLOR)
    ax_terrain.xaxis.pane.fill = False
    ax_terrain.yaxis.pane.fill = False
    ax_terrain.zaxis.pane.fill = False
    ax_terrain.xaxis.pane.set_edgecolor(BG_PANEL)
    ax_terrain.yaxis.pane.set_edgecolor(BG_PANEL)
    ax_terrain.zaxis.pane.set_edgecolor(BG_PANEL)
    ax_terrain.grid(True, alpha=0.3, color=BG_PANEL)
    ax_terrain.set_title("Fractal Terrain", color=TEXT_COLOR, fontsize=12, pad=8)

    # ── Tlacidla (vpravo) ──────────────────────────────────────────────────────
    btn_s_ax  = fig.add_axes([0.63, 0.86, 0.33, 0.055])
    btn_r_ax  = fig.add_axes([0.63, 0.80, 0.33, 0.055])
    btn_i_ax  = fig.add_axes([0.63, 0.74, 0.33, 0.055])
    btn_cl_ax = fig.add_axes([0.63, 0.68, 0.33, 0.055])

    btn_smooth = Button(btn_s_ax, "Jemny teren",
                        color=BG_PANEL, hovercolor="#1f3a1f")
    btn_rough  = Button(btn_r_ax, "Drsny teren (skaly)",
                        color=BG_PANEL, hovercolor="#2a1f3a")
    btn_island = Button(btn_i_ax, "Ostrov",
                        color=BG_PANEL, hovercolor="#3a2f1f")
    btn_clear  = Button(btn_cl_ax, "Clear canvas",
                        color=BG_PANEL, hovercolor="#2a2a2a")

    btn_smooth.label.set_color(COLOR_SMOOTH)
    btn_rough.label.set_color(COLOR_ROUGH)
    btn_island.label.set_color(COLOR_ISLAND)
    btn_clear.label.set_color(COLOR_CLEAR)

    for btn in (btn_smooth, btn_rough, btn_island, btn_clear):
        btn.label.set_fontsize(10)
        btn.label.set_fontweight("bold")

    # Informacny panel (vpravo dolu)
    ax_info = fig.add_axes([0.63, 0.30, 0.33, 0.36])
    ax_info.set_facecolor(BG_PANEL)
    ax_info.set_xticks([])
    ax_info.set_yticks([])
    for spine in ax_info.spines.values():
        spine.set_edgecolor(ACCENT)
        spine.set_linewidth(1.2)

    info_text = ax_info.text(
        0.5, 0.5,
        "Zvolte typ terenu:\n"
        "  Jemny / Drsny / Ostrov\n\n"
        "Algoritmus (Diamond-Square):\n"
        "  1. Stvorec rozdel na 2x2\n"
        "  2. Posun 5 vrcholov\n"
        "     o nahodnu hodnotu\n"
        "  3. Opakuj s mensim\n"
        "     posunom (roughness)\n\n"
        "Farebne vrstvy (3D):\n"
        "  voda -> piesok -> trava\n"
        "  hora -> skala -> snih",
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
        slider_ax, "Iter", 0, 1,
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
        "Kliknite na tlacidlo pre vygenerovanie terenu.",
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

    def _draw_terrain(frame_idx: int | None = None) -> None:
        """Vykresli teren pre dany frame."""
        # Zachovaj uhol kamery (ak uz ax existuje)
        try:
            elev = ax_terrain.elev
            azim = ax_terrain.azim
            app_state["view_elev"] = elev
            app_state["view_azim"] = azim
        except Exception:
            elev = app_state["view_elev"]
            azim = app_state["view_azim"]

        ax_terrain.clear()
        ax_terrain.set_facecolor(BG_DARK)
        ax_terrain.set_xlabel("X", color=TEXT_COLOR)
        ax_terrain.set_ylabel("Y", color=TEXT_COLOR)
        ax_terrain.set_zlabel("Z", color=TEXT_COLOR)
        ax_terrain.xaxis.pane.fill = False
        ax_terrain.yaxis.pane.fill = False
        ax_terrain.zaxis.pane.fill = False
        ax_terrain.xaxis.pane.set_edgecolor(BG_PANEL)
        ax_terrain.yaxis.pane.set_edgecolor(BG_PANEL)
        ax_terrain.zaxis.pane.set_edgecolor(BG_PANEL)
        ax_terrain.grid(True, alpha=0.3, color=BG_PANEL)

        frames = app_state["frames"]
        if len(frames) == 0:
            ax_terrain.set_title("Fractal Terrain",
                                 color=TEXT_COLOR, fontsize=12, pad=8)
            ax_terrain.view_init(elev=elev, azim=azim)
            fig.canvas.draw_idle()
            return

        if frame_idx is None:
            frame_idx = app_state["frame"]
        frame_idx = int(np.clip(frame_idx, 0, len(frames) - 1))

        heights = frames[frame_idx]
        n = heights.shape[0]
        x = np.arange(n)
        y = np.arange(n)
        X, Y = np.meshgrid(x, y)

        # Farebne vrstvy podla finalneho rozsahu vysok (stabilna farebna skala)
        ref_heights = frames[-1]
        h_min, h_max = ref_heights.min(), ref_heights.max()
        if h_max - h_min < 1e-9:
            norm = np.zeros_like(heights)
        else:
            norm = (heights - h_min) / (h_max - h_min)

        # Vyrob facecolors podla normalizovanych vysok (rozmery musia byt
        # (n-1, n-1) lebo plot_surface farby aplikuje na facety, nie vertexy)
        face_norm = 0.25 * (norm[:-1, :-1] + norm[1:, :-1] +
                            norm[:-1, 1:]  + norm[1:, 1:])
        face_colors = np.zeros((n - 1, n - 1, 4), dtype=float)
        for threshold, hex_color, _name in TERRAIN_LAYERS:
            mask = face_norm <= threshold
            unset = face_colors[..., 3] == 0
            target = mask & unset
            r = int(hex_color[1:3], 16) / 255.0
            g = int(hex_color[3:5], 16) / 255.0
            b = int(hex_color[5:7], 16) / 255.0
            face_colors[target] = (r, g, b, 1.0)
        unset = face_colors[..., 3] == 0
        if unset.any():
            face_colors[unset] = (1.0, 1.0, 1.0, 1.0)

        # Pri animacii s velkou mriezkou znizujeme rstride/cstride pre rychlost
        stride = max(1, n // 80)
        ax_terrain.plot_surface(
            X, Y, heights,
            facecolors=face_colors,
            rstride=stride, cstride=stride,
            linewidth=0.0, antialiased=False, shade=False,
        )

        # Hladina vody – poloprehladna modra plocha
        water_threshold = TERRAIN_LAYERS[1][0]   # hranica medzi plytkou vodou a piesokom
        water_level = h_min + water_threshold * (h_max - h_min)
        if heights.min() < water_level:
            water = np.full_like(heights, water_level)
            ax_terrain.plot_surface(
                X, Y, water,
                color="#2a6db0", alpha=0.35,
                rstride=max(1, n // 20), cstride=max(1, n // 20),
                linewidth=0.0, antialiased=False, shade=False,
            )

        # Pevne limity Z = stabilna kompozicia v animacii
        z_pad = 0.1 * (h_max - h_min + 1e-9)
        ax_terrain.set_zlim(h_min - z_pad, h_max + z_pad)
        ax_terrain.set_xlim(0, n - 1)
        ax_terrain.set_ylim(0, n - 1)

        # Vyber preset name pre titulok
        preset_name = ""
        pid = app_state["preset_id"]
        if pid and pid in PRESETS:
            preset_name = PRESETS[pid]["name"]

        title = (f"Diamond-Square  –  {preset_name}  ·  "
                 f"iter {frame_idx + 1}/{len(frames)}"
                 if preset_name else "Fractal Terrain")
        ax_terrain.set_title(title, color=TEXT_COLOR, fontsize=11, pad=8)

        ax_terrain.view_init(elev=elev, azim=azim)
        fig.canvas.draw_idle()

    # ── Generovanie a spustenie animacie ────────────────────────────────────────

    def _run_terrain(preset_id: str, color: str) -> None:
        """Vygeneruje teren podla presetu a spusti animaciu."""
        if app_state["anim_obj"] is not None:
            try:
                app_state["anim_obj"].event_source.stop()
            except Exception:
                pass

        _set_status("Generujem teren...", color=ACCENT)
        fig.canvas.draw_idle()
        plt.pause(0.01)

        preset = PRESETS[preset_id]
        seed = int(np.random.default_rng().integers(0, 2**31))
        frames = _build_progressive_heights(preset, seed=seed)

        # Pri ostrove pretransformuj vsetky framy pomocou radial falloff
        if preset_id == "island":
            frames = [_shape_island(f) for f in frames]

        app_state["preset_id"] = preset_id
        app_state["heights"]   = frames[-1]
        app_state["frames"]    = frames
        app_state["frame"]     = len(frames) - 1
        app_state["total"]     = len(frames)
        app_state["playing"]   = False
        app_state["color"]     = color

        n = len(frames)
        slider.valmin = 0
        slider.valmax = max(n - 1, 1)
        slider.set_val(n - 1)
        slider.ax.set_xlim(0, max(n - 1, 1))

        size = frames[-1].shape[0]
        layers_str = "\n".join(
            f"  {l[2]:13s} <= {int(l[0]*100):3d}%" for l in TERRAIN_LAYERS
        )
        _update_info(
            f"Preset: {preset['name']}\n\n"
            f"Mriezka: {size}x{size}\n"
            f"Iteracie: {preset['size_exp']}\n"
            f"Roughness: {preset['roughness']:.2f}\n"
            f"Init offset: {preset['initial_offset']:.2f}\n\n"
            f"Vrstvy (% z max vysky):\n"
            f"{layers_str}\n\n"
            f"Otacanie: tahaj mysou\n"
            f"v 3D oblasti."
        )

        _set_status(
            f"Hotovo! {preset['name']} vygenerovany ({n} iteracii).",
            color=color,
        )
        _draw_terrain(n - 1)

        anim_obj = anim_mod.FuncAnimation(
            fig, _anim_step, interval=400,
            blit=False, cache_frame_data=False,
        )
        app_state["anim_obj"] = anim_obj
        fig._anim = anim_obj

    # ── Animacia ────────────────────────────────────────────────────────────────

    def _set_frame(fi: int) -> None:
        n = len(app_state["frames"])
        if n == 0:
            return
        fi = int(np.clip(fi, 0, n - 1))
        app_state["frame"]       = fi
        app_state["_upd_slider"] = True
        slider.set_val(fi)
        app_state["_upd_slider"] = False
        _draw_terrain(fi)

        if fi == n - 1:
            _set_status(f"Kompletne vykreslenie: {n} iteracii.",
                        color=app_state["color"])
        else:
            _set_status(f"Iteracia {fi + 1} / {n}")

    def _anim_step(_fn: int) -> list:
        if not app_state["playing"]:
            return []
        nxt = app_state["frame"] + 1
        n = len(app_state["frames"])
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
        _draw_terrain(fi)

    def on_play(event) -> None:
        if len(app_state["frames"]) == 0:
            return
        if app_state["playing"]:
            app_state["playing"] = False
            btn_play.label.set_text("▶ Play")
        else:
            if app_state["frame"] >= len(app_state["frames"]) - 1:
                app_state["frame"] = 0
            app_state["playing"] = True
            btn_play.label.set_text("⏸ Pause")

    def on_prev(event) -> None:
        if len(app_state["frames"]) == 0:
            return
        app_state["playing"] = False
        btn_play.label.set_text("▶ Play")
        _set_frame(app_state["frame"] - 1)

    def on_next(event) -> None:
        if len(app_state["frames"]) == 0:
            return
        app_state["playing"] = False
        btn_play.label.set_text("▶ Play")
        _set_frame(app_state["frame"] + 1)

    def on_restart(event) -> None:
        if len(app_state["frames"]) == 0:
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

    def on_smooth(event):
        _run_terrain("smooth", COLOR_SMOOTH)

    def on_rough(event):
        _run_terrain("rough", COLOR_ROUGH)

    def on_island(event):
        _run_terrain("island", COLOR_ISLAND)

    btn_smooth.on_clicked(on_smooth)
    btn_rough.on_clicked(on_rough)
    btn_island.on_clicked(on_island)

    # ── Obsluha clear ───────────────────────────────────────────────────────────

    def on_clear(event) -> None:
        if app_state["anim_obj"] is not None:
            try:
                app_state["anim_obj"].event_source.stop()
            except Exception:
                pass

        app_state["preset_id"] = None
        app_state["heights"]   = None
        app_state["frames"]    = []
        app_state["frame"]     = 0
        app_state["total"]     = 0
        app_state["playing"]   = False
        app_state["anim_obj"]  = None
        btn_play.label.set_text("▶ Play")

        _update_info(
            "Zvolte typ terenu:\n"
            "  Jemny / Drsny / Ostrov\n\n"
            "Algoritmus (Diamond-Square):\n"
            "  1. Stvorec rozdel na 2x2\n"
            "  2. Posun 5 vrcholov\n"
            "     o nahodnu hodnotu\n"
            "  3. Opakuj s mensim\n"
            "     posunom (roughness)\n\n"
            "Farebne vrstvy (3D):\n"
            "  voda -> piesok -> trava\n"
            "  hora -> skala -> snih"
        )
        _set_status("Canvas vymazany.")
        _draw_terrain()

    btn_clear.on_clicked(on_clear)

    # ── Pociatocny stav ─────────────────────────────────────────────────────────
    _draw_terrain()
    plt.show()
