"""
lsystem.py – L-system generovanie a vykreslovanie.

L-system (Lindenmayer system) je definovany:
    - axiom   : pociatocny retazec (napr. "F+F+F+F")
    - rules   : slovnik prepisovacich pravidiel (napr. {"F": "F+F-F-FF+F+F-F"})
    - angle   : uhol otocenia v radianoch
    - nesting : pocet iteracii (vnoreni)

Symboly:
    F  – posun vpred (kresli ciaru)
    b  – posun vpred (nekresli)
    +  – otocenie vpravo o angle
    -  – otocenie vlavo o angle
    [  – uloz aktualnu poziciu a uhol (push)
    ]  – obnov poslednu ulozen poziciu a uhol (pop)
"""

import math
import numpy as np


def generate_string(axiom: str, rules: dict[str, str], nesting: int) -> str:
    """Aplikuje prepisovacie pravidla na axiom `nesting`-krat."""
    current = axiom
    for _ in range(nesting):
        result = []
        for ch in current:
            result.append(rules.get(ch, ch))
        current = "".join(result)
    return current


def compute_segments(
    instruction: str,
    angle: float,
    step_length: float,
    initial_angle: float = 0.0,
) -> tuple[list[tuple[float, float, float, float]], float, float, float, float]:
    """
    Interpretuje L-system retazec a vrati ciarove segmenty.

    Parametre
    ---------
    instruction   : vygenerovany L-system retazec
    angle         : uhol otocenia (radiany)
    step_length   : dlzka jedneho kroku F
    initial_angle : pociatocny uhol kresliaceho kurzora (radiany, 0 = doprava)

    Navratove hodnoty
    -----------------
    segments : list[(x0, y0, x1, y1)]
    xmin, ymin, xmax, ymax : bounding box
    """
    x, y = 0.0, 0.0
    a = initial_angle
    stack: list[tuple[float, float, float]] = []
    segments: list[tuple[float, float, float, float]] = []

    for ch in instruction:
        if ch == "F":
            nx = x + step_length * math.cos(a)
            ny = y + step_length * math.sin(a)
            segments.append((x, y, nx, ny))
            x, y = nx, ny
        elif ch == "b":
            x += step_length * math.cos(a)
            y += step_length * math.sin(a)
        elif ch == "+":
            a -= angle   # otocenie vpravo (clockwise)
        elif ch == "-":
            a += angle   # otocenie vlavo (counter-clockwise)
        elif ch == "[":
            stack.append((x, y, a))
        elif ch == "]":
            if stack:
                x, y, a = stack.pop()

    if not segments:
        return segments, 0, 0, 1, 1

    xs = [s[0] for s in segments] + [s[2] for s in segments]
    ys = [s[1] for s in segments] + [s[3] for s in segments]
    return segments, min(xs), min(ys), max(xs), max(ys)


# Predefinovane L-systemy zo zadania
PRESETS = {
    1: {
        "name": "Carpet",
        "axiom": "F+F+F+F",
        "rules": {"F": "F+F-F-FF+F+F-F"},
        "angle": math.radians(90),
        "nesting": 3,
        "step": 5,
    },
    2: {
        "name": "Koch Snowflake",
        "axiom": "F++F++F",
        "rules": {"F": "F+F--F+F"},
        "angle": math.radians(60),
        "nesting": 4,
        "step": 5,
    },
    3: {
        "name": "Weed 1",
        "axiom": "F",
        "rules": {"F": "F[+F]F[-F]F"},
        "angle": math.pi / 7,
        "nesting": 4,
        "step": 8,
        "initial_angle": math.pi / 2,
    },
    4: {
        "name": "Weed 2",
        "axiom": "F",
        "rules": {"F": "FF+[+F-F-F]-[-F+F+F]"},
        "angle": math.pi / 8,
        "nesting": 4,
        "step": 8,
        "initial_angle": math.pi / 2,
    },
}
