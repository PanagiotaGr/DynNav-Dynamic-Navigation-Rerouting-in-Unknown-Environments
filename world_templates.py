# world_templates.py
from __future__ import annotations
import numpy as np
from typing import Tuple, Dict

Coord = Tuple[int, int]


def base_world(unc_grid: np.ndarray):
    free_mask = np.isfinite(unc_grid)
    return free_mask


# --------------------------------------------------
# 1) Bottleneck world (wrapper γύρω από αυτό που έχεις)
# --------------------------------------------------
def world_bottleneck(I_grid: np.ndarray, wall_I=0.95, door_I=0.6, thickness=2):
    h, w = I_grid.shape
    I = I_grid.copy()

    wall_cols = (w // 2 - thickness, w // 2 + thickness)
    door_rows = (h // 2 - 2, h // 2 + 2)

    for c in range(wall_cols[0], wall_cols[1]):
        for r in range(h):
            I[r, c] = wall_I

    for r in range(door_rows[0], door_rows[1]):
        for c in range(wall_cols[0], wall_cols[1]):
            I[r, c] = door_I

    meta = {
        "type": "bottleneck",
        "wall_I": wall_I,
        "door_I": door_I,
        "wall_cols": wall_cols,
        "door_rows": door_rows,
    }
    return I, meta


# --------------------------------------------------
# 2) Cul-de-sac world (trap)
# --------------------------------------------------
def world_culdesac(I_grid: np.ndarray, trap_I=0.9):
    I = I_grid.copy()
    h, w = I.shape

    # vertical corridor
    x = w // 2
    for r in range(h // 4, 3 * h // 4):
        I[r, x] = trap_I

    # dead-end pocket
    for r in range(3 * h // 4, 3 * h // 4 + 4):
        for c in range(x - 3, x + 3):
            I[r, c] = trap_I

    meta = {
        "type": "culdesac",
        "trap_I": trap_I,
    }
    return I, meta


# --------------------------------------------------
# 3) Noisy corridor (gradual irreversibility)
# --------------------------------------------------
def world_noisy_corridor(I_grid: np.ndarray, I_min=0.3, I_max=0.9):
    I = I_grid.copy()
    h, w = I.shape

    x0 = w // 3
    x1 = 2 * w // 3

    for c in range(x0, x1):
        alpha = (c - x0) / max(1, (x1 - x0))
        val = I_min + alpha * (I_max - I_min)
        I[:, c] = np.maximum(I[:, c], val)

    meta = {
        "type": "noisy_corridor",
        "I_min": I_min,
        "I_max": I_max,
    }
    return I, meta
