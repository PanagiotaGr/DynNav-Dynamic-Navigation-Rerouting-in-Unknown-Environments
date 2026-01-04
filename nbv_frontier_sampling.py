# nbv_frontier_sampling.py
from __future__ import annotations
import numpy as np
from typing import List, Tuple

Coord = Tuple[int, int]


def find_frontier_cells(
    free_mask: np.ndarray,
    unc_grid: np.ndarray,
    unc_thresh: float = 0.6,
    known_thresh: float = 0.5,
) -> List[Coord]:
    """
    Classic frontier boundary:

    - current cell is free AND relatively "known": unc(y,x) <= known_thresh
    - has at least one 4-neighbor that is "unknown/high-unc": unc(ny,nx) >= unc_thresh

    This creates a boundary between explored (known-ish) and unexplored (unknown-ish).
    """
    h, w = free_mask.shape
    frontiers: List[Coord] = []

    for y in range(h):
        for x in range(w):
            if not free_mask[y, x]:
                continue

            # stand on a known-ish cell
            if unc_grid[y, x] > known_thresh:
                continue

            # neighbor unknown-ish
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if unc_grid[ny, nx] >= unc_thresh:
                        frontiers.append((y, x))
                        break

    return frontiers


def filter_reachable_frontiers(
    frontiers: List[Coord],
    R_grid: np.ndarray,
    R_min: float = 0.05,
) -> List[Coord]:
    """Keep only frontier cells that are reachable/returnable."""
    return [(y, x) for (y, x) in frontiers if float(R_grid[y, x]) >= R_min]


def sample_from_list(
    rng: np.random.Generator,
    pts: List[Coord],
    n: int,
) -> List[Coord]:
    """
    Sample n points from a list.
    Uses replacement if list is smaller than n.
    """
    if len(pts) == 0:
        return []
    replace = len(pts) < n
    idx = rng.choice(len(pts), size=n, replace=replace)
    return [pts[int(i)] for i in idx]
def filter_frontiers_by_I(
    frontiers: List[Coord],
    I_world: np.ndarray,
    I_cap: float = 0.6,
) -> List[Coord]:
    """Keep only frontier cells with I <= I_cap (avoid risky boundary points)."""
    return [(y, x) for (y, x) in frontiers if float(I_world[y, x]) <= I_cap]
