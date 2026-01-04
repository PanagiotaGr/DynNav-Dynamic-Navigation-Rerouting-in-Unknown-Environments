# returnability_map.py
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import heapq

Coord = Tuple[int, int]


@dataclass
class ReturnabilityConfig:
    tau: float = 0.85          # cells allowed if I <= tau
    step_cost: float = 1.0
    use_8conn: bool = False
    unreachable_cost: float = 1e9


def neighbors(y: int, x: int, h: int, w: int, use_8conn: bool):
    steps4 = [(-1,0),(1,0),(0,-1),(0,1)]
    steps8 = steps4 + [(-1,-1),(-1,1),(1,-1),(1,1)]
    steps = steps8 if use_8conn else steps4
    for dy, dx in steps:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w:
            yield ny, nx


def compute_return_cost_grid(
    free_mask: np.ndarray,
    I_grid: np.ndarray,
    base: Coord,
    cfg: Optional[ReturnabilityConfig] = None,
) -> np.ndarray:
    """
    Dijkstra from base over admissible cells (I<=tau) to compute minimal return cost.
    Cells not admissible or unreachable get cfg.unreachable_cost.
    """
    if cfg is None:
        cfg = ReturnabilityConfig()

    h, w = free_mask.shape
    dist = np.full((h, w), cfg.unreachable_cost, dtype=float)

    by, bx = base
    if (not free_mask[by, bx]) or (I_grid[by, bx] > cfg.tau):
        # base itself not admissible => everything unreachable
        return dist

    dist[by, bx] = 0.0
    pq = [(0.0, by, bx)]

    while pq:
        d, y, x = heapq.heappop(pq)
        if d != dist[y, x]:
            continue

        for ny, nx in neighbors(y, x, h, w, cfg.use_8conn):
            if not free_mask[ny, nx]:
                continue
            if I_grid[ny, nx] > cfg.tau:
                continue

            nd = d + cfg.step_cost
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                heapq.heappush(pq, (nd, ny, nx))

    return dist


def normalize_returnability(return_cost: np.ndarray, unreachable_cost: float = 1e9) -> np.ndarray:
    """
    Map return cost to [0,1] where:
      1 = easily returnable (low cost),
      0 = unreachable / very high cost.
    """
    rc = return_cost.copy()
    mask = rc < unreachable_cost * 0.5
    if not np.any(mask):
        return np.zeros_like(rc)

    v = rc[mask]
    vmin, vmax = float(v.min()), float(v.max())
    norm = np.zeros_like(rc, dtype=float)
    norm[mask] = 1.0 - (rc[mask] - vmin) / (vmax - vmin + 1e-9)
    # unreachable stays 0
    return norm
# returnability_map.py
