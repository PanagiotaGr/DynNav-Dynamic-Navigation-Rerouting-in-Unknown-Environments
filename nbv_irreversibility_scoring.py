# nbv_irreversibility_scoring.py
from __future__ import annotations
import numpy as np
import pandas as pd

from typing import Tuple, List, Dict

Coord = Tuple[int, int]


def sample_candidates(rng: np.random.Generator, free_mask: np.ndarray, n: int) -> List[Coord]:
    ys, xs = np.where(free_mask)
    idx = rng.choice(len(ys), size=n, replace=False if n <= len(ys) else True)
    return [(int(ys[i]), int(xs[i])) for i in idx]


def local_mean(grid: np.ndarray, y: int, x: int, r: int = 2) -> float:
    h, w = grid.shape
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    patch = grid[y0:y1, x0:x1]
    return float(np.mean(patch))


def ig_surrogate_from_uncertainty(unc_grid: np.ndarray, y: int, x: int, r: int = 2) -> float:
    """
    Simple IG surrogate: prefer higher uncertainty region (more to learn).
    You can replace this with your true entropy/IG map later.
    """
    return local_mean(unc_grid, y, x, r=r)


def score_candidates(
    candidates: List[Coord],
    unc_grid: np.ndarray,
    I_grid: np.ndarray,
    R_grid: np.ndarray,
    alpha: float,
    beta: float,
    r_local: int = 2,
) -> pd.DataFrame:
    rows = []
    for (y, x) in candidates:
        ig = ig_surrogate_from_uncertainty(unc_grid, y, x, r=r_local)
        i_loc = local_mean(I_grid, y, x, r=r_local)
        r_loc = local_mean(R_grid, y, x, r=r_local)

        score_ig = ig
        score_ig_i = ig - alpha * i_loc
        score_full = ig - alpha * i_loc - beta * (1.0 - r_loc)

        rows.append({
            "y": y, "x": x,
            "IG": ig,
            "I_local": i_loc,
            "R_local": r_loc,
            "score_IG": score_ig,
            "score_IG_I": score_ig_i,
            "score_IG_I_R": score_full,
        })

    return pd.DataFrame(rows)


def topk(df: pd.DataFrame, col: str, k: int = 10) -> pd.DataFrame:
    return df.sort_values(col, ascending=False).head(k).reset_index(drop=True)
