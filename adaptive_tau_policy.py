# adaptive_tau_policy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

Coord = Tuple[int, int]


@dataclass
class AdaptiveTauConfig:
    tau_min: float = 0.70
    tau_max: float = 0.95
    # if True, trust is computed from local neighborhood mean I around start
    local_radius: int = 0
    eps: float = 1e-9


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def local_mean_I(I_grid: np.ndarray, s: Coord, r: int) -> float:
    y, x = s
    h, w = I_grid.shape
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    patch = I_grid[y0:y1, x0:x1]
    return float(np.mean(patch))


def compute_self_trust(I_grid: np.ndarray, start: Coord, cfg: AdaptiveTauConfig) -> float:
    """
    Simple, interpretable proxy:
      trust S in [0,1], higher = more confident environment is recoverable.
    """
    if cfg.local_radius > 0:
        I0 = local_mean_I(I_grid, start, cfg.local_radius)
    else:
        I0 = float(I_grid[start])
    S = 1.0 - clip(I0, 0.0, 1.0)
    return clip(S, 0.0, 1.0)


def adaptive_tau_from_trust(S: float, cfg: AdaptiveTauConfig) -> float:
    """
    If trust is low (S small), request more permissive threshold (higher tau).
    tau_request = tau_min + (1-S)*(tau_max - tau_min)
    """
    S = clip(float(S), 0.0, 1.0)
    tau = cfg.tau_min + (1.0 - S) * (cfg.tau_max - cfg.tau_min)
    return clip(float(tau), cfg.tau_min, cfg.tau_max)
