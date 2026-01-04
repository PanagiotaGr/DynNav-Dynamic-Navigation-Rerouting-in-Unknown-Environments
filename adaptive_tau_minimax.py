# adaptive_tau_minimax.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import numpy as np

from irreversibility_planner import astar_irreversibility_constrained

Coord = Tuple[int, int]


@dataclass
class MinimaxTauConfig:
    tau_lo: float = 0.0
    tau_hi: float = 1.0
    tol: float = 0.005          # binary search resolution
    max_iters: int = 40
    margin: float = 0.0         # add safety margin above minimal feasible
    step_cost: float = 1.0


@dataclass
class MinimaxTauResult:
    feasible: bool
    tau_star: Optional[float]   # minimal feasible threshold
    tau_request: Optional[float]
    iters: int
    reason: str


def find_min_feasible_tau(
    free_mask: np.ndarray,
    I_grid: np.ndarray,
    start: Coord,
    goal: Coord,
    cfg: Optional[MinimaxTauConfig] = None,
) -> MinimaxTauResult:
    """
    Finds tau* = minimal tau such that a path exists under constraint max I <= tau.

    Uses binary search over tau with a feasibility oracle = hard planner success.
    """
    if cfg is None:
        cfg = MinimaxTauConfig()

    lo = float(cfg.tau_lo)
    hi = float(cfg.tau_hi)

    # quick check at hi
    res_hi = astar_irreversibility_constrained(
        free_mask=free_mask,
        irreversibility_grid=I_grid,
        start=start,
        goal=goal,
        tau=hi,
        step_cost=cfg.step_cost,
    )
    if not res_hi.success:
        return MinimaxTauResult(False, None, None, 0, f"infeasible even at tau_hi={hi}")

    # quick check at lo
    res_lo = astar_irreversibility_constrained(
        free_mask=free_mask,
        irreversibility_grid=I_grid,
        start=start,
        goal=goal,
        tau=lo,
        step_cost=cfg.step_cost,
    )
    if res_lo.success:
        tau_star = lo
        tau_req = min(1.0, tau_star + cfg.margin)
        return MinimaxTauResult(True, tau_star, tau_req, 0, "feasible at tau_lo")

    it = 0
    while (hi - lo) > cfg.tol and it < cfg.max_iters:
        mid = 0.5 * (lo + hi)
        res_mid = astar_irreversibility_constrained(
            free_mask=free_mask,
            irreversibility_grid=I_grid,
            start=start,
            goal=goal,
            tau=mid,
            step_cost=cfg.step_cost,
        )
        if res_mid.success:
            hi = mid
        else:
            lo = mid
        it += 1

    tau_star = hi  # minimal feasible approx
    tau_req = min(1.0, tau_star + cfg.margin)
    return MinimaxTauResult(True, float(tau_star), float(tau_req), it, "ok")
