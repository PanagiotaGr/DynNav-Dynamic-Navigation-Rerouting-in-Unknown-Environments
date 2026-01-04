# irreversibility_safe_mode.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np

from irreversibility_planner import astar_irreversibility_constrained

Coord = Tuple[int, int]


@dataclass
class SafeModeConfig:
    # Search for minimal feasible tau in [tau0, tau_max]
    tau_max: float = 1.0
    tau_step: float = 0.01  # resolution of relaxation search
    # If required tau gap is too large, we can choose to stop instead
    max_tau_gap: float = 0.25
    # If even at tau_max no path, stop
    stop_if_infeasible: bool = True


@dataclass
class SafeModeDecision:
    success: bool
    mode: str  # "NORMAL" | "SAFE_RELAX_TAU" | "SAFE_STOP"
    start: Coord
    goal: Coord
    tau_requested: float
    tau_used: Optional[float]
    tau_gap: Optional[float]
    reason: str

    path: List[Coord]
    expansions: int
    cost: float

    # Metrics on the returned path (if any)
    max_I_on_path: Optional[float]
    mean_I_on_path: Optional[float]


def _path_metrics(I_grid: np.ndarray, path: List[Coord]) -> Tuple[Optional[float], Optional[float]]:
    if not path:
        return None, None
    vals = np.array([I_grid[y, x] for (y, x) in path], dtype=float)
    return float(np.max(vals)), float(np.mean(vals))


def run_irreversibility_with_safe_mode(
    free_mask: np.ndarray,
    I_grid: np.ndarray,
    start: Coord,
    goal: Coord,
    tau0: float,
    step_cost: float = 1.0,
    cfg: Optional[SafeModeConfig] = None,
) -> SafeModeDecision:
    """
    Safe-mode wrapper around hard irreversibility-constrained planning.

    1) Try hard A* with tau0.
    2) If fails due to irreversibility constraint, find MINIMAL tau in [tau0, tau_max]
       that becomes feasible (minimal relaxation).
    3) If relaxation required is too large -> SAFE_STOP (optional).

    This implements a principled "graceful degradation" strategy.
    """
    if cfg is None:
        cfg = SafeModeConfig()

    # 1) try requested tau
    res0 = astar_irreversibility_constrained(
        free_mask=free_mask,
        irreversibility_grid=I_grid,
        start=start,
        goal=goal,
        tau=float(tau0),
        step_cost=float(step_cost),
    )

    if res0.success:
        maxI, meanI = _path_metrics(I_grid, res0.path)
        return SafeModeDecision(
            success=True,
            mode="NORMAL",
            start=start,
            goal=goal,
            tau_requested=float(tau0),
            tau_used=float(tau0),
            tau_gap=0.0,
            reason="ok",
            path=res0.path,
            expansions=int(res0.expansions),
            cost=float(res0.cost),
            max_I_on_path=maxI,
            mean_I_on_path=meanI,
        )

    # If failure not related to constraint, we stop
    # (e.g., start/goal invalid, out of bounds, etc.)
    if "irreversibility" not in str(res0.reason).lower() and "no path" not in str(res0.reason).lower():
        return SafeModeDecision(
            success=False,
            mode="SAFE_STOP",
            start=start,
            goal=goal,
            tau_requested=float(tau0),
            tau_used=None,
            tau_gap=None,
            reason=f"non-constraint failure: {res0.reason}",
            path=[],
            expansions=int(res0.expansions),
            cost=float("inf"),
            max_I_on_path=None,
            mean_I_on_path=None,
        )

    # 2) minimal relaxation search
    tau0 = float(tau0)
    tau_max = float(cfg.tau_max)
    tau_step = float(cfg.tau_step)

    # build candidate taus (inclusive)
    if tau_step <= 0:
        raise ValueError("tau_step must be > 0")

    taus = np.arange(tau0, tau_max + 1e-12, tau_step)
    best_tau = None
    best_res = None

    for tau in taus:
        res = astar_irreversibility_constrained(
            free_mask=free_mask,
            irreversibility_grid=I_grid,
            start=start,
            goal=goal,
            tau=float(tau),
            step_cost=float(step_cost),
        )
        if res.success:
            best_tau = float(tau)
            best_res = res
            break

    if best_res is None:
        # 3) still infeasible even at tau_max
        if cfg.stop_if_infeasible:
            return SafeModeDecision(
                success=False,
                mode="SAFE_STOP",
                start=start,
                goal=goal,
                tau_requested=float(tau0),
                tau_used=None,
                tau_gap=None,
                reason=f"infeasible even after relaxation up to tau_max={tau_max}",
                path=[],
                expansions=int(res0.expansions),
                cost=float("inf"),
                max_I_on_path=None,
                mean_I_on_path=None,
            )
        else:
            # return original failure
            return SafeModeDecision(
                success=False,
                mode="SAFE_STOP",
                start=start,
                goal=goal,
                tau_requested=float(tau0),
                tau_used=None,
                tau_gap=None,
                reason=str(res0.reason),
                path=[],
                expansions=int(res0.expansions),
                cost=float("inf"),
                max_I_on_path=None,
                mean_I_on_path=None,
            )

    tau_gap = best_tau - tau0
    if tau_gap > cfg.max_tau_gap:
        # relaxation too large -> stop (conservative safety)
        return SafeModeDecision(
            success=False,
            mode="SAFE_STOP",
            start=start,
            goal=goal,
            tau_requested=float(tau0),
            tau_used=best_tau,
            tau_gap=float(tau_gap),
            reason=f"required tau relaxation too large (gap={tau_gap:.3f} > max_tau_gap={cfg.max_tau_gap})",
            path=[],
            expansions=int(best_res.expansions),
            cost=float("inf"),
            max_I_on_path=None,
            mean_I_on_path=None,
        )

    maxI, meanI = _path_metrics(I_grid, best_res.path)
    return SafeModeDecision(
        success=True,
        mode="SAFE_RELAX_TAU",
        start=start,
        goal=goal,
        tau_requested=float(tau0),
        tau_used=float(best_tau),
        tau_gap=float(tau_gap),
        reason=f"relaxed tau to minimal feasible threshold",
        path=best_res.path,
        expansions=int(best_res.expansions),
        cost=float(best_res.cost),
        max_I_on_path=maxI,
        mean_I_on_path=meanI,
    )

