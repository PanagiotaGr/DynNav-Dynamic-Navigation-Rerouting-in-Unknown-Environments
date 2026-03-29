"""
belief_space_planner.py

Belief-space planning: robot state represented as Gaussian (mean, covariance).

The planner propagates uncertainty through motion and incorporates it into the
cost function. Compared to deterministic planning, this finds paths that trade
off distance against accumulated uncertainty.

State representation
--------------------
    belief = (μ, Σ)   μ ∈ R², Σ ∈ R²ˣ²

Motion model (linear with additive Gaussian noise)
--------------------
    μ_{t+1} = A μ_t + B u_t
    Σ_{t+1} = A Σ_t A^T + Q

Cost function
-------------
    J = Σ_t [ step_cost + lambda_uncert * tr(Σ_t) ]

where tr(Σ) is the trace (sum of variances) as a scalar uncertainty measure.

Comparison
----------
- Deterministic: λ = 0  (ignores uncertainty)
- Uncertainty-aware: λ > 0  (penalises high-uncertainty paths)
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Belief state
# ---------------------------------------------------------------------------

@dataclass
class BeliefState:
    mean: np.ndarray      # (2,) position estimate
    cov: np.ndarray       # (2, 2) covariance

    @property
    def uncertainty(self) -> float:
        """Scalar uncertainty = trace(Σ)."""
        return float(np.trace(self.cov))

    @property
    def pos(self) -> tuple[int, int]:
        """Grid cell (rounded mean)."""
        return (int(round(float(self.mean[0]))), int(round(float(self.mean[1]))))

    def copy(self) -> "BeliefState":
        return BeliefState(self.mean.copy(), self.cov.copy())


# ---------------------------------------------------------------------------
# Motion model
# ---------------------------------------------------------------------------

class LinearMotionModel:
    """
    μ_{t+1} = μ_t + u
    Σ_{t+1} = Σ_t + Q

    Simple additive noise model (constant velocity on a grid).
    """

    def __init__(self, Q: Optional[np.ndarray] = None):
        self.Q = Q if Q is not None else np.diag([0.01, 0.01])

    def propagate(self, belief: BeliefState, action: np.ndarray) -> BeliefState:
        new_mean = belief.mean + action
        new_cov = belief.cov + self.Q
        return BeliefState(new_mean, new_cov)


# ---------------------------------------------------------------------------
# Belief-space A*
# ---------------------------------------------------------------------------

NEIGHBORS_4: list[tuple[int, int]] = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def belief_astar(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    lambda_uncert: float = 1.0,
    init_cov: Optional[np.ndarray] = None,
    motion_noise: Optional[np.ndarray] = None,
) -> dict:
    """
    Belief-space A* on a 2-D grid.

    Parameters
    ----------
    grid           : (H, W) array, 0=free 1=obstacle
    start, goal    : (x, y) grid coordinates
    lambda_uncert  : weight on uncertainty in cost (0 = deterministic)
    init_cov       : initial covariance (default: 0.01 * I)
    motion_noise   : per-step noise Q (default: 0.01 * I)

    Returns
    -------
    dict with: path, belief_path, cost, expansions, found
    """
    H, W = grid.shape

    if init_cov is None:
        init_cov = np.diag([0.01, 0.01])
    model = LinearMotionModel(Q=motion_noise)

    def in_bounds(x, y):
        return 0 <= x < W and 0 <= y < H

    def free(x, y):
        return int(grid[y, x]) == 0

    def h(pos: tuple[int, int]) -> float:
        return math.sqrt((pos[0]-goal[0])**2 + (pos[1]-goal[1])**2)

    init_belief = BeliefState(
        mean=np.array([float(start[0]), float(start[1])]),
        cov=init_cov.copy(),
    )

    # g_cost: position → (min_cost, belief)
    g_cost: dict[tuple[int, int], float] = {start: 0.0}
    best_belief: dict[tuple[int, int], BeliefState] = {start: init_belief}
    parent: dict[tuple[int, int], Optional[tuple[int, int]]] = {start: None}
    belief_history: dict[tuple[int, int], BeliefState] = {start: init_belief}

    f0 = 0.0 + lambda_uncert * init_belief.uncertainty + h(start)
    open_pq = [(f0, 0.0, start)]
    closed: set[tuple[int, int]] = set()
    expansions = 0

    while open_pq:
        _, g_curr, curr = heapq.heappop(open_pq)
        if curr in closed:
            continue
        closed.add(curr)
        expansions += 1

        if curr == goal:
            # Reconstruct path
            path, node = [], curr
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            belief_path = [belief_history[p] for p in path]
            final_uncert = belief_history[goal].uncertainty
            return {
                "found": True,
                "path": path,
                "belief_path": belief_path,
                "cost": g_curr,
                "final_uncertainty": final_uncert,
                "expansions": expansions,
            }

        curr_belief = best_belief[curr]
        x, y = curr

        for dx, dy in NEIGHBORS_4:
            nx, ny = x + dx, y + dy
            if not in_bounds(nx, ny) or not free(nx, ny):
                continue
            neigh = (nx, ny)

            action = np.array([float(dx), float(dy)])
            new_belief = model.propagate(curr_belief, action)

            step_cost = 1.0 + lambda_uncert * new_belief.uncertainty
            tg = g_curr + step_cost

            if neigh not in g_cost or tg < g_cost[neigh]:
                g_cost[neigh] = tg
                best_belief[neigh] = new_belief
                parent[neigh] = curr
                belief_history[neigh] = new_belief
                f = tg + h(neigh)
                heapq.heappush(open_pq, (f, tg, neigh))

    return {"found": False, "path": None, "belief_path": [], "cost": float("inf"),
            "final_uncertainty": float("inf"), "expansions": expansions}


def compare_belief_vs_deterministic(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    lambda_values: list[float] | None = None,
) -> list[dict]:
    """
    Run belief-space planner with multiple λ values and report results.
    λ=0 → deterministic, λ>0 → uncertainty-aware.
    """
    if lambda_values is None:
        lambda_values = [0.0, 0.5, 1.0, 2.0]

    results = []
    for lam in lambda_values:
        r = belief_astar(grid, start, goal, lambda_uncert=lam)
        r["lambda"] = lam
        r["path_length"] = len(r["path"]) - 1 if r["path"] else float("inf")
        results.append(r)
    return results
