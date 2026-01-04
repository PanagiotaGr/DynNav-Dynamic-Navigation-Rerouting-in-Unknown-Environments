# risk_weighted_planner.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import heapq
import numpy as np

Coord = Tuple[int, int]


@dataclass
class RiskAStarResult:
    success: bool
    path: List[Coord]
    cost: float
    expansions: int
    reason: str


def manhattan(a: Coord, b: Coord) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct_path(came_from: Dict[Coord, Coord], start: Coord, goal: Coord) -> List[Coord]:
    cur = goal
    path = [cur]
    while cur != start:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path


def astar_risk_weighted(
    free_mask: np.ndarray,
    I_grid: np.ndarray,
    start: Coord,
    goal: Coord,
    lam: float = 1.0,
    step_cost: float = 1.0,
    risk_agg: str = "sum",
) -> RiskAStarResult:
    """
    Risk-weighted A* baseline.

    Objective (additive):
      J = sum(step_cost) + lam * sum(I(s))   if risk_agg="sum"
      J = sum(step_cost) + lam * max(I(s))   if risk_agg="max" (implemented as additive upper bound proxy)

    Note:
    - This is a SOFT approach: it never forbids I>tau cells.
    - Works on a 4-connected grid.
    """
    if lam < 0:
        raise ValueError("lam must be >= 0")

    h, w = free_mask.shape
    sy, sx = start
    gy, gx = goal

    if not (0 <= sy < h and 0 <= sx < w and 0 <= gy < h and 0 <= gx < w):
        return RiskAStarResult(False, [], float("inf"), 0, "start/goal out of bounds")

    if not free_mask[sy, sx]:
        return RiskAStarResult(False, [], float("inf"), 0, "start not free")
    if not free_mask[gy, gx]:
        return RiskAStarResult(False, [], float("inf"), 0, "goal not free")

    # Priority queue: (f, tie, node)
    open_heap: List[Tuple[float, int, Coord]] = []
    tie = 0

    g_cost: Dict[Coord, float] = {start: 0.0}
    came_from: Dict[Coord, Coord] = {}

    # For "max" risk aggregation, we track running max risk separately
    max_risk: Dict[Coord, float] = {start: float(I_grid[sy, sx])}

    def heuristic(n: Coord) -> float:
        # admissible geometric heuristic only for step cost part (risk part ignored in h)
        return step_cost * manhattan(n, goal)

    start_f = g_cost[start] + heuristic(start)
    heapq.heappush(open_heap, (start_f, tie, start))
    tie += 1

    expansions = 0
    visited = set()

    while open_heap:
        _, _, cur = heapq.heappop(open_heap)

        if cur in visited:
            continue
        visited.add(cur)
        expansions += 1

        if cur == goal:
            # reconstruct and compute reported costs
            path = reconstruct_path(came_from, start, goal)

            # geometric cost:
            geo = step_cost * (len(path) - 1)

            # risk cost:
            if risk_agg == "sum":
                risk_val = float(np.sum([I_grid[y, x] for (y, x) in path]))
            elif risk_agg == "max":
                risk_val = float(np.max([I_grid[y, x] for (y, x) in path]))
            else:
                return RiskAStarResult(False, [], float("inf"), expansions, "invalid risk_agg")

            total = geo + lam * risk_val
            return RiskAStarResult(True, path, total, expansions, "ok")

        cy, cx = cur
        neighs = [(cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)]

        for ny, nx in neighs:
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            if not free_mask[ny, nx]:
                continue

            nxt = (ny, nx)

            # base step cost
            new_geo = g_cost[cur] + step_cost

            if risk_agg == "sum":
                # add risk as additive per-state penalty (including entering next cell)
                new_g = new_geo + lam * float(I_grid[ny, nx])
                if nxt not in g_cost or new_g < g_cost[nxt]:
                    g_cost[nxt] = new_g
                    came_from[nxt] = cur
                    f = new_g + heuristic(nxt)
                    heapq.heappush(open_heap, (f, tie, nxt))
                    tie += 1

            elif risk_agg == "max":
                # track max risk so far; use a conservative additive proxy for priority:
                # g_proxy = geo + lam * max_risk_so_far
                new_max = max(max_risk[cur], float(I_grid[ny, nx]))
                new_g_proxy = new_geo + lam * new_max

                old = g_cost.get(nxt, float("inf"))
                # store proxy in g_cost for ordering
                if new_g_proxy < old:
                    g_cost[nxt] = new_g_proxy
                    max_risk[nxt] = new_max
                    came_from[nxt] = cur
                    f = new_g_proxy + heuristic(nxt)
                    heapq.heappush(open_heap, (f, tie, nxt))
                    tie += 1

            else:
                return RiskAStarResult(False, [], float("inf"), expansions, "invalid risk_agg")

    return RiskAStarResult(False, [], float("inf"), expansions, "no path (soft risk)")
