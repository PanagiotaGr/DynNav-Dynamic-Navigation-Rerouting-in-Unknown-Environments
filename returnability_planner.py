# returnability_planner.py
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import heapq

Coord = Tuple[int, int]


@dataclass
class PlanResult:
    success: bool
    cost: float
    expansions: int
    reason: str
    path: Optional[list[Coord]] = None
    max_I: float = 0.0
    mean_I: float = 0.0
    mean_R: float = 0.0


def neighbors(y: int, x: int, h: int, w: int):
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w:
            yield ny, nx


def astar_soft_I_R(
    free_mask: np.ndarray,
    I_grid: np.ndarray,
    R_grid: np.ndarray,
    start: Coord,
    goal: Coord,
    lam: float = 1.0,           # weight on I
    mu: float = 1.0,            # weight on (1-R)
    step_cost: float = 1.0,
    tau_cap: Optional[float] = None,   # HARD safety cap: allow only cells with I<=tau_cap
) -> PlanResult:
    """
    A* on 4-neighborhood with soft penalties:
      g += step_cost + lam * I(next) + mu * (1 - R(next))

    Hard guardrail (optional):
      if tau_cap is not None: we forbid transitions to cells with I > tau_cap

    heuristic: Manhattan distance * step_cost (admissible for step_cost part)
    """
    h, w = free_mask.shape
    sy, sx = start
    gy, gx = goal

    if not free_mask[sy, sx]:
        return PlanResult(False, np.inf, 0, "start not free")
    if not free_mask[gy, gx]:
        return PlanResult(False, np.inf, 0, "goal not free")

    if tau_cap is not None:
        if float(I_grid[sy, sx]) > float(tau_cap):
            return PlanResult(False, np.inf, 0, "start violates tau_cap")
        if float(I_grid[gy, gx]) > float(tau_cap):
            return PlanResult(False, np.inf, 0, "goal violates tau_cap")

    def heur(y: int, x: int) -> float:
        return (abs(y - gy) + abs(x - gx)) * step_cost

    gscore = np.full((h, w), np.inf, dtype=float)
    parent_y = np.full((h, w), -1, dtype=int)
    parent_x = np.full((h, w), -1, dtype=int)
    closed = np.zeros((h, w), dtype=bool)

    gscore[sy, sx] = 0.0
    pq = [(heur(sy, sx), 0.0, sy, sx)]
    expansions = 0

    while pq:
        f, g, y, x = heapq.heappop(pq)
        if closed[y, x]:
            continue
        closed[y, x] = True
        expansions += 1

        if (y, x) == (gy, gx):
            # reconstruct path
            path: list[Coord] = []
            cy, cx = gy, gx
            while not (cy == sy and cx == sx):
                path.append((cy, cx))
                py, px = parent_y[cy, cx], parent_x[cy, cx]
                if py < 0:
                    break
                cy, cx = py, px
            path.append((sy, sx))
            path.reverse()

            Ivals = [float(I_grid[p]) for p in path]
            Rvals = [float(R_grid[p]) for p in path]
            return PlanResult(
                True,
                float(gscore[gy, gx]),
                expansions,
                "ok",
                path=path,
                max_I=float(np.max(Ivals)),
                mean_I=float(np.mean(Ivals)),
                mean_R=float(np.mean(Rvals)),
            )

        for ny, nx in neighbors(y, x, h, w):
            if not free_mask[ny, nx]:
                continue
            if closed[ny, nx]:
                continue

            if tau_cap is not None and float(I_grid[ny, nx]) > float(tau_cap):
                continue

            step_pen = step_cost + lam * float(I_grid[ny, nx]) + mu * (1.0 - float(R_grid[ny, nx]))
            ng = gscore[y, x] + step_pen

            if ng < gscore[ny, nx]:
                gscore[ny, nx] = ng
                parent_y[ny, nx] = y
                parent_x[ny, nx] = x
                heapq.heappush(pq, (ng + heur(ny, nx), ng, ny, nx))

    return PlanResult(False, np.inf, expansions, "no path in free space")

