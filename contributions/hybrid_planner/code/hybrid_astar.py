"""
hybrid_astar.py

Hybrid A*: switches between learned uncertainty heuristic and admissible
Euclidean heuristic on a per-node basis via ConfidenceGate.

    f(n) = g(n) + h_selected(n)

where h_selected is chosen by the gate.

Bounded suboptimality
---------------------
When ε_budget is set, the learned heuristic is only used if it doesn't
exceed the admissible heuristic by more than ε_budget. This ensures the
returned path costs at most (1 + ε) × optimal.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from confidence_gate import ConfidenceGate, GateConfig, HeuristicChoice

NEIGHBORS_4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _euclidean(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _extract_features(
    node: tuple[int, int],
    goal: tuple[int, int],
    grid: np.ndarray,
) -> "np.ndarray":
    """11-D feature vector (shared with learned_uncertainty_astar)."""
    import sys, os
    # Import from sibling module if available
    try:
        from uncertainty_astar import extract_features
        return extract_features(node, goal, grid)
    except ImportError:
        pass

    x, y = node
    gx, gy = goal
    H, W = grid.shape
    dx, dy = gx - x, gy - y
    euclid = math.sqrt(dx * dx + dy * dy)
    manhattan = abs(dx) + abs(dy)
    chebyshev = max(abs(dx), abs(dy))
    free_n = sum(
        1 for nx, ny in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
        if 0 <= nx < W and 0 <= ny < H and grid[ny, nx] == 0
    )
    x_min, x_max = max(0, x-1), min(W, x+2)
    y_min, y_max = max(0, y-1), min(H, y+2)
    w = grid[y_min:y_max, x_min:x_max]
    od = float(np.mean(w != 0)) if w.size > 0 else 0.0
    return np.array(
        [dx, dy, euclid, manhattan, chebyshev, free_n, 4-free_n, od,
         1.0 if od > 0 else 0.0, x/max(W-1,1), y/max(H-1,1)],
        dtype=np.float32,
    )


@dataclass
class HybridAStarResult:
    path: Optional[list[tuple[int, int]]]
    expansions: int
    path_length: float
    learned_fraction: float   # fraction of nodes where learned heuristic was used
    h_choices: list[HeuristicChoice] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.path is not None


def hybrid_astar(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    learned_model,          # UncertaintyHeuristicNet or None
    gate_config: Optional[GateConfig] = None,
) -> HybridAStarResult:
    """
    Hybrid A* with confidence-gated heuristic selection.

    Parameters
    ----------
    grid          : (H, W) occupancy grid, 0=free 1=obstacle
    start, goal   : (x, y) coordinates
    learned_model : UncertaintyHeuristicNet (or None → pure Euclidean)
    gate_config   : ConfidenceGate configuration

    Returns
    -------
    HybridAStarResult
    """
    H, W = grid.shape
    gate = ConfidenceGate(gate_config)

    has_model = learned_model is not None

    try:
        import torch
        _torch = torch
    except ImportError:
        _torch = None
        has_model = False

    def in_bounds(x, y):
        return 0 <= x < W and 0 <= y < H

    def free(x, y):
        return int(grid[y, x]) == 0

    def heuristic(node):
        h_adm = _euclidean(node, goal)
        if not has_model:
            return h_adm, HeuristicChoice.ADMISSIBLE

        feat = _extract_features(node, goal, grid)
        with _torch.no_grad():
            x_t = _torch.from_numpy(feat)
            mean, std = learned_model(x_t)
        h_mean = float(mean.squeeze())
        h_std = float(std.squeeze())
        h_val, choice = gate.select(h_mean, h_std, h_adm)
        return h_val, choice

    h0, ch0 = heuristic(start)
    g_cost = {start: 0.0}
    parent = {start: None}
    choices: list[HeuristicChoice] = [ch0]

    open_pq = [(h0, 0.0, start)]
    closed: set[tuple[int, int]] = set()
    expansions = 0

    while open_pq:
        f_curr, g_curr, curr = heapq.heappop(open_pq)
        if curr in closed:
            continue
        closed.add(curr)
        expansions += 1

        if curr == goal:
            path: list[tuple[int, int]] = []
            node: Optional[tuple[int, int]] = curr
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return HybridAStarResult(
                path=path,
                expansions=expansions,
                path_length=float(len(path) - 1),
                learned_fraction=gate.learned_fraction,
                h_choices=choices,
            )

        x, y = curr
        for dx, dy in NEIGHBORS_4:
            nx, ny = x + dx, y + dy
            if not in_bounds(nx, ny) or not free(nx, ny):
                continue
            neigh = (nx, ny)
            tg = g_curr + 1.0
            if neigh in g_cost and tg >= g_cost[neigh]:
                continue
            g_cost[neigh] = tg
            parent[neigh] = curr
            h_val, choice = heuristic(neigh)
            choices.append(choice)
            heapq.heappush(open_pq, (tg + h_val, tg, neigh))

    return HybridAStarResult(
        path=None,
        expansions=expansions,
        path_length=float("inf"),
        learned_fraction=gate.learned_fraction,
    )
