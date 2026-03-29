"""
uncertainty_astar.py

A* search with uncertainty-aware heuristic.

    f(n) = g(n) + h_mean(n) + beta * h_std(n)

beta interpretation:
  beta  > 0  →  risk-averse:   penalises uncertain nodes, explores safer paths
  beta  = 0  →  mean-only:     ignores uncertainty (baseline learned A*)
  beta  < 0  →  risk-seeking:  prefers uncertain nodes, more aggressive search

The classic (Euclidean) A* is included for comparison.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import torch
    from. uncertainty_heuristic_net import UncertaintyHeuristicNet
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    UncertaintyHeuristicNet = None  # type: ignore
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Grid helpers  (identical to module 01 so grids are interchangeable)
# ---------------------------------------------------------------------------

NEIGHBORS_4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _in_bounds(x: int, y: int, W: int, H: int) -> bool:
    return 0 <= x < W and 0 <= y < H


def _is_free(grid: np.ndarray, x: int, y: int) -> bool:
    return int(grid[y, x]) == 0


# ---------------------------------------------------------------------------
# Feature extraction  (11-D, same as module 01)
# ---------------------------------------------------------------------------

def extract_features(
    node: tuple[int, int],
    goal: tuple[int, int],
    grid: np.ndarray,
) -> np.ndarray:
    """
    11-dimensional feature vector for a grid node.

    Features
    --------
    0  dx              signed x-distance to goal
    1  dy              signed y-distance to goal
    2  euclid          Euclidean distance to goal
    3  manhattan       Manhattan distance to goal
    4  chebyshev       Chebyshev distance to goal
    5  free_neighbors  number of free 4-neighbors (0..4)
    6  blocked_nbrs    4 - free_neighbors
    7  obst_density    fraction of 3x3 window that is obstacle
    8  near_obstacle   1.0 if obst_density > 0 else 0.0
    9  norm_x          x / (W-1)
    10 norm_y          y / (H-1)
    """
    x, y = node
    gx, gy = goal
    H, W = grid.shape

    dx, dy = gx - x, gy - y
    euclid = math.sqrt(dx * dx + dy * dy)
    manhattan = abs(dx) + abs(dy)
    chebyshev = max(abs(dx), abs(dy))

    free_neighbors = sum(
        1
        for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        if _in_bounds(nx, ny, W, H) and grid[ny, nx] == 0
    )
    blocked_nbrs = 4 - free_neighbors

    x_min, x_max = max(0, x - 1), min(W, x + 2)
    y_min, y_max = max(0, y - 1), min(H, y + 2)
    window = grid[y_min:y_max, x_min:x_max]
    obst_density = float(np.mean(window != 0)) if window.size > 0 else 0.0
    near_obstacle = 1.0 if obst_density > 0 else 0.0

    norm_x = x / max(W - 1, 1)
    norm_y = y / max(H - 1, 1)

    return np.array(
        [dx, dy, euclid, manhattan, chebyshev,
         free_neighbors, blocked_nbrs, obst_density, near_obstacle,
         norm_x, norm_y],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Heuristic wrappers
# ---------------------------------------------------------------------------

class EuclideanHeuristic:
    """Classic admissible heuristic (lower bound on grid cost)."""

    def h(
        self,
        node: tuple[int, int],
        goal: tuple[int, int],
        grid: np.ndarray,
    ) -> tuple[float, float]:
        """Returns (h_mean, h_std). std=0 for the classic heuristic."""
        x, y = node
        gx, gy = goal
        return math.sqrt((gx - x) ** 2 + (gy - y) ** 2), 0.0


class LearnedUncertaintyHeuristic:
    """
    Wraps UncertaintyHeuristicNet for use inside A*.

    Parameters
    ----------
    model : UncertaintyHeuristicNet
    beta  : float
        Risk parameter. See module docstring.
    """

    def __init__(
        self,
        model,
        beta: float = 0.0,
        device=None,
    ):
        self.model = model
        self.beta = beta
        self.device = device or (torch.device("cpu") if _TORCH_AVAILABLE else None)
        self.model.to(self.device)
        self.model.eval()

    def h(
        self,
        node: tuple[int, int],
        goal: tuple[int, int],
        grid: np.ndarray,
    ) -> tuple[float, float]:
        """Returns (h_mean, h_std) from the network."""
        feat = extract_features(node, goal, grid)
        with torch.no_grad():
            x_t = torch.from_numpy(feat).to(self.device)
            mean, std = self.model(x_t)
        return float(mean.squeeze()), float(std.squeeze())

    def f_score(self, g: float, h_mean: float, h_std: float) -> float:
        """f(n) = g(n) + h_mean(n) + beta * h_std(n)"""
        return g + h_mean + self.beta * h_std


# ---------------------------------------------------------------------------
# A* search
# ---------------------------------------------------------------------------

@dataclass
class AStarResult:
    path: Optional[list[tuple[int, int]]]
    expansions: int
    path_length: float
    h_means: list[float] = field(default_factory=list)
    h_stds: list[float] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.path is not None


def astar(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    heuristic,
    beta: float = 0.0,
) -> AStarResult:
    """
    General A* that works with any heuristic implementing `.h(node, goal, grid)`.

    For LearnedUncertaintyHeuristic:
        f(n) = g(n) + h_mean(n) + beta * h_std(n)

    For EuclideanHeuristic:
        f(n) = g(n) + euclidean(n, goal)   (standard A*)

    Parameters
    ----------
    grid      : (H, W) numpy array, 0=free 1=obstacle
    start     : (x, y)
    goal      : (x, y)
    heuristic : object with .h(node, goal, grid) -> (h_mean, h_std)
    beta      : risk parameter (used only for LearnedUncertaintyHeuristic)

    Returns
    -------
    AStarResult
    """
    H, W = grid.shape

    g_cost: dict[tuple[int, int], float] = {start: 0.0}
    parent: dict[tuple[int, int], Optional[tuple[int, int]]] = {start: None}
    h_means_log: list[float] = []
    h_stds_log: list[float] = []

    h_mean0, h_std0 = heuristic.h(start, goal, grid)
    f0 = g_cost[start] + h_mean0 + beta * h_std0
    open_pq: list[tuple[float, float, tuple[int, int]]] = [(f0, 0.0, start)]
    closed: set[tuple[int, int]] = set()
    expansions = 0

    while open_pq:
        f_curr, g_curr, curr = heapq.heappop(open_pq)
        x, y = curr

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
            return AStarResult(
                path=path,
                expansions=expansions,
                path_length=float(len(path) - 1),
                h_means=h_means_log,
                h_stds=h_stds_log,
            )

        for dx, dy in NEIGHBORS_4:
            nx, ny = x + dx, y + dy
            if not _in_bounds(nx, ny, W, H):
                continue
            if not _is_free(grid, nx, ny):
                continue

            neigh = (nx, ny)
            tentative_g = g_curr + 1.0

            if neigh in g_cost and tentative_g >= g_cost[neigh]:
                continue

            g_cost[neigh] = tentative_g
            parent[neigh] = curr

            h_mean, h_std = heuristic.h(neigh, goal, grid)
            h_means_log.append(h_mean)
            h_stds_log.append(h_std)

            f_val = tentative_g + h_mean + beta * h_std
            heapq.heappush(open_pq, (f_val, tentative_g, neigh))

    return AStarResult(
        path=None,
        expansions=expansions,
        path_length=float("inf"),
    )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_uncertainty_astar(
    model,
    beta: float = 0.0,
) -> LearnedUncertaintyHeuristic:
    """
    Factory for the three operating modes:

        beta > 0  →  risk-averse
        beta = 0  →  mean-only (no uncertainty)
        beta < 0  →  risk-seeking
    """
    return LearnedUncertaintyHeuristic(model=model, beta=beta)
