from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


GridPos = Tuple[int, int]


@dataclass
class SearchResult:
    found: bool
    path: List[GridPos]
    path_cost: float
    expansions: int
    fallback_rate: float
    mean_h_std: float


def reconstruct_path(
    came_from: Dict[GridPos, GridPos],
    current: GridPos,
) -> List[GridPos]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def get_neighbors(node: GridPos, grid: np.ndarray) -> List[GridPos]:
    rows, cols = grid.shape
    r, c = node
    nbrs = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
            nbrs.append((nr, nc))
    return nbrs


def hybrid_astar(
    grid: np.ndarray,
    start: GridPos,
    goal: GridPos,
    heuristic,
) -> SearchResult:
    heuristic.reset_stats()

    open_heap: List[Tuple[float, float, GridPos]] = []
    heapq.heappush(open_heap, (0.0, 0.0, start))

    came_from: Dict[GridPos, GridPos] = {}
    g_score: Dict[GridPos, float] = {start: 0.0}
    closed = set()

    expansions = 0

    while open_heap:
        _, current_g, current = heapq.heappop(open_heap)

        if current in closed:
            continue
        closed.add(current)
        expansions += 1

        if current == goal:
            path = reconstruct_path(came_from, current)
            return SearchResult(
                found=True,
                path=path,
                path_cost=g_score[current],
                expansions=expansions,
                fallback_rate=heuristic.stats.fallback_rate,
                mean_h_std=heuristic.stats.mean_std,
            )

        for nbr in get_neighbors(current, grid):
            tentative_g = current_g + 1.0

            if tentative_g < g_score.get(nbr, float("inf")):
                came_from[nbr] = current
                g_score[nbr] = tentative_g
                h = heuristic(nbr, goal, grid)
                f = tentative_g + h
                heapq.heappush(open_heap, (f, tentative_g, nbr))

    return SearchResult(
        found=False,
        path=[],
        path_cost=float("inf"),
        expansions=expansions,
        fallback_rate=heuristic.stats.fallback_rate,
        mean_h_std=heuristic.stats.mean_std,
    )
