"""
naive_replanner.py

Baseline: replan from scratch using classic A* every time an obstacle changes.
Used to compare against D* Lite's incremental approach.
"""

from __future__ import annotations

import heapq
import math
from typing import Optional

import numpy as np

NEIGHBORS_4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


class NaiveReplanner:
    """
    Replans from scratch with A* on every call to replan().

    Parameters
    ----------
    grid  : (H, W) array, 0=free 1=obstacle
    start : (x, y)
    goal  : (x, y)
    """

    def __init__(self, grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]):
        self.grid = grid.copy().astype(np.int32)
        self.start = start
        self.goal = goal
        self.H, self.W = grid.shape
        self.total_expansions: int = 0
        self.replan_count: int = 0

    def _astar(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> tuple[Optional[list[tuple[int, int]]], int]:
        H, W = self.H, self.W

        def in_bounds(x, y):
            return 0 <= x < W and 0 <= y < H

        def free(x, y):
            return int(self.grid[y, x]) == 0

        g: dict[tuple[int, int], float] = {start: 0.0}
        parent: dict[tuple[int, int], Optional[tuple[int, int]]] = {start: None}
        open_pq = [(_heuristic(start, goal), 0.0, start)]
        closed: set[tuple[int, int]] = set()
        expansions = 0

        while open_pq:
            _, g_curr, curr = heapq.heappop(open_pq)
            if curr in closed:
                continue
            closed.add(curr)
            expansions += 1

            if curr == goal:
                path = []
                node: Optional[tuple[int, int]] = curr
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path, expansions

            x, y = curr
            for dx, dy in NEIGHBORS_4:
                nx, ny = x + dx, y + dy
                if not in_bounds(nx, ny) or not free(nx, ny):
                    continue
                neigh = (nx, ny)
                tg = g_curr + 1.0
                if neigh not in g or tg < g[neigh]:
                    g[neigh] = tg
                    parent[neigh] = curr
                    f = tg + _heuristic(neigh, goal)
                    heapq.heappush(open_pq, (f, tg, neigh))

        return None, expansions

    def plan(self) -> Optional[list[tuple[int, int]]]:
        path, exp = self._astar(self.start, self.goal)
        self.total_expansions += exp
        return path

    def update_edge(self, x: int, y: int, blocked: bool) -> None:
        self.grid[y, x] = 1 if blocked else 0

    def replan(self, new_start: Optional[tuple[int, int]] = None) -> Optional[list[tuple[int, int]]]:
        if new_start is not None:
            self.start = new_start
        path, exp = self._astar(self.start, self.goal)
        self.total_expansions += exp
        self.replan_count += 1
        return path
