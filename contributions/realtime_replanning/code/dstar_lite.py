"""
dstar_lite.py

D* Lite — incremental replanning algorithm.

Reference: Koenig & Likhachev, "D* Lite", AAAI 2002.

Key idea
--------
Maintain rhs (one-step lookahead) and g (distance estimate) for every node.
A node is CONSISTENT when g == rhs, OVERCONSISTENT when g > rhs (better path
found), UNDERCONSISTENT when g < rhs (path became worse).

Only inconsistent nodes go into the priority queue.
On an obstacle change, only the affected nodes (and their predecessors) need
updating — not the whole graph.

API
---
    planner = DStarLite(grid, start, goal)
    path = planner.plan()                      # initial plan

    planner.update_edge(x, y, blocked=True)    # obstacle appeared/disappeared
    path = planner.replan()                    # incremental replan
"""

from __future__ import annotations

import heapq
import math
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INF = float("inf")
NEIGHBORS_4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


# ---------------------------------------------------------------------------
# Priority key helpers
# ---------------------------------------------------------------------------

def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# ---------------------------------------------------------------------------
# D* Lite
# ---------------------------------------------------------------------------

class DStarLite:
    """
    D* Lite on a 4-connected 2-D grid.

    Search runs backwards (goal → start) so that when the robot moves,
    only km (key modifier) needs updating, not a full re-initialization.

    Parameters
    ----------
    grid  : (H, W) int array, 0=free 1=obstacle
    start : (x, y)
    goal  : (x, y)
    """

    def __init__(
        self,
        grid: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
    ):
        self.grid = grid.copy().astype(np.int32)
        self.H, self.W = grid.shape
        self.start = start
        self.goal = goal

        self.g: dict[tuple[int, int], float] = {}
        self.rhs: dict[tuple[int, int], float] = {}
        self.km: float = 0.0                 # key modifier (accumulates as robot moves)

        self._heap: list = []                # (k1, k2, node)
        self._in_heap: set[tuple[int, int]] = set()

        self.expansions: int = 0
        self.replan_count: int = 0

        self._initialize()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _g(self, n: tuple[int, int]) -> float:
        return self.g.get(n, INF)

    def _rhs(self, n: tuple[int, int]) -> float:
        return self.rhs.get(n, INF)

    def _key(self, n: tuple[int, int]) -> tuple[float, float]:
        v = min(self._g(n), self._rhs(n))
        return (v + _heuristic(self.start, n) + self.km, v)

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.W and 0 <= y < self.H

    def _free(self, x: int, y: int) -> bool:
        return int(self.grid[y, x]) == 0

    def _neighbors(self, node: tuple[int, int]) -> list[tuple[int, int]]:
        x, y = node
        return [
            (x + dx, y + dy)
            for dx, dy in NEIGHBORS_4
            if self._in_bounds(x + dx, y + dy) and self._free(x + dx, y + dy)
        ]

    def _cost(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        """Edge cost: 1.0 if free, INF if blocked."""
        bx, by = b
        if not self._free(bx, by):
            return INF
        return 1.0

    def _push(self, node: tuple[int, int]) -> None:
        k = self._key(node)
        heapq.heappush(self._heap, (k[0], k[1], node))
        self._in_heap.add(node)

    def _top_key(self) -> tuple[float, float]:
        while self._heap:
            k1, k2, node = self._heap[0]
            curr_k = self._key(node)
            if (k1, k2) == curr_k:
                return (k1, k2)
            heapq.heappop(self._heap)
            if node in self._in_heap:
                heapq.heappush(self._heap, (curr_k[0], curr_k[1], node))
        return (INF, INF)

    def _pop(self) -> Optional[tuple[tuple[int, int], tuple[float, float]]]:
        while self._heap:
            k1, k2, node = heapq.heappop(self._heap)
            curr_k = self._key(node)
            if (k1, k2) == curr_k:
                self._in_heap.discard(node)
                return node, (k1, k2)
            heapq.heappush(self._heap, (curr_k[0], curr_k[1], node))
        return None

    def _update_vertex(self, u: tuple[int, int]) -> None:
        if u != self.goal:
            self.rhs[u] = min(
                self._cost(u, s) + self._g(s)
                for s in self._neighbors(u)
            ) if self._neighbors(u) else INF

        self._in_heap.discard(u)
        if self._g(u) != self._rhs(u):
            self._push(u)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        self.rhs[self.goal] = 0.0
        self.g[self.goal] = INF
        self._push(self.goal)

    def _compute_shortest_path(self) -> None:
        while True:
            top = self._top_key()
            s_key = self._key(self.start)
            if top >= s_key and self._rhs(self.start) == self._g(self.start):
                break

            result = self._pop()
            if result is None:
                break

            u, old_key = result
            self.expansions += 1
            new_key = self._key(u)

            if old_key < new_key:
                self._push(u)
            elif self._g(u) > self._rhs(u):
                self.g[u] = self._rhs(u)
                for pred in self._neighbors(u):
                    self._update_vertex(pred)
            else:
                self.g[u] = INF
                self._update_vertex(u)
                for pred in self._neighbors(u):
                    self._update_vertex(pred)

    def plan(self) -> Optional[list[tuple[int, int]]]:
        """Initial plan. Returns path start→goal or None."""
        self._compute_shortest_path()
        return self._extract_path()

    def update_edge(self, x: int, y: int, blocked: bool) -> None:
        """
        Mark cell (x, y) as blocked or free.
        Does NOT trigger replanning — call replan() afterwards.
        """
        new_val = 1 if blocked else 0
        if int(self.grid[y, x]) == new_val:
            return
        self.grid[y, x] = new_val

        node = (x, y)
        affected = [node] + self._neighbors(node)
        for u in affected:
            self._update_vertex(u)

    def replan(self, new_start: Optional[tuple[int, int]] = None) -> Optional[list[tuple[int, int]]]:
        """
        Incremental replan. Optionally update start (robot moved).
        Only re-expands inconsistent nodes.
        """
        if new_start is not None and new_start != self.start:
            self.km += _heuristic(self.start, new_start)
            self.start = new_start

        self.replan_count += 1
        self._compute_shortest_path()
        return self._extract_path()

    def _extract_path(self) -> Optional[list[tuple[int, int]]]:
        """Follow greedy gradient from start to goal."""
        if self._g(self.start) == INF:
            return None

        path = [self.start]
        current = self.start
        visited = {current}
        max_steps = self.H * self.W

        for _ in range(max_steps):
            if current == self.goal:
                return path

            nbrs = self._neighbors(current)
            if not nbrs:
                return None

            next_node = min(nbrs, key=lambda s: self._cost(current, s) + self._g(s))
            if self._cost(current, next_node) + self._g(next_node) == INF:
                return None
            if next_node in visited:
                return None

            path.append(next_node)
            visited.add(next_node)
            current = next_node

        return None
