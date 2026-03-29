"""
frontier_vs_infogain.py

Compare two exploration strategies:
  1. Frontier-based exploration  : always go to the nearest frontier cell
  2. Information-gain planning   : J = path_cost - lambda * information_gain

Both operate on the same partial map (cells unknown until visited).

Metrics collected over time
---------------------------
- explored_fraction : fraction of free cells revealed
- steps_taken       : total robot steps
- replans           : number of times the robot chose a new target

Visualization hook: explored_fraction_over_time (list per step)
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


NEIGHBORS_4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


# ---------------------------------------------------------------------------
# Map state
# ---------------------------------------------------------------------------

class ExplorationMap:
    """
    Maintains known and unknown cells.
    -1 = unknown, 0 = known free, 1 = known obstacle.
    """

    def __init__(self, true_grid: np.ndarray, sensor_radius: int = 3):
        self.true_grid = true_grid.astype(np.int32)
        self.H, self.W = true_grid.shape
        self.known = np.full((self.H, self.W), -1, dtype=np.int32)
        self.sensor_radius = sensor_radius

    def reveal(self, x: int, y: int) -> int:
        """Reveal cells within sensor_radius. Returns # newly revealed."""
        revealed = 0
        for dy in range(-self.sensor_radius, self.sensor_radius + 1):
            for dx in range(-self.sensor_radius, self.sensor_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.W and 0 <= ny < self.H:
                    if self.known[ny, nx] == -1:
                        self.known[ny, nx] = self.true_grid[ny, nx]
                        revealed += 1
        return revealed

    def frontiers(self) -> list[tuple[int, int]]:
        """Known free cells adjacent to at least one unknown cell."""
        result = []
        for y in range(self.H):
            for x in range(self.W):
                if self.known[y, x] != 0:
                    continue
                for dx, dy in NEIGHBORS_4:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.W and 0 <= ny < self.H and self.known[ny, nx] == -1:
                        result.append((x, y))
                        break
        return result

    @property
    def explored_fraction(self) -> float:
        total_free = int(np.sum(self.true_grid == 0))
        known_free = int(np.sum(self.known == 0))
        return known_free / total_free if total_free > 0 else 1.0

    def unknown_neighbors(self, x: int, y: int) -> int:
        """Count unknown cells within 1 step."""
        count = 0
        for dx, dy in NEIGHBORS_4:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.W and 0 <= ny < self.H and self.known[ny, nx] == -1:
                count += 1
        return count


# ---------------------------------------------------------------------------
# A* on known map
# ---------------------------------------------------------------------------

def _astar_known(
    known: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> Optional[list[tuple[int, int]]]:
    H, W = known.shape

    def free(x, y):
        return 0 <= x < W and 0 <= y < H and known[y, x] == 0

    g: dict = {start: 0.0}
    parent: dict = {start: None}
    dist = lambda a, b: math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    open_pq = [(dist(start, goal), 0.0, start)]
    closed: set = set()

    while open_pq:
        _, g_c, curr = heapq.heappop(open_pq)
        if curr in closed:
            continue
        closed.add(curr)
        if curr == goal:
            path, node = [], curr
            while node:
                path.append(node); node = parent[node]
            path.reverse()
            return path
        x, y = curr
        for dx, dy in NEIGHBORS_4:
            nx, ny = x + dx, y + dy
            if not free(nx, ny):
                continue
            tg = g_c + 1.0
            neigh = (nx, ny)
            if neigh not in g or tg < g[neigh]:
                g[neigh] = tg
                parent[neigh] = curr
                heapq.heappush(open_pq, (tg + dist(neigh, goal), tg, neigh))
    return None


# ---------------------------------------------------------------------------
# Exploration strategies
# ---------------------------------------------------------------------------

def _nearest_frontier(
    emap: ExplorationMap,
    robot: tuple[int, int],
) -> Optional[tuple[int, int]]:
    fronts = emap.frontiers()
    if not fronts:
        return None
    return min(fronts, key=lambda f: math.dist(robot, f))


def _infogain_frontier(
    emap: ExplorationMap,
    robot: tuple[int, int],
    lambda_gain: float = 1.5,
) -> Optional[tuple[int, int]]:
    """
    J(f) = dist(robot, f) - lambda * info_gain(f)
    info_gain = number of unknown cells within sensor radius of f
    """
    fronts = emap.frontiers()
    if not fronts:
        return None

    def info_gain(f: tuple[int, int]) -> float:
        x, y = f
        count = 0
        r = emap.sensor_radius
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < emap.W and 0 <= ny < emap.H and emap.known[ny, nx] == -1:
                    count += 1
        return float(count)

    return min(
        fronts,
        key=lambda f: math.dist(robot, f) - lambda_gain * info_gain(f)
    )


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class ExplorationResult:
    strategy: str
    steps: int
    replans: int
    final_explored_fraction: float
    explored_over_time: list[float] = field(default_factory=list)


def run_exploration(
    true_grid: np.ndarray,
    start: tuple[int, int],
    strategy: str = "frontier",
    max_steps: int = 500,
    sensor_radius: int = 3,
    lambda_gain: float = 1.5,
    seed: int = 0,
) -> ExplorationResult:
    """
    Run exploration until map is fully known or max_steps reached.

    Parameters
    ----------
    strategy : "frontier" or "infogain"
    """
    emap = ExplorationMap(true_grid, sensor_radius)
    robot = start
    emap.reveal(*robot)

    steps = 0
    replans = 0
    explored_over_time = [emap.explored_fraction]
    path_to_target: list[tuple[int, int]] = []
    current_target: Optional[tuple[int, int]] = None

    for _ in range(max_steps):
        if emap.explored_fraction >= 0.99:
            break

        # Select new target if path exhausted or target reached
        if not path_to_target or robot == current_target:
            if strategy == "frontier":
                current_target = _nearest_frontier(emap, robot)
            else:
                current_target = _infogain_frontier(emap, robot, lambda_gain)

            if current_target is None:
                break

            path_to_target = _astar_known(emap.known, robot, current_target) or []
            replans += 1
            if not path_to_target:
                break

        # Move one step
        if len(path_to_target) > 1:
            robot = path_to_target[1]
            path_to_target = path_to_target[1:]
            emap.reveal(*robot)
            steps += 1

        explored_over_time.append(emap.explored_fraction)

    return ExplorationResult(
        strategy=strategy,
        steps=steps,
        replans=replans,
        final_explored_fraction=emap.explored_fraction,
        explored_over_time=explored_over_time,
    )


def compare_exploration(
    true_grid: np.ndarray,
    start: tuple[int, int],
    max_steps: int = 500,
    sensor_radius: int = 3,
    seed: int = 0,
) -> list[ExplorationResult]:
    results = []
    for strategy in ["frontier", "infogain"]:
        r = run_exploration(true_grid, start, strategy=strategy,
                            max_steps=max_steps, sensor_radius=sensor_radius, seed=seed)
        results.append(r)
    return results
