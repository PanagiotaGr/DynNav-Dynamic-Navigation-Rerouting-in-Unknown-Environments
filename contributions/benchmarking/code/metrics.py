"""
metrics.py

Benchmark metrics for navigation algorithm evaluation.

Metrics
-------
path_cost           : total path length (number of steps)
optimal_path_cost   : optimal path length (A* reference)
suboptimality       : path_cost / optimal_path_cost
safety_violations   : steps where robot was within 1 cell of an obstacle
computation_time_ms : wall-clock time for planning (milliseconds)
node_expansions     : total nodes expanded during planning
found               : whether a path was found
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class BenchmarkResult:
    method: str
    env_type: str
    found: bool
    path_cost: float
    optimal_path_cost: float
    suboptimality: float
    safety_violations: int
    computation_time_ms: float
    node_expansions: int
    seed: int

    def as_dict(self) -> dict:
        return {
            "method":               self.method,
            "env_type":             self.env_type,
            "found":                int(self.found),
            "path_cost":            round(self.path_cost, 4),
            "optimal_path_cost":    round(self.optimal_path_cost, 4),
            "suboptimality":        round(self.suboptimality, 6),
            "safety_violations":    self.safety_violations,
            "computation_time_ms":  round(self.computation_time_ms, 3),
            "node_expansions":      self.node_expansions,
            "seed":                 self.seed,
        }


def count_safety_violations(
    path: list[tuple[int, int]],
    grid: np.ndarray,
) -> int:
    """Count path steps where any 4-neighbor is an obstacle."""
    H, W = grid.shape
    violations = 0
    for x, y in path:
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < W and 0 <= ny < H and grid[ny, nx] == 1:
                violations += 1
                break
    return violations


def timed_plan(plan_fn: Callable, *args, **kwargs):
    """
    Call plan_fn(*args, **kwargs), measure wall-clock time.
    Returns (result, elapsed_ms).
    """
    t0 = time.perf_counter()
    result = plan_fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return result, elapsed_ms
