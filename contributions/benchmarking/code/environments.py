"""
environments.py

Three benchmark environment types for navigation algorithm evaluation.

StaticEnvironment    : fixed obstacles, full map known
DynamicEnvironment   : obstacles appear/move during execution
PartialMapEnvironment: robot starts with unknown map, discovers via sensor radius
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class EnvType(Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    PARTIAL_MAP = "partial_map"


@dataclass
class Environment:
    env_type: EnvType
    grid: np.ndarray          # (H, W), 0=free 1=obstacle
    start: tuple[int, int]
    goal: tuple[int, int]
    known_grid: np.ndarray    # what the robot currently knows (starts unknown for partial)
    sensor_radius: int = 4    # used for partial-map only

    @property
    def H(self) -> int:
        return self.grid.shape[0]

    @property
    def W(self) -> int:
        return self.grid.shape[1]

    def update_known_map(self, robot_pos: tuple[int, int]) -> int:
        """
        Reveal cells within sensor_radius around robot_pos.
        Returns number of newly revealed cells.
        """
        x, y = robot_pos
        revealed = 0
        for dy in range(-self.sensor_radius, self.sensor_radius + 1):
            for dx in range(-self.sensor_radius, self.sensor_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.W and 0 <= ny < self.H:
                    if self.known_grid[ny, nx] == -1:
                        self.known_grid[ny, nx] = self.grid[ny, nx]
                        revealed += 1
        return revealed


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _make_grid(
    H: int, W: int, obstacle_prob: float, rng: np.random.Generator
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=np.int32)
    n_walls = int(H * W * obstacle_prob / 6)
    for _ in range(n_walls):
        r = int(rng.integers(1, H - 1))
        c = int(rng.integers(1, W - 1))
        length = int(rng.integers(3, 8))
        if bool(rng.integers(0, 2)):
            grid[r, c: min(c + length, W - 1)] = 1
        else:
            grid[r: min(r + length, H - 1), c] = 1
    for corner in [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]:
        grid[corner] = 0
    return grid


def _random_free(grid: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    H, W = grid.shape
    for _ in range(10000):
        r, c = int(rng.integers(0, H)), int(rng.integers(0, W))
        if grid[r, c] == 0:
            return c, r
    raise RuntimeError("No free cell found")


def make_static_env(
    H: int = 30, W: int = 30,
    obstacle_prob: float = 0.20,
    seed: int = 0,
) -> Environment:
    rng = np.random.default_rng(seed)
    grid = _make_grid(H, W, obstacle_prob, rng)
    start = _random_free(grid, rng)
    goal = _random_free(grid, rng)
    return Environment(
        env_type=EnvType.STATIC,
        grid=grid,
        start=start,
        goal=goal,
        known_grid=grid.copy(),
    )


def make_dynamic_env(
    H: int = 30, W: int = 30,
    obstacle_prob: float = 0.15,
    seed: int = 0,
) -> Environment:
    rng = np.random.default_rng(seed)
    grid = _make_grid(H, W, obstacle_prob, rng)
    start = _random_free(grid, rng)
    goal = _random_free(grid, rng)
    return Environment(
        env_type=EnvType.DYNAMIC,
        grid=grid,
        start=start,
        goal=goal,
        known_grid=grid.copy(),
    )


def make_partial_map_env(
    H: int = 30, W: int = 30,
    obstacle_prob: float = 0.20,
    sensor_radius: int = 4,
    seed: int = 0,
) -> Environment:
    rng = np.random.default_rng(seed)
    grid = _make_grid(H, W, obstacle_prob, rng)
    start = _random_free(grid, rng)
    goal = _random_free(grid, rng)
    known = np.full((H, W), -1, dtype=np.int32)  # -1 = unknown
    env = Environment(
        env_type=EnvType.PARTIAL_MAP,
        grid=grid,
        start=start,
        goal=goal,
        known_grid=known,
        sensor_radius=sensor_radius,
    )
    env.update_known_map(start)
    return env
