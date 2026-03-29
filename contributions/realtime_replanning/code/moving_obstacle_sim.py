"""
moving_obstacle_sim.py

Simulation of a robot navigating a grid while dynamic obstacles
appear, disappear, and move.

Obstacle models
---------------
- RandomObstacle: appears/disappears at random cells each step
- MovingObstacle: drifts in a random-walk direction

The Simulator drives execution:
  1. Robot follows current path (one step per tick).
  2. Obstacles update their positions.
  3. If any obstacle blocks the robot's next cell → replan.
  4. Robot calls planner.replan() (either DStarLite or NaiveReplanner).

Metrics collected per run
--------------------------
- total_steps   : steps taken by the robot
- replans       : number of replanning calls
- expansions    : total node expansions across all replans
- reached_goal  : whether the robot reached the goal
- path_blocked  : number of times the current path was invalidated
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Obstacle types
# ---------------------------------------------------------------------------

class RandomObstacle:
    """Appears and disappears at random free cells."""

    def __init__(self, grid: np.ndarray, rng: random.Random):
        self.H, self.W = grid.shape
        self.rng = rng
        self.pos: Optional[tuple[int, int]] = None

    def step(self, grid: np.ndarray) -> tuple[Optional[tuple[int, int]], Optional[tuple[int, int]]]:
        """Returns (old_pos, new_pos). old_pos is freed, new_pos is blocked."""
        old = self.pos
        if old is not None:
            grid[old[1], old[0]] = 0

        if self.rng.random() < 0.7:
            x = self.rng.randint(0, self.W - 1)
            y = self.rng.randint(0, self.H - 1)
            if grid[y, x] == 0:
                self.pos = (x, y)
                grid[y, x] = 1
            else:
                self.pos = None
        else:
            self.pos = None

        return old, self.pos


class MovingObstacle:
    """Moves with a persistent direction (random walk)."""

    DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def __init__(
        self,
        grid: np.ndarray,
        x: int,
        y: int,
        rng: random.Random,
        direction_persist: float = 0.7,
    ):
        self.H, self.W = grid.shape
        self.rng = rng
        self.pos = (x, y)
        self.direction = self.rng.choice(self.DIRS)
        self.persist = direction_persist
        grid[y, x] = 1

    def step(self, grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
        old = self.pos

        if self.rng.random() > self.persist:
            self.direction = self.rng.choice(self.DIRS)

        dx, dy = self.direction
        nx, ny = old[0] + dx, old[1] + dy

        if 0 <= nx < self.W and 0 <= ny < self.H and grid[ny, nx] == 0:
            grid[old[1], old[0]] = 0
            self.pos = (nx, ny)
            grid[ny, nx] = 1
        else:
            self.direction = self.rng.choice(self.DIRS)

        return old, self.pos


# ---------------------------------------------------------------------------
# Simulation result
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    method: str
    reached_goal: bool
    total_steps: int
    replans: int
    total_expansions: int
    path_blocked: int
    path_history: list[tuple[int, int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class Simulator:
    """
    Drives a robot along a dynamic grid using any replanner.

    Parameters
    ----------
    grid            : base (H, W) grid (modified in-place during simulation)
    start, goal     : robot start/goal
    planner         : DStarLite or NaiveReplanner instance (already initialised)
    n_moving_obs    : number of MovingObstacle instances
    n_random_obs    : number of RandomObstacle instances
    max_steps       : safety cap on simulation length
    seed            : RNG seed for reproducibility
    """

    def __init__(
        self,
        grid: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
        planner,
        n_moving_obs: int = 3,
        n_random_obs: int = 2,
        max_steps: int = 500,
        seed: int = 0,
    ):
        self.grid = grid
        self.H, self.W = grid.shape
        self.start = start
        self.goal = goal
        self.planner = planner
        self.max_steps = max_steps
        self.rng = random.Random(seed)

        self.moving_obs: list[MovingObstacle] = []
        self.random_obs: list[RandomObstacle] = []
        self._place_obstacles(n_moving_obs, n_random_obs)

    def _place_obstacles(self, n_moving: int, n_random: int) -> None:
        free_cells = [
            (x, y)
            for y in range(self.H)
            for x in range(self.W)
            if self.grid[y, x] == 0
            and (x, y) != self.start
            and (x, y) != self.goal
        ]
        self.rng.shuffle(free_cells)

        for x, y in free_cells[:n_moving]:
            self.moving_obs.append(
                MovingObstacle(self.grid, x, y, self.rng)
            )
        for _ in range(n_random):
            self.random_obs.append(RandomObstacle(self.grid, self.rng))

    def _path_clear(self, path: list[tuple[int, int]]) -> bool:
        """Check that the next few steps of the path are still free."""
        for x, y in path[:3]:
            if (x, y) != self.goal and int(self.grid[y, x]) == 1:
                return False
        return True

    def _collect_changed_cells(
        self,
        old_positions: list[Optional[tuple[int, int]]],
        new_positions: list[Optional[tuple[int, int]]],
    ) -> list[tuple[tuple[int, int], bool]]:
        changes: list[tuple[tuple[int, int], bool]] = []
        for old, new in zip(old_positions, new_positions):
            if old is not None:
                changes.append((old, False))
            if new is not None:
                changes.append((new, True))
        return changes

    def run(self, method_name: str = "planner") -> SimResult:
        """
        Execute the full simulation.

        Returns
        -------
        SimResult with all collected metrics.
        """
        robot = self.start
        path = self.planner.plan()
        replans = 0
        path_blocked = 0
        total_steps = 0
        path_history = [robot]

        if path is None:
            return SimResult(
                method=method_name,
                reached_goal=False,
                total_steps=0,
                replans=0,
                total_expansions=self.planner.total_expansions
                    if hasattr(self.planner, "total_expansions")
                    else self.planner.expansions,
                path_blocked=0,
            )

        for _ in range(self.max_steps):
            if robot == self.goal:
                break

            # --- move one step ---
            if len(path) > 1:
                next_cell = path[1]
                if int(self.grid[next_cell[1], next_cell[0]]) == 0:
                    robot = next_cell
                    path = path[1:]
                    total_steps += 1
                    path_history.append(robot)
                else:
                    path_blocked += 1

            # --- update obstacles ---
            old_pos_list, new_pos_list = [], []
            for obs in self.moving_obs:
                old, new = obs.step(self.grid)
                old_pos_list.append(old)
                new_pos_list.append(new)
            for obs in self.random_obs:
                old, new = obs.step(self.grid)
                old_pos_list.append(old)
                new_pos_list.append(new)

            changes = self._collect_changed_cells(old_pos_list, new_pos_list)

            # --- check if replan needed ---
            replan_needed = not self._path_clear(path) if path else True

            if replan_needed and changes:
                for cell, blocked in changes:
                    self.planner.update_edge(cell[0], cell[1], blocked)
                new_path = self.planner.replan(new_start=robot)
                replans += 1
                if new_path is not None:
                    path = new_path

        exp = (
            self.planner.total_expansions
            if hasattr(self.planner, "total_expansions")
            else self.planner.expansions
        )

        return SimResult(
            method=method_name,
            reached_goal=(robot == self.goal),
            total_steps=total_steps,
            replans=replans,
            total_expansions=exp,
            path_blocked=path_blocked,
            path_history=path_history,
        )
