"""
decentralized_nav.py

Decentralized multi-robot navigation with local coordination.

Each robot:
  - Plans independently using A*
  - Broadcasts its (position, intent/next_waypoint) to neighbours within
    communication range
  - Avoids collisions using a priority-based reservation scheme:
      lower robot_id has priority; others yield

Risk sharing
------------
Each robot maintains a local risk map. When a robot detects high-risk cells
(obstacle proximity, uncertainty), it shares that information.
Other robots adjust their cost function accordingly.

Comparison: centralised vs decentralised
-----------------------------------------
Centralised   : joint A* over combined state space (exponential — approximated
                here by sequential planning with full map knowledge)
Decentralised : each robot plans independently with local info exchange
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
# Robot agent
# ---------------------------------------------------------------------------

@dataclass
class RobotAgent:
    robot_id: int
    pos: tuple[int, int]
    goal: tuple[int, int]
    comm_range: float = 10.0      # communication radius
    path: list[tuple[int, int]] = field(default_factory=list)
    reached_goal: bool = False
    steps: int = 0
    replans: int = 0
    collision_avoidance_events: int = 0
    shared_risk: dict[tuple[int, int], float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Local A* with risk map
# ---------------------------------------------------------------------------

def _astar_with_risk(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    risk_map: dict[tuple[int, int], float],
    reserved: set[tuple[int, int]],
    lambda_risk: float = 1.0,
) -> Optional[list[tuple[int, int]]]:
    H, W = grid.shape

    def free(x, y):
        return 0 <= x < W and 0 <= y < H and int(grid[y, x]) == 0

    dist = lambda a, b: math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    g: dict = {start: 0.0}
    parent: dict = {start: None}
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
            neigh = (nx, ny)
            if neigh in reserved:
                continue  # yield to higher priority
            risk = risk_map.get(neigh, 0.0)
            tg = g_c + 1.0 + lambda_risk * risk
            if neigh not in g or tg < g[neigh]:
                g[neigh] = tg
                parent[neigh] = curr
                heapq.heappush(open_pq, (tg + dist(neigh, goal), tg, neigh))

    return None


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class DecentralizedResult:
    strategy: str
    all_reached: bool
    total_steps: int
    total_replans: int
    collision_events: int
    avg_path_length: float


def run_decentralized(
    grid: np.ndarray,
    robots: list[RobotAgent],
    max_steps: int = 300,
    share_risk: bool = True,
    lambda_risk: float = 0.5,
) -> DecentralizedResult:
    """
    Decentralised execution: each robot plans independently.
    Priority: lower robot_id always has right-of-way.
    """
    H, W = grid.shape

    # Initial planning
    for robot in robots:
        path = _astar_with_risk(grid, robot.pos, robot.goal, {}, set(), lambda_risk)
        robot.path = path or []

    collision_events = 0

    for step in range(max_steps):
        if all(r.reached_goal for r in robots):
            break

        # Collect current positions (for collision detection)
        positions = {r.robot_id: r.pos for r in robots}
        intended: dict[int, tuple[int, int]] = {}

        # Determine intended moves (higher priority robots first)
        reserved: set[tuple[int, int]] = set()

        for robot in sorted(robots, key=lambda r: r.robot_id):
            if robot.reached_goal:
                reserved.add(robot.pos)
                continue

            if robot.path and len(robot.path) > 1:
                next_cell = robot.path[1]
                if next_cell in reserved:
                    # Yield: replan around reserved cells
                    new_path = _astar_with_risk(
                        grid, robot.pos, robot.goal,
                        robot.shared_risk, reserved, lambda_risk
                    )
                    robot.path = new_path or [robot.pos]
                    robot.replans += 1
                    robot.collision_avoidance_events += 1
                    collision_events += 1
                    intended[robot.robot_id] = robot.pos
                else:
                    intended[robot.robot_id] = next_cell
            else:
                intended[robot.robot_id] = robot.pos

            reserved.add(intended.get(robot.robot_id, robot.pos))

        # Apply moves and share risk
        for robot in robots:
            if robot.reached_goal:
                continue
            new_pos = intended.get(robot.robot_id, robot.pos)
            robot.pos = new_pos
            robot.steps += 1
            if robot.path and len(robot.path) > 1 and robot.path[1] == new_pos:
                robot.path = robot.path[1:]

            if robot.pos == robot.goal:
                robot.reached_goal = True

        # Risk sharing: broadcast high-risk observations to nearby robots
        if share_risk:
            for r in robots:
                x, y = r.pos
                local_risk = 0.0
                for dx, dy in NEIGHBORS_4:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < W and 0 <= ny < H and int(grid[ny, nx]) == 1:
                        local_risk += 0.25
                if local_risk > 0:
                    for other in robots:
                        if other.robot_id != r.robot_id:
                            d = math.dist(r.pos, other.pos)
                            if d <= r.comm_range:
                                other.shared_risk[r.pos] = local_risk

    all_reached = all(r.reached_goal for r in robots)
    total_steps = sum(r.steps for r in robots)
    total_replans = sum(r.replans for r in robots)
    avg_path = total_steps / max(len(robots), 1)

    return DecentralizedResult(
        strategy="decentralized",
        all_reached=all_reached,
        total_steps=total_steps,
        total_replans=total_replans,
        collision_events=collision_events,
        avg_path_length=avg_path,
    )


def run_centralized(
    grid: np.ndarray,
    robots: list[RobotAgent],
    max_steps: int = 300,
    lambda_risk: float = 0.5,
) -> DecentralizedResult:
    """
    Centralised baseline: sequential planning with full knowledge.
    Robot i plans avoiding cells already claimed by robots 0..i-1.
    """
    reserved: set[tuple[int, int]] = set()
    for robot in sorted(robots, key=lambda r: r.robot_id):
        path = _astar_with_risk(grid, robot.pos, robot.goal, {}, reserved, lambda_risk)
        robot.path = path or [robot.pos]
        reserved.update(robot.path)

    # Execute (same movement loop)
    collision_events = 0
    for step in range(max_steps):
        if all(r.reached_goal for r in robots):
            break
        for robot in robots:
            if robot.reached_goal:
                continue
            if robot.path and len(robot.path) > 1:
                robot.pos = robot.path[1]
                robot.path = robot.path[1:]
                robot.steps += 1
            if robot.pos == robot.goal:
                robot.reached_goal = True

    all_reached = all(r.reached_goal for r in robots)
    return DecentralizedResult(
        strategy="centralized",
        all_reached=all_reached,
        total_steps=sum(r.steps for r in robots),
        total_replans=0,
        collision_events=collision_events,
        avg_path_length=sum(r.steps for r in robots) / max(len(robots), 1),
    )
