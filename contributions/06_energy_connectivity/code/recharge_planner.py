"""
recharge_planner.py

Energy-aware planner with recharge stations.

The robot has a limited battery (e0 joules). Each move consumes energy.
The planner finds the energy-feasible path to the goal, stopping at
recharge stations if necessary.

Comparison
----------
- shortest_path   : standard A* ignoring energy
- energy_optimal  : A* with energy consumption in cost
- recharge_aware  : A* that detours to stations when battery is low

State = (grid_cell, battery_level_discrete)
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


NEIGHBORS_4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


# ---------------------------------------------------------------------------
# Energy model
# ---------------------------------------------------------------------------

@dataclass
class BatteryConfig:
    capacity: float = 100.0      # full battery
    move_cost: float = 1.0       # energy per free step
    risk_coeff: float = 2.0      # extra energy in risky / obstacle-adjacent cells
    charge_rate: float = 100.0   # full recharge per stop at station


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

def energy_astar(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    stations: list[tuple[int, int]],
    battery: BatteryConfig,
    mode: str = "recharge_aware",
    risk_grid: Optional[np.ndarray] = None,
) -> dict:
    """
    Energy-aware A* with optional recharge stations.

    Parameters
    ----------
    grid      : (H, W) 0=free 1=obstacle
    start     : (x, y)
    goal      : (x, y)
    stations  : list of recharge station positions (x, y)
    battery   : BatteryConfig
    mode      : "shortest"        → ignore energy (classic A*)
                "energy_optimal"  → minimise energy, fail if exhausted
                "recharge_aware"  → allow recharge stops
    risk_grid : optional (H, W) float array, risk ∈ [0,1] per cell

    Returns
    -------
    dict: found, path, path_length, energy_used, recharge_stops, expansions
    """
    H, W = grid.shape
    station_set = set(stations)

    def in_bounds(x, y):
        return 0 <= x < W and 0 <= y < H

    def free(x, y):
        return int(grid[y, x]) == 0

    def step_energy(x, y) -> float:
        if mode == "shortest":
            return 0.0
        base = battery.move_cost
        if risk_grid is not None:
            base += battery.risk_coeff * float(risk_grid[y, x])
        return base

    def h(pos: tuple[int, int]) -> float:
        return math.sqrt((pos[0]-goal[0])**2 + (pos[1]-goal[1])**2)

    # State: (x, y, battery_level)   battery_level quantised to int (×10)
    SCALE = 10
    init_bat = int(round(battery.capacity * SCALE))
    max_bat = init_bat

    g_cost: dict[tuple[int, int, int], float] = {}
    parent: dict[tuple[int, int, int], Optional[tuple[int, int, int]]] = {}

    s0 = (start[0], start[1], init_bat)
    g_cost[s0] = 0.0
    parent[s0] = None

    open_pq = [(h(start), 0.0, s0)]
    closed: set = set()
    expansions = 0
    recharge_stops = [0]

    while open_pq:
        _, g_curr, state = heapq.heappop(open_pq)
        if state in closed:
            continue
        closed.add(state)
        expansions += 1

        x, y, bat = state

        if (x, y) == goal:
            path, node = [], state
            stops = 0
            while node is not None:
                px, py, _ = node
                path.append((px, py))
                pnode = parent[node]
                if pnode and (px, py) in station_set and (px, py) != start:
                    stops += 1
                node = pnode
            path.reverse()
            energy_used = battery.capacity - bat / SCALE
            return {
                "found": True,
                "path": path,
                "path_length": len(path) - 1,
                "energy_used": round(energy_used, 3),
                "recharge_stops": stops,
                "expansions": expansions,
                "mode": mode,
            }

        for dx, dy in NEIGHBORS_4:
            nx, ny = x + dx, y + dy
            if not in_bounds(nx, ny) or not free(nx, ny):
                continue

            e = step_energy(nx, ny)
            new_bat = bat - int(round(e * SCALE))

            # Check if at recharge station
            if (nx, ny) in station_set and mode == "recharge_aware":
                new_bat = max_bat

            if new_bat < 0 and mode != "shortest":
                continue  # energy exhausted

            new_bat = max(0, min(max_bat, new_bat))
            neigh_state = (nx, ny, new_bat)

            cost_increment = 1.0 if mode == "shortest" else (1.0 + e)
            tg = g_curr + cost_increment

            if neigh_state not in g_cost or tg < g_cost[neigh_state]:
                g_cost[neigh_state] = tg
                parent[neigh_state] = state
                f = tg + h((nx, ny))
                heapq.heappush(open_pq, (f, tg, neigh_state))

    return {"found": False, "path": None, "path_length": float("inf"),
            "energy_used": float("inf"), "recharge_stops": 0,
            "expansions": expansions, "mode": mode}


def compare_energy_modes(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    stations: list[tuple[int, int]],
    battery: Optional[BatteryConfig] = None,
) -> list[dict]:
    """Compare shortest, energy_optimal, and recharge_aware planners."""
    if battery is None:
        battery = BatteryConfig()
    results = []
    for mode in ["shortest", "energy_optimal", "recharge_aware"]:
        r = energy_astar(grid, start, goal, stations, battery, mode=mode)
        results.append(r)
    return results
