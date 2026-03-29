"""
replanning_compare.py

Run D* Lite vs Naive A* replanning head-to-head on identical scenarios
and report comparative statistics.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from dstar_lite import DStarLite
from naive_replanner import NaiveReplanner
from moving_obstacle_sim import Simulator, SimResult


# ---------------------------------------------------------------------------
# Grid factory (reuse from contributions/learned_uncertainty_astar)
# ---------------------------------------------------------------------------

def _make_random_grid(
    H: int = 30,
    W: int = 30,
    obstacle_prob: float = 0.15,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
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
    grid[0, 0] = grid[0, W - 1] = grid[H - 1, 0] = grid[H - 1, W - 1] = 0
    return grid


def _random_free(grid: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    H, W = grid.shape
    while True:
        r, c = int(rng.integers(0, H)), int(rng.integers(0, W))
        if grid[r, c] == 0:
            return c, r


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@dataclass
class CompareResult:
    scenario: int
    dstar_reached: bool
    naive_reached: bool
    dstar_expansions: int
    naive_expansions: int
    dstar_replans: int
    naive_replans: int
    dstar_steps: int
    naive_steps: int
    expansion_speedup: float   # naive / dstar  (>1 means dstar is faster)


def run_comparison(
    n_scenarios: int = 30,
    grid_h: int = 30,
    grid_w: int = 30,
    n_moving_obs: int = 3,
    n_random_obs: int = 2,
    max_steps: int = 400,
    seed: int = 0,
) -> list[CompareResult]:
    rng = np.random.default_rng(seed)
    results: list[CompareResult] = []

    for i in range(n_scenarios):
        base_grid = _make_random_grid(grid_h, grid_w, rng=rng)
        start = _random_free(base_grid, rng)
        goal = _random_free(base_grid, rng)
        if start == goal:
            continue

        scenario_seed = int(rng.integers(0, 2**31))

        # --- D* Lite ---
        grid_d = base_grid.copy()
        dstar = DStarLite(grid_d, start, goal)
        sim_d = Simulator(
            grid_d, start, goal, dstar,
            n_moving_obs=n_moving_obs,
            n_random_obs=n_random_obs,
            max_steps=max_steps,
            seed=scenario_seed,
        )
        res_d = sim_d.run("dstar_lite")

        # --- Naive A* ---
        grid_n = base_grid.copy()
        naive = NaiveReplanner(grid_n, start, goal)
        sim_n = Simulator(
            grid_n, start, goal, naive,
            n_moving_obs=n_moving_obs,
            n_random_obs=n_random_obs,
            max_steps=max_steps,
            seed=scenario_seed,
        )
        res_n = sim_n.run("naive_astar")

        exp_d = res_d.total_expansions or 1
        exp_n = res_n.total_expansions or 1
        speedup = exp_n / exp_d

        results.append(CompareResult(
            scenario=i,
            dstar_reached=res_d.reached_goal,
            naive_reached=res_n.reached_goal,
            dstar_expansions=res_d.total_expansions,
            naive_expansions=res_n.total_expansions,
            dstar_replans=res_d.replans,
            naive_replans=res_n.replans,
            dstar_steps=res_d.total_steps,
            naive_steps=res_n.total_steps,
            expansion_speedup=speedup,
        ))

    return results


def print_summary(results: list[CompareResult]) -> None:
    if not results:
        print("No results.")
        return

    reached_d = sum(r.dstar_reached for r in results) / len(results)
    reached_n = sum(r.naive_reached for r in results) / len(results)
    avg_speedup = np.mean([r.expansion_speedup for r in results])
    avg_exp_d = np.mean([r.dstar_expansions for r in results])
    avg_exp_n = np.mean([r.naive_expansions for r in results])

    print("\n" + "=" * 60)
    print("REPLANNING COMPARISON: D* Lite vs Naive A*")
    print("=" * 60)
    print(f"Scenarios evaluated : {len(results)}")
    print(f"D* Lite reach rate  : {reached_d:.2%}")
    print(f"Naive   reach rate  : {reached_n:.2%}")
    print(f"Avg expansions D*   : {avg_exp_d:.1f}")
    print(f"Avg expansions Naive: {avg_exp_n:.1f}")
    print(f"Expansion speedup   : {avg_speedup:.2f}x  (D* Lite vs Naive)")
    print("=" * 60 + "\n")


def save_csv(results: list[CompareResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "scenario", "dstar_reached", "naive_reached",
            "dstar_expansions", "naive_expansions",
            "dstar_replans", "naive_replans",
            "dstar_steps", "naive_steps", "expansion_speedup",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "scenario": r.scenario,
                "dstar_reached": int(r.dstar_reached),
                "naive_reached": int(r.naive_reached),
                "dstar_expansions": r.dstar_expansions,
                "naive_expansions": r.naive_expansions,
                "dstar_replans": r.dstar_replans,
                "naive_replans": r.naive_replans,
                "dstar_steps": r.dstar_steps,
                "naive_steps": r.naive_steps,
                "expansion_speedup": f"{r.expansion_speedup:.4f}",
            })
    print(f"Saved: {path}")
