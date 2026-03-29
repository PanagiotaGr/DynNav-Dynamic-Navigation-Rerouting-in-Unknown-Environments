"""
runner.py

Benchmark runner: evaluates multiple navigation algorithms across
multiple environments and seeds, collects metrics, exports CSV.

Supported algorithms
--------------------
- classic_astar   : Euclidean heuristic A*
- dstar_lite      : D* Lite incremental replanning
- naive_replan    : Naive A* replanning from scratch
- hybrid_astar    : Confidence-gated hybrid heuristic (requires torch)

Usage
-----
    python runner.py
    python runner.py --seeds 10 --envs static dynamic --out results/
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Add sibling module paths
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / "realtime_replanning" / "code"))
sys.path.insert(0, str(_HERE.parent.parent / "hybrid_planner" / "code"))

from environments import (
    make_static_env, make_dynamic_env, make_partial_map_env,
    Environment, EnvType,
)
from metrics import BenchmarkResult, count_safety_violations, timed_plan


# ---------------------------------------------------------------------------
# Algorithm wrappers
# ---------------------------------------------------------------------------

def _run_classic_astar(env: Environment) -> tuple[Optional[list], int, float]:
    from naive_replanner import NaiveReplanner
    planner = NaiveReplanner(env.known_grid.copy(), env.start, env.goal)
    path, elapsed = timed_plan(planner.plan)
    return path, planner.total_expansions, elapsed


def _run_dstar_lite(env: Environment) -> tuple[Optional[list], int, float]:
    from dstar_lite import DStarLite
    planner = DStarLite(env.known_grid.copy(), env.start, env.goal)
    path, elapsed = timed_plan(planner.plan)
    return path, planner.expansions, elapsed


def _run_hybrid(env: Environment, model=None) -> tuple[Optional[list], int, float]:
    from hybrid_astar import hybrid_astar
    result, elapsed = timed_plan(
        hybrid_astar, env.known_grid.copy(), env.start, env.goal, model
    )
    exp = result.expansions if result else 0
    return result.path if result else None, exp, elapsed


# ---------------------------------------------------------------------------
# Optimal reference (classic A* on true grid)
# ---------------------------------------------------------------------------

def _optimal_cost(env: Environment) -> float:
    from naive_replanner import NaiveReplanner
    planner = NaiveReplanner(env.grid.copy(), env.start, env.goal)
    path = planner.plan()
    return float(len(path) - 1) if path else float("inf")


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

def benchmark_one(
    method: str,
    env: Environment,
    seed: int,
    model=None,
) -> BenchmarkResult:
    if method == "classic_astar":
        path, exp, ms = _run_classic_astar(env)
    elif method == "dstar_lite":
        path, exp, ms = _run_dstar_lite(env)
    elif method == "hybrid_astar":
        path, exp, ms = _run_hybrid(env, model)
    else:
        raise ValueError(f"Unknown method: {method}")

    found = path is not None
    path_cost = float(len(path) - 1) if found else float("inf")
    opt = _optimal_cost(env)
    subopt = path_cost / opt if (opt > 0 and found) else float("inf")
    violations = count_safety_violations(path, env.grid) if found else 0

    return BenchmarkResult(
        method=method,
        env_type=env.env_type.value,
        found=found,
        path_cost=path_cost,
        optimal_path_cost=opt,
        suboptimality=subopt,
        safety_violations=violations,
        computation_time_ms=ms,
        node_expansions=exp,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Full benchmark sweep
# ---------------------------------------------------------------------------

def run_benchmark(
    methods: list[str],
    env_types: list[str],
    n_seeds: int = 10,
    grid_h: int = 30,
    grid_w: int = 30,
    model=None,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    env_factories = {
        "static":      make_static_env,
        "dynamic":     make_dynamic_env,
        "partial_map": make_partial_map_env,
    }

    for env_name in env_types:
        factory = env_factories[env_name]
        for seed in range(n_seeds):
            env = factory(H=grid_h, W=grid_w, seed=seed)
            for method in methods:
                try:
                    r = benchmark_one(method, env, seed, model)
                    results.append(r)
                except Exception as e:
                    print(f"[WARN] {method} on {env_name} seed={seed}: {e}")

    return results


def save_csv(results: list[BenchmarkResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        return
    fieldnames = list(results[0].as_dict().keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.as_dict())
    print(f"Saved: {path}  ({len(results)} rows)")


def print_summary(results: list[BenchmarkResult]) -> None:
    from collections import defaultdict
    groups: dict[tuple[str, str], list[BenchmarkResult]] = defaultdict(list)
    for r in results:
        groups[(r.method, r.env_type)].append(r)

    print("\n" + "=" * 95)
    print(f"{'Method':<20} {'Env':<14} {'Found':>6} {'PathCost':>10} "
          f"{'Subopt':>8} {'Expans':>8} {'Time(ms)':>10} {'SafeViol':>9}")
    print("=" * 95)
    for (method, env), rs in sorted(groups.items()):
        found = [r for r in rs if r.found]
        fr = len(found) / len(rs)
        pc = np.mean([r.path_cost for r in found]) if found else float("inf")
        so = np.mean([r.suboptimality for r in found]) if found else float("inf")
        ex = np.mean([r.node_expansions for r in found]) if found else 0
        ms = np.mean([r.computation_time_ms for r in rs])
        sv = np.mean([r.safety_violations for r in found]) if found else 0
        print(f"{method:<20} {env:<14} {fr:>6.0%} {pc:>10.2f} "
              f"{so:>8.4f} {ex:>8.0f} {ms:>10.3f} {sv:>9.1f}")
    print("=" * 95 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+",
                   default=["classic_astar", "dstar_lite"])
    p.add_argument("--envs",    nargs="+",
                   default=["static", "dynamic", "partial_map"])
    p.add_argument("--seeds",   type=int, default=10)
    p.add_argument("--out",     type=str,
                   default=str(Path(__file__).parent.parent / "results" / "benchmark.csv"))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    results = run_benchmark(
        methods=args.methods,
        env_types=args.envs,
        n_seeds=args.seeds,
    )
    print_summary(results)
    save_csv(results, args.out)
