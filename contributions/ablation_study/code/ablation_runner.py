"""
ablation_runner.py

Automated ablation study: toggle modules (risk, uncertainty, learned heuristic)
across all 2^N combinations and compare performance.

Each configuration is a dict of boolean flags:
    {
        "use_risk":         True/False,
        "use_uncertainty":  True/False,
        "use_learned":      True/False,
    }

For each combination, a planner is assembled from components and evaluated
on a fixed set of benchmark environments (reused from benchmarking module).

Output
------
- Summary table printed to stdout
- CSV with all results
- Per-metric best/worst configuration highlighted
"""

from __future__ import annotations

import csv
import heapq
import itertools
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "benchmarking" / "code"))
sys.path.insert(0, str(_HERE.parent.parent / "realtime_replanning" / "code"))
sys.path.insert(0, str(_HERE.parent.parent / "learned_uncertainty_astar" / "code"))


# ---------------------------------------------------------------------------
# Minimal self-contained planner for ablation
# ---------------------------------------------------------------------------

NEIGHBORS_4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def _ablated_astar(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    use_risk: bool,
    use_uncertainty: bool,
    use_learned: bool,
    model=None,
    lambda_risk: float = 1.0,
    beta_uncertainty: float = 0.5,
) -> tuple[Optional[list], int]:
    """
    Configurable A* for ablation.

    use_risk        : add obstacle-proximity penalty to g
    use_uncertainty : use h_std in f score (requires model)
    use_learned     : use learned h_mean (requires model)
    """
    H, W = grid.shape

    try:
        import torch
        _torch = torch
        _has_torch = True
    except ImportError:
        _has_torch = False

    def free(x, y):
        return 0 <= x < W and 0 <= y < H and int(grid[y, x]) == 0

    def risk_penalty(x, y):
        if not use_risk:
            return 0.0
        penalty = 0.0
        for dx, dy in NEIGHBORS_4:
            if 0 <= x+dx < W and 0 <= y+dy < H and grid[y+dy, x+dx] == 1:
                penalty += 0.3
        return penalty

    def heuristic(node):
        h_adm = _euclidean(node, goal)
        if not use_learned or model is None or not _has_torch:
            return h_adm
        try:
            from uncertainty_astar import extract_features
            feat = extract_features(node, goal, grid)
            with _torch.no_grad():
                x_t = _torch.from_numpy(feat)
                mean, std = model(x_t)
            h_mean = float(mean.squeeze())
            h_std = float(std.squeeze())
            if use_uncertainty:
                return h_mean + beta_uncertainty * h_std
            return h_mean
        except Exception:
            return h_adm

    g_cost = {start: 0.0}
    parent: dict = {start: None}
    open_pq = [(heuristic(start), 0.0, start)]
    closed: set = set()
    expansions = 0

    while open_pq:
        _, g_curr, curr = heapq.heappop(open_pq)
        if curr in closed:
            continue
        closed.add(curr)
        expansions += 1

        if curr == goal:
            path, node = [], curr
            while node:
                path.append(node); node = parent[node]
            path.reverse()
            return path, expansions

        x, y = curr
        for dx, dy in NEIGHBORS_4:
            nx, ny = x+dx, y+dy
            if not free(nx, ny):
                continue
            neigh = (nx, ny)
            step = 1.0 + risk_penalty(nx, ny)
            tg = g_curr + step
            if neigh not in g_cost or tg < g_cost[neigh]:
                g_cost[neigh] = tg
                parent[neigh] = curr
                f = tg + heuristic(neigh)
                heapq.heappush(open_pq, (f, tg, neigh))

    return None, expansions


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AblationConfig:
    use_risk: bool
    use_uncertainty: bool
    use_learned: bool

    @property
    def name(self) -> str:
        parts = []
        if self.use_risk:        parts.append("risk")
        if self.use_learned:     parts.append("learned")
        if self.use_uncertainty: parts.append("uncert")
        return "+".join(parts) if parts else "baseline"

    @staticmethod
    def all_combinations() -> list["AblationConfig"]:
        flags = ["use_risk", "use_uncertainty", "use_learned"]
        configs = []
        for vals in itertools.product([False, True], repeat=len(flags)):
            configs.append(AblationConfig(**dict(zip(flags, vals))))
        return configs


# ---------------------------------------------------------------------------
# Per-run result
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    config: AblationConfig
    seed: int
    found: bool
    path_cost: float
    optimal_cost: float
    node_expansions: int
    safety_violations: int

    @property
    def suboptimality(self) -> float:
        if not self.found or self.optimal_cost <= 0:
            return float("inf")
        return self.path_cost / self.optimal_cost


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_ablation(
    n_seeds: int = 20,
    grid_h: int = 25,
    grid_w: int = 25,
    model=None,
) -> list[AblationResult]:
    from environments import make_static_env
    from naive_replanner import NaiveReplanner
    from metrics import count_safety_violations

    configs = AblationConfig.all_combinations()
    results: list[AblationResult] = []

    for seed in range(n_seeds):
        env = make_static_env(H=grid_h, W=grid_w, seed=seed)

        # reference optimal
        ref = NaiveReplanner(env.grid.copy(), env.start, env.goal)
        ref_path = ref.plan()
        opt_cost = float(len(ref_path) - 1) if ref_path else float("inf")

        for cfg in configs:
            path, exp = _ablated_astar(
                env.grid.copy(), env.start, env.goal,
                use_risk=cfg.use_risk,
                use_uncertainty=cfg.use_uncertainty,
                use_learned=cfg.use_learned,
                model=model,
            )
            found = path is not None
            pc = float(len(path) - 1) if found else float("inf")
            sv = count_safety_violations(path, env.grid) if found else 0

            results.append(AblationResult(
                config=cfg,
                seed=seed,
                found=found,
                path_cost=pc,
                optimal_cost=opt_cost,
                node_expansions=exp,
                safety_violations=sv,
            ))

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_ablation_table(results: list[AblationResult]) -> None:
    from collections import defaultdict
    groups: dict[str, list[AblationResult]] = defaultdict(list)
    for r in results:
        groups[r.config.name].append(r)

    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print(f"{'Config':<22} {'Found':>6} {'Subopt':>8} {'Expans':>8} {'SafeViol':>9}")
    print("=" * 80)

    rows = []
    for name, rs in groups.items():
        found = [r for r in rs if r.found]
        fr = len(found) / len(rs)
        so = np.mean([r.suboptimality for r in found]) if found else float("inf")
        ex = np.mean([r.node_expansions for r in found]) if found else 0
        sv = np.mean([r.safety_violations for r in found]) if found else 0
        rows.append((name, fr, so, ex, sv))

    rows.sort(key=lambda x: (x[2] if x[2] != float("inf") else 9999))
    for name, fr, so, ex, sv in rows:
        so_str = f"{so:.4f}" if so != float("inf") else "  inf"
        print(f"{name:<22} {fr:>6.0%} {so_str:>8} {ex:>8.0f} {sv:>9.1f}")

    print("=" * 80 + "\n")
    if rows:
        best = rows[0]
        print(f"Best config by suboptimality: [{best[0]}]  subopt={best[2]:.4f}")


def save_ablation_csv(results: list[AblationResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "config", "use_risk", "use_uncertainty", "use_learned",
            "seed", "found", "path_cost", "optimal_cost",
            "suboptimality", "node_expansions", "safety_violations",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "config":           r.config.name,
                "use_risk":         int(r.config.use_risk),
                "use_uncertainty":  int(r.config.use_uncertainty),
                "use_learned":      int(r.config.use_learned),
                "seed":             r.seed,
                "found":            int(r.found),
                "path_cost":        round(r.path_cost, 4) if r.found else "inf",
                "optimal_cost":     round(r.optimal_cost, 4),
                "suboptimality":    round(r.suboptimality, 6) if r.found else "inf",
                "node_expansions":  r.node_expansions,
                "safety_violations": r.safety_violations,
            })
    print(f"Saved: {path}")
