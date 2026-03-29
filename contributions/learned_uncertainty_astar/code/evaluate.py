"""
evaluate.py

Evaluation of uncertainty-aware A* vs baselines.

Metrics
-------
expansions      : number of nodes expanded (search efficiency)
path_length     : cost of returned path (optimality)
suboptimality   : path_length / optimal_length  (1.0 = optimal)
mean_h_std      : average predicted uncertainty along expanded nodes
found_rate      : fraction of queries where a path was found

Usage
-----
    python evaluate.py                         # quick eval on 50 grids
    python evaluate.py --grids 200 --model results/uncertainty_heuristic.pt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from uncertainty_astar import (
    EuclideanHeuristic,
    LearnedUncertaintyHeuristic,
    astar,
    make_uncertainty_astar,
)
from train import build_dataset, load_model, make_random_grid, _random_free_cell


# ---------------------------------------------------------------------------
# Per-run result
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    method: str
    found: bool
    expansions: int
    path_length: float
    optimal_length: float
    mean_h_std: float = 0.0

    @property
    def suboptimality(self) -> float:
        if self.optimal_length <= 0 or not self.found:
            return float("inf")
        return self.path_length / self.optimal_length


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@dataclass
class EvalSummary:
    method: str
    n_queries: int
    found_rate: float
    mean_expansions: float
    mean_path_length: float
    mean_suboptimality: float
    mean_h_std: float = 0.0

    def __str__(self) -> str:
        return (
            f"[{self.method:30s}] "
            f"found={self.found_rate:.2%}  "
            f"expansions={self.mean_expansions:7.1f}  "
            f"path_len={self.mean_path_length:6.2f}  "
            f"subopt={self.mean_suboptimality:.4f}  "
            f"mean_h_std={self.mean_h_std:.4f}"
        )


def _summarise(method: str, results: list[RunResult]) -> EvalSummary:
    found = [r for r in results if r.found]
    n = len(results)
    return EvalSummary(
        method=method,
        n_queries=n,
        found_rate=len(found) / n if n > 0 else 0.0,
        mean_expansions=float(np.mean([r.expansions for r in found])) if found else 0.0,
        mean_path_length=float(np.mean([r.path_length for r in found])) if found else 0.0,
        mean_suboptimality=float(np.mean([r.suboptimality for r in found])) if found else float("inf"),
        mean_h_std=float(np.mean([r.mean_h_std for r in found])) if found else 0.0,
    )


def evaluate(
    n_grids: int = 50,
    model_path: Optional[str] = None,
    betas: list[float] | None = None,
    grid_h: int = 40,
    grid_w: int = 40,
    obstacle_prob: float = 0.20,
    seed: int = 0,
) -> list[EvalSummary]:
    """
    Compare classic A*, mean-only learned A*, and uncertainty A* (various betas).

    Parameters
    ----------
    n_grids     : number of evaluation grids
    model_path  : path to saved UncertaintyHeuristicNet (.pt)
                  If None, only classic A* is evaluated.
    betas       : list of beta values for uncertainty A*
                  defaults to [-1.0, 0.0, 0.5, 1.0]
    """
    if betas is None:
        betas = [-1.0, 0.0, 0.5, 1.0]

    rng = np.random.default_rng(seed)
    classic_h = EuclideanHeuristic()

    # Load model if provided
    learned_heuristics: dict[str, LearnedUncertaintyHeuristic] = {}
    if model_path is not None and Path(model_path).exists():
        model = load_model(model_path)
        for beta in betas:
            label = f"uncertainty_astar(beta={beta:+.1f})"
            learned_heuristics[label] = make_uncertainty_astar(model, beta=beta)
    else:
        if model_path is not None:
            print(f"[WARN] Model not found at {model_path}. Evaluating classic only.")

    # Accumulate per-method results
    all_results: dict[str, list[RunResult]] = {"classic_astar": []}
    for label in learned_heuristics:
        all_results[label] = []

    for i in range(n_grids):
        grid = make_random_grid(grid_h, grid_w, obstacle_prob, rng)
        start = _random_free_cell(grid, rng)
        goal = _random_free_cell(grid, rng)
        if start == goal:
            continue

        # --- classic (provides optimal reference) ---
        classic_res = astar(grid, start, goal, classic_h, beta=0.0)
        opt_len = classic_res.path_length if classic_res.found else float("inf")
        all_results["classic_astar"].append(RunResult(
            method="classic_astar",
            found=classic_res.found,
            expansions=classic_res.expansions,
            path_length=classic_res.path_length,
            optimal_length=opt_len,
            mean_h_std=0.0,
        ))

        # --- learned variants ---
        for label, h in learned_heuristics.items():
            res = astar(grid, start, goal, h, beta=h.beta)
            mean_std = float(np.mean(res.h_stds)) if res.h_stds else 0.0
            all_results[label].append(RunResult(
                method=label,
                found=res.found,
                expansions=res.expansions,
                path_length=res.path_length,
                optimal_length=opt_len,
                mean_h_std=mean_std,
            ))

    summaries = [_summarise(method, results) for method, results in all_results.items()]
    return summaries


def print_evaluation_table(summaries: list[EvalSummary]) -> None:
    print("\n" + "=" * 90)
    print("EVALUATION RESULTS")
    print("=" * 90)
    for s in summaries:
        print(s)
    print("=" * 90 + "\n")

    # Highlight expansion reduction vs classic
    classic = next((s for s in summaries if s.method == "classic_astar"), None)
    if classic and classic.mean_expansions > 0:
        print("Node expansion reduction vs. classic A*:")
        for s in summaries:
            if s.method == "classic_astar":
                continue
            reduction = 1.0 - s.mean_expansions / classic.mean_expansions
            print(f"  {s.method:40s}: {reduction:+.2%}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate uncertainty A*")
    p.add_argument("--grids",  type=int,   default=50)
    p.add_argument("--model",  type=str,   default="results/uncertainty_heuristic.pt")
    p.add_argument("--betas",  type=float, nargs="+", default=[-1.0, 0.0, 0.5, 1.0])
    p.add_argument("--seed",   type=int,   default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    summaries = evaluate(
        n_grids=args.grids,
        model_path=args.model,
        betas=args.betas,
        seed=args.seed,
    )
    print_evaluation_table(summaries)
