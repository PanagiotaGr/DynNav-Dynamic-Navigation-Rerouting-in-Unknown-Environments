from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from contributions.learned_uncertainty_astar.code.train import load_model
from contributions.hybrid_learned_astar.code.evaluate import evaluate_methods


def make_grid(size: int = 40, obstacle_prob: float = 0.2, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    grid = (rng.random((size, size)) < obstacle_prob).astype(np.int32)
    start = (0, 0)
    goal = (size - 1, size - 1)
    grid[start] = 0
    grid[goal] = 0
    return grid, start, goal


def build_eval_problems(n: int, size: int, obstacle_prob: float, seed: int):
    rng = np.random.default_rng(seed)
    problems = []
    for _ in range(n):
        grid, start, goal = make_grid(size=size, obstacle_prob=obstacle_prob, rng=rng)
        problems.append({"grid": grid, "start": start, "goal": goal})
    return problems


def save_plots(df: pd.DataFrame, out_dir: str):
    # expansions
    plt.figure(figsize=(10, 5))
    plt.bar(df["method"], df["mean_expansions"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean node expansions")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hybrid_expansions_bar.png"))
    plt.close()

    # tradeoff
    plt.figure(figsize=(6, 5))
    plt.scatter(df["mean_expansions"], df["mean_suboptimality"])
    for _, row in df.iterrows():
        plt.annotate(row["method"], (row["mean_expansions"], row["mean_suboptimality"]))
    plt.xlabel("Mean node expansions")
    plt.ylabel("Mean suboptimality")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hybrid_tradeoff_scatter.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-grids", type=int, default=100)
    parser.add_argument("--grid-size", type=int, default=40)
    parser.add_argument("--obstacle-prob", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-path",
        type=str,
        default="contributions/learned_uncertainty_astar/results/uncertainty_heuristic.pt",
    )
    parser.add_argument(
        "--taus",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0, 3.0],
    )
    args = parser.parse_args()

    out_dir = "contributions/hybrid_learned_astar/results"
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    learned_model = load_model(args.model_path, device=device)

    problems = build_eval_problems(
        n=args.eval_grids,
        size=args.grid_size,
        obstacle_prob=args.obstacle_prob,
        seed=args.seed,
    )

    df = evaluate_methods(
        problems=problems,
        learned_model=learned_model,
        tau_values=args.taus,
        device=device,
    )

    csv_path = os.path.join(out_dir, "hybrid_eval_results.csv")
    df.to_csv(csv_path, index=False)

    print("\nHYBRID EVALUATION RESULTS")
    print(df.to_string(index=False))

    save_plots(df, out_dir)
    print(f"\nSaved CSV to {csv_path}")
    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
