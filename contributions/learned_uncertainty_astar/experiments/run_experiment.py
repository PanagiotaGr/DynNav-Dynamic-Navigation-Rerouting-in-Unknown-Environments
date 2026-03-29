"""
run_experiment.py

Full end-to-end experiment:
  1. Generate dataset from random grids
  2. Train UncertaintyHeuristicNet
  3. Evaluate: classic A* vs learned A* (risk-averse / mean / risk-seeking)
  4. Save results CSV + model

Usage
-----
    python run_experiment.py
    python run_experiment.py --train-grids 500 --eval-grids 200 --epochs 100
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from contributions.learned_uncertainty_astar.code.train import build_dataset, train_model, save_model
from contributions.learned_uncertainty_astar.code.evaluate import evaluate, print_evaluation_table

# make the code/ directory importable
CODE_DIR = Path(__file__).resolve().parent.parent / "code"
sys.path.insert(0, str(CODE_DIR))

from contributions.learned_uncertainty_astar.code.train import build_dataset, train_model, save_model
from contributions.learned_uncertainty_astar.code.evaluate import evaluate, print_evaluation_table

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full experiment: train + eval")
    p.add_argument("--train-grids", type=int,   default=200,
                   help="Grids used to build the training dataset")
    p.add_argument("--eval-grids",  type=int,   default=100,
                   help="Grids used for evaluation")
    p.add_argument("--epochs",      type=int,   default=80)
    p.add_argument("--hidden",      type=int,   default=64)
    p.add_argument("--betas",       type=float, nargs="+",
                   default=[-1.0, -0.5, 0.0, 0.5, 1.0])
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--out-model",   type=str,
                   default=str(RESULTS_DIR / "uncertainty_heuristic.pt"))
    p.add_argument("--out-csv",     type=str,
                   default=str(RESULTS_DIR / "eval_results.csv"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # --- 1. Dataset ---
    print(f"\n[1/3] Building dataset from {args.train_grids} grids (seed={args.seed})...")
    X, y = build_dataset(n_grids=args.train_grids, seed=args.seed)
    print(f"      Dataset: {X.shape[0]:,} samples | {X.shape[1]} features")
    print(f"      Target range: [{y.min():.1f}, {y.max():.1f}]")

    # --- 2. Train ---
    print(f"\n[2/3] Training (epochs={args.epochs}, hidden={args.hidden})...")
    model = train_model(
        X, y,
        hidden_dim=args.hidden,
        epochs=args.epochs,
        seed=args.seed,
    )
    save_model(model, args.out_model)

    # --- 3. Evaluate ---
    print(f"\n[3/3] Evaluating on {args.eval_grids} grids (betas={args.betas})...")
    summaries = evaluate(
        n_grids=args.eval_grids,
        model_path=args.out_model,
        betas=args.betas,
        seed=args.seed + 1,
    )
    print_evaluation_table(summaries)

    # Save CSV
    csv_path = Path(args.out_csv)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "method", "n_queries", "found_rate",
            "mean_expansions", "mean_path_length",
            "mean_suboptimality", "mean_h_std",
        ])
        writer.writeheader()
        for s in summaries:
            writer.writerow({
                "method":             s.method,
                "n_queries":          s.n_queries,
                "found_rate":         f"{s.found_rate:.4f}",
                "mean_expansions":    f"{s.mean_expansions:.2f}",
                "mean_path_length":   f"{s.mean_path_length:.4f}",
                "mean_suboptimality": f"{s.mean_suboptimality:.6f}",
                "mean_h_std":         f"{s.mean_h_std:.6f}",
            })
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
