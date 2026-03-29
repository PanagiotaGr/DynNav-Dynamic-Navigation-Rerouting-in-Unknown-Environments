from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    csv_path = "contributions/hybrid_learned_astar/results/hybrid_eval_results.csv"
    out_dir = "contributions/hybrid_learned_astar/results"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # ---------------------------
    # Plot 1: Node expansions
    # ---------------------------
    plt.figure(figsize=(10, 5))
    plt.bar(df["method"], df["mean_expansions"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Mean node expansions")
    plt.title("Search efficiency comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "paper_expansions_comparison.png"), dpi=200)
    plt.close()

    # ---------------------------
    # Plot 2: Suboptimality
    # ---------------------------
    plt.figure(figsize=(10, 5))
    plt.bar(df["method"], df["mean_suboptimality"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Mean suboptimality")
    plt.title("Solution quality comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "paper_suboptimality_comparison.png"), dpi=200)
    plt.close()

    # ---------------------------
    # Plot 3: Trade-off scatter
    # ---------------------------
    plt.figure(figsize=(7, 5))
    plt.scatter(df["mean_expansions"], df["mean_suboptimality"])

    for _, row in df.iterrows():
        plt.annotate(
            row["method"],
            (row["mean_expansions"], row["mean_suboptimality"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )

    plt.xlabel("Mean node expansions")
    plt.ylabel("Mean suboptimality")
    plt.title("Efficiency vs. optimality trade-off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "paper_tradeoff_scatter.png"), dpi=200)
    plt.close()

    # ---------------------------
    # Plot 4: Hybrid tau sweep only
    # ---------------------------
    hybrid_df = df[df["method"].str.contains("hybrid_astar", na=False)].copy()
    if not hybrid_df.empty:
        hybrid_df["tau"] = hybrid_df["method"].str.extract(r"tau=([0-9.]+)").astype(float)

        plt.figure(figsize=(7, 5))
        plt.plot(hybrid_df["tau"], hybrid_df["mean_expansions"], marker="o")
        plt.xlabel("Tau")
        plt.ylabel("Mean node expansions")
        plt.title("Hybrid planner threshold sweep")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "paper_tau_sweep_expansions.png"), dpi=200)
        plt.close()

        plt.figure(figsize=(7, 5))
        plt.plot(hybrid_df["tau"], hybrid_df["mean_suboptimality"], marker="o")
        plt.xlabel("Tau")
        plt.ylabel("Mean suboptimality")
        plt.title("Hybrid planner quality vs. threshold")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "paper_tau_sweep_suboptimality.png"), dpi=200)
        plt.close()

    print("Saved plots to:", out_dir)


if __name__ == "__main__":
    main()
