# run_nbv_irreversibility_demo.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from irreversibility_map import (
    IrreversibilityConfig,
    build_irreversibility_map,
    load_grid_from_cell_table_csv,
)
from run_irreversibility_bottleneck_sweep import add_bottleneck_wall, pick_start_left_goal_right
from returnability_map import ReturnabilityConfig, compute_return_cost_grid, normalize_returnability
from nbv_irreversibility_scoring import sample_candidates, score_candidates, topk


def main():
    rng = np.random.default_rng(3)

    path_csv = "coverage_grid_with_uncertainty.csv"
    unc_grid = load_grid_from_cell_table_csv(
        path_csv, value_col="uncertainty", row_col=("row", "col"), fill_nan_with="max"
    )
    free_mask = np.isfinite(unc_grid)

    # Build irreversibility base map
    unc01 = (unc_grid - unc_grid.min()) / (unc_grid.max() - unc_grid.min() + 1e-9)
    feat_density = 1.0 - unc01
    cfgI = IrreversibilityConfig(w_uncert=0.60, w_sparsity=0.25, w_deadend=0.15, deadend_radius=2)
    I_grid = build_irreversibility_map(unc_grid, feat_density, free_mask, cfgI)

    # Apply bottleneck world (for visible effect)
    I_world, wall_cols, door_rows = add_bottleneck_wall(I_grid, wall_I=0.95, door_I=0.6, thickness=2)

    # Pick a base for returnability (use a start on left)
    start, goal = pick_start_left_goal_right(free_mask, I_world, wall_cols=wall_cols, I_max=0.50)

    # Returnability field under tau constraint
    tauR = 0.85
    cfgR = ReturnabilityConfig(tau=tauR, step_cost=1.0, use_8conn=False)
    rc = compute_return_cost_grid(free_mask, I_world, base=start, cfg=cfgR)
    R = normalize_returnability(rc, unreachable_cost=cfgR.unreachable_cost)

    # Candidate goals
    candidates = sample_candidates(rng, free_mask, n=500)

    # weights
    alpha = 0.8   # penalty on irreversibility
    beta = 0.5    # penalty on low returnability

    df = score_candidates(candidates, unc_grid, I_world, R, alpha=alpha, beta=beta, r_local=2)
    df.to_csv("nbv_goal_scores.csv", index=False)
    print("Saved: nbv_goal_scores.csv")
    print(f"Start={start}  (base for returnability), tauR={tauR}")
    print(f"alpha={alpha}, beta={beta}")

    # Top-K tables
    t1 = topk(df, "score_IG", 10)
    t2 = topk(df, "score_IG_I", 10)
    t3 = topk(df, "score_IG_I_R", 10)

    t1.to_csv("nbv_top10_ig.csv", index=False)
    t2.to_csv("nbv_top10_ig_minus_I.csv", index=False)
    t3.to_csv("nbv_top10_full.csv", index=False)
    print("Saved: nbv_top10_ig.csv, nbv_top10_ig_minus_I.csv, nbv_top10_full.csv")

    # Scatter plot: IG vs I_local colored by R_local (simple)
    plt.figure(figsize=(6, 5))
    plt.scatter(df["I_local"], df["IG"], s=12, alpha=0.4)
    plt.xlabel("I_local (irreversibility)")
    plt.ylabel("IG surrogate (uncertainty)")
    plt.title("Candidate goals: IG vs irreversibility")
    plt.grid(True, alpha=0.3)
    out = "nbv_scatter_IG_vs_I.png"
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    print("Saved:", out)
    plt.close()

    # Print summary means of top-k
    def summarize(name, tab):
        print(f"\nTop-10 {name}:")
        print(f"  mean IG={tab['IG'].mean():.3f}  mean I={tab['I_local'].mean():.3f}  mean R={tab['R_local'].mean():.3f}")

    summarize("IG only", t1)
    summarize("IG - alpha*I", t2)
    summarize("IG - alpha*I - beta*(1-R)", t3)


if __name__ == "__main__":
    main()
