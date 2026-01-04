# plot_nbv_frontier_top10_overlay.py
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


def main():
    path_csv = "coverage_grid_with_uncertainty.csv"
    unc_grid = load_grid_from_cell_table_csv(
        path_csv, value_col="uncertainty", row_col=("row", "col"), fill_nan_with="max"
    )
    free_mask = np.isfinite(unc_grid)

    unc01 = (unc_grid - unc_grid.min()) / (unc_grid.max() - unc_grid.min() + 1e-9)
    feat_density = 1.0 - unc01

    cfgI = IrreversibilityConfig(0.6, 0.25, 0.15, 2)
    I = build_irreversibility_map(unc_grid, feat_density, free_mask, cfgI)
    I_world, wall_cols, _ = add_bottleneck_wall(I, wall_I=0.95, door_I=0.6)

    start, _ = pick_start_left_goal_right(free_mask, I_world, wall_cols, I_max=0.5)

    cfgR = ReturnabilityConfig(tau=0.85)
    rc = compute_return_cost_grid(free_mask, I_world, base=start, cfg=cfgR)
    R = normalize_returnability(rc, unreachable_cost=cfgR.unreachable_cost)

    top = pd.read_csv("nbv_frontier_top10.csv")
    pts = list(zip(top["y"].astype(int), top["x"].astype(int)))

    xs = [p[1] for p in pts]
    ys = [p[0] for p in pts]

    # --- plot on Returnability ---
    plt.figure(figsize=(6, 5))
    plt.imshow(R, origin="upper")
    plt.scatter(xs, ys, s=80, marker="*", edgecolors="k", linewidths=0.4, label="Frontier Top-10")
    plt.scatter([start[1]], [start[0]], s=90, marker="o", edgecolors="k", linewidths=0.6, label="Base/Start")
    plt.title("Frontier NBV Top-10 on Returnability map")
    plt.xticks([]); plt.yticks([])
    plt.legend(loc="lower right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    out1 = "nbv_frontier_top10_on_R.png"
    plt.savefig(out1, dpi=220)
    print("Saved:", out1)
    plt.close()

    # --- plot on Irreversibility ---
    plt.figure(figsize=(6, 5))
    plt.imshow(I_world, origin="upper", vmin=0, vmax=1, cmap="inferno")
    plt.scatter(xs, ys, s=80, marker="*", edgecolors="k", linewidths=0.4, label="Frontier Top-10")
    plt.scatter([start[1]], [start[0]], s=90, marker="o", edgecolors="k", linewidths=0.6, label="Base/Start")
    plt.title("Frontier NBV Top-10 on Irreversibility map")
    plt.xticks([]); plt.yticks([])
    plt.legend(loc="lower right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    out2 = "nbv_frontier_top10_on_I.png"
    plt.savefig(out2, dpi=220)
    print("Saved:", out2)
    plt.close()


if __name__ == "__main__":
    main()
