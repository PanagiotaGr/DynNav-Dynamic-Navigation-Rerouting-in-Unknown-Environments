# run_returnability_demo.py
import numpy as np
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

    cfgI = IrreversibilityConfig(w_uncert=0.60, w_sparsity=0.25, w_deadend=0.15, deadend_radius=2)
    I_grid = build_irreversibility_map(unc_grid, feat_density, free_mask, cfgI)

    # bottleneck world
    I_bottle, wall_cols, door_rows = add_bottleneck_wall(I_grid, wall_I=0.95, door_I=0.6, thickness=2)
    start, goal = pick_start_left_goal_right(free_mask, I_bottle, wall_cols=wall_cols, I_max=0.50)

    tau = 0.85
    cfgR = ReturnabilityConfig(tau=tau, step_cost=1.0, use_8conn=False)
    rc = compute_return_cost_grid(free_mask, I_bottle, base=start, cfg=cfgR)
    R = normalize_returnability(rc, unreachable_cost=cfgR.unreachable_cost)

    print(f"Start={start}, Goal={goal}, tau={tau}")
    print(f"Returnability stats: min={R.min():.3f} max={R.max():.3f} mean={R.mean():.3f}")

    plt.figure(figsize=(6, 5))
    plt.imshow(R, origin="upper")
    plt.colorbar(label="returnability (1=easy return, 0=unreachable)")
    plt.scatter([start[1]], [start[0]], marker="o", s=60, label="start")
    plt.scatter([goal[1]], [goal[0]], marker="x", s=60, label="goal")
    plt.title("Returnability heatmap under irreversibility constraint")
    plt.legend(loc="lower right")
    plt.tight_layout()
    out = "returnability_heatmap.png"
    plt.savefig(out, dpi=220)
    print("Saved:", out)
    plt.close()


if __name__ == "__main__":
    main()
