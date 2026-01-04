# plot_nbv_topk_overlay.py
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


def load_topk(path: str):
    df = pd.read_csv(path)
    return list(zip(df["y"].astype(int).tolist(), df["x"].astype(int).tolist()))


def scatter_points(ax, pts, marker, label):
    if len(pts) == 0:
        return
    xs = [p[1] for p in pts]
    ys = [p[0] for p in pts]
    ax.scatter(xs, ys, s=55, marker=marker, label=label, edgecolors="k", linewidths=0.3)


def main():
    # rebuild same world + R field (consistent with demo)
    path_csv = "coverage_grid_with_uncertainty.csv"
    unc_grid = load_grid_from_cell_table_csv(
        path_csv, value_col="uncertainty", row_col=("row", "col"), fill_nan_with="max"
    )
    free_mask = np.isfinite(unc_grid)

    unc01 = (unc_grid - unc_grid.min()) / (unc_grid.max() - unc_grid.min() + 1e-9)
    feat_density = 1.0 - unc01

    cfgI = IrreversibilityConfig(w_uncert=0.60, w_sparsity=0.25, w_deadend=0.15, deadend_radius=2)
    I_grid = build_irreversibility_map(unc_grid, feat_density, free_mask, cfgI)
    I_world, wall_cols, door_rows = add_bottleneck_wall(I_grid, wall_I=0.95, door_I=0.6, thickness=2)

    start, goal = pick_start_left_goal_right(free_mask, I_world, wall_cols=wall_cols, I_max=0.50)

    tauR = 0.85
    cfgR = ReturnabilityConfig(tau=tauR, step_cost=1.0, use_8conn=False)
    rc = compute_return_cost_grid(free_mask, I_world, base=start, cfg=cfgR)
    R = normalize_returnability(rc, unreachable_cost=cfgR.unreachable_cost)

    # load top-k sets
    top_ig = load_topk("nbv_top10_ig.csv")
    top_ig_i = load_topk("nbv_top10_ig_minus_I.csv")
    top_full = load_topk("nbv_top10_full.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    panels = [
        ("Top-10: IG only", top_ig),
        ("Top-10: IG - αI", top_ig_i),
        ("Top-10: IG - αI - β(1-R)", top_full),
    ]

    for ax, (title, pts) in zip(axes, panels):
        ax.imshow(R, origin="upper")
        scatter_points(ax, pts, marker="*", label="top-10 goals")
        ax.scatter([start[1]], [start[0]], s=70, marker="o", label="base/start", edgecolors="k", linewidths=0.5)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    plt.suptitle(f"NBV goal selection over Returnability map (tauR={tauR})", y=1.02)
    plt.tight_layout()
    out = "nbv_topk_overlay_returnability.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    print("Saved:", out)
    plt.close()


if __name__ == "__main__":
    main()
