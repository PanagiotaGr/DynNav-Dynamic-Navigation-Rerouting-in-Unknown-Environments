# plot_nbv_topk_overlay_I.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from irreversibility_map import (
    IrreversibilityConfig,
    build_irreversibility_map,
    load_grid_from_cell_table_csv,
)
from run_irreversibility_bottleneck_sweep import add_bottleneck_wall, pick_start_left_goal_right


def load_topk(path: str):
    df = pd.read_csv(path)
    return list(zip(df["y"].astype(int), df["x"].astype(int)))


def scatter_points(ax, pts, marker, label, color):
    if len(pts) == 0:
        return
    xs = [p[1] for p in pts]
    ys = [p[0] for p in pts]
    ax.scatter(
        xs, ys,
        s=65,
        marker=marker,
        label=label,
        color=color,
        edgecolors="k",
        linewidths=0.4,
        zorder=5,
    )


def main():
    # rebuild the same world
    path_csv = "coverage_grid_with_uncertainty.csv"
    unc_grid = load_grid_from_cell_table_csv(
        path_csv,
        value_col="uncertainty",
        row_col=("row", "col"),
        fill_nan_with="max",
    )
    free_mask = np.isfinite(unc_grid)

    unc01 = (unc_grid - unc_grid.min()) / (unc_grid.max() - unc_grid.min() + 1e-9)
    feat_density = 1.0 - unc01

    cfgI = IrreversibilityConfig(
        w_uncert=0.60,
        w_sparsity=0.25,
        w_deadend=0.15,
        deadend_radius=2,
    )
    I_grid = build_irreversibility_map(unc_grid, feat_density, free_mask, cfgI)

    # bottleneck world
    I_world, wall_cols, door_rows = add_bottleneck_wall(
        I_grid, wall_I=0.95, door_I=0.6, thickness=2
    )

    start, _ = pick_start_left_goal_right(
        free_mask, I_world, wall_cols=wall_cols, I_max=0.50
    )

    # load top-k goal sets
    top_ig = load_topk("nbv_top10_ig.csv")
    top_ig_i = load_topk("nbv_top10_ig_minus_I.csv")
    top_full = load_topk("nbv_top10_full.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    panels = [
        ("Top-10: IG only", top_ig, "tab:red"),
        ("Top-10: IG − αI", top_ig_i, "tab:orange"),
        ("Top-10: IG − αI − β(1−R)", top_full, "tab:green"),
    ]

    for ax, (title, pts, color) in zip(axes, panels):
        im = ax.imshow(I_world, origin="upper", vmin=0, vmax=1, cmap="inferno")
        scatter_points(ax, pts, marker="*", label="top-10 goals", color=color)
        ax.scatter(
            [start[1]], [start[0]],
            s=80,
            marker="o",
            label="base / start",
            color="cyan",
            edgecolors="k",
            linewidths=0.6,
            zorder=6,
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75)
    cbar.set_label("Irreversibility I(s)")

    plt.suptitle("NBV Top-10 goal selection over Irreversibility map", y=1.02)
    plt.tight_layout()
    out = "nbv_topk_overlay_irreversibility.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    print("Saved:", out)
    plt.close()


if __name__ == "__main__":
    main()

