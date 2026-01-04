# plot_path_overlay_hard_vs_soft.py
import numpy as np
import matplotlib.pyplot as plt

from irreversibility_map import (
    IrreversibilityConfig,
    build_irreversibility_map,
    load_grid_from_cell_table_csv,
)
from run_irreversibility_bottleneck_sweep import add_bottleneck_wall, pick_start_left_goal_right
from irreversibility_planner import astar_irreversibility_constrained
from risk_weighted_planner import astar_risk_weighted


def plot_path(ax, path, label=None, linewidth=2.0):
    if not path:
        return
    ys = [p[0] for p in path]
    xs = [p[1] for p in path]
    ax.plot(xs, ys, linewidth=linewidth, label=label)


def main():
    path_csv = "coverage_grid_with_uncertainty.csv"

    unc_grid = load_grid_from_cell_table_csv(
        path_csv,
        value_col="uncertainty",
        row_col=("row", "col"),
        fill_nan_with="max",
    )
    free_mask = np.isfinite(unc_grid)

    # Proxy feature density
    unc01 = (unc_grid - unc_grid.min()) / (unc_grid.max() - unc_grid.min() + 1e-9)
    feat_density = 1.0 - unc01

    cfg = IrreversibilityConfig(w_uncert=0.60, w_sparsity=0.25, w_deadend=0.15, deadend_radius=2)
    I_grid = build_irreversibility_map(
        uncertainty_grid=unc_grid,
        feature_density_grid=feat_density,
        free_mask=free_mask,
        cfg=cfg,
    )

    # Bottleneck (same params)
    wall_I = 0.95
    door_I = 0.60
    thickness = 2
    I_bottle, wall_cols, door_rows = add_bottleneck_wall(I_grid, wall_I=wall_I, door_I=door_I, thickness=thickness)

    start, goal = pick_start_left_goal_right(free_mask, I_bottle, wall_cols=wall_cols, I_max=0.50)

    # HARD: tau fail/success examples
    tau_fail = 0.80
    tau_ok = 0.90
    hard_fail = astar_irreversibility_constrained(
        free_mask=free_mask,
        irreversibility_grid=I_bottle,
        start=start,
        goal=goal,
        tau=tau_fail,
        step_cost=1.0,
    )
    hard_ok = astar_irreversibility_constrained(
        free_mask=free_mask,
        irreversibility_grid=I_bottle,
        start=start,
        goal=goal,
        tau=tau_ok,
        step_cost=1.0,
    )

    # SOFT: lambda example (high lambda)
    lam = 12.0
    soft = astar_risk_weighted(
        free_mask=free_mask,
        I_grid=I_bottle,
        start=start,
        goal=goal,
        lam=lam,
        step_cost=1.0,
        risk_agg="sum",
    )

    print(f"Start={start} Goal={goal}")
    print(f"HARD tau={tau_fail}: success={hard_fail.success} reason={hard_fail.reason}")
    print(f"HARD tau={tau_ok}:   success={hard_ok.success} expansions={hard_ok.expansions}")
    print(f"SOFT lam={lam}:      success={soft.success} expansions={soft.expansions}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 7))

    # Background: irreversibility heatmap
    im = ax.imshow(I_bottle, origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Irreversibility I(s)")

    # Paths
    if hard_ok.success:
        plot_path(ax, hard_ok.path, label=f"HARD τ={tau_ok} (feasible)", linewidth=2.5)
    if soft.success:
        plot_path(ax, soft.path, label=f"SOFT λ={lam} (feasible)", linewidth=2.5)

    # Mark start/goal
    ax.scatter([start[1]], [start[0]], marker="o", s=80, label="Start")
    ax.scatter([goal[1]], [goal[0]], marker="X", s=90, label="Goal")

    # Title + legend
    ax.set_title("Path overlay: hard irreversibility constraint vs soft risk penalty")
    ax.set_xlabel("col (x)")
    ax.set_ylabel("row (y)")
    ax.legend(loc="upper right")

    out = "path_overlay_hard_vs_soft.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    print("Saved:", out)
    plt.close()


if __name__ == "__main__":
    main()
