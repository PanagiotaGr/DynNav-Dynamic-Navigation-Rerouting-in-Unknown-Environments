# run_returnability_weighted_demo.py
import numpy as np

from irreversibility_map import (
    IrreversibilityConfig,
    build_irreversibility_map,
    load_grid_from_cell_table_csv,
)
from run_irreversibility_bottleneck_sweep import add_bottleneck_wall, pick_start_left_goal_right
from returnability_map import ReturnabilityConfig, compute_return_cost_grid, normalize_returnability
from returnability_planner import astar_soft_I_R


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

    # returnability field under tau constraint
    tauR = 0.85
    cfgR = ReturnabilityConfig(tau=tauR, step_cost=1.0, use_8conn=False)
    rc = compute_return_cost_grid(free_mask, I_bottle, base=start, cfg=cfgR)
    R = normalize_returnability(rc, unreachable_cost=cfgR.unreachable_cost)

    lam = 2.0
    mu = 3.0
    tau_cap = 0.95  # hard envelope: don't allow I > 0.95 (prevents maxI=1.0)

    print(f"Start={start}, Goal={goal}, lam={lam}, mu={mu}, tauR={tauR}, tau_cap={tau_cap}")

    # A) risk-only (mu=0)
    A = astar_soft_I_R(
        free_mask, I_bottle, R, start, goal,
        lam=lam, mu=0.0, step_cost=1.0,
        tau_cap=None
    )
    print("\n[A] geo + lam*I")
    print(
        f"success={A.success} cost={A.cost:.2f} expansions={A.expansions} "
        f"maxI={A.max_I:.3f} meanI={A.mean_I:.3f} meanR={A.mean_R:.3f} "
        f"path_len={len(A.path) if A.path else 0} reason={A.reason}"
    )

    # B) risk + returnability with HARD cap
    B = astar_soft_I_R(
        free_mask, I_bottle, R, start, goal,
        lam=lam, mu=mu, step_cost=1.0,
        tau_cap=tau_cap
    )
    print("\n[B] geo + lam*I + mu*(1-R) + HARD cap(I<=tau_cap)")
    print(
        f"success={B.success} cost={B.cost:.2f} expansions={B.expansions} "
        f"maxI={B.max_I:.3f} meanI={B.mean_I:.3f} meanR={B.mean_R:.3f} "
        f"path_len={len(B.path) if B.path else 0} reason={B.reason}"
    )


if __name__ == "__main__":
    main()
