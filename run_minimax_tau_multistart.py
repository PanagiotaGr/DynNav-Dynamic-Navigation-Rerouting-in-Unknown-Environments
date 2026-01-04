# run_minimax_tau_multistart.py
import numpy as np
import pandas as pd

from irreversibility_map import (
    IrreversibilityConfig,
    build_irreversibility_map,
    load_grid_from_cell_table_csv,
)
from run_irreversibility_bottleneck_sweep import add_bottleneck_wall
from irreversibility_planner import astar_irreversibility_constrained
from irreversibility_safe_mode import SafeModeConfig, run_irreversibility_with_safe_mode
from adaptive_tau_minimax import MinimaxTauConfig, find_min_feasible_tau


def sample_free_cell(rng, free_mask):
    ys, xs = np.where(free_mask)
    idx = rng.integers(0, len(ys))
    return (int(ys[idx]), int(xs[idx]))


def main():
    rng = np.random.default_rng(7)

    path_csv = "coverage_grid_with_uncertainty.csv"
    unc_grid = load_grid_from_cell_table_csv(
        path_csv, value_col="uncertainty", row_col=("row", "col"), fill_nan_with="max"
    )
    free_mask = np.isfinite(unc_grid)

    unc01 = (unc_grid - unc_grid.min()) / (unc_grid.max() - unc_grid.min() + 1e-9)
    feat_density = 1.0 - unc01

    cfgI = IrreversibilityConfig(w_uncert=0.60, w_sparsity=0.25, w_deadend=0.15, deadend_radius=2)
    I_grid = build_irreversibility_map(unc_grid, feat_density, free_mask, cfgI)

    # Bottleneck world (keep it constant so τ* distribution is meaningful but varies with s,g)
    wall_I = 0.95
    door_I = 0.60
    thickness = 2
    I_bottle, wall_cols, door_rows = add_bottleneck_wall(I_grid, wall_I=wall_I, door_I=door_I, thickness=thickness)

    # Settings
    N = 60                      # number of random pairs
    fixed_tau0 = 0.80           # strict threshold
    minimax_margin = 0.01

    sm_cfg = SafeModeConfig(tau_max=1.0, tau_step=0.01, max_tau_gap=0.40, stop_if_infeasible=True)
    mm_cfg = MinimaxTauConfig(tau_lo=0.0, tau_hi=1.0, tol=0.005, max_iters=40, margin=minimax_margin)

    rows = []
    for k in range(N):
        # sample start/goal until distinct
        start = sample_free_cell(rng, free_mask)
        goal = sample_free_cell(rng, free_mask)
        while goal == start:
            goal = sample_free_cell(rng, free_mask)

        # 1) fixed strict hard planner
        fixed = astar_irreversibility_constrained(
            free_mask=free_mask,
            irreversibility_grid=I_bottle,
            start=start,
            goal=goal,
            tau=fixed_tau0,
            step_cost=1.0,
        )

        # 2) safe-mode relaxation starting from fixed_tau0
        dec = run_irreversibility_with_safe_mode(
            free_mask=free_mask,
            I_grid=I_bottle,
            start=start,
            goal=goal,
            tau0=fixed_tau0,
            step_cost=1.0,
            cfg=sm_cfg,
        )

        # 3) minimax tau* estimate + plan at tau_request
        mm = find_min_feasible_tau(free_mask, I_bottle, start, goal, mm_cfg)
        if mm.feasible:
            mm_plan = astar_irreversibility_constrained(
                free_mask=free_mask,
                irreversibility_grid=I_bottle,
                start=start,
                goal=goal,
                tau=float(mm.tau_request),
                step_cost=1.0,
            )
        else:
            mm_plan = None

        rows.append({
            "pair_id": k,
            "start_y": start[0], "start_x": start[1],
            "goal_y": goal[0], "goal_x": goal[1],
            "I_start": float(I_bottle[start]),
            "I_goal": float(I_bottle[goal]),

            # fixed
            "fixed_tau0": fixed_tau0,
            "fixed_success": int(fixed.success),
            "fixed_expansions": int(fixed.expansions),
            "fixed_cost": float(fixed.cost) if np.isfinite(fixed.cost) else np.inf,
            "fixed_reason": str(fixed.reason),

            # safe-mode
            "safe_success": int(dec.success),
            "safe_mode": dec.mode,
            "safe_tau_used": float(dec.tau_used) if dec.tau_used is not None else np.nan,
            "safe_tau_gap": float(dec.tau_gap) if dec.tau_gap is not None else np.nan,
            "safe_expansions": int(dec.expansions),
            "safe_cost": float(dec.cost) if np.isfinite(dec.cost) else np.inf,
            "safe_reason": dec.reason,

            # minimax
            "mm_feasible": int(mm.feasible),
            "tau_star": float(mm.tau_star) if mm.tau_star is not None else np.nan,
            "tau_request": float(mm.tau_request) if mm.tau_request is not None else np.nan,
            "mm_iters": int(mm.iters),
            "mm_success": int(mm_plan.success) if mm_plan is not None else 0,
            "mm_expansions": int(mm_plan.expansions) if mm_plan is not None else 0,
            "mm_cost": float(mm_plan.cost) if (mm_plan is not None and np.isfinite(mm_plan.cost)) else np.inf,
        })

        print(f"[{k+1:02d}/{N}] fixed={fixed.success} safe={dec.mode} mm_tau*={mm.tau_star}")

    out = "minimax_tau_multistart.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
