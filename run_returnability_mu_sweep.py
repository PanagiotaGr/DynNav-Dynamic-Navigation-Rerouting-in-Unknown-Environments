# run_returnability_mu_sweep.py
import numpy as np
import pandas as pd

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

    # world
    I_bottle, wall_cols, door_rows = add_bottleneck_wall(I_grid, wall_I=0.95, door_I=0.6, thickness=2)
    start, goal = pick_start_left_goal_right(free_mask, I_bottle, wall_cols=wall_cols, I_max=0.50)

    # returnability field
    tauR = 0.85
    cfgR = ReturnabilityConfig(tau=tauR, step_cost=1.0, use_8conn=False)
    rc = compute_return_cost_grid(free_mask, I_bottle, base=start, cfg=cfgR)
    R = normalize_returnability(rc, unreachable_cost=cfgR.unreachable_cost)

    lam = 2.0
    tau_cap = 0.95

    mus = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0]
    rows = []

    for mu in mus:
        res = astar_soft_I_R(
            free_mask, I_bottle, R, start, goal,
            lam=lam, mu=mu, step_cost=1.0, tau_cap=tau_cap
        )
        rows.append({
            "mu": mu,
            "success": int(res.success),
            "cost": float(res.cost) if res.success else np.inf,
            "expansions": int(res.expansions),
            "path_len": int(len(res.path)) if res.path else 0,
            "maxI": float(res.max_I),
            "meanI": float(res.mean_I),
            "meanR": float(res.mean_R),
            "reason": res.reason,
        })
        print(f"mu={mu:>4} success={res.success} cost={res.cost:.2f} meanR={res.mean_R:.3f} meanI={res.mean_I:.3f} exp={res.expansions}")

    out = "returnability_mu_sweep.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
