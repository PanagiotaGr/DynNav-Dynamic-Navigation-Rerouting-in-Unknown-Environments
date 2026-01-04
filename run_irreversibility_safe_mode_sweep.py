# run_irreversibility_safe_mode_sweep.py
import numpy as np
import pandas as pd

from irreversibility_map import (
    IrreversibilityConfig,
    build_irreversibility_map,
    load_grid_from_cell_table_csv,
)
from run_irreversibility_bottleneck_sweep import add_bottleneck_wall, pick_start_left_goal_right
from irreversibility_safe_mode import SafeModeConfig, run_irreversibility_with_safe_mode


def main():
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

    cfgI = IrreversibilityConfig(w_uncert=0.60, w_sparsity=0.25, w_deadend=0.15, deadend_radius=2)
    I_grid = build_irreversibility_map(
        uncertainty_grid=unc_grid,
        feature_density_grid=feat_density,
        free_mask=free_mask,
        cfg=cfgI,
    )

    wall_I = 0.95
    door_I = 0.60
    thickness = 2
    I_bottle, wall_cols, door_rows = add_bottleneck_wall(I_grid, wall_I=wall_I, door_I=door_I, thickness=thickness)

    start, goal = pick_start_left_goal_right(free_mask, I_bottle, wall_cols=wall_cols, I_max=0.50)

    sm_cfg = SafeModeConfig(tau_max=1.0, tau_step=0.01, max_tau_gap=0.30, stop_if_infeasible=True)

    tau0_list = np.round(np.arange(0.70, 0.96, 0.01), 2)

    rows = []
    for tau0 in tau0_list:
        dec = run_irreversibility_with_safe_mode(
            free_mask=free_mask,
            I_grid=I_bottle,
            start=start,
            goal=goal,
            tau0=float(tau0),
            step_cost=1.0,
            cfg=sm_cfg,
        )
        rows.append({
            "tau0": float(tau0),
            "success": int(dec.success),
            "mode": dec.mode,
            "tau_used": dec.tau_used if dec.tau_used is not None else np.nan,
            "tau_gap": dec.tau_gap if dec.tau_gap is not None else np.nan,
            "expansions": int(dec.expansions),
            "cost": float(dec.cost) if np.isfinite(dec.cost) else np.inf,
            "path_len": int(len(dec.path)),
            "max_I_on_path": dec.max_I_on_path if dec.max_I_on_path is not None else np.nan,
            "mean_I_on_path": dec.mean_I_on_path if dec.mean_I_on_path is not None else np.nan,
            "reason": dec.reason,
            "I_start": float(I_bottle[start]),
            "I_goal": float(I_bottle[goal]),
            "door_I": float(door_I),
            "wall_I": float(wall_I),
        })
        print(f"tau0={tau0:.2f} mode={dec.mode} success={dec.success} tau_used={dec.tau_used}")

    out = "irreversibility_safe_mode_sweep.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
