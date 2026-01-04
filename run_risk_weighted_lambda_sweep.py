# run_risk_weighted_lambda_sweep.py
import numpy as np
import pandas as pd

from irreversibility_map import (
    IrreversibilityConfig,
    build_irreversibility_map,
    load_grid_from_cell_table_csv,
)
from run_irreversibility_bottleneck_sweep import add_bottleneck_wall, pick_start_left_goal_right
from risk_weighted_planner import astar_risk_weighted


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

    cfg = IrreversibilityConfig(w_uncert=0.60, w_sparsity=0.25, w_deadend=0.15, deadend_radius=2)
    I_grid = build_irreversibility_map(
        uncertainty_grid=unc_grid,
        feature_density_grid=feat_density,
        free_mask=free_mask,
        cfg=cfg,
    )

    # Bottleneck configuration (same as before)
    wall_I = 0.95
    door_I = 0.60
    thickness = 2
    I_bottle, wall_cols, door_rows = add_bottleneck_wall(I_grid, wall_I=wall_I, door_I=door_I, thickness=thickness)

    # Fixed start/goal on opposite sides
    start, goal = pick_start_left_goal_right(free_mask, I_bottle, wall_cols=wall_cols, I_max=0.50)

    print(f"Start={start}, Goal={goal}, I_start={I_bottle[start]:.3f}, I_goal={I_bottle[goal]:.3f}")
    print(f"Door rows={door_rows}, wall cols={wall_cols}, door_I={door_I}, wall_I={wall_I}")

    # Lambda sweep (log-ish)
    lams = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]

    rows = []
    for lam in lams:
        res = astar_risk_weighted(
            free_mask=free_mask,
            I_grid=I_bottle,
            start=start,
            goal=goal,
            lam=lam,
            step_cost=1.0,
            risk_agg="sum",
        )

        if res.success:
            path_I = np.array([I_bottle[y, x] for (y, x) in res.path], dtype=float)
            maxI = float(np.max(path_I))
            meanI = float(np.mean(path_I))
            path_len = int(len(res.path))
            geo_cost = float(path_len - 1)  # step_cost=1
        else:
            maxI = np.nan
            meanI = np.nan
            path_len = 0
            geo_cost = np.inf

        rows.append({
            "lambda": float(lam),
            "success": int(res.success),
            "total_cost": float(res.cost) if np.isfinite(res.cost) else np.inf,
            "geo_cost": geo_cost,
            "path_len": path_len,
            "expansions": int(res.expansions),
            "max_I_on_path": maxI,
            "mean_I_on_path": meanI,
            "I_start": float(I_bottle[start]),
            "I_goal": float(I_bottle[goal]),
            "door_I": float(door_I),
            "wall_I": float(wall_I),
            "reason": res.reason,
        })

        print(f"lam={lam:.2f} success={res.success} expansions={res.expansions} maxI={maxI}")

    out_csv = "risk_weighted_lambda_sweep.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
