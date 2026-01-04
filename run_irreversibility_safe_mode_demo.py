# run_irreversibility_safe_mode_demo.py
import numpy as np

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

    # Bottleneck (same as before)
    wall_I = 0.95
    door_I = 0.60
    thickness = 2
    I_bottle, wall_cols, door_rows = add_bottleneck_wall(I_grid, wall_I=wall_I, door_I=door_I, thickness=thickness)

    start, goal = pick_start_left_goal_right(free_mask, I_bottle, wall_cols=wall_cols, I_max=0.50)

    print(f"Start={start}, Goal={goal}, I_start={I_bottle[start]:.3f}, I_goal={I_bottle[goal]:.3f}")
    print(f"Door rows={door_rows}, wall cols={wall_cols}, door_I={door_I}, wall_I={wall_I}")

    # request a too-strict tau to force safe-mode activation
    tau0 = 0.80

    sm_cfg = SafeModeConfig(
        tau_max=1.0,
        tau_step=0.01,
        max_tau_gap=0.30,
        stop_if_infeasible=True,
    )

    dec = run_irreversibility_with_safe_mode(
        free_mask=free_mask,
        I_grid=I_bottle,
        start=start,
        goal=goal,
        tau0=tau0,
        step_cost=1.0,
        cfg=sm_cfg,
    )

    print("\n=== SAFE MODE DECISION ===")
    print(f"mode={dec.mode} success={dec.success}")
    print(f"tau_requested={dec.tau_requested:.2f} tau_used={dec.tau_used} tau_gap={dec.tau_gap}")
    print(f"reason={dec.reason}")
    print(f"expansions={dec.expansions} cost={dec.cost}")
    print(f"maxI={dec.max_I_on_path} meanI={dec.mean_I_on_path}")
    print(f"path_len={len(dec.path)}")


if __name__ == "__main__":
    main()
