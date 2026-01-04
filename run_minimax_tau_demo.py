# run_minimax_tau_demo.py
import numpy as np

from irreversibility_map import (
    IrreversibilityConfig,
    build_irreversibility_map,
    load_grid_from_cell_table_csv,
)
from run_irreversibility_bottleneck_sweep import add_bottleneck_wall, pick_start_left_goal_right
from adaptive_tau_minimax import MinimaxTauConfig, find_min_feasible_tau
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

    print(f"Start={start}, Goal={goal}, I_start={I_bottle[start]:.3f}, I_goal={I_bottle[goal]:.3f}")
    print(f"Door rows={door_rows}, wall cols={wall_cols}, door_I={door_I}, wall_I={wall_I}")

    mm_cfg = MinimaxTauConfig(tau_lo=0.0, tau_hi=1.0, tol=0.0025, max_iters=40, margin=0.01)
    mm = find_min_feasible_tau(free_mask, I_bottle, start, goal, mm_cfg)

    print("\n=== MINIMAX τ ESTIMATE ===")
    print(f"feasible={mm.feasible} tau_star={mm.tau_star} tau_request={mm.tau_request} iters={mm.iters} reason={mm.reason}")

    # Now run safe-mode using the minimax request (should be near-feasible already)
    sm_cfg = SafeModeConfig(tau_max=1.0, tau_step=0.01, max_tau_gap=0.30, stop_if_infeasible=True)
    dec = run_irreversibility_with_safe_mode(
        free_mask=free_mask,
        I_grid=I_bottle,
        start=start,
        goal=goal,
        tau0=float(mm.tau_request),
        step_cost=1.0,
        cfg=sm_cfg,
    )

    print("\n=== MINIMAX τ + SAFE MODE ===")
    print(f"mode={dec.mode} success={dec.success}")
    print(f"tau_requested={dec.tau_requested:.3f} tau_used={dec.tau_used} tau_gap={dec.tau_gap}")
    print(f"maxI={dec.max_I_on_path} meanI={dec.mean_I_on_path} path_len={len(dec.path)} expansions={dec.expansions}")


if __name__ == "__main__":
    main()
