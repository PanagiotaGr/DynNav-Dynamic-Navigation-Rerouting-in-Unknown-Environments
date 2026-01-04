# build_world_context.py
import numpy as np
from dataclasses import dataclass

from irreversibility_map import (
    IrreversibilityConfig,
    build_irreversibility_map,
    load_grid_from_cell_table_csv,
)
from run_irreversibility_bottleneck_sweep import add_bottleneck_wall, pick_start_left_goal_right
from returnability_map import ReturnabilityConfig, compute_return_cost_grid, normalize_returnability


@dataclass
class WorldContext:
    unc_grid: np.ndarray
    free_mask: np.ndarray
    I_world: np.ndarray
    R: np.ndarray
    start: tuple
    wall_cols: tuple
    tauR: float


def build_bottleneck_context(
    path_csv: str = "coverage_grid_with_uncertainty.csv",
    tauR: float = 0.85,
    wall_I: float = 0.95,
    door_I: float = 0.6,
    thickness: int = 2,
) -> WorldContext:
    unc_grid = load_grid_from_cell_table_csv(
        path_csv, value_col="uncertainty", row_col=("row", "col"), fill_nan_with="max"
    )
    free_mask = np.isfinite(unc_grid)

    # irreversibility
    unc01 = (unc_grid - unc_grid.min()) / (unc_grid.max() - unc_grid.min() + 1e-9)
    feat_density = 1.0 - unc01
    cfgI = IrreversibilityConfig(0.60, 0.25, 0.15, 2)
    I = build_irreversibility_map(unc_grid, feat_density, free_mask, cfgI)

    I_world, wall_cols, _ = add_bottleneck_wall(I, wall_I=wall_I, door_I=door_I, thickness=thickness)

    start, _ = pick_start_left_goal_right(free_mask, I_world, wall_cols=wall_cols, I_max=0.50)

    # returnability
    cfgR = ReturnabilityConfig(tau=tauR, step_cost=1.0, use_8conn=False)
    rc = compute_return_cost_grid(free_mask, I_world, base=start, cfg=cfgR)
    R = normalize_returnability(rc, unreachable_cost=cfgR.unreachable_cost)

    return WorldContext(
        unc_grid=unc_grid,
        free_mask=free_mask,
        I_world=I_world,
        R=R,
        start=start,
        wall_cols=wall_cols,
        tauR=tauR,
    )
