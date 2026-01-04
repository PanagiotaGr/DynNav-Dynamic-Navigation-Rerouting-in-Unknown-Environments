# run_world_template_bench.py
import numpy as np
import pandas as pd

from irreversibility_map import (
    IrreversibilityConfig,
    build_irreversibility_map,
    load_grid_from_cell_table_csv,
)
from irreversibility_planner import astar_irreversibility_constrained
from irreversibility_safe_mode import SafeModeConfig, run_irreversibility_with_safe_mode
from adaptive_tau_minimax import MinimaxTauConfig, find_min_feasible_tau

from world_templates import world_bottleneck, world_culdesac, world_noisy_corridor


def sample_free_cell(rng, free_mask):
    ys, xs = np.where(free_mask)
    idx = rng.integers(0, len(ys))
    return (int(ys[idx]), int(xs[idx]))


def build_base_I():
    path_csv = "coverage_grid_with_uncertainty.csv"
    unc_grid = load_grid_from_cell_table_csv(
        path_csv, value_col="uncertainty", row_col=("row", "col"), fill_nan_with="max"
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
    return free_mask, I_grid


def apply_template(name, I_grid):
    if name == "bottleneck":
        return world_bottleneck(I_grid, wall_I=0.95, door_I=0.6, thickness=2)
    if name == "culdesac":
        return world_culdesac(I_grid, trap_I=0.90)
    if name == "noisy_corridor":
        return world_noisy_corridor(I_grid, I_min=0.30, I_max=0.90)
    raise ValueError(f"Unknown template: {name}")


def main():
    rng = np.random.default_rng(7)

    templates = ["bottleneck", "culdesac", "noisy_corridor"]
    N = 60
    fixed_tau0 = 0.80
    minimax_margin = 0.01

    free_mask, I_base = build_base_I()

    sm_cfg = SafeModeConfig(tau_max=1.0, tau_step=0.01, max_tau_gap=0.40, stop_if_infeasible=True)
    mm_cfg = MinimaxTauConfig(tau_lo=0.0, tau_hi=1.0, tol=0.005, max_iters=40, margin=minimax_margin)

    rows = []
    for tname in templates:
        I_world, meta = apply_template(tname, I_base)

        for k in range(N):
            start = sample_free_cell(rng, free_mask)
            goal = sample_free_cell(rng, free_mask)
            while goal == start:
                goal = sample_free_cell(rng, free_mask)

            fixed = astar_irreversibility_constrained(
                free_mask=free_mask,
                irreversibility_grid=I_world,
                start=start,
                goal=goal,
                tau=fixed_tau0,
                step_cost=1.0,
            )

            safe = run_irreversibility_with_safe_mode(
                free_mask=free_mask,
                I_grid=I_world,
                start=start,
                goal=goal,
                tau0=fixed_tau0,
                step_cost=1.0,
                cfg=sm_cfg,
            )

            mm = find_min_feasible_tau(free_mask, I_world, start, goal, mm_cfg)
            if mm.feasible:
                mm_plan = astar_irreversibility_constrained(
                    free_mask=free_mask,
                    irreversibility_grid=I_world,
                    start=start,
                    goal=goal,
                    tau=float(mm.tau_request),
                    step_cost=1.0,
                )
            else:
                mm_plan = None

            rows.append({
                "template": tname,
                "pair_id": k,
                "I_start": float(I_world[start]),
                "I_goal": float(I_world[goal]),
                "fixed_tau0": fixed_tau0,

                "fixed_success": int(fixed.success),
                "fixed_reason": str(fixed.reason),

                "safe_success": int(safe.success),
                "safe_mode": safe.mode,
                "safe_tau_used": float(safe.tau_used) if safe.tau_used is not None else np.nan,
                "safe_tau_gap": float(safe.tau_gap) if safe.tau_gap is not None else np.nan,
                "safe_reason": safe.reason,

                "tau_star": float(mm.tau_star) if mm.tau_star is not None else np.nan,
                "tau_request": float(mm.tau_request) if mm.tau_request is not None else np.nan,
                "mm_success": int(mm_plan.success) if mm_plan is not None else 0,
            })

        print(f"[done] template={tname} meta={meta}")

    out = "world_template_bench.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
