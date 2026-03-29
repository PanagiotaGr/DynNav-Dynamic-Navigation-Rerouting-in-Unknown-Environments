from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List
import time
import numpy as np
import pandas as pd

from contributions.hybrid_learned_astar.code.hybrid_astar import hybrid_astar
from contributions.hybrid_learned_astar.code.hybrid_heuristic import HybridHeuristic
from contributions.learned_uncertainty_astar.code.uncertainty_astar import (
    astar,
    EuclideanHeuristic as BaseEuclideanHeuristic,
    LearnedUncertaintyHeuristic,
)


@dataclass
class EvalSummary:
    method: str
    n_queries: int
    found_rate: float
    mean_expansions: float
    mean_path_length: float
    mean_suboptimality: float
    mean_h_std: float
    mean_fallback_rate: float
    mean_compute_time_ms: float


def _mean_std_from_result(res) -> float:
    if hasattr(res, "mean_h_std"):
        return float(res.mean_h_std)
    if hasattr(res, "h_stds") and len(res.h_stds) > 0:
        return float(np.mean(res.h_stds))
    return 0.0


def _path_length_from_result(res) -> float:
    if hasattr(res, "path_length"):
        return float(res.path_length)
    if hasattr(res, "path_cost"):
        return float(res.path_cost)
    return float("inf")


def _found_from_result(res) -> bool:
    if hasattr(res, "found"):
        return bool(res.found)
    path_len = _path_length_from_result(res)
    return np.isfinite(path_len)


def evaluate_methods(
    problems: List[dict],
    learned_model,
    tau_values: List[float],
    device: str = "cpu",
) -> pd.DataFrame:
    rows = []

    baseline_costs = []
    baseline_exp = []
    baseline_times = []

    for p in problems:
        t0 = time.perf_counter()
        res = astar(
            grid=p["grid"],
            start=p["start"],
            goal=p["goal"],
            heuristic=BaseEuclideanHeuristic(),
        )
        dt = (time.perf_counter() - t0) * 1000.0
        baseline_costs.append(_path_length_from_result(res))
        baseline_exp.append(res.expansions)
        baseline_times.append(dt)

    rows.append(
        asdict(
            EvalSummary(
                method="classic_astar",
                n_queries=len(problems),
                found_rate=float(np.mean(np.isfinite(baseline_costs))),
                mean_expansions=float(np.mean(baseline_exp)),
                mean_path_length=float(np.mean(baseline_costs)),
                mean_suboptimality=1.0,
                mean_h_std=0.0,
                mean_fallback_rate=0.0,
                mean_compute_time_ms=float(np.mean(baseline_times)),
            )
        )
    )

    learned_costs = []
    learned_exp = []
    learned_std = []
    learned_times = []

    learned_heuristic = LearnedUncertaintyHeuristic(
        learned_model,
        device=device,
    )

    for p, base_cost in zip(problems, baseline_costs):
        t0 = time.perf_counter()
        res = astar(
            grid=p["grid"],
            start=p["start"],
            goal=p["goal"],
            heuristic=learned_heuristic,
            beta=0.0,
        )
        dt = (time.perf_counter() - t0) * 1000.0
        learned_costs.append(_path_length_from_result(res))
        learned_exp.append(res.expansions)
        learned_std.append(_mean_std_from_result(res))
        learned_times.append(dt)

    rows.append(
        asdict(
            EvalSummary(
                method="learned_uncertainty_astar(beta=0.0)",
                n_queries=len(problems),
                found_rate=float(np.mean(np.isfinite(learned_costs))),
                mean_expansions=float(np.mean(learned_exp)),
                mean_path_length=float(np.mean(learned_costs)),
                mean_suboptimality=float(
                    np.nanmean(
                        [
                            c / b if np.isfinite(c) and np.isfinite(b) and b > 0 else np.nan
                            for c, b in zip(learned_costs, baseline_costs)
                        ]
                    )
                ),
                mean_h_std=float(np.mean(learned_std)) if learned_std else 0.0,
                mean_fallback_rate=0.0,
                mean_compute_time_ms=float(np.mean(learned_times)),
            )
        )
    )

    for tau in tau_values:
        costs = []
        exps = []
        stds = []
        fallbacks = []
        times = []

        heuristic = HybridHeuristic(
            learned_model=learned_model,
            tau=tau,
            device=device,
            use_smooth_blending=False,
        )

        for p in problems:
            t0 = time.perf_counter()
            res = hybrid_astar(
                grid=p["grid"],
                start=p["start"],
                goal=p["goal"],
                heuristic=heuristic,
            )
            dt = (time.perf_counter() - t0) * 1000.0
            costs.append(_path_length_from_result(res))
            exps.append(res.expansions)
            stds.append(_mean_std_from_result(res))
            fallbacks.append(res.fallback_rate)
            times.append(dt)

        rows.append(
            asdict(
                EvalSummary(
                    method=f"hybrid_astar(tau={tau:.2f})",
                    n_queries=len(problems),
                    found_rate=float(np.mean(np.isfinite(costs))),
                    mean_expansions=float(np.mean(exps)),
                    mean_path_length=float(np.mean(costs)),
                    mean_suboptimality=float(
                        np.nanmean(
                            [
                                c / b if np.isfinite(c) and np.isfinite(b) and b > 0 else np.nan
                                for c, b in zip(costs, baseline_costs)
                            ]
                        )
                    ),
                    mean_h_std=float(np.mean(stds)) if stds else 0.0,
                    mean_fallback_rate=float(np.mean(fallbacks)) if fallbacks else 0.0,
                    mean_compute_time_ms=float(np.mean(times)),
                )
            )
        )

    return pd.DataFrame(rows)
