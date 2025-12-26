"""
adaptive_belief_risk_planner.py

Wrapper around the belief–risk planner that adapts the lambda weight
using an online feedback policy.

NOTE:
You need to connect this script to your existing belief_risk_planner
implementation by implementing `run_belief_risk_planner` below.
"""

import csv
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

from adaptive_risk_policy import AdaptiveRiskPolicy, RiskContext


# ---------------------------------------------------------------------------
# 1. Adapter to your existing planner
# ---------------------------------------------------------------------------

def run_belief_risk_planner(
    lambda_value: float,
    scenario_id: int,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Offline adapter που αντλεί metrics από belief_risk_lambda_sweep.csv
    αντί να καλεί κανονικό planner.

    Ιδέα:
    - Διαβάζουμε μία φορά το CSV με τα αποτελέσματα του λ-sweep.
    - Για δοσμένο lambda_value, βρίσκουμε τη γραμμή με λ πιο κοντά σε αυτό.
    - Επιστρέφουμε fused_risk, geom_length κλπ από εκείνη τη γραμμή.

    Έτσι μπορούμε να δοκιμάσουμε την adaptive risk policy χωρίς
    να αγγίξουμε ακόμη τον πλήρη ROS/πλοηγό.
    """

    from pathlib import Path
    import math
    import csv

    csv_path = Path("belief_risk_lambda_sweep.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find {csv_path}. "
            "Make sure sweep_lambda_fused_risk.py has been run "
            "and the CSV is in the project root."
        )

    # --- cache για να μην ξαναδιαβάζουμε το CSV κάθε φορά ---
    # βάζουμε attribute στη function για να αποθηκεύσουμε τα rows
    if not hasattr(run_belief_risk_planner, "_cached_rows"):
        rows = []
        with csv_path.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        if not rows:
            raise RuntimeError(f"{csv_path} is empty.")
        run_belief_risk_planner._cached_rows = rows  # type: ignore
    else:
        rows = run_belief_risk_planner._cached_rows  # type: ignore

    # helper για να βρούμε τιμή με fallback σε πολλά πιθανά column names
    def get_with_fallback(row: Dict[str, str], candidates, default: float) -> float:
        for key in candidates:
            if key in row and row[key] != "":
                try:
                    return float(row[key])
                except ValueError:
                    continue
        return float(default)

    # προσπαθούμε να εντοπίσουμε ποια στήλη είναι το λ
    possible_lambda_cols = ["lambda", "lambda_value", "lambda_fused", "lam"]
    lambda_col = None
    for c in possible_lambda_cols:
        if c in rows[0]:
            lambda_col = c
            break
    if lambda_col is None:
        raise KeyError(
            "Could not find a lambda column in belief_risk_lambda_sweep.csv. "
            "Tried: " + ", ".join(possible_lambda_cols)
        )

    # βρίσκουμε τη γραμμή με λ πιο κοντά στο ζητούμενο lambda_value
    best_row = None
    best_diff = math.inf
    for row in rows:
        try:
            lam_row = float(row[lambda_col])
        except ValueError:
            continue
        diff = abs(lam_row - lambda_value)
        if diff < best_diff:
            best_diff = diff
            best_row = row

    if best_row is None:
        raise RuntimeError("Could not find any valid lambda row in CSV.")

    # εξαγωγή μετρικών με διάφορα πιθανά ονόματα στηλών
    fused_risk = get_with_fallback(
        best_row,
        [
            "fused_risk",
            "risk_fused",
            "R_fused",
            "risk",
            "fused_sum",
            "fused_mean",
        ],
        default=0.0,
    )



    geom_length = get_with_fallback(
        best_row,
        ["geom_length", "path_length", "length_cells", "L_geom"],
        default=0.0,
    )
    num_cells = int(
        get_with_fallback(
            best_row,
            ["num_cells", "n_cells", "length_cells"],
            default=max(1.0, geom_length),
        )
    )

    # εδώ δεν έχουμε πραγματικό distance_to_goal από το sweep → βάζουμε 0.0
    distance_to_goal = 0.0
    success = True  # το sweep λογικά είναι πάνω σε επιτυχημένες διαδρομές

    return {
        "fused_risk": fused_risk,
        "geom_length": geom_length,
        "num_cells": num_cells,
        "distance_to_goal": distance_to_goal,
        "success": success,
    }

# ---------------------------------------------------------------------------
# 2. Main adaptive loop
# ---------------------------------------------------------------------------

def run_adaptive_experiments(
    num_episodes: int,
    lambda_init: float,
    output_csv: Path,
    lambda_min: float = 0.0,
    lambda_max: float = 5.0,
    target_risk: float = 1.0,
    eta: float = 0.1,
    config_path: Optional[str] = None,
) -> None:
    """
    Run a sequence of planning episodes where lambda is updated adaptively.

    Parameters
    ----------
    num_episodes : int
        Number of planning episodes (scenarios, or replans) to run.
    lambda_init : float
        Initial lambda value.
    output_csv : Path
        CSV file where metrics and lambda evolution will be stored.
    lambda_min, lambda_max : float
        Bounds for lambda.
    target_risk : float
        Nominal target fused risk.
    eta : float
        Step size for the lambda update.
    config_path : Optional[str]
        Optional config file for the underlying planner.
    """
    policy = AdaptiveRiskPolicy(
        lambda_init=lambda_init,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        target_risk=target_risk,
        eta=eta,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "lambda_before",
                "lambda_after",
                "fused_risk",
                "target_risk_effective",
                "geom_length",
                "num_cells",
                "distance_to_goal",
                "success",
            ],
        )
        writer.writeheader()

        for episode in range(num_episodes):
            lambda_before = policy.get_lambda()

            # Here scenario_id could simply be the episode index,
            # or you can map it to specific stored grids / maps.
            scenario_id = episode

            planner_result = run_belief_risk_planner(
                lambda_value=lambda_before,
                scenario_id=scenario_id,
                config_path=config_path,
            )

            fused_risk = float(planner_result.get("fused_risk", 0.0))
            geom_length = float(planner_result.get("geom_length", 0.0))
            num_cells = int(planner_result.get("num_cells", 0))
            distance_to_goal = float(planner_result.get("distance_to_goal", 0.0))
            success = bool(planner_result.get("success", True))

            context = RiskContext(
                step_idx=episode,
                distance_to_goal=distance_to_goal,
                map_entropy=None,  # fill if you compute global entropy
            )

            # We want to know the effective target used internally:
            target_effective = policy._compute_effective_target(context)  # type: ignore

            lambda_after = policy.update(
                realized_risk=fused_risk,
                context=context,
            )

            writer.writerow(
                {
                    "episode": episode,
                    "lambda_before": lambda_before,
                    "lambda_after": lambda_after,
                    "fused_risk": fused_risk,
                    "target_risk_effective": target_effective,
                    "geom_length": geom_length,
                    "num_cells": num_cells,
                    "distance_to_goal": distance_to_goal,
                    "success": int(success),
                }
            )

            print(
                f"[Episode {episode}] "
                f"lambda: {lambda_before:.3f} -> {lambda_after:.3f}, "
                f"fused_risk: {fused_risk:.3f}, "
                f"geom_length: {geom_length:.3f}, "
                f"success: {success}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive belief–risk planning experiments."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of planning episodes to run.",
    )
    parser.add_argument(
        "--lambda_init",
        type=float,
        default=0.5,
        help="Initial lambda value.",
    )
    parser.add_argument(
        "--lambda_min",
        type=float,
        default=0.0,
        help="Minimum lambda value.",
    )
    parser.add_argument(
        "--lambda_max",
        type=float,
        default=5.0,
        help="Maximum lambda value.",
    )
    parser.add_argument(
        "--target_risk",
        type=float,
        default=1.0,
        help="Nominal target fused risk.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.1,
        help="Step size for lambda update.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config file path for the planner.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/adaptive_belief_risk_runs.csv",
        help="Output CSV file for logging results.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_path = Path(args.output_csv)
    run_adaptive_experiments(
        num_episodes=args.episodes,
        lambda_init=args.lambda_init,
        output_csv=output_path,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        target_risk=args.target_risk,
        eta=args.eta,
        config_path=args.config,
    )
