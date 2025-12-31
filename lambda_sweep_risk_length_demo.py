"""
lambda_sweep_risk_length_demo.py

Demo για το trade-off:

    J_λ = length + λ * risk

σε ένα απλό multi-robot σενάριο, χρησιμοποιώντας τον ίδιο
"safe-mode" κόσμο (3 ρομπότ, 4 targets με high/low risk).

Για κάθε τιμή του λ:

- λύνουμε το assignment με cost ≈ distance + λ * risk
- μετράμε total_distance, total_risk, max_risk, total_cost
- αποθηκεύουμε σε CSV: lambda_sweep_risk_length_results.csv

Έτσι μπορείς μετά να κάνεις:
- plot length vs risk
- να δείξεις Pareto-like curve
- να το βάλεις σαν θεωρητικό/πειραματικό παράδειγμα για J = length + λ·risk.
"""

import csv
from typing import List, Dict, Tuple

import numpy as np

from multi_robot_risk_allocation import (
    RobotState,
    CellTarget,
    assign_with_risk_profiles,
)


OUTPUT_CSV = "lambda_sweep_risk_length_results.csv"


# ---------------------------------------------------
# Helpers
# ---------------------------------------------------

def build_fixed_world(num_robots=3, num_targets=4):
    """
    Ίδιος κόσμος με το safe-mode παράδειγμα:

    Robots:
        r0 ~ (0,0)
        r1 ~ (5,0)
        r2 ~ (0,5)

    Targets:
        t0: πολύ κοντά στο r0 αλλά high-risk
        t1: λίγο πιο μακριά από r0 αλλά low-risk
        t2: κοντά στο r1, low-risk
        t3: κοντά στο r2, low-ish risk
    """
    robots = [
        RobotState(id=0, x=0.0, y=0.0),
        RobotState(id=1, x=5.0, y=0.0),
        RobotState(id=2, x=0.0, y=5.0),
    ][:num_robots]

    targets = [
        CellTarget(id=0, x=0.1, y=0.0, risk=0.9),   # very high risk, very close to r0
        CellTarget(id=1, x=3.0, y=0.0, risk=0.1),   # further, low risk
        CellTarget(id=2, x=5.0, y=0.5, risk=0.1),   # near r1, low risk
        CellTarget(id=3, x=0.0, y=5.5, risk=0.2),   # near r2, low-ish risk
    ][:num_targets]

    return robots, targets


def compute_assignment_metrics(
    robots: List[RobotState],
    targets: List[CellTarget],
    assignments: List[Tuple[int, int]],
    total_cost: float,
) -> Dict[str, float]:
    robot_map: Dict[int, RobotState] = {r.id: r for r in robots}
    target_map: Dict[int, CellTarget] = {t.id: t for t in targets}

    distances = []
    risks = []

    for rid, tid in assignments:
        r = robot_map[rid]
        t = target_map[tid]
        dist = float(np.hypot(r.x - t.x, r.y - t.y))
        distances.append(dist)
        risks.append(float(t.risk))

    if not distances:
        return {
            "total_distance": 0.0,
            "mean_distance": 0.0,
            "total_risk": 0.0,
            "mean_risk": 0.0,
            "max_risk": 0.0,
            "total_cost": float(total_cost),
        }

    total_distance = float(sum(distances))
    mean_distance = float(total_distance / len(distances))
    total_risk = float(sum(risks))
    mean_risk = float(total_risk / len(risks))
    max_risk = float(max(risks))

    return {
        "total_distance": total_distance,
        "mean_distance": mean_distance,
        "total_risk": total_risk,
        "mean_risk": mean_risk,
        "max_risk": max_risk,
        "total_cost": float(total_cost),
    }


def write_header(path: str):
    header = [
        "lambda",
        "assignments",
        "total_distance",
        "total_risk",
        "max_risk",
        "total_cost",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_row(path: str, lam: float, assignments, metrics: Dict[str, float]):
    row = [
        lam,
        str(assignments),
        round(metrics["total_distance"], 4),
        round(metrics["total_risk"], 4),
        round(metrics["max_risk"], 4),
        round(metrics["total_cost"], 4),
    ]
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# ---------------------------------------------------
# Main λ-sweep
# ---------------------------------------------------

def main():
    robots, targets = build_fixed_world(num_robots=3, num_targets=4)

    print("Robots:")
    for r in robots:
        print(" ", r)
    print("\nTargets:")
    for t in targets:
        print(" ", t)

    lambdas = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

    write_header(OUTPUT_CSV)

    print("\n=== λ-sweep for J_λ = length + λ * risk ===\n")

    # risk_profile = 1 για όλα τα robots -> cost ~ dist + λ·risk
    base_risk_profile = {r.id: 1.0 for r in robots}

    for lam in lambdas:
        assignments, total_cost = assign_with_risk_profiles(
            robots,
            targets,
            risk_profile=base_risk_profile,
            base_w_dist=1.0,
            base_w_risk=float(lam),
        )
        metrics = compute_assignment_metrics(robots, targets, assignments, total_cost)

        append_row(OUTPUT_CSV, lam, assignments, metrics)

        print(f"λ = {lam:.1f}")
        print(f"  assignments: {assignments}")
        print(
            f"  total_distance = {metrics['total_distance']:.3f}, "
            f"total_risk = {metrics['total_risk']:.3f}, "
            f"max_risk = {metrics['max_risk']:.3f}, "
            f"total_cost = {metrics['total_cost']:.3f}"
        )
        print()


if __name__ == "__main__":
    main()
