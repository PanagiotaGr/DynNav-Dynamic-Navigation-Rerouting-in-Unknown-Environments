"""
run_multi_robot_risk_experiment.py

Πειράματα με πολλαπλά ρομπότ και risk-aware allocation policies.
Χρησιμοποιεί το multi_robot_risk_allocation.py για να συγκρίνει policies:

- uniform_risk: όλα τα ρομπότ ίδιο weight στο ρίσκο
- explorer_monitor: ένας explorer (λιγότερο ευαίσθητος) + ένας conservative
- all_conservative: όλα τα ρομπότ πολύ ευαίσθητα στο ρίσκο

Για κάθε trial:
- Δημιουργεί τυχαίες θέσεις ρομπότ και targets.
- Κάθε target έχει risk ∈ [0, 1].
- Τρέχει allocation και μετρά:
  - total_distance,
  - mean_distance,
  - total_risk (sum risk assigned targets),
  - mean_risk,
  - max_risk,
  - total_cost (distance + risk weighted).

Αποτελέσματα γράφονται σε CSV: multi_robot_risk_results.csv
"""

import csv
import random
from typing import List, Dict, Tuple
from multi_robot_disagreement import disagreement_maxmin, disagreement_variance
import numpy as np
from multi_robot_risk_allocation import (
    RobotState,
    CellTarget,
    assign_tasks_hungarian,
    assign_with_risk_profiles,
)


# --------------------------------------------------------------------------- #
# Scenario generation
# --------------------------------------------------------------------------- #

def generate_random_robots(num_robots: int,
                           x_range: Tuple[float, float] = (0.0, 10.0),
                           y_range: Tuple[float, float] = (0.0, 10.0)
                           ) -> List[RobotState]:
    robots: List[RobotState] = []
    for i in range(num_robots):
        x = random.uniform(*x_range)
        y = random.uniform(*y_range)
        robots.append(RobotState(id=i, x=x, y=y))
    return robots


def generate_random_targets(num_targets: int,
                            x_range: Tuple[float, float] = (0.0, 10.0),
                            y_range: Tuple[float, float] = (0.0, 10.0),
                            risk_mode: str = "mixed"
                            ) -> List[CellTarget]:
    """
    risk_mode:
        - "mixed": ομοιόμορφο risk στη [0, 1]
        - "clustered_high": κάποια περιοχή έχει πιο υψηλό avg risk
    """
    targets: List[CellTarget] = []

    if risk_mode == "clustered_high":
        # ορίζουμε μία "επικίνδυνη" περιοχή π.χ. στο πάνω δεξί τεταρτημόριο
        high_risk_center = (8.0, 8.0)
        high_risk_radius = 3.0

        for j in range(num_targets):
            x = random.uniform(*x_range)
            y = random.uniform(*y_range)

            dist_to_hr = np.hypot(x - high_risk_center[0], y - high_risk_center[1])
            # όσο πιο κοντά στο high-risk center, τόσο μεγαλύτερο risk
            base_risk = random.uniform(0.0, 0.5)
            extra = max(0.0, (high_risk_radius - dist_to_hr) / high_risk_radius)
            risk = min(1.0, base_risk + 0.5 * extra)
            targets.append(CellTarget(id=j, x=x, y=y, risk=risk))

    else:  # "mixed"
        for j in range(num_targets):
            x = random.uniform(*x_range)
            y = random.uniform(*y_range)
            risk = random.uniform(0.0, 1.0)
            targets.append(CellTarget(id=j, x=x, y=y, risk=risk))

    return targets

def per_robot_risk_burden(
    robots: List[RobotState],
    targets: List[CellTarget],
    assignments,
) -> Dict[int, float]:
    """
    Returns dict robot_id -> sum(target.risk) for targets assigned to that robot.
    Handles common assignment formats:
      - list of (robot_id, target_id)
      - dict robot_id -> list of target_id
    """
    risk_by_target = {t.id: float(t.risk) for t in targets}
    burden = {r.id: 0.0 for r in robots}

    # case 1: dict robot_id -> [target_id,...]
    if isinstance(assignments, dict):
        for rid, tlist in assignments.items():
            for tid in tlist:
                if tid in risk_by_target:
                    burden[int(rid)] += risk_by_target[tid]
        return burden

    # case 2: list/iterable of pairs
    try:
        for pair in assignments:
            if pair is None:
                continue
            rid, tid = pair
            rid = int(rid)
            tid = int(tid)
            if tid in risk_by_target and rid in burden:
                burden[rid] += risk_by_target[tid]
    except Exception:
        # If format is unexpected, leave burdens at 0.0
        pass

    return burden

# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def compute_assignment_metrics(
    robots: List[RobotState],
    targets: List[CellTarget],
    assignments: List[Tuple[int, int]],
    total_cost: float,
):
    """
    Υπολογίζει μετρικές για μία assignment:
    - συνολική απόσταση
    - μέση απόσταση
    - συνολικό risk
    - μέσο risk
    - max risk
    """
    # map από id -> object για γρήγορη πρόσβαση
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

    if len(distances) == 0:
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


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #

def run_policy_uniform_risk(
    robots: List[RobotState],
    targets: List[CellTarget],
    w_dist: float,
    w_risk: float,
):
    assignments, total_cost = assign_tasks_hungarian(
        robots, targets, w_dist=w_dist, w_risk=w_risk
    )
    metrics = compute_assignment_metrics(robots, targets, assignments, total_cost)
    return assignments, metrics


def run_policy_explorer_monitor(
    robots: List[RobotState],
    targets: List[CellTarget],
    base_w_dist: float,
    base_w_risk: float,
):
    """
    Παράδειγμα:
    - robot 0: explorer (λιγότερο ευαίσθητος σε ρίσκο, alpha=0.5)
    - robot 1: neutral (alpha=1.0)
    - robot 2: conservative (alpha=2.0)   αν υπάρχει
    - τα υπόλοιπα robots: alpha=1.0
    """
    risk_profile: Dict[int, float] = {}
    if len(robots) > 0:
        risk_profile[robots[0].id] = 0.5  # explorer
    if len(robots) > 1:
        risk_profile[robots[1].id] = 1.0  # neutral
    if len(robots) > 2:
        risk_profile[robots[2].id] = 2.0  # conservative

    assignments, total_cost = assign_with_risk_profiles(
        robots,
        targets,
        risk_profile=risk_profile,
        base_w_dist=base_w_dist,
        base_w_risk=base_w_risk,
    )
    metrics = compute_assignment_metrics(robots, targets, assignments, total_cost)
    return assignments, metrics


def run_policy_all_conservative(
    robots: List[RobotState],
    targets: List[CellTarget],
    base_w_dist: float,
    base_w_risk: float,
    alpha_conservative: float = 3.0,
):
    """
    Όλα τα ρομπότ είναι πολύ ευαίσθητα στο ρίσκο.
    """
    risk_profile: Dict[int, float] = {
        r.id: alpha_conservative for r in robots
    }
    assignments, total_cost = assign_with_risk_profiles(
        robots,
        targets,
        risk_profile=risk_profile,
        base_w_dist=base_w_dist,
        base_w_risk=base_w_risk,
    )
    metrics = compute_assignment_metrics(robots, targets, assignments, total_cost)
    return assignments, metrics


# --------------------------------------------------------------------------- #
# Main experiment loop
# --------------------------------------------------------------------------- #

def main():
    random.seed(42)
    np.random.seed(42)

    output_csv = "multi_robot_risk_results.csv"
    num_trials = 200
    num_robots = 3
    num_targets = 6

    w_dist = 1.0
    w_risk = 2.0

    fieldnames = [
        "trial",
        "policy",
        "num_robots",
        "num_targets",
        "risk_mode",
        "total_distance",
        "mean_distance",
        "total_risk",
        "mean_risk",
        "max_risk",
        "total_cost",
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for trial in range(num_trials):
            # Μπορείς να εναλλάσσεις risk_mode ανάλογα με το σενάριο
            risk_mode = "clustered_high" if trial % 2 == 0 else "mixed"

            robots = generate_random_robots(num_robots)
            targets = generate_random_targets(num_targets, risk_mode=risk_mode)

            # 1) Uniform risk policy
            _, metrics_uniform = run_policy_uniform_risk(
                robots, targets, w_dist=w_dist, w_risk=w_risk
            )
            row_uniform = {
                "trial": trial,
                "policy": "uniform_risk",
                "num_robots": num_robots,
                "num_targets": num_targets,
                "risk_mode": risk_mode,
                **metrics_uniform,
            }
            writer.writerow(row_uniform)

            # 2) Explorer + monitor policy
            _, metrics_explmon = run_policy_explorer_monitor(
                robots, targets, base_w_dist=w_dist, base_w_risk=w_risk
            )
            row_explmon = {
                "trial": trial,
                "policy": "explorer_monitor",
                "num_robots": num_robots,
                "num_targets": num_targets,
                "risk_mode": risk_mode,
                **metrics_explmon,
            }
            writer.writerow(row_explmon)

            # 3) All conservative policy
            _, metrics_allcons = run_policy_all_conservative(
                robots, targets, base_w_dist=w_dist, base_w_risk=w_risk
            )
            row_allcons = {
                "trial": trial,
                "policy": "all_conservative",
                "num_robots": num_robots,
                "num_targets": num_targets,
                "risk_mode": risk_mode,
                **metrics_allcons,
            }
            writer.writerow(row_allcons)

    print(f"Saved results to {output_csv}")


if __name__ == "__main__":
    main()
