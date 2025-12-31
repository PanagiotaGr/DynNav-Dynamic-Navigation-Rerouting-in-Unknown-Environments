"""
multi_robot_safe_mode_logged.py

Logged έκδοση του πειράματος self-trust + safe mode.

- Χρησιμοποιεί τον ίδιο τύπο κόσμου και policies με το multi_robot_safe_mode_experiment.
- Κάθε step:
    * ενημερώνει self-trust για κάθε ρομπότ
    * επιλέγει NORMAL / SAFE policy
    * υπολογίζει metrics (distance, risk, cost)
    * γράφει όλα τα αποτελέσματα σε CSV αρχείο:
        multi_robot_safe_mode_results.csv
"""

import csv
import random
from typing import List, Dict, Tuple

import numpy as np

from multi_robot_risk_allocation import (
    RobotState,
    CellTarget,
    assign_with_risk_profiles,
)
from self_trust_manager import SelfTrustManager


RESULTS_FILE = "multi_robot_safe_mode_results.csv"


# ---------------------------------------------------
# CSV helper functions
# ---------------------------------------------------

def init_csv():
    header = [
        "step",
        "policy",
        "any_safe_mode",
        "robot_id",
        "self_trust_S",
        "expected_drift",
        "observed_drift",
        "calibration_error",
        "total_distance",
        "total_risk",
        "max_risk",
        "total_cost",
    ]

    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def log_row(
    step: int,
    policy: str,
    any_safe_mode: bool,
    robot_id: int,
    S: float,
    expected_drift: float,
    observed_drift: float,
    calibration_error: float,
    metrics: Dict[str, float],
):
    row = [
        step,
        policy,
        int(any_safe_mode),
        robot_id,
        round(S, 4),
        round(expected_drift, 4),
        round(observed_drift, 4),
        round(calibration_error, 4),
        round(metrics["total_distance"], 4),
        round(metrics["total_risk"], 4),
        round(metrics["max_risk"], 4),
        round(metrics["total_cost"], 4),
    ]
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# ---------------------------------------------------
# Metrics helper (ίδιο με αυτό που έχεις ήδη)
# ---------------------------------------------------

def compute_assignment_metrics(
    robots: List[RobotState],
    targets: List[CellTarget],
    assignments: List[Tuple[int, int]],
    total_cost: float,
):
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


# ---------------------------------------------------
# Fixed world (ίδιο με το καλό παράδειγμα που έτρεξες)
# ---------------------------------------------------

def build_fixed_world(num_robots=3, num_targets=4):
    """
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


# ---------------------------------------------------
# Policies (ίδιες με πριν)
# ---------------------------------------------------

def run_normal_policy(robots, targets):
    risk_profile = {}
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
        base_w_dist=1.0,
        base_w_risk=4.0,
    )
    metrics = compute_assignment_metrics(robots, targets, assignments, total_cost)
    return "NORMAL_POLICY", assignments, metrics


def run_safe_policy(robots, targets):
    risk_profile = {r.id: 4.0 for r in robots}
    assignments, total_cost = assign_with_risk_profiles(
        robots,
        targets,
        risk_profile=risk_profile,
        base_w_dist=1.0,
        base_w_risk=4.0,
    )
    metrics = compute_assignment_metrics(robots, targets, assignments, total_cost)
    return "SAFE_MODE_POLICY", assignments, metrics


# ---------------------------------------------------
# Main experiment with logging
# ---------------------------------------------------

def main():
    random.seed(0)
    np.random.seed(0)

    init_csv()

    trust = SelfTrustManager(
        alpha_calib=0.4,
        alpha_drift=0.6,
        safe_threshold=0.7,
    )

    num_robots = 3
    for rid in range(num_robots):
        trust.register_robot(rid)

    robots, targets = build_fixed_world(num_robots=num_robots, num_targets=4)

    print("Robots:")
    for r in robots:
        print(" ", r)

    print("\nTargets:")
    for t in targets:
        print(" ", t)

    print("\n===== LOGGED SELF-TRUST + SAFE MODE EXPERIMENT =====")

    for step in range(10):
        print(f"\n--- STEP {step} ---")

        # "εύκολα" steps 0–4, "δύσκολα" 5–9
        if step < 5:
            calibration_error = np.clip(np.random.normal(0.15, 0.05), 0.0, 1.0)
            observed_drift = np.clip(np.random.normal(0.25, 0.05), 0.0, 1.0)
        else:
            calibration_error = np.clip(np.random.normal(0.35, 0.07), 0.0, 1.0)
            observed_drift = np.clip(np.random.normal(0.9, 0.05), 0.0, 1.0)

        expected_drift = np.clip(np.random.normal(0.25, 0.05), 0.0, 1.0)

        any_safe = False
        trust_values: Dict[int, float] = {}

        # Update self-trust per robot
        for rid in range(num_robots):
            S, safe = trust.update(
                robot_id=rid,
                calibration_error=float(calibration_error),
                expected_drift=float(expected_drift),
                observed_drift=float(observed_drift),
            )
            trust_values[rid] = S
            mode = "SAFE" if safe else "NORMAL"
            print(f"Robot {rid} | S={S:.3f} | {mode}")
            if safe:
                any_safe = True

        # Υπολογίζουμε ΚΑΙ τις δύο πολιτικές
        pol_norm, assign_norm, metrics_norm = run_normal_policy(robots, targets)
        pol_safe, assign_safe, metrics_safe = run_safe_policy(robots, targets)

        # Επιλογή active policy με βάση self-trust
        if any_safe:
            active_policy = pol_safe
            active_assign = assign_safe
            active_metrics = metrics_safe
        else:
            active_policy = pol_norm
            active_assign = assign_norm
            active_metrics = metrics_norm

        print(f"\nExpected drift:  {expected_drift:.3f}")
        print(f"Observed drift:  {observed_drift:.3f}")
        print(f"Calibration err: {calibration_error:.3f}")
        print(f"Active policy:   {active_policy}")
        print(f"Active assign:   {active_assign}")
        print(
            "Active metrics:  "
            f"dist={active_metrics['total_distance']:.3f}, "
            f"risk={active_metrics['total_risk']:.3f}, "
            f"max_risk={active_metrics['max_risk']:.3f}, "
            f"cost={active_metrics['total_cost']:.3f}"
        )

        print("\n[Comparison NORMAL vs SAFE]")
        print(
            f"  NORMAL: dist={metrics_norm['total_distance']:.3f}, "
            f"risk={metrics_norm['total_risk']:.3f}, "
            f"max_risk={metrics_norm['max_risk']:.3f}, "
            f"cost={metrics_norm['total_cost']:.3f}"
        )
        print(
            f"  SAFE  : dist={metrics_safe['total_distance']:.3f}, "
            f"risk={metrics_safe['total_risk']:.3f}, "
            f"max_risk={metrics_safe['max_risk']:.3f}, "
            f"cost={metrics_safe['total_cost']:.3f}"
        )

        # Logging: μία γραμμή ανά ρομπότ, με τα ίδια mission-level metrics
        for rid in range(num_robots):
            log_row(
                step=step,
                policy=active_policy,
                any_safe_mode=any_safe,
                robot_id=rid,
                S=trust_values[rid],
                expected_drift=expected_drift,
                observed_drift=observed_drift,
                calibration_error=calibration_error,
                metrics=active_metrics,
            )


if __name__ == "__main__":
    main()
