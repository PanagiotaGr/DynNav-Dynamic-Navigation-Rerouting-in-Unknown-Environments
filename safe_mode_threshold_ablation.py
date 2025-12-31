"""
safe_mode_threshold_ablation.py

Αblation study για το safe_mode threshold τ.

Για κάθε τ σε ένα σύνολο τιμών:
- τρέχουμε ένα μικρό online πείραμα (10 βήματα)
- ενημερώνουμε self-trust S για κάθε ρομπότ
- επιλέγουμε policy (NORMAL / SAFE) ανά βήμα
- καταγράφουμε mission-level metrics ανά policy:
    total_distance, total_risk, max_risk, total_cost

Στο τέλος γράφουμε σε CSV:
    safe_mode_threshold_ablation_results.csv

με γραμμές:
    threshold, policy, num_steps, mean_total_distance,
    mean_total_risk, mean_max_risk, mean_total_cost
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

OUTPUT_CSV = "safe_mode_threshold_ablation_results.csv"


# ---------------------------------------------------
# Βοηθητικά: ίδιος κόσμος & metrics με τα προηγούμενα
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
        CellTarget(id=0, x=0.1, y=0.0, risk=0.9),
        CellTarget(id=1, x=3.0, y=0.0, risk=0.1),
        CellTarget(id=2, x=5.0, y=0.5, risk=0.1),
        CellTarget(id=3, x=0.0, y=5.5, risk=0.2),
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


def run_normal_policy(robots, targets):
    # explorer / neutral / conservative, όπως στο safe-mode demo
    risk_profile = {}
    if len(robots) > 0:
        risk_profile[robots[0].id] = 0.5
    if len(robots) > 1:
        risk_profile[robots[1].id] = 1.0
    if len(robots) > 2:
        risk_profile[robots[2].id] = 2.0

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
    # όλα τα ρομπότ super conservative
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
# Ablation main
# ---------------------------------------------------

def write_header(path: str):
    header = [
        "threshold",
        "policy",
        "num_steps",
        "mean_total_distance",
        "mean_total_risk",
        "mean_max_risk",
        "mean_total_cost",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_summary_row(
    path: str,
    threshold: float,
    policy: str,
    num_steps: int,
    mean_total_distance: float,
    mean_total_risk: float,
    mean_max_risk: float,
    mean_total_cost: float,
):
    row = [
        threshold,
        policy,
        num_steps,
        round(mean_total_distance, 4),
        round(mean_total_risk, 4),
        round(mean_max_risk, 4),
        round(mean_total_cost, 4),
    ]
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def main():
    random.seed(0)
    np.random.seed(0)

    thresholds = [0.5, 0.6, 0.7, 0.8]

    write_header(OUTPUT_CSV)

    for tau in thresholds:
        print(f"\n########## Threshold τ = {tau:.2f} ##########")

        # new self-trust manager for κάθε threshold
        trust = SelfTrustManager(
            alpha_calib=0.4,
            alpha_drift=0.6,
            safe_threshold=float(tau),
        )

        num_robots = 3
        for rid in range(num_robots):
            trust.register_robot(rid)

        robots, targets = build_fixed_world(num_robots=num_robots, num_targets=4)

        # συλλέγουμε metrics ανά policy
        metrics_per_policy: Dict[str, Dict[str, list]] = {
            "NORMAL_POLICY": {
                "total_distance": [],
                "total_risk": [],
                "max_risk": [],
                "total_cost": [],
            },
            "SAFE_MODE_POLICY": {
                "total_distance": [],
                "total_risk": [],
                "max_risk": [],
                "total_cost": [],
            },
        }

        for step in range(10):
            print(f"\n--- STEP {step} ---")

            # ίδια λογική με πριν: 0–4 "εύκολα", 5–9 "δύσκολα"
            if step < 5:
                calibration_error = np.clip(np.random.normal(0.15, 0.05), 0.0, 1.0)
                observed_drift = np.clip(np.random.normal(0.25, 0.05), 0.0, 1.0)
            else:
                calibration_error = np.clip(np.random.normal(0.35, 0.07), 0.0, 1.0)
                observed_drift = np.clip(np.random.normal(0.9, 0.05), 0.0, 1.0)

            expected_drift = np.clip(np.random.normal(0.25, 0.05), 0.0, 1.0)

            any_safe = False

            for rid in range(num_robots):
                S, safe = trust.update(
                    robot_id=rid,
                    calibration_error=float(calibration_error),
                    expected_drift=float(expected_drift),
                    observed_drift=float(observed_drift),
                )
                mode = "SAFE" if safe else "NORMAL"
                print(f"Robot {rid} | S={S:.3f} | {mode}")
                if safe:
                    any_safe = True

            pol_norm, assign_norm, metrics_norm = run_normal_policy(robots, targets)
            pol_safe, assign_safe, metrics_safe = run_safe_policy(robots, targets)

            if any_safe:
                active_policy = pol_safe
                active_metrics = metrics_safe
                active_assign = assign_safe
            else:
                active_policy = pol_norm
                active_metrics = metrics_norm
                active_assign = assign_norm

            print(f"Active policy: {active_policy}")
            print(f"Assignments:   {active_assign}")
            print(
                f"Metrics: dist={active_metrics['total_distance']:.3f}, "
                f"risk={active_metrics['total_risk']:.3f}, "
                f"max_risk={active_metrics['max_risk']:.3f}, "
                f"cost={active_metrics['total_cost']:.3f}"
            )

            # αποθήκευση metrics για το active policy
            mp = metrics_per_policy[active_policy]
            mp["total_distance"].append(active_metrics["total_distance"])
            mp["total_risk"].append(active_metrics["total_risk"])
            mp["max_risk"].append(active_metrics["max_risk"])
            mp["total_cost"].append(active_metrics["total_cost"])

        # στο τέλος: summary per policy για αυτό το threshold
        for policy, mdict in metrics_per_policy.items():
            if not mdict["total_distance"]:
                continue  # policy που δεν ενεργοποιήθηκε ποτέ

            arr_dist = np.array(mdict["total_distance"], dtype=float)
            arr_risk = np.array(mdict["total_risk"], dtype=float)
            arr_maxr = np.array(mdict["max_risk"], dtype=float)
            arr_cost = np.array(mdict["total_cost"], dtype=float)

            mean_dist = float(arr_dist.mean())
            mean_risk = float(arr_risk.mean())
            mean_maxr = float(arr_maxr.mean())
            mean_cost = float(arr_cost.mean())
            num_steps = len(arr_dist)

            print(f"\nSummary for τ={tau:.2f}, policy={policy}:")
            print(f"  steps used: {num_steps}")
            print(f"  mean distance: {mean_dist:.3f}")
            print(f"  mean risk:     {mean_risk:.3f}")
            print(f"  mean max risk: {mean_maxr:.3f}")
            print(f"  mean cost:     {mean_cost:.3f}")

            append_summary_row(
                OUTPUT_CSV,
                threshold=tau,
                policy=policy,
                num_steps=num_steps,
                mean_total_distance=mean_dist,
                mean_total_risk=mean_risk,
                mean_max_risk=mean_maxr,
                mean_total_cost=mean_cost,
            )


if __name__ == "__main__":
    main()
