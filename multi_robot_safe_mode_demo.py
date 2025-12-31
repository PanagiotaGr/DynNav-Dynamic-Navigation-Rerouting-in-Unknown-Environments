"""
multi_robot_safe_mode_demo.py

Παράδειγμα ενσωμάτωσης Self-Trust + Safe Mode
σε multi-robot risk-aware σύστημα.

Δεν χρειάζεται ROS εδώ. Είναι framework-level demo:
- κάθε robot έχει self-trust S ∈ [0,1]
- αν πέσει κάτω από threshold μπαίνει σε SAFE MODE
- SAFE MODE => πιο conservative policy
- otherwise => normal / explorer policy

Μετά μπορείς:
- να το καλέσεις μέσα από multi_robot_coverage_sim.py
- ή run_multi_robot_experiment.py
"""

import random
import numpy as np

from multi_robot_risk_allocation import (
    RobotState,
    CellTarget,
    assign_with_risk_profiles,
    assign_tasks_hungarian,
)

from self_trust_manager import SelfTrustManager


# --------------------------------------------
# Generate dummy robots + dummy risky targets
# --------------------------------------------
def generate_dummy_world(num_robots=3, num_targets=6):
    robots = [
        RobotState(
            id=i,
            x=random.uniform(0, 10),
            y=random.uniform(0, 10)
        )
        for i in range(num_robots)
    ]

    targets = []
    for j in range(num_targets):
        x = random.uniform(0, 10)
        y = random.uniform(0, 10)

        # μερικές περιοχές high risk
        risk = np.clip(
            random.uniform(0.0, 1.0)
            + (0.5 if x > 6 and y > 6 else 0.0),
            0.0,
            1.0
        )

        targets.append(CellTarget(id=j, x=x, y=y, risk=risk))

    return robots, targets


# ---------------------------------------------------
# Policies (Normal vs Safe Mode)
# ---------------------------------------------------
def run_normal_policy(robots, targets):
    """
    Explorer/monitor στυλ policy
    """
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
        base_w_risk=2.0,
    )

    return "NORMAL_POLICY", assignments, total_cost


def run_safe_policy(robots, targets):
    """
    ΌΛΟΙ conservative
    """
    risk_profile = {r.id: 3.0 for r in robots}

    assignments, total_cost = assign_with_risk_profiles(
        robots,
        targets,
        risk_profile=risk_profile,
        base_w_dist=1.0,
        base_w_risk=2.0,
    )

    return "SAFE_MODE_POLICY", assignments, total_cost


# ---------------------------------------------------
# MAIN LOOP – εδώ κολλάει το Self Trust
# ---------------------------------------------------
def main():
    random.seed(7)
    np.random.seed(7)

    trust = SelfTrustManager(
        alpha_calib=0.6,
        alpha_drift=0.4,
        safe_threshold=0.5
    )

    num_robots = 3
    for rid in range(num_robots):
        trust.register_robot(rid)

    robots, targets = generate_dummy_world(
        num_robots=num_robots,
        num_targets=6
    )

    print("Robots:")
    for r in robots:
        print(r)

    print("\nTargets:")
    for t in targets:
        print(t)

    print("\n===== ONLINE LOOP =====")

    for step in range(10):
        print(f"\n--- STEP {step} ---")

        # (Demo) simulation signals
        calibration_error = np.clip(np.random.normal(0.2, 0.15), 0.0, 1.0)
        expected_drift = np.clip(np.random.normal(0.2, 0.1), 0.0, 1.0)

        # Harder world occasionally
        observed_drift = (
            np.clip(np.random.normal(0.25, 0.1), 0.0, 1.0)
            if step < 5
            else np.clip(np.random.normal(0.6, 0.15), 0.0, 1.0)
        )

        any_safe_mode = False

        for rid in range(num_robots):
            S, safe = trust.update(
                robot_id=rid,
                calibration_error=float(calibration_error),
                expected_drift=float(expected_drift),
                observed_drift=float(observed_drift),
            )

            mode = "SAFE MODE" if safe else "NORMAL"
            print(f"Robot {rid} | S={S:.3f} | {mode}")

            if safe:
                any_safe_mode = True

        # ----------------------
        # Policy Switch
        # ----------------------
        if any_safe_mode:
            policy, assignments, cost = run_safe_policy(robots, targets)
        else:
            policy, assignments, cost = run_normal_policy(robots, targets)

        print(f"Active Policy: {policy}")
        print("Assignments:", assignments)
        print("Cost:", cost)


if __name__ == "__main__":
    main()
