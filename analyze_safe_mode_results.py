"""
analyze_safe_mode_results.py

Διαβάζει το multi_robot_safe_mode_results.csv και
υπολογίζει συνοπτικά στατιστικά για:

- NORMAL_POLICY
- SAFE_MODE_POLICY

με βάση total_distance, total_risk, max_risk, total_cost.
"""

import csv
from collections import defaultdict
import numpy as np


INPUT_CSV = "multi_robot_safe_mode_results.csv"


def load_results(path: str):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def summarize_by_policy(rows):
    """
    Ομαδοποίηση ανά policy.
    Παρότι το CSV έχει μία γραμμή ανά robot ανά step,
    εδώ θα κάνουμε aggregate ανά (step, policy) ώστε να
    μετράμε κάθε mission μία φορά.
    """

    # key: (step, policy) -> metrics
    missions = {}

    for r in rows:
        step = int(r["step"])
        policy = r["policy"]
        key = (step, policy)

        # τα mission-level metrics είναι ίδια για όλα τα robots σε ίδιο step
        if key not in missions:
            missions[key] = {
                "total_distance": float(r["total_distance"]),
                "total_risk": float(r["total_risk"]),
                "max_risk": float(r["max_risk"]),
                "total_cost": float(r["total_cost"]),
            }

    # τώρα κάνουμε aggregate ανά policy
    per_policy = defaultdict(lambda: defaultdict(list))

    for (step, policy), m in missions.items():
        per_policy[policy]["total_distance"].append(m["total_distance"])
        per_policy[policy]["total_risk"].append(m["total_risk"])
        per_policy[policy]["max_risk"].append(m["max_risk"])
        per_policy[policy]["total_cost"].append(m["total_cost"])

    return per_policy


def print_summary(per_policy):
    print("Analysis of multi_robot_safe_mode_results.csv:\n")

    for policy, metrics_dict in per_policy.items():
        print(f"=== Policy: {policy} ===")
        for metric_name, values in metrics_dict.items():
            arr = np.array(values, dtype=float)
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            print(f" {metric_name:>14}: mean = {mean:.3f}, std = {std:.3f}")
        print()


def main():
    rows = load_results(INPUT_CSV)
    if not rows:
        print("No rows found in CSV.")
        return

    per_policy = summarize_by_policy(rows)
    print_summary(per_policy)


if __name__ == "__main__":
    main()
