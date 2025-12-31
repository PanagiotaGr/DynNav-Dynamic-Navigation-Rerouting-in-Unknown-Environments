"""
analyze_multi_robot_risk_results.py

Ανάλυση των αποτελεσμάτων από run_multi_robot_risk_experiment.py.

Υπολογίζει μέση τιμή και τυπική απόκλιση ανά policy για:
- total_distance
- total_risk
- max_risk
- total_cost
"""

import csv
from collections import defaultdict
from math import sqrt


def mean(values):
    return sum(values) / len(values) if values else 0.0


def std(values):
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return sqrt(var)


def main():
    input_csv = "multi_robot_risk_results.csv"

    # policy -> metric -> list[float]
    data = defaultdict(lambda: defaultdict(list))

    with open(input_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            policy = row["policy"]
            data[policy]["total_distance"].append(float(row["total_distance"]))
            data[policy]["total_risk"].append(float(row["total_risk"]))
            data[policy]["max_risk"].append(float(row["max_risk"]))
            data[policy]["total_cost"].append(float(row["total_cost"]))

    print(f"Analysis of {input_csv}:\n")
    for policy, metrics in data.items():
        print(f"=== Policy: {policy} ===")

        for metric_name, values in metrics.items():
            m = mean(values)
            s = std(values)
            print(f"{metric_name:>15}: mean = {m:.3f}, std = {s:.3f}")
        print()


if __name__ == "__main__":
    main()
