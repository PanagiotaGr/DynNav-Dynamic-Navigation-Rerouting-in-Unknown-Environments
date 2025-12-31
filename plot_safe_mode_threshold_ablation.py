"""
plot_safe_mode_threshold_ablation.py

Διαβάζει το safe_mode_threshold_ablation_results.csv και
φτιάχνει plots για:

- mean_total_distance vs threshold
- mean_total_risk vs threshold
- mean_max_risk vs threshold
- mean_total_cost vs threshold

ξεχωριστά για NORMAL_POLICY και SAFE_MODE_POLICY.

Αποθηκεύει δύο PNG:
- safe_mode_threshold_distance_risk.png
- safe_mode_threshold_maxrisk_cost.png
"""

import csv
from collections import defaultdict

import matplotlib.pyplot as plt


INPUT_CSV = "safe_mode_threshold_ablation_results.csv"


def load_results(path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def group_by_threshold_and_policy(rows):
    """
    Επιστρέφει dict:

    data[policy][threshold] = {
        "mean_total_distance": ...,
        "mean_total_risk": ...,
        "mean_max_risk": ...,
        "mean_total_cost": ...,
    }
    """
    data = defaultdict(dict)

    for r in rows:
        tau = float(r["threshold"])
        policy = r["policy"]

        data[policy][tau] = {
            "mean_total_distance": float(r["mean_total_distance"]),
            "mean_total_risk": float(r["mean_total_risk"]),
            "mean_max_risk": float(r["mean_max_risk"]),
            "mean_total_cost": float(r["mean_total_cost"]),
        }

    return data


def plot_distance_and_risk(data):
    """
    Plot:
        threshold (x) vs distance / risk (y)
    για NORMAL_POLICY και SAFE_MODE_POLICY.
    """
    policies = sorted(data.keys())
    thresholds = sorted(next(iter(data.values())).keys())

    plt.figure(figsize=(8, 5))

    # Distance
    for policy in policies:
        xs = []
        ys = []
        for tau in thresholds:
            if tau in data[policy]:
                xs.append(tau)
                ys.append(data[policy][tau]["mean_total_distance"])
        plt.plot(xs, ys, marker="o", label=f"{policy} - distance")

    plt.xlabel("Safe-mode threshold τ")
    plt.ylabel("Mean total distance")
    plt.title("Effect of safe-mode threshold on mean distance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("safe_mode_threshold_distance.png", dpi=200)

    # Risk
    plt.figure(figsize=(8, 5))
    for policy in policies:
        xs = []
        ys = []
        for tau in thresholds:
            if tau in data[policy]:
                xs.append(tau)
                ys.append(data[policy][tau]["mean_total_risk"])
        plt.plot(xs, ys, marker="o", label=f"{policy} - risk")

    plt.xlabel("Safe-mode threshold τ")
    plt.ylabel("Mean total risk")
    plt.title("Effect of safe-mode threshold on mean risk")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("safe_mode_threshold_risk.png", dpi=200)


def plot_maxrisk_and_cost(data):
    """
    Plot:
        threshold (x) vs max_risk / cost (y)
    """
    policies = sorted(data.keys())
    thresholds = sorted(next(iter(data.values())).keys())

    # max_risk
    plt.figure(figsize=(8, 5))
    for policy in policies:
        xs = []
        ys = []
        for tau in thresholds:
            if tau in data[policy]:
                xs.append(tau)
                ys.append(data[policy][tau]["mean_max_risk"])
        plt.plot(xs, ys, marker="o", label=f"{policy} - max risk")

    plt.xlabel("Safe-mode threshold τ")
    plt.ylabel("Mean max risk")
    plt.title("Effect of safe-mode threshold on max risk")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("safe_mode_threshold_maxrisk.png", dpi=200)

    # total_cost
    plt.figure(figsize=(8, 5))
    for policy in policies:
        xs = []
        ys = []
        for tau in thresholds:
            if tau in data[policy]:
                xs.append(tau)
                ys.append(data[policy][tau]["mean_total_cost"])
        plt.plot(xs, ys, marker="o", label=f"{policy} - total cost")

    plt.xlabel("Safe-mode threshold τ")
    plt.ylabel("Mean total cost")
    plt.title("Effect of safe-mode threshold on total cost")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("safe_mode_threshold_cost.png", dpi=200)


def main():
    rows = load_results(INPUT_CSV)
    if not rows:
        print("No rows in", INPUT_CSV)
        return

    data = group_by_threshold_and_policy(rows)

    plot_distance_and_risk(data)
    plot_maxrisk_and_cost(data)

    print("Saved plots:")
    print("  safe_mode_threshold_distance.png")
    print("  safe_mode_threshold_risk.png")
    print("  safe_mode_threshold_maxrisk.png")
    print("  safe_mode_threshold_cost.png")


if __name__ == "__main__":
    main()
