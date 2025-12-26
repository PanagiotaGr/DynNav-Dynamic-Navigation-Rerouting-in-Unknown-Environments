import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_adaptive_results(csv_path: Path):
    episodes = []
    lambda_before = []
    lambda_after = []
    fused_risk = []
    target_risk_eff = []

    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            lambda_before.append(float(row["lambda_before"]))
            lambda_after.append(float(row["lambda_after"]))
            fused_risk.append(float(row["fused_risk"]))
            target_risk_eff.append(float(row["target_risk_effective"]))

    return episodes, lambda_before, lambda_after, fused_risk, target_risk_eff


def main():
    csv_path = Path("results/adaptive_belief_risk_runs.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run adaptive_belief_risk_planner.py first."
        )

    episodes, lam_b, lam_a, fused, target_eff = load_adaptive_results(csv_path)

    # Plot lambda evolution
    plt.figure()
    plt.plot(episodes, lam_b, marker="o", label="lambda_before")
    plt.plot(episodes, lam_a, marker="x", label="lambda_after")
    plt.xlabel("Episode")
    plt.ylabel("Lambda")
    plt.title("Adaptive lambda evolution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/adaptive_lambda_evolution.png")

    # Plot fused risk vs target
    plt.figure()
    plt.plot(episodes, fused, marker="o", label="fused_risk")
    plt.plot(episodes, target_eff, marker="x", label="target_risk_effective")
    plt.xlabel("Episode")
    plt.ylabel("Risk")
    plt.title("Fused risk vs effective target")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/adaptive_risk_tracking.png")

    # Αν θέλεις και interactive:
    # plt.show()


if __name__ == "__main__":
    main()
