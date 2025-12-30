# ================================================================
# Risk Explanation Experiment (Path-Level Tradeoffs)
# ================================================================

from risk_explainer import PathStats, choose_best_path, explain_preference


def main():
    # Ορίζουμε τρεις υποθετικές "διαδρομές"
    path_A = PathStats(
        name="A",
        length=10.0,
        drift_exposure=5.0,
        uncertainty_exposure=2.0,
    )
    path_B = PathStats(
        name="B",
        length=11.0,
        drift_exposure=3.0,
        uncertainty_exposure=1.5,
    )
    path_C = PathStats(
        name="C",
        length=13.0,
        drift_exposure=2.0,
        uncertainty_exposure=1.0,
    )

    paths = [path_A, path_B, path_C]

    lambda_risk = 1.0

    best = choose_best_path(paths, lambda_risk=lambda_risk)
    print(f"[RISK-INFO] Best path according to cost = length + λ*(drift+unc): {best.name}")

    # Εξηγήσεις μεταξύ ζευγών
    exp_AB = explain_preference(path_A, path_B, lambda_risk=lambda_risk)
    exp_BC = explain_preference(path_B, path_C, lambda_risk=lambda_risk)
    exp_AC = explain_preference(path_A, path_C, lambda_risk=lambda_risk)

    print("\n[RISK-EXPLANATIONS]")
    print("A vs B:")
    print(" ", exp_AB)
    print("\nB vs C:")
    print(" ", exp_BC)
    print("\nA vs C:")
    print(" ", exp_AC)


if __name__ == "__main__":
    main()
