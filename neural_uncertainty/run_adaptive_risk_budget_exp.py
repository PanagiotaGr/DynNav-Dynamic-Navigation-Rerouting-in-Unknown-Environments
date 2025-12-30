# ================================================================
# Adaptive Risk Budget Experiment
#
# Δείχνει πώς αλλάζει η επιλογή διαδρομής
# όταν προσαρμόζουμε το λ (risk weight) με βάση Self-Trust S.
# ================================================================

from risk_explainer import PathStats, choose_best_path, explain_preference


def lambda_from_self_trust(S: float) -> float:
    """
    Απλό παράδειγμα adaptive λ:

    - S κοντά στο 1  → εμπιστευόμαστε πολύ το σύστημα → χαμηλό λ (δέχεται περισσότερο ρίσκο)
    - S κοντά στο 0  → δεν εμπιστευόμαστε → υψηλό λ (πολύ συντηρητικό)

    Λογική: λ(S) = λ_min + (λ_max - λ_min) * (1 - S)
    """

    lambda_min = 0.3
    lambda_max = 3.0

    lam = lambda_min + (lambda_max - lambda_min) * (1.0 - S)
    return lam


def run_for_self_trust(S: float):
    print(f"\n[ARB-INFO] Self-Trust S = {S:.2f}")

    lam = lambda_from_self_trust(S)
    print(f"[ARB-INFO] Adaptive λ = {lam:.2f}")

    # Τρεις υποθετικές διαδρομές όπως στο προηγούμενο πείραμα
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

    best = choose_best_path(paths, lambda_risk=lam)
    print(f"[ARB-RESULT] Best path for S={S:.2f}, λ={lam:.2f} is: {best.name}")

    # Εξηγούμε τουλάχιστον μία βασική σύγκριση σε σχέση με την A
    for p in paths:
        if p.name != best.name:
            exp = explain_preference(best, p, lambda_risk=lam)
            print(f"[ARB-EXPLAIN] {best.name} vs {p.name}:")
            print(" ", exp)


def main():
    # Δοκιμάζουμε τρία επίπεδα Self-Trust
    # Χαμηλό S → πολύ συντηρητικό (μεγάλο λ)
    # Μεσαίο S → balanced
    # Υψηλό S → πιο "επιθετικό" navigation
    self_trust_values = [0.2, 0.5, 0.8]

    for S in self_trust_values:
        run_for_self_trust(S)


if __name__ == "__main__":
    main()
