"""
save_simple_planner_results.py

Τρέχει:
- τον baseline risk-aware planner (χωρίς ανθρώπινη προτίμηση)
- τον human-aware risk planner (με HumanPreference)

και αποθηκεύει τα αποτελέσματα σε ένα CSV:

    simple_planner_results.csv

Κάθε γραμμή του CSV αντιστοιχεί σε ένα path candidate
και περιέχει:
- ανήκει σε baseline ή human mode
- ποιο path είναι (A/B/C)
- length, risk
- flags dark / low-feature
- lambda_robot, lambda_effective
- cost
- αν είναι το best path για το αντίστοιχο mode.
"""

import csv
import os

from simple_risk_planner import SimpleRiskPlanner
from simple_risk_planner_human import SimpleHumanAwareRiskPlanner


def get_candidates():
    """
    Ορίζουμε κοινά candidates για όλα τα runs,
    ώστε baseline vs human να είναι συγκρίσιμα.
    """
    candidates = [
        {"name": "A", "length": 10.0, "risk": 0.2, "is_dark": False, "is_low_feature": False},
        {"name": "B", "length": 8.0, "risk": 0.6, "is_dark": True,  "is_low_feature": True},
        {"name": "C", "length": 12.0, "risk": 0.1, "is_dark": False, "is_low_feature": True},
    ]
    return candidates


def run_baseline(lambda_robot: float, candidates, csv_writer):
    """
    Τρέχει τον SimpleRiskPlanner και γράφει τα αποτελέσματα στο CSV.
    """
    planner = SimpleRiskPlanner(lambda_robot=lambda_robot)

    # Υπολογίζουμε cost για κάθε path
    costs = []
    for c in candidates:
        cost = planner.evaluate_path_cost(c["length"], c["risk"])
        costs.append((c, cost))

    # Βρίσκουμε το καλύτερο path
    best_c, best_cost = planner.select_best_path(candidates)
    best_name = best_c["name"]

    # Γράφουμε όλες τις γραμμές
    for c, cost in costs:
        row = {
            "mode": "baseline",
            "human_pref_text": "",
            "path_name": c["name"],
            "length": c["length"],
            "risk": c["risk"],
            "is_dark": c.get("is_dark", False),
            "is_low_feature": c.get("is_low_feature", False),
            "lambda_robot": lambda_robot,
            "lambda_effective": lambda_robot,  # baseline -> ίδιο με robot
            "cost": cost,
            "is_best": int(c["name"] == best_name),
        }
        csv_writer.writerow(row)


def run_human_mode(lambda_robot: float, human_pref_text: str, human_influence_scale: float,
                   candidates, csv_writer):
    """
    Τρέχει τον SimpleHumanAwareRiskPlanner και γράφει τα αποτελέσματα στο CSV.
    """
    planner = SimpleHumanAwareRiskPlanner(
        lambda_robot=lambda_robot,
        human_pref_text=human_pref_text,
        human_influence_scale=human_influence_scale,
    )

    costs = []
    lambda_effs = []

    for c in candidates:
        cost, lambda_eff = planner.evaluate_path_cost(
            path_length=c["length"],
            path_risk=c["risk"],
            is_dark=c.get("is_dark", False),
            is_low_feature=c.get("is_low_feature", False),
        )
        costs.append((c, cost))
        lambda_effs.append(lambda_eff)

    # Βρίσκουμε το καλύτερο path
    best, best_cost, best_lambda = planner.select_best_path(candidates)
    best_name = best["name"]

    # Γράφουμε όλες τις γραμμές
    for (c, cost), lambda_eff in zip(costs, lambda_effs):
        row = {
            "mode": "human",
            "human_pref_text": human_pref_text,
            "path_name": c["name"],
            "length": c["length"],
            "risk": c["risk"],
            "is_dark": c.get("is_dark", False),
            "is_low_feature": c.get("is_low_feature", False),
            "lambda_robot": lambda_robot,
            "lambda_effective": lambda_eff,
            "cost": cost,
            "is_best": int(c["name"] == best_name),
        }
        csv_writer.writerow(row)


def main():
    results_path = "simple_planner_results.csv"
    file_exists = os.path.exists(results_path)

    fieldnames = [
        "mode",
        "human_pref_text",
        "path_name",
        "length",
        "risk",
        "is_dark",
        "is_low_feature",
        "lambda_robot",
        "lambda_effective",
        "cost",
        "is_best",
    ]

    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # Αν το αρχείο δεν υπήρχε, γράφουμε header
        if not file_exists:
            writer.writeheader()

        candidates = get_candidates()
        lambda_robot = 1.0

        # 1) Baseline run
        run_baseline(lambda_robot=lambda_robot, candidates=candidates, csv_writer=writer)

        # 2) Human-aware runs με διαφορετικές προτιμήσεις
        human_prefs = [
            "Προτίμηση: πιο ασφαλής διαδρομή, ακόμη κι αν είναι αργή",
            "Προτίμηση: φτάσε γρήγορα, αποδέχομαι ρίσκο",
            "Αποφεύγετε σκοτεινές / low-feature περιοχές",
        ]
        human_influence_scale = 1.0

        for pref_text in human_prefs:
            run_human_mode(
                lambda_robot=lambda_robot,
                human_pref_text=pref_text,
                human_influence_scale=human_influence_scale,
                candidates=candidates,
                csv_writer=writer,
            )

    print(f"Αποτελέσματα αποθηκεύτηκαν στο '{results_path}'.")


if __name__ == "__main__":
    main()
