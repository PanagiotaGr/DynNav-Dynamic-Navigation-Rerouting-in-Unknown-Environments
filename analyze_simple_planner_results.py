"""
analyze_simple_planner_results.py

Διαβάζει το simple_planner_results.csv και υπολογίζει:

- Για κάθε mode (baseline / human) και κάθε human_pref_text:
    * πόσες φορές κέρδισε κάθε path (A/B/C)
    * μέσο cost ανά path
    * μέσο lambda_effective

Έτσι μπορείς να δεις:
- πώς αλλάζει η επιλογή path μεταξύ baseline και human mode
- τι κάνει κάθε preference στην πράξη.
"""

import csv
from collections import defaultdict
from typing import Dict, Tuple, List


def load_results(csv_path: str):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Μετατροπή types
            r["length"] = float(r["length"])
            r["risk"] = float(r["risk"])
            r["is_dark"] = (r["is_dark"] == "True")
            r["is_low_feature"] = (r["is_low_feature"] == "True")
            r["lambda_robot"] = float(r["lambda_robot"])
            r["lambda_effective"] = float(r["lambda_effective"])
            r["cost"] = float(r["cost"])
            r["is_best"] = int(r["is_best"])
            rows.append(r)
    return rows


def aggregate_by_mode_pref(rows) -> Dict[Tuple[str, str], Dict]:
    """
    Ομαδοποιεί ανά (mode, human_pref_text).

    Για κάθε group κρατάει:
        - per-path stats:
            count, best_count, sum_cost, sum_lambda_effective
        - συνολικό #samples
    """
    agg = {}

    for r in rows:
        mode = r["mode"]
        pref = r["human_pref_text"] if mode == "human" else ""  # empty για baseline
        key = (mode, pref)

        if key not in agg:
            agg[key] = {
                "total_rows": 0,
                "paths": defaultdict(lambda: {
                    "count": 0,
                    "best_count": 0,
                    "sum_cost": 0.0,
                    "sum_lambda_effective": 0.0,
                }),
            }

        group = agg[key]
        group["total_rows"] += 1

        pname = r["path_name"]
        pstats = group["paths"][pname]

        pstats["count"] += 1
        pstats["sum_cost"] += r["cost"]
        pstats["sum_lambda_effective"] += r["lambda_effective"]
        if r["is_best"] == 1:
            pstats["best_count"] += 1

    return agg


def print_summary(agg):
    """
    Τυπώνει συγκεντρωτικά αποτελέσματα σε ανθρώπινη μορφή.
    """
    for (mode, pref), group in agg.items():
        print("=" * 80)
        print(f"Mode: {mode}")
        if mode == "human":
            print(f"Human preference: {pref}")
        else:
            print("(no human preference)")
        print(f"Total rows (paths evaluated): {group['total_rows']}")
        print()

        # Συγκεντρώνουμε paths με κάποιο fixed order για σταθερότητα
        path_names = sorted(group["paths"].keys())

        for pname in path_names:
            stats = group["paths"][pname]
            count = stats["count"]
            best_count = stats["best_count"]
            if count > 0:
                avg_cost = stats["sum_cost"] / count
                avg_lambda_eff = stats["sum_lambda_effective"] / count
            else:
                avg_cost = float("nan")
                avg_lambda_eff = float("nan")

            print(f"Path {pname}:")
            print(f"  count           = {count}")
            print(f"  best_count      = {best_count}")
            if count > 0:
                print(f"  best_rate       = {best_count / count:.3f}")
                print(f"  avg_cost        = {avg_cost:.3f}")
                print(f"  avg_lambda_eff  = {avg_lambda_eff:.3f}")
            print()
        print()


def main():
    csv_path = "simple_planner_results.csv"
    print(f"Φόρτωση αποτελεσμάτων από: {csv_path}")
    rows = load_results(csv_path)
    if not rows:
        print("Δεν βρέθηκαν γραμμές στο CSV. Τρέξε πρώτα το save_simple_planner_results.py.")
        return

    agg = aggregate_by_mode_pref(rows)
    print_summary(agg)


if __name__ == "__main__":
    main()
