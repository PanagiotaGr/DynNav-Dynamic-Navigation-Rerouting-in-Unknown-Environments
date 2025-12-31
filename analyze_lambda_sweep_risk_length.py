"""
analyze_lambda_sweep_risk_length.py

Διαβάζει το lambda_sweep_risk_length_results.csv και
τυπώνει ένα μικρό summary ανά τιμή λ για:

- total_distance
- total_risk
- max_risk
- total_cost

Αυτό είναι χρήσιμο για να περιγράψεις στο κείμενο
το trade-off length vs risk όσο αυξάνεται το λ.
"""

import csv


INPUT_CSV = "lambda_sweep_risk_length_results.csv"


def load_rows(path: str):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def main():
    rows = load_rows(INPUT_CSV)
    if not rows:
        print("No rows found in", INPUT_CSV)
        return

    print("Analysis of lambda_sweep_risk_length_results.csv:\n")

    for r in rows:
        lam = float(r["lambda"])
        assignments = r["assignments"]
        td = float(r["total_distance"])
        tr = float(r["total_risk"])
        mr = float(r["max_risk"])
        tc = float(r["total_cost"])

        print(f"λ = {lam:.2f}")
        print(f"  assignments   = {assignments}")
        print(f"  total_distance = {td:.3f}")
        print(f"  total_risk     = {tr:.3f}")
        print(f"  max_risk       = {mr:.3f}")
        print(f"  total_cost     = {tc:.3f}")
        print()


if __name__ == "__main__":
    main()
