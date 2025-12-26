import pandas as pd
import numpy as np


def select_best_under_budget(df, budget):
    """
    Δεδομένο risk budget (μέγιστο fused_mean),
    επιλέγει τη λύση με ελάχιστο geometric_length
    μεταξύ όσων έχουν fused_mean <= budget.
    """
    feasible = df[df["fused_mean"] <= budget]

    if feasible.empty:
        return None

    # αν υπάρχουν πολλές με ίδιο geom length, παίρνουμε αυτή με το μικρότερο λ
    min_geom = feasible["geometric_length"].min()
    candidates = feasible[feasible["geometric_length"] == min_geom]
    best = candidates.sort_values("lambda").iloc[0]
    return best


def main():
    csv_path = "belief_risk_lambda_sweep.csv"
    df = pd.read_csv(csv_path)
    print("[INFO] Loaded lambda sweep:")
    print(df)

    # Μπορείς να αλλάξεις / προσθέσεις budgets εδώ
    risk_budgets = [0.4, 0.3, 0.25, 0.23, 0.22]

    rows_out = []
    print("\n=== RISK BUDGET ANALYSIS (constraint on fused_mean) ===")
    for b in risk_budgets:
        best = select_best_under_budget(df, b)
        if best is None:
            print(f"\nBudget fused_mean <= {b:.3f}:")
            print("  -> No feasible path (all paths have higher fused_mean).")
            rows_out.append(
                {
                    "risk_budget": b,
                    "feasible": False,
                    "lambda": np.nan,
                    "geometric_length": np.nan,
                    "fused_mean": np.nan,
                    "fused_sum": np.nan,
                    "total_cost": np.nan,
                }
            )
        else:
            print(f"\nBudget fused_mean <= {b:.3f}:")
            print(
                f"  -> Selected λ = {best['lambda']} "
                f"(geom_len = {best['geometric_length']:.3f}, "
                f"fused_mean = {best['fused_mean']:.3f}, "
                f"fused_sum = {best['fused_sum']:.3f}, "
                f"total_cost = {best['total_cost']:.3f})"
            )
            rows_out.append(
                {
                    "risk_budget": b,
                    "feasible": True,
                    "lambda": best["lambda"],
                    "geometric_length": best["geometric_length"],
                    "fused_mean": best["fused_mean"],
                    "fused_sum": best["fused_sum"],
                    "total_cost": best["total_cost"],
                }
            )

    out_df = pd.DataFrame(rows_out)
    out_csv = "belief_risk_budget_selection.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\n[INFO] Saved budget selection table to {out_csv}")


if __name__ == "__main__":
    main()
