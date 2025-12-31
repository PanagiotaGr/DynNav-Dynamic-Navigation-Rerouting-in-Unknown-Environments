"""
plot_lambda_sweep_tradeoff.py

Φτιάχνει ένα 2D plot με:

    x = total_distance
    y = total_risk

για κάθε τιμή λ από το lambda_sweep_risk_length_results.csv.

Έτσι φαίνεται καθαρά το trade-off:
- μικρό λ → μικρή απόσταση, μεγάλο ρίσκο
- μεγάλο λ → μεγαλύτερη απόσταση, μικρό ρίσκο
"""

import csv
import matplotlib.pyplot as plt

INPUT_CSV = "lambda_sweep_risk_length_results.csv"


def load_rows(path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def main():
    rows = load_rows(INPUT_CSV)
    if not rows:
        print("No rows in", INPUT_CSV)
        return

    xs = []
    ys = []
    labels = []

    for r in rows:
        lam = float(r["lambda"])
        td = float(r["total_distance"])
        tr = float(r["total_risk"])
        xs.append(td)
        ys.append(tr)
        labels.append(lam)

    plt.figure(figsize=(6, 5))
    plt.plot(xs, ys, marker="o")
    for x, y, lam in zip(xs, ys, labels):
        plt.text(x + 0.03, y + 0.03, f"λ={lam:g}", fontsize=8)

    plt.xlabel("Total distance")
    plt.ylabel("Total risk")
    plt.title("Length–Risk trade-off under J_λ = length + λ·risk")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lambda_sweep_length_risk_tradeoff.png", dpi=200)

    print("Saved plot: lambda_sweep_length_risk_tradeoff.png")


if __name__ == "__main__":
    main()
