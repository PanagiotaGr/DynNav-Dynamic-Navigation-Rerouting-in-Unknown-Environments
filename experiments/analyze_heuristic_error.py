import csv
import math
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent.parent
ERR_CSV = BASE / "phd_experiments" / "heuristic_error_data.csv"
OUT_MD = BASE / "phd_experiments" / "heuristic_error_summary.md"
PLOTS_DIR = BASE / "phd_experiments" / "error_plots"


def load_error_data():
    records = []
    with ERR_CSV.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "instance_id": int(row["instance_id"]),
                "x": int(row["state_x"]),
                "y": int(row["state_y"]),
                "h_star": float(row["h_star"]),
                "h_pred": float(row["h_pred"]),
                "error": float(row["error"]),
                "abs_error": float(row["abs_error"]),
                "rel_error": float(row["rel_error"])
            })
    return records


def plot_hist_error(errors, out):
    plt.figure()
    plt.hist(errors, bins=50)
    plt.title("Histogram of heuristic error (h_pred - h_star)")
    plt.xlabel("Error")
    plt.ylabel("Count")
    plt.savefig(out)
    plt.close()


def plot_scatter_hstar_error(h_star, abs_error, out):
    plt.figure()
    plt.scatter(h_star, abs_error, s=5, alpha=0.3)
    plt.title("Abs Error vs h*(s)")
    plt.xlabel("h*(s)")
    plt.ylabel("|error|")
    plt.savefig(out)
    plt.close()


def plot_scatter_predstar_error(h_pred, abs_error, out):
    plt.figure()
    plt.scatter(h_pred, abs_error, s=5, alpha=0.3)
    plt.title("Abs Error vs h_pred(s)")
    plt.xlabel("h_pred(s)")
    plt.ylabel("|error|")
    plt.savefig(out)
    plt.close()


def main():
    if not ERR_CSV.exists():
        print("[ERROR] No error data to analyze.")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_error_data()

    errors = [r["error"] for r in data]
    abs_errors = [r["abs_error"] for r in data]
    h_star_vals = [r["h_star"] for r in data]
    h_pred_vals = [r["h_pred"] for r in data]

    # 1. plot histogram
    plot_hist_error(errors, PLOTS_DIR / "hist_error.png")

    # 2. abs error vs h*(s)
    plot_scatter_hstar_error(h_star_vals, abs_errors, PLOTS_DIR / "scatter_hstar_abs.png")

    # 3. abs error vs h_pred(s)
    plot_scatter_predstar_error(h_pred_vals, abs_errors, PLOTS_DIR / "scatter_hpred_abs.png")

    # 4. summary markdown
    mean_err = sum(errors) / len(errors)
    mean_abs = sum(abs_errors) / len(abs_errors)

    lines = []
    lines.append("# Heuristic Error Analysis Summary\n")
    lines.append(f"- Total samples: {len(data)}")
    lines.append(f"- Mean signed error: {mean_err:.4f}")
    lines.append(f"- Mean absolute error: {mean_abs:.4f}\n")
    lines.append("## Figures\n")
    lines.append(f"![](error_plots/hist_error.png)")
    lines.append(f"![](error_plots/scatter_hstar_abs.png)")
    lines.append(f"![](error_plots/scatter_hpred_abs.png)")

    OUT_MD.write_text("\n".join(lines))
    print(f"[INFO] Wrote summary to {OUT_MD}")


if __name__ == "__main__":
    main()
