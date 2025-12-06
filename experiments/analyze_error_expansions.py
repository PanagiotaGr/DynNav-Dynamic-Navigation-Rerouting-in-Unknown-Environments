import csv
import math
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent

ERROR_CSV = BASE / "phd_experiments" / "heuristic_error_data.csv"
ASTAR_CSV = BASE / "phd_experiments" / "astar_phd_results.csv"
OUT_MD = BASE / "phd_experiments" / "error_expansions_summary.md"
PLOTS = BASE / "phd_experiments" / "correlation_plots"


def load_error_data():
    per_inst = defaultdict(list)
    with ERROR_CSV.open("r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            inst = int(r["instance_id"])
            e   = float(r["error"])
            ae  = float(r["abs_error"])
            per_inst[inst].append((e,ae))
    return per_inst


def load_expansion_data():
    exp = {}
    cost = {}
    with ASTAR_CSV.open("r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["method"] != "learned":
                continue
            inst = int(r["instance_id"])
            exp[inst] = int(r["expansions"])
            cost[inst] = float(r["path_cost"])
    return exp, cost


def compute_metrics(per_inst_err):
    per_inst_metrics = {}
    for inst, vals in per_inst_err.items():
        signed = [v[0] for v in vals]
        absvals = [v[1] for v in vals]
        over = sum(1 for s in signed if s > 0) / len(vals)
        under = sum(1 for s in signed if s < 0) / len(vals)

        per_inst_metrics[inst] = {
            "mean_signed": sum(signed)/len(signed),
            "mean_abs": sum(absvals)/len(absvals),
            "over_rate": over,
            "under_rate": under
        }
    return per_inst_metrics


def scatter(x, y, labelx, labely, out):
    plt.figure()
    plt.scatter(x,y,alpha=0.6,s=16)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.title(f"{labely} vs {labelx}")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)

    per_inst_err = load_error_data()
    expansions, cost = load_expansion_data()

    metrics = compute_metrics(per_inst_err)

    # scatter 1: mean_abs_error vs expansions
    xs = []
    ys = []
    for inst,m in metrics.items():
        if inst in expansions:
            xs.append(m["mean_abs"])
            ys.append(expansions[inst])

    scatter(xs, ys, "mean_abs_error", "expansions", PLOTS/"abs_vs_exp.png")

    # scatter 2: overestimation_rate vs path_cost
    xs = []
    ys = []
    for inst,m in metrics.items():
        if inst in cost:
            xs.append(m["over_rate"])
            ys.append(cost[inst])

    scatter(xs, ys, "over_rate", "path_cost", PLOTS/"over_vs_cost.png")

    # save summary
    lines = []
    lines.append("# Error vs Expansions Correlation Summary\n")
    lines.append("Scatter plots generated:\n")
    lines.append("- mean_abs_error vs expansions")
    lines.append("- overestimation_rate vs path_cost\n")
    lines.append("Figures stored in `phd_experiments/correlation_plots/`.\n")

    OUT_MD.write_text("\n".join(lines))
    print("[INFO] Wrote summary")


if __name__ == "__main__":
    main()
