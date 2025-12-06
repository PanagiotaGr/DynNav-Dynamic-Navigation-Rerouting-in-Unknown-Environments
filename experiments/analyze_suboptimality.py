import csv
import math
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_CSV = BASE_DIR / "phd_experiments" / "astar_phd_results.csv"
OUT_MARKDOWN = BASE_DIR / "phd_experiments" / "astar_suboptimality_summary.md"


def safe_div(a, b):
    if b == 0:
        return float("inf")
    return a / b


def mean_std(values):
    if not values:
        return float("nan"), float("nan")
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    return m, math.sqrt(var)


def main():
    if not RESULTS_CSV.exists():
        print(f"[ERROR] Results file not found: {RESULTS_CSV}")
        return

    # 1) Διαβάζουμε όλα τα rows και ομαδοποιούμε ανά (instance, grid)
    by_instance = defaultdict(dict)
    methods_set = set()

    with RESULTS_CSV.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inst = int(row["instance_id"])
            grid = row["grid_type"]
            method = row["method"]
            success = int(row["success"])
            if success != 1:
                continue

            cost = float(row["path_cost"])
            by_instance[(inst, grid)][method] = cost
            methods_set.add(method)

    # classic είναι η reference
    if "classic" not in methods_set:
        print("[ERROR] No classic baseline found.")
        return

    # 2) Φτιάχνουμε λίστες C/C* για κάθε μέθοδο ≠ classic
    methods = sorted(m for m in methods_set if m != "classic")
    ratios = {m: [] for m in methods}

    for (inst, grid), methods_dict in by_instance.items():
        if "classic" not in methods_dict:
            continue
        c_star = methods_dict["classic"]

        for m in methods:
            if m in methods_dict:
                ratio = safe_div(methods_dict[m], c_star)
                ratios[m].append(ratio)

    # 3) Φτιάχνουμε markdown table
    lines = []
    lines.append("# Suboptimality Analysis (C/C*)\n")
    lines.append("| Method | Mean | Median | Std | % Optimal | Worst Case |")
    lines.append("|--------|------|--------|-----|-----------|------------|")

    for m in methods:
        vals = ratios[m]
        if not vals:
            continue
        mean_v, std_v = mean_std(vals)
        med_v = sorted(vals)[len(vals) // 2]
        pct_opt = sum(1 for v in vals if abs(v - 1.0) < 1e-9) / len(vals) * 100
        worst = max(vals)

        lines.append(
            f"| {m} | {mean_v:.4f} | {med_v:.4f} | {std_v:.4f} | "
            f"{pct_opt:.1f}% | {worst:.4f} |"
        )

    OUT_MARKDOWN.write_text("\n".join(lines))
    print(f"[INFO] Wrote suboptimality markdown to: {OUT_MARKDOWN}")

    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
