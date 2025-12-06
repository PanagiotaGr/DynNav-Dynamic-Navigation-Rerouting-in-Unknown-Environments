import csv
import math
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_CSV = BASE_DIR / "phd_experiments" / "data_sweep_results.csv"
OUT_MD = BASE_DIR / "phd_experiments" / "data_sweep_summary.md"


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

    # group by method (classic, learned_frac_10, ...)
    groups = defaultdict(lambda: {"expansions": [], "runtime_sec": [], "path_cost": [], "fraction": None})

    with RESULTS_CSV.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["success"]) != 1:
                continue
            method = row["method"]
            frac = float(row["fraction"])
            exp = int(row["expansions"])
            t = float(row["runtime_sec"])
            c = float(row["path_cost"])

            g = groups[method]
            g["expansions"].append(exp)
            g["runtime_sec"].append(t)
            g["path_cost"].append(c)
            g["fraction"] = frac

    lines = []
    lines.append("# Data Sweep: Training Set Size vs Performance\n")
    lines.append("| Method | Fraction | #Instances | Expansions (mean±std) | Runtime [s] (mean±std) | Path cost (mean±std) |")
    lines.append("|--------|----------|------------|------------------------|------------------------|----------------------|")

    # sort so that classic first, then by fraction
    def sort_key(item):
        method, info = item
        if method == "classic":
            return (0, 0.0)
        frac = info["fraction"] if info["fraction"] is not None else 1.0
        return (1, frac)

    for method, info in sorted(groups.items(), key=sort_key):
        n = len(info["expansions"])
        exp_m, exp_s = mean_std(info["expansions"])
        t_m, t_s = mean_std(info["runtime_sec"])
        c_m, c_s = mean_std(info["path_cost"])
        frac = info["fraction"]

        frac_str = f"{frac:.2f}" if frac is not None else "-"

        lines.append(
            f"| {method} | {frac_str} | {n} | "
            f"{exp_m:.1f} ± {exp_s:.1f} | "
            f"{t_m:.4f} ± {t_s:.4f} | "
            f"{c_m:.2f} ± {c_s:.2f} |"
        )

    OUT_MD.write_text("\n".join(lines))
    print(f"[INFO] Wrote data sweep summary to {OUT_MD}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
