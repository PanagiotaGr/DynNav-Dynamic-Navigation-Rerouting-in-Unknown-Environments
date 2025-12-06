import csv
import math
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_CSV = BASE_DIR / "phd_experiments" / "ablation_hidden_results.csv"
OUT_MD = BASE_DIR / "phd_experiments" / "ablation_hidden_summary.md"


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

    # group by method (classic, learned_h32, learned_h64, learned_h128, ...)
    groups = defaultdict(lambda: {"expansions": [], "runtime_sec": [], "path_cost": []})

    with RESULTS_CSV.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["success"]) != 1:
                continue
            method = row["method"]
            exp = int(row["expansions"])
            t = float(row["runtime_sec"])
            c = float(row["path_cost"])
            groups[method]["expansions"].append(exp)
            groups[method]["runtime_sec"].append(t)
            groups[method]["path_cost"].append(c)

    lines = []
    lines.append("# Ablation: Hidden Size vs Performance\n")
    lines.append("| Method | #Instances | Expansions (mean±std) | Runtime [s] (mean±std) | Path cost (mean±std) |")
    lines.append("|--------|------------|------------------------|------------------------|----------------------|")

    for method in sorted(groups.keys()):
        data = groups[method]
        n = len(data["expansions"])
        exp_m, exp_s = mean_std(data["expansions"])
        t_m, t_s = mean_std(data["runtime_sec"])
        c_m, c_s = mean_std(data["path_cost"])

        lines.append(
            f"| {method} | {n} | "
            f"{exp_m:.1f} ± {exp_s:.1f} | "
            f"{t_m:.4f} ± {t_s:.4f} | "
            f"{c_m:.2f} ± {c_s:.2f} |"
        )

    OUT_MD.write_text("\n".join(lines))
    print(f"[INFO] Wrote ablation summary to {OUT_MD}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

