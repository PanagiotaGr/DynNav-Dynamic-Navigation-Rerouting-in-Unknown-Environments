import csv
import math
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_CSV = BASE_DIR / "phd_experiments" / "astar_residual_results.csv"
OUT_MD = BASE_DIR / "phd_experiments" / "astar_residual_summary.md"


def mean_std(vals):
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return m, math.sqrt(var)


def main():
    if not RESULTS_CSV.exists():
        print(f"[ERROR] No results at {RESULTS_CSV}")
        return

    groups = defaultdict(lambda: {"expansions": [], "runtime_sec": [], "path_cost": []})

    with RESULTS_CSV.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["success"]) != 1:
                continue
            key = (row["grid_type"], row["method"])
            groups[key]["expansions"].append(int(row["expansions"]))
            groups[key]["runtime_sec"].append(float(row["runtime_sec"]))
            groups[key]["path_cost"].append(float(row["path_cost"]))

    lines = []
    lines.append("# Residual Heuristic vs Classic – Summary\n")
    lines.append("| Grid type | Method | #Inst | Expansions (mean±std) | Runtime [s] (mean±std) | Path cost (mean±std) |")
    lines.append("|-----------|--------|-------|------------------------|------------------------|----------------------|")

    for (grid, method) in sorted(groups.keys()):
        data = groups[(grid, method)]
        n = len(data["expansions"])
        exp_m, exp_s = mean_std(data["expansions"])
        t_m, t_s = mean_std(data["runtime_sec"])
        c_m, c_s = mean_std(data["path_cost"])

        lines.append(
            f"| {grid} | {method} | {n} | "
            f"{exp_m:.1f} ± {exp_s:.1f} | "
            f"{t_m:.4f} ± {t_s:.4f} | "
            f"{c_m:.2f} ± {c_s:.2f} |"
        )

    OUT_MD.write_text("\n".join(lines))
    print(f"[INFO] Wrote residual summary to {OUT_MD}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
