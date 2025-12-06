import csv
import math
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent  # root του project

RESULTS_CSV = BASE_DIR / "phd_experiments" / "astar_phd_results.csv"
OUT_MARKDOWN = BASE_DIR / "phd_experiments" / "astar_phd_summary.md"


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

    # group by (grid_type, method)
    groups = defaultdict(lambda: {"expansions": [], "runtime_sec": [], "path_cost": []})

    with RESULTS_CSV.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            grid_type = row["grid_type"]
            method = row["method"]
            key = (grid_type, method)

            success = int(row["success"])
            if success != 1:
                # αν θες να συμπεριλάβεις και failures, αφαίρεσε αυτό το continue
                continue

            expansions = int(row["expansions"])
            runtime_sec = float(row["runtime_sec"])
            path_cost = float(row["path_cost"])

            groups[key]["expansions"].append(expansions)
            groups[key]["runtime_sec"].append(runtime_sec)
            groups[key]["path_cost"].append(path_cost)

    # Ετοιμάζουμε markdown table
    lines = []
    lines.append("# A* Learned vs Classic – Summary\n")
    lines.append("")
    lines.append("| Grid type | Method | #Instances | Expansions (mean±std) | Runtime [s] (mean±std) | Path cost (mean±std) |")
    lines.append("|-----------|--------|------------|------------------------|------------------------|----------------------|")

    # ταξινόμηση για πιο ωραία εκτύπωση
    for (grid_type, method) in sorted(groups.keys()):
        data = groups[(grid_type, method)]
        n = len(data["expansions"])

        exp_mean, exp_std = mean_std(data["expansions"])
        t_mean, t_std = mean_std(data["runtime_sec"])
        c_mean, c_std = mean_std(data["path_cost"])

        lines.append(
            f"| {grid_type} | {method} | {n} | "
            f"{exp_mean:.1f} ± {exp_std:.1f} | "
            f"{t_mean:.4f} ± {t_std:.4f} | "
            f"{c_mean:.2f} ± {c_std:.2f} |"
        )

    text = "\n".join(lines)

    OUT_MARKDOWN.parent.mkdir(parents=True, exist_ok=True)
    OUT_MARKDOWN.write_text(text)
    print(f"[INFO] Wrote summary markdown to {OUT_MARKDOWN}")

    # Εκτυπώνουμε και στην κονσόλα
    print()
    print(text)


if __name__ == "__main__":
    main()
