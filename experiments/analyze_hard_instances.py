import csv
import math
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_CSV = BASE_DIR / "phd_experiments" / "astar_phd_results.csv"
OUT_MD = BASE_DIR / "phd_experiments" / "hard_instances_summary.md"
OUT_CSV = BASE_DIR / "phd_experiments" / "hard_instances.csv"


def mean_std(values):
    if not values:
        return float("nan"), float("nan")
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    return m, math.sqrt(var)


def main(top_k_per_grid: int = 10):
    if not RESULTS_CSV.exists():
        print(f"[ERROR] Results file not found: {RESULTS_CSV}")
        return

    # by_inst[key] = {method_name: row_dict}
    by_inst = defaultdict(dict)

    with RESULTS_CSV.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["success"]) != 1:
                continue
            inst = int(row["instance_id"])
            grid = row["grid_type"]
            method = row["method"]
            key = (inst, grid)
            by_inst[key][method] = row

    records = []

    for (inst, grid), methods in by_inst.items():
        if "classic" not in methods:
            # χρειάζομαι classic σαν reference
            continue

        classic = methods["classic"]
        exp_classic = int(classic["expansions"])
        cost_classic = float(classic["path_cost"])

        # βασικό difficulty metric: expansions του classic
        difficulty = exp_classic

        # optional metrics για learned
        exp_learned = None
        cost_learned = None
        speedup_learned = None
        subopt_learned = None

        if "learned" in methods:
            learned = methods["learned"]
            exp_learned = int(learned["expansions"])
            cost_learned = float(learned["path_cost"])

            if exp_learned > 0:
                speedup_learned = exp_classic / exp_learned
            if cost_classic > 0:
                subopt_learned = cost_learned / cost_classic

        # μπορούμε να βάλουμε κι άλλα (π.χ. uncertainty_k0, κ.λπ.), αλλά ας ξεκινήσουμε έτσι
        records.append({
            "instance_id": inst,
            "grid_type": grid,
            "difficulty_classic_exp": difficulty,
            "exp_classic": exp_classic,
            "cost_classic": cost_classic,
            "exp_learned": exp_learned if exp_learned is not None else "",
            "cost_learned": cost_learned if cost_learned is not None else "",
            "speedup_learned": speedup_learned if speedup_learned is not None else "",
            "subopt_learned": subopt_learned if subopt_learned is not None else "",
        })

    # χωρίζουμε ανά grid_type και διαλέγουμε τα top-K πιο δύσκολα
    by_grid = defaultdict(list)
    for rec in records:
        by_grid[rec["grid_type"]].append(rec)

    hard_instances = []
    for grid, recs in by_grid.items():
        recs_sorted = sorted(recs, key=lambda r: r["difficulty_classic_exp"], reverse=True)
        top_recs = recs_sorted[:top_k_per_grid]
        hard_instances.extend(top_recs)

    # γράφουμε detailed CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "instance_id",
        "grid_type",
        "difficulty_classic_exp",
        "exp_classic",
        "cost_classic",
        "exp_learned",
        "cost_learned",
        "speedup_learned",
        "subopt_learned",
    ]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in hard_instances:
            writer.writerow(rec)

    print(f"[INFO] Wrote hard instances CSV to {OUT_CSV}")

    # γράφουμε summary markdown
    lines = []
    lines.append("# Hard Instances Analysis\n")
    lines.append(
        "Αναγνωρίζουμε τα πιο δύσκολα instances (με βάση τα expansions του classic A*) "
        "για κάθε τύπο grid.\n"
    )
    lines.append("| Grid type | Instance ID | Expansions (classic) | Path cost (classic) | Expansions (learned) | Speedup learned | C_learned / C* |")
    lines.append("|-----------|-------------|----------------------|---------------------|----------------------|-----------------|-----------------|")

    # sort for pretty printing: by grid_type, then descending difficulty
    hard_instances_sorted = sorted(
        hard_instances,
        key=lambda r: (r["grid_type"], -r["difficulty_classic_exp"]),
    )

    for rec in hard_instances_sorted:
        grid = rec["grid_type"]
        inst = rec["instance_id"]
        exp_c = rec["exp_classic"]
        cost_c = rec["cost_classic"]
        exp_l = rec["exp_learned"] if rec["exp_learned"] != "" else "-"
        speed = rec["speedup_learned"] if rec["speedup_learned"] != "" else "-"
        subopt = rec["subopt_learned"] if rec["subopt_learned"] != "" else "-"

        lines.append(
            f"| {grid} | {inst} | {exp_c} | {cost_c:.1f} | {exp_l} | {speed} | {subopt} |"
        )

    OUT_MD.write_text("\n".join(lines))
    print(f"[INFO] Wrote hard instances summary to {OUT_MD}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
