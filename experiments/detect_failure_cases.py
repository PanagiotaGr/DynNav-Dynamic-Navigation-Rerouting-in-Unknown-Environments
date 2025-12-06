import csv
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

ERROR_CSV = BASE / "phd_experiments" / "heuristic_error_data.csv"
ASTAR_CSV = BASE / "phd_experiments" / "astar_phd_results.csv"
OUT_CSV = BASE / "phd_experiments" / "failure_cases.csv"
OUT_MD  = BASE / "phd_experiments" / "failure_cases_summary.md"


def load_error_stats():
    """
    Επιστρέφει ανά instance:
    - mean signed error
    - mean abs error
    - over-estimation rate
    """
    per_inst = defaultdict(list)

    with ERROR_CSV.open("r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            inst = int(r["instance_id"])
            e = float(r["error"])
            ae = float(r["abs_error"])
            per_inst[inst].append((e, ae))

    stats = {}
    for inst, vals in per_inst.items():
        signed = [v[0] for v in vals]
        absvals = [v[1] for v in vals]
        over_rate = sum(1 for s in signed if s > 0) / len(vals)
        under_rate = sum(1 for s in signed if s < 0) / len(vals)
        stats[inst] = {
            "mean_signed": sum(signed) / len(signed),
            "mean_abs": sum(absvals) / len(absvals),
            "over_rate": over_rate,
            "under_rate": under_rate,
        }
    return stats


def load_astar_stats():
    """
    Διαβάζουμε το astar_phd_results.csv και για κάθε instance μαζεύουμε:
    - grid_type
    - classic expansions / cost (C*)
    - learned expansions / cost (C_learned)
    """
    per_inst = {}

    with ASTAR_CSV.open("r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            inst = int(r["instance_id"])
            grid = r["grid_type"]
            method = r["method"]
            success = int(r["success"])
            if success != 1:
                continue

            if inst not in per_inst:
                per_inst[inst] = {
                    "grid_type": grid,
                    "classic_cost": None,
                    "classic_exp": None,
                    "learned_cost": None,
                    "learned_exp": None,
                }

            if method == "classic":
                per_inst[inst]["classic_cost"] = float(r["path_cost"])
                per_inst[inst]["classic_exp"] = int(r["expansions"])
            elif method == "learned":
                per_inst[inst]["learned_cost"] = float(r["path_cost"])
                per_inst[inst]["learned_exp"] = int(r["expansions"])

    return per_inst


def main(
    TH_ABS_ERROR=5.0,
    TH_OVER_RATE=0.7,
    TH_SUBOPT=1.05,
):
    if not ERROR_CSV.exists():
        print(f"[ERROR] Missing {ERROR_CSV}")
        return
    if not ASTAR_CSV.exists():
        print(f"[ERROR] Missing {ASTAR_CSV}")
        return

    err_stats = load_error_stats()
    astar_stats = load_astar_stats()

    rows = []

    for inst, ainfo in astar_stats.items():
        if inst not in err_stats:
            continue

        einfo = err_stats[inst]

        grid_type = ainfo["grid_type"]
        c_star = ainfo["classic_cost"]
        c_learn = ainfo["learned_cost"]
        exp_star = ainfo["classic_exp"]
        exp_learn = ainfo["learned_exp"]

        if c_star is None or c_learn is None:
            continue

        subopt = c_learn / (c_star + 1e-6)
        mean_abs = einfo["mean_abs"]
        over_rate = einfo["over_rate"]

        # Κριτήριο failure
        fail_abs = mean_abs > TH_ABS_ERROR
        fail_over = over_rate > TH_OVER_RATE
        fail_subopt = subopt > TH_SUBOPT

        is_failure = fail_abs or fail_over or fail_subopt

        rows.append({
            "instance_id": inst,
            "grid_type": grid_type,
            "classic_expansions": exp_star,
            "classic_cost": c_star,
            "learned_expansions": exp_learn,
            "learned_cost": c_learn,
            "suboptimality": subopt,
            "mean_abs_error": mean_abs,
            "over_rate": over_rate,
            "fail_abs_error": int(fail_abs),
            "fail_over_rate": int(fail_over),
            "fail_subopt": int(fail_subopt),
            "is_failure": int(is_failure),
        })

    # γράφουμε CSV για όλα τα instances με αυτά τα stats
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "instance_id",
        "grid_type",
        "classic_expansions",
        "classic_cost",
        "learned_expansions",
        "learned_cost",
        "suboptimality",
        "mean_abs_error",
        "over_rate",
        "fail_abs_error",
        "fail_over_rate",
        "fail_subopt",
        "is_failure",
    ]

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # summary markdown
    total = len(rows)
    failures = [r for r in rows if r["is_failure"] == 1]

    lines = []
    lines.append("# Failure Cases Detection\n")
    lines.append(f"- Total instances (with stats): {total}")
    lines.append(f"- Failure cases (any criterion): {len(failures)}\n")
    lines.append("Thresholds used:\n")
    lines.append(f"- mean_abs_error > {TH_ABS_ERROR}")
    lines.append(f"- over_rate > {TH_OVER_RATE}")
    lines.append(f"- C_learned / C* > {TH_SUBOPT}\n")
    lines.append("## Top failure cases (sorted by suboptimality)\n")
    lines.append("| Instance | Grid | C_learned/C* | mean_abs_error | over_rate | classic_exp | learned_exp |")
    lines.append("|----------|------|--------------|----------------|-----------|-------------|-------------|")

    failures_sorted = sorted(failures, key=lambda r: r["suboptimality"], reverse=True)[:20]
    for r in failures_sorted:
        lines.append(
            f"| {r['instance_id']} | {r['grid_type']} | "
            f"{r['suboptimality']:.3f} | {r['mean_abs_error']:.2f} | {r['over_rate']:.2f} | "
            f"{r['classic_expansions']} | {r['learned_expansions']} |"
        )

    OUT_MD.write_text("\n".join(lines))
    print(f"[INFO] Wrote failure cases CSV to {OUT_CSV}")
    print(f"[INFO] Wrote summary MD to {OUT_MD}")


if __name__ == "__main__":
    main()
