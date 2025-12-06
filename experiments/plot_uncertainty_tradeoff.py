import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_CSV = BASE_DIR / "phd_experiments" / "astar_phd_results.csv"
PLOTS_DIR = BASE_DIR / "phd_experiments"


def load_results():
    data = []
    with RESULTS_CSV.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["instance_id"] = int(row["instance_id"])
            row["success"] = int(row["success"])
            row["expansions"] = int(row["expansions"])
            row["runtime_sec"] = float(row["runtime_sec"])
            row["path_cost"] = float(row["path_cost"])
            data.append(row)
    return data


def compute_c_over_cstar(data):
    """
    Επιστρέφει dict:
    per (grid_type, method) -> list of C/C*
    """
    by_inst_grid = defaultdict(dict)
    for row in data:
        if row["success"] != 1:
            continue
        key = (row["instance_id"], row["grid_type"])
        by_inst_grid[key][row["method"]] = row["path_cost"]

    ratios = defaultdict(list)
    for (inst, grid), costs in by_inst_grid.items():
        if "classic" not in costs:
            continue
        c_star = costs["classic"]
        for method, c in costs.items():
            if method == "classic":
                continue
            ratios[(grid, method)].append(c / c_star if c_star > 0 else 1.0)

    return ratios


def main():
    if not RESULTS_CSV.exists():
        print(f"[ERROR] No results at {RESULTS_CSV}")
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_results()
    ratios = compute_c_over_cstar(data)

    # μέσο expansions per (grid, method)
    exp_stats = defaultdict(list)
    for row in data:
        if row["success"] != 1:
            continue
        key = (row["grid_type"], row["method"])
        exp_stats[key].append(row["expansions"])

    exp_mean = {
        key: sum(v) / len(v) for key, v in exp_stats.items() if v
    }
    ratio_mean = {
        key: sum(v) / len(v) for key, v in ratios.items() if v
    }

    grid_types = sorted(set(row["grid_type"] for row in data))
    k_values = [0.0, 1.0, 2.0, 3.0]

    # Plot expansions vs k για κάθε grid type
    for grid in grid_types:
        ks_plot = []
        exp_plot = []
        c_over_plot = []

        for k in k_values:
            method = f"uncertainty_k{k:g}"
            key = (grid, method)
            if key in exp_mean:
                ks_plot.append(k)
                exp_plot.append(exp_mean[key])
                c_over_plot.append(ratio_mean.get(key, 1.0))

        if not ks_plot:
            continue

        # 1) Expansions vs k
        plt.figure()
        plt.plot(ks_plot, exp_plot, marker="o")
        plt.xlabel("k (LCB parameter)")
        plt.ylabel("Mean expansions")
        plt.title(f"Uncertainty A* expansions vs k ({grid})")
        plt.grid(True)
        out_path = PLOTS_DIR / f"uncertainty_expansions_{grid}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved {out_path}")

        # 2) C/C* vs k
        plt.figure()
        plt.plot(ks_plot, c_over_plot, marker="o")
        plt.xlabel("k (LCB parameter)")
        plt.ylabel("Mean C/C*")
        plt.title(f"Uncertainty A* path cost ratio vs k ({grid})")
        plt.grid(True)
        out_path = PLOTS_DIR / f"uncertainty_c_over_cstar_{grid}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved {out_path}")


if __name__ == "__main__":
    main()

