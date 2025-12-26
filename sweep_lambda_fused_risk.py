import numpy as np
import pandas as pd
import math

from belief_risk_planner import GridCell, astar_risk_aware


def load_fused_grid(csv_path: str) -> np.ndarray:
    """
    Φορτώνει grid [H, W] με uncertainty_fused
    από coverage_grid_with_uncertainty_pose.csv.

    Περιμένει στήλες: row, col, uncertainty_fused.
    """
    df = pd.read_csv(csv_path)
    required_cols = ["row", "col", "uncertainty_fused"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Λείπει η στήλη '{c}' από το {csv_path}.")

    max_row = int(df["row"].max())
    max_col = int(df["col"].max())
    grid = np.full((max_row + 1, max_col + 1), np.nan, dtype=float)

    for _, row in df.iterrows():
        i = int(row["row"])
        j = int(row["col"])
        grid[i, j] = float(row["uncertainty_fused"])

    return grid


def path_metrics_basic(rows: np.ndarray, cols: np.ndarray, grid_fused: np.ndarray):
    """
    Υπολογίζει basic metrics για path:
      - length (cells)
      - geometric length
      - fused_sum / fused_mean / fused_max
    """
    assert rows.shape == cols.shape
    n = len(rows)

    length_cells = n

    # Geometric length: sum sqrt(Δi^2 + Δj^2)
    geom_len = 0.0
    for k in range(1, n):
        di = rows[k] - rows[k - 1]
        dj = cols[k] - cols[k - 1]
        geom_len += math.sqrt(di * di + dj * dj)

    H, W = grid_fused.shape
    fused_vals = []
    for i, j in zip(rows, cols):
        if 0 <= i < H and 0 <= j < W:
            fused_vals.append(grid_fused[i, j])
        else:
            fused_vals.append(np.nan)

    fused_vals = np.array(fused_vals, dtype=float)
    valid = fused_vals[~np.isnan(fused_vals)]
    if valid.size == 0:
        fused_sum = np.nan
        fused_mean = np.nan
        fused_max = np.nan
    else:
        fused_sum = float(valid.sum())
        fused_mean = float(valid.mean())
        fused_max = float(valid.max())

    return {
        "length_cells": length_cells,
        "geometric_length": geom_len,
        "fused_sum": fused_sum,
        "fused_mean": fused_mean,
        "fused_max": fused_max,
    }


def main():
    csv_path = "coverage_grid_with_uncertainty_pose.csv"
    print("[INFO] Loading fused uncertainty grid...")
    grid_unc = load_fused_grid(csv_path)
    H, W = grid_unc.shape
    print(f"[INFO] Grid shape: {H} x {W}")

    start = GridCell(0, 0)
    goal = GridCell(H - 1, W - 1)

    lambda_values = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    rows_out = []

    for lam in lambda_values:
        print(f"\n[INFO] Running A* with λ = {lam} ...")
        path, total_cost = astar_risk_aware(grid_unc, start, goal, lambda_risk=lam)
        if path is None:
            print("[WARN] Δεν βρέθηκε διαδρομή.")
            row = {
                "lambda": lam,
                "found_path": False,
                "total_cost": np.nan,
                "length_cells": np.nan,
                "geometric_length": np.nan,
                "fused_sum": np.nan,
                "fused_mean": np.nan,
                "fused_max": np.nan,
            }
        else:
            coords = np.array([(c.i, c.j) for c in path], dtype=int)
            rows = coords[:, 0]
            cols = coords[:, 1]
            metrics = path_metrics_basic(rows, cols, grid_unc)
            print(f"[INFO] Path length (cells): {metrics['length_cells']}")
            print(f"[INFO] Geometric length: {metrics['geometric_length']:.3f}")
            print(f"[INFO] Fused sum: {metrics['fused_sum']:.3f}, mean: {metrics['fused_mean']:.3f}, max: {metrics['fused_max']:.3f}")
            print(f"[INFO] Total A* cost: {total_cost:.3f}")

            row = {
                "lambda": lam,
                "found_path": True,
                "total_cost": total_cost,
                **metrics,
            }

        rows_out.append(row)

    df_out = pd.DataFrame(rows_out)
    out_csv = "belief_risk_lambda_sweep.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\n[INFO] Saved lambda sweep metrics to {out_csv}")


if __name__ == "__main__":
    main()
