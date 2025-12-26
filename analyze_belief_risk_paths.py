import numpy as np
import pandas as pd
import math


def load_grids(csv_path: str):
    """
    Φορτώνει από coverage_grid_with_uncertainty_pose.csv τα πεδία:
      - uncertainty_fused
      - pose_uncertainty_norm
    και τα βάζει σε 2D grids [H, W].
    """
    df = pd.read_csv(csv_path)
    required = ["row", "col", "uncertainty_fused", "pose_uncertainty_norm"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Λείπει η στήλη '{c}' από το {csv_path}.")

    max_row = int(df["row"].max())
    max_col = int(df["col"].max())
    fused = np.full((max_row + 1, max_col + 1), np.nan, dtype=float)
    pose = np.full((max_row + 1, max_col + 1), np.nan, dtype=float)

    for _, row in df.iterrows():
        i = int(row["row"])
        j = int(row["col"])
        fused[i, j] = float(row["uncertainty_fused"])
        pose[i, j] = float(row["pose_uncertainty_norm"])

    return fused, pose


def load_path(path_csv: str):
    """
    Φορτώνει path από CSV χωρίς header:
      col0 = row, col1 = col
    """
    arr = np.loadtxt(path_csv, delimiter=",", dtype=int)
    if arr.ndim == 1:
        arr = arr.reshape(1, 2)
    rows = arr[:, 0]
    cols = arr[:, 1]
    return rows, cols


def path_metrics(
    rows: np.ndarray,
    cols: np.ndarray,
    grid_fused: np.ndarray,
    grid_pose: np.ndarray,
):
    """
    Υπολογίζει διάφορα metrics για το path:
      - length (cells)
      - geometric_length (άθροισμα Euclidean βημάτων)
      - sum/mean/max fused
      - sum/mean/max pose_only
    """
    assert rows.shape == cols.shape
    n = len(rows)

    # path length σε cells
    length_cells = n

    # geometric length: sum over steps sqrt(Δi^2 + Δj^2)
    geom_len = 0.0
    for k in range(1, n):
        di = rows[k] - rows[k - 1]
        dj = cols[k] - cols[k - 1]
        geom_len += math.sqrt(di * di + dj * dj)

    fused_vals = []
    pose_vals = []

    H, W = grid_fused.shape
    for i, j in zip(rows, cols):
        if 0 <= i < H and 0 <= j < W:
            fused_vals.append(grid_fused[i, j])
            pose_vals.append(grid_pose[i, j])
        else:
            fused_vals.append(np.nan)
            pose_vals.append(np.nan)

    fused_vals = np.array(fused_vals, dtype=float)
    pose_vals = np.array(pose_vals, dtype=float)

    # αγνοούμε NaNs στον υπολογισμό
    def stats(arr):
        arr_valid = arr[~np.isnan(arr)]
        if arr_valid.size == 0:
            return np.nan, np.nan, np.nan
        return float(arr_valid.sum()), float(arr_valid.mean()), float(arr_valid.max())

    fused_sum, fused_mean, fused_max = stats(fused_vals)
    pose_sum, pose_mean, pose_max = stats(pose_vals)

    metrics = {
        "length_cells": length_cells,
        "geometric_length": geom_len,
        "fused_sum": fused_sum,
        "fused_mean": fused_mean,
        "fused_max": fused_max,
        "pose_sum": pose_sum,
        "pose_mean": pose_mean,
        "pose_max": pose_max,
    }
    return metrics


def main():
    grid_csv = "coverage_grid_with_uncertainty_pose.csv"
    path_fused_csv = "belief_risk_path_case2_pose.csv"
    path_pose_csv = "belief_risk_path_case2_pose_only.csv"

    print("[INFO] Loading grids...")
    fused, pose = load_grids(grid_csv)

    print("[INFO] Loading fused path...")
    rows_fused, cols_fused = load_path(path_fused_csv)
    print("[INFO] Loading pose-only path...")
    rows_pose, cols_pose = load_path(path_pose_csv)

    print("[INFO] Computing metrics for fused path...")
    fused_metrics = path_metrics(rows_fused, cols_fused, fused, pose)
    print("[INFO] Computing metrics for pose-only path...")
    pose_metrics_dict = path_metrics(rows_pose, cols_pose, fused, pose)

    # ωραία εκτύπωση
    def pretty_print(name, m):
        print(f"\n=== {name} ===")
        print(f"Length (cells): {m['length_cells']:.0f}")
        print(f"Geometric length: {m['geometric_length']:.3f}")
        print(f"Fused uncertainty: sum={m['fused_sum']:.3f}, mean={m['fused_mean']:.3f}, max={m['fused_max']:.3f}")
        print(f"Pose uncertainty:  sum={m['pose_sum']:.3f}, mean={m['pose_mean']:.3f}, max={m['pose_max']:.3f}")

    pretty_print("FUSED-RISK PATH (λ on uncertainty_fused)", fused_metrics)
    pretty_print("POSE-RISK PATH (λ on pose_uncertainty_norm)", pose_metrics_dict)

    # γράφουμε και σε CSV για report
    rows = []
    fused_metrics_out = {"path_type": "fused_path", **fused_metrics}
    pose_metrics_out = {"path_type": "pose_only_path", **pose_metrics_dict}
    rows.append(fused_metrics_out)
    rows.append(pose_metrics_out)

    df_out = pd.DataFrame(rows)
    out_csv = "belief_risk_path_metrics.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\n[INFO] Saved metrics to {out_csv}")


if __name__ == "__main__":
    main()
