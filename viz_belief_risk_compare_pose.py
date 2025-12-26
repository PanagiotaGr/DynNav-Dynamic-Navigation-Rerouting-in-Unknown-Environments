import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_grids(csv_path: str):
    """
    Φορτώνει:
      - fused uncertainty (uncertainty_fused)
      - pose-only uncertainty (pose_uncertainty_norm)
    σε δύο grids [H, W].
    """
    df = pd.read_csv(csv_path)
    print("[INFO] Columns in CSV:", df.columns.tolist())

    required_cols = ["row", "col", "uncertainty_fused", "pose_uncertainty_norm"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Λείπει η στήλη '{c}' από το {csv_path}.")

    max_row = int(df["row"].max())
    max_col = int(df["col"].max())
    fused = np.full((max_row + 1, max_col + 1), np.nan, dtype=float)
    pose_only = np.full((max_row + 1, max_col + 1), np.nan, dtype=float)

    for _, row in df.iterrows():
        i = int(row["row"])
        j = int(row["col"])
        fused[i, j] = float(row["uncertainty_fused"])
        pose_only[i, j] = float(row["pose_uncertainty_norm"])

    return fused, pose_only


def load_path(path_csv: str):
    """
    Φορτώνει path από CSV (2 στήλες: row, col, χωρίς header).
    """
    arr = np.loadtxt(path_csv, delimiter=",", dtype=int)
    if arr.ndim == 1:
        arr = arr.reshape(1, 2)
    rows = arr[:, 0]
    cols = arr[:, 1]
    return rows, cols


def main():
    grid_csv = "coverage_grid_with_uncertainty_pose.csv"
    path_fused_csv = "belief_risk_path_case2_pose.csv"
    path_pose_csv = "belief_risk_path_case2_pose_only.csv"

    print("[INFO] Loading grids...")
    fused, pose_only = load_grids(grid_csv)
    H, W = fused.shape
    print(f"[INFO] Grid shape: {H} x {W}")

    print("[INFO] Loading paths...")
    rows_fused, cols_fused = load_path(path_fused_csv)
    rows_pose, cols_pose = load_path(path_pose_csv)
    print(f"[INFO] Path fused length: {len(rows_fused)}")
    print(f"[INFO] Path pose-only length: {len(rows_pose)}")

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    # Left: fused
    im0 = axs[0].imshow(
        fused,
        origin="lower",
        interpolation="nearest",
    )
    axs[0].plot(cols_fused, rows_fused, marker="o", linewidth=2, markersize=3)
    axs[0].set_title("Risk-aware path on fused uncertainty")
    axs[0].set_xlabel("col")
    axs[0].set_ylabel("row")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04, label="uncertainty_fused")

    # Right: pose-only
    im1 = axs[1].imshow(
        pose_only,
        origin="lower",
        interpolation="nearest",
    )
    axs[1].plot(cols_pose, rows_pose, marker="o", linewidth=2, markersize=3)
    axs[1].set_title("Risk-aware path on pose-only uncertainty")
    axs[1].set_xlabel("col")
    axs[1].set_ylabel("row")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04, label="pose_uncertainty_norm")

    plt.tight_layout()
    out_img = "belief_risk_compare_fused_vs_pose.png"
    plt.savefig(out_img, dpi=200)
    print(f"[INFO] Saved figure to {out_img}")
    # Αν θες να το δεις live:
    # plt.show()


if __name__ == "__main__":
    main()
