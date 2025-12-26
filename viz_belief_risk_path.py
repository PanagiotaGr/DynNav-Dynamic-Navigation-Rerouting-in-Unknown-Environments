import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_uncertainty_grid(csv_path: str) -> np.ndarray:
    """
    Φορτώνει το coverage_grid_with_uncertainty_pose.csv
    και επιστρέφει 2D grid [H, W] με uncertainty_fused.
    Περιμένει στήλες: row, col, uncertainty_fused.
    """
    df = pd.read_csv(csv_path)
    print("[INFO] Columns in CSV:", df.columns.tolist())

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


def load_path(path_csv: str):
    """
    Φορτώνει path από belief_risk_path_case2_pose.csv.
    Περιμένει δύο στήλες (row, col) χωρίς header.
    """
    arr = np.loadtxt(path_csv, delimiter=",", dtype=int)
    # Αν το path είναι ένα μόνο σημείο, arr.shape θα είναι (2,), κάνε το (1, 2)
    if arr.ndim == 1:
        arr = arr.reshape(1, 2)
    rows = arr[:, 0]
    cols = arr[:, 1]
    return rows, cols


def main():
    grid_csv = "coverage_grid_with_uncertainty_pose.csv"
    path_csv = "belief_risk_path_case2_pose.csv"

    print("[INFO] Loading grid...")
    grid = load_uncertainty_grid(grid_csv)
    H, W = grid.shape
    print(f"[INFO] Grid shape: {H} x {W}")

    print("[INFO] Loading path...")
    rows, cols = load_path(path_csv)
    print(f"[INFO] Path length: {len(rows)} cells")

    # Plot
    plt.figure(figsize=(6, 8))
    # imshow με origin='lower' για να το βλέπουμε "ανθρώπινα"
    im = plt.imshow(
        grid,
        origin="lower",
        interpolation="nearest",
    )
    plt.colorbar(im, label="uncertainty_fused")

    # Path overlay (προσοχή: x = col, y = row)
    plt.plot(cols, rows, marker="o", linewidth=2, markersize=3)

    plt.title("Risk-aware path on fused uncertainty grid")
    plt.xlabel("col")
    plt.ylabel("row")

    out_img = "belief_risk_path_case2_pose.png"
    plt.tight_layout()
    plt.savefig(out_img, dpi=200)
    print(f"[INFO] Saved figure to {out_img}")
    # Αν θες να το βλέπεις κατευθείαν:
    # plt.show()


if __name__ == "__main__":
    main()
