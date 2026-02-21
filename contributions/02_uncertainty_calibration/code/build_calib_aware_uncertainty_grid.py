import numpy as np
import pandas as pd
import torch
from neural_uncertainty.calibration_methods import AffineCalibrator

from train_drift_uncertainty_net import (
    CSV_PATH,
    MODEL_PATH,
    NORM_STATS_PATH,
    UncertaintyMLP,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Πόσο επιθετικά διορθώνουμε με βάση το calibration error
ALPHA = 1.0
N_CLUSTERS = 4  # μπορείς να γίνει  αλλαγή π.χ. σε 3,5,...


def load_model_and_stats():
    """
    Φορτώνει το εκπαιδευμένο μοντέλο drift-uncertainty
    και τα normalization statistics (x_mean, x_std, y_mean, y_std).
    """
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    input_dim = ckpt["input_dim"]
    feature_cols = ckpt["feature_cols"]
    target_col = ckpt["target_col"]

    model = UncertaintyMLP(input_dim=input_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    stats = np.load(NORM_STATS_PATH)
    x_mean = stats["x_mean"]
    x_std = stats["x_std"]
    y_mean = stats["y_mean"]
    y_std = stats["y_std"]

    return model, feature_cols, target_col, x_mean, x_std, y_mean, y_std


def main():
    print("[INFO] Loading model and stats...")
    model, feature_cols, target_col, x_mean, x_std, y_mean, y_std = load_model_and_stats()

    print("[INFO] Loading drift dataset:", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    num_df = df.select_dtypes(include=[np.number])

    # Features & target
    for c in feature_cols:
        if c not in num_df.columns:
            raise ValueError(f"Feature column '{c}' λείπει από το drift_dataset.csv.")

    if target_col not in num_df.columns:
        raise ValueError(f"Target column '{target_col}' λείπει από το drift_dataset.csv.")

    X = num_df[feature_cols].to_numpy(dtype=np.float32)
    y = num_df[target_col].to_numpy(dtype=np.float32).reshape(-1, 1)

    n = len(num_df)
    print(f"[INFO] Dataset size: {n}")

    # normalize inputs
    Xn = (X - x_mean) / x_std

    print("[INFO] Running model to predict mean and sigma...")
    with torch.no_grad():
        xb = torch.from_numpy(Xn).to(DEVICE)
        mean_n, log_std = model(xb)
        mean_n = mean_n.cpu().numpy()
        log_std = log_std.cpu().numpy()

    # unnormalize σε original scale
    mean_orig = mean_n * y_std + y_mean
    std_norm = np.exp(log_std)
    std_orig = std_norm * y_std  # [N, 1]

    err = y - mean_orig
    abs_err = np.abs(err)
    sigma = std_orig + 1e-8  # avoid divide-by-zero
    z = abs_err / sigma          # [N, 1]
    z_flat = z.reshape(-1)

    print("[INFO] Overall mean z =", float(z_flat.mean()))

    # -----------------------------
    # Clustering στο feature space
    # -----------------------------
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError(
            "Χρειάζεται scikit-learn. Τρέξε:\n"
            "  pip install scikit-learn\n"
            "και ξανατρέξε αυτό το script."
        )

    print(f"[INFO] Clustering in feature space into {N_CLUSTERS} clusters...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init=10)
    clusters = kmeans.fit_predict(Xn)  # [N]

    # -----------------------------
    # Calibration per cluster
    # -----------------------------
    cluster_mean_z = []
    cluster_factor = []

    for k in range(N_CLUSTERS):
        mask = clusters == k
        if mask.sum() == 0:
            mean_z_k = np.nan
            factor_k = 1.0
        else:
            mean_z_k = float(z_flat[mask].mean())
            # factor = 1 + ALPHA * (mean_z - 1)
            # αν mean_z > 1 => over-confident => αυξάνουμε risk
            # αν mean_z < 1 => under-confident => μειώνουμε risk (προαιρετικό)
            factor_k = 1.0 + ALPHA * (mean_z_k - 1.0)
            # clip για να μην ξεφύγουμε
            factor_k = float(np.clip(factor_k, 0.5, 2.0))

        cluster_mean_z.append(mean_z_k)
        cluster_factor.append(factor_k)
        print(f"[INFO] Cluster {k}: mean z = {mean_z_k:.3f}, factor = {factor_k:.3f}")

    cluster_mean_z = np.array(cluster_mean_z, dtype=float)
    cluster_factor = np.array(cluster_factor, dtype=float)

    # -----------------------------
    # Apply calibration-aware factor
    # -----------------------------
    factors = cluster_factor[clusters]  # [N]
    sigma_calib = (sigma.reshape(-1) * factors).reshape(-1, 1)

    print("[INFO] mean(sigma) before:", float(sigma.mean()))
    print("[INFO] mean(sigma_calib) after:", float(sigma_calib.mean()))

    # -----------------------------
    # Build 1D grid (H = n, W = 1)
    # row = index, col = 0
    # -----------------------------
    H = n
    W = 1

    rows = np.arange(H, dtype=int)
    cols = np.zeros(H, dtype=int)

    grid_calib = np.full((H, W), np.nan, dtype=float)

    for r, c, s in zip(rows, cols, sigma_calib.reshape(-1)):
        if 0 <= r < H and 0 <= c < W:
            grid_calib[r, c] = float(s)

    # Flatten σε CSV
    out_rows = []
    for r in range(H):
        for c in range(W):
            s = grid_calib[r, c]
            out_rows.append(
                {
                    "row": r,
                    "col": c,
                    "calib_sigma": s,
                }
            )

    out_df = pd.DataFrame(out_rows)
    out_csv = "calib_learned_uncertainty_grid.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved calibration-aware uncertainty grid to {out_csv}")
    print("[INFO] Grid shape:", H, "x", W)


if __name__ == "__main__":
    main()
