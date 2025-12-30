# ================================================================
# Experimental Comparison:
#   - Mode A: Χωρίς online adaptation
#   - Mode B: Με online adaptation
# ================================================================

import os
import numpy as np
import pandas as pd
import torch

from online_drift_uncertainty_exp import OnlineDriftUncertaintyAdapterExp

FEATURE_COLS = ["entropy", "local_uncertainty", "speed"]
TARGET_COL = "drift"


def load_drift_dataset(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}. Columns: {list(df.columns)}")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Columns: {list(df.columns)}")

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    return X, y


def mse(a, b) -> float:
    return float(np.mean((a - b) ** 2))


def run_mode(
    X: np.ndarray,
    y: np.ndarray,
    enable_online_update: bool,
    device: str,
    model_path: str,
    seed: int = 0,
):
    """
    Τρέχει ένα πείραμα:
      - αν enable_online_update == False: δεν γίνονται καθόλου online updates
      - αν enable_online_update == True : γίνονται updates κάθε 8 δείγματα
    Επιστρέφει metrics σε dict.
    """

    # Για να είναι δίκαιη η σύγκριση, σταθεροποιούμε τα seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    N = X.shape[0]
    N0 = int(0.4 * N)  # 40% baseline, 60% online segment

    X_base = X[:N0]
    y_base = y[:N0]
    X_online = X[N0:]
    y_online = y[N0:]

    # Δημιουργία adapter με την ίδια αρχικοποίηση
    adapter = OnlineDriftUncertaintyAdapterExp(
        model_path=None,          # για τη σύγκριση ξεκινάμε πάντα από scratch
        input_dim=X.shape[1],
        lr=1e-4,
        device=device,
        max_buffer_size=512,
        weight_decay=1e-6,
        dropout_p=0.1,
    )

    # ================= Baseline =================
    y_pred_base = adapter.predict(X_base).reshape(-1)
    base_mse = mse(y_pred_base, y_base)

    # ================= Online Segment =================
    mse_before = []
    mse_after = []
    var_before = []
    var_after = []

    window = 64

    for i in range(X_online.shape[0]):
        x_i = X_online[i]
        y_i = y_online[i]

        # Πρόβλεψη ΠΡΙΝ το update (με epistemic uncertainty)
        mean_b, var_b = adapter.predict_with_uncertainty(x_i, n_samples=20)
        pred_b = mean_b.reshape(-1)[0]
        var_b_scalar = var_b.reshape(-1)[0]

        adapter.add_observation(x_i, np.array(y_i, dtype=np.float32))

        # Online update λογική ανά mode
        if enable_online_update and (i + 1) % 8 == 0:
            loss = adapter.online_update(batch_size=32)
        else:
            loss = 0.0

        # Πρόβλεψη ΜΕΤΑ το update
        mean_a, var_a = adapter.predict_with_uncertainty(x_i, n_samples=20)
        pred_a = mean_a.reshape(-1)[0]
        var_a_scalar = var_a.reshape(-1)[0]

        mse_before.append((pred_b - y_i) ** 2)
        mse_after.append((pred_a - y_i) ** 2)
        var_before.append(var_b_scalar)
        var_after.append(var_a_scalar)

        # Μικρό log για να βλέπουμε ότι τρέχει
        if (i + 1) % 100 == 0:
            mb = float(np.mean(mse_before[-window:]))
            ma = float(np.mean(mse_after[-window:]))
            vb = float(np.mean(var_before[-window:]))
            va = float(np.mean(var_after[-window:]))
            mode_name = "ONLINE" if enable_online_update else "NO-UPDATE"
            print(
                f"[{mode_name} STEP {i+1}/{X_online.shape[0]}] "
                f"rolling MSE before={mb:.6f} | after={ma:.6f} | "
                f"VAR before={vb:.6f} | VAR after={va:.6f} | last loss={loss:.6f}"
            )

    metrics = {
        "enable_online_update": enable_online_update,
        "base_mse": base_mse,
        "online_mse_before": float(np.mean(mse_before)),
        "online_mse_after": float(np.mean(mse_after)),
        "online_var_before": float(np.mean(var_before)),
        "online_var_after": float(np.mean(var_after)),
    }

    return metrics


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "drift_dataset.csv")
    model_path_dummy = os.path.join(base_dir, "drift_uncertainty_net_exp_compare.pt")

    print(f"[CMP-INFO] Loading dataset from: {csv_path}")
    X, y = load_drift_dataset(csv_path)
    print(f"[CMP-INFO] Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")

    N = X.shape[0]
    if N < 200:
        print("[CMP-WARN] Πολύ λίγα samples για ουσιαστικό online πείραμα.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CMP-INFO] Using device: {device}")

    # -----------------------------
    # Mode A: Χωρίς online updates
    # -----------------------------
    print("\n[CMP-INFO] Running Mode A: NO ONLINE ADAPTATION")
    metrics_no_update = run_mode(
        X=X,
        y=y,
        enable_online_update=False,
        device=device,
        model_path=model_path_dummy,
        seed=0,
    )

    # -----------------------------
    # Mode B: Με online updates
    # -----------------------------
    print("\n[CMP-INFO] Running Mode B: WITH ONLINE ADAPTATION")
    metrics_online = run_mode(
        X=X,
        y=y,
        enable_online_update=True,
        device=device,
        model_path=model_path_dummy,
        seed=0,
    )

    # -----------------------------
    # Σύγκριση σε μορφή "πίνακα"
    # -----------------------------
    def fmt(b: bool) -> str:
        return "YES" if b else "NO"

    print("\n================ COMPARISON SUMMARY ================")
    print("Mode                 | Online Update | Baseline MSE | Online MSE (before) | Online MSE (after) | Avg VAR (before) | Avg VAR (after)")
    print("---------------------+--------------+--------------+----------------------+---------------------+------------------+-----------------")
    print(
        f"No-Update            | {fmt(metrics_no_update['enable_online_update']):>12} | "
        f"{metrics_no_update['base_mse']:.6f}   | "
        f"{metrics_no_update['online_mse_before']:.6f}          | "
        f"{metrics_no_update['online_mse_after']:.6f}         | "
        f"{metrics_no_update['online_var_before']:.6f}       | "
        f"{metrics_no_update['online_var_after']:.6f}"
    )
    print(
        f"With-Online-Adapt    | {fmt(metrics_online['enable_online_update']):>12} | "
        f"{metrics_online['base_mse']:.6f}   | "
        f"{metrics_online['online_mse_before']:.6f}          | "
        f"{metrics_online['online_mse_after']:.6f}         | "
        f"{metrics_online['online_var_before']:.6f}       | "
        f"{metrics_online['online_var_after']:.6f}"
    )
    print("====================================================")


if __name__ == "__main__":
    main()
