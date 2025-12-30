# ================================================================
# Experimental Self-Trust Analysis over Online Drift Adaptation
# ================================================================

import os
import numpy as np
import pandas as pd
import torch

from online_drift_uncertainty_exp import OnlineDriftUncertaintyAdapterExp
from self_awareness_controller import SelfAwarenessController, SelfTrustConfig, NavigationMode

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


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "drift_dataset.csv")

    print(f"[ST-INFO] Loading dataset from: {csv_path}")
    X, y = load_drift_dataset(csv_path)
    print(f"[ST-INFO] Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")

    N = X.shape[0]
    if N < 200:
        print("[ST-WARN] Πολύ λίγα samples για ουσιαστικό self-trust πείραμα.")

    N0 = int(0.4 * N)

    X_base = X[:N0]
    y_base = y[:N0]
    X_online = X[N0:]
    y_online = y[N0:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ST-INFO] Using device: {device}")

    # Για αυτό το πείραμα ξεκινάμε πάντα από scratch
    adapter = OnlineDriftUncertaintyAdapterExp(
        model_path=None,
        input_dim=X.shape[1],
        lr=1e-4,
        device=device,
        max_buffer_size=512,
        weight_decay=1e-6,
        dropout_p=0.1,
    )

    # ================= Baseline =================
    print("[ST-INFO] Computing baseline error & variance...")
    y_pred_base = adapter.predict(X_base).reshape(-1)
    base_mse = mse(y_pred_base, y_base)

    # Εκτίμηση baseline epistemic variance
    var_base_list = []
    for i in range(X_base.shape[0]):
        _, var_b = adapter.predict_with_uncertainty(X_base[i], n_samples=10)
        var_base_list.append(var_b.reshape(-1)[0])
    base_var = float(np.mean(var_base_list))

    print(f"[ST-INFO] Baseline MSE = {base_mse:.6f}, baseline VAR = {base_var:.6f}")

    # Διαμόρφωση Self-Trust controller
    cfg = SelfTrustConfig(
        base_mse=base_mse,
        base_var=base_var,
        alpha_err=1.0,
        alpha_var=1.0,
        normal_threshold=0.7,
        cautious_threshold=0.4,
    )
    controller = SelfAwarenessController(cfg)

    # ================= Online Phase =================
    print("[ST-INFO] Starting self-trust analysis in online phase...")

    mse_list = []
    var_list = []
    S_list = []
    mode_counts = {
        NavigationMode.NORMAL: 0,
        NavigationMode.CAUTIOUS: 0,
        NavigationMode.SAFE_STOP: 0,
    }

    # Μικρό online adaptation όπως πριν (κάθε 8 δείγματα)
    for i in range(X_online.shape[0]):
        x_i = X_online[i]
        y_i = y_online[i]

        mean_b, var_b = adapter.predict_with_uncertainty(x_i, n_samples=20)
        pred_b = mean_b.reshape(-1)[0]
        var_b_scalar = var_b.reshape(-1)[0]
        error_sq = (pred_b - y_i) ** 2

        # Self-Trust & mode
        S, mode = controller.evaluate_step(error_sq, var_b_scalar)

        mse_list.append(error_sq)
        var_list.append(var_b_scalar)
        S_list.append(S)
        mode_counts[mode] += 1

        # Online update κάθε 8 δείγματα
        adapter.add_observation(x_i, np.array(y_i, dtype=np.float32))
        if (i + 1) % 8 == 0:
            _ = adapter.online_update(batch_size=32)

        if (i + 1) % 100 == 0:
            print(
                f"[ST-STEP {i+1}/{X_online.shape[0]}] "
                f"error={error_sq:.6f} | var={var_b_scalar:.6f} | S={S:.3f} | mode={mode.name}"
            )

    mse_online = float(np.mean(mse_list))
    var_online = float(np.mean(var_list))
    S_avg = float(np.mean(S_list))

    total_online = X_online.shape[0]
    normal_ratio = mode_counts[NavigationMode.NORMAL] / total_online
    cautious_ratio = mode_counts[NavigationMode.CAUTIOUS] / total_online
    safe_stop_ratio = mode_counts[NavigationMode.SAFE_STOP] / total_online

    # ================= Summary =================
    print("\n================ SELF-TRUST SUMMARY ================")
    print(f"Baseline MSE: {base_mse:.6f}")
    print(f"Baseline VAR: {base_var:.6f}")
    print(f"Online MSE (mean): {mse_online:.6f}")
    print(f"Online VAR (mean): {var_online:.6f}")
    print(f"Average Self-Trust S: {S_avg:.3f}")
    print("\nMode occupancy (online segment):")
    print(f"  NORMAL    : {mode_counts[NavigationMode.NORMAL]} steps "
          f"({normal_ratio*100:.1f}%)")
    print(f"  CAUTIOUS  : {mode_counts[NavigationMode.CAUTIOUS]} steps "
          f"({cautious_ratio*100:.1f}%)")
    print(f"  SAFE_STOP : {mode_counts[NavigationMode.SAFE_STOP]} steps "
          f"({safe_stop_ratio*100:.1f}%)")
    print("====================================================")


if __name__ == "__main__":
    main()
