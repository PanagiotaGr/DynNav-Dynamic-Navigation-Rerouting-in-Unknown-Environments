# ================================================================
# Online Drift Adaptation with Epistemic Uncertainty Monitoring
# ================================================================

import os
import numpy as np
import pandas as pd
import torch

from online_drift_uncertainty import OnlineDriftUncertaintyAdapter


FEATURE_COLS = ["entropy", "local_uncertainty", "speed"]
TARGET_COL = "drift"


def load_drift_dataset(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found")

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    return X, y


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    csv_path = os.path.join(base_dir, "drift_dataset.csv")
    model_path = os.path.join(base_dir, "drift_uncertainty_net.pt")

    print(f"[INFO] Loading dataset from: {csv_path}")
    X, y = load_drift_dataset(csv_path)
    print(f"[INFO] Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")

    N = X.shape[0]
    if N < 200:
        print("[WARN] Πολύ λίγα samples για ουσιαστικό online πείραμα.")

    N0 = int(0.4 * N)

    X_base = X[:N0]
    y_base = y[:N0]
    X_online = X[N0:]
    y_online = y[N0:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    adapter = OnlineDriftUncertaintyAdapter(
        model_path=model_path,
        input_dim=X.shape[1],
        lr=1e-4,
        device=device,
        max_buffer_size=512,
        weight_decay=1e-6,
        dropout_p=0.1,
    )

    # ================= Baseline =================
    print("[INFO] Evaluating baseline performance...")
    y_pred = adapter.predict(X_base).reshape(-1)
    base_mse = mse(y_pred, y_base)
    print(f"[RESULT] Baseline MSE: {base_mse:.6f}")

    # ================= Online Phase =================
    print("[INFO] Starting online adaptation...")

    mse_before = []
    mse_after = []
    var_before = []
    var_after = []

    window = 64

    for i in range(X_online.shape[0]):
        x_i = X_online[i]
        y_i = y_online[i]

        mean_b, var_b = adapter.predict_with_uncertainty(x_i, n_samples=20)
        pred_b = mean_b.reshape(-1)[0]

        adapter.add_observation(x_i, np.array(y_i, dtype=np.float32))

        if (i + 1) % 8 == 0:
            loss = adapter.online_update(batch_size=32)
        else:
            loss = 0.0

        mean_a, var_a = adapter.predict_with_uncertainty(x_i, n_samples=20)
        pred_a = mean_a.reshape(-1)[0]

        mse_before.append((pred_b - y_i) ** 2)
        mse_after.append((pred_a - y_i) ** 2)
        var_before.append(var_b.reshape(-1)[0])
        var_after.append(var_a.reshape(-1)[0])

        if (i + 1) % 100 == 0:
            mb = np.mean(mse_before[-window:])
            ma = np.mean(mse_after[-window:])
            print(
                f"[STEP {i+1}/{X_online.shape[0]}] "
                f"rolling MSE before={mb:.6f} | after={ma:.6f} | last loss={loss:.6f}"
            )

    print("\n================ SUMMARY ================")
    print(f"Overall MSE BEFORE: {float(np.mean(mse_before)):.6f}")
    print(f"Overall MSE AFTER : {float(np.mean(mse_after)):.6f}")
    print(f"Avg VAR BEFORE: {float(np.mean(var_before)):.6f}")
    print(f"Avg VAR AFTER : {float(np.mean(var_after)):.6f}")
    print("=========================================")

    out_model_path = os.path.join(base_dir, "drift_uncertainty_net_online_adapted.pt")
    adapter.save_model(out_model_path)
    print(f"[INFO] Saved adapted model: {out_model_path}")


if __name__ == "__main__":
    main()
