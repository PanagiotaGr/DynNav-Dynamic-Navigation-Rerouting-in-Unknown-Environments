# ================================================================
# Trigger-Based Online Drift Adaptation - Fixed vs Triggered Updates
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
    device: str,
    mode: str,
    seed: int = 0,
):
    """
    Τρέχει ένα πείραμα online adaptation με δύο modes:
      - mode == "fixed"   : online update κάθε 8 δείγματα
      - mode == "trigger" : update μόνο όταν error/variance > thresholds

    Επιστρέφει metrics σε dict.
    """

    assert mode in ("fixed", "trigger"), "mode must be 'fixed' or 'trigger'"

    np.random.seed(seed)
    torch.manual_seed(seed)

    N = X.shape[0]
    N0 = int(0.4 * N)  # 40% baseline, 60% online

    X_base = X[:N0]
    y_base = y[:N0]
    X_online = X[N0:]
    y_online = y[N0:]

    # Ξεκινάμε ΠΑΝΤΑ από scratch για δίκαιη σύγκριση
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
    y_pred_base = adapter.predict(X_base).reshape(-1)
    base_mse = mse(y_pred_base, y_base)

    # Εκτίμηση "τυπικού" variance στο baseline για threshold
    var_base_list = []
    for i in range(X_base.shape[0]):
        mean_b, var_b = adapter.predict_with_uncertainty(X_base[i], n_samples=10)
        var_base_list.append(var_b.reshape(-1)[0])
    avg_var_base = float(np.mean(var_base_list))

    # Thresholds για trigger mode
    # - error threshold = baseline MSE
    # - variance threshold = baseline mean variance
    tau_err = base_mse
    tau_var = avg_var_base

    print(
        f"[TRG-INFO] Mode={mode} | base_mse={base_mse:.6f} | "
        f"tau_err={tau_err:.6f} | tau_var={tau_var:.6f}"
    )

    mse_before = []
    mse_after = []
    var_before = []
    var_after = []

    window = 64
    update_count = 0
    trigger_err_count = 0
    trigger_var_count = 0

    for i in range(X_online.shape[0]):
        x_i = X_online[i]
        y_i = y_online[i]

        # Πρόβλεψη ΠΡΙΝ το update
        mean_b, var_b = adapter.predict_with_uncertainty(x_i, n_samples=20)
        pred_b = mean_b.reshape(-1)[0]
        var_b_scalar = var_b.reshape(-1)[0]
        error_now = (pred_b - y_i) ** 2

        adapter.add_observation(x_i, np.array(y_i, dtype=np.float32))

        # --------------------------------------
        # Fixed vs Trigger update policy
        # --------------------------------------
        do_update = False

        if mode == "fixed":
            # update κάθε 8 δείγματα
            if (i + 1) % 8 == 0:
                do_update = True

        elif mode == "trigger":
            # update όταν error ή variance είναι υψηλά
            cond_err = error_now > tau_err
            cond_var = var_b_scalar > tau_var

            if cond_err or cond_var:
                do_update = True
                if cond_err:
                    trigger_err_count += 1
                if cond_var:
                    trigger_var_count += 1

        if do_update:
            loss = adapter.online_update(batch_size=32)
            update_count += 1
        else:
            loss = 0.0

        # Πρόβλεψη ΜΕΤΑ το update
        mean_a, var_a = adapter.predict_with_uncertainty(x_i, n_samples=20)
        pred_a = mean_a.reshape(-1)[0]
        var_a_scalar = var_a.reshape(-1)[0]

        mse_before.append(error_now)
        mse_after.append((pred_a - y_i) ** 2)
        var_before.append(var_b_scalar)
        var_after.append(var_a_scalar)

        if (i + 1) % 100 == 0:
            mb = float(np.mean(mse_before[-window:]))
            ma = float(np.mean(mse_after[-window:]))
            vb = float(np.mean(var_before[-window:]))
            va = float(np.mean(var_after[-window:]))
            print(
                f"[TRG-STEP {mode} {i+1}/{X_online.shape[0]}] "
                f"rolling MSE before={mb:.6f} | after={ma:.6f} | "
                f"VAR before={vb:.6f} | VAR after={va:.6f} | "
                f"last loss={loss:.6f}"
            )

    metrics = {
        "mode": mode,
        "base_mse": base_mse,
        "tau_err": tau_err,
        "tau_var": tau_var,
        "online_mse_before": float(np.mean(mse_before)),
        "online_mse_after": float(np.mean(mse_after)),
        "online_var_before": float(np.mean(var_before)),
        "online_var_after": float(np.mean(var_after)),
        "update_count": update_count,
        "trigger_err_count": trigger_err_count,
        "trigger_var_count": trigger_var_count,
        "num_online_samples": int(X_online.shape[0]),
    }

    return metrics


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "drift_dataset.csv")

    print(f"[TRG-INFO] Loading dataset from: {csv_path}")
    X, y = load_drift_dataset(csv_path)
    print(f"[TRG-INFO] Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")

    if X.shape[0] < 200:
        print("[TRG-WARN] Πολύ λίγα samples για ουσιαστικό online πείραμα.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TRG-INFO] Using device: {device}")

    # Mode A: fixed schedule updates
    print("\n[TRG-INFO] Running Mode A: FIXED UPDATES (every 8 samples)")
    metrics_fixed = run_mode(
        X=X,
        y=y,
        device=device,
        mode="fixed",
        seed=0,
    )

    # Mode B: trigger-based updates
    print("\n[TRG-INFO] Running Mode B: TRIGGER-BASED UPDATES")
    metrics_trigger = run_mode(
        X=X,
        y=y,
        device=device,
        mode="trigger",
        seed=0,
    )

    # Summary table
    print("\n================ TRIGGER COMPARISON SUMMARY ================")
    print(
        "Mode     | Base MSE | Online MSE (before) | Online MSE (after) | "
        "Avg VAR (before) | Avg VAR (after) | Updates | Trig-by-Err | Trig-by-Var"
    )
    print(
        "---------+----------+---------------------+---------------------+"
        "------------------+-----------------+---------+------------+-----------"
    )

    def row(m):
        return (
            f"{m['mode']:^8} | "
            f"{m['base_mse']:.6f} | "
            f"{m['online_mse_before']:.6f}          | "
            f"{m['online_mse_after']:.6f}         | "
            f"{m['online_var_before']:.6f}       | "
            f"{m['online_var_after']:.6f}      | "
            f"{m['update_count']:7d} | "
            f"{m['trigger_err_count']:10d} | "
            f"{m['trigger_var_count']:9d}"
        )

    print(row(metrics_fixed))
    print(row(metrics_trigger))
    print("============================================================")


if __name__ == "__main__":
    main()
