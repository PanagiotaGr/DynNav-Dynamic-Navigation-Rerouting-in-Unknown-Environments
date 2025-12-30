# ================================================================
# OOD / Drift Detection Evaluation using Uncertainty
#
# - Εκπαίδευση (online) στο 60% του dataset
# - Το υπόλοιπο 40% = in-distribution (ID) test set
# - Δημιουργία synthetic OOD δειγμάτων μακριά από το training distribution
# - Χρήση της predicted variance ως OOD score
# - Υπολογισμός ROC AUC για OOD vs ID
# - Αποθήκευση CSV + ROC plot + histograms
# ================================================================

import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from real_world_dataset_loader import RealWorldDatasetLoader
from online_drift_uncertainty_exp import OnlineDriftUncertaintyAdapterExp

FEATURE_COLS = ["entropy", "local_uncertainty", "speed"]
TARGET_COL = "drift"


def compute_roc_auc(scores: np.ndarray, labels: np.ndarray):
    """
    Υπολογίζει ROC curve + AUC χειροκίνητα (χωρίς sklearn).
    labels: 0 = ID, 1 = OOD
    scores: όσο μεγαλύτερο, τόσο πιο πιθανό OOD.
    """
    # sort κατά score φθίνουσα (ψηλότερο score = πιο OOD)
    order = np.argsort(-scores)
    scores_sorted = scores[order]
    labels_sorted = labels[order]

    P = float((labels == 1).sum())  # OOD positives
    N = float((labels == 0).sum())  # ID negatives

    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.5

    tprs = []
    fprs = []

    tp = 0.0
    fp = 0.0
    prev_score = None

    # Σαρώνουμε thresholds στα μοναδικά scores
    for s, y in zip(scores_sorted, labels_sorted):
        if prev_score is None or s != prev_score:
            tprs.append(tp / P)
            fprs.append(fp / N)
            prev_score = s
        if y == 1:
            tp += 1.0
        else:
            fp += 1.0

    # Τελευταίο σημείο (1,1)
    tprs.append(1.0)
    fprs.append(1.0)

    fprs = np.array(fprs, dtype=np.float32)
    tprs = np.array(tprs, dtype=np.float32)

    # AUC via trapezoidal rule
    # Προσοχή: πρέπει να είναι ταξινομημένα κατά FPR
    idx = np.argsort(fprs)
    fprs = fprs[idx]
    tprs = tprs[idx]
    auc = float(np.trapezoid(tprs, fprs))
    return fprs, tprs, auc


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "drift_dataset.csv")

    print(f"[OOD] Loading dataset from: {csv_path}")
    loader = RealWorldDatasetLoader(csv_path)
    X, y = loader.build_feature_matrix(FEATURE_COLS, TARGET_COL)

    N = len(X)
    N_train = int(0.6 * N)
    X_train, y_train = X[:N_train], y[:N_train]
    X_id, y_id = X[N_train:], y[N_train:]

    print(f"[OOD] N={N}, N_train={N_train}, N_id={len(X_id)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[OOD] Using device: {device}")

    adapter = OnlineDriftUncertaintyAdapterExp(
        model_path=None,
        input_dim=X.shape[1],
        lr=1e-4,
        device=device,
        max_buffer_size=512,
        weight_decay=1e-6,
        dropout_p=0.1,
    )

    # ================== Online Training ==================
    print("[OOD] Online training on training set...")

    for i in range(N_train):
        x_i = X_train[i]
        y_i = y_train[i]
        adapter.add_observation(x_i, np.array(y_i, dtype=np.float32))
        if (i + 1) % 16 == 0:
            _ = adapter.online_update(batch_size=64)

    print("[OOD] Training phase done.")

    # ================== Construct Synthetic OOD ==================
    # OOD: samples far from training distribution (per-feature)
    mu = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-6

    # Π.χ. 5 * std shift
    num_ood = len(X_id)
    X_ood = mu + 5.0 * std * np.random.randn(num_ood, X.shape[1]).astype(np.float32)

    print(f"[OOD] Generated {num_ood} synthetic OOD samples.")

    # ================== Compute Uncertainty Scores ==================
    print("[OOD] Computing variance-based scores for ID and OOD sets...")

    var_id_list = []
    var_ood_list = []

    for i in range(len(X_id)):
        mean_i, var_i = adapter.predict_with_uncertainty(X_id[i], n_samples=30)
        var_scalar = var_i.reshape(-1)[0]
        var_id_list.append(var_scalar)

    for i in range(len(X_ood)):
        mean_o, var_o = adapter.predict_with_uncertainty(X_ood[i], n_samples=30)
        var_scalar = var_o.reshape(-1)[0]
        var_ood_list.append(var_scalar)

    var_id = np.array(var_id_list, dtype=np.float32)
    var_ood = np.array(var_ood_list, dtype=np.float32)

    print(f"[OOD] ID variance:   mean={var_id.mean():.6f}, std={var_id.std():.6f}")
    print(f"[OOD] OOD variance:  mean={var_ood.mean():.6f}, std={var_ood.std():.6f}")

    # ================== ROC / AUC ==================
    scores = np.concatenate([var_id, var_ood], axis=0)
    labels = np.concatenate([np.zeros_like(var_id), np.ones_like(var_ood)], axis=0)

    fprs, tprs, auc = compute_roc_auc(scores, labels)
    print(f"[OOD] ROC AUC (variance as OOD score) = {auc:.4f}")

    # ================== Save results ==================
    out_dir = os.path.join(base_dir, "logs_ood")
    os.makedirs(out_dir, exist_ok=True)

    # Per-sample CSV
    df = pd.DataFrame({
        "score_variance": scores,
        "label": labels,  # 0=ID, 1=OOD
    })
    csv_path_out = os.path.join(out_dir, "ood_variance_scores.csv")
    df.to_csv(csv_path_out, index=False)

    # Summary CSV
    summary = {
        "id_var_mean": [float(var_id.mean())],
        "id_var_std": [float(var_id.std())],
        "ood_var_mean": [float(var_ood.mean())],
        "ood_var_std": [float(var_ood.std())],
        "auc": [float(auc)],
        "num_id": [int(len(var_id))],
        "num_ood": [int(len(var_ood))],
    }
    df_summary = pd.DataFrame(summary)
    summary_path = os.path.join(out_dir, "ood_variance_summary.csv")
    df_summary.to_csv(summary_path, index=False)

    print(f"[OOD] Saved per-sample scores to: {csv_path_out}")
    print(f"[OOD] Saved summary to:         {summary_path}")

    # ================== Plots ==================
    # ROC curve
    plt.figure(figsize=(6, 6))
    plt.plot(fprs, tprs, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", label="random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("OOD Detection ROC (variance as score)")
    plt.grid(True)
    plt.legend()
    roc_path = os.path.join(out_dir, "ood_roc_variance.png")
    plt.savefig(roc_path, dpi=200)
    plt.close()
    print(f"[OOD] Saved ROC plot to: {roc_path}")

    # Histograms ID vs OOD variance
    plt.figure(figsize=(8, 4))
    plt.hist(var_id, bins=20, alpha=0.6, label="ID")
    plt.hist(var_ood, bins=20, alpha=0.6, label="OOD")
    plt.xlabel("Predicted variance")
    plt.ylabel("Count")
    plt.title("ID vs OOD variance distribution")
    plt.legend()
    hist_path = os.path.join(out_dir, "ood_variance_hist.png")
    plt.savefig(hist_path, dpi=200)
    plt.close()
    print(f"[OOD] Saved variance histogram to: {hist_path}")

    print("\n[OOD] OOD detection experiment finished 🎯")


if __name__ == "__main__":
    main()
