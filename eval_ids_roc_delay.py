import numpy as np
import matplotlib.pyplot as plt

from attack_aware_ukf import AttackAwareUKF, SensorConfig, f_constant_velocity, h_position


def simulate_one(seed: int, T: int = 250, attack_start: int = 80):
    """
    Returns:
      nis_vo: (T,) NIS sequence for VO
      y_true: (T,) binary labels (0 normal, 1 attack)
    """
    rng = np.random.RandomState(seed)

    n = 4
    dt = 0.1
    Q = np.diag([1e-4, 1e-4, 1e-3, 1e-3])

    ukf = AttackAwareUKF(n=n, f=f_constant_velocity, Q=Q)
    ukf.set_state(np.array([0, 0, 1, 0.5], float), np.diag([0.1, 0.1, 0.2, 0.2]))

    vo_cfg = SensorConfig(
        name="vo",
        R_base=np.diag([0.05**2, 0.05**2]),
        nis_p=0.99,            # used inside filter for trust; ROC we will do externally
        trust_decay=0.25,
        trust_recover=0.02,
        dropout_trust=0.06
    )
    wheel_cfg = SensorConfig(
        name="wheel",
        R_base=np.diag([0.10**2, 0.10**2]),
        nis_p=0.99,
        trust_decay=0.12,
        trust_recover=0.03,
        dropout_trust=0.06
    )

    ukf.add_sensor(vo_cfg, h_position)
    ukf.add_sensor(wheel_cfg, h_position)

    x_true = np.array([0, 0, 1, 0.5], float)

    nis_vo = np.zeros(T, float)
    y_true = np.zeros(T, int)

    for t in range(T):
        x_true = f_constant_velocity(x_true, dt)
        ukf.predict(dt)

        z_vo = x_true[:2] + rng.randn(2) * 0.05
        z_w  = x_true[:2] + rng.randn(2) * 0.10

        if t >= attack_start:
            # stealthy slowly-growing bias
            bias = np.array([0.002*(t-attack_start), -0.0015*(t-attack_start)])
            z_vo = z_vo + bias
            y_true[t] = 1

        tel_vo = ukf.update("vo", z_vo)
        _ = ukf.update("wheel", z_w)

        nis_vo[t] = tel_vo["nis"]

    return nis_vo, y_true


def roc_pr_from_scores(scores: np.ndarray, labels: np.ndarray, thresholds: np.ndarray):
    """
    Compute ROC + PR points for given thresholds.
    y_pred = 1 if score >= thr else 0
    """
    roc = []
    pr = []
    for thr in thresholds:
        y_pred = (scores >= thr).astype(int)

        TP = int(np.sum((y_pred == 1) & (labels == 1)))
        FP = int(np.sum((y_pred == 1) & (labels == 0)))
        TN = int(np.sum((y_pred == 0) & (labels == 0)))
        FN = int(np.sum((y_pred == 0) & (labels == 1)))

        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0

        precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
        recall = TPR

        roc.append([thr, FPR, TPR, TP, FP, TN, FN])
        pr.append([thr, precision, recall, TP, FP, TN, FN])

    return np.array(roc, float), np.array(pr, float)


def auc_trapz(x: np.ndarray, y: np.ndarray) -> float:
    # sort by x ascending for integration
    idx = np.argsort(x)
    xs = x[idx]
    ys = y[idx]
    return float(np.trapz(ys, xs))


def detection_delay(scores: np.ndarray, labels: np.ndarray, thr: float, attack_start: int):
    """
    For a single run:
      return delay steps from attack_start to first detection (score>=thr),
      or np.inf if never detected after attack_start.
    """
    det_idx = np.where(scores[attack_start:] >= thr)[0]
    if det_idx.size == 0:
        return np.inf
    return int(det_idx[0])  # steps after attack_start


def main():
    # --- experiment setup ---
    T = 250
    attack_start = 80
    seeds = list(range(30))

    # Collect all scores/labels for ROC/PR
    all_scores = []
    all_labels = []

    # Also keep per-run sequences for delay stats
    per_run_scores = []
    per_run_labels = []

    for s in seeds:
        nis_vo, y = simulate_one(seed=s, T=T, attack_start=attack_start)
        per_run_scores.append(nis_vo)
        per_run_labels.append(y)
        all_scores.append(nis_vo)
        all_labels.append(y)

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    # Threshold sweep: use quantiles of scores to cover range well
    qs = np.linspace(0.0, 1.0, 250)
    thresholds = np.unique(np.quantile(all_scores[~np.isnan(all_scores)], qs))
    thresholds = thresholds[np.isfinite(thresholds)]
    if thresholds.size < 10:
        thresholds = np.linspace(np.nanmin(all_scores), np.nanmax(all_scores), 200)

    roc, pr = roc_pr_from_scores(all_scores, all_labels, thresholds)

    # AUCs
    roc_auc = auc_trapz(roc[:, 1], roc[:, 2])          # AUC over FPR-TPR
    pr_auc = auc_trapz(pr[:, 2], pr[:, 1])             # AUC over recall-precision (rough)

    # Pick an operating point:
    # Example: choose threshold that gives FPR <= 1% and max TPR under that constraint.
    fpr_target = 0.01
    feasible = roc[roc[:, 1] <= fpr_target]
    if feasible.size > 0:
        best = feasible[np.argmax(feasible[:, 2])]
        thr_star = float(best[0])
        fpr_star = float(best[1])
        tpr_star = float(best[2])
    else:
        # fallback: maximize Youden J = TPR - FPR
        j = roc[:, 2] - roc[:, 1]
        thr_star = float(roc[np.argmax(j), 0])
        fpr_star = float(roc[np.argmax(j), 1])
        tpr_star = float(roc[np.argmax(j), 2])

    # Detection delay stats at chosen thr
    delays = []
    for nis_vo, y in zip(per_run_scores, per_run_labels):
        d = detection_delay(nis_vo, y, thr_star, attack_start)
        delays.append(d)
    delays = np.array(delays, float)

    # Summarize delay (ignore inf separately)
    detected = np.isfinite(delays)
    det_rate = float(np.mean(detected))
    if np.any(detected):
        delay_mean = float(np.mean(delays[detected]))
        delay_median = float(np.median(delays[detected]))
        delay_p90 = float(np.quantile(delays[detected], 0.90))
    else:
        delay_mean = delay_median = delay_p90 = np.inf

    # Save CSVs
    np.savetxt(
        "ids_roc.csv",
        roc,
        delimiter=",",
        header="thr,FPR,TPR,TP,FP,TN,FN",
        comments=""
    )
    np.savetxt(
        "ids_pr.csv",
        pr,
        delimiter=",",
        header="thr,precision,recall,TP,FP,TN,FN",
        comments=""
    )
    delay_rows = np.array([[thr_star, fpr_star, tpr_star, roc_auc, pr_auc, det_rate, delay_mean, delay_median, delay_p90]], float)
    np.savetxt(
        "ids_delay_stats.csv",
        delay_rows,
        delimiter=",",
        header="thr_star,FPR_star,TPR_star,ROC_AUC,PR_AUC,detect_rate,delay_mean_steps,delay_median_steps,delay_p90_steps",
        comments=""
    )

    # Plots
    plt.figure()
    plt.plot(roc[:, 1], roc[:, 2])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(f"ROC (AUC={roc_auc:.3f})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ids_roc.png", dpi=160)

    plt.figure()
    plt.plot(pr[:, 2], pr[:, 1])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall (AUC≈{pr_auc:.3f})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ids_pr.png", dpi=160)

    print("Saved:")
    print(" - ids_roc.csv, ids_pr.csv, ids_delay_stats.csv")
    print(" - ids_roc.png, ids_pr.png")
    print("")
    print("Chosen operating point:")
    print(f"  thr*={thr_star:.6g}  FPR*={fpr_star:.4f}  TPR*={tpr_star:.4f}")
    print(f"  detect_rate={det_rate:.3f}  delay_mean={delay_mean:.2f} steps  median={delay_median:.2f}  p90={delay_p90:.2f}")


if __name__ == "__main__":
    main()
