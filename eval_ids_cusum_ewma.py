import numpy as np
import matplotlib.pyplot as plt
from attack_aware_ukf import AttackAwareUKF, SensorConfig, f_constant_velocity, h_position


def simulate_nis(seed: int, T: int = 250, attack_start: int = 80):
    rng = np.random.RandomState(seed)

    n = 4
    dt = 0.1
    Q = np.diag([1e-4, 1e-4, 1e-3, 1e-3])

    ukf = AttackAwareUKF(n=n, f=f_constant_velocity, Q=Q)
    ukf.set_state(np.array([0, 0, 1, 0.5], float), np.diag([0.1, 0.1, 0.2, 0.2]))

    vo_cfg = SensorConfig(
        name="vo",
        R_base=np.diag([0.05**2, 0.05**2]),
        nis_p=0.99,
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

    nis = np.zeros(T, float)
    y = np.zeros(T, int)

    for t in range(T):
        x_true = f_constant_velocity(x_true, dt)
        ukf.predict(dt)

        z_vo = x_true[:2] + rng.randn(2) * 0.05
        z_w  = x_true[:2] + rng.randn(2) * 0.10

        if t >= attack_start:
            bias = np.array([0.002*(t-attack_start), -0.0015*(t-attack_start)])
            z_vo = z_vo + bias
            y[t] = 1

        tel_vo = ukf.update("vo", z_vo)
        _ = ukf.update("wheel", z_w)

        nis[t] = tel_vo["nis"]

    return nis, y


def ewma_score(x: np.ndarray, lam: float = 0.05):
    """
    EWMA on NIS.
    s_t = lam*x_t + (1-lam)*s_{t-1}
    """
    s = np.zeros_like(x, dtype=float)
    s[0] = x[0]
    for t in range(1, len(x)):
        s[t] = lam * x[t] + (1.0 - lam) * s[t-1]
    return s


def cusum_score(x: np.ndarray, k: float = 0.2):
    """
    One-sided CUSUM on NIS:
      g_t = max(0, g_{t-1} + (x_t - k))
    If x_t has a persistent mean shift upward, g grows quickly.

    k acts like reference/drift (bigger k -> less sensitive).
    """
    g = np.zeros_like(x, dtype=float)
    for t in range(1, len(x)):
        g[t] = max(0.0, g[t-1] + (x[t] - k))
    return g


def roc_pr(scores: np.ndarray, labels: np.ndarray, thresholds: np.ndarray):
    roc_rows = []
    pr_rows = []
    for thr in thresholds:
        y_pred = (scores >= thr).astype(int)
        TP = int(np.sum((y_pred == 1) & (labels == 1)))
        FP = int(np.sum((y_pred == 1) & (labels == 0)))
        TN = int(np.sum((y_pred == 0) & (labels == 0)))
        FN = int(np.sum((y_pred == 0) & (labels == 1)))

        TPR = TP / (TP + FN) if (TP + FN) else 0.0
        FPR = FP / (FP + TN) if (FP + TN) else 0.0
        precision = TP / (TP + FP) if (TP + FP) else 1.0
        recall = TPR

        roc_rows.append([thr, FPR, TPR, TP, FP, TN, FN])
        pr_rows.append([thr, precision, recall, TP, FP, TN, FN])

    return np.array(roc_rows, float), np.array(pr_rows, float)


def auc_trapz(x, y):
    idx = np.argsort(x)
    return float(np.trapz(y[idx], x[idx]))


def pick_thr_by_fpr(roc_arr: np.ndarray, fpr_target: float = 0.01):
    feasible = roc_arr[roc_arr[:, 1] <= fpr_target]
    if feasible.size:
        best = feasible[np.argmax(feasible[:, 2])]
        return float(best[0]), float(best[1]), float(best[2])
    # fallback: Youden J
    j = roc_arr[:, 2] - roc_arr[:, 1]
    i = int(np.argmax(j))
    return float(roc_arr[i, 0]), float(roc_arr[i, 1]), float(roc_arr[i, 2])


def delay_for_run(scores: np.ndarray, thr: float, attack_start: int):
    idx = np.where(scores[attack_start:] >= thr)[0]
    if idx.size == 0:
        return np.inf
    return int(idx[0])


def summarize_delays(delays: np.ndarray):
    detected = np.isfinite(delays)
    det_rate = float(np.mean(detected))
    if np.any(detected):
        mean = float(np.mean(delays[detected]))
        med = float(np.median(delays[detected]))
        p90 = float(np.quantile(delays[detected], 0.90))
    else:
        mean = med = p90 = np.inf
    return det_rate, mean, med, p90


def main():
    T = 250
    attack_start = 80
    seeds = list(range(30))

    # Collect
    nis_all = []
    y_all = []
    nis_runs = []
    y_runs = []

    for s in seeds:
        nis, y = simulate_nis(s, T=T, attack_start=attack_start)
        nis_runs.append(nis)
        y_runs.append(y)
        nis_all.append(nis)
        y_all.append(y)

    nis_all = np.concatenate(nis_all)
    y_all = np.concatenate(y_all)

    # Build EWMA and CUSUM sequences per run then concat
    ew_all = []
    cu_all = []
    for nis in nis_runs:
        ew_all.append(ewma_score(nis, lam=0.05))
        cu_all.append(cusum_score(nis, k=1.5))  # k tuned for NIS scale (we’ll tune later)
    ew_all = np.concatenate(ew_all)
    cu_all = np.concatenate(cu_all)

    # Threshold grids (quantiles)
    def thr_grid(x):
        qs = np.linspace(0, 1, 300)
        th = np.unique(np.quantile(x[np.isfinite(x)], qs))
        th = th[np.isfinite(th)]
        if th.size < 50:
            th = np.linspace(np.nanmin(x), np.nanmax(x), 300)
        return th

    thr_nis = thr_grid(nis_all)
    thr_ew  = thr_grid(ew_all)
    thr_cu  = thr_grid(cu_all)

    # ROC/PR
    roc_nis, pr_nis = roc_pr(nis_all, y_all, thr_nis)
    roc_ew,  pr_ew  = roc_pr(ew_all,  y_all, thr_ew)
    roc_cu,  pr_cu  = roc_pr(cu_all,  y_all, thr_cu)

    # AUCs
    auc_nis = auc_trapz(roc_nis[:, 1], roc_nis[:, 2])
    auc_ew  = auc_trapz(roc_ew[:, 1],  roc_ew[:, 2])
    auc_cu  = auc_trapz(roc_cu[:, 1],  roc_cu[:, 2])

    # Pick operating point at FPR<=1%
    thrN, fprN, tprN = pick_thr_by_fpr(roc_nis, 0.01)
    thrE, fprE, tprE = pick_thr_by_fpr(roc_ew,  0.01)
    thrC, fprC, tprC = pick_thr_by_fpr(roc_cu,  0.01)

    # Delays per run
    delays_nis = []
    delays_ew = []
    delays_cu = []

    for nis, y in zip(nis_runs, y_runs):
        ew = ewma_score(nis, lam=0.05)
        cu = cusum_score(nis, k=1.5)
        delays_nis.append(delay_for_run(nis, thrN, attack_start))
        delays_ew.append(delay_for_run(ew,  thrE, attack_start))
        delays_cu.append(delay_for_run(cu,  thrC, attack_start))

    delays_nis = np.array(delays_nis, float)
    delays_ew  = np.array(delays_ew,  float)
    delays_cu  = np.array(delays_cu,  float)

    detN, meanN, medN, p90N = summarize_delays(delays_nis)
    detE, meanE, medE, p90E = summarize_delays(delays_ew)
    detC, meanC, medC, p90C = summarize_delays(delays_cu)

    # Save CSV summaries
    rows = np.array([
        ["raw_nis", thrN, fprN, tprN, auc_nis, detN, meanN, medN, p90N],
        ["ewma",    thrE, fprE, tprE, auc_ew,  detE, meanE, medE, p90E],
        ["cusum",   thrC, fprC, tprC, auc_cu,  detC, meanC, medC, p90C],
    ], dtype=object)

    # Write manually to preserve method column
    with open("ids_methods_summary.csv", "w") as f:
        f.write("method,thr_star,FPR_star,TPR_star,ROC_AUC,detect_rate,delay_mean_steps,delay_median_steps,delay_p90_steps\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    np.savetxt("ids_roc_raw.csv", roc_nis, delimiter=",", header="thr,FPR,TPR,TP,FP,TN,FN", comments="")
    np.savetxt("ids_roc_ewma.csv", roc_ew,  delimiter=",", header="thr,FPR,TPR,TP,FP,TN,FN", comments="")
    np.savetxt("ids_roc_cusum.csv", roc_cu, delimiter=",", header="thr,FPR,TPR,TP,FP,TN,FN", comments="")

    # Plots ROC comparison
    plt.figure()
    plt.plot(roc_nis[:, 1], roc_nis[:, 2], label=f"raw NIS (AUC={auc_nis:.3f})")
    plt.plot(roc_ew[:, 1],  roc_ew[:, 2],  label=f"EWMA (AUC={auc_ew:.3f})")
    plt.plot(roc_cu[:, 1],  roc_cu[:, 2],  label=f"CUSUM (AUC={auc_cu:.3f})")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC: raw vs EWMA vs CUSUM")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ids_roc_compare.png", dpi=160)

    print("Saved:")
    print(" - ids_methods_summary.csv")
    print(" - ids_roc_raw.csv, ids_roc_ewma.csv, ids_roc_cusum.csv")
    print(" - ids_roc_compare.png")
    print("")
    print("Operating points at FPR<=1%:")
    print(f" raw:  thr={thrN:.4g} FPR={fprN:.4f} TPR={tprN:.4f} det_rate={detN:.3f} delay_mean={meanN:.2f}")
    print(f" ewma: thr={thrE:.4g} FPR={fprE:.4f} TPR={tprE:.4f} det_rate={detE:.3f} delay_mean={meanE:.2f}")
    print(f" cusum:thr={thrC:.4g} FPR={fprC:.4f} TPR={tprC:.4f} det_rate={detC:.3f} delay_mean={meanC:.2f}")


if __name__ == "__main__":
    main()
