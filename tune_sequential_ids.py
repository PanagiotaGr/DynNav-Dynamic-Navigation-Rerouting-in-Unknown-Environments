import numpy as np
import matplotlib.pyplot as plt

from attack_aware_ukf import AttackAwareUKF, SensorConfig, f_constant_velocity, h_position
from sequential_ids import EWMADetector, EWMAConfig, CUSUMDetector, CUSUMConfig


def simulate_nis(seed: int, T: int, attack_start: int):
    rng = np.random.RandomState(seed)

    n = 4
    dt = 0.1
    Q = np.diag([1e-4, 1e-4, 1e-3, 1e-3])

    ukf = AttackAwareUKF(n=n, f=f_constant_velocity, Q=Q)
    ukf.set_state(np.array([0, 0, 1, 0.5], float), np.diag([0.1, 0.1, 0.2, 0.2]))

    vo_cfg = SensorConfig(name="vo", R_base=np.diag([0.05**2, 0.05**2]))
    wheel_cfg = SensorConfig(name="wheel", R_base=np.diag([0.10**2, 0.10**2]))
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


def evaluate_detector(scores_runs, labels_runs, thr):
    # per-step confusion over all steps
    scores = np.concatenate(scores_runs)
    labels = np.concatenate(labels_runs)
    pred = (scores >= thr).astype(int)

    TP = int(np.sum((pred == 1) & (labels == 1)))
    FP = int(np.sum((pred == 1) & (labels == 0)))
    TN = int(np.sum((pred == 0) & (labels == 0)))
    FN = int(np.sum((pred == 0) & (labels == 1)))

    TPR = TP / (TP + FN) if (TP + FN) else 0.0
    FPR = FP / (FP + TN) if (FP + TN) else 0.0

    return TPR, FPR


def detection_delay(scores, thr, attack_start):
    idx = np.where(scores[attack_start:] >= thr)[0]
    return np.inf if idx.size == 0 else int(idx[0])


def main():
    T = 250
    attack_start = 80
    seeds = list(range(30))

    nis_runs = []
    y_runs = []
    for s in seeds:
        nis, y = simulate_nis(s, T=T, attack_start=attack_start)
        nis_runs.append(nis)
        y_runs.append(y)

    # Parameter grids
    lam_grid = [0.02, 0.05, 0.1, 0.15]
    k_grid = [0.5, 1.0, 1.5, 2.0, 2.5]
    # We'll tune CUSUM threshold h internally using quantiles of score under normal to meet FPR target.
    fpr_target = 0.01

    results = []

    for lam in lam_grid:
        # EWMA scores per run
        ew_runs = []
        for nis in nis_runs:
            det = EWMADetector(EWMAConfig(lam=lam))
            ew = np.array([det.update(v) for v in nis], float)
            ew_runs.append(ew)

        # choose threshold by normal quantile to approx meet FPR target
        ew_all = np.concatenate(ew_runs)
        y_all = np.concatenate(y_runs)
        normal_scores = ew_all[y_all == 0]
        thr_ew = float(np.quantile(normal_scores, 1.0 - fpr_target))

        TPR, FPR = evaluate_detector(ew_runs, y_runs, thr_ew)

        delays = [detection_delay(ew, thr_ew, attack_start) for ew in ew_runs]
        delays = np.array(delays, float)
        det_rate = float(np.mean(np.isfinite(delays)))
        delay_mean = float(np.mean(delays[np.isfinite(delays)])) if np.any(np.isfinite(delays)) else np.inf

        results.append(["EWMA", lam, np.nan, thr_ew, TPR, FPR, det_rate, delay_mean])

    for k in k_grid:
        cu_runs = []
        for nis in nis_runs:
            det = CUSUMDetector(CUSUMConfig(k=k, h=1e9))  # h will be selected later
            cu = np.array([det.update(v) for v in nis], float)
            cu_runs.append(cu)

        cu_all = np.concatenate(cu_runs)
        y_all = np.concatenate(y_runs)
        normal_scores = cu_all[y_all == 0]
        thr_cu = float(np.quantile(normal_scores, 1.0 - fpr_target))

        TPR, FPR = evaluate_detector(cu_runs, y_runs, thr_cu)

        delays = [detection_delay(cu, thr_cu, attack_start) for cu in cu_runs]
        delays = np.array(delays, float)
        det_rate = float(np.mean(np.isfinite(delays)))
        delay_mean = float(np.mean(delays[np.isfinite(delays)])) if np.any(np.isfinite(delays)) else np.inf

        results.append(["CUSUM", np.nan, k, thr_cu, TPR, FPR, det_rate, delay_mean])

    # Save
    with open("tuning_results.csv", "w") as f:
        f.write("method,lam,k,thr,TPR,FPR,detect_rate,delay_mean_steps\n")
        for r in results:
            f.write(",".join(str(x) for x in r) + "\n")

    # Simple plot: delay vs param
    ew = [r for r in results if r[0] == "EWMA"]
    cu = [r for r in results if r[0] == "CUSUM"]

    plt.figure()
    plt.plot([r[1] for r in ew], [r[7] for r in ew], marker="o", label="EWMA delay_mean")
    plt.xlabel("lam")
    plt.ylabel("mean delay (steps)")
    plt.title("EWMA tuning at FPR≈1% (threshold by normal quantile)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("tune_ewma_delay.png", dpi=160)

    plt.figure()
    plt.plot([r[2] for r in cu], [r[7] for r in cu], marker="o", label="CUSUM delay_mean")
    plt.xlabel("k")
    plt.ylabel("mean delay (steps)")
    plt.title("CUSUM tuning at FPR≈1% (threshold by normal quantile)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("tune_cusum_delay.png", dpi=160)

    print("Saved: tuning_results.csv, tune_ewma_delay.png, tune_cusum_delay.png")


if __name__ == "__main__":
    main()
