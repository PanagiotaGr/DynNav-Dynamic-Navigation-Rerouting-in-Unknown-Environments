import os
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths στα summary που ήδη έχεις φτιάξει
PATH_CALIB = os.path.join(BASE, "logs_calibration", "uncertainty_calibration_summary.csv")
PATH_CALIB_ENS = os.path.join(BASE, "logs_calibration_ensemble", "uncertainty_calibration_summary_ensemble.csv")
PATH_OOD = os.path.join(BASE, "logs_ood", "ood_variance_summary.csv")
PATH_BENCH = os.path.join(BASE, "logs_benchmark", "benchmark_summary.csv")

OUT_DIR = os.path.join(BASE, "research_results")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "research_summary.csv")


def safe_load(path, name):
    if not os.path.exists(path):
        print(f"[WARN] Missing {name}: {path}")
        return None
    print(f"[OK] Loaded {name}: {path}")
    return pd.read_csv(path)


def main():
    print("\n================ RESEARCH SUMMARY =================\n")

    calib = safe_load(PATH_CALIB, "Calibration")
    calib_ens = safe_load(PATH_CALIB_ENS, "Ensemble Calibration")
    ood = safe_load(PATH_OOD, "OOD Variance Summary")
    bench = safe_load(PATH_BENCH, "Planning Benchmark")

    rows = []

    # ----- Dropout-based calibration -----
    if calib is not None and len(calib) > 0:
        rows.append({
            "module": "uncertainty_calibration_dropout",
            "mse": float(calib.loc[0, "mse"]),
            "nll": float(calib.loc[0, "nll"]),
            "uce": float(calib.loc[0, "uce"]),
            "num_samples": int(calib.loc[0, "num_samples"]),
        })

    # ----- Ensemble calibration -----
    if calib_ens is not None and len(calib_ens) > 0:
        rows.append({
            "module": "uncertainty_calibration_ensemble",
            "mse": float(calib_ens.loc[0, "mse"]),
            "nll": float(calib_ens.loc[0, "nll"]),
            "uce": float(calib_ens.loc[0, "uce"]),
            "num_samples": int(calib_ens.loc[0, "num_samples"]),
        })

    # ----- OOD detection (variance-based) -----
    if ood is not None and len(ood) > 0:
        row = ood.loc[0]
        rows.append({
            "module": "ood_detection_variance",
            "auc": float(row["auc"]),
            "id_var_mean": float(row["id_var_mean"]),
            "id_var_std": float(row["id_var_std"]),
            "ood_var_mean": float(row["ood_var_mean"]),
            "ood_var_std": float(row["ood_var_std"]),
            "num_id": int(row["num_id"]),
            "num_ood": int(row["num_ood"]),
        })

    # ----- Planning benchmark per policy -----
    if bench is not None and len(bench) > 0:
        for _, r in bench.iterrows():
            rows.append({
                "module": f"benchmark_policy_{r['policy']}",
                "mean_length": float(r["mean_length"]),
                "mean_drift": float(r["mean_drift"]),
                "mean_unc": float(r["mean_unc"]),
                "mean_cost": float(r["mean_cost"]),
                "success_rate": float(r["success_rate"]),
                "safe_mode_ratio": float(r["safe_mode_ratio"]),
                "num_trials": int(r["num_trials"]),
            })

    if not rows:
        print("[ERROR] No results found to summarize.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)

    print(df.to_string(index=False))
    print(f"\n[RESULT] Saved unified research summary to:\n{OUT_PATH}")
    print("\n====================================================\n")


if __name__ == "__main__":
    main()
