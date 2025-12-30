# ================================================================
# Benchmark: Navigation Policies under Drift & Uncertainty
#
# Συγκρίνουμε 4 πολιτικές επιλογής διαδρομής:
#  - baseline_shortest: αγνοεί drift/uncertainty, επιλέγει μόνο το μικρότερο μήκος
#  - fixed_risk: σταθερό λ για cost = length + λ*(drift + κ*uncertainty)
#  - adaptive_risk: λ προσαρμόζεται από self-trust S ∈ [0,1]
#  - ood_aware: σε "δύσκολα" περιβάλλοντα (high difficulty) ενεργοποιεί safe-mode
#
# Παράγει:
#  - logs_benchmark/benchmark_results.csv
#  - logs_benchmark/benchmark_summary.csv
#  - logs_benchmark/benchmark_bar_metrics.png
#  - logs_benchmark/benchmark_boxplots.png
# ================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sample_paths(rng: np.random.Generator):
    """
    Δημιουργεί 3 υποψήφιες διαδρομές (A, B, C) με συνθετικά χαρακτηριστικά:
    - length: γύρω από 10–20
    - drift_exposure: 0–5
    - uncertainty_exposure: 0–0.01
    """
    paths = []
    names = ["A", "B", "C"]
    for name in names:
        length = float(rng.normal(loc=12.0, scale=2.0))
        length = max(length, 5.0)  # δεν θέλουμε αρνητικά/γελοία μικρά

        drift = float(rng.uniform(0.0, 5.0))
        unc = float(rng.uniform(0.0, 0.01))

        paths.append({
            "name": name,
            "length": length,
            "drift": drift,
            "unc": unc,
        })
    return paths


def compute_cost(path, lam: float, k_unc: float = 100.0):
    """
    Κόστος για risk-aware planner:
      cost = length + λ * (drift + k_unc * uncertainty)
    όπου k_unc είναι scaling factor για να φέρει τα μεγέθη σε παρόμοια κλίμακα.
    """
    length = path["length"]
    drift = path["drift"]
    unc = path["unc"]
    return length + lam * (drift + k_unc * unc)


def choose_baseline_shortest(paths):
    # αγνοεί drift/uncertainty, επιλέγει μόνο το μικρότερο μήκος
    lengths = [p["length"] for p in paths]
    idx = int(np.argmin(lengths))
    return paths[idx]


def choose_fixed_risk(paths, lam: float):
    costs = [compute_cost(p, lam) for p in paths]
    idx = int(np.argmin(costs))
    return paths[idx], costs[idx]


def lambda_from_self_trust(S: float, lam_min: float = 0.5, lam_max: float = 3.0):
    """
    Self-trust S ∈ [0,1]:
      - χαμηλό S → κοντά στο lam_max (πιο συντηρητικό)
      - υψηλό S → κοντά στο lam_min (πιο επιθετικό)
    """
    S_clamped = float(np.clip(S, 0.0, 1.0))
    return lam_max - S_clamped * (lam_max - lam_min)


def choose_adaptive_risk(paths, S: float):
    lam = lambda_from_self_trust(S)
    costs = [compute_cost(p, lam) for p in paths]
    idx = int(np.argmin(costs))
    return paths[idx], costs[idx], lam


def choose_ood_aware(paths, env_difficulty: float, ood_threshold: float = 0.75):
    """
    Αν το env_difficulty > ood_threshold:
      - θεωρείται "δύσκολο / OOD"
      - ενεργοποιούμε safe-mode → επιλέγουμε path με ελάχιστο (drift + κ*unc)
    Αλλιώς:
      - συμπεριφέρεται σαν fixed-risk με μέτριο λ
    Επιστρέφει:
      chosen_path, cost, mode_str ("NORMAL" ή "SAFE")
    """
    if env_difficulty > ood_threshold:
        # SAFE mode: αγνοούμε μήκος, δίνουμε προτεραιότητα σε ασφάλεια
        k_unc = 100.0
        safe_scores = [p["drift"] + k_unc * p["unc"] for p in paths]
        idx = int(np.argmin(safe_scores))
        chosen = paths[idx]
        # Κόστος εδώ μπορούμε να το ορίσουμε σαν "effective" cost:
        cost = chosen["length"] + 3.0 * safe_scores[idx]
        return chosen, cost, "SAFE"
    else:
        lam = 1.5
        costs = [compute_cost(p, lam) for p in paths]
        idx = int(np.argmin(costs))
        chosen = paths[idx]
        return chosen, costs[idx], "NORMAL"


def run_benchmark(num_trials: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)

    results = []

    for trial in range(num_trials):
        # "περιβάλλον" με έναν δείκτη δυσκολίας (0–1)
        env_difficulty = float(rng.uniform(0.0, 1.0))
        # self-trust S ~ αντίστροφο της δυσκολίας (απλά για το πείραμα)
        S_env = 1.0 - env_difficulty  # όσο πιο δύσκολο το περιβάλλον, τόσο μικρότερο S

        paths = sample_paths(rng)

        # ---------------- Baseline (shortest only) ----------------
        p_base = choose_baseline_shortest(paths)
        results.append({
            "trial": trial,
            "policy": "baseline_shortest",
            "env_difficulty": env_difficulty,
            "self_trust_S": S_env,
            "chosen_path": p_base["name"],
            "length": p_base["length"],
            "drift": p_base["drift"],
            "unc": p_base["unc"],
            "cost": p_base["length"],  # δεν υπάρχει risk term
            "mode": "NORMAL",
        })

        # ---------------- Fixed-risk planner ----------------
        lam_fixed = 1.5
        p_fix, cost_fix = choose_fixed_risk(paths, lam=lam_fixed)
        results.append({
            "trial": trial,
            "policy": "fixed_risk",
            "env_difficulty": env_difficulty,
            "self_trust_S": S_env,
            "chosen_path": p_fix["name"],
            "length": p_fix["length"],
            "drift": p_fix["drift"],
            "unc": p_fix["unc"],
            "cost": cost_fix,
            "mode": "NORMAL",
        })

        # ---------------- Adaptive-risk planner ----------------
        p_adapt, cost_adapt, lam_adapt = choose_adaptive_risk(paths, S=S_env)
        results.append({
            "trial": trial,
            "policy": "adaptive_risk",
            "env_difficulty": env_difficulty,
            "self_trust_S": S_env,
            "chosen_path": p_adapt["name"],
            "length": p_adapt["length"],
            "drift": p_adapt["drift"],
            "unc": p_adapt["unc"],
            "cost": cost_adapt,
            "mode": "NORMAL",
            "lambda_used": lam_adapt,
        })

        # ---------------- OOD-aware planner ----------------
        p_ood, cost_ood, mode_ood = choose_ood_aware(paths, env_difficulty)
        results.append({
            "trial": trial,
            "policy": "ood_aware",
            "env_difficulty": env_difficulty,
            "self_trust_S": S_env,
            "chosen_path": p_ood["name"],
            "length": p_ood["length"],
            "drift": p_ood["drift"],
            "unc": p_ood["unc"],
            "cost": cost_ood,
            "mode": mode_ood,
        })

    df = pd.DataFrame(results)

    # ορίζουμε ένα απλό "success" κριτήριο:
    #   - επιτυχημένη πλοήγηση αν drift < 3.5
    df["success"] = df["drift"] < 3.5

    return df


def summarize_results(df: pd.DataFrame):
    """
    Φτιάχνει summary ανά policy:
      - mean length, drift, unc, cost
      - success rate
      - % SAFE mode (για ood_aware)
    """
    summaries = []
    for policy, group in df.groupby("policy"):
        mean_len = group["length"].mean()
        mean_drift = group["drift"].mean()
        mean_unc = group["unc"].mean()
        mean_cost = group["cost"].mean()
        success_rate = group["success"].mean()
        # % SAFE mode (μόνο για ood_aware έχει νόημα, αλλά δεν πειράζει)
        safe_ratio = (group["mode"] == "SAFE").mean()

        summaries.append({
            "policy": policy,
            "mean_length": mean_len,
            "mean_drift": mean_drift,
            "mean_unc": mean_unc,
            "mean_cost": mean_cost,
            "success_rate": success_rate,
            "safe_mode_ratio": safe_ratio,
            "num_trials": len(group),
        })

    df_summary = pd.DataFrame(summaries)
    return df_summary


def make_plots(df: pd.DataFrame, df_summary: pd.DataFrame, out_dir: str):
    # --------- Bar plot: mean metrics per policy ---------
    policies = df_summary["policy"].values
    x = np.arange(len(policies))

    mean_drift = df_summary["mean_drift"].values
    mean_length = df_summary["mean_length"].values
    success_rate = df_summary["success_rate"].values

    plt.figure(figsize=(10, 5))
    width = 0.25

    plt.bar(x - width, mean_drift, width=width, label="Mean Drift")
    plt.bar(x, mean_length, width=width, label="Mean Length")
    plt.bar(x + width, success_rate, width=width, label="Success Rate")

    plt.xticks(x, policies, rotation=20)
    plt.ylabel("Value")
    plt.title("Benchmark Metrics per Policy")
    plt.legend()
    plt.tight_layout()

    bar_path = os.path.join(out_dir, "benchmark_bar_metrics.png")
    plt.savefig(bar_path, dpi=200)
    plt.close()

    # --------- Boxplot: drift per policy ---------
    plt.figure(figsize=(8, 5))
    data_drift = [df[df["policy"] == p]["drift"].values for p in policies]
    plt.boxplot(data_drift, labels=policies)
    plt.ylabel("Drift")
    plt.title("Drift Distribution per Policy")
    plt.tight_layout()

    box_path = os.path.join(out_dir, "benchmark_boxplots.png")
    plt.savefig(box_path, dpi=200)
    plt.close()


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, "logs_benchmark")
    os.makedirs(out_dir, exist_ok=True)

    print("[BMARK] Running benchmark experiments...")
    df = run_benchmark(num_trials=200, seed=0)

    results_path = os.path.join(out_dir, "benchmark_results.csv")
    df.to_csv(results_path, index=False)
    print(f"[BMARK] Saved per-trial results to: {results_path}")

    df_summary = summarize_results(df)
    summary_path = os.path.join(out_dir, "benchmark_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"[BMARK] Saved summary to: {summary_path}")

    print("\n[BMARK] Summary:")
    print(df_summary.to_string(index=False))

    print("\n[BMARK] Generating plots...")
    make_plots(df, df_summary, out_dir)
    print(f"[BMARK] Saved plots to: {out_dir}")

    print("\n[BMARK] Benchmark finished 🎯")


if __name__ == "__main__":
    main()
