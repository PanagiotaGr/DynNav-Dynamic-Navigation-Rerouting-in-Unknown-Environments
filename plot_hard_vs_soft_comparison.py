# plot_hard_vs_soft_comparison.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    # HARD: irreversibility constraint
    df_tau = pd.read_csv("irreversibility_bottleneck_tau_sweep.csv")
    # SOFT: risk-weighted A*
    df_lam = pd.read_csv("risk_weighted_lambda_sweep.csv")

    # For hard results: compute geometric cost from path_len (step_cost=1)
    df_tau = df_tau.copy()
    df_tau["geo_cost"] = df_tau["path_len"].replace(0, np.nan) - 1

    # For soft results: geo_cost already there, but keep consistent NaN for failures
    df_lam = df_lam.copy()
    df_lam.loc[df_lam["success"] == 0, "geo_cost"] = np.nan
    df_lam.loc[df_lam["success"] == 0, "max_I_on_path"] = np.nan

    # ---- Plot: 3 rows x 2 cols (tau on left, lambda on right) ----
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))

    # Row 1: Feasibility
    axs[0, 0].plot(df_tau["tau"], df_tau["success"], marker="o")
    axs[0, 0].set_title("HARD constraint: feasibility vs τ")
    axs[0, 0].set_xlabel("τ (irreversibility threshold)")
    axs[0, 0].set_ylabel("success (0/1)")
    axs[0, 0].set_ylim(-0.05, 1.05)
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(df_lam["lambda"], df_lam["success"], marker="o")
    axs[0, 1].set_title("SOFT penalty: feasibility vs λ")
    axs[0, 1].set_xlabel("λ (risk weight)")
    axs[0, 1].set_ylabel("success (0/1)")
    axs[0, 1].set_ylim(-0.05, 1.05)
    axs[0, 1].set_xscale("symlog", linthresh=0.1)
    axs[0, 1].grid(True, alpha=0.3)

    # Row 2: Max irreversibility on path
    axs[1, 0].plot(df_tau["tau"], df_tau["max_I_on_path"], marker="o")
    axs[1, 0].set_title("HARD: max I along feasible path")
    axs[1, 0].set_xlabel("τ")
    axs[1, 0].set_ylabel("max I on path")
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].plot(df_lam["lambda"], df_lam["max_I_on_path"], marker="o")
    axs[1, 1].set_title("SOFT: max I along path (NOT controlled)")
    axs[1, 1].set_xlabel("λ")
    axs[1, 1].set_ylabel("max I on path")
    axs[1, 1].set_xscale("symlog", linthresh=0.1)
    axs[1, 1].grid(True, alpha=0.3)

    # Row 3: Geometric cost (path length)
    axs[2, 0].plot(df_tau["tau"], df_tau["geo_cost"], marker="o")
    axs[2, 0].set_title("HARD: geometric cost vs τ (feasible only)")
    axs[2, 0].set_xlabel("τ")
    axs[2, 0].set_ylabel("geometric cost (len-1)")
    axs[2, 0].grid(True, alpha=0.3)

    axs[2, 1].plot(df_lam["lambda"], df_lam["geo_cost"], marker="o")
    axs[2, 1].set_title("SOFT: geometric cost vs λ")
    axs[2, 1].set_xlabel("λ")
    axs[2, 1].set_ylabel("geometric cost (len-1)")
    axs[2, 1].set_xscale("symlog", linthresh=0.1)
    axs[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = "hard_vs_soft_comparison.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    print("Saved:", out)

    # Print a short scientific summary
    # Hard critical tau
    df_tau_s = df_tau[df_tau["success"] == 1]
    if len(df_tau_s) > 0:
        print("Hard constraint critical tau ~", float(df_tau_s["tau"].min()))
        print("Hard failures:", df_tau[df_tau["success"] == 0]["reason"].value_counts().to_dict())

    # Soft always succeeds here
    print("Soft maxI unique values:", sorted(set(df_lam["max_I_on_path"].dropna().round(3).tolist())))


if __name__ == "__main__":
    main()
