# plot_minimax_tau_multistart.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("minimax_tau_multistart.csv")

    # 1) Histogram of tau_star
    vals = df["tau_star"].dropna().values
    plt.figure()
    plt.hist(vals, bins=15)
    plt.xlabel("tau_star (minimal feasible irreversibility threshold)")
    plt.ylabel("count")
    plt.title("Distribution of minimax feasibility thresholds")
    plt.grid(True, alpha=0.3)
    out1 = "tau_star_hist.png"
    plt.savefig(out1, dpi=220, bbox_inches="tight")
    print("Saved:", out1)
    plt.close()

    # 2) Success rate by policy
    fixed_rate = df["fixed_success"].mean()
    safe_rate = df["safe_success"].mean()
    mm_rate = df["mm_success"].mean()

    plt.figure()
    plt.bar(["fixed_tau0", "safe_mode", "minimax_tau"], [fixed_rate, safe_rate, mm_rate])
    plt.ylim(0, 1.05)
    plt.ylabel("success rate")
    plt.title("Success rate comparison (fixed vs safe-mode vs minimax tau)")
    plt.grid(True, axis="y", alpha=0.3)
    out2 = "success_rate_policies.png"
    plt.savefig(out2, dpi=220, bbox_inches="tight")
    print("Saved:", out2)
    plt.close()

    # 3) Safe-mode tau_gap distribution (only activated)
    gaps = df.loc[df["safe_mode"] == "SAFE_RELAX_TAU", "safe_tau_gap"].dropna().values
    plt.figure()
    if len(gaps) > 0:
        plt.hist(gaps, bins=15)
    plt.xlabel("safe tau_gap (tau_used - tau0)")
    plt.ylabel("count")
    plt.title("Safe-mode relaxation gap distribution")
    plt.grid(True, alpha=0.3)
    out3 = "safe_mode_tau_gap_hist.png"
    plt.savefig(out3, dpi=220, bbox_inches="tight")
    print("Saved:", out3)
    plt.close()

    # 4) Expansions comparison (successful only)
    df_s = df.copy()
    plt.figure()
    plt.scatter(df_s["fixed_expansions"], df_s["mm_expansions"])
    plt.xlabel("fixed expansions")
    plt.ylabel("minimax expansions")
    plt.title("Search effort: fixed vs minimax (per pair)")
    plt.grid(True, alpha=0.3)
    out4 = "expansions_fixed_vs_minimax.png"
    plt.savefig(out4, dpi=220, bbox_inches="tight")
    print("Saved:", out4)
    plt.close()

    print("\nRates:")
    print("fixed:", fixed_rate, "safe:", safe_rate, "minimax:", mm_rate)
    print("safe activations:", int((df["safe_mode"] == "SAFE_RELAX_TAU").sum()), "/", len(df))


if __name__ == "__main__":
    main()
