# plot_irreversibility_safe_mode_sweep.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("irreversibility_safe_mode_sweep.csv")

    # Activation: SAFE_RELAX_TAU counts
    df["safe_activated"] = (df["mode"] == "SAFE_RELAX_TAU").astype(int)

    # 1) mode vs tau0 (activation)
    plt.figure()
    plt.plot(df["tau0"], df["safe_activated"], marker="o")
    plt.xlabel("requested tau0")
    plt.ylabel("safe mode activated (0/1)")
    plt.title("Safe-mode activation vs requested irreversibility threshold")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    out1 = "safe_mode_activation_vs_tau0.png"
    plt.savefig(out1, dpi=220, bbox_inches="tight")
    print("Saved:", out1)
    plt.close()

    # 2) tau_used vs tau0
    plt.figure()
    plt.plot(df["tau0"], df["tau_used"], marker="o")
    plt.xlabel("requested tau0")
    plt.ylabel("tau_used (after safe-mode)")
    plt.title("Minimal relaxation: tau_used vs tau0")
    plt.grid(True, alpha=0.3)
    out2 = "safe_mode_tau_used_vs_tau0.png"
    plt.savefig(out2, dpi=220, bbox_inches="tight")
    print("Saved:", out2)
    plt.close()

    # 3) tau_gap vs tau0
    plt.figure()
    plt.plot(df["tau0"], df["tau_gap"], marker="o")
    plt.xlabel("requested tau0")
    plt.ylabel("tau_gap = tau_used - tau0")
    plt.title("Relaxation gap required for feasibility")
    plt.grid(True, alpha=0.3)
    out3 = "safe_mode_tau_gap_vs_tau0.png"
    plt.savefig(out3, dpi=220, bbox_inches="tight")
    print("Saved:", out3)
    plt.close()

    # summary
    rate = float(df["safe_activated"].mean())
    print("\nSafe-mode activation rate:", rate)
    print("Modes:", df["mode"].value_counts().to_dict())


if __name__ == "__main__":
    main()
