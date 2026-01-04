# plot_returnability_mu_sweep.py
import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("returnability_mu_sweep.csv").sort_values("mu")

    # cost vs mu
    plt.figure()
    plt.plot(df["mu"], df["cost"], marker="o")
    plt.xlabel("mu")
    plt.ylabel("total cost")
    plt.title("Cost vs mu (returnability weight)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out1 = "returnability_cost_vs_mu.png"
    plt.savefig(out1, dpi=220)
    print("Saved:", out1)
    plt.close()

    # meanR vs mu
    plt.figure()
    plt.plot(df["mu"], df["meanR"], marker="o")
    plt.xlabel("mu")
    plt.ylabel("mean returnability on path")
    plt.title("Mean returnability vs mu")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out2 = "returnability_meanR_vs_mu.png"
    plt.savefig(out2, dpi=220)
    print("Saved:", out2)
    plt.close()

    # meanI vs mu
    plt.figure()
    plt.plot(df["mu"], df["meanI"], marker="o")
    plt.xlabel("mu")
    plt.ylabel("mean irreversibility on path")
    plt.title("Mean irreversibility vs mu")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out3 = "returnability_meanI_vs_mu.png"
    plt.savefig(out3, dpi=220)
    print("Saved:", out3)
    plt.close()


if __name__ == "__main__":
    main()
