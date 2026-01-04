# plot_risk_weighted_lambda_sweep.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("risk_weighted_lambda_sweep.csv")
    df_s = df[df["success"] == 1].copy()

    # 1) max I vs lambda
    plt.figure()
    plt.plot(df["lambda"], df["max_I_on_path"], marker="o")
    plt.xscale("symlog", linthresh=0.1)
    plt.xlabel("lambda (risk weight)")
    plt.ylabel("max I along path")
    plt.title("Soft risk weighting does not control max irreversibility")
    plt.grid(True, alpha=0.3)
    out1 = "risk_weighted_maxI_vs_lambda.png"
    plt.savefig(out1, dpi=200, bbox_inches="tight")
    print("Saved:", out1)
    plt.close()

    # 2) mean I vs lambda
    plt.figure()
    plt.plot(df["lambda"], df["mean_I_on_path"], marker="o")
    plt.xscale("symlog", linthresh=0.1)
    plt.xlabel("lambda (risk weight)")
    plt.ylabel("mean I along path")
    plt.title("Mean irreversibility decreases with lambda (soft risk)")
    plt.grid(True, alpha=0.3)
    out2 = "risk_weighted_meanI_vs_lambda.png"
    plt.savefig(out2, dpi=200, bbox_inches="tight")
    print("Saved:", out2)
    plt.close()

    # 3) geometric cost vs lambda (successful)
    plt.figure()
    if len(df_s) > 0:
        plt.plot(df_s["lambda"], df_s["geo_cost"], marker="o")
        plt.xscale("symlog", linthresh=0.1)
    plt.xlabel("lambda (risk weight)")
    plt.ylabel("geometric cost (path length - 1)")
    plt.title("Path length trade-off under soft risk weighting")
    plt.grid(True, alpha=0.3)
    out3 = "risk_weighted_geocost_vs_lambda.png"
    plt.savefig(out3, dpi=200, bbox_inches="tight")
    print("Saved:", out3)
    plt.close()

    # 4) expansions vs lambda (successful)
    plt.figure()
    if len(df_s) > 0:
        plt.plot(df_s["lambda"], df_s["expansions"], marker="o")
        plt.xscale("symlog", linthresh=0.1)
    plt.xlabel("lambda (risk weight)")
    plt.ylabel("A* expansions")
    plt.title("Search effort vs lambda (soft risk)")
    plt.grid(True, alpha=0.3)
    out4 = "risk_weighted_expansions_vs_lambda.png"
    plt.savefig(out4, dpi=200, bbox_inches="tight")
    print("Saved:", out4)
    plt.close()

    # Summary
    print("\nSummary:")
    print(df[["lambda", "success", "geo_cost", "max_I_on_path", "mean_I_on_path", "expansions"]].to_string(index=False))


if __name__ == "__main__":
    main()

