import pandas as pd
import matplotlib.pyplot as plt

def main(csv_path="connectivity_sweep.csv"):
    df = pd.read_csv(csv_path)

    # Keep only successful runs
    ok = df[df["status"]=="ok"].copy()

    # Aggregate
    agg = ok.groupby(["planner","gamma"], dropna=False).agg(
        path_len_mean=("path_len","mean"),
        expansions_mean=("expansions","mean"),
        conn_pen_mean=("path_conn_pen_sum","mean"),
        risk_sum_mean=("path_risk_sum","mean"),
    ).reset_index()

    print(agg)

    # Plot expansions vs gamma for connectivity-aware
    sub = agg[agg["planner"]=="risk_plus_connectivity"].sort_values("gamma")
    plt.figure()
    plt.plot(sub["gamma"], sub["expansions_mean"], marker="o")
    plt.xlabel("gamma (connectivity weight)")
    plt.ylabel("mean node expansions")
    plt.title("Expansions vs gamma (risk+connectivity planner)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("connectivity_expansions_vs_gamma.png", dpi=200)

    # Compare baselines (single point)
    base = agg[agg["planner"].isin(["geometry_only","risk_only"])]
    if len(base) > 0:
        plt.figure()
        plt.bar(base["planner"], base["path_len_mean"])
        plt.ylabel("mean path length")
        plt.title("Baseline comparison (mean path length)")
        plt.tight_layout()
        plt.savefig("connectivity_baseline_pathlen.png", dpi=200)

    print("[OK] saved plots: connectivity_expansions_vs_gamma.png, connectivity_baseline_pathlen.png")

if __name__ == "__main__":
    main()
