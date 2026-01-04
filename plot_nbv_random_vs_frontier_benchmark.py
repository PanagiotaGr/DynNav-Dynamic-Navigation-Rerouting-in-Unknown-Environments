# plot_nbv_random_vs_frontier_benchmark.py
import pandas as pd
import matplotlib.pyplot as plt


ORDER = ["random_global", "random_frontier", "frontier_scored"]


def barplot_metric(df, metric, title, out):
    stats = df.groupby("method")[metric].agg(["mean", "std"]).reindex(ORDER)

    methods = stats.index.tolist()
    means = stats["mean"].tolist()
    stds = stats["std"].tolist()

    plt.figure(figsize=(7.2, 4.6))
    plt.bar(methods, means, yerr=stds, capsize=6)
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    print("Saved:", out)
    plt.close()


def main():
    df = pd.read_csv("nbv_random_vs_frontier_benchmark.csv")

    print("\nSummary (mean ± std):")
    for metric in ["mean_IG", "mean_I", "mean_R"]:
        s = df.groupby("method")[metric].agg(["mean", "std"]).reindex(ORDER)
        print("\n", metric)
        print(s)

    barplot_metric(df, "mean_IG", "Top-10 average IG (3 methods)", "bench_top10_meanIG_3methods.png")
    barplot_metric(df, "mean_I", "Top-10 average Irreversibility (3 methods)", "bench_top10_meanI_3methods.png")
    barplot_metric(df, "mean_R", "Top-10 average Returnability (3 methods)", "bench_top10_meanR_3methods.png")


if __name__ == "__main__":
    main()
