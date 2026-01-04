# plot_world_template_bench.py
import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("world_template_bench.csv")

    # success rates per template
    g = df.groupby("template")
    fixed = g["fixed_success"].mean()
    safe = g["safe_success"].mean()
    mm = g["mm_success"].mean()

    # bar plot
    plt.figure(figsize=(7, 4))
    x = range(len(fixed.index))
    labels = list(fixed.index)

    plt.bar([i - 0.25 for i in x], fixed.values, width=0.25, label="fixed tau0")
    plt.bar([i for i in x], safe.values, width=0.25, label="safe-mode")
    plt.bar([i + 0.25 for i in x], mm.values, width=0.25, label="minimax tau")

    plt.xticks(list(x), labels, rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("success rate")
    plt.title("Success rate by world template")
    plt.grid(True, axis="y", alpha=0.3)
    out1 = "world_template_success_rates.png"
    plt.savefig(out1, dpi=220, bbox_inches="tight")
    print("Saved:", out1)
    plt.close()

    # tau_star distribution per template (simple hist in separate figures)
    for t in df["template"].unique():
        sub = df[df["template"] == t]
        vals = sub["tau_star"].dropna().values
        plt.figure()
        plt.hist(vals, bins=15)
        plt.xlabel("tau_star")
        plt.ylabel("count")
        plt.title(f"tau_star distribution - {t}")
        plt.grid(True, alpha=0.3)
        out = f"tau_star_hist_{t}.png"
        plt.savefig(out, dpi=220, bbox_inches="tight")
        print("Saved:", out)
        plt.close()


if __name__ == "__main__":
    main()
