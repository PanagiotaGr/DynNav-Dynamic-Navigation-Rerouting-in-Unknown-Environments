# plot_failure_taxonomy.py
import pandas as pd
import matplotlib.pyplot as plt


def plot_counts(series, title, out):
    counts = series.value_counts()

    plt.figure(figsize=(8, 4))
    plt.bar(counts.index.astype(str), counts.values)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("count")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.savefig(out, dpi=220, bbox_inches="tight")
    print("Saved:", out)
    plt.close()


def main():
    df = pd.read_csv("minimax_tau_multistart_with_failures.csv")

    # Fixed policy failure types
    plot_counts(
        df["fixed_failure_type"],
        "Failure taxonomy (fixed tau0 policy)",
        "failure_taxonomy_fixed.png",
    )

    # Safe-mode failure types
    plot_counts(
        df["safe_failure_type"],
        "Failure taxonomy (safe-mode policy)",
        "failure_taxonomy_safe_mode.png",
    )


if __name__ == "__main__":
    main()

