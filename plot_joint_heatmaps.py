import pandas as pd
import matplotlib.pyplot as plt

def heatmap(df, value_col, out_png, title):
    pivot = df.pivot(index="delta", columns="gamma", values=value_col)
    plt.figure()
    plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
    plt.yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
    plt.xlabel("gamma")
    plt.ylabel("delta")
    plt.title(title)
    plt.colorbar(label=value_col)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)

def main():
    df = pd.read_csv("energy_connectivity_joint_sweep.csv")
    ok = df[df["status"]=="ok"].copy()
    agg = ok.groupby(["delta","gamma"]).agg(
        disconnect_mean=("disconnect_steps","mean"),
        safe_mean=("safe_mode_activations","mean"),
        pathlen_mean=("path_len","mean"),
    ).reset_index()

    heatmap(agg, "disconnect_mean", "heatmap_disconnect.png", "Mean disconnect_steps (lower is better)")
    heatmap(agg, "safe_mean", "heatmap_safe_mode.png", "Mean safe_mode_activations (lower is better)")
    heatmap(agg, "pathlen_mean", "heatmap_pathlen.png", "Mean path length (cost)")

    print("[OK] saved heatmaps: heatmap_disconnect.png, heatmap_safe_mode.png, heatmap_pathlen.png")

if __name__ == "__main__":
    main()
