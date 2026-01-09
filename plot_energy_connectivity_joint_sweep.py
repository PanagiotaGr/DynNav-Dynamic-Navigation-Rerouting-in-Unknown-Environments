import pandas as pd
import matplotlib.pyplot as plt

def main(csv_path="energy_connectivity_joint_sweep.csv"):
    df = pd.read_csv(csv_path)
    ok = df[df["status"]=="ok"].copy()

    agg = ok.groupby(["delta","gamma"]).agg(
        path_len_mean=("path_len","mean"),
        safe_act_rate=("safe_mode_activations","mean"),
        disconnect_mean=("disconnect_steps","mean"),
        replans_mean=("replans","mean"),
    ).reset_index()

    print(agg.head(20))

    # For each delta: plot disconnect vs gamma
    for d in sorted(agg["delta"].unique()):
        sub = agg[agg["delta"]==d].sort_values("gamma")
        plt.figure()
        plt.plot(sub["gamma"], sub["disconnect_mean"], marker="o")
        plt.xlabel("gamma (connectivity weight)")
        plt.ylabel("mean disconnect_steps")
        plt.title(f"Disconnect steps vs gamma (delta={d})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"joint_disconnect_vs_gamma_delta_{d}.png", dpi=200)

    # Safe mode activation vs gamma for delta=1.0 (example)
    d0 = 1.0
    sub = agg[agg["delta"]==d0].sort_values("gamma")
    if len(sub) > 0:
        plt.figure()
        plt.plot(sub["gamma"], sub["safe_act_rate"], marker="o")
        plt.xlabel("gamma (connectivity weight)")
        plt.ylabel("mean safe_mode_activations")
        plt.title(f"Safe mode activations vs gamma (delta={d0})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"joint_safe_mode_vs_gamma_delta_{d0}.png", dpi=200)

    print("[OK] saved joint plots: joint_disconnect_vs_gamma_delta_*.png and joint_safe_mode_vs_gamma_delta_1.0.png")

if __name__ == "__main__":
    main()
