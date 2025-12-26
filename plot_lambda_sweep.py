import pandas as pd
import matplotlib.pyplot as plt


def main():
    csv_path = "belief_risk_lambda_sweep.csv"
    df = pd.read_csv(csv_path)
    print("[INFO] Loaded lambda sweep:")
    print(df)

    lambdas = df["lambda"].values
    fused_mean = df["fused_mean"].values
    total_cost = df["total_cost"].values
    geom_len = df["geometric_length"].values

    plt.figure(figsize=(8, 6))

    # Subplot 1: λ vs fused_mean
    plt.subplot(2, 1, 1)
    plt.plot(lambdas, fused_mean, marker="o")
    plt.xlabel("λ (risk weight)")
    plt.ylabel("mean fused uncertainty")
    plt.title("Effect of λ on mean fused uncertainty along path")
    plt.grid(True)

    # Subplot 2: λ vs total_cost (και geometric length)
    plt.subplot(2, 1, 2)
    plt.plot(lambdas, total_cost, marker="o", label="total cost")
    plt.plot(lambdas, geom_len, marker="x", linestyle="--", label="geometric length")
    plt.xlabel("λ (risk weight)")
    plt.ylabel("Cost / length")
    plt.title("Effect of λ on total A* cost and geometric length")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    out_img = "lambda_sweep_fused_risk.png"
    plt.savefig(out_img, dpi=200)
    print(f"[INFO] Saved figure to {out_img}")
    # plt.show()  # αν θέλεις να το ανοίγει και στην οθόνη


if __name__ == "__main__":
    main()
