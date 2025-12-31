"""
plot_multi_robot_risk_results.py

Φτιάχνει boxplots για τα αποτελέσματα multi-robot risk-aware allocation από
το multi_robot_risk_results.csv.

Για κάθε metric (π.χ. total_distance, total_risk, max_risk, total_cost)
δημιουργεί ένα boxplot με άξονα x τις policies και άξονα y το metric.

Αποθηκεύει τα plots σε PNG αρχεία, π.χ.:
    boxplot_total_distance.png
    boxplot_total_risk.png
"""

import csv
from collections import defaultdict

import matplotlib.pyplot as plt


def load_results(input_csv: str):
    """
    Διαβάζει το CSV και επιστρέφει δομή:
        data[policy][metric] -> list[float]
    """
    data = defaultdict(lambda: defaultdict(list))

    with open(input_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            policy = row["policy"]
            # metrics που μας ενδιαφέρουν
            for metric_name in ["total_distance", "total_risk", "max_risk", "total_cost"]:
                value = float(row[metric_name])
                data[policy][metric_name].append(value)

    return data


def plot_boxplot_per_metric(
    data,
    metric_name: str,
    output_png: str,
    ylabel: str,
    title: str,
):
    """
    Δημιουργεί και αποθηκεύει boxplot για ένα metric.

    Parameters
    ----------
    data : dict[policy][metric] -> list[float]
    metric_name : str
        π.χ. "total_distance"
    output_png : str
        όνομα αρχείου εξόδου (π.χ. "boxplot_total_distance.png")
    ylabel : str
        label για τον άξονα y
    title : str
        τίτλος του γραφήματος
    """
    policies = sorted(data.keys())
    values_per_policy = [data[pol][metric_name] for pol in policies]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.boxplot(values_per_policy, labels=policies, showmeans=True)

    ax.set_xlabel("Policy")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)
    print(f"Saved {output_png}")


def main():
    input_csv = "multi_robot_risk_results.csv"
    data = load_results(input_csv)

    # 1) Boxplot για συνολική απόσταση
    plot_boxplot_per_metric(
        data,
        metric_name="total_distance",
        output_png="boxplot_total_distance.png",
        ylabel="Total distance (sum over robots)",
        title="Multi-robot total distance per policy",
    )

    # 2) Boxplot για συνολικό ρίσκο
    plot_boxplot_per_metric(
        data,
        metric_name="total_risk",
        output_png="boxplot_total_risk.png",
        ylabel="Total assigned risk",
        title="Multi-robot total risk per policy",
    )

    # 3) Boxplot για μέγιστο ρίσκο
    plot_boxplot_per_metric(
        data,
        metric_name="max_risk",
        output_png="boxplot_max_risk.png",
        ylabel="Max assigned risk",
        title="Maximum risk per trial and policy",
    )

    # 4) Boxplot για συνολικό cost
    plot_boxplot_per_metric(
        data,
        metric_name="total_cost",
        output_png="boxplot_total_cost.png",
        ylabel="Total cost (distance + risk)",
        title="Total cost per policy",
    )


if __name__ == "__main__":
    main()
