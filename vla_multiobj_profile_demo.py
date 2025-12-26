"""
vla_multiobj_profile_demo.py

Demo: Εφαρμόζουμε διαφορετικά VLA intents σε multi-objective
υποψηφίους (multiobj_candidates.csv) και αλλάζουμε το ranking των paths.

Χρησιμοποιεί:
- path length
- fused risk / uncertainty
- entropy / IG
- coverage

και τα συνδυάζει με βάρη από vla_risk_profiles.py.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

from vla_risk_profiles import get_planner_profile, PlannerWeights


def load_candidates(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found.")

    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"{csv_path} is empty.")
    return rows


def extract_metric(
    row: Dict[str, str],
    candidates,
    default: float,
) -> float:
    """Βρίσκει την πρώτη διαθέσιμη στήλη από τη λίστα candidates."""
    for key in candidates:
        if key in row and row[key] != "":
            try:
                return float(row[key])
            except ValueError:
                continue
    return float(default)


def normalize(values: List[float]) -> List[float]:
    """Απλή min-max κανονικοποίηση σε [0, 1]."""
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax - vmin < 1e-9:
        # όλα ίδια -> γύρνα 0.5
        return [0.5 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def compute_scores(
    rows: List[Dict[str, str]],
    weights: PlannerWeights,
) -> List[Tuple[float, Dict[str, str]]]:
    """
    Υπολογίζει score για κάθε candidate:
    - Θέλουμε:
        * μικρό length, μικρό risk
        * μεγάλο entropy, μεγάλο coverage
    - Για "καλό" score:
        * length_score = 1 - norm_length
        * risk_score = 1 - norm_risk
        * entropy_score = norm_entropy
        * coverage_score = norm_coverage
    """

    # Συλλογή raw metrics
    lengths = []
    risks = []
    entropies = []
    coverages = []

    for row in rows:
        length = extract_metric(
            row,
            ["geom_length", "path_length", "length_cells", "L_geom"],
            default=0.0,
        )
        risk = extract_metric(
            row,
            ["fused_risk", "risk_fused", "R_fused", "risk", "fused_sum", "fused_mean"],
            default=0.0,
        )
        entropy = extract_metric(
            row,
            ["entropy", "entropy_reduction", "ig", "information_gain"],
            default=0.0,
        )
        coverage = extract_metric(
            row,
            ["coverage", "coverage_percent", "coverage_ratio"],
            default=0.0,
        )

        lengths.append(length)
        risks.append(risk)
        entropies.append(entropy)
        coverages.append(coverage)

    # Κανονικοποίηση
    n_length = normalize(lengths)
    n_risk = normalize(risks)
    n_entropy = normalize(entropies)
    n_cov = normalize(coverages)

    scores_with_rows: List[Tuple[float, Dict[str, str]]] = []

    for idx, row in enumerate(rows):
        length_score = 1.0 - n_length[idx]   # μικρότερο length -> μεγαλύτερο score
        risk_score = 1.0 - n_risk[idx]       # μικρότερο risk -> μεγαλύτερο score
        entropy_score = n_entropy[idx]       # μεγαλύτερο entropy -> μεγαλύτερο score
        cov_score = n_cov[idx]               # μεγαλύτερο coverage -> μεγαλύτερο score

        score = (
            weights.length_weight * length_score
            + weights.risk_weight * risk_score
            + weights.entropy_weight * entropy_score
            + weights.coverage_weight * cov_score
        )

        scores_with_rows.append((score, row))

    return scores_with_rows


def main():
    parser = argparse.ArgumentParser(
        description="VLA intent-based multi-objective ranking demo."
    )
    parser.add_argument(
        "--intent",
        type=str,
        default="cautious_navigation",
        help="High-level intent (e.g. cautious_navigation, fast_navigation, exploration_focused, localization_focused).",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="multiobj_candidates.csv",
        help="Input CSV file with multi-objective candidates.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/multiobj_ranked.csv",
        help="Output CSV file with scores and sorted candidates.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="How many top candidates to print.",
    )
    args = parser.parse_args()

    intent = args.intent.strip().lower()
    weights = get_planner_profile(intent)
    print(f"[INFO] Using intent '{intent}' with weights: {weights}")

    input_path = Path(args.input_csv)
    rows = load_candidates(input_path)

    scores_with_rows = compute_scores(rows, weights)

    # sort descending by score (μεγαλύτερο score -> καλύτερο)
    scores_with_rows.sort(key=lambda x: x[0], reverse=True)

    # Εκτύπωση top-k
    print(f"[INFO] Top {args.top_k} candidates for intent '{intent}':")
    for rank, (score, row) in enumerate(scores_with_rows[: args.top_k], start=1):
        length = extract_metric(
            row,
            ["geom_length", "path_length", "length_cells", "L_geom"],
            default=0.0,
        )
        risk = extract_metric(
            row,
            ["fused_risk", "risk_fused", "R_fused", "risk", "fused_sum", "fused_mean"],
            default=0.0,
        )
        entropy = extract_metric(
            row,
            ["entropy", "entropy_reduction", "ig", "information_gain"],
            default=0.0,
        )
        coverage = extract_metric(
            row,
            ["coverage", "coverage_percent", "coverage_ratio"],
            default=0.0,
        )

        print(
            f"  #{rank:02d}  score={score:.3f}  "
            f"length={length:.3f}, risk={risk:.3f}, "
            f"entropy={entropy:.3f}, coverage={coverage:.3f}"
        )

    # Γράφουμε νέο CSV με extra στήλη "vla_score"
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Χρησιμοποιούμε τα ίδια fieldnames + μία νέα στήλη
    fieldnames = list(rows[0].keys())
    if "vla_score" not in fieldnames:
        fieldnames.append("vla_score")
    if "vla_intent" not in fieldnames:
        fieldnames.append("vla_intent")

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for score, row in scores_with_rows:
            row = dict(row)  # copy
            row["vla_score"] = f"{score:.6f}"
            row["vla_intent"] = intent
            writer.writerow(row)

    print(f"[INFO] Saved ranked candidates to {output_path}")


if __name__ == "__main__":
    main()
