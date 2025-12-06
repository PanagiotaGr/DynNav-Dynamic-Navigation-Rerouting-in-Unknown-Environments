#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MI-based Next Best View Planner
Ubuntu 24.04 / ROS2 Jazzy compatible (pure Python module)

- Διαβάζει occupancy / uncertainty grid από CSV (με header)
- Υπολογίζει entropy-based information gain για υποψήφιες views
"""

import numpy as np
import argparse
import os


class MINBVPlanner:
    def __init__(self, delta=0.35, eps=1e-6):
        """
        :param delta: update strength (sensor model)
        :param eps: numerical stability for log operations
        """
        self.delta = delta
        self.eps = eps

    # --------------------------------------------------------
    # Entropy of a single probability or array of probabilities
    # --------------------------------------------------------
    def cell_entropy(self, p):
        p = np.clip(p, self.eps, 1.0 - self.eps)
        return -(p * np.log(p) + (1 - p) * np.log(1 - p))

    # --------------------------------------------------------
    # Expected entropy after observing a cell with a simplified
    # hit/miss inverse sensor model.
    # --------------------------------------------------------
    def expected_entropy_after_measurement(self, p):
        # approximate update for hit/miss
        p_hit = np.clip(p + self.delta, self.eps, 1.0 - self.eps)
        p_miss = np.clip(p - self.delta, self.eps, 1.0 - self.eps)

        # probability the sensor will "hit" (cell occupied)
        q_hit = p
        q_miss = 1 - p

        H_hit = self.cell_entropy(p_hit)
        H_miss = self.cell_entropy(p_miss)

        return q_hit * H_hit + q_miss * H_miss

    # --------------------------------------------------------
    # Information Gain for a given viewpoint
    # visible_cells: list of (i, j)
    # prob_grid: 2D numpy array [H, W] of probabilities
    # --------------------------------------------------------
    def information_gain_view(self, prob_grid, visible_cells):
        if len(visible_cells) == 0:
            return 0.0

        p_vals = np.array([prob_grid[i, j] for (i, j) in visible_cells])

        H_before = self.cell_entropy(p_vals)
        H_after = self.expected_entropy_after_measurement(p_vals)

        IG_cells = H_before - H_after
        return float(np.sum(IG_cells))

    # --------------------------------------------------------
    # Compute IG for a list of candidate viewpoints
    # views: list of dicts -> { "id": k, "visible": [(i,j), ...] }
    # returns: dict {view_id: IG_value}
    # --------------------------------------------------------
    def compute_IG_for_candidates(self, prob_grid, views):
        ig_dict = {}

        for view in views:
            view_id = view["id"]
            visible = view["visible"]
            ig = self.information_gain_view(prob_grid, visible)
            ig_dict[view_id] = ig

        return ig_dict


# ------------------------------------------------------------
# CSV utilities
# ------------------------------------------------------------

def load_prob_grid_from_csv(path, normalize=False, max_val=None):
    """
    Διαβάζει ένα CSV και το μετατρέπει σε probability grid.

    Περιμένουμε CSV με header (π.χ. 'cell_id,...') και μόνο αριθμητικά
    δεδομένα από τη 2η γραμμή και κάτω.

    :param path: path του CSV
    :param normalize: αν True, γίνεται normalization του grid σε [0,1]
    :param max_val: μέγιστη τιμή για normalization (αν None, παίρνει το max του grid)
    :return: prob_grid (numpy array σε [0,1])
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV file not found: {path}")

    try:
        # Αγνοούμε τη γραμμή header (π.χ. 'cell_id,...')
        grid = np.genfromtxt(path, delimiter=",", skip_header=1)
    except ValueError as e:
        raise ValueError(
            f"Failed to load CSV as numeric grid. Check the file format. Original error: {e}"
        ) from e

    if grid.ndim == 1:
        # Αν κατά λάθος φορτωθεί ως 1D, προσπάθησε reshape αν γίνεται
        n = grid.shape[0]
        side = int(np.sqrt(n))
        if side * side == n:
            grid = grid.reshape((side, side))
        else:
            raise ValueError(
                f"Loaded grid is 1D of length {n} και δεν μπορώ να το κάνω 2D τετράγωνο."
            )

    if normalize:
        if max_val is None:
            max_val = np.nanmax(grid)
        if max_val <= 0:
            raise ValueError("max_val must be > 0 for normalization.")
        prob_grid = grid / max_val
    else:
        prob_grid = grid

    # Clip σε [0,1] για να είναι valid probabilities
    prob_grid = np.clip(prob_grid, 1e-6, 1.0 - 1e-6)
    return prob_grid


# ------------------------------------------------------------
# Example usage from terminal
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MI-based NBV planner using CSV grid.")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path σε CSV (π.χ. coverage_grid_with_uncertainty.csv)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Αν δοθεί, γίνεται normalization του grid σε [0,1]",
    )
    parser.add_argument(
        "--max-val",
        type=float,
        default=None,
        help="Μέγιστη τιμή για normalization (αν δεν δοθεί, χρησιμοποιείται το max του grid)",
    )

    args = parser.parse_args()

    # Φόρτωσε grid από CSV
    prob_grid = load_prob_grid_from_csv(
        args.csv,
        normalize=args.normalize,
        max_val=args.max_val,
    )

    H, W = prob_grid.shape

    # DEMO views – εδώ αργότερα θα βάλεις τα πραγματικά visible cells
    demo_views = [
        {
            "id": "v1",
            "visible": [(i, j) for i in range(3, 6) for j in range(3, 6)
                        if 0 <= i < H and 0 <= j < W],
        },
        {
            "id": "v2",
            "visible": [(i, j) for i in range(6, 9) for j in range(1, 4)
                        if 0 <= i < H and 0 <= j < W],
        },
        {
            "id": "v3",
            "visible": [(i, j) for i in range(1, 4) for j in range(7, 10)
                        if 0 <= i < H and 0 <= j < W],
        },
    ]

    planner = MINBVPlanner(delta=0.35)
    ig_results = planner.compute_IG_for_candidates(prob_grid, demo_views)

    print(f"Loaded grid from: {args.csv}")
    print("Information Gain per view (demo):")
    for vid, val in ig_results.items():
        print(f"  {vid}: {val:.4f}")
