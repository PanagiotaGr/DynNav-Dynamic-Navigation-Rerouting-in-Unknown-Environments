"""
multi_robot_coverage_sim.py

Simple multi-robot cooperative coverage simulation over an uncertainty / priority grid.

- Διαβάζει grid από CSV (π.χ. coverage_grid_with_uncertainty.csv ή priority_field.csv).
- Προσομοιώνει N ρομπότ που κινούνται σε discrete cells.
- Σε κάθε βήμα, κάθε ρομπότ επιλέγει έναν στόχο υψηλής προτεραιότητας
  με penalty στην απόσταση και αποφυγή conflicts με άλλα ρομπότ.
- Η "κίνηση" γίνεται εδώ απλοποιημένα ως τηλεμεταφορά στο στόχο (no path planning),
  αλλά το μοντέλο είναι αρκετό για να μελετήσεις cooperative policies.

Αποθηκεύει:
- per-step metrics σε CSV (coverage, mean priority, κτλ)
- per-robot trajectories σε CSV.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import math
import random


@dataclass
class Cell:
    x: int
    y: int
    priority: float      # υψηλότερο = καλύτερο για exploration
    covered: bool = False


@dataclass
class Robot:
    robot_id: int
    x: int
    y: int


def load_grid_from_csv(csv_path: Path) -> Tuple[List[Cell], int, int]:
    """
    Φορτώνει grid από CSV.

    Περιμένουμε στήλες π.χ.:
        - x, y (ή i, j, row, col)
        - priority (ή weight, score, uncertainty, entropy, coverage_priority)

    Αν δεν βρει priority, βάζει 1.0 παντού (ισοδύναμα cells).

    Επιστρέφει:
        - λίστα από Cell
        - width, height του grid (max+1)
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found.")

    cells: List[Cell] = []

    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"{csv_path} is empty.")

    # εντοπίζουμε ονόματα στηλών για x, y
    sample = rows[0]
    x_key_candidates = ["x", "i", "col", "j"]
    y_key_candidates = ["y", "j", "row", "i"]

    def find_key(cands):
        for c in cands:
            if c in sample:
                return c
        raise KeyError(f"Could not find any of {cands} in CSV headers: {sample.keys()}")

    x_key = find_key(x_key_candidates)
    y_key = find_key(y_key_candidates)

    # priority column
    priority_candidates = [
        "priority",
        "priority_value",
        "weight",
        "score",
        "uncertainty",
        "entropy",
        "coverage_priority",
    ]
    priority_key: Optional[str] = None
    for c in priority_candidates:
        if c in sample:
            priority_key = c
            break

    width = 0
    height = 0

    for row in rows:
        x = int(float(row[x_key]))
        y = int(float(row[y_key]))

        if priority_key is not None and row[priority_key] != "":
            try:
                p = float(row[priority_key])
            except ValueError:
                p = 1.0
        else:
            p = 1.0

        cells.append(Cell(x=x, y=y, priority=p, covered=False))

        width = max(width, x + 1)
        height = max(height, y + 1)

    return cells, width, height


def initialize_robots(num_robots: int, width: int, height: int) -> List[Robot]:
    """
    Αρχικοποίηση ρομπότ σε τυχαίες θέσεις στο grid.
    """
    robots: List[Robot] = []
    for rid in range(num_robots):
        # επιλέγουμε τυχαία cell στο grid
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        robots.append(Robot(robot_id=rid, x=x, y=y))
    return robots


def manhattan_distance(x1: int, y1: int, x2: int, y2: int) -> float:
    return abs(x1 - x2) + abs(y1 - y2)


def choose_goal_for_robot(
    robot: Robot,
    cells: List[Cell],
    reserved_targets: List[Tuple[int, int]],
    alpha_dist: float = 0.1,
) -> Optional[Cell]:
    """
    Επιλέγει cell υψηλής προτεραιότητας για το ρομπότ:

    objective ~= priority - alpha_dist * distance

    - Χρησιμοποιεί Manhattan distance.
    - Αποφεύγει cells που είναι ήδη reserved από άλλο ρομπότ
      σε αυτό το time step (reserved_targets).
    - Αγνοεί ήδη covered cells (priority effectively ~ 0).
    """

    best_cell: Optional[Cell] = None
    best_score = -math.inf

    for c in cells:
        if c.covered:
            continue
        if (c.x, c.y) in reserved_targets:
            continue

        dist = manhattan_distance(robot.x, robot.y, c.x, c.y)
        score = c.priority - alpha_dist * dist

        if score > best_score:
            best_score = score
            best_cell = c

    return best_cell


def compute_global_metrics(cells: List[Cell]) -> Dict[str, float]:
    """
    Υπολογίζει απλά global metrics:
    - coverage_ratio: ποσοστό covered cells
    - mean_priority: μέση priority (μηδενική για covered)
    """
    if not cells:
        return {"coverage_ratio": 0.0, "mean_priority": 0.0}

    total_cells = len(cells)
    covered_cells = sum(1 for c in cells if c.covered)

    # μέση "υπόλοιπη" priority
    total_priority = sum(c.priority for c in cells if not c.covered)
    mean_priority = total_priority / max(1, total_cells - covered_cells)

    coverage_ratio = covered_cells / total_cells

    return {
        "coverage_ratio": coverage_ratio,
        "mean_priority": mean_priority,
    }


def run_multi_robot_simulation(
    grid_csv: Path,
    num_robots: int,
    num_steps: int,
    alpha_dist: float,
    metrics_csv: Path,
    traj_csv: Path,
) -> None:
    """
    Κύρια συνάρτηση simulation.

    - Φορτώνει grid.
    - Τρέχει για num_steps.
    - Σε κάθε βήμα:
        * κάθε ρομπότ επιλέγει στόχο (χωρίς conflicts),
        * "πηγαίνει" εκεί και καλύπτει το κελί (covered = True).
    - Αποθηκεύει:
        * metrics ανά step,
        * trajectories ανά robot & step.
    """

    cells, width, height = load_grid_from_csv(grid_csv)
    print(f"[INFO] Loaded grid from {grid_csv} with size {width} x {height} "
          f"and {len(cells)} cells.")

    robots = initialize_robots(num_robots, width, height)
    print(f"[INFO] Initialized {num_robots} robots.")

    # Δημιουργία CSV writers
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    traj_csv.parent.mkdir(parents=True, exist_ok=True)

    with metrics_csv.open("w", newline="") as fm, traj_csv.open("w", newline="") as ft:
        metrics_writer = csv.DictWriter(
            fm,
            fieldnames=[
                "step",
                "coverage_ratio",
                "mean_priority",
            ],
        )
        metrics_writer.writeheader()

        traj_writer = csv.DictWriter(
            ft,
            fieldnames=[
                "step",
                "robot_id",
                "x",
                "y",
                "target_x",
                "target_y",
            ],
        )
        traj_writer.writeheader()

        for step in range(num_steps):
            reserved_targets: List[Tuple[int, int]] = []

            # Each robot chooses a target (sequentially → greedy assignment)
            for robot in robots:
                target_cell = choose_goal_for_robot(
                    robot,
                    cells,
                    reserved_targets,
                    alpha_dist=alpha_dist,
                )

                if target_cell is None:
                    # No more uncovered high-priority cells
                    traj_writer.writerow(
                        {
                            "step": step,
                            "robot_id": robot.robot_id,
                            "x": robot.x,
                            "y": robot.y,
                            "target_x": robot.x,
                            "target_y": robot.y,
                        }
                    )
                    continue

                # Reserve target to avoid conflicts
                reserved_targets.append((target_cell.x, target_cell.y))

            # Now "move" robots to their reserved targets
            # (we match in the same order)
            idx = 0
            for robot in robots:
                if idx < len(reserved_targets):
                    tx, ty = reserved_targets[idx]
                    idx += 1
                else:
                    tx, ty = robot.x, robot.y

                # Move robot
                robot.x, robot.y = tx, ty

                # Mark cell as covered
                for c in cells:
                    if c.x == tx and c.y == ty:
                        c.covered = True
                        break

                traj_writer.writerow(
                    {
                        "step": step,
                        "robot_id": robot.robot_id,
                        "x": robot.x,
                        "y": robot.y,
                        "target_x": tx,
                        "target_y": ty,
                    }
                )

            # Compute and log global metrics
            metrics = compute_global_metrics(cells)
            metrics_writer.writerow(
                {
                    "step": step,
                    "coverage_ratio": metrics["coverage_ratio"],
                    "mean_priority": metrics["mean_priority"],
                }
            )

            if step % 10 == 0 or step == num_steps - 1:
                print(
                    f"[Step {step}] coverage={metrics['coverage_ratio']:.3f}, "
                    f"mean_priority={metrics['mean_priority']:.3f}"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-robot cooperative coverage simulation."
    )
    parser.add_argument(
        "--grid_csv",
        type=str,
        default="coverage_grid_with_uncertainty.csv",
        help="Grid CSV file (e.g. coverage_grid_with_uncertainty.csv or priority_field.csv).",
    )
    parser.add_argument(
        "--num_robots",
        type=int,
        default=2,
        help="Number of robots in the simulation.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of simulation steps.",
    )
    parser.add_argument(
        "--alpha_dist",
        type=float,
        default=0.1,
        help="Distance penalty coefficient in goal selection.",
    )
    parser.add_argument(
        "--metrics_csv",
        type=str,
        default="results/multi_robot_metrics.csv",
        help="Output CSV for global coverage metrics.",
    )
    parser.add_argument(
        "--traj_csv",
        type=str,
        default="results/multi_robot_trajectories.csv",
        help="Output CSV for robot trajectories.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_multi_robot_simulation(
        grid_csv=Path(args.grid_csv),
        num_robots=args.num_robots,
        num_steps=args.steps,
        alpha_dist=args.alpha_dist,
        metrics_csv=Path(args.metrics_csv),
        traj_csv=Path(args.traj_csv),
    )
