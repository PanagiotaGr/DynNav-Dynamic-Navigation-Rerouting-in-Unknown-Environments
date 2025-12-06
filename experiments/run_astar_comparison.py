import csv
import random
from pathlib import Path
from typing import Set, Tuple

from .astar_core import astar_search
from .grid_world import GridWorld
from .heuristics import (
    manhattan_heuristic,
    LearnedHeuristic,
    UncertaintyAwareHeuristic,
    OnlineUpdateHeuristic,
)
from .online_astar import astar_with_online_updates


# ----------- Grid generators ----------- #


def generate_random_obstacles(width: int, height: int, density: float) -> Set[Tuple[int, int]]:
    obstacles = set()
    for x in range(width):
        for y in range(height):
            if random.random() < density:
                obstacles.add((x, y))
    return obstacles


def ensure_valid_start_goal(width, height, obstacles):
    start = (0, 0)
    goal = (width - 1, height - 1)
    if start in obstacles:
        obstacles.remove(start)
    if goal in obstacles:
        obstacles.remove(goal)
    return start, goal


def make_open_grid(width: int, height: int, density: float = 0.1) -> GridWorld:
    obstacles = generate_random_obstacles(width, height, density)
    start, goal = ensure_valid_start_goal(width, height, obstacles)
    return GridWorld(width, height, start, goal, obstacles)


def make_maze_like_grid(width: int, height: int) -> GridWorld:
    # προσωρινά: λίγο πιο πυκνά εμπόδια
    obstacles = generate_random_obstacles(width, height, density=0.3)
    start, goal = ensure_valid_start_goal(width, height, obstacles)
    return GridWorld(width, height, start, goal, obstacles)


def make_rooms_grid(width: int, height: int) -> GridWorld:
    obstacles = set()
    # απλό pattern "τοίχων" για rooms
    for x in range(0, width, 4):
        for y in range(height):
            if y % 3 != 0:
                obstacles.add((x, y))
    start, goal = ensure_valid_start_goal(width, height, obstacles)
    return GridWorld(width, height, start, goal, obstacles)


# ----------- Main experiment ----------- #


def run_experiments(
    out_csv: str = "phd_experiments/astar_phd_results.csv",
    num_instances: int = 50,
    width: int = 32,
    height: int = 32,
    learned_model_path: str = "heuristic_net.pt",
    learned_unc_model_path: str = "heuristic_net_unc.pt",
):
    random.seed(0)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "instance_id",
        "grid_type",
        "method",
        "success",
        "expansions",
        "runtime_sec",
        "path_cost",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        grid_types = ["open", "maze", "rooms"]

        # load learned net
        learned_h = None
        try:
            learned_h = LearnedHeuristic(learned_model_path)
            print(f"[INFO] Loaded learned heuristic from {learned_model_path}")
        except Exception as e:
            print("[WARNING] Could not load learned model:", e)

        # load uncertainty net (we will vary k later)
        unc_h = None
        k_values = [0.0, 1.0, 2.0, 3.0]
        try:
            unc_h = UncertaintyAwareHeuristic(learned_unc_model_path, k=1.0)
            print(f"[INFO] Loaded uncertainty-aware heuristic from {learned_unc_model_path}")
        except Exception as e:
            print("[WARNING] Could not load uncertainty-aware model:", e)

        inst_id = 0

        for grid_type in grid_types:
            for _ in range(num_instances):
                inst_id += 1

                # φτιάχνουμε το κατάλληλο πρόβλημα
                if grid_type == "open":
                    problem = make_open_grid(width, height, density=0.1)
                elif grid_type == "maze":
                    problem = make_maze_like_grid(width, height)
                elif grid_type == "rooms":
                    problem = make_rooms_grid(width, height)
                else:
                    raise ValueError(f"Unknown grid_type {grid_type}")

                # 1) Classic A*
                stats = astar_search(problem, manhattan_heuristic)
                writer.writerow({
                    "instance_id": inst_id,
                    "grid_type": grid_type,
                    "method": "classic",
                    "success": int(stats.success),
                    "expansions": stats.expansions,
                    "runtime_sec": stats.runtime_sec,
                    "path_cost": stats.path_cost,
                })

                # 2) Learned A*
                if learned_h is not None:
                    stats = astar_search(problem, lambda s, p: learned_h(s, p))
                    writer.writerow({
                        "instance_id": inst_id,
                        "grid_type": grid_type,
                        "method": "learned",
                        "success": int(stats.success),
                        "expansions": stats.expansions,
                        "runtime_sec": stats.runtime_sec,
                        "path_cost": stats.path_cost,
                    })

                # 3) Uncertainty-aware A* για διάφορα k
                if unc_h is not None:
                    for k in k_values:
                        unc_h.k = k
                        stats = astar_search(problem, lambda s, p: unc_h(s, p))
                        writer.writerow({
                            "instance_id": inst_id,
                            "grid_type": grid_type,
                            "method": f"uncertainty_k{k:g}",
                            "success": int(stats.success),
                            "expansions": stats.expansions,
                            "runtime_sec": stats.runtime_sec,
                            "path_cost": stats.path_cost,
                        })

                # 4) Online update A*
                online_h = OnlineUpdateHeuristic(manhattan_heuristic, eta=0.1)
                stats = astar_with_online_updates(problem, online_h)
                writer.writerow({
                    "instance_id": inst_id,
                    "grid_type": grid_type,
                    "method": "online_TD",
                    "success": int(stats.success),
                    "expansions": stats.expansions,
                    "runtime_sec": stats.runtime_sec,
                    "path_cost": stats.path_cost,
                })


if __name__ == "__main__":
    run_experiments(
        out_csv="phd_experiments/astar_phd_results.csv",
        num_instances=50,
        width=32,
        height=32,
    )
