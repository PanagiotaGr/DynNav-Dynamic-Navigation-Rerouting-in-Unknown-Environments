import csv
import random
from pathlib import Path
from typing import Set, Tuple

from .astar_core import astar_search
from .grid_world import GridWorld
from .heuristics import manhattan_heuristic
from .heuristics_residual import ResidualLearnedHeuristic


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
    obstacles.discard(start)
    obstacles.discard(goal)
    return start, goal


def make_open_grid(width: int, height: int, density: float = 0.1) -> GridWorld:
    obstacles = generate_random_obstacles(width, height, density)
    start, goal = ensure_valid_start_goal(width, height, obstacles)
    return GridWorld(width, height, start, goal, obstacles)


def make_maze_like_grid(width: int, height: int) -> GridWorld:
    obstacles = generate_random_obstacles(width, height, density=0.3)
    start, goal = ensure_valid_start_goal(width, height, obstacles)
    return GridWorld(width, height, start, goal, obstacles)


def make_rooms_grid(width: int, height: int) -> GridWorld:
    obstacles = set()
    for x in range(0, width, 4):
        for y in range(height):
            if y % 3 != 0:
                obstacles.add((x, y))
    start, goal = ensure_valid_start_goal(width, height, obstacles)
    return GridWorld(width, height, start, goal, obstacles)


def run_residual_experiments(
    out_csv: str = "phd_experiments/astar_residual_results.csv",
    num_instances: int = 50,
    width: int = 32,
    height: int = 32,
    residual_model_path: str = "heuristic_net_residual.pt",
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

        residual_h = ResidualLearnedHeuristic(
            model_path=residual_model_path,
            stats_path="planner_dataset_norm_stats_residual.npz",
            device="cpu",
            hidden=64,
        )
        print(f"[INFO] Loaded residual heuristic from {residual_model_path}")

        inst_id = 0
        for grid_type in grid_types:
            for _ in range(num_instances):
                inst_id += 1

                if grid_type == "open":
                    problem = make_open_grid(width, height, density=0.1)
                elif grid_type == "maze":
                    problem = make_maze_like_grid(width, height)
                elif grid_type == "rooms":
                    problem = make_rooms_grid(width, height)
                else:
                    raise ValueError(f"Unknown grid_type {grid_type}")

                # classic
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

                # residual-learned
                stats = astar_search(problem, lambda s, p: residual_h(s, p))
                writer.writerow({
                    "instance_id": inst_id,
                    "grid_type": grid_type,
                    "method": "residual_learned",
                    "success": int(stats.success),
                    "expansions": stats.expansions,
                    "runtime_sec": stats.runtime_sec,
                    "path_cost": stats.path_cost,
                })


if __name__ == "__main__":
    run_residual_experiments()
