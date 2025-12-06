import csv
import random
from pathlib import Path
from typing import Set, Tuple

import torch

from .astar_core import astar_search
from .grid_world import GridWorld
from .heuristics import manhattan_heuristic
from learned_heuristic import HeuristicNet


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


class LearnedHeuristicFraction:
    def __init__(self, model_path: str, hidden: int = 64, device: str = "cpu"):
        self.device = device
        self.model = HeuristicNet(in_dim=4, hidden=hidden)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    def __call__(self, state, problem):
        x, y = state
        gx, gy = problem.goal
        inp = torch.tensor([[x, y, gx, gy]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            h_pred = self.model(inp).squeeze().item()
        return max(0.0, float(h_pred))


def run_data_sweep(
    out_csv: str = "phd_experiments/data_sweep_results.csv",
    num_instances: int = 50,
    width: int = 32,
    height: int = 32,
):
    random.seed(0)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "instance_id",
        "method",
        "fraction",
        "success",
        "expansions",
        "runtime_sec",
        "path_cost",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        fractions = [0.1, 0.25, 0.5, 1.0]
        wrappers = []
        for frac in fractions:
            path = f"heuristic_net_frac_{int(frac*100)}.pt"
            try:
                wrappers.append((frac, LearnedHeuristicFraction(path, hidden=64)))
                print(f"[INFO] Loaded model {path}")
            except Exception as e:
                print(f"[WARNING] Could not load {path}: {e}")

        inst_id = 0
        for _ in range(num_instances):
            inst_id += 1
            problem = make_open_grid(width, height, density=0.1)

            # classic
            stats = astar_search(problem, manhattan_heuristic)
            writer.writerow({
                "instance_id": inst_id,
                "method": "classic",
                "fraction": 1.0,
                "success": int(stats.success),
                "expansions": stats.expansions,
                "runtime_sec": stats.runtime_sec,
                "path_cost": stats.path_cost,
            })

            for frac, wrapper in wrappers:
                stats = astar_search(problem, lambda s, p: wrapper(s, p))
                writer.writerow({
                    "instance_id": inst_id,
                    "method": f"learned_frac_{int(frac*100)}",
                    "fraction": frac,
                    "success": int(stats.success),
                    "expansions": stats.expansions,
                    "runtime_sec": stats.runtime_sec,
                    "path_cost": stats.path_cost,
                })


if __name__ == "__main__":
    run_data_sweep()
