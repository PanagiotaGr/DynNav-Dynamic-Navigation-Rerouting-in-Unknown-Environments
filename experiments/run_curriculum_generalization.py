import csv
import random
from pathlib import Path
from typing import Set, Tuple

import numpy as np
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


class DirectLearnedHeuristic:
    def __init__(self, model_path="heuristic_net.pt", stats_path="planner_dataset_norm_stats.npz", device="cpu", hidden=64):
        self.device = device
        self.model = HeuristicNet(in_dim=4, hidden=hidden)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        data = np.load(stats_path)
        self.mean = data["mean"]
        self.std = data["std"]

    def __call__(self, state, problem):
        x, y = state
        gx, gy = problem.goal
        feat = np.array([x, y, gx, gy], dtype=np.float32)
        feat = (feat - self.mean) / self.std
        inp = torch.tensor([feat], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            h = self.model(inp).squeeze().item()
        return max(0.0, float(h))


class CurriculumLearnedHeuristic:
    def __init__(self, model_path="heuristic_net_curriculum_8_16_32.pt",
                 stats_path="planner_dataset_norm_stats_curriculum.npz",
                 device="cpu", hidden=64):
        self.device = device
        self.model = HeuristicNet(in_dim=4, hidden=hidden)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        data = np.load(stats_path)
        self.mean = data["mean"]
        self.std = data["std"]

    def __call__(self, state, problem):
        x, y = state
        gx, gy = problem.goal
        feat = np.array([x, y, gx, gy], dtype=np.float32)
        feat = (feat - self.mean) / self.std
        inp = torch.tensor([feat], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            h = self.model(inp).squeeze().item()
        return max(0.0, float(h))


def run_curriculum_generalization(
    out_csv="phd_experiments/curriculum_generalization_results.csv",
    num_instances_per_size=50,
):
    random.seed(0)
    base_dir = Path(__file__).resolve().parent.parent
    out_path = base_dir / out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "instance_id",
        "grid_size",
        "method",
        "success",
        "expansions",
        "runtime_sec",
        "path_cost",
    ]

    direct_h = DirectLearnedHeuristic()
    curriculum_h = CurriculumLearnedHeuristic()

    grid_sizes = [32, 48]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        inst_id = 0

        for size in grid_sizes:
            for _ in range(num_instances_per_size):
                inst_id += 1
                problem = make_open_grid(size, size, density=0.1)

                # classic
                stats = astar_search(problem, manhattan_heuristic)
                writer.writerow({
                    "instance_id": inst_id,
                    "grid_size": size,
                    "method": "classic",
                    "success": int(stats.success),
                    "expansions": stats.expansions,
                    "runtime_sec": stats.runtime_sec,
                    "path_cost": stats.path_cost,
                })

                # direct learned
                stats = astar_search(problem, lambda s, p: direct_h(s, p))
                writer.writerow({
                    "instance_id": inst_id,
                    "grid_size": size,
                    "method": "learned_direct",
                    "success": int(stats.success),
                    "expansions": stats.expansions,
                    "runtime_sec": stats.runtime_sec,
                    "path_cost": stats.path_cost,
                })

                # curriculum learned
                stats = astar_search(problem, lambda s, p: curriculum_h(s, p))
                writer.writerow({
                    "instance_id": inst_id,
                    "grid_size": size,
                    "method": "learned_curriculum",
                    "success": int(stats.success),
                    "expansions": stats.expansions,
                    "runtime_sec": stats.runtime_sec,
                    "path_cost": stats.path_cost,
                })


if __name__ == "__main__":
    run_curriculum_generalization()
