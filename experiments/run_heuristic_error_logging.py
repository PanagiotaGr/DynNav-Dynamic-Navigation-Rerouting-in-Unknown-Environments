import csv
import heapq
import random
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import torch

from .grid_world import GridWorld
from .heuristics import manhattan_heuristic
from learned_heuristic import HeuristicNet


##############################################
# Compute h*(s) via Dijkstra from goal
##############################################

def dijkstra_from_goal(problem: GridWorld) -> Dict[Tuple[int,int], float]:
    goal = problem.goal
    dist = {goal: 0.0}
    pq = [(0.0, goal)]

    while pq:
        d, s = heapq.heappop(pq)
        if d > dist[s]:
            continue
        for s_next, cost in problem.successors(s):
            alt = d + cost
            if s_next not in dist or alt < dist[s_next]:
                dist[s_next] = alt
                heapq.heappush(pq, (alt, s_next))
    return dist


##############################################
# Build grid for testing
##############################################

def generate_random_obstacles(width, height, density=0.1):
    obs = set()
    for x in range(width):
        for y in range(height):
            if random.random() < density:
                obs.add((x, y))
    return obs


def ensure_valid_start_goal(width, height, obstacles):
    start = (0,0)
    goal = (width-1, height-1)
    obstacles.discard(start)
    obstacles.discard(goal)
    return start, goal


def make_open_grid(width, height, density=0.1):
    obs = generate_random_obstacles(width, height, density)
    start, goal = ensure_valid_start_goal(width, height, obs)
    return GridWorld(width, height, start, goal, obs)


##############################################
# Predict h(s)
##############################################

class LearnedHWrapper:
    def __init__(self, model_path="heuristic_net.pt", stats_path="planner_dataset_norm_stats.npz", hidden=64):
        data = np.load(stats_path)
        self.mean = data["mean"]
        self.std = data["std"]

        self.model = HeuristicNet(in_dim=4, hidden=hidden)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def __call__(self, s, problem):
        x,y = s
        gx,gy = problem.goal
        feat = np.array([x,y,gx,gy], dtype=np.float32)
        feat = (feat - self.mean) / self.std
        inp = torch.tensor([feat], dtype=torch.float32)
        with torch.no_grad():
            h = self.model(inp).squeeze().item()
        return max(0.0, float(h))


##############################################
# Main experiment
##############################################

def run_error_logging(
    out_csv="phd_experiments/heuristic_error_data.csv",
    num_instances=50,
    width=32,
    height=32
):
    base_dir = Path(__file__).resolve().parent.parent
    out_path = base_dir / out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)

    learned_h = LearnedHWrapper()

    fieldnames = [
        "instance_id",
        "state_x", "state_y",
        "h_star", "h_pred",
        "error", "abs_error", "rel_error",
        "f_pred",
        "grid_size"
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        inst_id = 0

        for _ in range(num_instances):
            inst_id += 1

            problem = make_open_grid(width, height, density=0.1)
            h_star_map = dijkstra_from_goal(problem)

            for (x,y), h_star in h_star_map.items():
                h_pred = learned_h((x,y), problem)

                err = h_pred - h_star
                abs_err = abs(err)
                rel_err = err / (h_star + 1e-6)

                f_pred = problem.cost((0,0),(x,y)) + h_pred if False else h_pred  # optional

                writer.writerow({
                    "instance_id": inst_id,
                    "state_x": x,
                    "state_y": y,
                    "h_star": h_star,
                    "h_pred": h_pred,
                    "error": err,
                    "abs_error": abs_err,
                    "rel_error": rel_err,
                    "f_pred": f_pred,
                    "grid_size": width
                })

    print(f"[INFO] Saved error log to {out_path}")


if __name__ == "__main__":
    run_error_logging()
