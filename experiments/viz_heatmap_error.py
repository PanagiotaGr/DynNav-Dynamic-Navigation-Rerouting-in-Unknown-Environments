import heapq
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from .grid_world import GridWorld
from .heuristics import manhattan_heuristic
from learned_heuristic import HeuristicNet


##############################################
# Dijkstra from goal → compute h*(s)
##############################################

def dijkstra_from_goal(problem: GridWorld):
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
# Learned heuristic wrapper
##############################################

class LearnedH:
    def __init__(self, model_path="heuristic_net.pt", stats="planner_dataset_norm_stats.npz"):
        data = np.load(stats)
        self.mean = data["mean"]
        self.std = data["std"]

        self.model = HeuristicNet(in_dim=4, hidden=64)
        self.model.load_state_dict(
            torch.load(model_path, map_location="cpu")
        )
        self.model.eval()

    def __call__(self, s, problem):
        x, y = s
        gx, gy = problem.goal
        feat = np.array([x, y, gx, gy], dtype=np.float32)
        feat = (feat - self.mean) / self.std
        inp = torch.tensor([feat], dtype=torch.float32)
        with torch.no_grad():
            h = self.model(inp).squeeze().item()
        return h


##############################################
# Heatmap helper
##############################################

def save_heatmap(matrix, title, out_file):
    plt.figure(figsize=(6,6))
    plt.imshow(matrix, cmap="viridis", origin="lower")
    plt.colorbar(label=title)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


##############################################
# Main visualization
##############################################

def viz_heatmap(grid_type="open", width=32, height=32, density=0.15):
    base = Path(__file__).resolve().parent.parent / "phd_experiments" / "heatmaps"
    base.mkdir(parents=True, exist_ok=True)

    # construct grid
    if grid_type == "open":
        obstacles = {(x,y) for x in range(width) for y in range(height)
                     if np.random.rand() < density}
        start = (0,0)
        goal = (width-1, height-1)
        obstacles.discard(start); obstacles.discard(goal)
        problem = GridWorld(width, height, start, goal, obstacles)
    else:
        raise NotImplementedError("Μπορούμε να προσθέσουμε maze/rooms αργότερα")

    # compute h*
    h_star_map = dijkstra_from_goal(problem)

    # compute predicted h
    learned = LearnedH()

    H_star = np.zeros((width, height), dtype=np.float32)
    H_pred = np.zeros((width, height), dtype=np.float32)
    H_err  = np.zeros((width, height), dtype=np.float32)

    for x in range(width):
        for y in range(height):
            s = (x,y)
            if s in problem.obstacles:
                H_star[x,y] = np.nan
                H_pred[x,y] = np.nan
                H_err[x,y]  = np.nan
                continue

            hstar = h_star_map.get(s, np.nan)
            hpred = learned(s, problem)

            H_star[x,y] = hstar
            H_pred[x,y] = hpred
            H_err[x,y]  = hpred - hstar if not np.isnan(hstar) else np.nan

    # Save heatmaps
    save_heatmap(H_star, "h*(s)", base / f"{grid_type}_hstar.png")
    save_heatmap(H_pred, "h_pred(s)", base / f"{grid_type}_hpred.png")
    save_heatmap(H_err,  "error(s)",  base / f"{grid_type}_herror.png")

    print("[INFO] Saved heatmaps to:", base)


if __name__ == "__main__":
    viz_heatmap()
