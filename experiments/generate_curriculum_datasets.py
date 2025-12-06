import heapq
import random
from pathlib import Path
from typing import Set, Tuple, List

import numpy as np

from .grid_world import GridWorld


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


def make_open_grid(width: int, height: int, density: float) -> GridWorld:
    obstacles = generate_random_obstacles(width, height, density)
    start, goal = ensure_valid_start_goal(width, height, obstacles)
    return GridWorld(width, height, start, goal, obstacles)


def dijkstra_from_goal(problem: GridWorld):
    """
    Τρέχουμε Dijkstra από το goal προς όλα τα reachable states
    για να πάρουμε cost-to-go h*(s).
    """
    goal = problem.goal
    dist = {goal: 0.0}
    pq: List[Tuple[float, Tuple[int, int]]] = [(0.0, goal)]

    while pq:
        d, s = heapq.heappop(pq)
        if d > dist[s]:
            continue
        # αντιστρέφουμε τις ακμές: κοιτάμε predecessors
        # αλλά επειδή το GridWorld δίνει successors(s), μπορούμε να τρέξουμε προς τα "εμπρός"
        # αρκεί να θεωρήσουμε cost συμμετρικό.
        for (s_next, cost) in problem.successors(s):
            # εδώ πάμε "ανάποδα": s_next -> s
            alt = d + cost
            if s_next not in dist or alt < dist[s_next]:
                dist[s_next] = alt
                heapq.heappush(pq, (alt, s_next))
    return dist


def build_dataset_for_size(
    grid_size: int,
    num_instances: int,
    obstacle_density: float,
    out_path: Path,
    seed: int = 0,
):
    random.seed(seed)
    X_all = []
    y_all = []

    for inst in range(num_instances):
        problem = make_open_grid(grid_size, grid_size, density=obstacle_density)
        dist = dijkstra_from_goal(problem)

        gx, gy = problem.goal

        for (x, y), h_star in dist.items():
            X_all.append([x, y, gx, gy])
            y_all.append(h_star)

        print(f"[size={grid_size}] instance {inst+1}/{num_instances}: collected {len(dist)} states")

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, X=X, y=y)
    print(f"[size={grid_size}] Saved dataset to {out_path} with {len(X)} samples")


def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data_curriculum"
    data_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        (8,  200, 0.1),
        (16, 200, 0.1),
        (32, 200, 0.1),
    ]

    for grid_size, num_inst, density in configs:
        out_path = data_dir / f"planner_dataset_size{grid_size}.npz"
        build_dataset_for_size(
            grid_size=grid_size,
            num_instances=num_inst,
            obstacle_density=density,
            out_path=out_path,
            seed=grid_size,  # διαφορετικό seed per size
        )


if __name__ == "__main__":
    main()
