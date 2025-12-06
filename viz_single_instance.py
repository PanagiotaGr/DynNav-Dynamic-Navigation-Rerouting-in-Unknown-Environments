import random
from pathlib import Path
from typing import List, Tuple, Set, Any

import matplotlib.pyplot as plt

from .grid_world import GridWorld
from .heuristics import (
    manhattan_heuristic,
    LearnedHeuristic,
    UncertaintyAwareHeuristic,
)
from .astar_core import SearchStats
from .online_astar import astar_with_online_updates
from .heuristics import OnlineUpdateHeuristic


# Μικρό A* με trace για visualization
def astar_with_trace(problem, heuristic_fn):
    import heapq
    import time

    start_time = time.time()
    start = problem.get_start()
    goal_test = problem.is_goal

    open_heap = []
    g = {start: 0.0}
    came_from = {}
    closed = set()
    expansions = 0
    tie_breaker = 0

    def reconstruct_path(came_from_, goal_):
        path = [goal_]
        while goal_ in came_from_:
            goal_ = came_from_[goal_]
            path.append(goal_)
        path.reverse()
        return path

    h0 = heuristic_fn(start, problem)
    heapq.heappush(open_heap, (g[start] + h0, tie_breaker, start))

    expanded_order: List[Any] = []
    best_goal_cost = None
    best_goal_state = None

    while open_heap:
        f, _, s = heapq.heappop(open_heap)
        if s in closed:
            continue
        closed.add(s)
        expansions += 1
        expanded_order.append(s)

        if goal_test(s):
            best_goal_cost = g[s]
            best_goal_state = s
            break

        for (s_next, cost) in problem.successors(s):
            g_new = g[s] + cost
            if s_next not in g or g_new < g[s_next]:
                g[s_next] = g_new
                came_from[s_next] = s
                h = heuristic_fn(s_next, problem)
                tie_breaker += 1
                heapq.heappush(open_heap, (g_new + h, tie_breaker, s_next))

    runtime_sec = time.time() - start_time

    if best_goal_state is None:
        stats = SearchStats(False, [], float("inf"), expansions, runtime_sec)
        return stats, expanded_order

    path = reconstruct_path(came_from, best_goal_state)
    stats = SearchStats(True, path, best_goal_cost, expansions, runtime_sec)
    return stats, expanded_order


def generate_open_grid(width: int, height: int, density: float = 0.1) -> GridWorld:
    obstacles: Set[Tuple[int, int]] = set()
    for x in range(width):
        for y in range(height):
            if random.random() < density:
                obstacles.add((x, y))
    start = (0, 0)
    goal = (width - 1, height - 1)
    obstacles.discard(start)
    obstacles.discard(goal)
    return GridWorld(width, height, start, goal, obstacles)


def plot_instance(problem: GridWorld, results: dict, out_path: Path):
    """
    results: dict[ method_name ] = {"path": [...], "expanded": set([...])}
    """
    width, height = problem.width, problem.height

    # background grid
    grid = [[0 for _ in range(width)] for _ in range(height)]
    for (x, y) in problem.obstacles:
        grid[y][x] = 1  # obstacle

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, origin="lower", cmap="gray_r")

    # draw expanded nodes (light gray)
    for method, info in results.items():
        expanded = info["expanded"]
        xs = [s[0] for s in expanded]
        ys = [s[1] for s in expanded]
        ax.scatter(xs, ys, s=5, alpha=0.1, label=f"{method} expanded")

    # draw paths with thicker lines
    colors = {
        "classic": "tab:blue",
        "learned": "tab:orange",
        "uncertainty_k0": "tab:green",
        "online_TD": "tab:red",
    }

    for method, info in results.items():
        path = info["path"]
        if not path:
            continue
        xs = [s[0] for s in path]
        ys = [s[1] for s in path]
        ax.plot(xs, ys, linewidth=2, color=colors.get(method, None), label=f"{method} path")

    # start & goal
    sx, sy = problem.start
    gx, gy = problem.goal
    ax.scatter([sx], [sy], c="yellow", s=80, marker="o", edgecolors="k", label="start")
    ax.scatter([gx], [gy], c="magenta", s=80, marker="*", edgecolors="k", label="goal")

    ax.set_title("A* variants on a single grid instance")
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.grid(False)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved visualization to {out_path}")


def main():
    random.seed(0)

    # φτιάχνουμε ένα μικρό instance για να φαίνεται καθαρά
    problem = generate_open_grid(width=32, height=32, density=0.2)

    # φορτώνουμε τα nets
    learned_h = LearnedHeuristic("heuristic_net.pt")
    unc_h = UncertaintyAwareHeuristic("heuristic_net_unc.pt", k=0.0)
    online_h = OnlineUpdateHeuristic(manhattan_heuristic, eta=0.1)

    results = {}

    # classic
    stats_c, expanded_c = astar_with_trace(problem, manhattan_heuristic)
    results["classic"] = {"path": stats_c.path, "expanded": set(expanded_c)}
    print("[classic] expansions:", stats_c.expansions, "cost:", stats_c.path_cost)

    # learned
    stats_l, expanded_l = astar_with_trace(problem, lambda s, p: learned_h(s, p))
    results["learned"] = {"path": stats_l.path, "expanded": set(expanded_l)}
    print("[learned] expansions:", stats_l.expansions, "cost:", stats_l.path_cost)

    # uncertainty (k=0)
    stats_u0, expanded_u0 = astar_with_trace(problem, lambda s, p: unc_h(s, p))
    results["uncertainty_k0"] = {"path": stats_u0.path, "expanded": set(expanded_u0)}
    print("[uncertainty_k0] expansions:", stats_u0.expansions, "cost:", stats_u0.path_cost)

    # online TD πάνω στο classic
    stats_td = astar_with_online_updates(problem, online_h)
    # για expanded nodes μπορούμε να ξανατρέξουμε με trace, αλλά για τώρα ας πάρουμε του classic
    results["online_TD"] = {"path": stats_td.path, "expanded": set(expanded_c)}
    print("[online_TD] expansions:", stats_td.expansions, "cost:", stats_td.path_cost)

    out_path = Path(__file__).resolve().parent.parent / "phd_experiments" / "viz_single_instance.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_instance(problem, results, out_path)


if __name__ == "__main__":
    main()
