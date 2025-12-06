import heapq
import time
from typing import Any, Tuple

from .astar_core import SearchStats, reconstruct_path

def astar_with_online_updates(problem, online_heuristic, max_expansions=1_000_000) -> SearchStats:
    """
    A* που, κάθε φορά που κοιτάει successors,
    κάνει TD update στο online heuristic.
    """
    start_time = time.time()
    start = problem.get_start()
    goal_test = problem.is_goal

    open_heap = []
    g = {start: 0.0}
    came_from = {}
    closed = set()
    expansions = 0
    tie_breaker = 0

    h0 = online_heuristic(start, problem)
    heapq.heappush(open_heap, (g[start] + h0, tie_breaker, start))

    best_goal_cost = None
    best_goal_state = None

    while open_heap and expansions < max_expansions:
        f, _, s = heapq.heappop(open_heap)
        if s in closed:
            continue
        closed.add(s)
        expansions += 1

        if goal_test(s):
            best_goal_cost = g[s]
            best_goal_state = s
            break

        for (s_next, cost) in problem.successors(s):
            # Online TD update
            online_heuristic.update(s, s_next, cost, problem)

            g_new = g[s] + cost
            if s_next not in g or g_new < g[s_next]:
                g[s_next] = g_new
                came_from[s_next] = s
                h = online_heuristic(s_next, problem)
                tie_breaker += 1
                heapq.heappush(open_heap, (g_new + h, tie_breaker, s_next))

    runtime_sec = time.time() - start_time

    if best_goal_state is None:
        return SearchStats(False, [], float("inf"), expansions, runtime_sec)
    path = reconstruct_path(came_from, best_goal_state)
    return SearchStats(True, path, best_goal_cost, expansions, runtime_sec)
