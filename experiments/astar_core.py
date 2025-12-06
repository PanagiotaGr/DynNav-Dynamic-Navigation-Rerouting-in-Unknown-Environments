import heapq
import time
from typing import Callable, Any, Dict, List, Tuple, Optional


class SearchStats:
    def __init__(
        self,
        success: bool,
        path: List[Any],
        path_cost: float,
        expansions: int,
        runtime_sec: float,
    ):
        self.success = success
        self.path = path
        self.path_cost = path_cost
        self.expansions = expansions
        self.runtime_sec = runtime_sec


class SearchProblem:
    """
    Abstract interface.
    Θα το υλοποιήσουμε για grid-world πιο κάτω.
    """
    def get_start(self) -> Any:
        raise NotImplementedError

    def is_goal(self, state: Any) -> bool:
        raise NotImplementedError

    def successors(self, state: Any) -> List[Tuple[Any, float]]:
        """
        Επιστρέφει [(next_state, cost), ...]
        """
        raise NotImplementedError


def reconstruct_path(came_from: Dict[Any, Any], goal: Any) -> List[Any]:
    path = [goal]
    while goal in came_from:
        goal = came_from[goal]
        path.append(goal)
    path.reverse()
    return path


def astar_search(
    problem: SearchProblem,
    heuristic_fn: Callable[[Any, SearchProblem], float],
    max_expansions: int = 1_000_000,
) -> SearchStats:
    """
    A* που κρατά αναλυτικά στατιστικά.
    heuristic_fn(state, problem) -> h(state)
    """
    start_time = time.time()

    start = problem.get_start()
    goal_test = problem.is_goal

    open_heap: List[Tuple[float, int, Any]] = []  # (f, tie_breaker, state)
    g: Dict[Any, float] = {start: 0.0}
    came_from: Dict[Any, Any] = {}

    h0 = heuristic_fn(start, problem)
    f0 = g[start] + h0

    heapq.heappush(open_heap, (f0, 0, start))

    closed = set()
    expansions = 0
    tie_breaker = 0

    best_goal_cost: Optional[float] = None
    best_goal_state: Optional[Any] = None

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
            g_new = g[s] + cost
            if s_next not in g or g_new < g[s_next]:
                g[s_next] = g_new
                came_from[s_next] = s
                h = heuristic_fn(s_next, problem)
                tie_breaker += 1
                heapq.heappush(open_heap, (g_new + h, tie_breaker, s_next))

    runtime_sec = time.time() - start_time

    if best_goal_state is None:
        return SearchStats(
            success=False,
            path=[],
            path_cost=float("inf"),
            expansions=expansions,
            runtime_sec=runtime_sec,
        )

    path = reconstruct_path(came_from, best_goal_state)
    return SearchStats(
        success=True,
        path=path,
        path_cost=best_goal_cost,
        expansions=expansions,
        runtime_sec=runtime_sec,
    )
