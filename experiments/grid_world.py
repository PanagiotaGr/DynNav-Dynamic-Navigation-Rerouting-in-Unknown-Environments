from typing import List, Tuple, Set
from .astar_core import SearchProblem


class GridWorld(SearchProblem):
    """
    4-connected grid με ομοιόμορφο κόστος 1.
    Συντεταγμένες: (x, y), με 0 <= x < width, 0 <= y < height
    Obstacles: σύνολο από (x,y)
    """
    def __init__(
        self,
        width: int,
        height: int,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        obstacles: Set[Tuple[int, int]],
    ):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = obstacles

    def in_bounds(self, s: Tuple[int, int]) -> bool:
        x, y = s
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, s: Tuple[int, int]) -> bool:
        return s not in self.obstacles

    def get_start(self) -> Tuple[int, int]:
        return self.start

    def is_goal(self, state: Tuple[int, int]) -> bool:
        return state == self.goal

    def successors(self, state: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        x, y = state
        candidates = [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ]
        res = []
        for nx, ny in candidates:
            s2 = (nx, ny)
            if self.in_bounds(s2) and self.passable(s2):
                res.append((s2, 1.0))  # ομοιόμορφο κόστος
        return res
