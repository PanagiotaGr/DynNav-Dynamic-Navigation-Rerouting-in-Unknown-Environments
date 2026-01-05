# failure_memory_map.py
from __future__ import annotations
import numpy as np

class FailureMemoryMap:
    """
    Persistent grid that accumulates failure evidence over episodes.
    M in [0, +inf). You can cap/normalize when used in planning.
    """

    def __init__(self, shape, decay: float = 0.98):
        self.M = np.zeros(shape, dtype=float)
        self.decay = float(decay)

    def update_episode(self, trajectory_cells, failed: bool, weight: float = 1.0):
        """
        trajectory_cells: list of (x,y)
        failed: whether the episode failed
        weight: update magnitude
        """
        # mild forgetting
        self.M *= self.decay

        if not failed:
            return

        # add failure mass along trajectory (or subset)
        for (x,y) in trajectory_cells:
            self.M[x,y] += weight

    def normalized(self, eps=1e-9):
        m = self.M
        if m.max() < eps:
            return np.zeros_like(m)
        return m / (m.max() + eps)
