# memory_aware_planner.py
from __future__ import annotations
from dataclasses import dataclass
import heapq
import numpy as np
from typing import Dict, Tuple, List, Optional

from failure_memory_map import FailureMemoryMap

@dataclass
class MemoryWeights:
    alpha_L: float = 1.0
    beta_R: float = 3.0
    delta_M: float = 6.0

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

class MemoryAwareAStar:
    def __init__(self, occ, risk, memory: FailureMemoryMap, start, goal, w: MemoryWeights):
        assert occ.shape == risk.shape == memory.M.shape
        self.occ = occ.astype(np.uint8)
        self.risk = risk.astype(float)
        self.mem = memory
        self.start = start
        self.goal = goal
        self.w = w
        self.H, self.W = occ.shape

    def _neighbors(self, x,y):
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx,ny = x+dx,y+dy
            if 0<=nx<self.H and 0<=ny<self.W and self.occ[nx,ny]==0:
                yield nx,ny

    def _heur(self, x,y):
        return float(manhattan((x,y), self.goal))*self.w.alpha_L

    def plan(self):
        INF = 1e18
        memN = self.mem.normalized()

        g: Dict[Tuple[int,int], float] = {self.start: 0.0}
        parent: Dict[Tuple[int,int], Tuple[int,int]] = {}
        pq = []
        heapq.heappush(pq, (self._heur(*self.start), 0.0, self.start))
        expansions = 0

        while pq:
            f, gc, u = heapq.heappop(pq)
            if gc != g.get(u, INF):
                continue
            expansions += 1
            if u == self.goal:
                break

            x,y = u
            for v in self._neighbors(x,y):
                nx,ny = v
                step_L = 1.0
                step_R = float(self.risk[nx,ny])
                step_M = float(memN[nx,ny])

                step_J = self.w.alpha_L*step_L + self.w.beta_R*step_R + self.w.delta_M*step_M
                ng = gc + step_J
                if ng < g.get(v, INF):
                    g[v] = ng
                    parent[v] = u
                    heapq.heappush(pq, (ng + self._heur(nx,ny), ng, v))

        if self.goal not in g:
            return [], {"success": False, "expansions": expansions, "reason": "no path"}

        # reconstruct
        path = [self.goal]
        cur = self.goal
        while cur != self.start:
            cur = parent[cur]
            path.append(cur)
        path.reverse()
        return path, {"success": True, "expansions": expansions, "cost": g[self.goal]}
