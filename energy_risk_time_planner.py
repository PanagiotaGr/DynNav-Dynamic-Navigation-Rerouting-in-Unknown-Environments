# energy_risk_time_planner.py
from __future__ import annotations
from dataclasses import dataclass
import heapq
import numpy as np
from typing import Dict, Tuple, List, Optional

from energy_model import EnergyModel, EnergyParams

@dataclass
class PlannerWeights:
    alpha_L: float = 1.0   # path length
    beta_R: float = 5.0    # risk weight
    gamma_E: float = 1.0   # energy usage weight

@dataclass
class PlanResult:
    path: List[Tuple[int, int]]
    total_L: float
    total_R: float
    total_E: float
    total_J: float
    expansions: int
    success: bool
    reason: str

def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

class EnergyRiskTimePlanner:
    """
    A* on grid with augmented energy state.
    State = (x, y, e_bin) where e_bin discretizes remaining energy.
    """

    def __init__(
        self,
        occ_grid: np.ndarray,        # 0 free, 1 obstacle
        risk_grid: np.ndarray,       # risk in [0, 1] (or any nonnegative)
        start: Tuple[int,int],
        goal: Tuple[int,int],
        weights: PlannerWeights,
        energy_params: EnergyParams,
        e_bins: int = 50,
        seed: int = 0,
    ):
        assert occ_grid.shape == risk_grid.shape
        self.occ = occ_grid.astype(np.uint8)
        self.risk = risk_grid.astype(float)
        self.H, self.W = self.occ.shape
        self.s = start
        self.g = goal
        self.w = weights
        self.em = EnergyModel(energy_params)
        self.e0 = energy_params.e0
        self.e_bins = int(e_bins)
        self.rng = np.random.default_rng(seed)

        # discretization
        self.e_min = 0.0
        self.e_max = float(self.e0)
        self.bin_size = (self.e_max - self.e_min) / self.e_bins

    def _to_bin(self, e: float) -> int:
        e = max(self.e_min, min(self.e_max, e))
        b = int(np.floor((e - self.e_min) / self.bin_size))
        return min(self.e_bins-1, max(0, b))

    def _bin_to_energy_lower(self, b: int) -> float:
        return self.e_min + b * self.bin_size

    def _neighbors(self, x: int, y: int):
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < self.H and 0 <= ny < self.W and self.occ[nx,ny] == 0:
                yield nx, ny

    def _heuristic(self, x: int, y: int) -> float:
        # admissible for L only; we keep it conservative for multi-objective
        return float(manhattan((x,y), self.g)) * self.w.alpha_L

    def plan(self) -> PlanResult:
        if self.occ[self.s] == 1 or self.occ[self.g] == 1:
            return PlanResult([],0,0,0,0,0,False,"start/goal blocked")

        # g_cost stores best-known J-cost to reach a state (x,y,e_bin)
        INF = 1e18
        g_cost: Dict[Tuple[int,int,int], float] = {}
        parent: Dict[Tuple[int,int,int], Tuple[int,int,int]] = {}

        # Also track components for reporting
        comp: Dict[Tuple[int,int,int], Tuple[float,float,float]] = {}  # (L,R,E)

        e0_bin = self._to_bin(self.e0)
        start_state = (self.s[0], self.s[1], e0_bin)

        g_cost[start_state] = 0.0
        comp[start_state] = (0.0, 0.0, 0.0)

        pq = []
        heapq.heappush(pq, (self._heuristic(*self.s), 0.0, start_state))

        expansions = 0
        best_goal_state: Optional[Tuple[int,int,int]] = None
        best_goal_cost = INF

        while pq:
            f, gc, st = heapq.heappop(pq)
            if gc != g_cost.get(st, INF):
                continue

            x, y, eb = st
            expansions += 1

            if (x, y) == self.g:
                best_goal_state = st
                best_goal_cost = gc
                break

            e_rem = self._bin_to_energy_lower(eb)

            for nx, ny in self._neighbors(x, y):
                # step risk is risk at destination cell (simple model)
                r_step = float(self.risk[nx, ny])
                e_step = self.em.step_energy_cost(r_step)

                new_e = e_rem - e_step
                if new_e < 0:
                    continue  # infeasible due to energy depletion

                neb = self._to_bin(new_e)
                nst = (nx, ny, neb)

                # components update
                L_prev, R_prev, E_prev = comp[st]
                L_new = L_prev + 1.0
                R_new = R_prev + r_step
                E_new = E_prev + e_step

                J_step = self.w.alpha_L*1.0 + self.w.beta_R*r_step + self.w.gamma_E*e_step
                ng = gc + J_step

                if ng < g_cost.get(nst, INF):
                    g_cost[nst] = ng
                    comp[nst] = (L_new, R_new, E_new)
                    parent[nst] = st
                    nf = ng + self._heuristic(nx, ny)
                    heapq.heappush(pq, (nf, ng, nst))

        if best_goal_state is None:
            return PlanResult([],0,0,0,0,expansions,False,"no feasible path (energy/obstacles)")

        # reconstruct path (ignore energy in final path output)
        path_states = []
        cur = best_goal_state
        while cur != start_state:
            path_states.append(cur)
            cur = parent[cur]
        path_states.append(start_state)
        path_states.reverse()

        path = [(x,y) for (x,y,_) in path_states]
        L, R, E = comp[best_goal_state]
        J = best_goal_cost

        return PlanResult(path, L, R, E, J, expansions, True, "ok")
