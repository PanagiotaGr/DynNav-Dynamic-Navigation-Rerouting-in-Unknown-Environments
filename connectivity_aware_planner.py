import heapq
import numpy as np
from typing import Tuple, List, Dict, Optional

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def neighbors4(u, H, W):
    y, x = u
    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        ny, nx = y+dy, x+dx
        if 0 <= ny < H and 0 <= nx < W:
            yield (ny, nx)

def build_simple_risk(occupancy: np.ndarray) -> np.ndarray:
    """
    Minimal risk proxy: high risk near obstacles.
    risk = 1 / (1 + dist_to_obstacle)
    """
    H, W = occupancy.shape
    obs = np.argwhere(occupancy > 0.5)
    risk = np.zeros((H,W), dtype=float)
    if len(obs) == 0:
        return risk
    for y in range(H):
        for x in range(W):
            d2 = np.min((obs[:,0]-y)**2 + (obs[:,1]-x)**2)
            d = np.sqrt(d2)
            risk[y,x] = 1.0 / (1.0 + d)
    # normalize to [0,1]
    risk = (risk - risk.min()) / (risk.max() - risk.min() + 1e-9)
    return risk

def astar_connectivity(
    occupancy: np.ndarray,
    start: Tuple[int,int],
    goal: Tuple[int,int],
    risk: np.ndarray,
    C: np.ndarray,
    lam: float = 1.0,
    gamma: float = 1.0,
) -> Tuple[Optional[List[Tuple[int,int]]], Dict]:
    """
    A* on 4-neighborhood with cost shaping:
      step_cost = 1 + lam*risk[v] + gamma*(1 - C[v])
    """
    H, W = occupancy.shape
    if occupancy[start] > 0.5 or occupancy[goal] > 0.5:
        return None, {"status":"blocked_start_or_goal"}

    g = {start: 0.0}
    parent = {start: None}
    pq = []
    heapq.heappush(pq, (manhattan(start, goal), start))
    expansions = 0

    while pq:
        f, u = heapq.heappop(pq)
        expansions += 1
        if u == goal:
            # reconstruct
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()

            # metrics
            path_risk = float(np.sum([risk[p] for p in path]))
            path_conn_pen = float(np.sum([(1.0 - C[p]) for p in path]))
            return path, {
                "status":"ok",
                "expansions": expansions,
                "path_len": len(path)-1,
                "path_risk_sum": path_risk,
                "path_conn_pen_sum": path_conn_pen
            }

        for v in neighbors4(u, H, W):
            if occupancy[v] > 0.5:
                continue
            step = 1.0 + lam * float(risk[v]) + gamma * float(1.0 - C[v])
            ng = g[u] + step
            if v not in g or ng < g[v]:
                g[v] = ng
                parent[v] = u
                h = manhattan(v, goal)
                heapq.heappush(pq, (ng + h, v))

    return None, {"status":"no_path", "expansions": expansions}
