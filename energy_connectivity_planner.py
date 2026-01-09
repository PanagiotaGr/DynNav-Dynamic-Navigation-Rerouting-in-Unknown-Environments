import heapq
import numpy as np
from typing import Tuple, List, Optional, Dict

from connectivity_map import ConnectivityMap
from connectivity_aware_planner import build_simple_risk, neighbors4, manhattan

def build_simple_energy_field(occupancy: np.ndarray) -> np.ndarray:
    """
    Minimal energy proxy per cell (for research baseline).
    - Free space has baseline energy 1
    - Near obstacles slightly higher (turning/slowdown)
    """
    H, W = occupancy.shape
    risk = build_simple_risk(occupancy)
    E = 1.0 + 0.5 * risk  # normalized-ish, smooth
    # normalize to [0,1] for cost shaping convenience
    E = (E - E.min()) / (E.max() - E.min() + 1e-9)
    return E

def astar_energy_connectivity(
    occupancy: np.ndarray,
    start: Tuple[int,int],
    goal: Tuple[int,int],
    risk: np.ndarray,
    energy: np.ndarray,
    C: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    delta: float = 1.0,
    gamma: float = 1.0,
) -> Tuple[Optional[List[Tuple[int,int]]], Dict]:
    """
    A* with step cost:
      step = alpha*1 + beta*risk[v] + delta*energy[v] + gamma*(1 - C[v])
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
            path = []
            cur = goal
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()

            path_len = len(path) - 1
            risk_sum = float(np.sum([risk[p] for p in path]))
            energy_sum = float(np.sum([energy[p] for p in path]))
            conn_pen_sum = float(np.sum([(1.0 - C[p]) for p in path]))
            snr_mean = float(np.mean([C[p] for p in path]))

            return path, {
                "status":"ok",
                "expansions": expansions,
                "path_len": path_len,
                "risk_sum": risk_sum,
                "energy_sum": energy_sum,
                "conn_penalty": conn_pen_sum,
                "C_mean": snr_mean
            }

        for v in neighbors4(u, H, W):
            if occupancy[v] > 0.5:
                continue
            step = (
                alpha * 1.0
                + beta * float(risk[v])
                + delta * float(energy[v])
                + gamma * float(1.0 - C[v])
            )
            ng = g[u] + step
            if v not in g or ng < g[v]:
                g[v] = ng
                parent[v] = u
                h = manhattan(v, goal)
                heapq.heappush(pq, (ng + h, v))

    return None, {"status":"no_path", "expansions": expansions}

def simulate_safe_mode(
    occupancy: np.ndarray,
    start: Tuple[int,int],
    goal: Tuple[int,int],
    ap_xy=(10.0, 10.0),
    seed: int = 0,
    alpha=1.0, beta=2.0, delta=1.0,
    gamma=1.0,
    Cmin=0.35,
    k=4,
    gamma_safe=4.0,
) -> Dict:
    """
    Plan once with (gamma). Traverse path; if C<Cmin for k consecutive steps -> safe mode:
      replan from current with gamma_safe (more connectivity-averse).
    Returns metrics including disconnect_steps and safe_mode_activations.
    """
    occ = occupancy.astype(float)
    risk = build_simple_risk(occ)
    energy = build_simple_energy_field(occ)

    cm = ConnectivityMap(occ.shape, ap_xy=ap_xy, rng_seed=seed)
    _, C, P_loss = cm.build(occ, add_shadowing=True)

    # initial plan
    path, info = astar_energy_connectivity(occ, start, goal, risk, energy, C,
                                          alpha=alpha, beta=beta, delta=delta, gamma=gamma)
    if path is None:
        return {"status": info.get("status","fail"), "safe_mode_activations": 0, "disconnect_steps": 0}

    safe_mode = False
    safe_acts = 0
    disconnect_steps = 0
    low_run = 0
    replans = 0

    cur_idx = 0
    cur = path[cur_idx]

    # simulate following the path (cell by cell)
    while cur != goal:
        cur_idx += 1
        if cur_idx >= len(path):
            break
        cur = path[cur_idx]

        # count disconnection-ish steps
        if C[cur] < Cmin:
            disconnect_steps += 1
            low_run += 1
        else:
            low_run = 0

        # trigger safe mode
        if (not safe_mode) and (low_run >= k):
            safe_mode = True
            safe_acts += 1

            # replan from current with stronger connectivity weight
            new_path, new_info = astar_energy_connectivity(
                occ, cur, goal, risk, energy, C,
                alpha=alpha, beta=beta, delta=delta, gamma=gamma_safe
            )
            replans += 1
            if new_path is None:
                return {
                    "status":"safe_mode_replan_failed",
                    "safe_mode_activations": safe_acts,
                    "disconnect_steps": disconnect_steps,
                    "replans": replans
                }

            # stitch: keep current cell + rest
            path = [cur] + new_path[1:]
            cur_idx = 0  # restart index relative to new path

    # final summary
    return {
        "status":"ok",
        "safe_mode_activations": safe_acts,
        "disconnect_steps": disconnect_steps,
        "replans": replans,
        "path_len": len(path)-1,
    }

