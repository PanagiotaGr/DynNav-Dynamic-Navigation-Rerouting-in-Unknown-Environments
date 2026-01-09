import numpy as np
import pandas as pd

from connectivity_map import ConnectivityMap
from connectivity_aware_planner import astar_connectivity, build_simple_risk

def make_world(H=30, W=30, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    occ = np.zeros((H,W), dtype=float)

    # Random obstacle blocks
    for _ in range(6):
        y0 = rng.integers(2, H-8)
        x0 = rng.integers(2, W-8)
        h = rng.integers(3, 7)
        w = rng.integers(3, 7)
        occ[y0:y0+h, x0:x0+w] = 1.0

    start = (1, 1)
    goal = (H-2, W-2)
    occ[start] = 0.0
    occ[goal] = 0.0
    return occ, start, goal

def run(seed_list, gammas, lam=2.0, out_csv="connectivity_sweep.csv"):
    rows = []
    for s in seed_list:
        occ, start, goal = make_world(rng_seed=s)
        risk = build_simple_risk(occ)

        # AP fixed for now (center-ish)
        cm = ConnectivityMap(occ.shape, ap_xy=(10.0, 10.0), rng_seed=s)
        snr_db, C, P_loss = cm.build(occ, add_shadowing=True)

        # Baselines
        for name, (L, G) in {
            "geometry_only": (0.0, 0.0),
            "risk_only": (lam, 0.0),
        }.items():
            path, info = astar_connectivity(occ, start, goal, risk, C, lam=L, gamma=G)
            rows.append({
                "seed": s,
                "planner": name,
                "lam": L,
                "gamma": G,
                **info
            })

        # Connectivity-aware sweep
        for g in gammas:
            path, info = astar_connectivity(occ, start, goal, risk, C, lam=lam, gamma=float(g))
            rows.append({
                "seed": s,
                "planner": "risk_plus_connectivity",
                "lam": lam,
                "gamma": float(g),
                **info
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv} with {len(df)} rows")

if __name__ == "__main__":
    seeds = list(range(0, 20))          # 20 worlds
    gammas = [0.0, 0.5, 1.0, 2.0, 4.0]   # sweep
    run(seeds, gammas, lam=2.0, out_csv="connectivity_sweep.csv")
