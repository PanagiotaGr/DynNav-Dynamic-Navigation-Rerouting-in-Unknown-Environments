import numpy as np
import pandas as pd

from energy_connectivity_planner import simulate_safe_mode
from run_connectivity_sweep import make_world  # reuse same world generator

def run(out_csv="energy_connectivity_joint_sweep.csv"):
    seeds = list(range(0, 30))

    gammas = [0.0, 0.5, 1.0, 2.0, 4.0]     # connectivity weight
    deltas = [0.0, 0.5, 1.0, 2.0]          # energy weight

    rows = []
    for s in seeds:
        occ, start, goal = make_world(rng_seed=s)

        for d in deltas:
            for g in gammas:
                res = simulate_safe_mode(
                    occ, start, goal,
                    seed=s,
                    alpha=1.0, beta=2.0, delta=float(d),
                    gamma=float(g),
                    Cmin=0.35, k=4,
                    gamma_safe=4.0
                )
                rows.append({
                    "seed": s,
                    "delta": float(d),
                    "gamma": float(g),
                    **res
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv} with {len(df)} rows")

if __name__ == "__main__":
    run()
