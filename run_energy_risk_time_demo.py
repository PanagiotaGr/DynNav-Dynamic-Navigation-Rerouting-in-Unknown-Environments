# run_energy_risk_time_demo.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from energy_risk_time_planner import EnergyRiskTimePlanner, PlannerWeights
from energy_model import EnergyParams

def make_world(H=60, W=80, seed=0, start=(2,2), goal=(55,75)):
    rng = np.random.default_rng(seed)
    occ = np.zeros((H,W), dtype=np.uint8)

    # obstacles: fewer random rectangles
    for _ in range(4):
        h = rng.integers(5, 14)
        w = rng.integers(6, 18)
        x = rng.integers(0, H-h)
        y = rng.integers(0, W-w)
        occ[x:x+h, y:y+w] = 1

    # risk grid: smooth-ish random field
    base = rng.random((H,W))
    risk = base.copy()
    for _ in range(6):
        risk = 0.25*risk + 0.75*(np.roll(risk,1,0)+np.roll(risk,-1,0)+np.roll(risk,1,1)+np.roll(risk,-1,1))/4.0
    risk = (risk - risk.min())/(risk.max()-risk.min()+1e-9)

    # ensure borders are traversable
    occ[0,:]=0; occ[-1,:]=0; occ[:,0]=0; occ[:,-1]=0

    # --- GUARANTEE CONNECTIVITY: carve a corridor from start to goal ---
    sx, sy = start
    gx, gy = goal
    steps = max(abs(gx - sx), abs(gy - sy)) + 1
    xs = np.linspace(sx, gx, steps).astype(int)
    ys = np.linspace(sy, gy, steps).astype(int)

    for x, y in zip(xs, ys):
        x0 = max(0, x-1); x1 = min(H, x+2)
        y0 = max(0, y-1); y1 = min(W, y+2)
        occ[x0:x1, y0:y1] = 0

    # ensure start/goal free
    occ[start] = 0
    occ[goal] = 0

    return occ, risk

def main():
    occ, risk = make_world(seed=1)
    start = (2,2)
    goal = (55,75)
    occ, risk = make_world(seed=1, start=start, goal=goal)

    # sweep gamma (energy weight) to show tradeoffs
    rows = []
    for gamma in [0.0, 0.5, 1.0, 2.0, 5.0]:
        weights = PlannerWeights(alpha_L=1.0, beta_R=4.0, gamma_E=gamma)
        eparams = EnergyParams(e0=300.0, move_cost=1.0, risk_energy_coeff=0.8)

        planner = EnergyRiskTimePlanner(
            occ_grid=occ,
            risk_grid=risk,
            start=start,
            goal=goal,
            weights=weights,
            energy_params=eparams,
            e_bins=60,
            seed=0
        )
        res = planner.plan()
        rows.append({
            "gamma_E": gamma,
            "success": int(res.success),
            "reason": res.reason,
            "L": res.total_L,
            "R": res.total_R,
            "E": res.total_E,
            "J": res.total_J,
            "expansions": res.expansions
        })

    df = pd.DataFrame(rows)
    df.to_csv("energy_risk_time_sweep.csv", index=False)
    print(df)

    # plot trade-offs
    plt.figure()
    plt.plot(df["gamma_E"], df["L"], marker="o")
    plt.xlabel("gamma_E")
    plt.ylabel("Path length L")
    plt.tight_layout()
    plt.savefig("energy_sweep_L.png")

    plt.figure()
    plt.plot(df["gamma_E"], df["R"], marker="o")
    plt.xlabel("gamma_E")
    plt.ylabel("Total risk R")
    plt.tight_layout()
    plt.savefig("energy_sweep_R.png")

    plt.figure()
    plt.plot(df["gamma_E"], df["E"], marker="o")
    plt.xlabel("gamma_E")
    plt.ylabel("Total energy E")
    plt.tight_layout()
    plt.savefig("energy_sweep_E.png")

    plt.figure()
    plt.plot(df["gamma_E"], df["expansions"], marker="o")
    plt.xlabel("gamma_E")
    plt.ylabel("A* expansions")
    plt.tight_layout()
    plt.savefig("energy_sweep_expansions.png")

if __name__ == "__main__":
    main()
