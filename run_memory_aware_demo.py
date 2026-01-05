# run_memory_aware_demo.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from failure_memory_map import FailureMemoryMap
from memory_aware_planner import MemoryAwareAStar, MemoryWeights

def make_world(H=60, W=80, seed=0):
    rng = np.random.default_rng(seed)
    occ = np.zeros((H,W), dtype=np.uint8)

    # corridor + cul-de-sac style obstacles
    occ[10:50, 30] = 1
    occ[10:50, 50] = 1
    occ[10, 30:50] = 1
    occ[50, 30:50] = 1
    occ[30, 50:70] = 0  # corridor extension
    occ[20:40, 70] = 1  # dead-end boundary

    # risk field
    base = rng.random((H,W))
    risk = base.copy()
    for _ in range(5):
        risk = 0.3*risk + 0.7*(np.roll(risk,1,0)+np.roll(risk,-1,0)+np.roll(risk,1,1)+np.roll(risk,-1,1))/4.0
    risk = (risk - risk.min())/(risk.max()-risk.min()+1e-9)

    occ[0,:]=0; occ[-1,:]=0; occ[:,0]=0; occ[:,-1]=0
    return occ, risk

def simulate_failure(path, fail_zone_center=(30,40), radius=6):
    # “Failure” if trajectory enters a hazardous region (proxy)
    cx,cy = fail_zone_center
    for (x,y) in path:
        if (x-cx)**2 + (y-cy)**2 <= radius**2:
            return True
    return False

def plot_map(occ, risk, memN, path, fname):
    plt.figure()
    plt.imshow(risk, origin="lower")
    plt.contour(occ, levels=[0.5], linewidths=1)
    plt.imshow(memN, origin="lower", alpha=0.35)
    if path:
        xs = [p[1] for p in path]
        ys = [p[0] for p in path]
        plt.plot(xs, ys)
    plt.tight_layout()
    plt.savefig(fname)

def main():
    occ, risk = make_world(seed=2)
    start = (5,5)
    goal  = (55,75)

    mem = FailureMemoryMap(occ.shape, decay=0.995)

    rows = []
    w = MemoryWeights(alpha_L=1.0, beta_R=2.0, delta_M=8.0)

    # Run multiple episodes; memory should steer away from failure region over time
    for ep in range(1, 16):
        planner = MemoryAwareAStar(occ, risk, mem, start, goal, w)
        path, info = planner.plan()

        failed = False
        if info["success"]:
            failed = simulate_failure(path)
            mem.update_episode(path, failed=failed, weight=1.0)

        memN = mem.normalized()
        rows.append({
            "episode": ep,
            "success": int(info["success"]),
            "failed": int(failed),
            "path_len": len(path),
            "expansions": info["expansions"],
            "mem_max": float(mem.M.max())
        })

        if ep in [1, 5, 10, 15]:
            plot_map(occ, risk, memN, path, f"memory_episode_{ep}.png")

    df = pd.DataFrame(rows)
    df.to_csv("memory_aware_results.csv", index=False)
    print(df)

    plt.figure()
    plt.plot(df["episode"], df["failed"], marker="o")
    plt.xlabel("episode")
    plt.ylabel("failure (1/0)")
    plt.tight_layout()
    plt.savefig("memory_failures_over_time.png")

if __name__ == "__main__":
    main()
