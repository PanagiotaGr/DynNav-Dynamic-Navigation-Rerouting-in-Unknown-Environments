"""
Experiment: Formal Safety Shields — STL + CBF evaluation
=========================================================
Simulates a robot navigating with and without the safety shield.
Measures: constraint violations, command correction magnitude, path length.

Usage:
    python experiments/eval_safety_shields.py \
        --n_episodes 50 --out_csv results/shield_eval.csv
"""
import argparse, csv, logging, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from formal_safety_shields import (
    STLMonitor, STLAtom, STLAlways, CBFSafetyFilter, CBFConfig, SafetyShield
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("eval_shield")


def build_shield(safety_radius=0.4):
    dist_spec = STLAtom(
        lambda s: np.linalg.norm(s[:2]) - 0.1,
        name="non_zero_position"
    )
    monitor = STLMonitor([("non_zero", STLAlways(dist_spec, 0, 5))])
    cbf = CBFSafetyFilter(CBFConfig(safety_radius=safety_radius))
    return SafetyShield(monitor, cbf)


def run_episode(shield, obstacles, n_steps=40, use_shield=True):
    rng = np.random.default_rng()
    pos = np.array([0.0, 0.0])
    violations, corrections = 0, []

    for _ in range(n_steps):
        u_des = rng.uniform(-0.3, 0.3, 2)
        if use_shield:
            u_safe, info = shield.step(u_des, pos, obstacles, pos)
            corrections.append(info["cbf"]["correction_norm"])
            if info["cbf"]["safety_violated"]:
                violations += 1
        else:
            u_safe = u_des
            min_d = min(np.linalg.norm(pos - o) for o in obstacles)
            if min_d < 0.4:
                violations += 1
            corrections.append(0.0)
        pos = pos + u_safe * 0.1

    return violations, np.mean(corrections)


def run(args):
    rows = []
    for ep in range(args.n_episodes):
        rng = np.random.default_rng(ep)
        obs = [rng.uniform(-1, 1, 2) for _ in range(5)]

        shield_on = build_shield()
        v_on, c_on = run_episode(shield_on, obs, use_shield=True)

        shield_off = build_shield()
        v_off, c_off = run_episode(shield_off, obs, use_shield=False)

        rows.append({
            "episode": ep,
            "violations_shielded": v_on,
            "violations_unshielded": v_off,
            "mean_correction": round(c_on, 5),
            "reduction_pct": round((v_off - v_on) / max(v_off, 1) * 100, 2),
        })
        logger.info("[%2d] shield=%d unshielded=%d corr=%.4f",
                    ep, v_on, v_off, c_on)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    logger.info("Saved to %s", args.out_csv)
    avg_red = np.mean([r["reduction_pct"] for r in rows])
    logger.info("Average violation reduction: %.1f%%", avg_red)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=50)
    ap.add_argument("--out_csv", default="results/shield_eval.csv")
    run(ap.parse_args())

if __name__ == "__main__":
    main()
