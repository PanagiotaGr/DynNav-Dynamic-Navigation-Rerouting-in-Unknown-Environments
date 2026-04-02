"""
Experiment: Diffusion Occupancy Maps — risk estimation benchmark
================================================================
Compares diffusion-based risk maps against a deterministic baseline
(inflated binary occupancy) on path safety metrics.

Usage:
    python experiments/eval_diffusion_occupancy.py \
        --n_scenarios 30 --n_samples 10 --out_csv results/diffusion_eval.csv
"""
import argparse, csv, logging, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from diffusion_occupancy import DiffusionOccupancyPredictor, DiffusionOccupancyConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("eval_diff")


def random_occupancy(h, w, density=0.15, seed=None):
    rng = np.random.default_rng(seed)
    return (rng.random((h, w)) < density).astype(float)


def random_path(h, w, length=20, seed=None):
    rng = np.random.default_rng(seed)
    r0, c0 = rng.integers(0, h), rng.integers(0, w)
    path = [(r0, c0)]
    for _ in range(length - 1):
        dr, dc = rng.integers(-1, 2), rng.integers(-1, 2)
        r = int(np.clip(path[-1][0] + dr, 0, h - 1))
        c = int(np.clip(path[-1][1] + dc, 0, w - 1))
        path.append((r, c))
    return path


def run(args):
    cfg = DiffusionOccupancyConfig(n_samples=args.n_samples)
    predictor = DiffusionOccupancyPredictor(cfg)
    rows = []

    for i in range(args.n_scenarios):
        occ = random_occupancy(cfg.grid_h, cfg.grid_w, seed=i)
        path = random_path(cfg.grid_h, cfg.grid_w, seed=i)

        t0 = time.perf_counter()
        risk = predictor.predict_risk([occ])
        elapsed = time.perf_counter() - t0

        diff_cost = predictor.risk_weighted_cost(path, risk, lambda_risk=2.0)
        det_cost = float(len(path)) + 2.0 * sum(occ[r, c] for r, c in path)

        rows.append({
            "scenario": i,
            "diff_risk_cost": round(diff_cost, 4),
            "det_risk_cost": round(det_cost, 4),
            "mean_cvar": round(float(risk["cvar_95"].mean()), 4),
            "mean_std": round(float(risk["std"].mean()), 4),
            "elapsed_s": round(elapsed, 4),
        })
        logger.info("[%2d] diff=%.2f det=%.2f cvar=%.3f time=%.3fs",
                    i, diff_cost, det_cost, risk["cvar_95"].mean(), elapsed)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    logger.info("Saved to %s", args.out_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_scenarios", type=int, default=30)
    ap.add_argument("--n_samples", type=int, default=10)
    ap.add_argument("--out_csv", default="results/diffusion_eval.csv")
    run(ap.parse_args())

if __name__ == "__main__":
    main()
