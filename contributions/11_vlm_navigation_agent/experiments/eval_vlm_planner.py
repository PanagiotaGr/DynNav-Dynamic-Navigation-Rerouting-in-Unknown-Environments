"""
Experiment: VLM Navigation Agent — offline evaluation
======================================================
Evaluates the VLM planner on a directory of RGB frames + ground-truth
semantic labels.  Produces a CSV with per-frame metrics and a summary plot.

Usage:
    python experiments/eval_vlm_planner.py \
        --frames_dir data/vlm_frames \
        --labels     data/vlm_labels.json \
        --model      llava-1.6 \
        --endpoint   http://localhost:11434/api/chat \
        --out_csv    results/vlm_eval.csv
"""

import argparse
import csv
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("eval_vlm")


def load_frames_stub(frames_dir: str, n: int = 20) -> list[np.ndarray]:
    """
    Stub: returns random RGB frames.
    Replace with: sorted(glob.glob(frames_dir + '/*.jpg'))
    """
    logger.info("Generating %d synthetic frames (stub mode)", n)
    return [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(n)]


def load_labels_stub(n: int = 20) -> list[str]:
    regions = ["corridor", "doorway", "open_space", "obstacle_zone", "room"]
    return [regions[i % len(regions)] for i in range(n)]


def run_eval(args):
    from vlm_planner import VLMNavigationPlanner, VLMPlannerConfig

    cfg = VLMPlannerConfig(
        model_name=args.model,
        api_endpoint=args.endpoint,
        confidence_threshold=args.conf_threshold,
    )
    planner = VLMNavigationPlanner(config=cfg)

    frames = load_frames_stub(args.frames_dir, n=args.n_frames)
    gt_labels = load_labels_stub(n=args.n_frames)

    rows = []
    correct = 0

    for i, (frame, gt) in enumerate(zip(frames, gt_labels)):
        t0 = time.perf_counter()
        goal = planner.plan(frame)
        latency = time.perf_counter() - t0

        predicted = goal.region_label if goal else "none"
        conf = goal.confidence if goal else 0.0
        match = int(predicted == gt)
        correct += match

        rows.append({
            "frame": i,
            "gt_label": gt,
            "predicted_label": predicted,
            "confidence": round(conf, 4),
            "match": match,
            "latency_s": round(latency, 4),
        })
        logger.info("[%3d] gt=%-15s pred=%-15s conf=%.2f match=%d",
                    i, gt, predicted, conf, match)

    accuracy = correct / len(rows) if rows else 0.0
    summary = planner.session_summary()
    logger.info("Accuracy: %.3f | Summary: %s", accuracy, summary)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Append summary row
    with open(args.out_csv, "a") as f:
        f.write(f"\n# accuracy={accuracy:.4f}, total_goals={summary['total']}\n")

    logger.info("Results saved to %s", args.out_csv)
    return accuracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", default="data/vlm_frames")
    ap.add_argument("--labels", default="data/vlm_labels.json")
    ap.add_argument("--model", default="llava-1.6")
    ap.add_argument("--endpoint", default="http://localhost:11434/api/chat")
    ap.add_argument("--conf_threshold", type=float, default=0.55)
    ap.add_argument("--n_frames", type=int, default=20)
    ap.add_argument("--out_csv", default="results/vlm_eval.csv")
    args = ap.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
