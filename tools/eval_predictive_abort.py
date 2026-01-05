#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate predictive abort predictions (ROC/PR + lead time).")
    p.add_argument("--pred", "-p", required=True, help="CSV output from tools/predictive_abort.py")
    p.add_argument("--event-col", default="safe_mode", help="Event column in pred CSV (default: safe_mode)")
    p.add_argument("--score-col", default="p_fail_k", help="Score/prob column (default: p_fail_k)")
    p.add_argument("--time-col", default="t", help="Time column (default: t). If missing, uses index.")
    p.add_argument("--threshold", type=float, default=0.7, help="Threshold for lead-time computation (default: 0.7)")
    p.add_argument("--thresholds", type=int, default=21, help="Number of thresholds in [0,1] (default: 21)")
    p.add_argument("--min-gap", type=int, default=5, help="Min gap (steps) between event starts (default: 5)")
    p.add_argument("--max-lookback", type=int, default=200, help="Max lookback steps for lead time (default: 200)")
    return p.parse_args()


def auc_trapz(x: np.ndarray, y: np.ndarray) -> float:
    # numpy.trapezoid is the non-deprecated version
    return float(np.trapezoid(y, x))


def roc_curve(y: np.ndarray, s: np.ndarray):
    order = np.argsort(-s)
    y = y[order]
    s = s[order]

    P = np.sum(y == 1)
    N = np.sum(y == 0)
    if P == 0 or N == 0:
        return None

    tps = 0
    fps = 0
    tpr = [0.0]
    fpr = [0.0]

    last_score = None
    for yi, si in zip(y, s):
        if last_score is None:
            last_score = si
        if si != last_score:
            tpr.append(tps / P)
            fpr.append(fps / N)
            last_score = si

        if yi == 1:
            tps += 1
        else:
            fps += 1

    tpr.append(tps / P)
    fpr.append(fps / N)
    return np.array(fpr), np.array(tpr)


def pr_curve(y: np.ndarray, s: np.ndarray):
    order = np.argsort(-s)
    y = y[order]
    s = s[order]

    P = np.sum(y == 1)
    if P == 0:
        return None

    tps = 0
    fps = 0
    precision = [1.0]
    recall = [0.0]

    last_score = None
    for yi, si in zip(y, s):
        if last_score is None:
            last_score = si
        if si != last_score:
            prec = tps / max(1, (tps + fps))
            rec = tps / P
            precision.append(float(prec))
            recall.append(float(rec))
            last_score = si

        if yi == 1:
            tps += 1
        else:
            fps += 1

    prec = tps / max(1, (tps + fps))
    rec = tps / P
    precision.append(float(prec))
    recall.append(float(rec))
    return np.array(recall), np.array(precision)


def metrics_at_threshold(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    pred = (s >= thr).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"thr": thr, "precision": precision, "recall": recall, "f1": f1, "fpr": fpr, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def event_starts_indices(event: np.ndarray, min_gap: int) -> list[int]:
    e = (event > 0.5).astype(int)
    starts = []
    last = -10**9
    for i in range(1, len(e)):
        if e[i] == 1 and e[i - 1] == 0:
            if i - last >= min_gap:
                starts.append(i)
                last = i
    return starts


def lead_times_from_score(starts: list[int], score: np.ndarray, thr: float, max_lookback: int) -> list[int]:
    """
    For each event start s, find earliest time j in [s-max_lookback, s] where score[j] >= thr.
    Lead = s - j (>=0). If none found -> ignore.
    """
    leads = []
    for s in starts:
        j0 = max(0, s - max_lookback)
        j_found = None
        for j in range(j0, s + 1):
            if score[j] >= thr:
                j_found = j
                break
        if j_found is not None:
            leads.append(s - j_found)
    return leads


def main() -> int:
    pred_path = Path(parse_args().pred)
    args = parse_args()

    if not pred_path.exists():
        print(f"ERROR: pred file not found: {pred_path}")
        return 2

    # robust: detect empty file
    if pred_path.stat().st_size == 0:
        print(f"ERROR: pred file is empty (0 bytes): {pred_path}")
        return 2

    try:
        df = pd.read_csv(pred_path)
    except pd.errors.EmptyDataError:
        print(f"ERROR: pred file has no columns to parse: {pred_path}")
        return 2

    if args.event_col not in df.columns or args.score_col not in df.columns:
        print("ERROR: missing columns.")
        print("Need:", args.event_col, args.score_col)
        print("Available:", df.columns.tolist())
        return 2

    y_event = pd.to_numeric(df[args.event_col], errors="coerce").fillna(0).to_numpy().astype(int)
    s = pd.to_numeric(df[args.score_col], errors="coerce").fillna(0).to_numpy().astype(float)

    # IMPORTANT: evaluate classification on EVENT STARTS (0->1), not raw event occupancy
    y = np.zeros_like(y_event, dtype=int)
    if len(y_event) >= 2:
        y[1:] = (y_event[1:] == 1) & (y_event[:-1] == 0)

    # ROC/PR on event-start label
    roc = roc_curve(y, s)
    pr = pr_curve(y, s)

    if roc is None:
        print("ROC undefined (need both positive and negative labels).")
    else:
        fpr, tpr = roc
        print(f"ROC-AUC: {auc_trapz(fpr, tpr):.4f}")

    if pr is None:
        print("PR undefined (need positives).")
    else:
        recall, precision = pr
        print(f"PR-AUC : {auc_trapz(recall, precision):.4f}")

    thrs = np.linspace(0.0, 1.0, args.thresholds)
    rows = [metrics_at_threshold(y, s, float(th)) for th in thrs]
    mdf = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    print("\nTop thresholds by F1 (event-start classification):")
    print(mdf.head(8).to_string(index=False))

    starts = event_starts_indices(y_event, min_gap=args.min_gap)
    leads = lead_times_from_score(starts, s, thr=args.threshold, max_lookback=args.max_lookback)

    print(f"\nEvent starts detected (from {args.event_col} 0->1): {len(starts)}")
    if len(leads) == 0:
        print(f"No lead-time computed with threshold={args.threshold} (score never crossed before start).")
    else:
        L = np.array(leads, dtype=int)
        print(
            "Lead time (steps before event start, using score threshold) "
            f"mean={L.mean():.2f} median={np.median(L):.1f} min={L.min()} max={L.max()} n={len(L)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
