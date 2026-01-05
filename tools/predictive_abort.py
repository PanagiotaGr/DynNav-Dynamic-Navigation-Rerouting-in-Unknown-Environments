#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import os
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Run predictive abort inference using a trained model.")
    p.add_argument("--model", "-m", required=True, help="Path to model .npz from train_abort_predictor.py")
    p.add_argument("--input", "-i", required=True, help="CSV time-series log file.")
    p.add_argument("--threshold", type=float, default=0.7, help="Abort threshold on p_fail (default: 0.7)")
    p.add_argument("--out", "-o", default="results/abort_predictions.csv", help="Output CSV with p_fail/abort.")
    p.add_argument("--verify-write", action="store_true", help="After writing, read back and verify non-empty.")
    return p.parse_args()


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def rolling_slope(x: np.ndarray) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    t = np.arange(n, dtype=np.float64)
    t_mean = t.mean()
    denom = np.sum((t - t_mean) ** 2) + 1e-12
    a = np.sum((t - t_mean) * (x - x.mean())) / denom
    return float(a)


def build_features_for_row(df: pd.DataFrame, signals: list[str], window: int, i: int) -> np.ndarray:
    feats = []
    for s in signals:
        v = pd.to_numeric(df[s], errors="coerce").to_numpy(dtype=np.float64)

        feats.append(v[i])

        j0 = max(0, i - window + 1)
        w = v[j0 : i + 1]
        w = w[np.isfinite(w)]
        if len(w) == 0:
            feats.extend([np.nan, np.nan, np.nan])
        else:
            feats.append(float(np.mean(w)))
            feats.append(float(np.std(w, ddof=1)) if len(w) > 1 else 0.0)
            feats.append(rolling_slope(w))

    return np.array(feats, dtype=np.float64)


def atomic_to_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_csv(tmp, index=False)

    # fsync to be safe
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())

    tmp.replace(out_path)


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    in_path = Path(args.input)
    out_path = Path(args.out)

    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}")
        return 2
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}")
        return 2

    M = np.load(model_path, allow_pickle=True)
    w = M["w"].astype(np.float64)
    b = float(M["b"])
    mu = M["mu"].astype(np.float64)
    sigma = M["sigma"].astype(np.float64)
    signals = [str(x) for x in M["signals"].tolist()]
    window = int(M["window"])

    df = pd.read_csv(in_path)

    missing = [c for c in signals if c not in df.columns]
    if missing:
        print("ERROR: missing signal columns in input:", missing)
        print("Model expects:", signals)
        print("Available:", df.columns.tolist())
        return 2

    p_fail = np.zeros(len(df), dtype=np.float64)
    abort = np.zeros(len(df), dtype=np.int64)

    for i in range(len(df)):
        x = build_features_for_row(df, signals, window, i)
        z = (x - mu) / sigma
        z = np.where(np.isfinite(z), z, 0.0)
        p = float(sigmoid(z @ w + b))
        p_fail[i] = p
        abort[i] = 1 if p >= args.threshold else 0

    out = df.copy()
    out["p_fail_k"] = p_fail
    out["abort_pred"] = abort

    atomic_to_csv(out, out_path)

    print(f"Wrote predictions -> {out_path}")
    print("Abort rate:", float(out["abort_pred"].mean()))

    if args.verify_write:
        if out_path.stat().st_size == 0:
            print(f"ERROR: wrote 0 bytes to {out_path}")
            return 2
        back = pd.read_csv(out_path)
        print(f"[verify] read back rows={len(back)} cols={len(back.columns)} size={out_path.stat().st_size} bytes")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
