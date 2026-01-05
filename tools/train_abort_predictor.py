#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a simple predictive-abort model (logistic regression, no sklearn)."
    )
    p.add_argument("--input", "-i", required=True, help="CSV time-series log file.")
    p.add_argument("--out", "-o", default="results/abort_model.npz", help="Output model path (.npz).")

    p.add_argument("--time-col", default=None, help="Optional time column (else uses row index).")
    p.add_argument("--signals", nargs="+", required=True, help="Signal columns to use as features.")
    p.add_argument("--event-col", required=True, help="Column indicating event (0/1 or bool).")

    p.add_argument("--horizon", type=int, default=10, help="Predict event within next k steps (default: 10).")
    p.add_argument("--window", type=int, default=15, help="Window length for trend features (default: 15).")

    # NEW: label mode
    p.add_argument(
        "--label-mode",
        choices=["future_any", "future_start"],
        default="future_start",
        help=(
            "future_any: y[t]=1 if event occurs in (t+1..t+horizon). "
            "future_start: y[t]=1 if an event START (0->1) occurs in (t+1..t+horizon). "
            "Default: future_start (recommended)."
        ),
    )

    p.add_argument("--lr", type=float, default=0.05, help="Learning rate (default: 0.05).")
    p.add_argument("--epochs", type=int, default=400, help="Training epochs (default: 400).")
    p.add_argument("--l2", type=float, default=1e-3, help="L2 regularization (default: 1e-3).")
    p.add_argument("--train-frac", type=float, default=0.8, help="Train fraction (default: 0.8).")
    return p.parse_args()


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def event_starts(event: np.ndarray) -> np.ndarray:
    """start[t]=1 when event switches 0->1 at t."""
    e = (event > 0.5).astype(np.int64)
    s = np.zeros_like(e, dtype=np.int64)
    if len(e) >= 2:
        s[1:] = (e[1:] == 1) & (e[:-1] == 0)
    return s.astype(np.float64)


def make_label_future(event: np.ndarray, horizon: int, mode: str) -> np.ndarray:
    """
    future_any: y[t]=1 if any event occurs in (t+1..t+horizon)
    future_start: y[t]=1 if any event START (0->1) occurs in (t+1..t+horizon)
    """
    n = len(event)
    e = (event > 0.5).astype(np.float64)
    if mode == "future_start":
        e = event_starts(e)

    y = np.zeros(n, dtype=np.float64)
    for t in range(n):
        lo = t + 1
        hi = min(n, t + 1 + horizon)
        y[t] = 1.0 if np.any(e[lo:hi] > 0.5) else 0.0
    return y


def rolling_slope(x: np.ndarray) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    t = np.arange(n, dtype=np.float64)
    t_mean = t.mean()
    denom = np.sum((t - t_mean) ** 2) + 1e-12
    a = np.sum((t - t_mean) * (x - x.mean())) / denom
    return float(a)


def build_features(df: pd.DataFrame, signals: list[str], window: int) -> tuple[np.ndarray, list[str]]:
    n = len(df)
    feats = []
    names = []

    for s in signals:
        v = pd.to_numeric(df[s], errors="coerce").to_numpy(dtype=np.float64)

        feats.append(v.copy())
        names.append(f"{s}_cur")

        roll_mean = np.full(n, np.nan, dtype=np.float64)
        roll_std = np.full(n, np.nan, dtype=np.float64)
        roll_slope = np.full(n, np.nan, dtype=np.float64)

        for i in range(n):
            j0 = max(0, i - window + 1)
            w = v[j0 : i + 1]
            w = w[np.isfinite(w)]
            if len(w) == 0:
                continue
            roll_mean[i] = float(np.mean(w))
            roll_std[i] = float(np.std(w, ddof=1)) if len(w) > 1 else 0.0
            roll_slope[i] = rolling_slope(w)

        feats.extend([roll_mean, roll_std, roll_slope])
        names.extend([f"{s}_meanW", f"{s}_stdW", f"{s}_slopeW"])

    X = np.vstack(feats).T
    return X, names


def standardize_train(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(X_train, axis=0)
    sigma = np.nanstd(X_train, axis=0)
    sigma = np.where(sigma < 1e-9, 1.0, sigma)
    return mu, sigma


def apply_standardize(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    Z = (X - mu) / sigma
    Z = np.where(np.isfinite(Z), Z, 0.0)
    return Z


def train_logreg(X: np.ndarray, y: np.ndarray, lr: float, epochs: int, l2: float) -> tuple[np.ndarray, float]:
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0

    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        grad_w = (X.T @ (p - y)) / n + l2 * w
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w
        b -= lr * grad_b

    return w, b


def eval_metrics(p: np.ndarray, y: np.ndarray) -> dict:
    eps = 1e-12
    ll = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    yhat = (p >= 0.5).astype(np.float64)
    acc = float(np.mean(yhat == y))
    base = float(np.mean(y))
    return {"logloss": float(ll), "acc@0.5": acc, "base_rate": base}


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}")
        return 2

    df = pd.read_csv(in_path)

    missing = [c for c in args.signals + [args.event_col] if c not in df.columns]
    if missing:
        print("ERROR: missing columns:", missing)
        print("Available:", df.columns.tolist())
        return 2

    event = pd.to_numeric(df[args.event_col], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    event = (event > 0.5).astype(np.float64)

    y = make_label_future(event, horizon=args.horizon, mode=args.label_mode)

    X, feat_names = build_features(df, args.signals, window=args.window)

    valid_rows = np.any(np.isfinite(X), axis=1)
    X = X[valid_rows]
    y = y[valid_rows]

    n = len(y)
    if n < 50:
        print(f"WARNING: only {n} valid rows. Consider using a longer log.")
    split = int(args.train_frac * n)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    mu, sigma = standardize_train(X_train)
    Z_train = apply_standardize(X_train, mu, sigma)
    Z_test = apply_standardize(X_test, mu, sigma)

    w, b = train_logreg(Z_train, y_train, lr=args.lr, epochs=args.epochs, l2=args.l2)

    p_train = sigmoid(Z_train @ w + b)
    p_test = sigmoid(Z_test @ w + b) if len(y_test) else np.array([])

    print("Train metrics:", eval_metrics(p_train, y_train))
    if len(y_test):
        print("Test metrics :", eval_metrics(p_test, y_test))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        w=w,
        b=b,
        mu=mu,
        sigma=sigma,
        feat_names=np.array(feat_names, dtype=object),
        signals=np.array(args.signals, dtype=object),
        window=args.window,
        horizon=args.horizon,
        event_col=args.event_col,
        label_mode=args.label_mode,
    )
    print(f"Saved model -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
