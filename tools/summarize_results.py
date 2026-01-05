#!/usr/bin/env python3
"""
summarize_results.py

Reads a CSV file with experiment results and prints a compact summary.
Optionally writes the summary to a CSV.

Features:
- Auto-detect numeric metric columns (or choose with --metrics)
- Group-by a column (e.g., Algorithm)
- Optional success rate from a success column
- Optional warnings when group sample size is too small
- Optional 95% bootstrap confidence intervals (CI) for metric means

Usage:
  python3 tools/summarize_results.py --input benchmark_results.csv
  python3 tools/summarize_results.py --input benchmark_results.csv --group-by Algorithm
  python3 tools/summarize_results.py --input benchmark_results.csv --group-by Algorithm --warn-small-n
  python3 tools/summarize_results.py --input benchmark_results.csv --group-by Algorithm --ci --out results/summary.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize experiment results from a CSV.")
    p.add_argument("--input", "-i", required=True, help="Path to input CSV (e.g., benchmark_results.csv).")
    p.add_argument("--group-by", "-g", default=None, help="Column name to group by (e.g., Algorithm).")
    p.add_argument(
        "--metrics",
        "-m",
        nargs="*",
        default=None,
        help="Numeric metric columns to summarize (default: auto-detect numeric columns).",
    )
    p.add_argument(
        "--success-col",
        default=None,
        help="Optional column that indicates success (boolean or 0/1). If provided, we compute success rate.",
    )
    p.add_argument("--out", "-o", default=None, help="Optional output CSV path to write the summary table.")

    # New: CI + warnings
    p.add_argument(
        "--ci",
        action="store_true",
        help="Compute 95% bootstrap confidence intervals for metric means (requires n>=2 per group).",
    )
    p.add_argument(
        "--ci-iters",
        type=int,
        default=2000,
        help="Bootstrap iterations for CI (default: 2000).",
    )
    p.add_argument(
        "--warn-small-n",
        action="store_true",
        help="Add a warning column if any group has n<2 (std and CI are not meaningful).",
    )
    return p.parse_args()


def auto_numeric_metrics(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    exclude = set(exclude or [])
    numeric_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
    return numeric_cols


def coerce_success(series: pd.Series) -> pd.Series:
    """
    Convert common representations to boolean:
    - bool
    - 0/1
    - "true"/"false", "yes"/"no"
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0.0

    s = series.astype(str).str.strip().str.lower()
    truthy = {"true", "1", "yes", "y", "success", "ok"}
    falsy = {"false", "0", "no", "n", "fail", "failure", "error"}
    return s.apply(lambda x: True if x in truthy else (False if x in falsy else False))


def bootstrap_mean_ci(x: pd.Series, iters: int = 2000, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Non-parametric bootstrap CI for the mean.
    Returns (lower, upper) bounds for (1-alpha) CI.
    Requires at least 2 valid samples.
    """
    x = pd.to_numeric(x, errors="coerce").dropna().to_numpy()
    n = len(x)
    if n < 2:
        return (float("nan"), float("nan"))

    import numpy as np

    rng = np.random.default_rng(0)  # deterministic for reproducibility
    means = np.empty(iters, dtype=float)
    for i in range(iters):
        sample = rng.choice(x, size=n, replace=True)
        means[i] = float(sample.mean())

    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def summarize(
    df: pd.DataFrame,
    group_by: Optional[str],
    metrics: List[str],
    success_col: Optional[str],
    compute_ci: bool,
    ci_iters: int,
    warn_small_n: bool,
) -> pd.DataFrame:
    if group_by is not None and group_by not in df.columns:
        raise ValueError(f"--group-by column '{group_by}' not found. Available: {list(df.columns)}")

    # Base grouping
    if group_by is None:
        grouped = [(None, df)]
    else:
        grouped = list(df.groupby(group_by, dropna=False))

    rows = []
    for key, g in grouped:
        row = {}
        if group_by is not None:
            row[group_by] = key

        n = int(len(g))
        row["n"] = n

        # Warning for small n
        if warn_small_n:
            row["warning_small_n"] = 1 if n < 2 else 0

        # Success rate (optional)
        if success_col:
            if success_col not in g.columns:
                raise ValueError(f"--success-col '{success_col}' not found. Available: {list(df.columns)}")
            succ = coerce_success(g[success_col])
            row["success_rate"] = float(succ.mean())

        # Metrics: mean/std (+ optional CI)
        for m in metrics:
            if m not in g.columns:
                continue  # skip missing columns silently

            col = pd.to_numeric(g[m], errors="coerce")
            valid = col.dropna()

            row[f"{m}_mean"] = float(valid.mean()) if len(valid) > 0 else float("nan")
            row[f"{m}_std"] = float(valid.std(ddof=1)) if len(valid) > 1 else 0.0

            if compute_ci:
                if len(valid) >= 2:
                    lo, hi = bootstrap_mean_ci(valid, iters=ci_iters)
                    row[f"{m}_ci95_lo"] = lo
                    row[f"{m}_ci95_hi"] = hi
                else:
                    row[f"{m}_ci95_lo"] = float("nan")
                    row[f"{m}_ci95_hi"] = float("nan")

        rows.append(row)

    out = pd.DataFrame(rows)

    # Sort best-effort
    if group_by is not None and group_by in out.columns:
        out = out.sort_values(by=[group_by]).reset_index(drop=True)

    return out


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: input file not found: {in_path}", file=sys.stderr)
        return 2

    df = pd.read_csv(in_path)

    exclude: List[str] = []
    if args.group_by:
        exclude.append(args.group_by)
    if args.success_col:
        exclude.append(args.success_col)

    metrics = args.metrics or auto_numeric_metrics(df, exclude=exclude)

    try:
        summary_df = summarize(
            df,
            args.group_by,
            metrics,
            args.success_col,
            compute_ci=args.ci,
            ci_iters=args.ci_iters,
            warn_small_n=args.warn_small_n,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    with pd.option_context("display.max_columns", 200, "display.width", 160):
        print(summary_df)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_path, index=False)
        print(f"\nWrote summary to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
