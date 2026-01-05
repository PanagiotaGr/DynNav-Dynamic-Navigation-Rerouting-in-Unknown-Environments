#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Reshape world_template_bench merged CSV from wide to long format.")
    p.add_argument("--input", "-i", default="results/multiseed/merged.csv", help="Input merged CSV.")
    p.add_argument("--out", "-o", default="results/multiseed/merged_long.csv", help="Output long-format CSV.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.out)

    df = pd.read_csv(in_path)

    base_cols = ["template", "pair_id", "Seed", "I_start", "I_goal"]
    base_cols = [c for c in base_cols if c in df.columns]

    # Build long rows for each policy
    long_frames = []

    # fixed policy
    if "fixed_success" in df.columns:
        cols = base_cols + [c for c in ["fixed_tau0", "fixed_success", "fixed_reason"] if c in df.columns]
        fixed = df[cols].copy()
        fixed = fixed.rename(
            columns={
                "fixed_tau0": "tau0",
                "fixed_success": "success",
                "fixed_reason": "reason",
            }
        )
        fixed["Policy"] = "fixed"
        long_frames.append(fixed)

    # safe policy
    if "safe_success" in df.columns:
        cols = base_cols + [c for c in ["safe_success", "safe_mode", "safe_tau_used", "safe_tau_gap", "safe_reason"] if c in df.columns]
        safe = df[cols].copy()
        safe = safe.rename(
            columns={
                "safe_success": "success",
                "safe_mode": "mode",
                "safe_tau_used": "tau_used",
                "safe_tau_gap": "tau_gap",
                "safe_reason": "reason",
            }
        )
        safe["Policy"] = "safe"
        long_frames.append(safe)

    # minimax policy
    if "mm_success" in df.columns:
        cols = base_cols + [c for c in ["mm_success", "tau_star", "tau_request"] if c in df.columns]
        mm = df[cols].copy()
        mm = mm.rename(
            columns={
                "mm_success": "success",
                "tau_star": "tau_star",
                "tau_request": "tau_request",
            }
        )
        mm["Policy"] = "minimax"
        long_frames.append(mm)

    if not long_frames:
        raise SystemExit("No policy columns found (expected fixed_success / safe_success / mm_success).")

    out = pd.concat(long_frames, ignore_index=True)

    # Ensure success is numeric (0/1)
    out["success"] = pd.to_numeric(out["success"], errors="coerce").fillna(0).astype(int)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Wrote long-format CSV: {out_path}")
    print(f"Rows: {len(out)} | Columns: {list(out.columns)}")
    print(out.head(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

