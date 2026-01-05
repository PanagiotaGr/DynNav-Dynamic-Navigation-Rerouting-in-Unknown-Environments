#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Run a benchmark script multiple times with different seeds.")
    p.add_argument("--script", required=True, help="Benchmark script to run (e.g., run_world_template_bench.py).")
    p.add_argument("--seeds", type=int, default=20, help="How many seeds/runs (default: 20).")
    p.add_argument("--seed0", type=int, default=0, help="Starting seed (default: 0).")
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use (default: current venv python).",
    )

    # NEW: where the benchmark writes its CSV (we will copy it after each run)
    p.add_argument(
        "--produced-csv",
        required=True,
        help="Path to the CSV produced by the benchmark script (e.g., world_template_bench.csv).",
    )
    p.add_argument(
        "--out-dir",
        default="results/multiseed",
        help="Directory to store per-seed CSVs and the merged output (default: results/multiseed).",
    )
    p.add_argument(
        "--merged-name",
        default="merged.csv",
        help="Filename for merged CSV inside out-dir (default: merged.csv).",
    )
    p.add_argument(
        "--add-seed-col",
        action="store_true",
        help="Add a Seed column to each row before merging.",
    )

    p.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to the benchmark script after '--'. Example: -- --world hard --episodes 1",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    script_path = Path(args.script)
    if not script_path.exists():
        print(f"ERROR: script not found: {script_path}", file=sys.stderr)
        return 2

    produced_csv = Path(args.produced_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extra = args.extra_args or []
    if len(extra) > 0 and extra[0] == "--":
        extra = extra[1:]

    per_seed_paths = []

    for i in range(args.seeds):
        seed = args.seed0 + i
        env = os.environ.copy()
        env["SEED"] = str(seed)

        cmd = [args.python, str(script_path), *extra]
        print(f"\n[Run {i+1}/{args.seeds}] SEED={seed}")
        print(" ".join(cmd))

        ret = subprocess.call(cmd, env=env)
        if ret != 0:
            print(f"ERROR: run failed with exit code {ret} at seed={seed}", file=sys.stderr)
            return ret

        # After run: copy the produced CSV into a per-seed file
        if not produced_csv.exists():
            print(
                f"ERROR: expected produced CSV not found after run: {produced_csv}\n"
                f"Tip: check what the script prints as 'Saved: ...' and pass that to --produced-csv",
                file=sys.stderr,
            )
            return 2

        seed_path = out_dir / f"{produced_csv.stem}_seed_{seed:03d}.csv"
        shutil.copy2(produced_csv, seed_path)
        per_seed_paths.append(seed_path)
        print(f"[saved] {seed_path}")

    # Merge all per-seed CSVs
    frames = []
    for p in per_seed_paths:
        df = pd.read_csv(p)
        if args.add_seed_col:
            # try to parse seed from filename suffix
            try:
                seed_str = p.stem.split("_seed_")[-1]
                df["Seed"] = int(seed_str)
            except Exception:
                df["Seed"] = None
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged_path = out_dir / args.merged_name
    merged.to_csv(merged_path, index=False)
    print(f"\nMerged {len(per_seed_paths)} files -> {merged_path} (rows={len(merged)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

