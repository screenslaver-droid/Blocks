"""
merge_results_b3.py
====================
Concatenates the 40 per-N block_b3_results_N####.csv files from
sweep_array_b3.cmd into one block_b3_results.csv, re-derives
block_b3_summary.csv with build_summary_b3() over the FULL merged table
(so per-class eps_int mean/std are computed once, correctly, across the
whole N sweep -- not per-subjob), and re-runs the elbow heuristic.

Usage
-----
    python merge_results_b3.py [--dir DIR] [--out block_b3_results.csv]
                                [--summary block_b3_summary.csv]
"""
import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from block_b3_integration_accuracy import (  # noqa: E402
    build_summary_b3, detect_elbow, print_summary_table,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="Directory holding block_b3_results_N*.csv")
    ap.add_argument("--pattern", default="block_b3_results_N*.csv")
    ap.add_argument("--out", default="block_b3_results.csv")
    ap.add_argument("--summary", default="block_b3_summary.csv")
    ap.add_argument("--elbow-out", default="block_b3_elbow.csv")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not paths:
        print(f"No files matched {args.pattern} in {args.dir} — did "
              f"sweep_array_b3.cmd finish? (check qstat -an / "
              f"blockB3_sweep_N*.log)", file=sys.stderr)
        sys.exit(1)

    print(f"Merging {len(paths)} files:")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        print(f"  {p}: {len(df):,} rows")
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    dupe_key = ["event_id", "N", "seed"] if {"event_id", "N", "seed"}.issubset(merged.columns) else None
    if dupe_key:
        before = len(merged)
        merged = merged.drop_duplicates(subset=dupe_key, keep="last")
        if len(merged) != before:
            print(f"  Dropped {before - len(merged)} duplicate rows "
                  f"(re-run of a subset?) keyed on {dupe_key}")

    out_path = os.path.join(args.dir, args.out)
    merged.to_csv(out_path, index=False)
    print(f"\nMerged results -> {out_path}  ({len(merged):,} rows total)")

    summary = build_summary_b3(merged)
    summary_path = os.path.join(args.dir, args.summary)
    summary.to_csv(summary_path, index=False)
    print(f"Merged summary -> {summary_path}")

    elbow = detect_elbow(summary)
    elbow_path = os.path.join(args.dir, args.elbow_out)
    elbow.to_csv(elbow_path, index=False)
    print(f"Elbow heuristic -> {elbow_path}")

    print_summary_table(summary, elbow)


if __name__ == "__main__":
    main()
