"""
merge_results.py
=================
Concatenates the per-N block_a_results_N####.csv files produced by the
40 sweep_array.cmd subjobs into one block_a_results.csv, then re-derives
block_a_summary.csv from the FULL merged table with the original
build_summary() (so per-class means/stds are computed once, correctly,
across the whole N sweep, not per-subjob).

Usage
-----
    python merge_results.py [--dir DIR] [--out block_a_results.csv]
                             [--summary block_a_summary.csv]
"""
import argparse
import glob
import os
import sys

import pandas as pd

# Reuse the exact aggregation logic from the main script rather than
# re-deriving it here, so the merged summary is identical to what a
# single non-cluster run would have produced.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from block_a_solver_benchmark import build_summary, print_summary_table  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="Directory holding block_a_results_N*.csv")
    ap.add_argument("--pattern", default="block_a_results_N*.csv")
    ap.add_argument("--out", default="block_a_results.csv")
    ap.add_argument("--summary", default="block_a_summary.csv")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not paths:
        print(f"No files matched {args.pattern} in {args.dir} — did the "
              f"array job finish? (check qstat -an / blockA_sweep_N*.log)",
              file=sys.stderr)
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

    summary = build_summary(merged)
    summary_path = os.path.join(args.dir, args.summary)
    summary.to_csv(summary_path, index=False)
    print(f"Merged summary -> {summary_path}")

    print_summary_table(summary)


if __name__ == "__main__":
    main()
