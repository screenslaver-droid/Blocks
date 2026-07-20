"""
plot_results.py
================
Reads the merged block_a_summary.csv (per lifecycle_class x N mean/std,
from build_summary() / merge_results.py) and produces four figures, one
small-multiple panel per lifecycle_class in each:

  1. fig_nfe_vs_N.png          NFE: IMEX total vs DOPRI5, mean +/- std, vs N
  2. fig_wall_time_vs_N.png    Wall-clock seconds: IMEX vs DOPRI5, vs N
  3. fig_nfe_ratio_vs_N.png    NFE ratio (DOPRI5 / IMEX), vs N
  4. fig_lambda_max_vs_N.png   lambda_max(L): topological vs D-weighted, vs N
                                (blocks.tex Figure A-4)

Deliberately only depends on pandas + matplotlib + the summary CSV — it
does NOT import block_a_solver_benchmark.py, so it can run anywhere
(laptop, login node, compute node) without needing jax/diffrax/h5py/the
DEM-fetch stack installed.

Usage
-----
    python plot_results.py                                  # defaults
    python plot_results.py --summary block_a_summary.csv --out-dir figures
    python plot_results.py --log-y                           # log-scale y axes
    python plot_results.py --classes RAPID_GROWTH STEADY     # subset of classes
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless -- no display needed on a login/compute node
import matplotlib.pyplot as plt
import pandas as pd


def _grid_shape(n: int) -> tuple:
    """Small-multiple grid dimensions for n classes: prefer <=4 columns."""
    ncols = min(4, n) or 1
    nrows = -(-n // ncols)  # ceil
    return nrows, ncols


def _panel_grid(n_classes: int, figsize_per_panel=(4.2, 3.2)):
    nrows, ncols = _grid_shape(n_classes)
    fig, axes = plt.subplots(
        nrows, ncols, squeeze=False,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        sharex=True,
    )
    return fig, axes.flatten(), nrows, ncols


def _hide_unused(axes, n_used: int):
    for ax in axes[n_used:]:
        ax.set_visible(False)


def _plot_two_series(ax, sub: pd.DataFrame, x: str,
                      series: list, log_y: bool):
    """series: list of (mean_col, std_col_or_None, label, color)"""
    for mean_col, std_col, label, color in series:
        if mean_col not in sub.columns:
            continue
        y = sub[mean_col]
        ax.plot(sub[x], y, marker="o", markersize=3, linewidth=1.6,
                 label=label, color=color)
        if std_col and std_col in sub.columns:
            std = sub[std_col].fillna(0.0)
            lo = (y - std).clip(lower=0) if log_y else (y - std)
            ax.fill_between(sub[x], lo, y + std, alpha=0.15, color=color, linewidth=0)
    if log_y:
        ax.set_yscale("log")


def plot_nfe_vs_N(df: pd.DataFrame, classes: list, out_path: str, log_y: bool):
    fig, axes, nrows, ncols = _panel_grid(len(classes))
    for ax, cls in zip(axes, classes):
        sub = df[df["lifecycle_class"] == cls].sort_values("N")
        _plot_two_series(ax, sub, "N", [
            ("nfe_imex_total_mean", "nfe_imex_total_std", "IMEX (total)", "tab:blue"),
            ("nfe_dopri5_mean",     "nfe_dopri5_std",     "DOPRI5",       "tab:orange"),
        ], log_y)
        ax.set_title(cls, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes[-ncols:]:
        ax.set_xlabel("N")
    for ax in axes[::ncols]:
        ax.set_ylabel("NFE" + (" (log)" if log_y else ""))
    _hide_unused(axes, len(classes))
    fig.suptitle("Block A: NFE vs N — IMEX Strang vs DOPRI5, by lifecycle class")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_wall_time_vs_N(df: pd.DataFrame, classes: list, out_path: str, log_y: bool):
    fig, axes, nrows, ncols = _panel_grid(len(classes))
    for ax, cls in zip(axes, classes):
        sub = df[df["lifecycle_class"] == cls].sort_values("N")
        _plot_two_series(ax, sub, "N", [
            ("wall_imex_mean",   "wall_imex_std",   "IMEX (total)", "tab:blue"),
            ("wall_dopri5_mean", "wall_dopri5_std", "DOPRI5",       "tab:orange"),
        ], log_y)
        ax.set_title(cls, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes[-ncols:]:
        ax.set_xlabel("N")
    for ax in axes[::ncols]:
        ax.set_ylabel("wall time (s)" + (" (log)" if log_y else ""))
    _hide_unused(axes, len(classes))
    fig.suptitle("Block A: wall-clock time vs N — IMEX Strang vs DOPRI5, by lifecycle class")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_nfe_ratio_vs_N(df: pd.DataFrame, classes: list, out_path: str, log_y: bool):
    fig, axes, nrows, ncols = _panel_grid(len(classes))
    for ax, cls in zip(axes, classes):
        sub = df[df["lifecycle_class"] == cls].sort_values("N")
        _plot_two_series(ax, sub, "N", [
            ("nfe_ratio_mean", "nfe_ratio_std", "DOPRI5 / IMEX", "tab:green"),
        ], log_y)
        ax.axhline(1.0, color="gray", linewidth=1, linestyle="--")
        ax.set_title(cls, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes[-ncols:]:
        ax.set_xlabel("N")
    for ax in axes[::ncols]:
        ax.set_ylabel("NFE ratio" + (" (log)" if log_y else ""))
    _hide_unused(axes, len(classes))
    fig.suptitle("Block A: NFE ratio (DOPRI5 / IMEX) vs N, by lifecycle class "
                 "— note nfe_reaction (and so nfe_imex_total/ratio) is a nominal "
                 "stage-count proxy, not exact; see README", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_lambda_max_vs_N(df: pd.DataFrame, classes: list, out_path: str, log_y: bool):
    fig, axes, nrows, ncols = _panel_grid(len(classes))
    for ax, cls in zip(axes, classes):
        sub = df[df["lifecycle_class"] == cls].sort_values("N")
        _plot_two_series(ax, sub, "N", [
            ("lambda_max_topo_mean", "lambda_max_topo_std", "unit weights (topo)",  "tab:purple"),
            ("lambda_max_w_mean",    "lambda_max_w_std",    "D * RBF (weighted)",   "tab:red"),
        ], log_y)
        ax.set_title(cls, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes[-ncols:]:
        ax.set_xlabel("N")
    for ax in axes[::ncols]:
        ax.set_ylabel("lambda_max(L)" + (" (log)" if log_y else ""))
    _hide_unused(axes, len(classes))
    fig.suptitle("Block A / Figure A-4: lambda_max(L) vs N, by lifecycle class")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default="block_a_summary.csv",
                     help="Path to the merged summary CSV (build_summary() output)")
    ap.add_argument("--out-dir", default="figures",
                     help="Directory to write the four PNG figures into")
    ap.add_argument("--classes", nargs="*", default=None,
                     help="Subset of lifecycle_class values to plot "
                          "(default: every class present in the summary)")
    ap.add_argument("--log-y", action="store_true",
                     help="Log-scale the y axis on every figure (NFE/wall-time "
                          "commonly span orders of magnitude across N=50..2000)")
    args = ap.parse_args()

    if not os.path.exists(args.summary):
        print(f"Summary file not found: {args.summary}\n"
              f"(run merge_results.py first if this is a cluster sweep)",
              file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.summary)
    required = {"lifecycle_class", "N"}
    missing = required - set(df.columns)
    if missing:
        print(f"Summary CSV is missing expected columns: {missing}", file=sys.stderr)
        sys.exit(1)

    classes = args.classes or sorted(df["lifecycle_class"].unique().tolist())
    unknown = set(classes) - set(df["lifecycle_class"].unique())
    if unknown:
        print(f"WARNING: these --classes aren't in the summary and will be "
              f"skipped: {sorted(unknown)}", file=sys.stderr)
        classes = [c for c in classes if c not in unknown]
    if not classes:
        print("No classes to plot.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Classes: {classes}")
    print(f"N range in summary: {df['N'].min()}..{df['N'].max()} "
          f"({df['N'].nunique()} distinct values)")

    jobs = [
        ("fig_nfe_vs_N.png",       plot_nfe_vs_N),
        ("fig_wall_time_vs_N.png", plot_wall_time_vs_N),
        ("fig_nfe_ratio_vs_N.png", plot_nfe_ratio_vs_N),
        ("fig_lambda_max_vs_N.png", plot_lambda_max_vs_N),
    ]
    for fname, fn in jobs:
        out_path = os.path.join(args.out_dir, fname)
        fn(df, classes, out_path, args.log_y)
        print(f"  wrote {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()