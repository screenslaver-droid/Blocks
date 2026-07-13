"""
steady_spatial_diagnostic.py
============================
Diagnostic for the STEADY class: tests whether switching from t=0 spatial
metrics to time-averaged metrics would meaningfully change subclass assignments,
and whether either version produces sharper J(N) peaks.

Three questions answered, in order:

  Q1. Rank-correlation check
      How similar are t=0 vs time-averaged spatial metrics within STEADY?
      If Spearman r > ~0.85, temporal averaging won't move enough events
      to change the subclass distribution.

  Q2. Subclass stability check
      For events that *would* change subclass under temporal averaging,
      do their J(N) curves look different from events that don't change?
      (Requires per-event N-sweep CSV from diagnostic_n_sweep.py.)

  Q3. J(N) plateau width distribution
      Measures the width of the 95%-of-peak plateau in J(N) per event.
      Narrow peaks -> N* is well-defined. Wide plateaus -> flatness is intrinsic.
      Compares changers vs non-changers.

Inputs
------
  --catalogue   Path to event_catalogue.csv from Growth_Decay_Classify.py
  --sweep_dir   Directory containing per-event sweep CSVs.
                File naming is auto-detected: the script lists all *.csv files
                in sweep_dir and matches them to event IDs by substring search,
                so filenames like <id>_sweep.csv, sweep_<id>.csv, or <id>.csv
                all work. Pass --sweep_name_debug to print what was found.
  --data_root   SEVIR data root — the folder that DIRECTLY contains CATALOG.csv
                and the SEVIR_VIL_* subdirectories. The script searches
                data_root, data_root/2019, and data_root/.. automatically.
  --catalog_csv Full path to CATALOG.csv (defaults to data_root/CATALOG.csv)
  --out_dir     Output directory for plots and summary CSV

Usage
-----
  # Full run (Q1 + Q2 + Q3):
  python steady_spatial_diagnostic.py ^
      --catalogue event_catalogue.csv ^
      --sweep_dir sweep_results/ ^
      --data_root C:/path/to/SEVIR ^
      --out_dir steady_diagnostic_out/

  # Q2 + Q3 only (skip raw VIL re-read):
  python steady_spatial_diagnostic.py ^
      --catalogue event_catalogue.csv ^
      --sweep_dir sweep_results/ ^
      --skip_recompute

  # Debug sweep file discovery:
  python steady_spatial_diagnostic.py ^
      --catalogue event_catalogue.csv ^
      --sweep_dir sweep_results/ ^
      --skip_recompute --sweep_name_debug
"""

import argparse
import logging
import os
import warnings

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    logging.warning("matplotlib not found — plots will be skipped.")

try:
    from skimage.measure import label as sk_label, regionprops
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    logging.warning("scikit-image not found — Q1 spatial re-computation disabled.")

warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Constants (match Growth_Decay_Classify.py) ───────────────────────────────
NOISE_THRESH      = 20
ACTIVE_THRESH     = 20
ISOLATED_MAX_AREA = 5_000
MCS_MIN_AREA      = 30_000

# ── Joint-score constants (must mirror Build_summary_df_hpc_v2 exactly) ──────
# These are used to reconstruct J(N) on-the-fly from raw sweep metric columns,
# because per-event *_sweep.csv files do NOT contain a pre-computed J column.
# J is only produced by build_n_summary_hpc.py at the aggregation stage.
_HW               = 384
_V_MAX            = 9
_T_FC             = 12
_W_ISV, _W_BR, _W_FAE, _W_MDV = 0.40, 0.30, 0.15, 0.15
_ALPHA_R, _ALPHA_D, _ALPHA_T   = 0.10, 0.60, 0.30
_BETA, _S_THRESH, _GAMMA       = 5.0, 0.60, 0.10
_DEFAULT_OUTER_ALPHA            = 0.70
_SOFT_PEAK_RATIO                = 0.95

# Lifecycle-adaptive outer_alpha (matches LIFECYCLE_ALPHA in build summary)
_LIFECYCLE_ALPHA = {
    "RAPID_GROWTH": 1.40,
    "GROWTH_DECAY": 1.20,
    "EPISODIC":     0.95,
    "PLATEAU":      0.65,
    "RAPID_DECAY":  0.65,
    "STEADY":       0.55,
    "QUIESCENT":    0.35,
}

# Raw-metric column names written by diagnostic_n_sweep_hpc.py
# Listed in preference order; first match wins.
_ISV_COLS = ("intra_sp_var_mean",)
_LFS_COLS = ("LFS",)
_FAE_COLS = ("FAE",)
_MDV_COLS = ("MDV",)
_BR_COLS  = ("boundary_recall_mean",)
_TBR_COLS = ("temporal_br_std",)


# ── Subclass assignment ───────────────────────────────────────────────────────
def assign_spatial_subclass(n_cells: float, max_area: float,
                             active_frac: float) -> str:
    if active_frac < 0.01:
        return "quiescent"
    if max_area >= MCS_MIN_AREA:
        return "mcs"
    if n_cells == 1 and max_area < ISOLATED_MAX_AREA:
        return "isolated_cell"
    return "cluster"


# ============================================================================ #
# PATH RESOLUTION  —  robust multi-candidate search
# ============================================================================ #
def _candidate_roots(data_root: str) -> list:
    """
    Return a list of directories to try as the SEVIR root.
    Covers the common mistake of passing .../SEVIR/2019 instead of .../SEVIR.
    """
    candidates = [data_root]
    # One level up (handles data_root = .../SEVIR/2019)
    parent = os.path.dirname(data_root)
    if parent and parent != data_root:
        candidates.append(parent)
    # One level down (handles data_root = .../SEVIR, files in .../SEVIR/2019/...)
    for sub in ["2019", "2018", "data"]:
        p = os.path.join(data_root, sub)
        if os.path.isdir(p):
            candidates.append(p)
    return candidates


def resolve_hdf5_path(data_root: str, catalog_filename: str) -> str | None:
    """
    Try multiple root candidates and multiple path-reconstruction strategies
    for a given catalog file_name entry.

    Strategies per root:
      A) root / catalog_filename                  (as-is)
      B) root / basename(catalog_filename)        (strip any leading dirs)
      C) root / parts[-2] / parts[-3] / parts[-1] (reordered triplet from
                                                   Growth_Decay_Classify.py)
    """
    norm = catalog_filename.replace("\\", "/")
    parts = norm.split("/")
    basename = parts[-1]

    for root in _candidate_roots(data_root):
        # Strategy A
        p = os.path.join(root, *parts)
        if os.path.exists(p):
            return p
        # Strategy B
        p = os.path.join(root, basename)
        if os.path.exists(p):
            return p
        # Strategy C  (original Growth_Decay_Classify reorder for 3-part paths)
        if len(parts) == 3:
            p = os.path.join(root, parts[1], parts[0], parts[2])
            if os.path.exists(p):
                return p
        # Strategy D: walk up to two levels looking for the basename
        for dirpath, _, filenames in os.walk(root):
            if basename in filenames:
                return os.path.join(dirpath, basename)
            # Don't recurse too deep — stop after 2 subdirectory levels
            depth = dirpath.replace(root, "").count(os.sep)
            if depth >= 2:
                break

    return None


# ============================================================================ #
# Q1 — Compute time-averaged spatial metrics from raw VIL
# ============================================================================ #
def compute_temporal_spatial_metrics(vil_frames: np.ndarray) -> dict:
    """
    Per-frame connected-component analysis; returns time-averaged metrics
    and their std across frames.
    """
    if not HAS_SKIMAGE:
        return {}

    T = vil_frames.shape[0]
    n_cells_arr    = np.zeros(T)
    max_area_arr   = np.zeros(T)
    active_frac_arr= np.zeros(T)

    for t in range(T):
        frame = vil_frames[t]
        active_frac_arr[t] = float((frame > ACTIVE_THRESH).mean())
        props = regionprops(sk_label(frame > NOISE_THRESH))
        if props:
            n_cells_arr[t]  = len(props)
            max_area_arr[t] = max(p.area for p in props)

    mn, mx, ma = (float(n_cells_arr.mean()),
                  float(max_area_arr.mean()),
                  float(active_frac_arr.mean()))
    return {
        "mean_n_cells":      mn,
        "std_n_cells":       float(n_cells_arr.std()),
        "mean_max_area":     mx,
        "std_max_area":      float(max_area_arr.std()),
        "mean_active_frac":  ma,
        "std_active_frac":   float(active_frac_arr.std()),
        "temporal_subclass": assign_spatial_subclass(mn, mx, ma),
    }


def recompute_temporal_metrics_for_steady(
    df_steady: pd.DataFrame,
    catalog_path: str,
    data_root: str,
) -> pd.DataFrame:
    """
    Opens raw VIL HDF5 files and adds temporal spatial metric columns to df_steady.
    Logs detailed path-resolution info so failures are diagnosable.
    """
    log.info("[Q1] Re-reading raw VIL for STEADY events …")
    catalog = pd.read_csv(catalog_path, low_memory=False)
    vil_df  = catalog[catalog["img_type"] == "vil"].copy()

    steady_ids = set(df_steady["id"].values)
    results    = {}
    path_failures = 0

    grouped = vil_df.groupby("file_name")
    for relative_path, group in grouped:
        target_ids = set(group["id"].values) & steady_ids
        if not target_ids:
            continue

        full_path = resolve_hdf5_path(data_root, relative_path)
        if not full_path:
            path_failures += 1
            if path_failures <= 3:          # only spam the first few
                log.warning(f"  [Q1] Cannot resolve path: {relative_path!r}")
                log.warning(f"       Searched roots: {_candidate_roots(data_root)}")
            continue

        try:
            with h5py.File(full_path, "r") as f:
                if "id" not in f or "vil" not in f:
                    log.warning(f"  [Q1] HDF5 missing 'id' or 'vil' key: {full_path}")
                    continue
                file_ids = [
                    x.decode("utf-8") if isinstance(x, bytes) else str(x)
                    for x in f["id"][:]
                ]
                for i, eid in enumerate(file_ids):
                    if eid not in target_ids:
                        continue
                    data = f["vil"][i]
                    # Normalise to (T, H, W)
                    if data.ndim == 3 and data.shape[2] < data.shape[0]:
                        data = data.transpose(2, 0, 1)
                    results[eid] = compute_temporal_spatial_metrics(data)

        except Exception as e:
            log.warning(f"  [Q1] Error reading {full_path}: {e}")

    if path_failures > 3:
        log.warning(f"  [Q1] … and {path_failures - 3} more unresolved paths. "
                    f"Check --data_root; should be the folder containing CATALOG.csv.")

    log.info(f"[Q1] Temporal metrics computed for {len(results)} / "
             f"{len(steady_ids)} STEADY events.")
    if len(results) == 0:
        log.error(
            "[Q1] Zero events loaded. Common causes:\n"
            "  1) --data_root points to a year subfolder (e.g. .../SEVIR/2019).\n"
            "     Fix: set --data_root to the parent that contains CATALOG.csv.\n"
            "  2) CATALOG.csv img_type values differ — check: "
            f"{catalog['img_type'].unique().tolist()}\n"
            "  3) HDF5 files use a different internal key. "
            f"Keys in first file: {_probe_hdf5_keys(vil_df, data_root)}"
        )

    temporal_df = pd.DataFrame.from_dict(results, orient="index")
    temporal_df.index.name = "id"
    temporal_df = temporal_df.reset_index()
    return df_steady.merge(temporal_df, on="id", how="left")


def _probe_hdf5_keys(vil_df: pd.DataFrame, data_root: str) -> list:
    """Open the first resolvable HDF5 and return its top-level keys."""
    for rel_path in vil_df["file_name"].unique()[:5]:
        p = resolve_hdf5_path(data_root, rel_path)
        if p:
            try:
                with h5py.File(p, "r") as f:
                    return list(f.keys())
            except Exception:
                pass
    return ["<could not open any file>"]


# ============================================================================ #
# Q1 — Rank-correlation analysis
# ============================================================================ #
def rank_correlation_analysis(df_steady: pd.DataFrame,
                              out_dir: str) -> tuple:
    pairs = [
        ("n_cells_t0",         "mean_n_cells",    "N cells"),
        ("max_cell_area_t0",   "mean_max_area",   "Max cell area"),
        ("active_fraction_t0", "mean_active_frac","Active fraction"),
    ]

    rows = []
    for col_t0, col_mean, label in pairs:
        if col_t0 not in df_steady.columns or col_mean not in df_steady.columns:
            log.warning(f"  [Q1] Skipping {label}: column '{col_t0}' or "
                        f"'{col_mean}' missing from dataframe.")
            continue
        valid = df_steady[[col_t0, col_mean]].dropna()
        if len(valid) < 5:
            log.warning(f"  [Q1] Too few valid rows for {label} ({len(valid)}).")
            continue
        r_sp, p_sp = spearmanr(valid[col_t0], valid[col_mean])
        r_pe, p_pe = pearsonr(valid[col_t0],  valid[col_mean])
        rows.append({
            "metric":     label,
            "spearman_r": round(r_sp, 4),
            "spearman_p": round(p_sp, 6),
            "pearson_r":  round(r_pe, 4),
            "pearson_p":  round(p_pe, 6),
            "n_valid":    len(valid),
        })
        log.info(f"  {label:<20}  Spearman ρ={r_sp:.3f} (p={p_sp:.4f})  "
                 f"Pearson r={r_pe:.3f}")

    corr_df = pd.DataFrame(rows)

    # Subclass change fraction
    if "temporal_subclass" in df_steady.columns:
        df_steady["spatial_subclass_t0"] = df_steady.apply(
            lambda r: assign_spatial_subclass(
                r["n_cells_t0"], r["max_cell_area_t0"], r["active_fraction_t0"]
            ), axis=1
        )
        df_steady["subclass_changed"] = (
            df_steady["spatial_subclass_t0"] != df_steady["temporal_subclass"]
        )
        n_changed = int(df_steady["subclass_changed"].sum())
        pct = 100 * n_changed / len(df_steady)
        log.info(f"\n[Q1] Subclass would change for {n_changed} / "
                 f"{len(df_steady)} STEADY events ({pct:.1f}%).")
        changed_rows = df_steady[df_steady["subclass_changed"]][
            ["spatial_subclass_t0", "temporal_subclass"]
        ]
        if len(changed_rows):
            log.info("  Change matrix:\n" + str(
                pd.crosstab(changed_rows["spatial_subclass_t0"],
                            changed_rows["temporal_subclass"])
            ))
    else:
        df_steady["subclass_changed"] = False

    # Scatter plots
    if HAS_MPL and rows:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, (col_t0, col_mean, label) in zip(axes, pairs):
            if col_t0 not in df_steady.columns or col_mean not in df_steady.columns:
                ax.set_visible(False)
                continue
            valid = df_steady[[col_t0, col_mean, "subclass_changed"]].dropna()
            if valid.empty:
                ax.set_visible(False)
                continue
            colors = valid["subclass_changed"].map({True: "crimson", False: "steelblue"})
            ax.scatter(valid[col_t0], valid[col_mean],
                       c=colors, alpha=0.5, s=18, linewidths=0)
            lo = min(valid[col_t0].min(), valid[col_mean].min())
            hi = max(valid[col_t0].max(), valid[col_mean].max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6)
            r_sp_val = corr_df.loc[corr_df["metric"] == label, "spearman_r"].values
            r_str = f"ρ={r_sp_val[0]:.3f}" if len(r_sp_val) else ""
            ax.set_title(f"{label}\n{r_str}", fontsize=9)
            ax.set_xlabel("t=0 value", fontsize=8)
            ax.set_ylabel("Time-averaged value", fontsize=8)
            ax.tick_params(labelsize=7)
        from matplotlib.lines import Line2D
        fig.legend(handles=[
            Line2D([0],[0], marker="o", color="w", markerfacecolor="crimson",
                   markersize=6, label="Subclass would change"),
            Line2D([0],[0], marker="o", color="w", markerfacecolor="steelblue",
                   markersize=6, label="Subclass unchanged"),
        ], loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.05))
        fig.suptitle("STEADY: t=0 vs time-averaged spatial metrics",
                     fontsize=10, y=1.02)
        plt.tight_layout()
        out_path = os.path.join(out_dir, "q1_rank_correlation_scatter.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"[Q1] Scatter plot → {out_path}")

    return df_steady, corr_df


# ============================================================================ #
# J(N) RECONSTRUCTION FROM RAW METRIC COLUMNS
# ============================================================================ #
def _first_col(df: pd.DataFrame, candidates: tuple) -> str | None:
    """Return the first candidate column name that exists in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _norm_drop(arr: np.ndarray) -> np.ndarray:
    """0 at arr[0], 1 at arr.min() — normalised reduction."""
    return (arr[0] - arr) / (arr[0] - arr.min() + 1e-12)


def _norm_gain(arr: np.ndarray) -> np.ndarray:
    """0 at arr[0], 1 at arr.max() — normalised improvement."""
    return (arr - arr[0]) / (arr.max() - arr[0] + 1e-12)


def _norm_level(arr: np.ndarray) -> np.ndarray:
    """0 at minimum, 1 at maximum — normalised absolute level."""
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)


def _sigmoid(x: np.ndarray, beta: float, threshold: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-beta * (x - threshold)))


def _estimate_e_super(n: int) -> int:
    R_max         = _V_MAX * _T_FC
    avg_sp_area   = (_HW * _HW) / max(n, 1)
    avg_neighbors = np.pi * R_max ** 2 / avg_sp_area
    return int(n * min(avg_neighbors, n - 1))


def _compute_j_from_raw(df: pd.DataFrame,
                         lifecycle_class: str | None = None) -> pd.DataFrame | None:
    """
    Reconstruct J(N) from the raw metric columns present in a per-event
    *_sweep.csv file, replicating the logic in compute_joint_scores() from
    build_n_summary_hpc.py.

    Per-event sweep CSVs contain only the raw metrics written by
    diagnostic_n_sweep_hpc.py (ISV, LFS, FAE, MDV, BR, TBR, N).  The J
    column is produced by the aggregation step (build_n_summary_hpc.py) and
    is therefore absent from individual files.  This function fills that gap
    so Q2/Q3 can operate on per-event CSVs without modification to the sweep
    writer.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded sweep CSV (must contain 'N' and the raw metric columns).
    lifecycle_class : str or None
        Used to select the per-class outer_alpha from _LIFECYCLE_ALPHA.
        Falls back to _DEFAULT_OUTER_ALPHA if None or unknown.

    Returns
    -------
    df with a 'J_imex' column added (in-place copy), or None if required
    columns are missing.
    """
    isv_col = _first_col(df, _ISV_COLS)
    lfs_col = _first_col(df, _LFS_COLS)
    fae_col = _first_col(df, _FAE_COLS)
    mdv_col = _first_col(df, _MDV_COLS)
    br_col  = _first_col(df, _BR_COLS)
    tbr_col = _first_col(df, _TBR_COLS)

    missing = [name for name, col in [
        ("ISV", isv_col), ("LFS", lfs_col), ("FAE", fae_col),
        ("MDV", mdv_col), ("BR",  br_col),  ("TBR", tbr_col),
    ] if col is None]
    if missing:
        log.warning(f"  [J-reconstruct] Cannot compute J: missing columns {missing}. "
                    f"Available: {df.columns.tolist()}")
        return None

    df = df.copy().sort_values("N").reset_index(drop=True)
    ns  = df["N"].astype(float).values
    isv = df[isv_col].astype(float).values
    lfs = df[lfs_col].astype(float).values
    fae = df[fae_col].astype(float).values
    mdv = df[mdv_col].astype(float).values
    br  = df[br_col].astype(float).values
    tbr = df[tbr_col].astype(float).values

    if len(ns) < 3:
        log.warning("  [J-reconstruct] Fewer than 3 N values — skipping.")
        return None

    # ── QUALITY ──────────────────────────────────────────────────────────────
    # ISV_gain: normalised drop
    isv_gain = (isv[0] - isv) / (isv[0] - isv.min() + 1e-12)

    # BR_gain: normalised rise, elbow-gated (mirrors build summary exactly)
    br_gain_raw = (br - br[0]) / (br.max() - br[0] + 1e-12)

    # Elbow on (1 - BR)
    br_inv = 1.0 - br
    if len(ns) >= 3:
        pts   = np.column_stack([ns, br_inv]).astype(float)
        rng   = pts.max(axis=0) - pts.min(axis=0) + 1e-12
        pts_n = (pts - pts.min(axis=0)) / rng
        line  = pts_n[-1] - pts_n[0]
        line /= np.linalg.norm(line) + 1e-12
        dists = [np.linalg.norm((p - pts_n[0]) - np.dot(p - pts_n[0], line) * line)
                 for p in pts_n]
        br_elbow_idx = int(np.argmax(dists))
    else:
        br_elbow_idx = 0
    br_cap        = float(br_gain_raw[br_elbow_idx])
    br_gain_gated = np.minimum(br_gain_raw, br_cap)

    # FAE_pen and MDV_pen: normalised absolute level
    fae_pen = (fae - fae.min()) / (fae.max() - fae.min() + 1e-12)
    mdv_pen = (mdv - mdv.min()) / (mdv.max() - mdv.min() + 1e-12)

    Q = (_W_ISV * isv_gain
       + _W_BR  * br_gain_gated
       - _W_FAE * fae_pen
       - _W_MDV * mdv_pen)

    # ── STIFFNESS ─────────────────────────────────────────────────────────────
    lfs_stiff    = (lfs - lfs.min()) / (lfs.max() - lfs.min() + 1e-12)
    e_super_arr  = np.array([float(_estimate_e_super(int(n))) for n in ns])
    esuper_stiff = _norm_level(e_super_arr)
    tbr_stiff    = (tbr - tbr.min()) / (tbr.max() - tbr.min() + 1e-12)

    S = _ALPHA_R * lfs_stiff + _ALPHA_D * esuper_stiff + _ALPHA_T * tbr_stiff

    # ── PENALTY + INTERACTION ─────────────────────────────────────────────────
    outer_alpha = _LIFECYCLE_ALPHA.get(lifecycle_class, _DEFAULT_OUTER_ALPHA) \
                  if lifecycle_class else _DEFAULT_OUTER_ALPHA
    penalty     = outer_alpha * _sigmoid(S, _BETA, _S_THRESH)
    interaction = _GAMMA * Q * S

    J = Q - penalty - interaction

    df["J_imex"] = J
    return df


# ============================================================================ #
# ============================================================================ #
# SWEEP FILE DISCOVERY
# ============================================================================ #
_SWEEP_SUFFIXES = ("_sweep", "_n_sweep", "_metrics", "_result", "_out")


def _bare_id(filename: str) -> str:
    """Strip extension and any known sweep suffix to recover the bare event ID."""
    stem = os.path.splitext(filename)[0]
    for suf in _SWEEP_SUFFIXES:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def _build_sweep_index(sweep_dir: str,
                       event_ids: list,
                       debug: bool = False) -> dict:
    """
    Scan sweep_dir and build bare_event_id -> filepath for every *.csv found.
    The directory contains N-sweep results for ALL events; we index everything
    and load_sweep_for_event simply looks up each STEADY ID directly.
    No cross-matching between filenames and event_ids needed.
    """
    if not sweep_dir or not os.path.isdir(sweep_dir):
        log.error(f"[sweep] Directory not found: {sweep_dir!r}")
        return {}

    all_csvs = sorted(f for f in os.listdir(sweep_dir)
                      if f.lower().endswith(".csv"))
    log.info(f"[sweep] {len(all_csvs)} CSV files found in {sweep_dir}")

    event_to_path = {_bare_id(f): os.path.join(sweep_dir, f) for f in all_csvs}

    # Check overlap with the STEADY IDs we actually need
    id_set  = set(event_ids)
    overlap = id_set & set(event_to_path)
    log.info(f"[sweep] STEADY events with a sweep file: "
             f"{len(overlap)} / {len(id_set)}")

    if debug:
        log.info("[sweep] Sample bare IDs from filenames (first 5):")
        for bid in list(event_to_path)[:5]:
            log.info(f"  {bid!r}")
        log.info("[sweep] Sample STEADY event IDs (first 5):")
        for eid in list(id_set)[:5]:
            log.info(f"  {eid!r}")

    if not overlap:
        log.error(
            "[sweep] No STEADY event IDs found in sweep directory.\n"
            "  Run with --sweep_name_debug to compare bare IDs vs event IDs.\n"
            "  Likely cause: sweep was run on a different catalogue version."
        )

    return event_to_path


# Columns reported once so repeated noise is suppressed
_COLUMNS_REPORTED = False


def load_sweep_for_event(event_to_path: dict,
                         event_id: str,
                         lifecycle_class: str | None = None) -> pd.DataFrame | None:
    """
    Load the sweep CSV for event_id from the pre-built event->path index.

    Root-cause fix (Q2/Q3 zero-match bug)
    ──────────────────────────────────────
    Per-event *_sweep.csv files written by diagnostic_n_sweep_hpc.py contain
    only the raw metric columns (ISV, LFS, FAE, MDV, BR, TBR, N).  The J(N)
    column is produced exclusively by the aggregation step in
    build_n_summary_hpc.py and is therefore absent from individual CSVs.

    The old code returned None whenever no J column was found, causing every
    STEADY event to be counted as "missing" and Q2/Q3 to produce zero records.

    Fix: after loading the CSV, if no J column is present but the raw metric
    columns are present, compute J(N) on-the-fly via _compute_j_from_raw(),
    which replicates the compute_joint_scores() logic from the build summary
    (same BR elbow-gating, same sigmoid penalty, same interaction term).

    Parameters
    ----------
    event_to_path : dict
        Pre-built index from _build_sweep_index().
    event_id : str
        Event identifier to look up.
    lifecycle_class : str or None
        If provided, used to select the per-class outer_alpha (LIFECYCLE_ALPHA)
        when reconstructing J.  Pass the event's 'lifecycle_class' value.

    Returns
    -------
    pd.DataFrame with columns ['N', 'J_imex', ...] sorted by N, or None.
    """
    global _COLUMNS_REPORTED

    path = event_to_path.get(event_id)
    if path is None:
        return None

    try:
        df = pd.read_csv(path)

        # ── Report columns from the very first file opened ────────────────────
        if not _COLUMNS_REPORTED:
            log.info(f"[sweep] First sweep CSV opened: {path}")
            log.info(f"[sweep] Columns found: {df.columns.tolist()}")
            log.info("[sweep] Script expects: 'N' + either a pre-computed J column "
                     "(j_imex*, j_ark*, 'J') OR raw metric columns "
                     "(intra_sp_var_mean / LFS / FAE / MDV / boundary_recall_mean / "
                     "temporal_br_std) from which J is reconstructed on-the-fly.")
            _COLUMNS_REPORTED = True

        # ── N column ──────────────────────────────────────────────────────────
        n_col = next((c for c in df.columns if c.strip().upper() == "N"), None)
        if n_col is None:
            log.warning(f"  [sweep] No 'N' column in {path}. "
                        f"Columns: {df.columns.tolist()}")
            return None
        if n_col != "N":
            df = df.rename(columns={n_col: "N"})

        # ── J column: prefer pre-computed; reconstruct from raw if absent ─────
        j_col = next(
            (c for c in df.columns
             if c.lower().startswith("j_imex") or c.lower().startswith("j_ark")),
            None
        )
        if j_col is None:
            j_col = next((c for c in df.columns if c.strip().upper() == "J"), None)

        if j_col is not None:
            # Pre-computed J column found — just normalise the name.
            if j_col != "J_imex":
                df = df.rename(columns={j_col: "J_imex"})
        else:
            # No J column: reconstruct from raw metric columns.
            # This is the normal case for per-event *_sweep.csv files that were
            # written by diagnostic_n_sweep_hpc.py (which does not compute J).
            log.debug(f"  [sweep] No J column in {path} — reconstructing from "
                      f"raw metrics (ISV/LFS/FAE/MDV/BR/TBR).")
            df = _compute_j_from_raw(df, lifecycle_class=lifecycle_class)
            if df is None:
                log.warning(
                    f"  [sweep] Could not reconstruct J for {event_id} from {path}. "
                    f"Check that the sweep CSV contains the expected raw metric columns: "
                    f"intra_sp_var_mean, LFS, FAE, MDV, boundary_recall_mean, "
                    f"temporal_br_std."
                )
                return None

        return df.sort_values("N").reset_index(drop=True)

    except Exception as e:
        log.warning(f"  [sweep] Error loading {path}: {e}")
        return None


# ============================================================================ #
# J(N) shape metrics
# ============================================================================ #
def plateau_width_95(sweep: pd.DataFrame) -> float:
    j, n = sweep["J_imex"].values, sweep["N"].values
    jmax = j.max()
    if jmax <= 0 or len(n) < 3:
        return np.nan
    above = n[j >= 0.95 * jmax]
    if len(above) < 2:
        return float(n[-1] - n[0])
    return float(above[-1] - above[0])


def n_star_from_sweep(sweep: pd.DataFrame) -> float:
    j, n = sweep["J_imex"].values, sweep["N"].values
    jmax = j.max()
    if jmax <= 0:
        return float(n[np.argmax(j)])
    candidates = n[j >= 0.95 * jmax]
    return float(candidates[0]) if len(candidates) else float(n[np.argmax(j)])


# ============================================================================ #
# Q2 — Subclass stability in J(N) space
# ============================================================================ #
def subclass_stability_analysis(df_steady: pd.DataFrame,
                                event_to_path: dict,
                                out_dir: str) -> pd.DataFrame:
    if "subclass_changed" not in df_steady.columns:
        df_steady["subclass_changed"] = False

    records, missing = [], 0
    for _, row in df_steady.iterrows():
        sweep = load_sweep_for_event(event_to_path, row["id"],
                                     lifecycle_class=row.get("lifecycle_class"))
        if sweep is None:
            missing += 1
            continue
        records.append({
            "id":               row["id"],
            "subclass_changed": bool(row.get("subclass_changed", False)),
            "spatial_t0":       row.get("spatial_subclass_t0",
                                        row.get("spatial_type", "unknown")),
            "spatial_temporal": row.get("temporal_subclass",
                                        row.get("spatial_type", "unknown")),
            "plateau_width":    plateau_width_95(sweep),
            "n_star":           n_star_from_sweep(sweep),
            "j_max":            float(sweep["J_imex"].max()),
        })

    if missing:
        log.warning(f"[Q2] Sweep CSV not found for {missing} / "
                    f"{len(df_steady)} STEADY events.")
        if missing == len(df_steady):
            log.error(
                "[Q2] Zero sweep files matched. Common causes:\n"
                "  1) --sweep_dir points to the wrong folder.\n"
                "  2) Sweep CSV filenames don't contain the event ID.\n"
                "  Run with --sweep_name_debug to list what's in sweep_dir."
            )

    if not records:
        log.warning("[Q2] No sweep data found — skipping Q2/Q3.")
        return pd.DataFrame()

    res = pd.DataFrame(records)

    log.info("\n[Q2] Plateau width and N* by subclass-change status:")
    for changed, grp in res.groupby("subclass_changed"):
        label = "CHANGERS" if changed else "NON-CHANGERS"
        pw = grp["plateau_width"].dropna()
        ns = grp["n_star"].dropna()
        log.info(f"  {label} (n={len(grp)})")
        log.info(f"    plateau_width  mean={pw.mean():.0f}  "
                 f"median={pw.median():.0f}  std={pw.std():.0f} sp")
        log.info(f"    N*             mean={ns.mean():.0f}  "
                 f"median={ns.median():.0f}  std={ns.std():.0f} sp")

    changers     = res[res["subclass_changed"]]["plateau_width"].dropna()
    non_changers = res[~res["subclass_changed"]]["plateau_width"].dropna()

    if len(changers) >= 3 and len(non_changers) >= 3:
        from scipy.stats import mannwhitneyu
        stat, p = mannwhitneyu(changers, non_changers, alternative="less")
        log.info(f"\n[Q2] Mann-Whitney U (changers narrower?): U={stat:.1f}, p={p:.4f}")
        if p < 0.05:
            log.info("  → CHANGERS have significantly narrower J(N) plateaus.")
            log.info("    Temporal averaging fixes genuine misclassification.")
        else:
            log.info("  → No significant plateau-width difference.")
            log.info("    Flatness is intrinsic; subdivision approach won't help.")
    else:
        log.info(f"[Q2] Too few changers (n={len(changers)}) for Mann-Whitney test.")

    return res


# ============================================================================ #
# Q3 — Plateau width distribution + example J(N) curves
# ============================================================================ #
def plateau_width_distribution(res: pd.DataFrame, out_dir: str):
    if res.empty or not HAS_MPL:
        return

    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1])
    ax0 = fig.add_subplot(gs[0])

    valid_pw = res["plateau_width"].dropna()
    if valid_pw.empty:
        log.warning("[Q3] All plateau_width values are NaN — no J > 0 events.")
        return

    bins = np.linspace(0, valid_pw.quantile(0.98) + 50, 30)
    for changed, grp in res.groupby("subclass_changed"):
        label = ("Would change subclass" if changed else "Subclass unchanged")
        color = "crimson" if changed else "steelblue"
        vals  = grp["plateau_width"].dropna()
        ax0.hist(vals, bins=bins, alpha=0.55, color=color,
                 label=f"{label} (n={len(vals)})", edgecolor="white", lw=0.3)

    ax0.axvline(200, color="k", lw=1, ls="--", alpha=0.5, label="±100 sp tolerance")
    ax0.set_xlabel("J(N) plateau width at 95% of peak (superpixels)", fontsize=9)
    ax0.set_ylabel("Count", fontsize=9)
    ax0.set_title("STEADY: J(N) plateau width distribution", fontsize=10)
    ax0.legend(fontsize=8)
    ax0.tick_params(labelsize=8)

    ax1 = fig.add_subplot(gs[1])
    ax1.set_xlabel("N (superpixels)", fontsize=9)
    ax1.set_ylabel("J(N)", fontsize=9)
    ax1.set_title("Example J(N) curves\n(narrowest vs widest plateau)", fontsize=10)
    ax1.tick_params(labelsize=8)
    ax1.text(0.5, 0.5, "Populated by plot_example_jn_curves()",
             transform=ax1.transAxes, ha="center", va="center",
             fontsize=8, color="gray")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "q3_plateau_width_distribution.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"[Q3] Plateau width distribution → {out_path}")


def plot_example_jn_curves(res: pd.DataFrame,
                           event_to_path: dict,
                           out_dir: str,
                           n_examples: int = 5):
    if res.empty or not HAS_MPL:
        return

    res_valid = res.dropna(subset=["plateau_width"])
    if len(res_valid) < 2:
        return

    sorted_by_width = res_valid.sort_values("plateau_width")
    narrow = sorted_by_width.head(n_examples)
    wide   = sorted_by_width.tail(n_examples)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
    for ax, subset, title in [
        (axes[0], narrow, f"Narrowest {n_examples} plateaus (sharp N*)"),
        (axes[1], wide,   f"Widest {n_examples} plateaus (ambiguous N*)"),
    ]:
        for _, row in subset.iterrows():
            sweep = load_sweep_for_event(event_to_path, row["id"],
                                         lifecycle_class=row.get("lifecycle_class"))
            if sweep is None:
                continue
            color = "crimson" if row["subclass_changed"] else "steelblue"
            ls    = "--" if row["subclass_changed"] else "-"
            ax.plot(sweep["N"], sweep["J_imex"],
                    color=color, lw=1.2, ls=ls, alpha=0.75,
                    label=f"{str(row['id'])[:12]} (pw={row['plateau_width']:.0f})")
        ax.axhline(0, color="k", lw=0.7, ls="--", alpha=0.4)
        ax.set_xlabel("N (superpixels)", fontsize=9)
        ax.set_ylabel("J(N) [IMEX]", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=6.5, loc="lower right")
        ax.tick_params(labelsize=8)

    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([0],[0], color="crimson",   lw=1.2, ls="--",
               label="Would change subclass"),
        Line2D([0],[0], color="steelblue", lw=1.2, ls="-",
               label="Subclass unchanged"),
    ], loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("STEADY: J(N) curves coloured by subclass stability", fontsize=10)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "q2_jn_curves_by_stability.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"[Q2] J(N) curves plot → {out_path}")


# ============================================================================ #
# Summary
# ============================================================================ #
def write_summary(df_steady, corr_df, res, out_dir):
    df_steady.to_csv(os.path.join(out_dir, "steady_augmented.csv"), index=False)
    log.info(f"Augmented STEADY catalogue → {out_dir}/steady_augmented.csv")

    if not corr_df.empty:
        corr_df.to_csv(os.path.join(out_dir, "steady_rank_correlations.csv"),
                       index=False)
        log.info(f"Rank correlations → {out_dir}/steady_rank_correlations.csv")

    if not res.empty:
        res.to_csv(os.path.join(out_dir, "steady_sweep_summary.csv"), index=False)
        log.info(f"Sweep summary → {out_dir}/steady_sweep_summary.csv")

    print("\n" + "=" * 65)
    print("  INTERPRETATION GUIDE")
    print("=" * 65)
    print("""
Q1 — Spearman rho (t=0 vs time-averaged spatial metrics)
  rho > 0.85 on all metrics  ->  classifiers are equivalent;
    flatness is intrinsic, not a t=0 artefact.
  rho < 0.70 on n_cells or max_area  ->  check Q2.

Q2 — Mann-Whitney U test (do changers have narrower J(N) plateaus?)
  p < 0.05  ->  temporal averaging fixes genuine misclassification.
  p >= 0.05  ->  flatness is independent of spatial metric used;
    use range recommendation (750-950) rather than further subdivision.

Q3 — Plateau width distribution
  Most events > 400 sp  ->  N* is genuinely underspecified; report range.
  Bimodal distribution  ->  two populations; check spatial_t0 breakdown.
""")
    print("=" * 65 + "\n")


# ============================================================================ #
# CLI
# ============================================================================ #
def parse_args():
    p = argparse.ArgumentParser(
        description="STEADY class spatial diagnostic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--catalogue",
                   default=r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\SEVIR\event_catalogue.csv")
    p.add_argument("--sweep_dir",
                   default=r"C:\Users\Siddharth Nair\results")
    p.add_argument("--data_root",
                   default=r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\SEVIR")
    p.add_argument("--catalog_csv",
                   default=r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\SEVIR\CATALOG.csv")
    p.add_argument("--out_dir",
                   default=r"C:\Users\Siddharth Nair\results\steady_diagnostic_out")
    p.add_argument("--skip_recompute", action="store_true",
                   help="Skip raw VIL re-read; use existing catalogue spatial columns.")
    p.add_argument("--sweep_name_debug", action="store_true",
                   help="Print the first 10 sweep CSV filenames found in sweep_dir.")
    p.add_argument("--n_examples", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    log.info(f"Loading catalogue: {args.catalogue}")
    df_all    = pd.read_csv(args.catalogue, low_memory=False)
    df_steady = (df_all[df_all["lifecycle_class"] == "STEADY"]
                 .copy().reset_index(drop=True))
    log.info(f"STEADY events: {len(df_steady)}")
    if len(df_steady) == 0:
        log.error("No STEADY events found. Check lifecycle_class column.")
        return

    corr_df = pd.DataFrame()
    res     = pd.DataFrame()

    # ── Q1 ────────────────────────────────────────────────────────────────────
    if not args.skip_recompute:
        if not HAS_SKIMAGE:
            log.warning("scikit-image not available — skipping Q1.")
        else:
            df_steady = recompute_temporal_metrics_for_steady(
                df_steady, args.catalog_csv, args.data_root
            )
            df_steady, corr_df = rank_correlation_analysis(
                df_steady, args.out_dir
            )
    else:
        log.info("[Q1] Skipped (--skip_recompute).")
        if "spatial_type" in df_steady.columns:
            df_steady["spatial_subclass_t0"] = df_steady["spatial_type"]
        df_steady["subclass_changed"] = False

    # ── Build sweep index once ────────────────────────────────────────────────
    event_to_path = _build_sweep_index(
        args.sweep_dir,
        event_ids=df_steady["id"].tolist(),
        debug=args.sweep_name_debug,
    )

    # ── Q2 ────────────────────────────────────────────────────────────────────
    if event_to_path:
        res = subclass_stability_analysis(df_steady, event_to_path, args.out_dir)
        if not res.empty:
            plot_example_jn_curves(res, event_to_path, args.out_dir, args.n_examples)
    else:
        log.warning("[Q2/Q3] No sweep files matched any STEADY event ID — skipping.")

    # ── Q3 ────────────────────────────────────────────────────────────────────
    if not res.empty:
        plateau_width_distribution(res, args.out_dir)

    write_summary(df_steady, corr_df, res, args.out_dir)


if __name__ == "__main__":
    main()