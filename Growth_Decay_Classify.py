"""
Growth_Decay_Classify.py  —  Rich SEVIR Event Classification Pipeline
======================================================================
Two-pass scan over the entire SEVIR VIL catalogue:

  Pass 1  Collects raw statistics (slope, R², spatial metrics) for every
          event. Percentile thresholds are then derived from the full slope
          distribution — replacing the arbitrary 1000 / -1000 magic numbers.

  Pass 2  Classifies each event into a 7-class lifecycle taxonomy that maps
          directly to which ADR equation terms dominate. Also detects temporal
          phase sequences (Growth / Decay / Plateau / Quiescent segments) via
          piecewise linear fitting using ruptures (if installed) or a simple
          sign-change fallback.

Output CSV columns (one row per event)
---------------------------------------
  id, slope, r_squared, slope_norm, n_phases, phase_sequence,
  peak_mass, peak_frame, trough_mass, mean_mass, active_fraction_t0,
  n_cells_t0, max_cell_area_t0, spatial_type, lifecycle_class,
  intensity_percentile, duration_frames, recommended_N_hint

Lifecycle classes
-----------------
  RAPID_GROWTH   — strong monotone increase  (Reaction+ dominated)
  RAPID_DECAY    — strong monotone decrease  (Reaction- dominated)
  GROWTH_DECAY   — lifecycle [G→D]            (full microphysics cycle)
  STEADY         — low |slope|, high R²       (Advection dominated)
  EPISODIC       — multiple alternating G/D   (repeated initiation)
  PLATEAU        — high mass, flat slope      (mature MCS)
  QUIESCENT      — mass below noise floor     (background / clear air)

Usage
-----
    python Growth_Decay_Classify.py                   # interactive
    python Growth_Decay_Classify.py --out my_cat.csv  # direct output path
    python Growth_Decay_Classify.py --top 20          # print top-N per class
"""

import argparse
import logging
import os
import sys
import warnings
from collections import Counter

import h5py
import numpy as np
import pandas as pd
from scipy.stats import linregress
from tqdm import tqdm

try:
    from skimage.measure import label as sk_label, regionprops
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    logging.warning("scikit-image not found — spatial metrics will be skipped.")

try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --------------------------------------------------------------------------- #
# LOGGING
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# CONFIGURATION
# --------------------------------------------------------------------------- #
DATA_ROOT    = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\SEVIR"
CATALOG_PATH = os.path.join(DATA_ROOT, "CATALOG.csv")
DEFAULT_OUT  = os.path.join(DATA_ROOT, "event_catalogue.csv")

# Fixed physics thresholds (independent of slope units)
NOISE_THRESH  = 20     # VIL pixel value: below = background
ACTIVE_THRESH = 20     # same — used for active_fraction computation

# Percentile-based slope thresholds (derived in pass 1 from data)
# These are the *percentile* levels, not absolute values.
GROWTH_PERCENTILE = 85   # top 15% by slope → GROWTH candidate
DECAY_PERCENTILE  = 15   # bottom 15% by slope → DECAY candidate
STABLE_ABS_PCTL   = 25   # events with |slope| below 25th pctl of |slope| → STEADY

# Phase segmentation
MIN_PHASE_FRAMES = 3    # a segment shorter than this is merged into neighbours
RUPTURES_PEN     = 12   # ruptures PELT penalty — higher = fewer breakpoints

# Spatial type thresholds (pixels at 384×384, ~1 km/px)
ISOLATED_MAX_AREA = 5_000   # px  — single cell upper bound (~70 km²)
MCS_MIN_AREA      = 30_000  # px  — mesoscale system lower bound (~300 km²)

# Recommended N hint: scales with active fraction and n_cells
N_HINT_BASE = 500
N_HINT_MAX  = 2500


# --------------------------------------------------------------------------- #
# PATH RESOLVER  (identical to original)
# --------------------------------------------------------------------------- #
def get_local_path(catalog_filename: str) -> str | None:
    p1 = os.path.join(DATA_ROOT, catalog_filename)
    if os.path.exists(p1):
        return p1
    parts = catalog_filename.replace("\\", "/").split("/")
    if len(parts) == 3:
        p2 = os.path.join(DATA_ROOT, parts[1], parts[0], parts[2])
        if os.path.exists(p2):
            return p2
    return None


# --------------------------------------------------------------------------- #
# RAW STATISTICS  (Pass 1 per-event computation)
# --------------------------------------------------------------------------- #
def compute_raw_stats(vil_frames: np.ndarray) -> dict:
    """
    Compute all frame-level statistics from a raw VIL array (T, H, W).

    Returns a dict of scalar features.
    """
    T, H, W = vil_frames.shape

    # ── Mass series ──────────────────────────────────────────────────────────
    mask        = vil_frames > NOISE_THRESH
    mass_series = np.where(mask, vil_frames, 0).sum(axis=(1, 2)).astype(float)

    # ── Linear regression on full series ────────────────────────────────────
    x = np.arange(T, dtype=float)
    slope, intercept, r_value, _, _ = linregress(x, mass_series)
    r_squared = r_value ** 2

    # ── Peak / trough / mean ─────────────────────────────────────────────────
    peak_mass   = float(mass_series.max())
    peak_frame  = int(mass_series.argmax())
    trough_mass = float(mass_series.min())
    mean_mass   = float(mass_series.mean())

    # ── Active fraction at t=0 ───────────────────────────────────────────────
    active_fraction_t0 = float((vil_frames[0] > ACTIVE_THRESH).mean())

    # ── Spatial analysis at t=0 via connected components ────────────────────
    n_cells_t0      = 0
    max_cell_area   = 0
    spatial_type    = "unknown"

    if HAS_SKIMAGE:
        blob_map = sk_label(vil_frames[0] > NOISE_THRESH)
        props    = regionprops(blob_map)
        if props:
            n_cells_t0    = len(props)
            max_cell_area = int(max(p.area for p in props))
            if n_cells_t0 == 1 and max_cell_area < ISOLATED_MAX_AREA:
                spatial_type = "isolated_cell"
            elif max_cell_area >= MCS_MIN_AREA:
                spatial_type = "mcs"
            elif active_fraction_t0 < 0.01:
                spatial_type = "quiescent"
            else:
                spatial_type = "cluster"
        else:
            spatial_type = "quiescent"

    return {
        "slope":              float(slope),
        "r_squared":          float(r_squared),
        "peak_mass":          peak_mass,
        "peak_frame":         peak_frame,
        "trough_mass":        trough_mass,
        "mean_mass":          mean_mass,
        "active_fraction_t0": active_fraction_t0,
        "n_cells_t0":         n_cells_t0,
        "max_cell_area_t0":   max_cell_area,
        "spatial_type":       spatial_type,
        "duration_frames":    T,
        "_mass_series":       mass_series,   # carried for phase analysis, dropped later
    }


# --------------------------------------------------------------------------- #
# PHASE SEGMENTATION
# --------------------------------------------------------------------------- #
_PHASE_LABELS = {1: "G", -1: "D", 0: "P"}   # growth / decay / plateau


def _slope_sign(series: np.ndarray) -> int:
    s, _, r, _, _ = linregress(np.arange(len(series)), series)
    if abs(s) < 1e-6:
        return 0
    return 1 if s > 0 else -1


def segment_phases(mass_series: np.ndarray) -> tuple[int, str]:
    """
    Detect temporal lifecycle phases in a mass time-series.

    Uses ruptures.Pelt if available; falls back to a simple rolling
    slope sign-change detector.

    Returns (n_phases, phase_sequence_string).
    Example: n_phases=3, phase_sequence="G→D→P"
    """
    T = len(mass_series)
    if T < 4:
        return 1, _PHASE_LABELS.get(_slope_sign(mass_series), "P")

    # ── Ruptures changepoint detection ───────────────────────────────────────
    if HAS_RUPTURES and T >= 8:
        try:
            signal = mass_series.reshape(-1, 1).astype(float)
            algo   = rpt.Pelt(model="rbf", min_size=MIN_PHASE_FRAMES).fit(signal)
            bkpts  = algo.predict(pen=RUPTURES_PEN)  # last element = T
            # Build segment labels
            segments = []
            prev = 0
            for bp in bkpts:
                seg = mass_series[prev:bp]
                if len(seg) >= MIN_PHASE_FRAMES:
                    segments.append(_slope_sign(seg))
                prev = bp
            if not segments:
                segments = [_slope_sign(mass_series)]
            # Collapse consecutive identical phases
            collapsed = [segments[0]]
            for s in segments[1:]:
                if s != collapsed[-1]:
                    collapsed.append(s)
            phase_seq = "→".join(_PHASE_LABELS.get(s, "P") for s in collapsed)
            return len(collapsed), phase_seq
        except Exception:
            pass  # fall through to simple method

    # ── Simple fallback: split at midpoint and compare slopes ────────────────
    mid  = T // 2
    s1   = _slope_sign(mass_series[:mid])
    s2   = _slope_sign(mass_series[mid:])
    if s1 == s2:
        return 1, _PHASE_LABELS.get(s1, "P")
    return 2, f"{_PHASE_LABELS.get(s1,'P')}→{_PHASE_LABELS.get(s2,'P')}"


# --------------------------------------------------------------------------- #
# LIFECYCLE CLASSIFIER  (Pass 2 — requires derived thresholds)
# --------------------------------------------------------------------------- #
def classify_event(stats: dict, thresholds: dict) -> str:
    """
    Assign one of 7 lifecycle classes using derived thresholds and phase info.

    thresholds keys: growth_thresh, decay_thresh, stable_abs_thresh,
                     high_mass_thresh, noise_mass_thresh
    """
    slope    = stats["slope"]
    r2       = stats["r_squared"]
    n_phases = stats["n_phases"]
    phases   = stats["phase_sequence"]
    mean_m   = stats["mean_mass"]
    act_frac = stats["active_fraction_t0"]

    G = thresholds["growth_thresh"]
    D = thresholds["decay_thresh"]
    S = thresholds["stable_abs_thresh"]
    H = thresholds["high_mass_thresh"]
    N = thresholds["noise_mass_thresh"]

    # QUIESCENT — almost no precipitable mass throughout
    if mean_m < N or act_frac < 0.005:
        return "QUIESCENT"

    # EPISODIC — multiple alternating growth/decay phases
    if n_phases >= 3 and any(c in phases for c in ["G→D→G", "D→G→D"]):
        return "EPISODIC"

    # GROWTH_DECAY — full lifecycle [G then D]
    if n_phases == 2 and phases in ("G→D", "G→P→D"):
        return "GROWTH_DECAY"

    # PLATEAU — high mass but flat slope
    if mean_m > H and abs(slope) < S:
        return "PLATEAU"

    # STEADY — consistently low slope magnitude, high R² (clean advection)
    if abs(slope) < S and r2 > 0.6:
        return "STEADY"

    # RAPID_GROWTH / RAPID_DECAY — strong monotone signal
    if slope > G:
        return "RAPID_GROWTH"
    if slope < D:
        return "RAPID_DECAY"

    # Catch-all for mid-range, multi-phase events
    if n_phases >= 2:
        return "EPISODIC"

    return "STEADY"


# --------------------------------------------------------------------------- #
# N HINT
# --------------------------------------------------------------------------- #
def recommend_n(stats: dict) -> int:
    """
    Suggest a starting N for the diagnostic sweep based on active fraction
    and number of detected cells. This is a heuristic, not a final answer.
    """
    n = N_HINT_BASE
    n += int(stats["active_fraction_t0"] * 1000)
    n += stats.get("n_cells_t0", 1) * 50
    return int(np.clip(n, N_HINT_BASE, N_HINT_MAX))


# --------------------------------------------------------------------------- #
# PASS 1  — Scan all events, collect raw stats
# --------------------------------------------------------------------------- #
def pass1_collect(catalog: pd.DataFrame) -> list[dict]:
    """
    Open each HDF5 file once, compute raw stats for every VIL event inside.
    Returns a list of stat dicts.
    """
    vil_df  = catalog[catalog["img_type"] == "vil"].copy()
    grouped = vil_df.groupby("file_name")
    records = []

    log.info(f"[Pass 1] Scanning {len(grouped)} files …")
    for relative_path, group in tqdm(grouped, total=len(grouped), unit="file"):
        full_path = get_local_path(relative_path)
        if not full_path:
            continue

        target_ids = set(group["id"].values)
        try:
            with h5py.File(full_path, "r") as f:
                if "id" not in f or "vil" not in f:
                    continue
                file_ids = f["id"][:]
                file_ids = [x.decode("utf-8") if isinstance(x, bytes) else str(x)
                            for x in file_ids]
                for i, event_id in enumerate(file_ids):
                    if event_id not in target_ids:
                        continue
                    data = f["vil"][i]           # shape varies: (T, H, W) or (H, W, T)
                    # Normalise to (T, H, W)
                    if data.ndim == 3 and data.shape[2] < data.shape[0]:
                        data = data.transpose(2, 0, 1)
                    stats = compute_raw_stats(data)
                    stats["id"] = event_id
                    records.append(stats)
        except Exception as e:
            log.warning(f"  Error reading {relative_path}: {e}")

    log.info(f"[Pass 1] Collected stats for {len(records):,} events.")
    return records


# --------------------------------------------------------------------------- #
# DERIVE THRESHOLDS  (from pass-1 slope distribution)
# --------------------------------------------------------------------------- #
def derive_thresholds(records: list[dict]) -> dict:
    slopes    = np.array([r["slope"]     for r in records])
    masses    = np.array([r["mean_mass"] for r in records])
    abs_slopes = np.abs(slopes)

    thresholds = {
        "growth_thresh":    float(np.percentile(slopes,     GROWTH_PERCENTILE)),
        "decay_thresh":     float(np.percentile(slopes,     DECAY_PERCENTILE)),
        "stable_abs_thresh":float(np.percentile(abs_slopes, STABLE_ABS_PCTL)),
        "high_mass_thresh": float(np.percentile(masses,     75)),
        "noise_mass_thresh":float(np.percentile(masses,     10)),
    }

    log.info("[Thresholds] Derived from slope distribution:")
    log.info(f"   Growth   slope > {thresholds['growth_thresh']:.1f}")
    log.info(f"   Decay    slope < {thresholds['decay_thresh']:.1f}")
    log.info(f"   Stable |slope| < {thresholds['stable_abs_thresh']:.1f}")
    log.info(f"   Plateau  mass  > {thresholds['high_mass_thresh']:.0f}")
    log.info(f"   Quiet    mass  < {thresholds['noise_mass_thresh']:.0f}")

    return thresholds


# --------------------------------------------------------------------------- #
# PASS 2  — Phase segment + classify + enrich
# --------------------------------------------------------------------------- #
def pass2_classify(records: list[dict], thresholds: dict) -> pd.DataFrame:
    slopes    = np.array([r["slope"] for r in records])
    slope_pct = {r["id"]: float(np.searchsorted(np.sort(slopes), r["slope"]) / len(slopes) * 100)
                 for r in records}

    rows = []
    log.info(f"[Pass 2] Classifying {len(records):,} events …")
    for stats in tqdm(records, unit="event"):
        mass_series = stats.pop("_mass_series")   # remove temp key

        n_phases, phase_seq = segment_phases(mass_series)
        stats["n_phases"]       = n_phases
        stats["phase_sequence"] = phase_seq

        lc = classify_event(stats, thresholds)
        stats["lifecycle_class"]      = lc
        stats["intensity_percentile"] = round(slope_pct[stats["id"]], 1)
        stats["slope_norm"]           = round(
            float((stats["slope"] - slopes.mean()) / (slopes.std() + 1e-9)), 4)
        stats["recommended_N_hint"]   = recommend_n(stats)

        rows.append(stats)

    # Build DataFrame with column order matching docstring
    col_order = [
        "id", "slope", "r_squared", "slope_norm", "n_phases", "phase_sequence",
        "peak_mass", "peak_frame", "trough_mass", "mean_mass",
        "active_fraction_t0", "n_cells_t0", "max_cell_area_t0", "spatial_type",
        "lifecycle_class", "intensity_percentile", "duration_frames",
        "recommended_N_hint",
    ]
    df = pd.DataFrame(rows)
    # Keep only known columns (extras from future extension are fine but sorted last)
    extra = [c for c in df.columns if c not in col_order]
    df = df[col_order + extra]
    return df


# --------------------------------------------------------------------------- #
# SUMMARY REPORT
# --------------------------------------------------------------------------- #
def print_summary(df: pd.DataFrame, top_n: int = 10):
    sep  = "=" * 65
    print(f"\n{sep}")
    print(f"  SEVIR Event Classification Summary  |  {len(df):,} events")
    print(sep)

    class_counts = df["lifecycle_class"].value_counts()
    print("\n  Class distribution:")
    for cls, cnt in class_counts.items():
        pct = cnt / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"    {cls:<18} {cnt:>5}  ({pct:4.1f}%)  {bar}")

    print("\n  Spatial type distribution:")
    for stype, cnt in df["spatial_type"].value_counts().items():
        print(f"    {stype:<18} {cnt:>5}")

    for cls in class_counts.index[:5]:
        subset = df[df["lifecycle_class"] == cls]
        print(f"\n  [{cls}] — top {min(top_n, len(subset))} by |slope|")
        print(f"    {'Event ID':<15}  {'Slope':>9}  {'R²':>5}  {'Phases':<14}  N_hint")
        print("    " + "-" * 58)
        col = "slope" if "DECAY" not in cls else "slope"
        ascending = "DECAY" in cls
        for _, row in subset.sort_values("slope", ascending=ascending).head(top_n).iterrows():
            print(f"    {row['id']:<15}  {row['slope']:>9.1f}  "
                  f"{row['r_squared']:>5.2f}  {row['phase_sequence']:<14}  "
                  f"{row['recommended_N_hint']}")

    print(f"\n{sep}\n")


# --------------------------------------------------------------------------- #
# ENTRY POINT
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="SEVIR bulk event classifier (growth/decay/lifecycle)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--out",  type=str, default=DEFAULT_OUT,
                        help="Output CSV path (default: event_catalogue.csv in DATA_ROOT)")
    parser.add_argument("--top",  type=int, default=10,
                        help="Number of top events to print per class")
    args = parser.parse_args()

    if not os.path.exists(CATALOG_PATH):
        print(f"ERROR: Catalog not found:\n  {CATALOG_PATH}")
        sys.exit(1)

    log.info(f"Loading catalog: {CATALOG_PATH}")
    catalog = pd.read_csv(CATALOG_PATH, low_memory=False)
    log.info(f"  {len(catalog):,} catalog rows loaded.")

    if not HAS_RUPTURES:
        log.warning("ruptures not installed — using simple midpoint phase fallback.")
        log.warning("  Install: pip install ruptures")
    if not HAS_SKIMAGE:
        log.warning("scikit-image not installed — spatial metrics disabled.")
        log.warning("  Install: pip install scikit-image")

    # ── Pass 1: collect raw stats ─────────────────────────────────────────────
    records = pass1_collect(catalog)
    if not records:
        log.error("No events loaded — check DATA_ROOT and file paths.")
        sys.exit(1)

    # ── Derive percentile thresholds ──────────────────────────────────────────
    thresholds = derive_thresholds(records)

    # ── Pass 2: segment phases + classify ────────────────────────────────────
    df = pass2_classify(records, thresholds)

    # ── Save catalogue ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    log.info(f"Catalogue saved → {args.out}")

    # ── Also save threshold values alongside the catalogue ───────────────────
    thresh_path = args.out.replace(".csv", "_thresholds.csv")
    pd.DataFrame([thresholds]).to_csv(thresh_path, index=False)
    log.info(f"Thresholds saved → {thresh_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print_summary(df, top_n=args.top)

    # ── Stratified sample hint for diagnostic_n_sweep.py ─────────────────────
    print("  Stratified sample for N-sweep (10 per class):")
    sample_ids = (
        df.groupby("lifecycle_class", group_keys=False)
          .apply(lambda g: g.sample(min(10, len(g)), random_state=42))
    )["id"].tolist()
    sample_path = args.out.replace(".csv", "_sweep_sample.txt")
    with open(sample_path, "w") as f:
        f.write("\n".join(sample_ids))
    print(f"  {len(sample_ids)} event IDs written → {sample_path}")
    print(f"  Use with: python diagnostic_n_sweep.py --events_file {sample_path}\n")


if __name__ == "__main__":
    main()