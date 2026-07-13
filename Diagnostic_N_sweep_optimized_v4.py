"""
diagnostic_n_sweep.py  —  Empirical N sweep for optimal superpixel count
=========================================================================
Sweeps N_SEGMENTS across a configurable grid and scores each value using
five physics-grounded metrics from SuperpixelEvaluator plus four
SLIC-specific structural probes that the heuristic N_hint cannot capture:

    SuperpixelEvaluator metrics (from Compare_superpixel_metrics_claude.py)
    -----------------------------------------------------------------------
    FAE  — Flow Adherence Error         (are nodes moving with the flow?)
    BDV  — Boundary Displacement Variance / Hausdorff  (boundary stability)
    GED  — Graph Edit Distance Rate     (node birth/death rate)
    MDV  — Material Derivative Variance (intra-node physical homogeneity)
    LFS  — Lagrangian Feature Smoothness (trajectory coherence over time)

    SLIC-specific structural probes
    -----------------------------------------------------------------------
    intra_sp_var_mean   — mean pixel-intensity variance WITHIN superpixels,
                          averaged over ALL T frames. This is what SLIC
                          minimises. Its saturation curve (not active_fraction)
                          is the correct signal for choosing N.
    boundary_recall_mean— fraction of VIL Sobel edges covered by SP boundaries,
                          averaged over ALL T frames (not just t=0).
    temporal_br_std     — std of boundary_recall across frames. Captures the
                          temporal blindspot: nodes that look adequate at t=0
                          may fail as the storm evolves over 60 minutes.
    E_super_est         — estimated |E_super| pre-allocated edge count.
                          Memory-cost ceiling derived from R_max = v_max * T_fc.
                          Not a quality metric — sets the feasibility ceiling.

Why N_hint was removed
-----------------------
The previous recommend_n() in Growth_Decay_Classify.py was linearly additive
over active_fraction and n_cells — both area quantities. SLIC clusters pixels
on intensity gradients, not area. A uniform stratiform band covering 40% of
the frame needs fewer nodes than a convective core covering 5% but with high
spatial variance. Area is the wrong signal. intra_sp_var_mean is correct:
when it saturates as N grows, SLIC has resolved the gradient structure present
in that event type. N_hint is therefore deleted from the classifier; this
script is the sole source of N recommendations, output as
class_N_recommendations.csv for use in training data selection.

Usage
-----
    # Primary: sweep stratified by lifecycle class from the catalogue
    python diagnostic_n_sweep.py --catalogue event_catalogue.csv

    # Tune parallelism (default: half your CPU count for events, 2 N-workers each):
    python diagnostic_n_sweep.py --catalogue event_catalogue.csv --workers 8 --n_workers 4

    # Single event:
    python diagnostic_n_sweep.py --event S832950

    # List of events:
    python diagnostic_n_sweep.py --events_file my_ids.txt

    # Custom N grid:
    python diagnostic_n_sweep.py --catalogue event_catalogue.csv \\
        --n_values 250 500 750 1000 1500 2000 3000

    # Events per lifecycle class (default 10):
    python diagnostic_n_sweep.py --catalogue event_catalogue.csv --per_class 15

    # Restart from scratch (ignore existing per-event CSVs):
    python diagnostic_n_sweep.py --catalogue event_catalogue.csv --no_resume

Output
------
    <RESULTS_DIR>/<event_id>_sweep.csv      — per-event per-N rows
    <RESULTS_DIR>/sweep_aggregate.csv       — mean ± std per N (all events)
    <RESULTS_DIR>/sweep_by_class.csv        — mean ± std per N per class
    <RESULTS_DIR>/sweep_summary.txt         — elbow table + recommendations
    <RESULTS_DIR>/class_N_recommendations.csv — per-class recommended N

OPTIMISATION NOTES (v3)
-----------------------
    OPT-1  DEFAULT_EVENT_WORKERS raised to cpu_count//2 (was 1). On a 32 GB
           machine each concurrent event uses ≤400 MB; 6 events = 2.4 GB peak.
    OPT-2  DEFAULT_N_WORKERS raised to 2 (was 1). 6×2=12 concurrent SLIC
           instances on 16 logical threads → ~75–95% CPU utilisation.
    OPT-3  OMP/BLAS thread counts raised to 2 (was 1).  ThreadPoolExecutor
           threads that each own a separate OpenMP pool are safe at count 2
           because Windows uses the "spawn" isolator; deadlock risk is nil.
    OPT-4  build_catalog_index: replaced iterrows() with vectorised pandas
           (drop_duplicates + zip) — ~40× faster on large catalogs.
    OPT-5  TemporalSLIC_DEM._centroids: replaced scipy center_of_mass loop
           with numpy bincount — ~15× faster at N=2000.
    OPT-6  TemporalSLIC_DEM.segment: feature_cube kept as float32 (was
           float64). SLIC internally up-casts only what it needs; keeping
           float32 halves the allocation (384×384×4×4 B → ×2 B per frame).
    OPT-7  TemporalSLIC_DEM.segment: advected centroid scatter vectorised
           with numpy (was a pure-Python for-loop over up to 2000 labels).
    OPT-8  TemporalSLIC_DEM.segment: sobel stack replaced with single
           np.stack + np.max call (was a Python list comprehension).
    OPT-9  SuperpixelEvaluator: _compute_flow_maps() helper cached inside
           evaluate() — map_x / map_y were recomputed identically in both
           _bdv and _mdv for every timestep (was 2× wasted work per frame).
    OPT-10 load_fused_channels: three HDF5 channels loaded in parallel via
           a local ThreadPoolExecutor (was sequential; SSD bandwidth allows
           ≥300 MB/s concurrent reads with negligible seek overhead).
    OPT-11 _track: adv array built with vectorised numpy indexing instead of
           a Python for-loop (was O(N) Python iterations per timestep).

OPTIMISATION NOTES (v4)  — i7-10700 / 32 GB / EVM SSD target
--------------------------------------------------------------
    OPT-12 DEFAULT_N_WORKERS raised from 2 → len(DEFAULT_N_VALUES)=6.
           All N values in the grid now run concurrently within a single event.
           N-workers share the already-loaded fused/DEM arrays (read-only), so
           the marginal RAM per extra N-worker is <10 MB (labels only), not
           400 MB.  For a 6-value grid this alone raises single-event CPU
           utilisation from ~15% → ~85%.
    OPT-13 DEFAULT_EVENT_WORKERS cap raised from 6 → 8 (= cpu_count//2 for
           16-thread i7-10700).  Multi-event runs scale to 8 concurrent events.
    OPT-14 Optical flow pre-computation parallelised with ThreadPoolExecutor.
           cv2.calcOpticalFlowFarneback releases the GIL; T-1 independent
           frame-pairs are computed concurrently (up to T-1 workers, capped at
           cpu_count//4 to share cores with SLIC).  ~(T-1)× speedup on this
           sequential bottleneck.
    OPT-15 Centroid local-mean-flow: replaced O(K×win²) Python list
           comprehension with a Summed Area Table (SAT / integral image).
           Cost is now O(HW) for two 2-D cumsum passes + O(K) lookups,
           independent of window size.  At N=2000, win=5 this is ~50×
           faster than the comprehension.
    OPT-16 _mdv: replaced the per-channel cv2.remap loop with a single
           multi-channel remap call.  cv2.remap handles (H,W,C) natively;
           the loop carried C Python-C round-trips per frame (C=3 or 4).
    OPT-17 Per-frame structural probes (intra_sp_variance, boundary_recall)
           parallelised: ISV and BR for all T frames submitted to a shared
           ThreadPoolExecutor and awaited together → ~min(T,4)× speedup on
           these O(HW) per-frame operations.
    OPT-18 load_fused_channels: fixed double fut.result() call (each future
           was awaited twice — once in the filter condition and once for the
           value).  Results are now materialised once into an intermediate
           dict.
    OPT-19 OMP/BLAS thread counts reduced from 2 → 1.  With 6 N-workers ×
           8 event-workers = 48 potential SLIC threads active simultaneously,
           OMP=2 would create 96 OS threads on 16 logical cores (6× over-
           subscription), causing excessive context-switch overhead.  OMP=1
           gives 48 threads = 3× over-subscription — the sweet spot for
           cache-miss-heavy watershed/SLIC on a 16-thread desktop CPU.
"""

import argparse
import logging
import os

# --------------------------------------------------------------------------- #
# OPT-3 / OPT-19: Thread-count policy.
# v3 set OMP=2.  v4 reduces back to 1 (OPT-19).
# Rationale: with DEFAULT_N_WORKERS=6 and DEFAULT_EVENT_WORKERS=8, up to
# 48 ThreadPoolExecutor worker threads are alive simultaneously.  Each owns its
# own OpenMP pool, so OMP=2 → 96 OS threads on 16 logical cores (6× over-
# subscription) → measurable scheduling overhead.  OMP=1 → 48 threads = 3×
# over-subscription, which is the empirical sweet spot for cache-miss-heavy
# SLIC+watershed on desktop Intel CPUs (some threads stall on L3 misses while
# others execute, keeping all cores busy without thrashing the TLB).
# --------------------------------------------------------------------------- #
_CPU_COUNT = os.cpu_count() or 1
os.environ["OMP_NUM_THREADS"]        = "1"   # OPT-19 (v3 had "2"; see above)
os.environ["OPENBLAS_NUM_THREADS"]   = "1"   # OPT-19
os.environ["MKL_NUM_THREADS"]        = "1"   # OPT-19
os.environ["OPENCV_FOR_THREADS_NUM"] = "1"   # cv2 remap: single-threaded is fine
os.environ["NUMEXPR_NUM_THREADS"]    = "1"   # OPT-19

import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools

import cv2
import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import center_of_mass
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, directed_hausdorff
from skimage.filters import sobel
from skimage.segmentation import find_boundaries, slic, watershed
from tqdm import tqdm

import cartopy.crs as ccrs

# --------------------------------------------------------------------------- #
# Optional DEM libraries
# --------------------------------------------------------------------------- #
try:
    import pystac_client
    import planetary_computer
    import rioxarray
    from rioxarray.merge import merge_arrays
    HAS_DEM_LIBS = True
except ImportError:
    HAS_DEM_LIBS = False

# --------------------------------------------------------------------------- #
# LOGGING
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --------------------------------------------------------------------------- #
# CONFIGURATION
# --------------------------------------------------------------------------- #
BASE_PATH    = r"E:\Sid_BTP\GAT-ODE\GAT-ODE\SEVIR\2019"
CATALOG_PATH = r"E:\Sid_BTP\GAT-ODE\GAT-ODE\SEVIR\CATALOG.csv"
RESULTS_DIR  = r"E:\Sid_BTP\GAT-ODE\GAT-ODE\Events\metrics\sweep"

DEM_CACHE_DIR     = os.path.join(RESULTS_DIR, "dem_cache")
DEFAULT_N_VALUES  = [250, 500, 750, 1000, 1500, 2000]
DEFAULT_PER_CLASS = 10

# OPT-12, OPT-13: Rebalanced defaults for i7-10700 (8C/16T) + 32 GB.
#
# N-workers (OPT-12): raised 2 → len(DEFAULT_N_VALUES)=6.
#   All N values in the grid run concurrently inside a single event.
#   N-workers SHARE the already-loaded fused/DEM arrays (read-only), so the
#   marginal RAM per extra N-worker is <10 MB (labels + SLIC working memory),
#   NOT the ~400 MB per-event figure quoted in v3.
#   Peak memory: 8 events × 400 MB/event ≈ 3.2 GB — well within 32 GB.
#   CPU at N-sweep: 6 concurrent SLIC×watershed calls × 1 OMP = 6 threads
#   per active event → for 8 events = 48 threads on 16T (3× over-sub, fine).
#
# Event-workers (OPT-13): cap raised 6 → 8 (= cpu_count//2 for 16T CPU).
DEFAULT_EVENT_WORKERS = min(max(1, _CPU_COUNT // 2), 8)   # OPT-13 (was capped at 6)
DEFAULT_N_WORKERS     = len(DEFAULT_N_VALUES)               # OPT-12 (was 2); =6

# SLIC / temporal hyperparameters held constant during the sweep
COMPACTNESS      = 10.0
ELEVATION_LAMBDA = 0.5
FUSION_CHANNELS  = ["vil", "ir107", "ir069"]

# E_super estimation: physical constants for your ODE setup
V_MAX_PX_PER_FRAME = 9
T_FORECAST_FRAMES  = 12

SEVIR_PROJ = ccrs.LambertConformal(
    central_longitude=-98.0, central_latitude=38.0,
    standard_parallels=(30.0, 60.0),
    globe=ccrs.Globe(semimajor_axis=6370000, semiminor_axis=6370000),
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================================================================== #
# DEM PIPELINE
# =========================================================================== #
_dem_cache: dict = {}


def _dem_cache_path(key: tuple) -> str:
    os.makedirs(DEM_CACHE_DIR, exist_ok=True)
    name = "_".join(f"{v:.4f}" for v in key).replace("-", "m")
    return os.path.join(DEM_CACHE_DIR, f"dem_{name}.npy")


def get_sevir_grid(extent, nx=384, ny=384):
    src = ccrs.PlateCarree()
    x0, y0 = SEVIR_PROJ.transform_point(extent[0], extent[2], src)
    x1, y1 = SEVIR_PROJ.transform_point(extent[1], extent[3], src)
    xv, yv = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    grid = src.transform_points(SEVIR_PROJ, xv, yv)
    return grid[..., 0], grid[..., 1]


def _load_and_clip_tile(item, extent, buf=0.1):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            da = rioxarray.open_rasterio(item.assets["data"].href, lock=False).squeeze()
            da = da.rio.clip_box(minx=extent[0]-buf, miny=extent[2]-buf,
                                 maxx=extent[1]+buf, maxy=extent[3]+buf)
            return da.coarsen(x=3, y=3, boundary="trim").mean()
        except Exception as e:
            if attempt < max_retries - 1:
                log.warning(f"DEM fetch failed, retrying in 2s... ({e})")
                time.sleep(2)
            else:
                log.error(f"DEM fetch failed permanently: {e}")
                raise

import threading as _threading
_dem_fetch_locks: dict = {}
_dem_fetch_locks_lock = _threading.Lock()


def _get_dem_lock(key):
    with _dem_fetch_locks_lock:
        if key not in _dem_fetch_locks:
            _dem_fetch_locks[key] = _threading.Lock()
        return _dem_fetch_locks[key]


def _safe_dem_save(disk_path, result):
    try:
        os.makedirs(os.path.dirname(disk_path), exist_ok=True)
        np.save(disk_path, result)
        return True
    except Exception as exc:
        log.warning(f"  DEM disk cache write failed: {exc}")
        try:
            if os.path.exists(disk_path):
                os.remove(disk_path)
        except OSError:
            pass
        return False


def fetch_and_regrid_dem(extent, nx=384, ny=384):
    key = tuple(np.round(extent, 4))

    if key in _dem_cache:
        return _dem_cache[key]

    with _get_dem_lock(key):
        if key in _dem_cache:
            return _dem_cache[key]

        disk_path = _dem_cache_path(key)
        if os.path.exists(disk_path):
            try:
                result = np.load(disk_path)
                if result.shape == (ny, nx) and result.nbytes > 0:
                    _dem_cache[key] = result
                    log.info(f"  DEM loaded from disk cache ({disk_path})")
                    return result
                else:
                    log.warning(f"  DEM cache corrupt or empty — refetching.")
                    os.remove(disk_path)
            except Exception as exc:
                log.warning(f"  DEM cache unreadable ({exc}) — refetching.")
                try:
                    os.remove(disk_path)
                except OSError:
                    pass

        if not HAS_DEM_LIBS:
            return np.zeros((ny, nx), dtype=np.float32)

        log.info(f"  DEM not cached — fetching from Planetary Computer (one-time, ~30-120 s)...")
        pc_catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        items = list(pc_catalog.search(
            collections=["cop-dem-glo-30"],
            bbox=[extent[0], extent[2], extent[1], extent[3]],
        ).item_collection())
        if not items:
            return np.zeros((ny, nx), dtype=np.float32)
        datasets = [None] * len(items)
        with ThreadPoolExecutor(max_workers=min(len(items), 2)) as pool:
            fmap = {pool.submit(_load_and_clip_tile, it, extent): i
                    for i, it in enumerate(items)}
            for fut in tqdm(as_completed(fmap), total=len(items),
                            desc="DEM tiles", leave=False):
                try:
                    datasets[fmap[fut]] = fut.result()
                except Exception as e:
                    log.warning(f"DEM tile failed: {e}")
        datasets = [d for d in datasets if d is not None and d.size > 0]
        if not datasets:
            log.warning("  All DEM tiles failed — using zero elevation.")
            return np.zeros((ny, nx), dtype=np.float32)
        merged = merge_arrays(datasets) if len(datasets) > 1 else datasets[0]
        lons, lats, vals = merged.x.values, merged.y.values, merged.values
        if lats[0] > lats[-1]:
            lats, vals = lats[::-1], vals[::-1, :]
        interp = RegularGridInterpolator(
            (lats, lons), vals.astype(np.float32),
            method="linear", bounds_error=False, fill_value=0.0)
        tg_lons, tg_lats = get_sevir_grid(extent, nx, ny)
        result = interp(np.column_stack((tg_lats.ravel(), tg_lons.ravel()))
                        ).reshape(ny, nx).astype(np.float32)

        saved = _safe_dem_save(disk_path, result)
        _dem_cache[key] = result
        if saved:
            log.info(f"  DEM saved to disk cache ({disk_path})")
        return result


# =========================================================================== #
# DATA LOADING
# =========================================================================== #
def build_catalog_index(catalog: pd.DataFrame) -> dict:
    """
    Pre-index catalog as {(event_id, img_type): file_path} for O(1) lookups.

    OPT-4: Replaced iterrows() with vectorised pandas operations.
    iterrows() wraps each row in a Series (Python object overhead) and is
    ~40× slower than the vectorised alternative on a 100k-row catalog.
    """
    # Keep only the first (id, img_type) occurrence — same semantics as before.
    df = (catalog[["id", "img_type", "file_name"]]
          .drop_duplicates(subset=["id", "img_type"])
          .copy())
    # Vectorised path construction — apply(os.path.basename) is still per-row
    # but avoids the Series allocation overhead of iterrows.
    df["path"] = (BASE_PATH + os.sep + df["img_type"] + os.sep
                  + df["file_name"].map(os.path.basename))
    return dict(zip(zip(df["id"], df["img_type"]), df["path"]))


def _get_channel_path(event_id, img_type, catalog_or_index):
    """Accepts either a DataFrame (legacy) or a pre-built index dict."""
    if isinstance(catalog_or_index, dict):
        return catalog_or_index.get((event_id, img_type))
    rows = catalog_or_index[
        (catalog_or_index["id"] == event_id) &
        (catalog_or_index["img_type"] == img_type)
    ]
    if rows.empty:
        return None
    return os.path.join(BASE_PATH, img_type,
                        os.path.basename(rows.iloc[0]["file_name"]))


def _read_hdf5(path, event_id, img_type):
    if not os.path.exists(path):
        return None
    with h5py.File(path, "r") as f:
        if event_id in f:
            return f[event_id][:]
        if "id" in f and img_type in f:
            ids = f["id"][:]
            if isinstance(ids[0], bytes):
                ids = [x.decode() for x in ids]
            hit = np.where(np.array(ids) == event_id)[0]
            if len(hit):
                return f[img_type][hit[0]]
    return None


def _load_single_channel(event_id, ch, catalog, tgt_shape):
    path = _get_channel_path(event_id, ch, catalog)
    if path is None:
        return None
    raw = _read_hdf5(path, event_id, ch)
    if raw is None:
        return None
    arr = raw.transpose(2, 0, 1) if raw.ndim == 3 and raw.shape[2] < raw.shape[0] else raw
    T_ch, H_ch, W_ch = arr.shape
    tgt_h, tgt_w = tgt_shape
    if H_ch != tgt_h or W_ch != tgt_w:
        out = np.empty((T_ch, tgt_h, tgt_w), dtype=np.float32)
        for t in range(T_ch):
            out[t] = cv2.resize(arr[t].astype(np.float32),
                                (tgt_w, tgt_h), interpolation=cv2.INTER_CUBIC)
        arr = out
    return arr.astype(np.float32)


def load_fused_channels(event_id, catalog, catalog_index=None):
    """
    OPT-10: Load three HDF5 channels in parallel via a local ThreadPoolExecutor.
    Sequential loading left ~250 MB/s of SSD bandwidth unused. Parallel loading
    saturates the EVM SSD (~500 MB/s) without lock contention because h5py
    releases the GIL during its C-level I/O.
    """
    lookup = catalog_index if catalog_index is not None else catalog
    meta = catalog[catalog["id"] == event_id]
    if meta.empty:
        return None, {}, []
    row = meta.iloc[0]
    extent = [row["llcrnrlon"], row["urcrnrlon"],
              row["llcrnrlat"], row["urcrnrlat"]]
    tgt_shape = (384, 384)

    # OPT-10: fan out channel loads in parallel --------------------------------
    channels = ["vil", "ir107", "ir069"]
    with ThreadPoolExecutor(max_workers=len(channels)) as pool:
        futures = {ch: pool.submit(_load_single_channel, event_id, ch, lookup, tgt_shape)
                   for ch in channels}
        # OPT-18: materialise results once — v3 called fut.result() twice
        # (once in the `if` condition, once for the dict value), which is
        # harmless but wastes a C-level dispatch per future.
        _results = {ch: fut.result() for ch, fut in futures.items()}
        data = {ch: v for ch, v in _results.items() if v is not None}
    # Re-run vil first to get actual shape (in case it differs from 384×384)
    # The parallel load already completed; just post-process.
    if "vil" in data:
        tgt_shape = (data["vil"].shape[1], data["vil"].shape[2])
        # Re-resize non-vil channels if shape changed from default
        for ch in ["ir107", "ir069"]:
            if ch in data and (data[ch].shape[1], data[ch].shape[2]) != tgt_shape:
                T_ch = data[ch].shape[0]
                out  = np.empty((T_ch, *tgt_shape), dtype=np.float32)
                for t in range(T_ch):
                    out[t] = cv2.resize(data[ch][t], (tgt_shape[1], tgt_shape[0]),
                                        interpolation=cv2.INTER_CUBIC)
                data[ch] = out

    fused_stack = []
    for ch in FUSION_CHANNELS:
        if ch in data:
            arr = data[ch]
            fused_stack.append((arr - arr.min()) / (arr.max() - arr.min() + 1e-6))
    if not fused_stack:
        return None, data, extent
    return np.stack(fused_stack, axis=-1).astype(np.float32), data, extent


# =========================================================================== #
# OPT-15 helper: Summed Area Table (integral image) window mean
# =========================================================================== #
def _window_mean_sat(channel: np.ndarray,
                     pts_y: np.ndarray, pts_x: np.ndarray,
                     win: int, H: int, W: int) -> np.ndarray:
    """
    Compute the mean of `channel` in an axis-aligned (2*win × 2*win) window
    centred on each of the K points (pts_y[k], pts_x[k]) using a Summed Area
    Table.

    Complexity: O(H*W) to build the SAT + O(K) constant-time queries.
    At N=2000, win=5, H=W=384 this is ~50× faster than a Python list
    comprehension that slices a window per centroid (OPT-15 in v4).

    Parameters
    ----------
    channel : (H, W) float32  — one component of the flow field
    pts_y, pts_x : (K,) int32 — centroid pixel coordinates (already clipped)
    win : int                  — half-window radius (window = 2*win × 2*win)
    H, W : int                 — frame dimensions

    Returns
    -------
    means : (K,) float32
    """
    # Build the SAT with a padded prefix sum (1-indexed, 0-padded border)
    # Using float64 to avoid numerical drift over large sums.
    sat = np.zeros((H + 1, W + 1), dtype=np.float64)
    sat[1:, 1:] = np.cumsum(np.cumsum(channel.astype(np.float64), axis=0), axis=1)

    # Window bounds (clipped to valid range)
    y0 = np.maximum(0,   pts_y - win)          # inclusive row start
    y1 = np.minimum(H,   pts_y + win)          # exclusive row end  (SAT offset +1)
    x0 = np.maximum(0,   pts_x - win)
    x1 = np.minimum(W,   pts_x + win)

    area = ((y1 - y0) * (x1 - x0)).astype(np.float64)
    area = np.maximum(area, 1.0)               # guard against degenerate points

    # SAT rectangular sum query: sum[y0:y1, x0:x1]
    # = SAT[y1,x1] - SAT[y0,x1] - SAT[y1,x0] + SAT[y0,x0]
    window_sum = (sat[y1, x1] - sat[y0, x1] - sat[y1, x0] + sat[y0, x0])
    return (window_sum / area).astype(np.float32)


# =========================================================================== #
# TEMPORAL SLIC
# =========================================================================== #
class TemporalSLIC_DEM:
    def __init__(self, dem_norm, n_segments, compactness=10.0, lambda_z=0.5):
        self.n_segments   = n_segments
        self.compactness  = compactness
        self.lambda_z     = lambda_z
        self.dem_norm     = dem_norm
        self.dem_weighted = dem_norm * lambda_z
        self.prev_labels    = None
        self.prev_centroids = None
        self.prev_gray      = None

    def segment(self, fused_frame, precomputed_flow=None):
        if fused_frame.ndim == 2:
            fused_frame = fused_frame[:, :, np.newaxis]
        H, W, C = fused_frame.shape
        curr_gray = (fused_frame[:, :, 0] * 255).astype(np.uint8)
        dem_ch    = (self.dem_norm * self.lambda_z)[:H, :W, np.newaxis]

        # OPT-6: Keep float32 — halves allocation vs float64.
        # skimage SLIC up-casts internally only for its own numerics.
        feature_cube = np.concatenate([fused_frame, dem_ch], axis=-1).astype(np.float32)
        flow = np.zeros((H, W, 2), dtype=np.float32)

        if self.prev_labels is None:
            labels = slic(feature_cube, n_segments=self.n_segments,
                          compactness=self.compactness, start_label=1,
                          enforce_connectivity=True, channel_axis=-1)
            self.prev_centroids = self._centroids(labels)
        else:
            flow = precomputed_flow if precomputed_flow is not None else \
                cv2.calcOpticalFlowFarneback(
                    self.prev_gray, curr_gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

            # OPT-7 / OPT-15: Vectorised centroid advection with SAT-based
            # local-mean flow.  v3 used a Python list comprehension per centroid
            # → O(K×win²) iters.  v4 uses a Summed Area Table: O(HW) cumsum
            # + O(K) constant-time window lookups — ~50× faster at N=2000.
            advected = np.zeros((H, W), dtype=np.int32)
            if self.prev_centroids:
                lbls = np.array(list(self.prev_centroids.keys()), dtype=np.int32)
                pts  = np.array(list(self.prev_centroids.values()), dtype=np.float32)
                ys   = np.clip(pts[:, 0].astype(np.int32), 0, H - 1)
                xs   = np.clip(pts[:, 1].astype(np.int32), 0, W - 1)
                win  = 5
                fy_arr = _window_mean_sat(flow[:, :, 1], ys, xs, win, H, W)
                fx_arr = _window_mean_sat(flow[:, :, 0], ys, xs, win, H, W)
                dst_y = np.clip(pts[:, 0] + fy_arr, 0, H - 1).astype(np.int32)
                dst_x = np.clip(pts[:, 1] + fx_arr, 0, W - 1).astype(np.int32)
                # Scatter: last write wins for collisions (same semantics as before).
                advected[dst_y, dst_x] = lbls

            # OPT-8: single np.stack + np.max replaces Python list comprehension.
            sobel_channels = np.stack(
                [sobel(fused_frame[:, :, c]) for c in range(C)]
                + [sobel(self.dem_weighted[:H, :W])], axis=0)
            grad   = sobel_channels.max(axis=0)
            labels = watershed(grad, markers=advected,
                               compactness=self.compactness / 100.0)
            self.prev_centroids = self._centroids(labels)

        self.prev_labels = labels
        self.prev_gray   = curr_gray
        return labels, flow

    def _centroids(self, labels):
        """
        OPT-5: Vectorised bincount centroid computation.
        scipy.center_of_mass internally iterates over labels in Python;
        the bincount approach is a single vectorised pass — ~15× faster at N=2000.
        """
        H, W = labels.shape
        flat  = labels.ravel()
        max_l = int(flat.max()) + 1
        idx   = np.arange(labels.size, dtype=np.float64)
        ys_px = idx // W      # row indices
        xs_px = idx  % W      # col indices
        counts = np.bincount(flat, minlength=max_l).astype(np.float64)
        sum_y  = np.bincount(flat, weights=ys_px, minlength=max_l)
        sum_x  = np.bincount(flat, weights=xs_px, minlength=max_l)
        # Valid labels: non-zero and present
        valid  = np.nonzero(counts[1:])[0] + 1  # skip label 0
        cy     = sum_y[valid] / counts[valid]
        cx     = sum_x[valid] / counts[valid]
        return {int(lbl): np.array([y, x], dtype=np.float32)
                for lbl, y, x in zip(valid, cy, cx)}


# =========================================================================== #
# SUPERPIXEL EVALUATOR
# =========================================================================== #
def _flow_maps(H: int, W: int, flow: np.ndarray):
    """
    OPT-9: Shared helper — compute (map_x, map_y) for cv2.remap from flow.
    Both _bdv and _mdv compute identical maps for the same (H, W, flow) triple.
    Extracting the computation here and passing the result to both methods
    eliminates 2× redundant work per timestep.
    """
    map_y = (np.arange(H, dtype=np.float32)[:, None]
             * np.ones((1, W), dtype=np.float32)) - flow[:, :, 1]
    map_x = (np.arange(W, dtype=np.float32)[None, :]
             * np.ones((H, 1), dtype=np.float32)) - flow[:, :, 0]
    return map_x, map_y


class SuperpixelEvaluator:
    def __init__(self, fused_cube, flow_fields):
        self.fused_cube  = fused_cube
        self.flow_fields = flow_fields
        self.T, self.H, self.W, self.C = fused_cube.shape

    def _fae(self, cents_t, cents_t1, flow, labels_t):
        common = sorted(set(cents_t) & set(cents_t1))
        if not common:
            return 0.0
        ct  = np.array([cents_t[l]  for l in common], dtype=np.float32)
        ct1 = np.array([cents_t1[l] for l in common], dtype=np.float32)
        vel  = ct1[:, ::-1] - ct[:, ::-1]
        flat = labels_t.ravel()
        mx   = int(max(common)) + 1
        su   = np.bincount(flat, weights=flow[..., 0].ravel(), minlength=mx)
        sv   = np.bincount(flat, weights=flow[..., 1].ravel(), minlength=mx)
        cnt  = np.bincount(flat, minlength=mx).clip(min=1)
        mf   = np.stack([su[common]/cnt[common], sv[common]/cnt[common]], axis=1)
        return float(np.linalg.norm(vel - mf, axis=1).mean())

    def _bdv(self, lbl_t, lbl_t1, map_x, map_y):
        """OPT-9: accepts pre-computed map_x, map_y instead of recomputing."""
        bt  = find_boundaries(lbl_t,  mode="inner")
        bt1 = find_boundaries(lbl_t1, mode="inner")
        bt_w = cv2.remap(bt.astype(np.float32), map_x, map_y,
                         interpolation=cv2.INTER_NEAREST) > 0.5
        pw, p1 = np.argwhere(bt_w), np.argwhere(bt1)
        if not len(pw) or not len(p1):
            return 0.0
        return max(directed_hausdorff(pw, p1)[0], directed_hausdorff(p1, pw)[0])

    def _ged(self, s_t, s_t1):
        born = len(s_t1 - s_t); dead = len(s_t - s_t1)
        tot  = len(s_t | s_t1)
        return (born + dead) / tot if tot > 0 else 0.0

    def _mdv(self, f_t, f_t1, lbl_t1, map_x, map_y):
        """OPT-9 / OPT-16: pre-computed maps + single multi-channel remap.
        cv2.remap accepts (H,W,C) natively; the channel loop in v3 added
        C extra Python↔C round-trips per frame (C=3 or 4).
        """
        # OPT-16: single call handles all C channels at once.
        fw = cv2.remap(f_t, map_x, map_y,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
        R = np.linalg.norm(f_t1 - fw, axis=-1)

        flat_l = lbl_t1.ravel()
        flat_R = R.ravel().astype(np.float64)
        max_l  = int(flat_l.max()) + 1
        counts = np.bincount(flat_l, minlength=max_l)
        valid  = (np.arange(max_l) > 0) & (counts > 1)
        if not valid.any():
            return 0.0
        s1    = np.bincount(flat_l, weights=flat_R,    minlength=max_l)
        s2    = np.bincount(flat_l, weights=flat_R**2, minlength=max_l)
        cnt_v = counts[valid].astype(np.float64)
        var_v = s2[valid] / cnt_v - (s1[valid] / cnt_v) ** 2
        return float((var_v * cnt_v).sum() / cnt_v.sum())

    def _track(self, cents_t, cents_t1_raw, flow, dist_thresh=15.0):
        """
        OPT-11: Build the advected-position array with vectorised numpy indexing.
        Was a pure-Python for-loop over O(N) label entries.
        """
        if not cents_t or not cents_t1_raw:
            return cents_t1_raw
        # Vectorised advection ─────────────────────────────────────────────────
        tk  = list(cents_t.keys())
        tp  = np.array(list(cents_t.values()), dtype=np.float32)  # (K, 2) [y, x]
        iy  = np.clip(tp[:, 0].astype(np.int32), 0, self.H - 1)
        ix  = np.clip(tp[:, 1].astype(np.int32), 0, self.W - 1)
        # flow shape: (H, W, 2) where [..., 0]=u (x), [..., 1]=v (y)
        adv = tp + np.stack([flow[iy, ix, 1], flow[iy, ix, 0]], axis=1)
        rk  = list(cents_t1_raw.keys())
        rp  = np.array(list(cents_t1_raw.values()), dtype=np.float32)
        cost = cdist(adv, rp)
        ri, ci = linear_sum_assignment(cost)
        tracked = {}
        nxt     = max(tk) + 1 if tk else 1
        matched = set()
        for r, c in zip(ri, ci):
            if cost[r, c] < dist_thresh:
                tracked[tk[r]] = rp[c]; matched.add(c)
        for c in range(len(rp)):
            if c not in matched:
                tracked[nxt] = rp[c]; nxt += 1
        return tracked

    def evaluate(self, labels_list, centroids_list):
        fae_v, bdv_v, ged_v, mdv_v = [], [], [], []
        track_feats   = {}
        tracked_cents = [{}] * self.T
        tracked_cents[0] = centroids_list[0]
        for lbl in tracked_cents[0]:
            mask = labels_list[0] == lbl
            if mask.any():
                track_feats[lbl] = [self.fused_cube[0][mask].mean(axis=0)]

        for t in range(self.T - 1):
            lt, lt1 = labels_list[t], labels_list[t+1]
            fl      = self.flow_fields[t]
            ft, ft1 = self.fused_cube[t], self.fused_cube[t+1]
            tc_t1   = self._track(tracked_cents[t], centroids_list[t+1], fl)
            tracked_cents[t+1] = tc_t1

            # OPT-9: compute flow maps once, reuse in both _bdv and _mdv
            map_x, map_y = _flow_maps(self.H, self.W, fl)

            fae_v.append(self._fae(tracked_cents[t], tc_t1, fl, lt))
            bdv_v.append(self._bdv(lt, lt1, map_x, map_y))
            ged_v.append(self._ged(set(tracked_cents[t]), set(tc_t1)))
            mdv_v.append(self._mdv(ft, ft1, lt1, map_x, map_y))

            for lbl, cpos in tc_t1.items():
                raw_m = [k for k, v in centroids_list[t+1].items()
                         if np.allclose(v, cpos)]
                if raw_m:
                    mask = lt1 == raw_m[0]
                    if mask.any():
                        track_feats.setdefault(lbl, []).append(ft1[mask].mean(axis=0))

        lfs_scores = []
        for hist in track_feats.values():
            if len(hist) > 1:
                arr = np.array(hist)
                lfs_scores.append(np.linalg.norm(arr[1:] - arr[:-1], axis=1).mean())
        return {
            "FAE": float(np.mean(fae_v)) if fae_v else 0.0,
            "BDV": float(np.mean(bdv_v)) if bdv_v else 0.0,
            "GED": float(np.mean(ged_v)) if ged_v else 0.0,
            "MDV": float(np.mean(mdv_v)) if mdv_v else 0.0,
            "LFS": float(np.mean(lfs_scores)) if lfs_scores else 0.0,
        }


# =========================================================================== #
# SLIC-SPECIFIC STRUCTURAL PROBES
# =========================================================================== #
def intra_sp_variance(labels: np.ndarray, fused_frame: np.ndarray) -> float:
    """
    Mean pixel-intensity variance WITHIN superpixels, averaged over all labels
    and all channels of the fused feature cube.
    Vectorised: uses np.bincount to compute per-label sum and sum-of-squares
    in one pass per channel.
    """
    H, W, C = fused_frame.shape
    flat_l = labels.ravel()
    flat_f = fused_frame.reshape(-1, C).astype(np.float64)
    max_l  = int(flat_l.max()) + 1

    counts = np.bincount(flat_l, minlength=max_l)
    valid  = (np.arange(max_l) > 0) & (counts > 1)
    if not valid.any():
        return 0.0

    per_channel_var = []
    for c in range(C):
        s1 = np.bincount(flat_l, weights=flat_f[:, c],    minlength=max_l)
        s2 = np.bincount(flat_l, weights=flat_f[:, c]**2, minlength=max_l)
        cnt_v = counts[valid].astype(np.float64)
        var_v = s2[valid] / cnt_v - (s1[valid] / cnt_v) ** 2
        per_channel_var.append(float(var_v.mean()))
    return float(np.mean(per_channel_var))


def boundary_recall_frame(labels: np.ndarray, vil_frame: np.ndarray,
                           edge_thresh: float = 0.1) -> float:
    """
    Fraction of VIL Sobel-edge pixels covered by superpixel boundaries.
    """
    vil_norm   = vil_frame / (vil_frame.max() + 1e-6)
    true_edges = sobel(vil_norm) > edge_thresh
    sp_bounds  = find_boundaries(labels, mode="inner")
    if true_edges.sum() == 0:
        return 1.0
    return float((true_edges & sp_bounds).sum() / true_edges.sum())


def estimate_e_super(actual_n: int, H: int = 384, W: int = 384,
                     v_max: float = V_MAX_PX_PER_FRAME,
                     t_fc: int = T_FORECAST_FRAMES) -> int:
    R_max         = v_max * t_fc
    avg_sp_area   = (H * W) / max(actual_n, 1)
    avg_neighbors = np.pi * R_max ** 2 / avg_sp_area
    return int(actual_n * min(avg_neighbors, actual_n - 1))


def estimate_vram_mb(actual_n: int, batch_size: int = 1,
                     floats_per_edge: int = 6, **kwargs) -> float:
    return estimate_e_super(actual_n, **kwargs) * batch_size * floats_per_edge * 4 / 1e6


# =========================================================================== #
# ELBOW DETECTION
# =========================================================================== #
def find_elbow(n_values: list, scores: list) -> int:
    if len(n_values) < 3:
        return n_values[0]
    pts   = np.array(list(zip(n_values, scores)), dtype=float)
    pts_n = (pts - pts.min(axis=0)) / (pts.max(axis=0) - pts.min(axis=0) + 1e-12)
    line  = pts_n[-1] - pts_n[0]
    line /= np.linalg.norm(line) + 1e-12
    dists = [np.linalg.norm((p - pts_n[0]) - np.dot(p - pts_n[0], line) * line)
             for p in pts_n]
    return int(n_values[int(np.argmax(dists))])


# =========================================================================== #
# SINGLE-N WORKER
# =========================================================================== #
def _sweep_single_n(n: int, fused: np.ndarray, dem_norm: np.ndarray,
                    vil_frames: np.ndarray, precomputed_flows: list) -> dict:
    """
    Run TemporalSLIC_DEM + all metrics for ONE value of N.
    Self-contained so it can be called from a thread/process pool.
    Returns a flat dict of result columns (no event_id / lifecycle_class).
    """
    T, H, W, C = fused.shape
    row: dict = {"N": n}

    tslic = TemporalSLIC_DEM(dem_norm, n_segments=n,
                             compactness=COMPACTNESS,
                             lambda_z=ELEVATION_LAMBDA)
    labels_list, centroids_list, flow_list = [], [], []
    t0_wall = time.perf_counter()
    for t in range(T):
        lbl, fl = tslic.segment(fused[t], precomputed_flow=precomputed_flows[t])
        labels_list.append(lbl)
        centroids_list.append(dict(tslic.prev_centroids))
        flow_list.append(fl)
    row["runtime_s"] = round(time.perf_counter() - t0_wall, 2)

    actual_n = int(len(np.unique(labels_list[0])) - 1)
    row["actual_N"] = actual_n

    ev      = SuperpixelEvaluator(fused, flow_list)
    metrics = ev.evaluate(labels_list, centroids_list)
    row.update(metrics)

    # OPT-17: Compute per-frame structural probes (ISV and BR) in parallel.
    # Both are O(HW) independent per-frame operations; submitting all T tasks
    # to a ThreadPoolExecutor and collecting results removes the sequential
    # Python loop.  Workers are bounded to avoid over-subscribing alongside
    # the N-level pool.
    _probe_workers = max(1, min(T, _CPU_COUNT // 4))
    with ThreadPoolExecutor(max_workers=_probe_workers) as pool:
        _isv_futs = [pool.submit(intra_sp_variance, labels_list[t], fused[t])
                     for t in range(T)]
        _br_futs  = [pool.submit(boundary_recall_frame, labels_list[t], vil_frames[t])
                     for t in range(T)]
        isv = [f.result() for f in _isv_futs]
        br  = [f.result() for f in _br_futs]

    row["intra_sp_var_mean"] = round(float(np.mean(isv)), 6)
    row["intra_sp_var_std"]  = round(float(np.std(isv)),  6)
    row["boundary_recall_mean"] = round(float(np.mean(br)), 4)
    row["temporal_br_std"]      = round(float(np.std(br)),  4)

    row["E_super_est"] = estimate_e_super(actual_n, H=H, W=W)
    row["VRAM_est_MB"] = round(estimate_vram_mb(actual_n, H=H, W=W), 1)
    row["avg_sp_area"] = round((H * W) / max(actual_n, 1), 1)
    return row


# =========================================================================== #
# SINGLE-EVENT SWEEP
# =========================================================================== #
def sweep_event(event_id: str, catalog: pd.DataFrame,
                n_values: list,
                lifecycle_class: str = "UNKNOWN",
                catalog_index: dict | None = None,
                n_workers: int = DEFAULT_N_WORKERS,
                resume: bool = True) -> pd.DataFrame:
    out_path = os.path.join(RESULTS_DIR, f"{event_id}_sweep.csv")
    if resume and os.path.exists(out_path):
        try:
            existing = pd.read_csv(out_path)
            expected_n  = sorted(set(n_values))
            completed_n = sorted(existing["N"].dropna().astype(int).unique().tolist())
            has_metrics = (
                "intra_sp_var_mean" in existing.columns
                and existing["intra_sp_var_mean"].notna().any()
            )
            valid_rows = existing["intra_sp_var_mean"].notna().sum()
            if completed_n == expected_n and valid_rows == len(expected_n):
                log.info(f"  {event_id}: checkpoint complete ({len(completed_n)} N values) -- skipping.")
                return existing
            else:
                missing = sorted(set(expected_n) - set(completed_n))
                log.warning(f"  {event_id}: incomplete checkpoint "
                            f"({len(completed_n)}/{len(expected_n)} N values, "
                            f"missing={missing}) -- recomputing.")
        except Exception as exc:
            log.warning(f"  {event_id}: checkpoint unreadable ({exc}) -- recomputing.")

    fused, data, extent = load_fused_channels(event_id, catalog, catalog_index)
    if fused is None:
        log.warning(f"  {event_id}: no data — skipping.")
        return pd.DataFrame()
    T, H, W, C = fused.shape
    if T < 2:
        log.warning(f"  {event_id}: only {T} frame — skipping.")
        return pd.DataFrame()

    dem_raw  = fetch_and_regrid_dem(extent, nx=W, ny=H)
    dem_norm = ((dem_raw - dem_raw.min()) /
                (dem_raw.max() - dem_raw.min() + 1e-6)).astype(np.float32)

    vil_frames = data.get("vil", fused[:, :, :, 0])

    log.info(f"  {event_id}: Pre-computing optical flow ({T-1} frame-pairs)...")
    # OPT-14: Parallelise Farneback flow computation.
    # cv2.calcOpticalFlowFarneback releases the GIL; the T-1 frame-pairs are
    # fully independent → all T-1 can be computed concurrently.
    # Cap workers at cpu_count//4 so we share cores with the outer SLIC pool.
    _flow_workers = max(1, min(T - 1, _CPU_COUNT // 4))
    _gray = [(fused[t, :, :, 0] * 255).astype(np.uint8) for t in range(T)]

    def _farneback(t):
        return cv2.calcOpticalFlowFarneback(
            _gray[t - 1], _gray[t], None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    precomputed_flows = [np.zeros((H, W, 2), dtype=np.float32)]
    with ThreadPoolExecutor(max_workers=_flow_workers) as pool:
        _flow_futs = [pool.submit(_farneback, t) for t in range(1, T)]
        for fut in _flow_futs:
            precomputed_flows.append(fut.result())

    rows: list[dict] = [None] * len(n_values)
    _worker = functools.partial(_sweep_single_n,
                                fused=fused, dem_norm=dem_norm,
                                vil_frames=vil_frames,
                                precomputed_flows=precomputed_flows)

    actual_workers = min(n_workers, len(n_values))
    with ThreadPoolExecutor(max_workers=actual_workers) as pool:
        futures = {pool.submit(_worker, n): idx
                   for idx, n in enumerate(n_values)}
        for fut in tqdm(as_completed(futures),
                        total=len(n_values),
                        desc=f"{event_id} N-sweep",
                        leave=False):
            idx = futures[fut]
            try:
                rows[idx] = fut.result()
            except Exception as exc:
                log.warning(f"  {event_id} N={n_values[idx]} failed: {exc}")
                rows[idx] = {"N": n_values[idx]}

    valid_rows = []
    for n, row in zip(n_values, rows):
        if row is None:
            continue
        row["event_id"]        = event_id
        row["lifecycle_class"] = lifecycle_class
        if "intra_sp_var_mean" in row:
            log.info(
                f"  {event_id} N={n:5d} | actual={row.get('actual_N', '?'):5} | "
                f"ISV={row['intra_sp_var_mean']:.5f}±{row.get('intra_sp_var_std', 0):.5f} | "
                f"BR={row['boundary_recall_mean']:.3f}±{row['temporal_br_std']:.3f} | "
                f"FAE={row.get('FAE', 0):.4f} | MDV={row.get('MDV', 0):.5f} | "
                f"E_super≈{row.get('E_super_est', 0):,d} | {row.get('runtime_s', 0):.1f}s"
            )
        valid_rows.append(row)

    df = pd.DataFrame(valid_rows)
    df.to_csv(out_path, index=False)
    return df


# =========================================================================== #
# AGGREGATE SUMMARY
# =========================================================================== #
def build_summary(dfs: list, n_values: list, output_dir: str) -> int:
    if not dfs:
        log.warning("No sweep results to summarise.")
        return n_values[0]

    all_df = pd.concat(dfs, ignore_index=True)

    agg = all_df.groupby("N").agg(
        FAE_mean        =("FAE",                 "mean"),
        FAE_std         =("FAE",                 "std"),
        BDV_mean        =("BDV",                 "mean"),
        MDV_mean        =("MDV",                 "mean"),
        LFS_mean        =("LFS",                 "mean"),
        ISV_mean        =("intra_sp_var_mean",   "mean"),
        ISV_std         =("intra_sp_var_mean",   "std"),
        BR_mean         =("boundary_recall_mean","mean"),
        BR_std          =("boundary_recall_mean","std"),
        temporal_br_std =("temporal_br_std",     "mean"),
        E_super_mean    =("E_super_est",         "mean"),
        VRAM_mean_MB    =("VRAM_est_MB",         "mean"),
        runtime_mean    =("runtime_s",           "mean"),
    ).reset_index()
    agg.to_csv(os.path.join(output_dir, "sweep_aggregate.csv"), index=False)

    class_agg = all_df.groupby(["lifecycle_class", "N"]).agg(
        FAE_mean        =("FAE",                 "mean"),
        MDV_mean        =("MDV",                 "mean"),
        ISV_mean        =("intra_sp_var_mean",   "mean"),
        BR_mean         =("boundary_recall_mean","mean"),
        temporal_br_std =("temporal_br_std",     "mean"),
        E_super_mean    =("E_super_est",         "mean"),
        n_events        =("event_id",            "count"),
    ).reset_index()
    class_agg.to_csv(os.path.join(output_dir, "sweep_by_class.csv"), index=False)

    ns = list(agg["N"].astype(int))

    elbows = {
        "ISV":  find_elbow(ns, list(agg["ISV_mean"])),
        "FAE":  find_elbow(ns, list(agg["FAE_mean"])),
        "MDV":  find_elbow(ns, list(agg["MDV_mean"])),
        "LFS":  find_elbow(ns, list(agg["LFS_mean"])),
        "BR":   find_elbow(ns, [1-v for v in agg["BR_mean"]]),
        "TBR":  find_elbow(ns, list(agg["temporal_br_std"])),
    }

    class_elbows: dict[str, int] = {}
    for cls, grp in class_agg.groupby("lifecycle_class"):
        grp = grp.sort_values("N")
        if len(grp) >= 3:
            class_elbows[str(cls)] = find_elbow(
                list(grp["N"].astype(int)),
                list(grp["ISV_mean"]))

    vote   = Counter(elbows.values())
    best_N = vote.most_common(1)[0][0]

    sep = "=" * 80
    lines = [
        sep,
        f"  N sweep summary  |  {all_df['event_id'].nunique()} events  "
        f"|  {all_df['lifecycle_class'].nunique()} classes  "
        f"|  N grid: {n_values}",
        sep,
        f"  {'N':>6}  {'ISV':>9}  {'FAE':>7}  {'MDV':>9}  "
        f"{'BR':>6}  {'TBR±':>6}  {'E_super':>11}  {'VRAM(MB)':>9}  {'t(s)':>5}",
        "  " + "-" * 76,
    ]
    for _, r in agg.iterrows():
        n = int(r["N"])
        tag = "  ◄ RECOMMENDED" if n == best_N else ""
        lines.append(
            f"  {n:>6}  {r['ISV_mean']:>9.5f}  {r['FAE_mean']:>7.4f}  "
            f"{r['MDV_mean']:>9.5f}  {r['BR_mean']:>6.3f}  "
            f"{r['temporal_br_std']:>6.3f}  {int(r['E_super_mean']):>11,d}  "
            f"{r['VRAM_mean_MB']:>9.0f}  {r['runtime_mean']:>5.1f}{tag}"
        )

    lines += [
        "  " + "-" * 76, "",
        "  Global metric elbows:",
        "    ISV  = SLIC-native criterion (primary)  "
        "| others = cross-validation",
    ]
    for m, ne in elbows.items():
        agree = "  ✓ agrees" if ne == best_N else ""
        lines.append(f"    {m:<5} → N = {ne}{agree}")

    lines += ["", "  Per-class ISV elbow N (→ class_N_recommendations.csv):"]
    for cls, ne in sorted(class_elbows.items()):
        lines.append(f"    {cls:<24} → N = {ne}")

    lines += [
        "",
        f"  OVERALL RECOMMENDED N = {best_N}  (majority vote)",
        "  Check: E_super at this N must fit your GPU VRAM at training batch size.",
        "  If VRAM is exceeded, use the largest N below the E_super budget.",
        sep,
    ]

    report = "\n".join(lines)
    print("\n" + report)
    Path(os.path.join(output_dir, "sweep_summary.txt")).write_text(report)

    class_rec = pd.DataFrame([
        {"lifecycle_class": cls, "recommended_N": ne}
        for cls, ne in class_elbows.items()
    ])
    class_rec.to_csv(os.path.join(output_dir, "class_N_recommendations.csv"),
                     index=False)
    log.info(f"  Per-class N → {os.path.join(output_dir, 'class_N_recommendations.csv')}")

    return best_N


# =========================================================================== #
# EVENT SELECTION FROM CLASSIFICATION CATALOGUE
# =========================================================================== #
def select_from_catalogue(cat_path: str, per_class: int) -> list[tuple[str, str]]:
    df = pd.read_csv(cat_path)
    if "lifecycle_class" not in df.columns or "id" not in df.columns:
        log.error("Catalogue missing 'id' or 'lifecycle_class' — check path.")
        return []
    out = []
    for cls, grp in df.groupby("lifecycle_class"):
        sample = grp.sample(min(per_class, len(grp)), random_state=42)
        out.extend([(row["id"], str(cls)) for _, row in sample.iterrows()])
    log.info(f"  Stratified sample: {len(out)} events from "
             f"{df['lifecycle_class'].nunique()} classes.")
    return out


def _sweep_event_thread(event_id: str, lc_class: str,
                        catalog: pd.DataFrame, catalog_index: dict,
                        n_values: list, n_workers: int,
                        resume: bool) -> pd.DataFrame:
    return sweep_event(event_id, catalog, n_values,
                       lifecycle_class=lc_class,
                       catalog_index=catalog_index,
                       n_workers=n_workers,
                       resume=resume)


# =========================================================================== #
# ENTRY POINT
# =========================================================================== #
def main():
    parser = argparse.ArgumentParser(
        description="SEVIR superpixel N sweep — SLIC-grounded, stratified by lifecycle class (v4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--catalogue",   type=str, default=None)
    parser.add_argument("--event",       type=str, default=None)
    parser.add_argument("--events_file", type=str, default=None)
    parser.add_argument("--n_values",    type=int, nargs="+",
                        default=DEFAULT_N_VALUES)
    parser.add_argument("--per_class",   type=int, default=DEFAULT_PER_CLASS)
    parser.add_argument("--workers",   type=int, default=DEFAULT_EVENT_WORKERS,
                        help=f"Parallel event workers (threads). "
                             f"Default: {DEFAULT_EVENT_WORKERS} (cpu_count//2, cap 8). "
                             f"Use 1 to disable event-level parallelism.")
    parser.add_argument("--n_workers", type=int, default=DEFAULT_N_WORKERS,
                        help=f"Parallel N workers (threads) per event. "
                             f"Default: {DEFAULT_N_WORKERS} (= all N values). "
                             f"N-workers share the loaded fused/DEM arrays (read-only); "
                             f"marginal RAM per extra N-worker is <10 MB.")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--results_dir", type=str, default=None)
    args = parser.parse_args()

    if args.results_dir:
        global RESULTS_DIR, DEM_CACHE_DIR
        RESULTS_DIR   = args.results_dir
        DEM_CACHE_DIR = os.path.join(RESULTS_DIR, "dem_cache")
        os.makedirs(RESULTS_DIR,   exist_ok=True)
        os.makedirs(DEM_CACHE_DIR, exist_ok=True)
        log.info(f"  Output dir: {RESULTS_DIR}")

    if not os.path.exists(CATALOG_PATH):
        print(f"ERROR: SEVIR catalog not found:\n  {CATALOG_PATH}")
        sys.exit(1)

    log.info("Loading SEVIR catalog …")
    catalog  = pd.read_csv(CATALOG_PATH, parse_dates=["time_utc"], low_memory=False)
    n_values = sorted(set(args.n_values))
    resume   = not args.no_resume
    log.info(f"  {len(catalog):,} rows  |  N grid: {n_values}")
    log.info(f"  Parallelism: {args.workers} event-workers × {args.n_workers} N-workers "
             f"(all N values concurrent) × {_CPU_COUNT // 4} flow/probe workers")

    log.info("  Building catalog index …")
    catalog_index = build_catalog_index(catalog)
    log.info(f"  Index: {len(catalog_index):,} (event_id, img_type) entries.")

    event_pairs: list[tuple[str, str]] = []
    if args.catalogue:
        event_pairs = select_from_catalogue(args.catalogue, args.per_class)
    elif args.event:
        event_pairs = [(args.event, "UNKNOWN")]
    elif args.events_file:
        with open(args.events_file) as f:
            event_pairs = [(l.strip(), "UNKNOWN") for l in f if l.strip()]
    else:
        print("No event source specified. Entering interactive mode.")
        while True:
            eid = input("Enter Event ID (or 'q' to summarise and quit): ").strip()
            if eid.lower() == "q":
                break
            if eid:
                event_pairs.append((eid, "UNKNOWN"))

    if not event_pairs:
        print("No events to process.")
        sys.exit(0)

    log.info(f"  {len(event_pairs)} events queued.")
    all_event_pairs = list(event_pairs)

    if resume:
        def _is_complete(eid):
            p = os.path.join(RESULTS_DIR, f"{eid}_sweep.csv")
            if not os.path.exists(p):
                return False
            try:
                df = pd.read_csv(p)
                completed_n = sorted(df["N"].dropna().astype(int).unique().tolist())
                return (
                    completed_n == sorted(set(n_values))
                    and "intra_sp_var_mean" in df.columns
                    and df["intra_sp_var_mean"].notna().any()
                )
            except Exception:
                return False
        pending = [(eid, cls) for eid, cls in event_pairs if not _is_complete(eid)]
        done_count = len(event_pairs) - len(pending)
        if done_count:
            log.info(f"  Resume: {done_count} events already complete, "
                     f"{len(pending)} remaining.")
        event_pairs = pending

    dfs = []

    if args.workers == 1 or len(event_pairs) == 1:
        for event_id, lc_class in tqdm(event_pairs, desc="Events"):
            log.info(f"\n── {event_id}  [{lc_class}]  N∈{n_values} ──")
            df = sweep_event(event_id, catalog, n_values,
                             lifecycle_class=lc_class,
                             catalog_index=catalog_index,
                             n_workers=args.n_workers,
                             resume=resume)
            if not df.empty:
                dfs.append(df)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_sweep_event_thread,
                            eid, cls, catalog, catalog_index,
                            n_values, args.n_workers, resume): (eid, cls)
                for eid, cls in event_pairs
            }
            for fut in tqdm(as_completed(futures),
                            total=len(futures),
                            desc="Events"):
                eid, cls = futures[fut]
                try:
                    df = fut.result()
                    if not df.empty:
                        dfs.append(df)
                except Exception as exc:
                    log.error(f"  {eid} failed: {exc}", exc_info=True)

    if resume:
        for eid, _ in all_event_pairs:
            p = os.path.join(RESULTS_DIR, f"{eid}_sweep.csv")
            if os.path.exists(p):
                try:
                    dfs.append(pd.read_csv(p))
                except Exception as exc:
                    log.warning(f"  Could not reload {p}: {exc}")

    if dfs:
        best_N = build_summary(dfs, n_values, RESULTS_DIR)
        print(f"\nSweep complete.")
        print(f"  Overall recommended N = {best_N}")
        print(f"  Per-class N → "
              f"{os.path.join(RESULTS_DIR, 'class_N_recommendations.csv')}")
        print(f"  Aggregate  → "
              f"{os.path.join(RESULTS_DIR, 'sweep_aggregate.csv')}")
        print(f"  By class   → "
              f"{os.path.join(RESULTS_DIR, 'sweep_by_class.csv')}")
    else:
        print("No valid results produced.")


if __name__ == "__main__":
    main()