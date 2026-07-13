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
"""

import argparse
import logging
import os
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
BASE_PATH    = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\SEVIR\2019"
CATALOG_PATH = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\SEVIR\CATALOG.csv"
RESULTS_DIR  = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\Events\metrics\sweep"

DEM_CACHE_DIR     = os.path.join(RESULTS_DIR, "dem_cache")  # persisted across runs
DEFAULT_N_VALUES  = [250, 500, 750, 1000, 1500, 2000, 3000]
DEFAULT_PER_CLASS = 10   # events sampled per lifecycle class
# Safe defaults for low-memory machines (8 GB).  Override via CLI if you have more RAM.
# Rule of thumb: each concurrent event needs ~200–400 MB (fused array + SLIC working mem).
# With 8 GB available after OS (~6 GB usable), 1 event worker is safe.
DEFAULT_EVENT_WORKERS = 1   # increase with --workers if you have >16 GB RAM
DEFAULT_N_WORKERS     = 1   # N values run sequentially within each event

# SLIC / temporal hyperparameters held constant during the sweep
COMPACTNESS      = 10.0
ELEVATION_LAMBDA = 0.5
FUSION_CHANNELS  = ["vil", "ir107", "ir069"]

# E_super estimation: physical constants for your ODE setup
# v_max: max storm displacement per 5-min SEVIR frame in pixels (~30 m/s at 1 km/px)
# t_fc : forecast horizon in frames (12 frames * 5 min = 60 min)
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
# Two-level cache:
#   L1 — in-memory dict  : free repeat lookups within the same run
#   L2 — disk .npy files : free repeat lookups across runs / restarts
#
# Why this matters: each DEM fetch is 30–120 s of HTTP traffic to
# Planetary Computer. With 65 events this dominated total runtime (~2 h/event).
# After the first run the disk cache makes DEM lookup instantaneous.
_dem_cache: dict = {}


def _dem_cache_path(key: tuple) -> str:
    """Stable filename for a (lon0, lon1, lat0, lat1) extent key."""
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
    da = rioxarray.open_rasterio(item.assets["data"].href, lock=False).squeeze()
    da = da.rio.clip_box(minx=extent[0]-buf, miny=extent[2]-buf,
                         maxx=extent[1]+buf, maxy=extent[3]+buf)
    return da.coarsen(x=3, y=3, boundary="trim").mean()


def fetch_and_regrid_dem(extent, nx=384, ny=384):
    key = tuple(np.round(extent, 4))

    # L1: in-memory
    if key in _dem_cache:
        return _dem_cache[key]

    # L2: disk — skips the entire Planetary Computer round-trip on repeat runs
    disk_path = _dem_cache_path(key)
    if os.path.exists(disk_path):
        result = np.load(disk_path)
        _dem_cache[key] = result
        log.info(f"  DEM loaded from disk cache ({disk_path})")
        return result

    if not HAS_DEM_LIBS:
        return np.zeros((ny, nx), dtype=np.float32)

    log.info(f"  DEM not cached — fetching from Planetary Computer (one-time, ~30–120 s)…")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    items = list(catalog.search(
        collections=["cop-dem-glo-30"],
        bbox=[extent[0], extent[2], extent[1], extent[3]],
    ).item_collection())
    if not items:
        return np.zeros((ny, nx), dtype=np.float32)
    datasets = [None] * len(items)
    with ThreadPoolExecutor(max_workers=min(len(items), 8)) as pool:
        fmap = {pool.submit(_load_and_clip_tile, it, extent): i
                for i, it in enumerate(items)}
        for fut in tqdm(as_completed(fmap), total=len(items),
                        desc="DEM tiles", leave=False):
            try: datasets[fmap[fut]] = fut.result()
            except Exception as e: log.warning(f"DEM tile failed: {e}")
    datasets = [d for d in datasets if d is not None and d.size > 0]
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

    # Persist to both caches
    np.save(disk_path, result)
    _dem_cache[key] = result
    log.info(f"  DEM saved to disk cache ({disk_path})")
    return result


# =========================================================================== #
# DATA LOADING
# =========================================================================== #
def build_catalog_index(catalog: pd.DataFrame) -> dict:
    """
    Pre-index catalog as {(event_id, img_type): file_path} for O(1) lookups
    instead of filtering the full DataFrame on every channel load call.
    Call once in main() and pass the result everywhere.
    """
    idx = {}
    for _, row in catalog.iterrows():
        key = (row["id"], row["img_type"])
        if key not in idx:
            idx[key] = os.path.join(BASE_PATH, row["img_type"],
                                    os.path.basename(row["file_name"]))
    return idx


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
    catalog       : full DataFrame (for extent lookup via event_id filter)
    catalog_index : optional pre-built dict from build_catalog_index() for fast
                    channel-path resolution.  Falls back to DataFrame scan when None.
    """
    lookup = catalog_index if catalog_index is not None else catalog
    meta = catalog[catalog["id"] == event_id]
    if meta.empty:
        return None, {}, []
    row = meta.iloc[0]
    extent = [row["llcrnrlon"], row["urcrnrlon"],
              row["llcrnrlat"], row["urcrnrlat"]]
    data, tgt_shape = {}, (384, 384)
    vil = _load_single_channel(event_id, "vil", lookup, tgt_shape)
    if vil is not None:
        data["vil"] = vil
        tgt_shape = (vil.shape[1], vil.shape[2])
    for ch in ["ir107", "ir069"]:
        arr = _load_single_channel(event_id, ch, lookup, tgt_shape)
        if arr is not None:
            data[ch] = arr
    fused_stack = []
    for ch in FUSION_CHANNELS:
        if ch in data:
            arr = data[ch]
            fused_stack.append((arr - arr.min()) / (arr.max() - arr.min() + 1e-6))
    if not fused_stack:
        return None, data, extent
    return np.stack(fused_stack, axis=-1).astype(np.float32), data, extent


# =========================================================================== #
# TEMPORAL SLIC  (verbatim from Compare_superpixel_metrics_claude.py)
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

    def segment(self, fused_frame, use_flow=True):
        if fused_frame.ndim == 2:
            fused_frame = fused_frame[:, :, np.newaxis]
        H, W, C = fused_frame.shape
        curr_gray    = (fused_frame[:, :, 0] * 255).astype(np.uint8)
        dem_ch       = (self.dem_norm * self.lambda_z)[:H, :W, np.newaxis]
        feature_cube = np.concatenate([fused_frame, dem_ch], axis=-1).astype(np.float64)
        flow = np.zeros((H, W, 2), dtype=np.float32)

        if self.prev_labels is None:
            labels = slic(feature_cube, n_segments=self.n_segments,
                          compactness=self.compactness, start_label=1,
                          enforce_connectivity=True, channel_axis=-1)
            self.prev_centroids = self._centroids(labels)
        else:
            if use_flow:
                flow = cv2.calcOpticalFlowFarneback(
                    self.prev_gray, curr_gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            advected = np.zeros((H, W), dtype=np.int32)
            for lbl, (y0, x0) in self.prev_centroids.items():
                y0, x0 = int(y0), int(x0)
                if 0 <= y0 < H and 0 <= x0 < W:
                    win = 5
                    fy = np.mean(flow[max(0,y0-win):min(H,y0+win),
                                      max(0,x0-win):min(W,x0+win), 1])
                    fx = np.mean(flow[max(0,y0-win):min(H,y0+win),
                                      max(0,x0-win):min(W,x0+win), 0])
                    advected[int(np.clip(y0+fy, 0, H-1)),
                             int(np.clip(x0+fx, 0, W-1))] = lbl
            grad = np.max(
                np.stack([sobel(fused_frame[:, :, c]) for c in range(C)]
                         + [sobel(self.dem_weighted)], axis=0), axis=0)
            labels = watershed(grad, markers=advected,
                               compactness=self.compactness / 100.0)
            self.prev_centroids = self._centroids(labels)

        self.prev_labels = labels
        self.prev_gray   = curr_gray
        return labels, flow

    def _centroids(self, labels):
        unique = np.unique(labels)
        unique = unique[unique != 0]
        cms    = center_of_mass(np.ones_like(labels), labels, unique)
        return {lbl: np.array(c) for lbl, c in zip(unique, cms)}


# =========================================================================== #
# SUPERPIXEL EVALUATOR  (verbatim from Compare_superpixel_metrics_claude.py)
# =========================================================================== #
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

    def _bdv(self, lbl_t, lbl_t1, flow):
        bt  = find_boundaries(lbl_t,  mode="inner")
        bt1 = find_boundaries(lbl_t1, mode="inner")
        map_y = (np.arange(self.H, dtype=np.float32)[:, None]
                 * np.ones((1, self.W))) - flow[:, :, 1]
        map_x = (np.arange(self.W, dtype=np.float32)[None, :]
                 * np.ones((self.H, 1))) - flow[:, :, 0]
        bt_w  = cv2.remap(bt.astype(np.float32),
                          map_x.astype(np.float32), map_y.astype(np.float32),
                          interpolation=cv2.INTER_NEAREST) > 0.5
        pw, p1 = np.argwhere(bt_w), np.argwhere(bt1)
        if not len(pw) or not len(p1):
            return 0.0
        return max(directed_hausdorff(pw, p1)[0], directed_hausdorff(p1, pw)[0])

    def _ged(self, s_t, s_t1):
        born = len(s_t1 - s_t); dead = len(s_t - s_t1)
        tot  = len(s_t | s_t1)
        return (born + dead) / tot if tot > 0 else 0.0

    def _mdv(self, f_t, f_t1, lbl_t1, flow):
        map_y = (np.arange(self.H, dtype=np.float32)[:, None]
                 * np.ones((1, self.W))) - flow[:, :, 1]
        map_x = (np.arange(self.W, dtype=np.float32)[None, :]
                 * np.ones((self.H, 1))) - flow[:, :, 0]
        fw = np.empty_like(f_t)
        for c in range(self.C):
            fw[..., c] = cv2.remap(f_t[..., c],
                                   map_x.astype(np.float32),
                                   map_y.astype(np.float32),
                                   interpolation=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
        R = np.linalg.norm(f_t1 - fw, axis=-1)  # (H, W)

        # ── Vectorised weighted variance over superpixels ──────────────────────
        flat_l = lbl_t1.ravel()
        flat_R = R.ravel().astype(np.float64)
        max_l  = int(flat_l.max()) + 1
        counts = np.bincount(flat_l, minlength=max_l)
        valid  = (np.arange(max_l) > 0) & (counts > 1)
        if not valid.any():
            return 0.0
        s1     = np.bincount(flat_l, weights=flat_R,    minlength=max_l)
        s2     = np.bincount(flat_l, weights=flat_R**2, minlength=max_l)
        cnt_v  = counts[valid].astype(np.float64)
        var_v  = s2[valid] / cnt_v - (s1[valid] / cnt_v) ** 2
        # pixel-count-weighted mean variance (matches original weighting)
        return float((var_v * cnt_v).sum() / cnt_v.sum())

    def _track(self, cents_t, cents_t1_raw, flow, dist_thresh=15.0):
        if not cents_t or not cents_t1_raw:
            return cents_t1_raw
        adv = {}
        for lbl, (y, x) in cents_t.items():
            iy, ix = int(np.clip(y, 0, self.H-1)), int(np.clip(x, 0, self.W-1))
            adv[lbl] = np.array([y + flow[iy, ix, 1], x + flow[iy, ix, 0]])
        tk, tp = list(adv.keys()), np.array(list(adv.values()))
        rk, rp = list(cents_t1_raw.keys()), np.array(list(cents_t1_raw.values()))
        cost = cdist(tp, rp)
        ri, ci = linear_sum_assignment(cost)
        tracked = {}; nxt = max(tk)+1 if tk else 1; matched = set()
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
            fae_v.append(self._fae(tracked_cents[t], tc_t1, fl, lt))
            bdv_v.append(self._bdv(lt, lt1, fl))
            ged_v.append(self._ged(set(tracked_cents[t]), set(tc_t1)))
            mdv_v.append(self._mdv(ft, ft1, lt1, fl))
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

    Vectorised implementation: uses np.bincount to compute per-label sum and
    sum-of-squares in one pass per channel, then derives variance analytically.
    This avoids a Python loop over potentially thousands of superpixel labels,
    giving a ≥10× speedup at large N values.
    """
    H, W, C = fused_frame.shape
    flat_l = labels.ravel()
    flat_f = fused_frame.reshape(-1, C).astype(np.float64)
    max_l  = int(flat_l.max()) + 1

    counts = np.bincount(flat_l, minlength=max_l)
    # valid labels: label != 0, count > 1
    valid  = (np.arange(max_l) > 0) & (counts > 1)
    if not valid.any():
        return 0.0

    per_channel_var = []
    for c in range(C):
        s1 = np.bincount(flat_l, weights=flat_f[:, c],    minlength=max_l)
        s2 = np.bincount(flat_l, weights=flat_f[:, c]**2, minlength=max_l)
        cnt_v = counts[valid].astype(np.float64)
        # Var = E[x²] − E[x]²
        var_v = s2[valid] / cnt_v - (s1[valid] / cnt_v) ** 2
        per_channel_var.append(float(var_v.mean()))
    return float(np.mean(per_channel_var))


def boundary_recall_frame(labels: np.ndarray, vil_frame: np.ndarray,
                           edge_thresh: float = 0.1) -> float:
    """
    Fraction of VIL Sobel-edge pixels covered by superpixel boundaries.

    Computed per frame so the caller can average over all T frames and
    compute temporal_br_std — which quantifies boundary drift relative to
    storm edges across the full 60-minute integration window.

    Computing this at t=0 only (as the old N_hint did) misses RAPID_GROWTH
    events that start with a minimal active fraction and explode within
    30 minutes: the t=0 N is chosen for the quiescent state, then fails
    catastrophically as the storm grows into it.
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
    """
    Estimate |E_super| — pre-allocated edges in the GAT-ODE graph.

    R_max = v_max * t_fc  (max node displacement over the forecast horizon).
    Under uniform superpixel packing, each node's neighborhood radius in
    pixel space is sqrt(H*W/N), giving:
        avg_neighbors ≈ π * R_max² / (H*W/N)

    This is the memory-cost ceiling. It is not a quality metric and does not
    enter the elbow calculation. It constrains the feasible range of N given
    available GPU VRAM, and must be checked after the quality elbows are found.
    """
    R_max         = v_max * t_fc
    avg_sp_area   = (H * W) / max(actual_n, 1)
    avg_neighbors = np.pi * R_max ** 2 / avg_sp_area
    return int(actual_n * min(avg_neighbors, actual_n - 1))


def estimate_vram_mb(actual_n: int, batch_size: int = 1,
                     floats_per_edge: int = 6, **kwargs) -> float:
    """Float32 VRAM for E_super tensor: edges * batch * floats * 4 bytes / 1e6."""
    return estimate_e_super(actual_n, **kwargs) * batch_size * floats_per_edge * 4 / 1e6


# =========================================================================== #
# ELBOW DETECTION
# =========================================================================== #
def find_elbow(n_values: list, scores: list) -> int:
    """
    Maximum perpendicular distance from the chord joining the first and last
    points of the score-vs-N curve — standard knee detection.

    Both axes are normalised to [0,1] before computing distances. This ensures
    that metrics with very different absolute scales (FAE in px/frame,
    intra_sp_var in normalised intensity²) produce geometrically comparable
    elbows rather than one axis dominating.
    """
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
# SINGLE-N WORKER  (called in parallel for each N value within sweep_event)
# =========================================================================== #
def _sweep_single_n(n: int, fused: np.ndarray, dem_norm: np.ndarray,
                    vil_frames: np.ndarray) -> dict:
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
        lbl, fl = tslic.segment(fused[t], use_flow=True)
        labels_list.append(lbl)
        centroids_list.append(dict(tslic.prev_centroids))
        flow_list.append(fl)
    row["runtime_s"] = round(time.perf_counter() - t0_wall, 2)

    actual_n = int(len(np.unique(labels_list[0])) - 1)
    row["actual_N"] = actual_n

    ev      = SuperpixelEvaluator(fused, flow_list)
    metrics = ev.evaluate(labels_list, centroids_list)
    row.update(metrics)

    isv = [intra_sp_variance(labels_list[t], fused[t]) for t in range(T)]
    row["intra_sp_var_mean"] = round(float(np.mean(isv)), 6)
    row["intra_sp_var_std"]  = round(float(np.std(isv)),  6)

    br = [boundary_recall_frame(labels_list[t], vil_frames[t]) for t in range(T)]
    row["boundary_recall_mean"] = round(float(np.mean(br)), 4)
    row["temporal_br_std"]      = round(float(np.std(br)),  4)

    row["E_super_est"] = estimate_e_super(actual_n, H=H, W=W)
    row["VRAM_est_MB"] = round(estimate_vram_mb(actual_n, H=H, W=W), 1)
    row["avg_sp_area"] = round((H * W) / max(actual_n, 1), 1)
    return row


# =========================================================================== #
# SINGLE-EVENT SWEEP  (N values run in parallel via ThreadPoolExecutor)
# =========================================================================== #
def sweep_event(event_id: str, catalog: pd.DataFrame,
                n_values: list,
                lifecycle_class: str = "UNKNOWN",
                catalog_index: dict | None = None,
                n_workers: int = DEFAULT_N_WORKERS,
                resume: bool = True) -> pd.DataFrame:
    """
    Run the N sweep for one event. Returns a DataFrame with one row per N.

    Parameters
    ----------
    catalog_index : pre-built dict from build_catalog_index() for fast lookups.
    n_workers     : number of threads for parallel N evaluation (default 2).
                    Each N instantiates its own TemporalSLIC + OpenCV pipeline;
                    threads are sufficient because OpenCV/SLIC release the GIL
                    during heavy compute.
    resume        : if True, skip events whose output CSV already exists.
    """
    out_path = os.path.join(RESULTS_DIR, f"{event_id}_sweep.csv")
    if resume and os.path.exists(out_path):
        log.info(f"  {event_id}: checkpoint found — loading existing results.")
        return pd.read_csv(out_path)

    # ── Load data ─────────────────────────────────────────────────────────────
    fused, data, extent = load_fused_channels(event_id, catalog, catalog_index)
    if fused is None:
        log.warning(f"  {event_id}: no data — skipping.")
        return pd.DataFrame()
    T, H, W, C = fused.shape
    if T < 2:
        log.warning(f"  {event_id}: only {T} frame — skipping.")
        return pd.DataFrame()

    # ── DEM (fetched once, shared across all N workers) ───────────────────────
    dem_raw  = fetch_and_regrid_dem(extent, nx=W, ny=H)
    dem_norm = ((dem_raw - dem_raw.min()) /
                (dem_raw.max() - dem_raw.min() + 1e-6)).astype(np.float32)

    vil_frames = data.get("vil", fused[:, :, :, 0])

    # ── Parallel N sweep ──────────────────────────────────────────────────────
    # Each N is independent (own TemporalSLIC instance), so we fan-out with
    # ThreadPoolExecutor. Using threads (not processes) avoids pickling the
    # large fused/dem arrays and keeps memory in one process.
    rows: list[dict] = [None] * len(n_values)
    _worker = functools.partial(_sweep_single_n,
                                fused=fused, dem_norm=dem_norm,
                                vil_frames=vil_frames)

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

    # ── Attach metadata and log ───────────────────────────────────────────────
    valid_rows = []
    for n, row in zip(n_values, rows):
        if row is None:
            continue
        row["event_id"]       = event_id
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
# AGGREGATE SUMMARY  (global + per-class)
# =========================================================================== #
def build_summary(dfs: list, n_values: list, output_dir: str) -> int:
    """
    Aggregate sweep results across all events, find elbows per metric and per
    lifecycle class, write CSV outputs and a human-readable summary.
    Returns the overall recommended N (majority vote across metric elbows).
    """
    if not dfs:
        log.warning("No sweep results to summarise.")
        return n_values[0]

    all_df = pd.concat(dfs, ignore_index=True)

    # ── Global aggregate ─────────────────────────────────────────────────────
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

    # ── Per-class aggregate ───────────────────────────────────────────────────
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

    # ── Elbow detection — global ──────────────────────────────────────────────
    # ISV is the primary criterion (SLIC-native). The other four are
    # cross-validation: if they agree with ISV, the recommendation is robust.
    elbows = {
        "ISV":  find_elbow(ns, list(agg["ISV_mean"])),
        "FAE":  find_elbow(ns, list(agg["FAE_mean"])),
        "MDV":  find_elbow(ns, list(agg["MDV_mean"])),
        "LFS":  find_elbow(ns, list(agg["LFS_mean"])),
        "BR":   find_elbow(ns, [1-v for v in agg["BR_mean"]]),  # inverted: maximise
        "TBR":  find_elbow(ns, list(agg["temporal_br_std"])),   # minimise drift
    }

    # ── Elbow detection — per lifecycle class on ISV ──────────────────────────
    # Class-level elbows matter because a QUIESCENT event saturates ISV at a
    # much lower N than a RAPID_GROWTH event with a high-variance convective
    # core. These are saved as class_N_recommendations.csv.
    class_elbows: dict[str, int] = {}
    for cls, grp in class_agg.groupby("lifecycle_class"):
        grp = grp.sort_values("N")
        if len(grp) >= 3:
            class_elbows[str(cls)] = find_elbow(
                list(grp["N"].astype(int)),
                list(grp["ISV_mean"]))

    # ── Overall recommendation: majority vote across metric elbows ────────────
    vote   = Counter(elbows.values())
    best_N = vote.most_common(1)[0][0]

    # ── Report ────────────────────────────────────────────────────────────────
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

    # Save per-class recommendations for downstream use
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
    """
    Stratified sample from the Growth_Decay_Classify.py output CSV.
    Returns list of (event_id, lifecycle_class) pairs.
    """
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
    """
    Thread-safe wrapper around sweep_event.

    Why threads, not processes:
      * catalog and catalog_index are shared read-only — threads access them
        directly with zero copy overhead.
      * The bottleneck per event is I/O (HDF5 load, DEM fetch) + OpenCV/SLIC/
        NumPy compute. All of these release the GIL, so threads get real
        concurrency without the memory explosion of spawning processes.
      * ProcessPoolExecutor required pickling the full catalog per task and
        spawning N fresh Python runtimes (cartopy, cv2, skimage …), causing
        OOM kills before any event started.
    """
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
        description="SEVIR superpixel N sweep — SLIC-grounded, stratified by lifecycle class",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--catalogue",   type=str, default=None,
                        help="event_catalogue.csv from Growth_Decay_Classify.py (recommended)")
    parser.add_argument("--event",       type=str, default=None,
                        help="Single event ID")
    parser.add_argument("--events_file", type=str, default=None,
                        help="Text file with one event ID per line")
    parser.add_argument("--n_values",    type=int, nargs="+",
                        default=DEFAULT_N_VALUES,
                        help="N values to sweep (space-separated)")
    parser.add_argument("--per_class",   type=int, default=DEFAULT_PER_CLASS,
                        help="Events per lifecycle class when using --catalogue")
    # ── Parallelism controls ──────────────────────────────────────────────────
    parser.add_argument("--workers",   type=int, default=DEFAULT_EVENT_WORKERS,
                        help=f"Parallel event workers (threads). "
                             f"Default: {DEFAULT_EVENT_WORKERS} (half your CPU count). "
                             f"Use 1 to disable event-level parallelism.")
    parser.add_argument("--n_workers", type=int, default=DEFAULT_N_WORKERS,
                        help=f"Parallel N workers (threads) per event. "
                             f"Default: {DEFAULT_N_WORKERS}. "
                             f"Total concurrent SLIC instances = workers × n_workers.")
    parser.add_argument("--no_resume", action="store_true",
                        help="Ignore existing per-event CSVs and recompute everything.")
    args = parser.parse_args()

    if not os.path.exists(CATALOG_PATH):
        print(f"ERROR: SEVIR catalog not found:\n  {CATALOG_PATH}")
        sys.exit(1)

    log.info("Loading SEVIR catalog …")
    catalog  = pd.read_csv(CATALOG_PATH, parse_dates=["time_utc"], low_memory=False)
    n_values = sorted(set(args.n_values))
    resume   = not args.no_resume
    log.info(f"  {len(catalog):,} rows  |  N grid: {n_values}")
    log.info(f"  Parallelism: {args.workers} event workers × {args.n_workers} N workers")

    # ── Build catalog index once ──────────────────────────────────────────────
    log.info("  Building catalog index …")
    catalog_index = build_catalog_index(catalog)
    log.info(f"  Index: {len(catalog_index):,} (event_id, img_type) entries.")

    # ── Collect event pairs ───────────────────────────────────────────────────
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

    # Keep the full list so the resume block can load completed CSVs into the
    # summary without re-deriving it from scratch.
    all_event_pairs = list(event_pairs)

    # ── Filter already-done events when resuming ──────────────────────────────
    if resume:
        pending = [(eid, cls) for eid, cls in event_pairs
                   if not os.path.exists(
                       os.path.join(RESULTS_DIR, f"{eid}_sweep.csv"))]
        done_count = len(event_pairs) - len(pending)
        if done_count:
            log.info(f"  Resume: {done_count} events already complete, "
                     f"{len(pending)} remaining.")
        event_pairs = pending

    # ── Run sweep — parallel events via ThreadPoolExecutor ───────────────────
    dfs = []

    if args.workers == 1 or len(event_pairs) == 1:
        # Single-threaded path: simpler, easier to debug
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
        # Multi-threaded path.
        # Threads share catalog + catalog_index directly (no copying or pickling).
        # OpenCV, SLIC, and NumPy all release the GIL during heavy compute, so
        # threads achieve real concurrency here. Each thread runs one event at a
        # time; within that event, a second ThreadPoolExecutor handles N values.
        # Total concurrent SLIC instances = --workers × --n_workers.
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

    # Load completed CSVs so the summary includes events skipped by resume.
    # all_event_pairs holds the full list before it was trimmed to pending.
    if resume:
        for eid, _ in all_event_pairs:
            p = os.path.join(RESULTS_DIR, f"{eid}_sweep.csv")
            if os.path.exists(p):
                try:
                    dfs.append(pd.read_csv(p))
                except Exception as exc:
                    log.warning(f"  Could not reload {p}: {exc}")

    # ── Build aggregate summary ───────────────────────────────────────────────
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