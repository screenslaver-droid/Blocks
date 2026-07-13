"""
diagnostic_n_sweep_hpc.py  —  HPC-adapted N sweep for PARAM Rudra
==================================================================
Adapted from Diagnostic_N_sweep_optimized_v3.py for PARAM Rudra.

Key changes from v3
-------------------
HPC-1  DEM loading is disk-only.  All Planetary Computer / pystac /
       rioxarray calls are removed entirely.  The sweep reads pre-fetched
       .npy files produced by prefetch_dem.py.  If a tile is missing, the
       event runs with zero-elevation DEM and a warning is emitted.

HPC-2  Dense continuous N grid: default is N ∈ {50, 100, 150, ..., 2000}
       (40 values, step=50).  This produces smooth metric-vs-N curves
       suitable for publication-quality elbow detection and plotting,
       compared to the 6-point discrete grid used for local testing.

HPC-3  Parallelism rescaled for a 48-core PARAM Rudra CPU node:
         DEFAULT_EVENT_WORKERS = 6   (6 concurrent events)
         DEFAULT_N_WORKERS     = 8   (8 N values per event in parallel)
         Total processes:       6 × 8 = 48  ← saturates the node exactly.
       OMP/BLAS thread counts kept at 1 to avoid oversubscription
       (6 × 8 × 1 = 48 hardware threads in use).

HPC-4  Continuous elbow detection in build_summary: with 40 N values the
       standard perpendicular-distance method is well-conditioned; no
       change to find_elbow() needed.

HPC-5  build_summary now writes matplotlib plots:
         sweep_aggregate_plot.png  — all metrics vs N (global mean ± std)
         sweep_by_class_plot.png   — ISV, LFS, BR curves per lifecycle class
       Plots are saved to RESULTS_DIR; no display (Agg backend, safe on HPC).

HPC-6  SLURM Job Array support (--array_id / --n_chunks).
       The full event catalogue is split into n_chunks equal-sized slices
       using a stratified interleaved scheme: events are sorted by
       (lifecycle_class, event_id) and distributed round-robin so every
       chunk contains a proportional mix of all lifecycle classes.
       Each SLURM array task receives one chunk via --array_id.
       Summary / plotting is intentionally SKIPPED in array tasks to
       avoid a Lustre write-race — run with --aggregate_only after all
       array tasks finish to produce the final summary and plots.

       Workflow
       --------
       Step 1 — submit the array job (17 tasks × ~50 events each):
         sbatch run_sweep_array.sh   (uses --array_id $SLURM_ARRAY_TASK_ID)

       Step 2 — after all tasks complete, aggregate:
         python diagnostic_n_sweep_hpc.py \\
             --base_path ... --catalog ... --dem_cache ... \\
             --results_dir /scratch/USER/diagnostic_sweep/results \\
             --catalogue   /scratch/USER/sevir/event_catalogue.csv \\
             --aggregate_only

Usage (single-node, no array)
-------------------------------
    python diagnostic_n_sweep_hpc.py \\
        --catalogue  /scratch/USER/sevir/event_catalogue.csv \\
        --base_path  /scratch/USER/sevir/2019 \\
        --catalog    /scratch/USER/sevir/CATALOG.csv \\
        --dem_cache  /scratch/USER/diagnostic_sweep/dem_cache \\
        --results_dir /scratch/USER/diagnostic_sweep/results \\
        --per_class  3 \\
        --workers    6 \\
        --n_workers  8

    # Dense custom grid (e.g. step=25 for extra resolution):
    python diagnostic_n_sweep_hpc.py ... --n_step 25

    # Resume a partially completed run:
    python diagnostic_n_sweep_hpc.py ... --resume
"""

import argparse
import logging
import os

# ── HPC thread tuning ────────────────────────────────────────────────────────
# 6 event procs × 8 N procs × 1 OMP = 48 threads on a 48-core node.
# Keep OMP=1 to avoid oversubscription; processes provide the true parallelism.
_CPU_COUNT = os.cpu_count() or 1
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["OPENBLAS_NUM_THREADS"]   = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["OPENCV_FOR_THREADS_NUM"] = "1"
os.environ["NUMEXPR_NUM_THREADS"]    = "1"

import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, directed_hausdorff
from skimage.filters import sobel
from skimage.segmentation import find_boundaries, slic, watershed
from tqdm import tqdm

import cartopy.crs as ccrs

import matplotlib
matplotlib.use("Agg")   # non-interactive — safe on HPC login/compute nodes
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================================
# CONFIGURATION  (overridden at runtime by CLI --* arguments)
# ============================================================================
BASE_PATH    = "/scratch/ce24d903/SEVIR/2019"
CATALOG_PATH = "/scratch/ce24d903/SEVIR/CATALOG.csv"
RESULTS_DIR  = "/scratch/ce24d903/diagnostic_sweep/results"
DEM_CACHE_DIR = "/scratch/ce24d903/SEVIR/dem_cache"

# HPC-2: dense N grid — step of 50 gives 40 values from 50 to 2000.
# Produces smooth curves without discrete staircasing.
DEFAULT_N_STEP   = 50
DEFAULT_N_MIN    = 50
DEFAULT_N_MAX    = 2000
DEFAULT_N_VALUES = list(range(DEFAULT_N_MIN, DEFAULT_N_MAX + 1, DEFAULT_N_STEP))

DEFAULT_PER_CLASS = 999

# HPC-3: 6 × 8 = 48 processes — saturates a 48-core node.
DEFAULT_EVENT_WORKERS = 6
DEFAULT_N_WORKERS     = 8

# SLIC / temporal hyperparameters (unchanged from v3)
COMPACTNESS          = 10.0
ELEVATION_LAMBDA     = 0.5
FUSION_CHANNELS      = ["vil", "ir107", "ir069"]
V_MAX_PX_PER_FRAME   = 9
T_FORECAST_FRAMES    = 12

SEVIR_PROJ = ccrs.LambertConformal(
    central_longitude=-98.0, central_latitude=38.0,
    standard_parallels=(30.0, 60.0),
    globe=ccrs.Globe(semimajor_axis=6370000, semiminor_axis=6370000),
)

# ============================================================================
# HPC-1: DISK-ONLY DEM LOADING  (no network calls)
# ============================================================================

def _dem_cache_path(key: tuple) -> str:
    """Identical filename logic to prefetch_dem.py — do NOT change."""
    name = "_".join(f"{v:.4f}" for v in key).replace("-", "m")
    return os.path.join(DEM_CACHE_DIR, f"dem_{name}.npy")


def load_dem_from_cache(extent, nx=384, ny=384) -> np.ndarray:
    """
    HPC-1: Load DEM from pre-fetched disk cache only.
    No Planetary Computer calls.  If the tile is missing, returns
    a zero-elevation array and emits a warning.
    """
    key       = tuple(np.round(extent, 4))
    disk_path = _dem_cache_path(key)

    if os.path.exists(disk_path):
        try:
            result = np.load(disk_path)
            if result.shape == (ny, nx) and result.nbytes > 0:
                log.info(f"  DEM loaded from cache: {os.path.basename(disk_path)}")
                return result
            log.warning(f"  DEM cache shape mismatch at {disk_path} — using zeros.")
        except Exception as exc:
            log.warning(f"  DEM cache unreadable ({exc}) — using zeros.")
    else:
        log.warning(
            f"  DEM cache MISS for extent {[round(v,3) for v in extent]}.\n"
            f"  Expected: {disk_path}\n"
            f"  Run prefetch_dem.py before submitting this job.")

    return np.zeros((ny, nx), dtype=np.float32)


def get_sevir_grid(extent, nx=384, ny=384):
    src = ccrs.PlateCarree()
    x0, y0 = SEVIR_PROJ.transform_point(extent[0], extent[2], src)
    x1, y1 = SEVIR_PROJ.transform_point(extent[1], extent[3], src)
    xv, yv = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    grid = src.transform_points(SEVIR_PROJ, xv, yv)
    return grid[..., 0], grid[..., 1]


# ============================================================================
# DATA LOADING  (unchanged from v3)
# ============================================================================

def build_catalog_index(catalog: pd.DataFrame) -> dict:
    df = (catalog[["id", "img_type", "file_name"]]
          .drop_duplicates(subset=["id", "img_type"])
          .copy())
    df["path"] = (BASE_PATH + os.sep + df["img_type"] + os.sep
                  + df["file_name"].map(os.path.basename))
    return dict(zip(zip(df["id"], df["img_type"]), df["path"]))


def _get_channel_path(event_id, img_type, catalog_or_index):
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
    lookup = catalog_index if catalog_index is not None else catalog
    meta   = catalog[catalog["id"] == event_id]
    if meta.empty:
        return None, {}, []
    row    = meta.iloc[0]
    extent = [row["llcrnrlon"], row["urcrnrlon"],
              row["llcrnrlat"], row["urcrnrlat"]]
    tgt_shape = (384, 384)

    channels = ["vil", "ir107", "ir069"]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {ch: pool.submit(_load_single_channel, event_id, ch, lookup, tgt_shape)
                   for ch in channels}
        data = {ch: fut.result() for ch, fut in futures.items()
                if fut.result() is not None}

    if "vil" in data:
        tgt_shape = (data["vil"].shape[1], data["vil"].shape[2])
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


# ============================================================================
# TEMPORAL SLIC  (unchanged from v3)
# ============================================================================

class TemporalSLIC_DEM:
    def __init__(self, dem_norm, n_segments, compactness=10.0, lambda_z=0.5):
        self.n_segments     = n_segments
        self.compactness    = compactness
        self.lambda_z       = lambda_z
        self.dem_norm       = dem_norm
        self.dem_weighted   = dem_norm * lambda_z
        self.prev_labels    = None
        self.prev_centroids = None
        self.prev_gray      = None

    def segment(self, fused_frame, precomputed_flow=None):
        if fused_frame.ndim == 2:
            fused_frame = fused_frame[:, :, np.newaxis]
        H, W, C      = fused_frame.shape
        curr_gray    = (fused_frame[:, :, 0] * 255).astype(np.uint8)
        dem_ch       = (self.dem_norm * self.lambda_z)[:H, :W, np.newaxis]
        feature_cube = np.concatenate([fused_frame, dem_ch], axis=-1).astype(np.float32)

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

            advected = np.zeros((H, W), dtype=np.int32)
            if self.prev_centroids:
                lbls   = np.array(list(self.prev_centroids.keys()), dtype=np.int32)
                pts    = np.array(list(self.prev_centroids.values()), dtype=np.float32)
                ys     = np.clip(pts[:, 0].astype(np.int32), 0, H - 1)
                xs     = np.clip(pts[:, 1].astype(np.int32), 0, W - 1)
                win    = 5
                fy_arr = np.array([
                    flow[max(0, y-win):min(H, y+win),
                         max(0, x-win):min(W, x+win), 1].mean()
                    for y, x in zip(ys, xs)], dtype=np.float32)
                fx_arr = np.array([
                    flow[max(0, y-win):min(H, y+win),
                         max(0, x-win):min(W, x+win), 0].mean()
                    for y, x in zip(ys, xs)], dtype=np.float32)
                dst_y  = np.clip(pts[:, 0] + fy_arr, 0, H - 1).astype(np.int32)
                dst_x  = np.clip(pts[:, 1] + fx_arr, 0, W - 1).astype(np.int32)
                advected[dst_y, dst_x] = lbls

            sobel_channels = np.stack(
                [sobel(fused_frame[:, :, c]) for c in range(C)]
                + [sobel(self.dem_weighted[:H, :W])], axis=0)
            grad   = sobel_channels.max(axis=0)
            labels = watershed(grad, markers=advected,
                               compactness=self.compactness / 100.0)
            self.prev_centroids = self._centroids(labels)

        self.prev_labels = labels
        self.prev_gray   = curr_gray
        return labels, flow if self.prev_labels is not None else np.zeros(
            (H, W, 2), dtype=np.float32)

    def _centroids(self, labels):
        H, W   = labels.shape
        flat   = labels.ravel()
        max_l  = int(flat.max()) + 1
        idx    = np.arange(labels.size, dtype=np.float64)
        counts = np.bincount(flat, minlength=max_l).astype(np.float64)
        sum_y  = np.bincount(flat, weights=idx // W, minlength=max_l)
        sum_x  = np.bincount(flat, weights=idx  % W, minlength=max_l)
        valid  = np.nonzero(counts[1:])[0] + 1
        cy     = sum_y[valid] / counts[valid]
        cx     = sum_x[valid] / counts[valid]
        return {int(lbl): np.array([y, x], dtype=np.float32)
                for lbl, y, x in zip(valid, cy, cx)}


# ============================================================================
# SUPERPIXEL EVALUATOR  (unchanged from v3)
# ============================================================================

def _flow_maps(H, W, flow):
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
        ct   = np.array([cents_t[l]  for l in common], dtype=np.float32)
        ct1  = np.array([cents_t1[l] for l in common], dtype=np.float32)
        vel  = ct1[:, ::-1] - ct[:, ::-1]
        flat = labels_t.ravel()
        mx   = int(max(common)) + 1
        su   = np.bincount(flat, weights=flow[..., 0].ravel(), minlength=mx)
        sv   = np.bincount(flat, weights=flow[..., 1].ravel(), minlength=mx)
        cnt  = np.bincount(flat, minlength=mx).clip(min=1)
        mf   = np.stack([su[common]/cnt[common], sv[common]/cnt[common]], axis=1)
        return float(np.linalg.norm(vel - mf, axis=1).mean())

    def _bdv(self, lbl_t, lbl_t1, map_x, map_y):
        bt   = find_boundaries(lbl_t,  mode="inner")
        bt1  = find_boundaries(lbl_t1, mode="inner")
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
        fw = np.empty_like(f_t)
        for c in range(self.C):
            fw[..., c] = cv2.remap(f_t[..., c], map_x, map_y,
                                   interpolation=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
        R      = np.linalg.norm(f_t1 - fw, axis=-1)
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
        if not cents_t or not cents_t1_raw:
            return cents_t1_raw
        tk  = list(cents_t.keys())
        tp  = np.array(list(cents_t.values()), dtype=np.float32)
        iy  = np.clip(tp[:, 0].astype(np.int32), 0, self.H - 1)
        ix  = np.clip(tp[:, 1].astype(np.int32), 0, self.W - 1)
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
            map_x, map_y = _flow_maps(self.H, self.W, fl)
            fae_v.append(self._fae(tracked_cents[t], tc_t1, fl, lt))
            bdv_v.append(self._bdv(lt, lt1, map_x, map_y))
            ged_v.append(self._ged(set(tracked_cents[t]), set(tc_t1)))
            mdv_v.append(self._mdv(ft, ft1, lt1, map_x, map_y))

            for lbl, cpos in tc_t1.items():
                raw_m = [k for k, v in centroids_list[t+1].items()
                         if np.allclose(v, cpos)]
                lt1_arr = labels_list[t+1]
                if raw_m:
                    mask = lt1_arr == raw_m[0]
                    if mask.any():
                        track_feats.setdefault(lbl, []).append(
                            ft1[mask].mean(axis=0))

        lfs_scores = []
        for hist in track_feats.values():
            if len(hist) > 1:
                arr = np.array(hist)
                lfs_scores.append(np.linalg.norm(arr[1:] - arr[:-1], axis=1).mean())

        return {
            "FAE": float(np.mean(fae_v))    if fae_v    else 0.0,
            "BDV": float(np.mean(bdv_v))    if bdv_v    else 0.0,
            "GED": float(np.mean(ged_v))    if ged_v    else 0.0,
            "MDV": float(np.mean(mdv_v))    if mdv_v    else 0.0,
            "LFS": float(np.mean(lfs_scores)) if lfs_scores else 0.0,
        }


# ============================================================================
# SLIC STRUCTURAL PROBES  (unchanged from v3)
# ============================================================================

def intra_sp_variance(labels, fused_frame):
    H, W, C = fused_frame.shape
    flat_l  = labels.ravel()
    flat_f  = fused_frame.reshape(-1, C).astype(np.float64)
    max_l   = int(flat_l.max()) + 1
    counts  = np.bincount(flat_l, minlength=max_l)
    valid   = (np.arange(max_l) > 0) & (counts > 1)
    if not valid.any():
        return 0.0
    per_channel_var = []
    for c in range(C):
        s1    = np.bincount(flat_l, weights=flat_f[:, c],    minlength=max_l)
        s2    = np.bincount(flat_l, weights=flat_f[:, c]**2, minlength=max_l)
        cnt_v = counts[valid].astype(np.float64)
        var_v = s2[valid] / cnt_v - (s1[valid] / cnt_v) ** 2
        per_channel_var.append(float(var_v.mean()))
    return float(np.mean(per_channel_var))


def boundary_recall_frame(labels, vil_frame, edge_thresh=0.1):
    vil_norm   = vil_frame / (vil_frame.max() + 1e-6)
    true_edges = sobel(vil_norm) > edge_thresh
    sp_bounds  = find_boundaries(labels, mode="inner")
    if true_edges.sum() == 0:
        return 1.0
    return float((true_edges & sp_bounds).sum() / true_edges.sum())


def estimate_e_super(actual_n, H=384, W=384,
                     v_max=V_MAX_PX_PER_FRAME, t_fc=T_FORECAST_FRAMES):
    R_max         = v_max * t_fc
    avg_sp_area   = (H * W) / max(actual_n, 1)
    avg_neighbors = np.pi * R_max ** 2 / avg_sp_area
    return int(actual_n * min(avg_neighbors, actual_n - 1))


def estimate_vram_mb(actual_n, batch_size=1, floats_per_edge=6, **kwargs):
    return estimate_e_super(actual_n, **kwargs) * batch_size * floats_per_edge * 4 / 1e6


# ============================================================================
# ELBOW DETECTION  (unchanged from v3; works well with 40 N values)
# ============================================================================

def find_elbow(n_values, scores):
    if len(n_values) < 3:
        return n_values[0]
    pts   = np.array(list(zip(n_values, scores)), dtype=float)
    pts_n = (pts - pts.min(axis=0)) / (pts.max(axis=0) - pts.min(axis=0) + 1e-12)
    line  = pts_n[-1] - pts_n[0]
    line /= np.linalg.norm(line) + 1e-12
    dists = [np.linalg.norm((p - pts_n[0]) - np.dot(p - pts_n[0], line) * line)
             for p in pts_n]
    return int(n_values[int(np.argmax(dists))])


# ============================================================================
# SINGLE-N WORKER  (unchanged from v3)
# ============================================================================

def _sweep_single_n(n, fused, dem_norm, vil_frames, precomputed_flows):
    T, H, W, C = fused.shape
    row = {"N": n}
    tslic = TemporalSLIC_DEM(dem_norm, n_segments=n,
                             compactness=COMPACTNESS,
                             lambda_z=ELEVATION_LAMBDA)
    labels_list, centroids_list, flow_list = [], [], []
    t0 = time.perf_counter()
    for t in range(T):
        lbl, fl = tslic.segment(fused[t], precomputed_flow=precomputed_flows[t])
        labels_list.append(lbl)
        centroids_list.append(dict(tslic.prev_centroids))
        flow_list.append(fl)
    row["runtime_s"] = round(time.perf_counter() - t0, 2)

    actual_n     = int(len(np.unique(labels_list[0])) - 1)
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


# ============================================================================
# PROCESS-POOL WORKERS  (unchanged from v3)
# ============================================================================

_WORKER_SWEEP_DATA: dict = {}


def _init_n_worker(fused, dem_norm, vil_frames, precomputed_flows):
    global _WORKER_SWEEP_DATA
    _WORKER_SWEEP_DATA = {
        "fused": fused, "dem_norm": dem_norm,
        "vil_frames": vil_frames, "precomputed_flows": precomputed_flows,
    }


def _sweep_single_n_worker(n):
    d = _WORKER_SWEEP_DATA
    return _sweep_single_n(n, d["fused"], d["dem_norm"],
                           d["vil_frames"], d["precomputed_flows"])


def _init_event_worker(results_dir, dem_cache_dir):
    global RESULTS_DIR, DEM_CACHE_DIR
    RESULTS_DIR   = results_dir
    DEM_CACHE_DIR = dem_cache_dir
    os.makedirs(RESULTS_DIR,   exist_ok=True)
    os.makedirs(DEM_CACHE_DIR, exist_ok=True)


# ============================================================================
# SINGLE-EVENT SWEEP  (HPC-1: uses load_dem_from_cache instead of fetch)
# ============================================================================

def sweep_event(event_id, catalog, n_values,
                lifecycle_class="UNKNOWN",
                catalog_index=None,
                n_workers=DEFAULT_N_WORKERS,
                resume=True):
    out_path = os.path.join(RESULTS_DIR, f"{event_id}_sweep.csv")
    if resume and os.path.exists(out_path):
        try:
            existing    = pd.read_csv(out_path)
            expected_n  = sorted(set(n_values))
            completed_n = sorted(existing["N"].dropna().astype(int).unique().tolist())
            valid_rows  = existing["intra_sp_var_mean"].notna().sum() \
                          if "intra_sp_var_mean" in existing.columns else 0
            if completed_n == expected_n and valid_rows == len(expected_n):
                log.info(f"  {event_id}: checkpoint complete — skipping.")
                return existing
            else:
                missing = sorted(set(expected_n) - set(completed_n))
                log.warning(f"  {event_id}: incomplete checkpoint "
                            f"({len(completed_n)}/{len(expected_n)}, "
                            f"missing={missing[:5]}{'...' if len(missing)>5 else ''}) "
                            f"— recomputing.")
        except Exception as exc:
            log.warning(f"  {event_id}: checkpoint unreadable ({exc}) — recomputing.")

    fused, data, extent = load_fused_channels(event_id, catalog, catalog_index)
    if fused is None:
        log.warning(f"  {event_id}: no data — skipping.")
        return pd.DataFrame()
    T, H, W, C = fused.shape
    if T < 2:
        log.warning(f"  {event_id}: only {T} frame — skipping.")
        return pd.DataFrame()

    # HPC-1: disk-only DEM load
    dem_raw  = load_dem_from_cache(extent, nx=W, ny=H)
    dem_norm = ((dem_raw - dem_raw.min()) /
                (dem_raw.max() - dem_raw.min() + 1e-6)).astype(np.float32)

    vil_frames = data.get("vil", fused[:, :, :, 0])

    log.info(f"  {event_id}: Pre-computing optical flow ({T-1} pairs)...")
    precomputed_flows = [np.zeros((H, W, 2), dtype=np.float32)]
    for t in range(1, T):
        prev_gray = (fused[t-1, :, :, 0] * 255).astype(np.uint8)
        curr_gray = (fused[t,   :, :, 0] * 255).astype(np.uint8)
        precomputed_flows.append(
            cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0))

    rows = [None] * len(n_values)
    actual_workers = min(n_workers, len(n_values))
    with ProcessPoolExecutor(
            max_workers=actual_workers,
            initializer=_init_n_worker,
            initargs=(fused, dem_norm, vil_frames, precomputed_flows)) as pool:
        futures = {pool.submit(_sweep_single_n_worker, n): idx
                   for idx, n in enumerate(n_values)}
        for fut in tqdm(as_completed(futures), total=len(n_values),
                        desc=f"{event_id} N-sweep", leave=False):
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
                f"ISV={row['intra_sp_var_mean']:.5f} | "
                f"BR={row['boundary_recall_mean']:.3f} | "
                f"LFS={row.get('LFS', 0):.4f} | "
                f"E_super={row.get('E_super_est', 0):,d} | {row.get('runtime_s', 0):.1f}s"
            )
        valid_rows.append(row)

    df = pd.DataFrame(valid_rows)
    df.to_csv(out_path, index=False)
    return df


# ============================================================================
# EVENT SELECTION  (unchanged from v3)
# ============================================================================

def select_from_catalogue(cat_path, per_class):
    df = pd.read_csv(cat_path)
    if "lifecycle_class" not in df.columns or "id" not in df.columns:
        log.error("Catalogue missing 'id' or 'lifecycle_class'.")
        return []
    out = []
    for cls, grp in df.groupby("lifecycle_class"):
        sample = grp.sample(min(per_class, len(grp)), random_state=42)
        out.extend([(row["id"], str(cls)) for _, row in sample.iterrows()])
    log.info(f"  Stratified sample: {len(out)} events from "
             f"{df['lifecycle_class'].nunique()} classes.")
    return out


def _sweep_event_thread(event_id, lc_class, catalog, catalog_index,
                        n_values, n_workers, resume):
    return sweep_event(event_id, catalog, n_values,
                       lifecycle_class=lc_class,
                       catalog_index=catalog_index,
                       n_workers=n_workers,
                       resume=resume)


# ============================================================================
# HPC-6: SLURM JOB ARRAY — CATALOGUE CHUNKING
# ============================================================================

def slice_catalogue_for_array(cat_path: str,
                               array_id: int,
                               n_chunks: int) -> list:
    """
    Split the full event catalogue into n_chunks equal-sized slices and
    return the slice for array_id.

    Stratified interleaved scheme
    ------------------------------
    Events are sorted by (lifecycle_class, id) for determinism, then
    distributed round-robin: event at sorted position i → chunk (i % n_chunks).

    This guarantees every chunk gets a proportional mix of all lifecycle
    classes regardless of n_chunks or total event count.

    Example: 840 events, 17 chunks → each chunk gets ~49-50 events,
    ~7 from each of the 7 lifecycle classes.

    The assignment is purely deterministic — all 17 tasks independently
    reconstruct the same assignment from the same CSV with no coordination.

    Parameters
    ----------
    cat_path  : path to event_catalogue.csv (must have 'id', 'lifecycle_class')
    array_id  : SLURM_ARRAY_TASK_ID — which slice this task should process
    n_chunks  : total number of array tasks (must match #SBATCH --array=0-N)

    Returns
    -------
    List of (event_id, lifecycle_class) tuples for this chunk only.
    """
    df = pd.read_csv(cat_path)
    if "lifecycle_class" not in df.columns or "id" not in df.columns:
        log.error("Catalogue missing 'id' or 'lifecycle_class'.")
        sys.exit(1)

    if array_id < 0 or array_id >= n_chunks:
        log.error(f"array_id={array_id} out of range [0, {n_chunks - 1}].")
        sys.exit(1)

    # Deterministic sort → round-robin assignment
    df_sorted = (df[["id", "lifecycle_class"]]
                 .drop_duplicates(subset="id")
                 .sort_values(["lifecycle_class", "id"])
                 .reset_index(drop=True))

    df_sorted["_chunk"] = df_sorted.index % n_chunks
    chunk_df = df_sorted[df_sorted["_chunk"] == array_id].copy()

    pairs = [(row["id"], str(row["lifecycle_class"]))
             for _, row in chunk_df.iterrows()]

    log.info(
        f"  [Array {array_id}/{n_chunks - 1}]  "
        f"{len(pairs)} events  |  "
        f"{chunk_df['lifecycle_class'].nunique()} classes"
    )
    for cls, cnt in sorted(chunk_df.groupby("lifecycle_class")["id"].count().items()):
        log.info(f"    {cls:<24}: {cnt} events")

    return pairs


# ============================================================================
# HPC-5: CONTINUOUS PLOTTING
# ============================================================================

# Colour palette for lifecycle classes
_CLASS_COLORS = {
    "RAPID_GROWTH": "#e74c3c",
    "GROWTH_DECAY": "#e67e22",
    "EPISODIC":     "#f1c40f",
    "PLATEAU":      "#2ecc71",
    "STEADY":       "#3498db",
    "RAPID_DECAY":  "#9b59b6",
    "QUIESCENT":    "#95a5a6",
}
_DEFAULT_COLOR = "#34495e"


def _plot_aggregate(agg: pd.DataFrame, best_N: int, output_dir: str):
    """
    Global aggregate plot: ISV, LFS, FAE, MDV, BR, E_super vs N.
    Shaded bands show ±1 std where available.
    """
    ns = agg["N"].astype(int).values

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Diagnostic N-Sweep — Global Aggregate (all events)",
                 fontsize=14, fontweight="bold")

    metrics = [
        ("ISV_mean",  "ISV_std",       "Intra-SP Variance (ISV)",
         "Lower = better segmentation",  "dodgerblue",   True),
        ("LFS_mean",  None,            "Lagrangian Feature Smoothness (LFS)",
         "Lower = lower ODE stiffness",  "tomato",       True),
        ("FAE_mean",  None,            "Flow Adherence Error (FAE)",
         "Lower = better flow tracking", "darkorange",   True),
        ("MDV_mean",  None,            "Material Derivative Variance (MDV)",
         "Lower = better homogeneity",   "mediumseagreen", True),
        ("BR_mean",   "BR_std",        "Boundary Recall (BR)",
         "Higher = better edge coverage","mediumpurple", False),
        ("E_super_mean", None,         "Edge Count Estimate E_super",
         "VRAM cost proxy (lower N = cheaper)", "dimgray", True),
    ]

    for ax, (col, std_col, title, subtitle, color, lower_better) in \
            zip(axes.flat, metrics):
        y = agg[col].values
        ax.plot(ns, y, color=color, linewidth=2, label=col)
        if std_col and std_col in agg.columns:
            s = agg[std_col].fillna(0).values
            ax.fill_between(ns, y - s, y + s, color=color, alpha=0.15,
                            label="±1 std")
        ax.axvline(best_N, color="black", linestyle="--", linewidth=1.2,
                   label=f"Rec. N={best_N}")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("N (superpixels)")
        ax.set_ylabel(col)
        ax.text(0.97, 0.97, subtitle, transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color="gray")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "sweep_aggregate_plot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Aggregate plot saved → {path}")


def _plot_by_class(class_agg: pd.DataFrame, class_elbows: dict,
                   output_dir: str):
    """
    Per-class continuous curves: ISV, LFS, BR vs N — one line per class.
    Elbow markers are annotated on each curve.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Diagnostic N-Sweep — Per-Lifecycle-Class",
                 fontsize=14, fontweight="bold")

    plot_metrics = [
        ("ISV_mean",  "Intra-SP Variance (ISV)",
         "Lower = better SLIC quality"),
        ("LFS_mean",  "Lagrangian Feature Smoothness (LFS)" if "LFS_mean" in class_agg.columns
                      else "FAE_mean",
         "Lower = lower ODE stiffness"),
        ("BR_mean",   "Boundary Recall (BR)",
         "Higher = better edge coverage"),
    ]

    # Normalise LFS column name if missing
    for i, (col, title, subtitle) in enumerate(plot_metrics):
        if col not in class_agg.columns:
            # Fallback column map
            fb = {"LFS_mean": "FAE_mean"}
            col = fb.get(col, col)
            plot_metrics[i] = (col, title, subtitle)

    classes = sorted(class_agg["lifecycle_class"].unique())

    for ax, (col, title, subtitle) in zip(axes, plot_metrics):
        if col not in class_agg.columns:
            ax.set_visible(False)
            continue
        for cls in classes:
            grp   = class_agg[class_agg["lifecycle_class"] == cls].sort_values("N")
            ns    = grp["N"].astype(int).values
            ys    = grp[col].values
            color = _CLASS_COLORS.get(cls, _DEFAULT_COLOR)
            ax.plot(ns, ys, color=color, linewidth=2, label=cls)
            # Elbow marker
            elbow_n = class_elbows.get(str(cls))
            if elbow_n and elbow_n in grp["N"].values:
                ev = grp.loc[grp["N"] == elbow_n, col].values[0]
                ax.scatter([elbow_n], [ev], color=color, s=80, zorder=5,
                           marker="D", edgecolors="black", linewidths=0.8)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("N (superpixels)")
        ax.set_ylabel(col)
        ax.text(0.97, 0.97, subtitle, transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="gray")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "sweep_by_class_plot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Per-class plot saved → {path}")


def _plot_joint_score(agg: pd.DataFrame, alpha: float, best_N: int,
                      output_dir: str):
    """
    Joint score J(N) = quality(N) − alpha × stiffness(N) plot.
    Matches the DRIFT summary report interpretation.
    """
    ns    = agg["N"].astype(int).values
    isv   = agg["ISV_mean"].values
    lfs_col = "LFS_mean" if "LFS_mean" in agg.columns else "FAE_mean"
    lfs   = agg[lfs_col].values

    isv0, isv_last = isv[0], isv[-1]
    lfs0, lfs_last = lfs[0], lfs[-1]

    quality   = (isv0 - isv)   / (isv0 - isv_last + 1e-12)
    stiffness = (lfs  - lfs0)  / (lfs_last - lfs0 + 1e-12)
    J         = quality - alpha * stiffness

    best_J_N = int(ns[np.argmax(J)])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Joint Score J(N) = quality − {alpha}·stiffness",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(ns, quality,   color="dodgerblue", linewidth=2, label="quality(N)")
    ax.plot(ns, stiffness, color="tomato",     linewidth=2, label="stiffness(N)")
    ax.plot(ns, J,         color="black",      linewidth=2.5, linestyle="--",
            label=f"J(N)  α={alpha}")
    ax.axvline(best_J_N, color="green", linestyle=":", linewidth=1.5,
               label=f"argmax J = {best_J_N}")
    ax.axvline(best_N,   color="black", linestyle="--", linewidth=1.2,
               label=f"ISV-vote N = {best_N}")
    ax.set_xlabel("N")
    ax.set_ylabel("Score")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Normalised components")

    ax2 = axes[1]
    ax2.plot(ns, J, color="black", linewidth=2.5)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.axvline(best_J_N, color="green", linestyle=":", linewidth=1.5,
                label=f"argmax J = {best_J_N}")
    ax2.scatter([best_J_N], [float(J[np.argmax(J)])],
                color="green", s=120, zorder=5, marker="*")
    ax2.set_xlabel("N")
    ax2.set_ylabel("J(N)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Joint score J(N)")

    plt.tight_layout()
    path = os.path.join(output_dir, "sweep_joint_score_plot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Joint-score plot saved → {path}")
    return best_J_N


# ============================================================================
# AGGREGATE SUMMARY  (HPC-5: adds three continuous plots)
# ============================================================================

def build_summary(dfs, n_values, output_dir, alpha=1.5):
    if not dfs:
        log.warning("No sweep results to summarise.")
        return n_values[0]

    all_df = pd.concat(dfs, ignore_index=True)

    # ── aggregate over all events ────────────────────────────────────────────
    agg_cols = {
        "FAE_mean":       ("FAE",                 "mean"),
        "FAE_std":        ("FAE",                 "std"),
        "BDV_mean":       ("BDV",                 "mean"),
        "MDV_mean":       ("MDV",                 "mean"),
        "LFS_mean":       ("LFS",                 "mean"),
        "ISV_mean":       ("intra_sp_var_mean",   "mean"),
        "ISV_std":        ("intra_sp_var_mean",   "std"),
        "BR_mean":        ("boundary_recall_mean","mean"),
        "BR_std":         ("boundary_recall_mean","std"),
        "temporal_br_std":("temporal_br_std",     "mean"),
        "E_super_mean":   ("E_super_est",         "mean"),
        "VRAM_mean_MB":   ("VRAM_est_MB",         "mean"),
        "runtime_mean":   ("runtime_s",           "mean"),
    }
    agg = all_df.groupby("N").agg(**{k: v for k, v in agg_cols.items()}).reset_index()
    agg.to_csv(os.path.join(output_dir, "sweep_aggregate.csv"), index=False)

    class_agg = all_df.groupby(["lifecycle_class", "N"]).agg(
        FAE_mean         =("FAE",                 "mean"),
        MDV_mean         =("MDV",                 "mean"),
        LFS_mean         =("LFS",                 "mean"),
        ISV_mean         =("intra_sp_var_mean",   "mean"),
        BR_mean          =("boundary_recall_mean","mean"),
        temporal_br_std  =("temporal_br_std",     "mean"),
        E_super_mean     =("E_super_est",         "mean"),
        n_events         =("event_id",            "count"),
    ).reset_index()
    class_agg.to_csv(os.path.join(output_dir, "sweep_by_class.csv"), index=False)

    ns = list(agg["N"].astype(int))

    elbows = {
        "ISV": find_elbow(ns, list(agg["ISV_mean"])),
        "FAE": find_elbow(ns, list(agg["FAE_mean"])),
        "MDV": find_elbow(ns, list(agg["MDV_mean"])),
        "LFS": find_elbow(ns, list(agg["LFS_mean"])),
        "BR":  find_elbow(ns, [1 - v for v in agg["BR_mean"]]),
        "TBR": find_elbow(ns, list(agg["temporal_br_std"])),
    }

    class_elbows: dict = {}
    for cls, grp in class_agg.groupby("lifecycle_class"):
        grp = grp.sort_values("N")
        if len(grp) >= 3:
            class_elbows[str(cls)] = find_elbow(
                list(grp["N"].astype(int)), list(grp["ISV_mean"]))

    vote   = Counter(elbows.values())
    best_N = vote.most_common(1)[0][0]

    # ── HPC-5: plots ─────────────────────────────────────────────────────────
    _plot_aggregate(agg, best_N, output_dir)
    _plot_by_class(class_agg, class_elbows, output_dir)
    best_J_N = _plot_joint_score(agg, alpha, best_N, output_dir)

    # ── text summary ─────────────────────────────────────────────────────────
    sep = "=" * 80
    lines = [
        sep,
        f"  N sweep summary  |  {all_df['event_id'].nunique()} events  "
        f"|  {all_df['lifecycle_class'].nunique()} classes  "
        f"|  N grid: {n_values[0]}…{n_values[-1]} (step {n_values[1]-n_values[0]})",
        sep,
        f"  {'N':>6}  {'ISV':>9}  {'FAE':>7}  {'LFS':>8}  {'BR':>6}  "
        f"{'E_super':>11}  {'VRAM(MB)':>9}  {'t(s)':>5}",
        "  " + "-" * 68,
    ]
    for _, r in agg.iterrows():
        n = int(r["N"])
        tag = " ◄ REC" if n == best_N else ("  ◄ J" if n == best_J_N else "")
        lines.append(
            f"  {n:>6}  {r['ISV_mean']:>9.5f}  {r['FAE_mean']:>7.4f}  "
            f"{r['LFS_mean']:>8.5f}  {r['BR_mean']:>6.3f}  "
            f"{int(r['E_super_mean']):>11,d}  "
            f"{r['VRAM_mean_MB']:>9.0f}  {r['runtime_mean']:>5.1f}{tag}"
        )
    lines += [
        "  " + "-" * 68, "",
        f"  Global metric elbows (ISV=primary, others=cross-validation):",
    ]
    for m, ne in elbows.items():
        lines.append(f"    {m:<5} → N = {ne}")
    lines += [
        "",
        f"  OVERALL RECOMMENDED N (majority vote) = {best_N}",
        f"  Joint-score argmax   (α={alpha})       = {best_J_N}",
        "",
        "  Per-class ISV elbow:",
    ]
    for cls, ne in sorted(class_elbows.items()):
        lines.append(f"    {cls:<24} → N = {ne}")
    lines += ["", sep]

    report = "\n".join(lines)
    print("\n" + report)
    Path(os.path.join(output_dir, "sweep_summary.txt")).write_text(report)

    pd.DataFrame([{"lifecycle_class": c, "recommended_N": n}
                  for c, n in class_elbows.items()]).to_csv(
        os.path.join(output_dir, "class_N_recommendations.csv"), index=False)

    return best_N


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HPC N-sweep for SEVIR superpixels (PARAM Rudra)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # ── paths ────────────────────────────────────────────────────────────────
    parser.add_argument("--base_path",   type=str, default=None,
                        help="Path to SEVIR data directory (2019/)")
    parser.add_argument("--catalog",     type=str, default=None,
                        help="Path to SEVIR CATALOG.csv")
    parser.add_argument("--dem_cache",   type=str, default=None,
                        help="Path to pre-fetched DEM cache directory")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Output directory for results and plots")

    # ── event selection ──────────────────────────────────────────────────────
    parser.add_argument("--catalogue",   type=str, default=None,
                        help="Event classification catalogue CSV")
    parser.add_argument("--event",       type=str, default=None)
    parser.add_argument("--events_file", type=str, default=None)
    parser.add_argument("--per_class",   type=int, default=DEFAULT_PER_CLASS,
                        help="Events per class for non-array mode (default: 3). "
                             "Ignored when --array_id is used.")

    # ── HPC-6: job array ─────────────────────────────────────────────────────
    parser.add_argument(
        "--array_id", type=int, default=None,
        help="SLURM_ARRAY_TASK_ID (0-based).  When set, the catalogue is split "
             "into --n_chunks equal slices and this task processes slice array_id. "
             "Requires --catalogue.  Overrides --per_class / --event / --events_file.")
    parser.add_argument(
        "--n_chunks", type=int, default=17,
        help="Total number of array tasks; must match #SBATCH --array=0-N "
             "(default: 17).  Ignored when --array_id is not set.")
    parser.add_argument(
        "--aggregate_only", action="store_true",
        help="Skip event processing.  Load all existing per-event CSVs from "
             "--results_dir and rebuild the summary + plots.  Run this once "
             "after all array tasks have completed.")

    # ── N grid (HPC-2) ───────────────────────────────────────────────────────
    parser.add_argument("--n_min",    type=int, default=DEFAULT_N_MIN)
    parser.add_argument("--n_max",    type=int, default=DEFAULT_N_MAX)
    parser.add_argument("--n_step",   type=int, default=DEFAULT_N_STEP,
                        help="Step size for continuous N grid (default: 50)")
    parser.add_argument("--n_values", type=int, nargs="+", default=None,
                        help="Explicit N list (overrides --n_min/max/step)")

    # ── parallelism (HPC-3) ──────────────────────────────────────────────────
    parser.add_argument("--workers",   type=int, default=DEFAULT_EVENT_WORKERS,
                        help=f"Concurrent event processes (default {DEFAULT_EVENT_WORKERS})")
    parser.add_argument("--n_workers", type=int, default=DEFAULT_N_WORKERS,
                        help=f"Concurrent N processes per event (default {DEFAULT_N_WORKERS})")

    # ── misc ─────────────────────────────────────────────────────────────────
    parser.add_argument("--resume",    action="store_true", default=True)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--alpha",     type=float, default=1.5,
                        help="Stiffness weight in joint score J(N) (default: 1.5)")

    args = parser.parse_args()

    # ── path overrides ───────────────────────────────────────────────────────
    global BASE_PATH, CATALOG_PATH, DEM_CACHE_DIR, RESULTS_DIR
    if args.base_path:   BASE_PATH     = args.base_path
    if args.catalog:     CATALOG_PATH  = args.catalog
    if args.dem_cache:   DEM_CACHE_DIR = args.dem_cache
    if args.results_dir: RESULTS_DIR   = args.results_dir
    os.makedirs(RESULTS_DIR,   exist_ok=True)
    os.makedirs(DEM_CACHE_DIR, exist_ok=True)

    # ── N grid ───────────────────────────────────────────────────────────────
    if args.n_values:
        n_values = sorted(set(args.n_values))
    else:
        n_values = list(range(args.n_min, args.n_max + 1, args.n_step))
    resume = not args.no_resume

    # ── banner ───────────────────────────────────────────────────────────────
    array_tag = (f"  array task : {args.array_id}/{args.n_chunks - 1}"
                 if args.array_id is not None else "  mode       : single-job")
    log.info("=" * 70)
    log.info(f"PARAM Rudra HPC N-sweep  |  {len(n_values)} N values "
             f"({n_values[0]}…{n_values[-1]}, step={n_values[1]-n_values[0]})")
    log.info(array_tag)
    log.info(f"  results_dir : {RESULTS_DIR}")
    log.info(f"  dem_cache   : {DEM_CACHE_DIR}")
    log.info(f"  parallelism : {args.workers} events × {args.n_workers} N-workers "
             f"= {args.workers * args.n_workers} processes")
    log.info("=" * 70)

    # ── aggregate-only mode ──────────────────────────────────────────────────
    # Runs after all array tasks finish. Scans results_dir for per-event CSVs
    # and rebuilds the summary + plots without re-running any sweeps.
    if args.aggregate_only:
        log.info("--aggregate_only: scanning results_dir for per-event CSVs ...")
        all_dfs = []
        for f in sorted(Path(RESULTS_DIR).glob("*_sweep.csv")):
            try:
                all_dfs.append(pd.read_csv(f))
            except Exception as exc:
                log.warning(f"  Could not read {f.name}: {exc}")
        if not all_dfs:
            log.error("No per-event CSVs found. Ensure array tasks have completed.")
            sys.exit(1)
        log.info(f"  Loaded {len(all_dfs)} per-event CSVs.")
        best_N = build_summary(all_dfs, n_values, RESULTS_DIR, alpha=args.alpha)
        print(f"\nAggregation complete. Recommended N = {best_N}")
        print(f"  Plots → {RESULTS_DIR}/sweep_aggregate_plot.png")
        print(f"          {RESULTS_DIR}/sweep_by_class_plot.png")
        print(f"          {RESULTS_DIR}/sweep_joint_score_plot.png")
        return

    # ── SEVIR catalog (needed for all non-aggregate modes) ───────────────────
    if not os.path.exists(CATALOG_PATH):
        print(f"ERROR: SEVIR catalog not found: {CATALOG_PATH}")
        sys.exit(1)
    catalog       = pd.read_csv(CATALOG_PATH, parse_dates=["time_utc"],
                                low_memory=False)
    catalog_index = build_catalog_index(catalog)
    log.info(f"  SEVIR catalog: {len(catalog):,} rows")

    # ── event selection ──────────────────────────────────────────────────────
    # HPC-6: array mode uses slice_catalogue_for_array and ignores --per_class.
    # Single-job mode preserves the original per_class / event / events_file logic.
    if args.array_id is not None:
        # ── ARRAY MODE ───────────────────────────────────────────────────────
        if not args.catalogue:
            print("ERROR: --array_id requires --catalogue.")
            sys.exit(1)
        event_pairs = slice_catalogue_for_array(
            args.catalogue, args.array_id, args.n_chunks)
    else:
        # ── SINGLE-JOB MODE (unchanged from previous version) ────────────────
        event_pairs = []
        if args.catalogue:
            event_pairs = select_from_catalogue(args.catalogue, args.per_class)
        elif args.event:
            event_pairs = [(args.event, "UNKNOWN")]
        elif args.events_file:
            with open(args.events_file) as f:
                event_pairs = [(l.strip(), "UNKNOWN") for l in f if l.strip()]
        else:
            print("No event source. Use --catalogue, --event, or --events_file.")
            sys.exit(1)

    if not event_pairs:
        log.error("No events selected for this task.")
        sys.exit(1)

    all_event_pairs = list(event_pairs)

    # ── checkpoint filter ────────────────────────────────────────────────────
    if resume:
        def _is_complete(eid):
            p = os.path.join(RESULTS_DIR, f"{eid}_sweep.csv")
            if not os.path.exists(p):
                return False
            try:
                df = pd.read_csv(p)
                completed_n = sorted(df["N"].dropna().astype(int).unique().tolist())
                return (completed_n == sorted(set(n_values))
                        and "intra_sp_var_mean" in df.columns
                        and df["intra_sp_var_mean"].notna().sum() == len(n_values))
            except Exception:
                return False

        pending    = [(e, c) for e, c in event_pairs if not _is_complete(e)]
        done_count = len(event_pairs) - len(pending)
        if done_count:
            log.info(f"  Resume: {done_count} already complete, "
                     f"{len(pending)} pending.")
        event_pairs = pending

    # ── sweep ────────────────────────────────────────────────────────────────
    if not event_pairs:
        log.info("All events in this chunk already complete.")
    else:
        if args.workers == 1 or len(event_pairs) == 1:
            for eid, cls in tqdm(event_pairs, desc="Events"):
                sweep_event(eid, catalog, n_values, lifecycle_class=cls,
                            catalog_index=catalog_index,
                            n_workers=args.n_workers, resume=resume)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(_sweep_event_thread, eid, cls, catalog,
                                catalog_index, n_values, args.n_workers, resume
                                ): (eid, cls)
                    for eid, cls in event_pairs
                }
                for fut in tqdm(as_completed(futures), total=len(futures),
                                desc="Events"):
                    eid, cls = futures[fut]
                    try:
                        fut.result()
                    except Exception as exc:
                        log.error(f"  {eid} failed: {exc}", exc_info=True)

    # ── post-sweep: summary / plots ──────────────────────────────────────────
    # In ARRAY MODE: skip summary here.  All 17 tasks write to the same shared
    # results_dir concurrently — building the summary from a partial set would
    # produce misleading plots and text, and simultaneous writes to
    # sweep_summary.txt / PNG files would cause a Lustre write-race.
    # Run with --aggregate_only after all tasks finish.
    #
    # In SINGLE-JOB MODE: build the summary immediately from all CSVs present.
    if args.array_id is not None:
        log.info(
            f"\n  Array task {args.array_id} complete.\n"
            f"  Per-event CSVs written to {RESULTS_DIR}/\n"
            f"  When ALL {args.n_chunks} tasks have finished, run:\n\n"
            f"    python diagnostic_n_sweep_hpc.py \\\n"
            f"        --results_dir {RESULTS_DIR} \\\n"
            f"        --catalogue   <your_catalogue.csv> \\\n"
            f"        --aggregate_only\n"
        )
    else:
        # Single-job mode: reload all CSVs (including previously completed)
        all_dfs = []
        for eid, _ in all_event_pairs:
            p = os.path.join(RESULTS_DIR, f"{eid}_sweep.csv")
            if os.path.exists(p):
                try:
                    all_dfs.append(pd.read_csv(p))
                except Exception as exc:
                    log.warning(f"  Could not reload {p}: {exc}")

        if all_dfs:
            best_N = build_summary(all_dfs, n_values, RESULTS_DIR,
                                   alpha=args.alpha)
            print(f"\nSweep complete. Recommended N = {best_N}")
            print(f"  Plots → {RESULTS_DIR}/sweep_aggregate_plot.png")
            print(f"          {RESULTS_DIR}/sweep_by_class_plot.png")
            print(f"          {RESULTS_DIR}/sweep_joint_score_plot.png")
        else:
            print("No valid results produced.")


if __name__ == "__main__":
    main()