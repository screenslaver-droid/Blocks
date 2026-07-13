"""
run_metrics_pipeline.py  —  Superpixel Metrics Evaluation Pipeline
===================================================================
Accepts a SEVIR event ID, runs both segmentation algorithms, and scores
each using the full SuperpixelEvaluator metric suite:

    FAE  — Flow Adherence Error        (lower = better)
    BDV  — Boundary Displacement Variance / Hausdorff  (lower = better)
    GED  — Graph Edit Distance Rate    (informational — storm birth/death)
    MDV  — Material Derivative Variance (lower = better)
    LFS  — Lagrangian Feature Smoothness (lower = better)

Algorithms compared
-------------------
  1. TemporalSLIC_DEM          (Visualize_sevir_with_superpixel_fused.py)
     → Farneback flow + Lagrangian watershed, centroid-point advection.

  2. AdaptiveResidualTSLIC_DEM (Visualize_sevir_with_superpixel_fused_adaptive_residual_tslic.py)
     → Mean-member-flow advection, Birth/Death lifecycle, RAG construction.

Usage
-----
    python run_metrics_pipeline.py                 # interactive prompt
    python run_metrics_pipeline.py --event S832950 # direct event ID
    python run_metrics_pipeline.py --event S832950 --save_csv results.csv
"""

# --------------------------------------------------------------------------- #
# Standard imports
# --------------------------------------------------------------------------- #
import argparse
import logging
import os
import sys
import time

import cv2
import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import center_of_mass
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, directed_hausdorff
from skimage.filters import sobel
from skimage.measure import label as sk_label, regionprops
from skimage.segmentation import find_boundaries, mark_boundaries, slic, watershed
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# --------------------------------------------------------------------------- #
# CONFIGURATION  — edit paths to match your environment
# --------------------------------------------------------------------------- #
BASE_PATH    = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\SEVIR\2019"
CATALOG_PATH = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\SEVIR\CATALOG.csv"
RESULTS_DIR  = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\Events\metrics"

N_SEGMENTS       = 1500
COMPACTNESS      = 10.0
ELEVATION_LAMBDA = 0.5

# Adaptive Residual thresholds (normalised [0,1] space)
THRESH_DEATH = 0.05
THRESH_BIRTH = 0.15
MIN_SPACING  = 20

# Channels fused for segmentation (order is significant)
FUSION_CHANNELS = ["vil", "ir107", "ir069"]

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


def get_sevir_grid(extent, nx: int = 384, ny: int = 384):
    src  = ccrs.PlateCarree()
    x0, y0 = SEVIR_PROJ.transform_point(extent[0], extent[2], src)
    x1, y1 = SEVIR_PROJ.transform_point(extent[1], extent[3], src)
    xv, yv = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    grid   = src.transform_points(SEVIR_PROJ, xv, yv)
    return grid[..., 0], grid[..., 1]


def _load_and_clip_tile(item, extent, buf=0.1):
    da = rioxarray.open_rasterio(item.assets["data"].href, lock=False).squeeze()
    da = da.rio.clip_box(minx=extent[0] - buf, miny=extent[2] - buf,
                         maxx=extent[1] + buf, maxy=extent[3] + buf)
    return da.coarsen(x=3, y=3, boundary="trim").mean()


def fetch_and_regrid_dem(extent, nx: int = 384, ny: int = 384) -> np.ndarray:
    """Fetch Copernicus DEM-30 and regrid to the SEVIR pixel grid."""
    cache_key = tuple(np.round(extent, 4))
    if cache_key in _dem_cache:
        return _dem_cache[cache_key]
    if not HAS_DEM_LIBS:
        log.warning("DEM libs unavailable — using flat terrain.")
        return np.zeros((ny, nx), dtype=np.float32)

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
        for fut in tqdm(as_completed(fmap), total=len(items), desc="DEM tiles"):
            try:    datasets[fmap[fut]] = fut.result()
            except Exception as e: log.warning(f"DEM tile failed: {e}")

    datasets = [d for d in datasets if d is not None and d.size > 0]
    merged   = merge_arrays(datasets) if len(datasets) > 1 else datasets[0]
    lons, lats, vals = merged.x.values, merged.y.values, merged.values
    if lats[0] > lats[-1]:
        lats, vals = lats[::-1], vals[::-1, :]

    interp = RegularGridInterpolator(
        (lats, lons), vals.astype(np.float32),
        method="linear", bounds_error=False, fill_value=0.0,
    )
    tgt_lons, tgt_lats = get_sevir_grid(extent, nx, ny)
    result = interp(
        np.column_stack((tgt_lats.ravel(), tgt_lons.ravel()))
    ).reshape(ny, nx).astype(np.float32)
    _dem_cache[cache_key] = result
    return result


# =========================================================================== #
# DATA LOADING
# =========================================================================== #
def _get_channel_path(event_id: str, img_type: str, catalog: pd.DataFrame):
    rows = catalog[(catalog["id"] == event_id) & (catalog["img_type"] == img_type)]
    if rows.empty:
        return None
    return os.path.join(BASE_PATH, img_type, os.path.basename(rows.iloc[0]["file_name"]))


def _read_hdf5(path: str, event_id: str, img_type: str):
    if not os.path.exists(path):
        log.warning(f"  File not found: {path}")
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
    log.warning(f"  Event {event_id} not found in {path}")
    return None


def _load_single_channel(event_id: str, ch: str,
                          catalog: pd.DataFrame, tgt_shape: tuple):
    path = _get_channel_path(event_id, ch, catalog)
    if path is None:
        log.warning(f"  {ch}: no catalog entry.")
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


def load_fused_channels(event_id: str, catalog: pd.DataFrame):
    """
    Load satellite channels and build the (T, H, W, C) normalised feature cube.

    Returns:
        fused  : (T, H, W, C) float32 — normalised, for segmentation
        data   : {ch: (T, H, W) float32} — raw values, for reference
        extent : [llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat]
    """
    meta = catalog[catalog["id"] == event_id]
    if meta.empty:
        log.error(f"Event {event_id} not found in catalog.")
        return None, {}, []

    row    = meta.iloc[0]
    extent = [row["llcrnrlon"], row["urcrnrlon"], row["llcrnrlat"], row["urcrnrlat"]]

    data      = {}
    tgt_shape = (384, 384)
    vil = _load_single_channel(event_id, "vil", catalog, tgt_shape)
    if vil is not None:
        data["vil"] = vil
        tgt_shape   = (vil.shape[1], vil.shape[2])

    for ch in ["ir107", "ir069"]:
        arr = _load_single_channel(event_id, ch, catalog, tgt_shape)
        if arr is not None:
            data[ch] = arr

    fused_stack = []
    for ch in FUSION_CHANNELS:
        if ch in data:
            arr = data[ch]
            fused_stack.append((arr - arr.min()) / (arr.max() - arr.min() + 1e-6))
        else:
            log.warning(f"  Fusion channel '{ch}' missing — skipping.")

    if not fused_stack:
        log.error("  No fusion channels — cannot build feature cube.")
        return None, data, extent

    fused = np.stack(fused_stack, axis=-1).astype(np.float32)   # (T, H, W, C)
    log.info(f"  Fused cube: {fused.shape}")
    return fused, data, extent


# =========================================================================== #
# ALGORITHM 1 — TemporalSLIC_DEM
# (from Visualize_sevir_with_superpixel_fused.py)
# =========================================================================== #
class TemporalSLIC_DEM:
    """
    Lagrangian Temporal SLIC on a fused multi-channel + DEM feature cube.
    Centroids are advected using the single-pixel flow at each centroid position.
    Returns (labels, flow) per frame.
    """
    def __init__(self, dem_norm: np.ndarray, n_segments: int = 1500,
                 compactness: float = 10.0, lambda_z: float = 0.5):
        self.n_segments  = n_segments
        self.compactness = compactness
        self.lambda_z    = lambda_z
        self.dem_norm     = dem_norm
        self.dem_weighted = dem_norm * lambda_z

        self.prev_labels    = None
        self.prev_centroids = None   # {label: np.array([y, x])}
        self.prev_gray      = None

    def segment(self, fused_frame: np.ndarray, use_flow: bool = True):
        if fused_frame.ndim == 2:
            fused_frame = fused_frame[:, :, np.newaxis]
        H, W, C = fused_frame.shape

        gray_norm = fused_frame[:, :, 0]
        curr_gray = (gray_norm * 255).astype(np.uint8)

        dem_ch       = (self.dem_norm * self.lambda_z)[:H, :W, np.newaxis]
        feature_cube = np.concatenate([fused_frame, dem_ch], axis=-1).astype(np.float64)

        flow = np.zeros((H, W, 2), dtype=np.float32)

        if self.prev_labels is None:
            labels = slic(feature_cube, n_segments=self.n_segments,
                          compactness=self.compactness, start_label=1,
                          enforce_connectivity=True, channel_axis=-1)
            self.prev_centroids = self._calculate_centroids(labels)
        else:
            if use_flow:
                flow = cv2.calcOpticalFlowFarneback(
                    self.prev_gray, curr_gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
            advected_markers = np.zeros((H, W), dtype=np.int32)
            for lbl, centroid in self.prev_centroids.items():
                y_old, x_old = int(centroid[0]), int(centroid[1])
                if 0 <= y_old < H and 0 <= x_old < W:
                    window = 5
                    y1 = max(0, y_old - window); y2 = min(H, y_old + window)
                    x1 = max(0, x_old - window); x2 = min(W, x_old + window)
                    flow_y = np.mean(flow[y1:y2, x1:x2, 1])
                    flow_x = np.mean(flow[y1:y2, x1:x2, 0])
                    y_new  = int(np.clip(y_old + flow_y, 0, H - 1))
                    x_new  = int(np.clip(x_old + flow_x, 0, W - 1))
                    advected_markers[y_new, x_new] = lbl

            grad_layers = [sobel(fused_frame[:, :, c]) for c in range(C)]
            grad_layers.append(sobel(self.dem_weighted))
            combined_gradient = np.max(np.stack(grad_layers, axis=0), axis=0)

            labels = watershed(combined_gradient, markers=advected_markers,
                               compactness=self.compactness / 100.0)
            self.prev_centroids = self._calculate_centroids(labels)

        self.prev_labels = labels
        self.prev_gray   = curr_gray
        return labels, flow

    def _calculate_centroids(self, labels: np.ndarray) -> dict:
        unique = np.unique(labels)
        if len(unique) and unique[0] == 0:
            unique = unique[1:]
        cms = center_of_mass(np.ones_like(labels), labels, unique)
        return {lbl: np.array(c) for lbl, c in zip(unique, cms)}


# =========================================================================== #
# ALGORITHM 2 — AdaptiveResidualTSLIC_DEM
# (from Visualize_sevir_with_superpixel_fused_adaptive_residual_tslic.py)
# =========================================================================== #
class AdaptiveResidualTSLIC_DEM:
    """
    Adaptive Residual Temporal SLIC on a fused multi-channel + DEM feature cube.
    Key differences from TemporalSLIC_DEM:
      • Centers advected by MEAN flow of all member pixels (not just centroid).
      • Superpixel Death when fused intensity drops below thresh_death.
      • Superpixel Birth in high-residual regions not covered by existing centers.
      • Region Adjacency Graph (RAG) built every frame.
    Returns (labels, flow, rag) per frame.
    """
    def __init__(self, dem_norm: np.ndarray, n_segments: int = 1500,
                 compactness: float = 10.0, lambda_z: float = 0.5,
                 thresh_death: float = THRESH_DEATH, thresh_birth: float = THRESH_BIRTH,
                 min_spacing: int = MIN_SPACING):
        self.n_segments   = n_segments
        self.compactness  = compactness
        self.lambda_z     = lambda_z
        self.thresh_death = thresh_death
        self.thresh_birth = thresh_birth
        self.min_spacing  = min_spacing
        self.dem_norm     = dem_norm
        self.dem_weighted = dem_norm * lambda_z

        self.prev_labels  = None
        self.prev_centers = None   # (K, 2) array [y, x]
        self.prev_fused   = None
        self.prev_gray    = None

    def segment(self, fused_frame: np.ndarray, use_flow: bool = True):
        if fused_frame.ndim == 2:
            fused_frame = fused_frame[:, :, np.newaxis]
        H, W, C = fused_frame.shape

        curr_gray    = (fused_frame[:, :, 0] * 255).astype(np.uint8)
        dem_ch       = self.dem_norm[:H, :W, np.newaxis] * self.lambda_z
        feature_cube = np.concatenate([fused_frame, dem_ch], axis=-1).astype(np.float64)
        flow         = np.zeros((H, W, 2), dtype=np.float32)

        # Frame 0: initialise with multi-channel SLIC
        if self.prev_labels is None:
            labels = slic(feature_cube, n_segments=self.n_segments,
                          compactness=self.compactness, start_label=1,
                          enforce_connectivity=True, channel_axis=-1)
            self.prev_centers = self._labels_to_centers(labels)
            self.prev_labels  = labels
            self.prev_fused   = fused_frame
            self.prev_gray    = curr_gray
            rag = self._build_rag(labels, fused_frame)
            return labels, flow, rag

        # Frame t+: Adaptive Residual update
        if use_flow:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )

        new_centers = self._advect_centers(flow, H, W)
        pred_fused  = self._warp_fused(self.prev_fused, flow)
        residual    = np.max(np.abs(fused_frame - pred_fused), axis=-1)

        new_centers = self._apply_death(new_centers, fused_frame, H, W)
        new_centers = self._apply_birth(new_centers, fused_frame, residual, H, W)
        labels      = self._refine_watershed(fused_frame, new_centers, H, W)

        self.prev_centers = self._labels_to_centers(labels)
        rag = self._build_rag(labels, fused_frame)

        self.prev_labels = labels
        self.prev_fused  = fused_frame
        self.prev_gray   = curr_gray
        return labels, flow, rag

    # ------------------------------------------------------------------ #
    def _labels_to_centers(self, labels: np.ndarray) -> np.ndarray:
        unique = np.unique(labels)
        unique = unique[unique != 0]
        if len(unique) == 0:
            return np.empty((0, 2), dtype=np.float32)
        cms = center_of_mass(np.ones_like(labels), labels, unique)
        return np.array(cms, dtype=np.float32)

    def _advect_centers(self, flow: np.ndarray, H: int, W: int) -> np.ndarray:
        labels  = self.prev_labels
        centers = self.prev_centers
        if centers is None or len(centers) == 0:
            return np.empty((0, 2), dtype=np.float32)
        unique  = np.unique(labels)
        unique  = unique[unique != 0]
        flow_fy, flow_fx = flow[:, :, 1], flow[:, :, 0]
        new_centers = []
        for k, lbl in enumerate(unique):
            if k >= len(centers):
                break
            mask = (labels == lbl)
            mean_fy = float(flow_fy[mask].mean()) if mask.any() else 0.0
            mean_fx = float(flow_fx[mask].mean()) if mask.any() else 0.0
            y_old, x_old = centers[k]
            new_centers.append([
                float(np.clip(y_old + mean_fy, 0, H - 1)),
                float(np.clip(x_old + mean_fx, 0, W - 1)),
            ])
        return np.array(new_centers, dtype=np.float32) if new_centers \
               else np.empty((0, 2), dtype=np.float32)

    def _warp_fused(self, fused: np.ndarray, flow: np.ndarray) -> np.ndarray:
        H, W  = flow.shape[:2]
        map_y = (np.arange(H, dtype=np.float32)[:, None]
                 * np.ones((1, W), dtype=np.float32)) - flow[:, :, 1]
        map_x = (np.arange(W, dtype=np.float32)[None, :]
                 * np.ones((H, 1), dtype=np.float32)) - flow[:, :, 0]
        warped = np.empty_like(fused)
        for c in range(fused.shape[2]):
            warped[:, :, c] = cv2.remap(fused[:, :, c].astype(np.float32),
                                        map_x, map_y,
                                        interpolation=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REPLICATE)
        return warped

    def _apply_death(self, centers: np.ndarray,
                     fused_curr: np.ndarray, H: int, W: int) -> np.ndarray:
        if len(centers) == 0:
            return centers
        survived = []
        for cy, cx in centers:
            iy = int(np.clip(cy, 0, H - 1))
            ix = int(np.clip(cx, 0, W - 1))
            if float(fused_curr[iy, ix, :].max()) >= self.thresh_death:
                survived.append([cy, cx])
        return np.array(survived, dtype=np.float32) if survived \
               else np.empty((0, 2), dtype=np.float32)

    def _apply_birth(self, centers: np.ndarray, fused_curr: np.ndarray,
                     residual: np.ndarray, H: int, W: int) -> np.ndarray:
        birth_mask = (residual > self.thresh_birth)
        if not birth_mask.any():
            return centers
        blob_labels, n_blobs = sk_label(birth_mask, return_num=True)
        new_centers = list(centers)
        existing    = np.array(new_centers) if len(new_centers) > 0 else None
        for blob_id in range(1, n_blobs + 1):
            bmask = (blob_labels == blob_id)
            if fused_curr[bmask].max() < self.thresh_death:
                continue
            props = regionprops(bmask.astype(np.uint8))
            if not props:
                continue
            by, bx = props[0].centroid
            if existing is not None and len(existing) > 0:
                dists = np.hypot(existing[:, 0] - by, existing[:, 1] - bx)
                if dists.min() < self.min_spacing:
                    continue
            nc = [float(np.clip(by, 0, H - 1)), float(np.clip(bx, 0, W - 1))]
            new_centers.append(nc)
            existing = np.array(new_centers)
        return np.array(new_centers, dtype=np.float32)

    def _refine_watershed(self, fused_curr: np.ndarray,
                          centers: np.ndarray, H: int, W: int) -> np.ndarray:
        markers = np.zeros((H, W), dtype=np.int32)
        for lbl_idx, (cy, cx) in enumerate(centers, start=1):
            markers[int(np.clip(cy, 0, H - 1)), int(np.clip(cx, 0, W - 1))] = lbl_idx
        C = fused_curr.shape[2]
        grad_layers = [sobel(fused_curr[:, :, c]) for c in range(C)]
        grad_layers.append(sobel(self.dem_weighted[:H, :W]))
        combined_grad = np.max(np.stack(grad_layers, axis=0), axis=0)
        return watershed(combined_grad, markers=markers,
                         compactness=self.compactness / 100.0)

    def _build_rag(self, labels: np.ndarray, fused_curr: np.ndarray) -> dict:
        unique = np.unique(labels)
        unique = unique[unique != 0]
        nodes  = {int(lbl): fused_curr[labels == lbl].mean(axis=0) for lbl in unique}
        adj_set = set()
        for left, right in [(labels[:, :-1], labels[:, 1:]),
                             (labels[:-1, :], labels[1:, :])]:
            diff = left != right
            pa, pb = left[diff].ravel(), right[diff].ravel()
            for a, b in zip(pa, pb):
                if a != 0 and b != 0:
                    adj_set.add((int(min(a, b)), int(max(a, b))))
        edges = [(a, b, float(np.linalg.norm(nodes[a] - nodes[b])))
                 for a, b in adj_set if a in nodes and b in nodes]
        return {"nodes": nodes, "edges": edges}


# =========================================================================== #
# CENTROID ADAPTER UTILITIES
# =========================================================================== #
def centers_array_to_dict(centers: np.ndarray) -> dict:
    """
    Convert AdaptiveResidualTSLIC_DEM's (K, 2) center array into the
    {label: np.array([y, x])} dict format expected by SuperpixelEvaluator.
    The label assigned to center k is (k+1), consistent with watershed markers.
    """
    if centers is None or len(centers) == 0:
        return {}
    return {k + 1: np.array(centers[k]) for k in range(len(centers))}


# =========================================================================== #
# SUPERPIXEL EVALUATOR
# =========================================================================== #
class SuperpixelEvaluator:
    """
    Evaluates temporal superpixel segmentations using physical fluid-dynamics
    and graph-theory metrics on the fused satellite/DEM feature cube.

    Metrics
    -------
    FAE : Flow Adherence Error        — are superpixels moving with the flow?
    BDV : Boundary Displacement Variance (Hausdorff) — boundary stability.
    GED : Graph Edit Distance Rate    — superpixel birth/death rate.
    MDV : Material Derivative Variance — intra-SP physical consistency.
    LFS : Lagrangian Feature Smoothness — temporal smoothness along trajectories.
    """
    def __init__(self, fused_cube: np.ndarray, flow_fields: list):
        self.fused_cube  = fused_cube
        self.flow_fields = flow_fields
        self.T, self.H, self.W, self.C = fused_cube.shape

    # ----------------------------------------------------------------------- #
    def calculate_fae(self, centroids_t: dict, centroids_t1: dict,
                      optical_flow: np.ndarray, labels_t: np.ndarray) -> float:
        common = sorted(set(centroids_t) & set(centroids_t1))
        if not common:
            return 0.0
        coords_t  = np.array([centroids_t[l]  for l in common], dtype=np.float32)
        coords_t1 = np.array([centroids_t1[l] for l in common], dtype=np.float32)
        vel_actual = coords_t1[:, ::-1] - coords_t[:, ::-1]   # [dx, dy]

        flow_u = optical_flow[..., 0]
        flow_v = optical_flow[..., 1]
        flat   = labels_t.ravel()
        mx     = int(max(common)) + 1
        su = np.bincount(flat, weights=flow_u.ravel(), minlength=mx)
        sv = np.bincount(flat, weights=flow_v.ravel(), minlength=mx)
        cnt = np.bincount(flat, minlength=mx).clip(min=1)
        mean_flow = np.stack([su[common] / cnt[common],
                              sv[common] / cnt[common]], axis=1)
        return float(np.linalg.norm(vel_actual - mean_flow, axis=1).mean())

    def calculate_bdv(self, labels_t: np.ndarray, labels_t1: np.ndarray,
                      optical_flow: np.ndarray) -> float:
        bound_t  = find_boundaries(labels_t,  mode='inner')
        bound_t1 = find_boundaries(labels_t1, mode='inner')
        map_y = (np.arange(self.H, dtype=np.float32)[:, None]
                 * np.ones((1, self.W), dtype=np.float32)) - optical_flow[:, :, 1]
        map_x = (np.arange(self.W, dtype=np.float32)[None, :]
                 * np.ones((self.H, 1), dtype=np.float32)) - optical_flow[:, :, 0]
        bound_t_warped = cv2.remap(bound_t.astype(np.float32), map_x, map_y,
                                   interpolation=cv2.INTER_NEAREST) > 0.5
        pts_warp = np.argwhere(bound_t_warped)
        pts_t1   = np.argwhere(bound_t1)
        if len(pts_warp) == 0 or len(pts_t1) == 0:
            return 0.0
        d1 = directed_hausdorff(pts_warp, pts_t1)[0]
        d2 = directed_hausdorff(pts_t1, pts_warp)[0]
        return max(d1, d2)

    def calculate_ged_rate(self, tracked_t: set, tracked_t1: set) -> float:
        born  = len(tracked_t1 - tracked_t)
        dead  = len(tracked_t  - tracked_t1)
        total = len(tracked_t  | tracked_t1)
        return (born + dead) / total if total > 0 else 0.0

    def calculate_mdv(self, fused_t: np.ndarray, fused_t1: np.ndarray,
                      labels_t1: np.ndarray, optical_flow: np.ndarray) -> float:
        map_y = (np.arange(self.H, dtype=np.float32)[:, None]
                 * np.ones((1, self.W), dtype=np.float32)) - optical_flow[:, :, 1]
        map_x = (np.arange(self.W, dtype=np.float32)[None, :]
                 * np.ones((self.H, 1), dtype=np.float32)) - optical_flow[:, :, 0]
        fused_t_warped = np.empty_like(fused_t)
        for c in range(self.C):
            fused_t_warped[..., c] = cv2.remap(
                fused_t[..., c], map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE)
        R_pixel = np.linalg.norm(fused_t1 - fused_t_warped, axis=-1)
        unique  = np.unique(labels_t1)
        unique  = unique[unique != 0]
        mdv_sum, total_px = 0.0, 0
        for lbl in unique:
            mask = (labels_t1 == lbl)
            n    = int(mask.sum())
            if n > 1:
                mdv_sum  += np.var(R_pixel[mask]) * n
                total_px += n
        return mdv_sum / total_px if total_px > 0 else 0.0

    def _track_labels(self, centroids_t: dict, centroids_t1_raw: dict,
                      flow: np.ndarray, dist_thresh: float = 15.0) -> dict:
        if not centroids_t or not centroids_t1_raw:
            return centroids_t1_raw
        advected_t = {}
        for lbl, (y, x) in centroids_t.items():
            iy = int(np.clip(y, 0, self.H - 1))
            ix = int(np.clip(x, 0, self.W - 1))
            fy, fx = flow[iy, ix, 1], flow[iy, ix, 0]
            advected_t[lbl] = np.array([y + fy, x + fx])

        t_keys  = list(advected_t.keys())
        t_pts   = np.array([advected_t[k] for k in t_keys])
        t1_keys = list(centroids_t1_raw.keys())
        t1_pts  = np.array([centroids_t1_raw[k] for k in t1_keys])

        cost = cdist(t_pts, t1_pts)
        row_ind, col_ind = linear_sum_assignment(cost)

        tracked = {}
        next_id = max(t_keys) + 1 if t_keys else 1
        matched = set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < dist_thresh:
                tracked[t_keys[r]] = t1_pts[c]
                matched.add(c)
        for c in range(len(t1_pts)):
            if c not in matched:
                tracked[next_id] = t1_pts[c]
                next_id += 1
        return tracked

    def evaluate(self, labels_list: list, centroids_raw_list: list) -> dict:
        """
        Run the full metric suite over all frames.

        Args:
            labels_list       : T × (H, W) int32 label maps.
            centroids_raw_list: T × {label: np.array([y, x])} centroid dicts.

        Returns:
            dict with scalar values for FAE, BDV, GED, MDV, LFS.
        """
        metrics = {"FAE": [], "BDV": [], "GED": [], "MDV": []}
        track_features:  dict = {}
        tracked_centroids = [{}] * self.T
        tracked_centroids[0] = centroids_raw_list[0]

        for lbl in tracked_centroids[0]:
            mask = labels_list[0] == lbl
            if mask.any():
                track_features[lbl] = [self.fused_cube[0][mask].mean(axis=0)]

        for t in tqdm(range(self.T - 1), desc="  Evaluating frames", leave=False):
            lbl_t, lbl_t1   = labels_list[t], labels_list[t + 1]
            flow             = self.flow_fields[t]
            fused_t, fused_t1 = self.fused_cube[t], self.fused_cube[t + 1]

            tracked_c_t1 = self._track_labels(
                tracked_centroids[t], centroids_raw_list[t + 1], flow
            )
            tracked_centroids[t + 1] = tracked_c_t1

            metrics["FAE"].append(
                self.calculate_fae(tracked_centroids[t], tracked_c_t1, flow, lbl_t))
            metrics["BDV"].append(
                self.calculate_bdv(lbl_t, lbl_t1, flow))
            metrics["GED"].append(
                self.calculate_ged_rate(set(tracked_centroids[t]), set(tracked_c_t1)))
            metrics["MDV"].append(
                self.calculate_mdv(fused_t, fused_t1, lbl_t1, flow))

            # Update Lagrangian feature history for LFS
            for lbl, c_pos in tracked_c_t1.items():
                raw_matches = [k for k, v in centroids_raw_list[t + 1].items()
                               if np.allclose(v, c_pos)]
                if raw_matches:
                    mask = (lbl_t1 == raw_matches[0])
                    if mask.any():
                        feat = fused_t1[mask].mean(axis=0)
                        track_features.setdefault(lbl, []).append(feat)

        # Lagrangian Feature Smoothness
        lfs_scores = []
        for history in track_features.values():
            if len(history) > 1:
                arr   = np.array(history)
                diffs = np.linalg.norm(arr[1:] - arr[:-1], axis=1)
                lfs_scores.append(diffs.mean())
        metrics["LFS"] = float(np.mean(lfs_scores)) if lfs_scores else 0.0

        for key in ["FAE", "BDV", "GED", "MDV"]:
            metrics[key] = float(np.mean(metrics[key])) if metrics[key] else 0.0

        return metrics


# =========================================================================== #
# SEGMENTATION RUNNERS
# =========================================================================== #
def run_temporal_slic(fused: np.ndarray, dem_norm: np.ndarray):
    """
    Run TemporalSLIC_DEM over all frames.
    Returns (labels_list, centroids_dict_list, flow_list).
    """
    T = fused.shape[0]
    tslic = TemporalSLIC_DEM(dem_norm, n_segments=N_SEGMENTS,
                             compactness=COMPACTNESS, lambda_z=ELEVATION_LAMBDA)
    labels_list, centroids_list, flow_list = [], [], []
    for t in tqdm(range(T), desc="  TemporalSLIC_DEM", leave=False):
        labels, flow = tslic.segment(fused[t], use_flow=True)
        labels_list.append(labels)
        centroids_list.append(dict(tslic.prev_centroids))
        flow_list.append(flow)
    return labels_list, centroids_list, flow_list


def run_adaptive_slic(fused: np.ndarray, dem_norm: np.ndarray):
    """
    Run AdaptiveResidualTSLIC_DEM over all frames.
    Returns (labels_list, centroids_dict_list, flow_list).
    Centroid arrays are converted to dicts for metric compatibility.
    """
    T = fused.shape[0]
    aslic = AdaptiveResidualTSLIC_DEM(
        dem_norm, n_segments=N_SEGMENTS, compactness=COMPACTNESS,
        lambda_z=ELEVATION_LAMBDA, thresh_death=THRESH_DEATH,
        thresh_birth=THRESH_BIRTH, min_spacing=MIN_SPACING,
    )
    labels_list, centroids_list, flow_list = [], [], []
    for t in tqdm(range(T), desc="  AdaptiveResidualTSLIC_DEM", leave=False):
        labels, flow, _ = aslic.segment(fused[t], use_flow=True)
        labels_list.append(labels)
        # Convert (K, 2) array → {label: [y, x]} dict
        centroids_list.append(
            centers_array_to_dict(aslic.prev_centers)
        )
        flow_list.append(flow)
    return labels_list, centroids_list, flow_list


# =========================================================================== #
# RESULTS DISPLAY
# =========================================================================== #
def print_results_table(event_id: str, results: dict):
    """Print a side-by-side comparison table of both algorithms."""
    algos  = list(results.keys())
    keys   = ["FAE", "BDV", "GED", "MDV", "LFS"]
    labels = {
        "FAE": "Flow Adherence Error      ↓",
        "BDV": "Boundary Displacement Var ↓",
        "GED": "Graph Edit Distance Rate  ℹ",
        "MDV": "Material Deriv. Variance  ↓",
        "LFS": "Lagrangian Feature Smooth ↓",
    }
    col_w  = 22

    sep   = "+" + "-" * 34 + "+" + ("-" * col_w + "+") * len(algos)
    header = f"| {'Metric':<32} |" + "".join(f" {a:^{col_w-2}} |" for a in algos)

    print(f"\n{'=' * len(sep)}")
    print(f"  Superpixel Evaluation — Event: {event_id}")
    print(f"{'=' * len(sep)}")
    print(sep); print(header); print(sep)

    for k in keys:
        vals   = [results[a].get(k, float("nan")) for a in algos]
        row    = f"| {labels[k]:<32} |"
        # Highlight the better (lower) value for non-GED metrics
        if k != "GED" and not any(np.isnan(v) for v in vals):
            best = min(vals)
        else:
            best = None
        for v in vals:
            marker = " *" if (best is not None and v == best) else "  "
            row   += f" {v:>10.4f}{marker:>10} |"
        print(row)

    print(sep)
    print("  * = better (lower) value")
    print()


def save_csv(event_id: str, results: dict, path: str):
    rows = []
    for algo, metrics in results.items():
        row = {"event_id": event_id, "algorithm": algo}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"  Results saved → {path}")


# =========================================================================== #
# MAIN PIPELINE
# =========================================================================== #
def run_pipeline(event_id: str, catalog: pd.DataFrame, save_csv_path: str = None):
    print(f"\n{'=' * 60}")
    print(f"  Pipeline: Event {event_id}")
    print(f"{'=' * 60}")

    # ── 1. Load data ────────────────────────────────────────────────────────
    print("\n[1/4] Loading fused channels …")
    fused, data, extent = load_fused_channels(event_id, catalog)
    if fused is None:
        print("  ERROR: No data found. Check event ID and file paths.")
        return None

    T = fused.shape[0]
    print(f"       Frames : {T}   ({T * 300 // 60} min total)")
    print(f"       Cube   : {fused.shape}  (T × H × W × C)")

    if T < 2:
        print("  ERROR: Event has fewer than 2 frames — cannot evaluate temporal metrics.")
        return None

    # ── 2. Fetch DEM ────────────────────────────────────────────────────────
    print("\n[2/4] Fetching / gridding DEM …")
    dem_raw  = fetch_and_regrid_dem(extent)
    dem_norm = ((dem_raw - dem_raw.min()) /
                (dem_raw.max() - dem_raw.min() + 1e-6)).astype(np.float32)
    print(f"       DEM range: [{dem_raw.min():.0f} m, {dem_raw.max():.0f} m]")

    # ── 3. Run segmentation algorithms ──────────────────────────────────────
    print("\n[3/4] Running segmentation algorithms …")

    print("  → TemporalSLIC_DEM")
    t0 = time.perf_counter()
    tslic_labels, tslic_centroids, tslic_flows = run_temporal_slic(fused, dem_norm)
    tslic_time = time.perf_counter() - t0
    print(f"       Done in {tslic_time:.1f}s")

    print("  → AdaptiveResidualTSLIC_DEM")
    t0 = time.perf_counter()
    aslic_labels, aslic_centroids, aslic_flows = run_adaptive_slic(fused, dem_norm)
    aslic_time = time.perf_counter() - t0
    print(f"       Done in {aslic_time:.1f}s")

    # ── 4. Evaluate metrics ──────────────────────────────────────────────────
    print("\n[4/4] Computing evaluation metrics …")

    print("  → Evaluating TemporalSLIC_DEM …")
    tslic_eval = SuperpixelEvaluator(fused, tslic_flows)
    tslic_metrics = tslic_eval.evaluate(tslic_labels, tslic_centroids)
    tslic_metrics["runtime_s"] = tslic_time

    print("  → Evaluating AdaptiveResidualTSLIC_DEM …")
    aslic_eval = SuperpixelEvaluator(fused, aslic_flows)
    aslic_metrics = aslic_eval.evaluate(aslic_labels, aslic_centroids)
    aslic_metrics["runtime_s"] = aslic_time

    results = {
        "TemporalSLIC_DEM":         tslic_metrics,
        "AdaptiveResidualTSLIC_DEM": aslic_metrics,
    }

    # ── Display ──────────────────────────────────────────────────────────────
    print_results_table(event_id, results)

    # Runtime summary
    print("  Runtime:")
    print(f"    TemporalSLIC_DEM          : {tslic_time:7.1f} s")
    print(f"    AdaptiveResidualTSLIC_DEM : {aslic_time:7.1f} s")
    print()

    # ── Optional CSV save ────────────────────────────────────────────────────
    if save_csv_path:
        save_csv(event_id, results, save_csv_path)
    else:
        default_csv = os.path.join(RESULTS_DIR, f"{event_id}_metrics.csv")
        save_csv(event_id, results, default_csv)

    return results


# =========================================================================== #
# ENTRY POINT
# =========================================================================== #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SEVIR Superpixel Metrics Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--event",    type=str, default=None,
                        help="SEVIR event ID (e.g. S832950)")
    parser.add_argument("--save_csv", type=str, default=None,
                        help="Optional path to save results CSV")
    args = parser.parse_args()

    if not os.path.exists(CATALOG_PATH):
        print(f"ERROR: Catalog not found at:\n  {CATALOG_PATH}")
        sys.exit(1)

    print("Loading SEVIR catalog …")
    catalog = pd.read_csv(CATALOG_PATH, parse_dates=["time_utc"], low_memory=False)
    print(f"  {len(catalog):,} rows loaded.")

    if args.event:
        # ── Single-event mode (CLI) ──────────────────────────────────────────
        run_pipeline(args.event, catalog, save_csv_path=args.save_csv)
    else:
        # ── Interactive loop ────────────────────────────────────────────────
        print("\nEntering interactive mode. Type 'q' to quit.\n")
        while True:
            print("=" * 60)
            event_id = input("Enter Event ID (or 'q' to quit): ").strip()
            if event_id.lower() == "q":
                print("Goodbye.")
                break
            if not event_id:
                continue
            run_pipeline(event_id, catalog)