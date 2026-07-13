"""
visualize_b1_cc_vs_slic.py — Block B1: Convective Cell Boundary Alignment
==========================================================================
Visual diagnostic for Block B1 (Convective Cell Boundary Alignment).

Implements both external quality metrics for B1 per blocks_technical_v2_latest.md:

  METRIC 1 — F1_nested (boundary alignment, upper bound N_F1):
  1. Builds the SCIT-inspired multi-threshold nested reference (Steps 1a–1f):
     - Steps 1a–1e applied independently at τ_low, τ_med, τ_high with the
       *same* σ_cell calibrated at τ_med.  Strictly nested: C(τ_high) ⊆ C(τ_med) ⊆ C(τ_low).
     - Step 1f: ∂C_nested = ∂C_low ∪ ∂C_med ∪ ∂C_high  (union boundary).
     - Step 1g: optional watershed-from-maxima split of residual merged cells.
     - Step 1h: single ROI Ω_C = dilate(C_low, δ_margin); δ_margin = c_margin × sqrt(H×W/N_max).
  2. Runs SLIC (t=0 branch of TemporalSLIC) for one or more requested N.
  3. Computes three-stage ablation (Step 3):
       Stage 0 (original, flawed): Global Precision  × flat ∂C_med
       Stage 1 (ROI fix only):     ROI Precision      × flat ∂C_med
       Stage 2 (primary):          ROI Precision      × ∂C_nested
     F1_nested = Stage 2 result.  N_F1 = min{N : F1_nested(N) ≥ 0.95 × peak}.

  METRIC 2 — per-cell mIoU (instance scale, lower bound N_IoU):
  4. Per-cell IoU with one-to-one Hungarian matching at each level τ_k.
     mIoU(N; τ_k) = (1/M_k) Σ_i IoU(C_i); unmatched instances → IoU=0.
     PRIMARY (area-weighted): mIoU_weighted(N; τ_k) = Σ_i IoU(C_i) × Area(C_i)/ΣArea(C_j).
     This prevents one severe, badly-undersegmented core from being diluted
     by several small well-matched "noise" cores in the unweighted mean —
     the unweighted mean is still computed and reported alongside it.
     N_IoU(τ_high) = soft-argmax-at-95%-peak of mIoU_weighted(N; τ_high).

  VALIDATION INTERVAL (Step B5):
     N_IoU(τ_high) ≤ N*_class ≤ N_F1

ROI masking: δ_margin calibrated as c_margin × sqrt(H×W/N_max);
default c_margin=0.75 → δ_margin≈7 px at N_max=1500 (spec §1h).
Sensitivity sweep available via --margin_sweep.

Usage
-----
    python visualize_b1_cc_vs_slic.py --event_id <ID> --n_list 500 850 1150
    python visualize_b1_cc_vs_slic.py                      # interactive prompt
    python visualize_b1_cc_vs_slic.py --event_id <ID> --n_list 1150 --split_merged
    python visualize_b1_cc_vs_slic.py --event_id <ID> --n_list 1150 --sigma_sweep
    python visualize_b1_cc_vs_slic.py --event_id <ID> --n_list 500 1150 --margin_sweep

Dependencies
------------
    pip install h5py pandas numpy scipy scikit-image opencv-python matplotlib tqdm
    (DEM channel is optional; requires pystac_client, planetary_computer,
     rioxarray — gracefully skipped if unavailable.)
"""

import os
import argparse
import logging

import h5py
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from scipy.ndimage import (
    gaussian_filter, binary_fill_holes, label as cc_label,
    binary_dilation,
)
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import slic as skimage_slic, mark_boundaries, watershed
from skimage.feature import peak_local_max
from skimage.morphology import disk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION (paths match Visualize_sevir_with_superpixel_fused.py)
# ─────────────────────────────────────────────────────────────────────────────
BASE_PATH    = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\SEVIR\2019"
CATALOG_PATH = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\SEVIR\CATALOG.csv"
OUT_DIR      = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\Blocks\B\B1_Diagnostics"
os.makedirs(OUT_DIR, exist_ok=True)

FUSION_CHANNELS  = ["vil", "ir107", "ir069"]
COMPACTNESS      = 10.0
ELEVATION_LAMBDA = 0.5

# ── Absolute physical VIL thresholds (Preliminary section, blocks_technical_v2.md) ──
VIL_MAX_PHYS = 70.0   # kg/m^2, SEVIR clip value
TAU_LOW      = 4.4    # kg/m^2  — onset of detectable precipitation
TAU_MED      = 20.3   # kg/m^2  — moderate deep convection
TAU_HIGH     = 36.5   # kg/m^2  — severe / extreme convection

# ── Step 1f: SCIT-inspired multi-threshold nested reference ─────────────────
# Three threshold levels; processing order is low→med→high (ascending intensity).
# Nested structure by construction: C(τ_high) ⊆ C(τ_med) ⊆ C(τ_low).
LEVEL_NAMES = ["low", "med", "high"]
LEVEL_TAU   = {"low": TAU_LOW, "med": TAU_MED, "high": TAU_HIGH}

# ── Step 3 tolerances ────────────────────────────────────────────────────────
DEFAULT_TOLERANCE_PX = 0   # δ: boundary-match tolerance (spec: 2 px)

# δ_margin = c_margin × sqrt(H×W / N_max)  (spec Step 1h)
# At N_max=1500: sqrt(384×384/1500) ≈ 9.92 px.
# c_margin=0.75 → δ_margin ≈ 7 px (default; calibrate on 20-event set per spec).
DEFAULT_C_MARGIN  = 0.75
DEFAULT_MARGIN_PX = 7      # pre-computed at N_max=1500 with DEFAULT_C_MARGIN
MARGIN_SWEEP_VALUES = [5, 7, 10]   # c_margin ≈ {0.50, 0.75, 1.00} sensitivity sweep

# Finest N in the sweep — used to derive A_min = H*W / N_max (Step 1e)
# and δ_margin anchor (Step 1h).
DEFAULT_N_MAX_SWEEP = 1500

# Overlay colours for nested boundary panel (Figure 1, row 0, col 2)
LEVEL_BOUNDARY_COLORS = {
    "low":  (1.00, 0.90, 0.00),   # yellow
    "med":  (1.00, 0.20, 0.20),   # red
    "high": (0.95, 0.95, 0.95),   # near-white
}

# Optional DEM support (gracefully degraded, as in the original script)
try:
    import pystac_client
    import planetary_computer
    import rioxarray
    from rioxarray.merge import merge_arrays
    from scipy.interpolate import RegularGridInterpolator
    import cartopy.crs as ccrs
    HAS_DEM_LIBS = True
except ImportError:
    HAS_DEM_LIBS = False


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (same conventions as Visualize_sevir_with_superpixel_fused.py)
# ─────────────────────────────────────────────────────────────────────────────

def _get_channel_path(event_id, img_type, catalog):
    rows = catalog[(catalog["id"] == event_id) & (catalog["img_type"] == img_type)]
    if rows.empty:
        return None
    return os.path.join(BASE_PATH, img_type, os.path.basename(rows.iloc[0]["file_name"]))


def _read_hdf5(path, event_id, img_type):
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
    log.warning(f"  Event {event_id} not found inside {path}")
    return None


def _load_single_channel(event_id, ch, catalog, tgt_shape):
    path = _get_channel_path(event_id, ch, catalog)
    if path is None:
        log.warning(f"  {ch}: no catalog entry — skipping.")
        return None
    raw = _read_hdf5(path, event_id, ch)
    if raw is None:
        return None

    if raw.ndim == 3:
        arr = raw.transpose(2, 0, 1) if raw.shape[2] < raw.shape[0] else raw
    else:
        arr = raw

    T_ch, H_ch, W_ch = arr.shape
    tgt_h, tgt_w = tgt_shape
    if H_ch != tgt_h or W_ch != tgt_w:
        out = np.empty((T_ch, tgt_h, tgt_w), dtype=np.float32)
        for t in range(T_ch):
            out[t] = cv2.resize(arr[t].astype(np.float32), (tgt_w, tgt_h),
                                interpolation=cv2.INTER_CUBIC)
        arr = out
    return arr.astype(np.float32)


def load_channels(event_id, catalog):
    """Load raw VIL, IR107, IR069 (T,H,W) float32, plus the lon/lat extent."""
    meta_rows = catalog[catalog["id"] == event_id]
    if meta_rows.empty:
        log.error(f"Event {event_id} not found in catalog.")
        return {}, []
    extent_row = meta_rows.iloc[0]
    extent = [extent_row["llcrnrlon"], extent_row["urcrnrlon"],
              extent_row["llcrnrlat"], extent_row["urcrnrlat"]]

    data = {}
    tgt_shape = (384, 384)
    vil = _load_single_channel(event_id, "vil", catalog, tgt_shape)
    if vil is not None:
        data["vil"] = vil
        tgt_shape = (vil.shape[1], vil.shape[2])
    for ch in ["ir107", "ir069"]:
        arr = _load_single_channel(event_id, ch, catalog, tgt_shape)
        if arr is not None:
            data[ch] = arr
    return data, extent


def build_fused_cube_frame0(data):
    """
    Per-event, per-channel min-max [0,1] normalisation (Preliminary section
    of blocks_technical_v2.md) — used ONLY for SLIC feature construction,
    never for the physical-unit connected-component reference.

    Returns (H, W, C) float32 feature cube for frame t=0.
    """
    fused_stack = []
    for ch in FUSION_CHANNELS:
        if ch in data:
            arr0 = data[ch][0]
            mn, mx = arr0.min(), arr0.max()
            norm = (arr0 - mn) / (mx - mn + 1e-6)
            fused_stack.append(norm)
    if not fused_stack:
        raise ValueError("No fusion channels available for SLIC.")
    return np.stack(fused_stack, axis=-1).astype(np.float32)   # (H, W, C)


def fetch_dem_norm(extent, nx=384, ny=384):
    """Optional DEM channel, gracefully degraded to zeros if unavailable."""
    if not HAS_DEM_LIBS:
        return np.zeros((ny, nx), dtype=np.float32)
    try:
        proj = ccrs.LambertConformal(
            central_longitude=-98.0, central_latitude=38.0,
            standard_parallels=(30.0, 60.0),
            globe=ccrs.Globe(semimajor_axis=6370000, semiminor_axis=6370000),
        )
        src = ccrs.PlateCarree()
        x0, y0 = proj.transform_point(extent[0], extent[2], src)
        x1, y1 = proj.transform_point(extent[1], extent[3], src)
        xv, yv = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
        grid = src.transform_points(proj, xv, yv)
        tgt_lons, tgt_lats = grid[..., 0], grid[..., 1]

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

        das = []
        for it in items:
            da = rioxarray.open_rasterio(it.assets["data"].href, lock=False).squeeze()
            da = da.rio.clip_box(minx=extent[0]-0.1, miny=extent[2]-0.1,
                                 maxx=extent[1]+0.1, maxy=extent[3]+0.1)
            das.append(da.coarsen(x=3, y=3, boundary="trim").mean())
        merged = merge_arrays(das) if len(das) > 1 else das[0]
        lons, lats, vals = merged.x.values, merged.y.values, merged.values
        if lats[0] > lats[-1]:
            lats, vals = lats[::-1], vals[::-1, :]
        interp = RegularGridInterpolator((lats, lons), vals.astype(np.float32),
                                         method="linear", bounds_error=False, fill_value=0.0)
        dem_raw = interp(np.column_stack((tgt_lats.ravel(), tgt_lons.ravel()))
                         ).reshape(ny, nx).astype(np.float32)
        return (dem_raw - dem_raw.min()) / (dem_raw.max() - dem_raw.min() + 1e-6)
    except Exception as exc:
        log.warning(f"DEM fetch failed ({exc}) — using flat terrain.")
        return np.zeros((ny, nx), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# B1 STEP 1 — Reference storm cell construction
# ─────────────────────────────────────────────────────────────────────────────

def decode_vil_physical(vil_raw_frame0: np.ndarray) -> np.ndarray:
    """raw float32 (already cast from uint8) -> physical kg/m^2."""
    return np.clip(vil_raw_frame0, 0, 255) / 255.0 * VIL_MAX_PHYS


# ── Step 1b–1f for a pre-smoothed field ─────────────────────────────────────
def _build_from_smooth(vil_smooth, tau_cell, H, W, n_max_sweep,
                       split_merged, seed_min_distance):
    """
    Steps 1b–1f given an already-smoothed VIL field (Step 1a done externally).
    Called once per threshold level inside build_all_threshold_cells.

    Returns (C, M, areas):
        C     : (H,W) int32 label map  (0=background, 1..M=cell id)
        M     : number of retained cells after Step 1e
        areas : dict {cell_id: pixel area}
    """
    # 1b — Absolute physical threshold (NOT relative to event dynamic range)
    B_VIL = vil_smooth > tau_cell

    # 1c — Hole filling (removes interior dips from retrieval noise)
    B_filled = binary_fill_holes(B_VIL)

    # 1d — 8-connected component labelling
    #      8-connectivity: diagonal-only adjacency is a pixel-grid artefact,
    #      not evidence of two distinct cores (spec §1d).
    structure_8 = np.ones((3, 3), dtype=np.int32)
    C_raw, M_raw = cc_label(B_filled, structure=structure_8)

    # 1e — Minimum area tied to finest superpixel resolution in the sweep
    #      (not an arbitrary pixel count — see spec §1e for derivation)
    A_min = (H * W) / n_max_sweep   # = H*W / N_max(sweep)

    C = np.zeros_like(C_raw)
    areas = {}
    next_id = 1
    for k in range(1, M_raw + 1):
        mask_k = (C_raw == k)
        area_k = int(mask_k.sum())
        if area_k >= A_min:
            C[mask_k] = next_id
            areas[next_id] = area_k
            next_id += 1
    M = next_id - 1

    # 1g — Optional watershed-from-maxima split of residual merged cells
    #      (secondary mechanism — most merging resolved by Step 1f nesting)
    if split_merged and M > 0:
        C, areas = _split_merged_cells(C, vil_smooth, TAU_HIGH, seed_min_distance)
        M = int(C.max())

    return C, M, areas


# ── Step 1g — SCIT-inspired multi-threshold nested reference system ───────────
def build_all_threshold_cells(vil_phys, sigma_cell, n_max_sweep,
                               split_merged=False, seed_min_distance=5):
    """
    Step 1f: apply Steps 1a–1e independently at τ_low, τ_med, τ_high
    using the *same* σ_cell for all levels (SCIT-inspired nested ladder,
    spec §1f).

    Produces label maps C_k at each level.  The union boundary
    ∂C_nested = ∂C_low ∪ ∂C_med ∪ ∂C_high is assembled in
    compute_boundary_f1_all_stages (Step 3).

    Step 1g (optional watershed split of residual merged cells) is applied
    inside _build_from_smooth when split_merged=True.

    Step 1h (single ROI Ω_C = dilate(C_low, δ_margin)) is applied in
    compute_boundary_f1_all_stages — C_low is the spatial footprint that
    contains all nested levels by construction.

    Returns
    -------
    vil_smooth : (H,W) float32
        σ_cell-smoothed VIL field, shared across all threshold levels.

    level_data : dict  keyed by 'low', 'med', 'high'
        Each entry contains:
          C         : (H,W) int32   label map (0=bg, 1..M=cell id)
          M         : int            number of retained cells after A_min filter
          areas     : dict           {cell_id: pixel_area}
          boundary  : (H,W) bool    ∂C(τ_k) — 4-connected boundary of C>0
          cell_mask : (H,W) bool    (C > 0) — spatial footprint of level k
          active    : bool          True if M > 0

    Notes
    -----
    - Inactive levels (active=False) are excluded from the union boundary and
      from per-cell mIoU; report fraction of such events per class (spec §1h).
    - Nested structure: C(τ_high) ⊆ C(τ_med) ⊆ C(τ_low) by construction,
      so C_low is the correct ROI source (largest spatial footprint).
    """
    H, W = vil_phys.shape

    # Step 1a — single shared pre-smoothing (σ_cell invariant across thresholds)
    vil_smooth = gaussian_filter(
        vil_phys.astype(np.float64), sigma=sigma_cell
    ).astype(np.float32)

    level_data = {}
    for level in LEVEL_NAMES:
        tau = LEVEL_TAU[level]
        C, M, areas = _build_from_smooth(
            vil_smooth, tau, H, W, n_max_sweep, split_merged, seed_min_distance
        )
        boundary  = extract_boundary(C, require_positive=True)
        cell_mask = (C > 0)
        active    = (M > 0)
        level_data[level] = {
            "C": C, "M": M, "areas": areas,
            "boundary": boundary, "cell_mask": cell_mask, "active": active,
        }
        log.info(f"  τ_{level}={tau:.1f} kg/m²: M={M} cells, active={active}")

    return vil_smooth, level_data


# ── Retained for σ_cell calibration sweep (single-threshold, diagnostic only) ──
def build_reference_cells(
    vil_phys, tau_cell, sigma_cell=1.0,
    n_max_sweep=DEFAULT_N_MAX_SWEEP, split_merged=False,
    seed_min_distance=5, tau_seed=TAU_HIGH,
):
    """
    Single-threshold Steps 1a–1f pipeline.
    Kept for use by sigma_stabilisation_sweep (calibration at τ_med only).
    Production code calls build_all_threshold_cells instead.
    """
    H, W = vil_phys.shape
    vil_smooth = gaussian_filter(
        vil_phys.astype(np.float64), sigma=sigma_cell
    ).astype(np.float32)
    return _build_from_smooth(
        vil_smooth, tau_cell, H, W, n_max_sweep, split_merged, seed_min_distance
    )


def sigma_stabilisation_sweep(vil_phys, tau_cell, n_max_sweep,
                               sigmas=(0.5, 0.75, 1.0, 1.5, 2.0)):
    """
    Step 1a calibration diagnostic: number of retained cells M(σ_cell)
    across a grid. Select the smallest σ where M stabilises
    (changes < 5% per 0.25px step). Always run at τ_med per spec.
    """
    print("\n  sigma_cell stabilisation sweep (at τ_med = {:.1f} kg/m²):".format(TAU_MED))
    print("  " + "-" * 44)
    prev_M = None
    for s in sigmas:
        C, M, _ = build_reference_cells(vil_phys, tau_cell, sigma_cell=s,
                                         n_max_sweep=n_max_sweep, split_merged=False)
        delta = "" if prev_M is None else f"  (Δ={100*(M-prev_M)/max(prev_M,1):+.1f}%)"
        print(f"    sigma={s:4.2f} px  ->  M={M:3d} cells{delta}")
        prev_M = M
    print("  " + "-" * 44)
    print("  Choose the smallest sigma where successive M values differ by < 5%.\n")


def _split_merged_cells(C, vil_smooth, tau_seed, min_distance):
    """
    Step 1g (secondary mechanism): watershed-from-maxima split of residual
    merged convective cells that nesting (Step 1f) alone does not resolve.
    For each retained component, detect local maxima exceeding τ_high;
    if more than one is found, split via watershed on the inverted smooth field.
    """
    H, W = C.shape
    C_out = np.zeros_like(C)
    next_id = 1
    areas_out = {}

    for k in range(1, int(C.max()) + 1):
        mask_k = (C == k)
        if not mask_k.any():
            continue

        coords = peak_local_max(
            vil_smooth, min_distance=min_distance,
            threshold_abs=tau_seed, labels=mask_k.astype(np.int32),
        )

        if len(coords) <= 1:
            C_out[mask_k] = next_id
            areas_out[next_id] = int(mask_k.sum())
            next_id += 1
            continue

        markers = np.zeros((H, W), dtype=np.int32)
        for i, (ry, rx) in enumerate(coords, start=1):
            markers[ry, rx] = i

        split_labels = watershed(-vil_smooth, markers=markers, mask=mask_k)
        for sub_id in np.unique(split_labels):
            if sub_id == 0:
                continue
            sub_mask = (split_labels == sub_id)
            C_out[sub_mask] = next_id
            areas_out[next_id] = int(sub_mask.sum())
            next_id += 1

    return C_out, areas_out


# ─────────────────────────────────────────────────────────────────────────────
# SLIC (t=0 branch only — standard multi-channel SLIC)
# ─────────────────────────────────────────────────────────────────────────────

def compute_slic_t0(fused_frame0, n_segments, dem_norm=None,
                    compactness=COMPACTNESS, lambda_z=ELEVATION_LAMBDA):
    """
    Standard SLIC on the (H,W,C) fused cube at t=0, optionally with a
    DEM channel appended — exactly the `if self.prev_labels is None`
    branch of TemporalSLIC_DEM.segment() in the original script.
    """
    if dem_norm is not None:
        dem_ch = (dem_norm * lambda_z)[:, :, np.newaxis]
        feature_cube = np.concatenate([fused_frame0, dem_ch], axis=-1).astype(np.float64)
    else:
        feature_cube = fused_frame0.astype(np.float64)

    labels = skimage_slic(
        feature_cube, n_segments=n_segments, compactness=compactness,
        start_label=1, enforce_connectivity=True, channel_axis=-1,
    )
    return labels.astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# B1 STEPS 2–3d — Boundary extraction and ROI-masked F1
# ─────────────────────────────────────────────────────────────────────────────

def extract_boundary(label_map, require_positive=False):
    """
    4-connected boundary pixel mask.

    require_positive=True  : used for cell map C — boundary pixels must be
                              foreground (C>0); a pixel is on the boundary if
                              any 4-neighbour has a different label (including
                              label 0 = background).  [spec §1d: 4-connected
                              boundary-pixel convention, distinct from 8-connected
                              blob-identity convention]
    require_positive=False : used for SLIC — every pixel is foreground, so
                              simply test neighbour-label inequality.
    """
    H, W = label_map.shape

    up    = np.roll(label_map,  1, axis=0)
    down  = np.roll(label_map, -1, axis=0)
    left  = np.roll(label_map,  1, axis=1)
    right = np.roll(label_map, -1, axis=1)

    diff = ((label_map != up) | (label_map != down) |
            (label_map != left) | (label_map != right))

    # Correct edge-pixel wrap-around introduced by np.roll
    diff[0, :]   = ((label_map[0,  :] != down[0,  :]) |
                    (label_map[0,  :] != left[0,  :]) |
                    (label_map[0,  :] != right[0, :]))
    diff[-1, :]  = ((label_map[-1, :] != up[-1,  :]) |
                    (label_map[-1, :] != left[-1, :]) |
                    (label_map[-1, :] != right[-1,:]))
    diff[:, 0]   = ((label_map[:,  0] != up[:,   0]) |
                    (label_map[:,  0] != down[:,  0]) |
                    (label_map[:,  0] != right[:, 0]))
    diff[:, -1]  = ((label_map[:, -1] != up[:,  -1]) |
                    (label_map[:, -1] != down[:, -1]) |
                    (label_map[:, -1] != left[:, -1]))

    boundary = diff
    if require_positive:
        boundary = boundary & (label_map > 0)
    return boundary


def boundary_f1(boundary_cells, boundary_slic, cell_mask,
                tolerance_px=DEFAULT_TOLERANCE_PX, margin_px=DEFAULT_MARGIN_PX,
                use_roi=True):
    """
    Precision / Recall / F1 at a given pixel tolerance.

    use_roi=True  (Stages 1 & 2): denominator = |∂SLIC ∩ Ω_C| where
                  Ω_C = dilate(cell_mask, δ_margin).
    use_roi=False (Stage 0):      denominator = |∂SLIC| (global — the
                  "original, flawed" baseline the spec ablates against).

    Parameters
    ----------
    boundary_cells : (H,W) bool  — ∂C to match against (flat or nested union)
    boundary_slic  : (H,W) bool  — ∂SLIC
    cell_mask      : (H,W) bool  — source of Ω_C (use C_low for Steps 1 & 2)
    use_roi        : bool        — True for ROI-masked denominator
    """
    se_tol    = disk(tolerance_px)
    se_margin = disk(margin_px)

    dilated_cells = binary_dilation(boundary_cells, structure=se_tol)
    dilated_slic  = binary_dilation(boundary_slic,  structure=se_tol)

    if use_roi and cell_mask is not None:
        roi_mask          = binary_dilation(cell_mask, structure=se_margin)
        boundary_slic_roi = boundary_slic & roi_mask    # ∂SLIC_ROI (Step 1h / Step 3)
    else:
        roi_mask          = None
        boundary_slic_roi = boundary_slic               # global denominator

    n_denom = int(boundary_slic_roi.sum())
    n_cells = int(boundary_cells.sum())

    precision = float((boundary_slic & dilated_cells).sum()) / (n_denom + 1e-8)
    recall    = float((boundary_cells & dilated_slic).sum())  / (n_cells + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return float(precision), float(recall), float(f1), dilated_cells, dilated_slic, roi_mask


# ─────────────────────────────────────────────────────────────────────────────
# B1 STEP 3 — Three-stage ablation + per-cell mIoU + soft-argmax
# ─────────────────────────────────────────────────────────────────────────────

def compute_boundary_f1_all_stages(level_data, boundary_slic,
                                    tolerance_px=DEFAULT_TOLERANCE_PX,
                                    margin_px=DEFAULT_MARGIN_PX):
    """
    Step 3 — Boundary F1 with ROI-masked Precision and nested reference,
    plus explicit two-stage ablation (spec §Step 3).

    Stage 0 (original, flawed): Global Precision × flat ∂C_med
    Stage 1 (ROI fix only):     ROI Precision    × flat ∂C_med
    Stage 2 (primary):          ROI Precision    × ∂C_nested = ∂C_low ∪ ∂C_med ∪ ∂C_high

    The ROI is ALWAYS built from C_low (Step 1h):
        Ω_C = dilate(C_low, δ_margin)
    Nesting guarantees C_high ⊆ C_med ⊆ C_low, so C_low's footprint contains
    every nested level — no separate dilation per level is needed.

    The numerator of Precision_ROI does not need an explicit "∩ Ω_C"
    restriction: since δ_margin ≥ δ by construction, any ∂SLIC pixel within
    tolerance δ of ∂C_nested is automatically inside Ω_C (spec §Step 3).

    Parameters
    ----------
    level_data    : dict — output of build_all_threshold_cells
    boundary_slic : (H,W) bool — ∂SLIC at the current N
    tolerance_px  : int  — δ (spec default: 2 px)
    margin_px     : int  — δ_margin (default 7 px; must be > tolerance_px)

    Returns
    -------
    dict with:
      boundary_nested : (H,W) bool    ∂C_nested (union of all active levels)
      roi_mask        : (H,W) bool    Ω_C = dilate(C_low, δ_margin)
      dilated_nested  : (H,W) bool    dilate(∂C_nested, δ)
      dilated_slic    : (H,W) bool    dilate(∂SLIC, δ)
      precision_roi   : float   Stage 2 Precision_ROI
      recall          : float   Stage 2 Recall
      f1_nested       : float   Stage 2 F1 (primary metric)
      f1_stage0       : float   Stage 0 F1 (global P, flat ∂C_med)
      f1_stage1       : float   Stage 1 F1 (ROI P, flat ∂C_med)
      active          : bool    True if at least one level is active
      active_levels   : list[str]
      zero_cell       : bool    True if C_low is inactive (exclude from aggregate)
    """
    assert margin_px > tolerance_px, (
        f"Spec constraint violated: δ_margin={margin_px} must be > δ={tolerance_px}"
    )

    active_levels = [lv for lv in LEVEL_NAMES if level_data[lv]["active"]]

    # Zero-cell event: C_low inactive → Ω_C empty → undefined (spec §1h)
    if not level_data["low"]["active"]:
        return {
            "boundary_nested": np.zeros(boundary_slic.shape, dtype=bool),
            "roi_mask": np.zeros(boundary_slic.shape, dtype=bool),
            "dilated_nested": np.zeros(boundary_slic.shape, dtype=bool),
            "dilated_slic": np.zeros(boundary_slic.shape, dtype=bool),
            "precision_roi": 0.0, "recall": 0.0,
            "f1_nested": 0.0, "f1_stage0": 0.0, "f1_stage1": 0.0,
            "active": False, "active_levels": [], "zero_cell": True,
        }

    # ── ∂C_nested: union boundary (Step 1f) ─────────────────────────────────
    boundary_nested = np.zeros(boundary_slic.shape, dtype=bool)
    for lv in active_levels:
        boundary_nested |= level_data[lv]["boundary"]

    # ── Single ROI from C_low (Step 1h) ─────────────────────────────────────
    se_tol    = disk(tolerance_px)
    se_margin = disk(margin_px)
    cell_mask_low = level_data["low"]["cell_mask"]
    roi_mask  = binary_dilation(cell_mask_low, structure=se_margin)

    boundary_slic_roi = boundary_slic & roi_mask      # ∂SLIC_ROI
    dilated_nested    = binary_dilation(boundary_nested, structure=se_tol)
    dilated_slic      = binary_dilation(boundary_slic,   structure=se_tol)

    n_slic_roi = int(boundary_slic_roi.sum())
    n_nested   = int(boundary_nested.sum())

    # ── Stage 2 (primary): ROI Precision × ∂C_nested ────────────────────────
    p2 = float((boundary_slic & dilated_nested).sum()) / (n_slic_roi + 1e-8)
    r2 = float((boundary_nested & dilated_slic).sum())  / (n_nested  + 1e-8)
    f1_nested = 2 * p2 * r2 / (p2 + r2 + 1e-8)

    # ── Stages 0 & 1: flat ∂C_med as reference ───────────────────────────────
    # Fall back to ∂C_low if τ_med has no cells (rare for non-QUIESCENT events)
    ref_level = "med" if level_data["med"]["active"] else "low"
    boundary_flat = level_data[ref_level]["boundary"]
    dilated_flat  = binary_dilation(boundary_flat, structure=se_tol)
    n_flat        = int(boundary_flat.sum())
    n_slic_global = int(boundary_slic.sum())

    r_flat = float((boundary_flat & dilated_slic).sum()) / (n_flat + 1e-8)

    # Stage 0: Global Precision (no ROI) × flat ∂C
    p0       = float((boundary_slic & dilated_flat).sum()) / (n_slic_global + 1e-8)
    f1_stage0 = 2 * p0 * r_flat / (p0 + r_flat + 1e-8)

    # Stage 1: ROI Precision × flat ∂C
    p1       = float((boundary_slic & dilated_flat).sum()) / (n_slic_roi + 1e-8)
    f1_stage1 = 2 * p1 * r_flat / (p1 + r_flat + 1e-8)

    return {
        "boundary_nested": boundary_nested,
        "roi_mask": roi_mask,
        "dilated_nested": dilated_nested,
        "dilated_slic": dilated_slic,
        "precision_roi": p2, "recall": r2,
        "f1_nested": f1_nested,
        "f1_stage0": f1_stage0,
        "f1_stage1": f1_stage1,
        "active": True,
        "active_levels": active_levels,
        "zero_cell": False,
    }


def compute_per_cell_iou(level_data, slic_labels):
    """
    Per-cell IoU with one-to-one bipartite matching (Hungarian algorithm).

    For each nested level k ∈ {low, med, high} (spec §B1, per-cell IoU section):
        - Candidate SLIC superpixels: those overlapping at least one CC instance.
        - Build the (M_k × |candidates|) IoU matrix.
        - Solve one-to-one assignment via Hungarian algorithm.
        - Unmatched instances (when M_k > |candidates|) → IoU = 0, NOT dropped.
        - mIoU(N; τ_k)          = (1/M_k) Σ_i IoU(C_i)                  [unweighted]
        - mIoU_weighted(N; τ_k) = Σ_i IoU(C_i) × Area(C_i)/ΣArea(C_j)   [area-weighted]

    Why the area-weighted variant is the primary metric: an unweighted mean
    treats one 400px hail core that SLIC totally fails on (IoU=0) and three
    10px noise cores that happen to get wrapped (IoU≈1) as a 75% success
    rate — the metric is dominated by instance *count*, not by how much of
    the meteorologically significant area is actually well-segmented. Area
    weighting makes a failure on the dominant core dominate the score
    instead of being diluted by small instances, which sharpens the N-sweep
    peak instead of damping its variance. ΣArea(C_j) is the sum over ALL
    M_k instances (matched and unmatched) at this threshold, so an unmatched
    massive core still correctly craters the score via its large weight × 0.

    No Ω_C restriction needed: bipartite matching only involves superpixels
    overlapping at least one instance — background superpixels are excluded
    by the structural property of instance-matching (spec §per-cell IoU).

    Returns
    -------
    dict keyed by level name:
        miou          : float        unweighted mean IoU across all M_k instances (0 if M_k=0)
        miou_weighted : float        area-weighted mean IoU (primary metric; 0 if M_k=0)
        n_instances   : int          M_k
        instance_ious : list[float]  per-instance IoU
        instance_areas: list[float]  per-instance area (px), aligned with instance_ious
        active        : bool         True if M_k > 0
    """
    results = {}
    for level in LEVEL_NAMES:
        ld = level_data[level]
        if not ld["active"]:
            results[level] = {"miou": 0.0, "miou_weighted": 0.0, "n_instances": 0,
                               "instance_ious": [], "instance_areas": [], "active": False}
            continue

        C = ld["C"]
        M = ld["M"]

        # Per-instance physical area (px), needed for area weighting regardless
        # of whether any SLIC superpixel overlaps the instance.
        instance_areas = np.array(
            [int((C == cell_id).sum()) for cell_id in range(1, M + 1)],
            dtype=np.float64,
        )

        # Candidate SLIC superpixels: any that overlap at least one cell pixel
        cell_pixels   = (C > 0)
        candidate_ids = np.unique(slic_labels[cell_pixels])
        candidate_ids = candidate_ids[candidate_ids > 0]

        if len(candidate_ids) == 0:
            # No overlapping superpixels → all unmatched (IoU=0 for everyone,
            # so weighted and unweighted means are both 0 regardless of areas)
            results[level] = {"miou": 0.0, "miou_weighted": 0.0, "n_instances": M,
                               "instance_ious": [0.0] * M,
                               "instance_areas": instance_areas.tolist(), "active": True}
            continue

        # Build IoU matrix (M × |candidates|)
        iou_matrix = np.zeros((M, len(candidate_ids)), dtype=np.float32)
        for i, cell_id in enumerate(range(1, M + 1)):
            cell_mask = (C == cell_id)
            for j, slic_id in enumerate(candidate_ids):
                slic_mask    = (slic_labels == slic_id)
                intersection = int((cell_mask & slic_mask).sum())
                union        = int((cell_mask | slic_mask).sum())
                iou_matrix[i, j] = intersection / (union + 1e-8)

        # Hungarian matching (scipy minimises → negate IoU)
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        # Per-instance IoU: unmatched rows stay at 0 (spec: must NOT be dropped)
        instance_ious = np.zeros(M, dtype=np.float32)
        for r, c in zip(row_ind, col_ind):
            instance_ious[r] = iou_matrix[r, c]

        miou = float(instance_ious.mean())

        total_area = float(instance_areas.sum())
        if total_area > 0:
            miou_weighted = float(
                np.sum(instance_ious.astype(np.float64) * instance_areas) / total_area
            )
        else:
            miou_weighted = 0.0

        results[level] = {
            "miou": miou,
            "miou_weighted": miou_weighted,
            "n_instances": M,
            "instance_ious": instance_ious.tolist(),
            "instance_areas": instance_areas.tolist(),
            "active": True,
        }

    return results


def soft_argmax_n(values, n_list, threshold=0.95):
    """
    N_soft = min{ N : value(N) ≥ threshold × max_N value(N) }

    Used to characterise both F1_nested (→ N_F1, upper bound) and
    mIoU(N; τ_high) (→ N_IoU, lower bound) without over-interpreting
    noise within a plateau (spec §B1 soft-argmax, eq. 5.14 analogue).

    Returns the first N meeting the threshold, or None if all values ≤ 1e-6.
    """
    if not values or all(v < 1e-6 for v in values):
        return None
    peak    = max(values)
    cutoff  = threshold * peak
    for n, v in zip(n_list, values):
        if v >= cutoff:
            return n
    return n_list[int(np.argmax(values))]


# ─────────────────────────────────────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cell_colormap(M):
    """Distinct colours per cell id, with id=0 forced to black (background)."""
    if M == 0:
        return ListedColormap(["black"])
    base = plt.get_cmap("tab20")(np.linspace(0, 1, max(M, 1)))
    colors = np.vstack([[0, 0, 0, 1], base])
    return ListedColormap(colors)


def render_match_miss_panel(vil_phys, boundary_ref, boundary_slic,
                             dilated_ref, dilated_slic, roi_mask):
    """
    Diagnostic overlay attributing Precision/Recall failures to pixels.
    Works with either a single-level boundary (Stage 0/1) or ∂C_nested (Stage 2).

    Colour key
    ----------
    green      — TP: SLIC boundary within tolerance of reference boundary
    orange     — FP inside Ω_C: SLIC boundary in evaluation zone but not
                 near ∂C_ref → counted in denominator, hurts Precision_ROI
    dim ochre  — FP outside Ω_C: SLIC boundary in clear-air region →
                 NOT counted (Lagrangian tiling artefact; ROI masking removes it)
    magenta    — FN: reference boundary pixel not covered by any SLIC boundary
    white      — outline of the ROI evaluation zone Ω_C (spatial context)

    Parameters
    ----------
    boundary_ref : (H,W) bool — ∂C reference (flat ∂C_med or union ∂C_nested)
    dilated_ref  : (H,W) bool — dilate(boundary_ref, δ)
    roi_mask     : (H,W) bool — Ω_C = dilate(C_low, δ_margin)
    """
    H, W = vil_phys.shape
    panel = np.tile((vil_phys / (vil_phys.max() + 1e-6))[:, :, None], (1, 1, 3)) * 0.35

    tp_mask         = boundary_slic & dilated_ref
    fp_counted_mask = boundary_slic & ~dilated_ref & roi_mask    # inside Ω_C
    fp_ignored_mask = boundary_slic & ~dilated_ref & ~roi_mask   # outside Ω_C
    fn_mask         = boundary_ref  & ~dilated_slic

    roi_boundary = extract_boundary(roi_mask.astype(np.int32), require_positive=True)

    panel[fp_ignored_mask] = [0.55, 0.45, 0.10]   # dim ochre  — not counted
    panel[tp_mask]         = [0.00, 0.90, 0.00]   # green      — true positives
    panel[fp_counted_mask] = [1.00, 0.55, 0.00]   # orange     — counted FPs
    panel[fn_mask]         = [0.95, 0.00, 0.85]   # magenta    — false negatives
    panel[roi_boundary]    = [0.90, 0.90, 0.90]   # white      — Ω_C zone outline
    return np.clip(panel, 0, 1)


def render_nested_boundary_overlay(vil_phys, level_data):
    """
    Overlay all active threshold boundary sets on the VIL background.
    Colour key: yellow=τ_low, red=τ_med, white=τ_high
    Illustrates the nested structure: C(τ_high) ⊆ C(τ_med) ⊆ C(τ_low).
    """
    vmax = max(float(vil_phys.max()), 1.0)
    rgb  = plt.get_cmap("jet")(np.clip(vil_phys / vmax, 0, 1))[:, :, :3].copy()
    rgb  = (rgb * 0.55).astype(np.float64)    # dim background for contrast
    # Paint in ascending intensity order so higher levels overwrite lower ones
    for level in LEVEL_NAMES:
        if not level_data[level]["active"]:
            continue
        bnd   = level_data[level]["boundary"]
        color = LEVEL_BOUNDARY_COLORS[level]
        rgb[bnd] = color
    return np.clip(rgb, 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main diagnostic figure
# ─────────────────────────────────────────────────────────────────────────────

def visualize_event(event_id, vil_phys, vil_smooth, level_data, fused_frame0,
                    dem_norm, n_list, sigma_cell, tolerance_px,
                    split_merged, margin_px, out_dir):
    """
    Figure 1 layout
    ---------------
    Row 0 — Reference panels (shared):
      [0,0] VIL (kg/m²) with threshold contour lines at τ_low, τ_med, τ_high
      [0,1] τ_med reference cells (coloured by cell id); cell counts per level
      [0,2] Nested boundary overlay: τ_low=yellow, τ_med=red, τ_high=white

    Row i+1 — Per N value:
      [i+1,0] SLIC boundaries (cyan) on VIL
      [i+1,1] Match/miss panel (Stage 2: ∂C_nested, Ω_C from C_low)
      [i+1,2] Three-stage ablation table + per-cell mIoU per level

    Figure 2 (only when len(n_list) > 1) — metric curves vs N:
      Top panel    : F1_nested (Stage 2) vs N; also Stage 0 and Stage 1 for ablation
      Middle panel : per-cell mIoU(N; τ_k) for low, med, high
      Bottom panel : validation interval N_IoU(τ_high) ≤ N* ≤ N_F1
    """
    vmax_vil = max(float(vil_phys.max()), 1.0)
    rgb_vil  = plt.get_cmap("jet")(np.clip(vil_phys / vmax_vil, 0, 1))[:, :, :3]

    n_rows = 1 + len(n_list)
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, 3)

    # ── Row 0: Reference panels ────────────────────────────────────────────
    ax = axes[0, 0]
    ax.imshow(vil_phys, cmap="jet", vmin=0, vmax=vmax_vil, origin="lower")
    contour_spec = [
        (TAU_LOW,  "yellow", f"τ_low={TAU_LOW}"),
        (TAU_MED,  "red",    f"τ_med={TAU_MED}"),
        (TAU_HIGH, "white",  f"τ_high={TAU_HIGH}"),
    ]
    for tau_v, color, label in contour_spec:
        if vil_phys.max() > tau_v:
            cs = ax.contour(vil_phys, levels=[tau_v], colors=[color],
                            linewidths=0.9, origin="lower")
            ax.clabel(cs, fmt={tau_v: f"{tau_v:.1f}"}, fontsize=6, inline=True)
    ax.set_title(
        f"VIL (kg/m²)\nContours: yellow={TAU_LOW}  red={TAU_MED}  white={TAU_HIGH} kg/m²",
        fontsize=9,
    )
    ax.axis("off")

    M_med = level_data["med"]["M"]
    ax = axes[0, 1]
    cmap_cells = _cell_colormap(M_med)
    ax.imshow(level_data["med"]["C"], cmap=cmap_cells, origin="lower",
              vmin=0, vmax=max(M_med, 1))
    cell_count_str = "  ".join(
        f"τ_{lv}:M={level_data[lv]['M']}" for lv in LEVEL_NAMES
    )
    ax.set_title(
        f"Reference cells @ τ_med (primary)\n"
        f"{cell_count_str}\n"
        f"σ_cell={sigma_cell}px  split={split_merged}  A_min=H×W/{DEFAULT_N_MAX_SWEEP}",
        fontsize=9,
    )
    ax.axis("off")

    ax = axes[0, 2]
    ax.imshow(render_nested_boundary_overlay(vil_phys, level_data), origin="lower")
    ax.set_title(
        "Nested boundary overlay (Step 1f)\n"
        "yellow=τ_low  red=τ_med  white=τ_high\n"
        "∂C_nested = ∂C_low ∪ ∂C_med ∪ ∂C_high",
        fontsize=9,
    )
    ax.axis("off")

    # ── Per-N rows ─────────────────────────────────────────────────────────
    results_per_n = []

    for i, n_req in enumerate(n_list):
        row = i + 1
        slic_labels   = compute_slic_t0(fused_frame0, n_req, dem_norm=dem_norm)
        actual_n      = len(np.unique(slic_labels))
        boundary_slic = extract_boundary(slic_labels, require_positive=False)

        # Step 3: three-stage ablation (primary metric, used for the
        # Match/Miss panel and ablation table rendered below)
        stages = compute_boundary_f1_all_stages(
            level_data, boundary_slic,
            tolerance_px=tolerance_px, margin_px=margin_px,
        )

        # Step 3e: multi-threshold composite (Scheme G/P/F) — this is the
        # dict consumed by the summary table, convergence check, Scheme-G
        # printout, and _plot_f1_vs_n downstream
        mt = compute_multi_threshold_f1(
            level_data, vil_smooth, boundary_slic,
            tolerance_px=tolerance_px, margin_px=margin_px,
        )

        # Per-cell mIoU (instance-scale metric)
        miou_dict = compute_per_cell_iou(level_data, slic_labels)

        results_per_n.append((n_req, actual_n, mt, miou_dict))

        # [row, 0] SLIC overlay on VIL
        slic_overlay = mark_boundaries(rgb_vil, slic_labels, color=(0, 0.9, 1.0), mode="thick")
        axes[row, 0].imshow(slic_overlay, origin="lower")
        axes[row, 0].set_title(f"SLIC  N={n_req}  (actual={actual_n})", fontsize=10)
        axes[row, 0].axis("off")

        # [row, 1] Match/miss panel — Stage 2 (∂C_nested, single Ω_C from C_low)
        if stages["active"] and not stages["zero_cell"]:
            panel = render_match_miss_panel(
                vil_phys,
                stages["boundary_nested"],
                boundary_slic,
                stages["dilated_nested"],
                stages["dilated_slic"],
                stages["roi_mask"],
            )
        else:
            panel = (rgb_vil * 0.4).copy()
            panel[boundary_slic] = [0.0, 0.9, 1.0]

        axes[row, 1].imshow(panel, origin="lower")
        axes[row, 1].set_title(
            f"Match/miss — Stage 2: ∂C_nested  (δ={tolerance_px}px, Ω_margin={margin_px}px)\n"
            f"P_ROI={stages['precision_roi']:.3f}  R={stages['recall']:.3f}  "
            f"F1_nested={stages['f1_nested']:.3f}",
            fontsize=9,
        )
        axes[row, 1].axis("off")

        # [row, 2] Three-stage ablation + per-cell mIoU table
        axes[row, 2].axis("off")
        lines = [
            f"N={n_req}  actual={actual_n}  δ={tolerance_px}px  δ_margin={margin_px}px",
            "",
            "── Three-stage ablation (Step 3) ──────────────────",
            f"  Stage 0 (global P, flat ∂C_med): F1={stages['f1_stage0']:.3f}",
            f"  Stage 1 (ROI P,  flat ∂C_med):   F1={stages['f1_stage1']:.3f}",
            f"  Stage 2 (ROI P,  ∂C_nested) [★]: F1={stages['f1_nested']:.3f}",
            f"     Precision_ROI={stages['precision_roi']:.3f}  Recall={stages['recall']:.3f}",
            "",
        ]
        if stages["zero_cell"]:
            lines.append("  [zero-cell event: C_low inactive, excluded from aggregate]")
        else:
            lines.append(f"  Active levels: {', '.join(stages['active_levels'])}")
            bnd_sizes = {lv: int(level_data[lv]["boundary"].sum())
                         for lv in stages["active_levels"]}
            lines.append(
                "  |∂C|: " + "  ".join(
                    f"τ_{lv}={bnd_sizes[lv]}" for lv in stages["active_levels"]
                )
            )

        lines += ["", "── Per-cell mIoU (Hungarian matching) ─────────────"]
        for lv in LEVEL_NAMES:
            md = miou_dict[lv]
            tau_v = LEVEL_TAU[lv]
            if md["active"]:
                lines.append(
                    f"  mIoU(τ_{lv}={tau_v:.1f}): {md['miou_weighted']:.3f} [area-wtd]  "
                    f"({md['miou']:.3f} unweighted)  [M={md['n_instances']} instances]"
                )
            else:
                lines.append(f"  mIoU(τ_{lv}={tau_v:.1f}): [inactive — M=0]")

        axes[row, 2].text(
            0.04, 0.97, "\n".join(lines),
            transform=axes[row, 2].transAxes,
            va="top", ha="left", fontsize=8.0,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#FFFFF0", alpha=0.95),
        )

    legend_text = (
        "Match/miss (Stage 2, ∂C_nested):  green=TP   orange=FP inside Ω_C (hurts P_ROI)   "
        "dim ochre=FP outside Ω_C (not counted)   magenta=FN   white=Ω_C boundary"
    )
    fig.text(0.5, 0.005, legend_text, ha="center", fontsize=8, style="italic")

    active_str = "  ".join(
        f"τ_{lv}:M={level_data[lv]['M']}"
        for lv in LEVEL_NAMES if level_data[lv]["active"]
    )
    fig.suptitle(
        f"B1 Diagnostic — Event {event_id}  [nested reference, union boundary]\n"
        f"Active levels: {active_str} | σ_cell={sigma_cell}px | "
        f"split={split_merged} | ROI δ_margin={margin_px}px",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    out_path = os.path.join(out_dir, f"{event_id}_b1_diagnostic_{n_list}.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved figure -> {out_path}")

    if len(n_list) > 1:
        _plot_f1_vs_n(event_id, results_per_n, margin_px, tolerance_px, out_dir)

    return results_per_n


LEVEL_PLOT_STYLE = {
    "low":  dict(color="goldenrod", marker="o", mec=None, label=f"τ_low = {TAU_LOW} kg/m²"),
    "med":  dict(color="red",       marker="s", mec=None, label=f"τ_med = {TAU_MED} kg/m²"),
    "high": dict(color="dimgray",   marker="^", mec="k",  label=f"τ_high = {TAU_HIGH} kg/m²"),
}


def _plot_per_level_panel(ax, ns, series, active_levels, ylabel, title,
                           mark_argmax=False, reference_series=None,
                           reference_note=None):
    """
    Shared helper: plot one metric (Precision_ROI / Recall / F1_bound / mIoU)
    for each active threshold level vs N, on a single axis.

    If reference_series is given (same shape as series), an unlabelled thin
    dotted line in the same per-level color is drawn underneath the primary
    curve — used to show the legacy unweighted mIoU mean alongside the
    area-weighted primary curve for direct visual comparison.
    """
    plotted_any = False
    for lv in LEVEL_NAMES:
        if lv not in active_levels:
            continue
        st = LEVEL_PLOT_STYLE[lv]
        vals = series[lv]
        if reference_series is not None:
            ax.plot(ns, reference_series[lv], ":", color=st["color"],
                     lw=1.2, alpha=0.5)
        ax.plot(ns, vals, "-", marker=st["marker"], color=st["color"],
                 lw=2.0, ms=7, markeredgecolor=st["mec"], label=st["label"])
        plotted_any = True
        if mark_argmax and any(v > 1e-6 for v in vals):
            idx = int(np.argmax(vals))
            ax.axvline(ns[idx], color=st["color"], lw=1.0, ls=":", alpha=0.8)

    ax.set_xlabel("N (superpixels)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 1.05)
    if plotted_any:
        ax.legend(fontsize=7.5)
    if reference_note:
        ax.text(0.02, 0.02, reference_note, transform=ax.transAxes,
                fontsize=7, color="gray", style="italic", va="bottom")
    ax.grid(alpha=0.3)


def _plot_f1_vs_n(event_id, results_per_n, margin_px, tolerance_px, out_dir):
    """
    Multi-panel metrics-vs-N figure (single-event mirror of the aggregate
    Fig B-1, extended to show every metric in the B1 spec, not just F1).

    Panels:
      [0,0] Precision_ROI(N; τ_k)      — per active threshold level
      [0,1] Recall(N; τ_k)             — per active threshold level
      [1,0] F1_bound(N; τ_k)           — per active threshold level
      [1,1] Per-cell mIoU(N; τ_k)      — area-weighted (primary), Hungarian-matched,
                                         per active level; legacy unweighted mean
                                         shown as a thin dotted reference
      [2,:] Composite F1_multi^G/P/F   — event-adaptive aggregate (Step 3e)

    Validation conditions visualised (spec §3e):
      - argmax F1_multi^G ≈ argmax F1_multi^P ≈ argmax F1_multi^F  (convergence)
      - F1_bound(N; τ_high) recall limits composite until N ≈ N*
      - F1_bound(N; τ_med)  may peak well below N* (single-threshold artefact)
      - mIoU_weighted(N; τ_high) gives the instance-scale lower bound N_IoU,
        sharpened relative to the unweighted mean so a single under-segmented
        massive core is not diluted by several well-matched tiny ones
    """
    ns = [r[0] for r in results_per_n]
    active_levels = results_per_n[0][2]["active_levels"]

    precision_roi = {lv: [r[2]["per_level"][lv]["precision"] for r in results_per_n]
                      for lv in LEVEL_NAMES}
    recall        = {lv: [r[2]["per_level"][lv]["recall"]    for r in results_per_n]
                      for lv in LEVEL_NAMES}
    f1_bound      = {lv: [r[2]["per_level"][lv]["f1"]         for r in results_per_n]
                      for lv in LEVEL_NAMES}
    miou_weighted = {lv: [r[3][lv]["miou_weighted"] for r in results_per_n]
                      for lv in LEVEL_NAMES}
    miou_unweighted = {lv: [r[3][lv]["miou"]         for r in results_per_n]
                      for lv in LEVEL_NAMES}

    f1_G = [r[2]["f1_multi_G"] for r in results_per_n]
    f1_P = [r[2]["f1_multi_P"] for r in results_per_n]
    f1_F = [r[2]["f1_multi_F"] for r in results_per_n]

    fig = plt.figure(figsize=(13, 14), constrained_layout=True)
    gs = fig.add_gridspec(3, 2)
    ax_p    = fig.add_subplot(gs[0, 0])
    ax_r    = fig.add_subplot(gs[0, 1])
    ax_f1   = fig.add_subplot(gs[1, 0])
    ax_miou = fig.add_subplot(gs[1, 1])
    ax_comp = fig.add_subplot(gs[2, :])

    _plot_per_level_panel(ax_p, ns, precision_roi, active_levels,
                          "Precision_ROI", "Precision_ROI vs N (per level)")
    _plot_per_level_panel(ax_r, ns, recall, active_levels,
                          "Recall", "Recall vs N (per level)")
    _plot_per_level_panel(ax_f1, ns, f1_bound, active_levels,
                          "F1_bound", "F1_bound vs N (per level)",
                          mark_argmax=True)
    _plot_per_level_panel(ax_miou, ns, miou_weighted, active_levels,
                          "mIoU (area-weighted)", "Per-cell mIoU vs N (per level, area-weighted)",
                          mark_argmax=True, reference_series=miou_unweighted,
                          reference_note="dotted = legacy unweighted mean")

    # ── Composite F1_multi (Step 3e) ────────────────────────────────────────
    ax_comp.plot(ns, f1_G, "ko-",  lw=2.5, ms=8,
                 label="F1_multi^G — gradient-magnitude (primary)")
    ax_comp.plot(ns, f1_P, "b--s", lw=1.8, ms=7,
                 label="F1_multi^P — inverse-perimeter (secondary)")
    ax_comp.plot(ns, f1_F, "g-.^", lw=1.8, ms=7,
                 label="F1_multi^F — fixed weights (baseline)")

    # Convergence check: mark argmax of each composite
    for vals, color, scheme_label in [
        (f1_G, "black", "G"),
        (f1_P, "blue",  "P"),
        (f1_F, "green", "F"),
    ]:
        if any(v > 1e-6 for v in vals):
            idx = int(np.argmax(vals))
            ax_comp.axvline(
                ns[idx], color=color, lw=1.1, ls=":",
                label=f"argmax F1^{scheme_label} = N={ns[idx]} ({vals[idx]:.3f})",
            )

    ax_comp.set_xlabel("N (superpixels)", fontsize=10)
    ax_comp.set_ylabel("Composite F1_multi", fontsize=10)
    ax_comp.set_title(
        "Composite F1_multi vs N (Step 3e)\n"
        "Convergence check: argmax under G ≈ P ≈ F validates scheme robustness",
        fontsize=10,
    )
    ax_comp.set_ylim(0, 1.05)
    ax_comp.legend(fontsize=7.5)
    ax_comp.grid(alpha=0.3)

    fig.suptitle(
        f"B1 Metrics vs N — Event {event_id}\n"
        f"(ROI δ_margin={margin_px}px, tolerance δ={tolerance_px}px; "
        f"active levels: {', '.join(active_levels)})",
        fontsize=12,
    )

    out_path2 = os.path.join(out_dir, f"{event_id}_b1_metrics_vs_N.png")
    fig.savefig(out_path2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved metrics-vs-N plot -> {out_path2}")

def compute_gradient_magnitude(vil_smooth):
    """
    Central-difference |∇VIL_smooth| on the full (H,W) domain.

    Used by Scheme G (Step 3e): g_k = mean of this field sampled at ∂C(τ_k).
    By physics, the steepest VIL gradients occur at the edge of the most
    intense cells (∂C(τ_high)), so Scheme G gives w_high > w_med > w_low
    for convective events — not by assumption, but because |∇VIL| is largest
    where the field transitions sharply from the severe-convection interior
    to the surrounding moderate region.

    The gradient is computed on the σ_cell*-smoothed field already produced
    in Step 1a; no additional smoothing is needed (spec §3e Scheme G).
    """
    gy, gx = np.gradient(vil_smooth.astype(np.float64))
    return np.sqrt(gx**2 + gy**2).astype(np.float32)


def compute_scheme_weights(level_data, grad_mag, active_levels):
    """
    Per-level weights under Scheme G, P, and F for the given active levels.

    All three weight vectors sum to 1 over active_levels. Inactive levels
    are excluded from the denominator (normalisation absorbs them, no explicit
    redistribution rule needed — spec §3e inactive-threshold handling).

    Scheme G (Primary) — Boundary Gradient-Magnitude Weighting
        g_k = (1/|∂C(τ_k)|) Σ_{p ∈ ∂C(τ_k)} |∇VIL_smooth(p)|
        w_k^G = g_k / Σ_{j:active} g_j
        ISV-aligned: g_k directly measures the sharpness of the τ_k contour,
        the exact property that determines whether a SLIC boundary at that
        location reduces within-superpixel variance.

    Scheme P (Secondary) — Inverse-Perimeter Weighting
        w_k^P = (1/|∂C(τ_k)|) / Σ_{j:active} (1/|∂C(τ_j)|)
        Shorter, more compact contours correspond to harder-to-resolve features.
        Nested structure guarantees |∂C(τ_high)| ≤ |∂C(τ_med)| ≤ |∂C(τ_low)|,
        so w_high^P ≥ w_med^P ≥ w_low^P.

    Scheme F (Reference Baseline) — Fixed Weights, renormalised over active levels
        w_low=0.20, w_med=0.30, w_high=0.50 (prior expectation; class-average
        approximation of what Scheme G produces for convective events).

    Parameters
    ----------
    level_data    : dict — output of build_all_threshold_cells
    grad_mag      : (H,W) float32 — |∇VIL_smooth| computed by compute_gradient_magnitude
    active_levels : list[str] — level names with M > 0

    Returns
    -------
    w_G, w_P, w_F : dict[str -> float]  per-level weights, each summing to 1
    g_vals        : dict[str -> float]  mean |∇VIL| sampled at ∂C(τ_k)
    """
    # ── Scheme G ─────────────────────────────────────────────────────────────
    g_vals = {}
    for level in active_levels:
        bnd = level_data[level]["boundary"]
        n_bnd = int(bnd.sum())
        g_vals[level] = float(grad_mag[bnd].mean()) if n_bnd > 0 else 0.0

    g_sum = sum(g_vals.values())
    if g_sum > 1e-12:
        w_G = {lv: g_vals[lv] / g_sum for lv in active_levels}
    else:
        # Degenerate: all boundaries carry equal mean gradient → equal weights
        eq = 1.0 / len(active_levels)
        w_G = {lv: eq for lv in active_levels}

    # ── Scheme P ─────────────────────────────────────────────────────────────
    p_vals = {
        lv: 1.0 / (int(level_data[lv]["boundary"].sum()) + 1e-8)
        for lv in active_levels
    }
    p_sum = sum(p_vals.values())
    w_P = {lv: p_vals[lv] / p_sum for lv in active_levels}

    # ── Scheme F ─────────────────────────────────────────────────────────────
    f_raw = {lv: FIXED_WEIGHTS_F[lv] for lv in active_levels}
    f_sum = sum(f_raw.values())
    w_F = {lv: f_raw[lv] / f_sum for lv in active_levels}

    return w_G, w_P, w_F, g_vals


def compute_multi_threshold_f1(level_data, vil_smooth, boundary_slic,
                                tolerance_px=DEFAULT_TOLERANCE_PX,
                                margin_px=DEFAULT_MARGIN_PX):
    """
    Steps 3a–3e: per-threshold ROI-masked F1 and event-adaptive composite.

    Calls boundary_f1 once per active threshold level (each level has its own
    cell_mask and therefore its own Ω_C(τ_k)), then aggregates under all three
    weighting schemes (G, P, F).

    Parameters
    ----------
    level_data    : dict — output of build_all_threshold_cells
    vil_smooth    : (H,W) float32 — σ_cell*-smoothed VIL (for Scheme G gradient)
    boundary_slic : (H,W) bool — ∂SLIC at the current N
    tolerance_px  : int  — δ (spec default: 2 px)
    margin_px     : int  — δ_margin (spec default: 7 px; must be > tolerance_px)

    Returns
    -------
    dict with:
      per_level     : dict[level] → {precision, recall, f1, active,
                                      dilated_cells, dilated_slic, roi_mask}
      active_levels : list[str]
      f1_multi_G    : float   composite under Scheme G (primary)
      f1_multi_P    : float   composite under Scheme P (secondary)
      f1_multi_F    : float   composite under Scheme F (baseline)
      w_G, w_P, w_F : dict[str -> float]
      g_vals        : dict[str -> float]  mean |∇VIL| per active level
    """
    assert margin_px > tolerance_px, (
        f"Spec constraint violated: δ_margin={margin_px} must be > δ={tolerance_px}"
    )

    active_levels = [lv for lv in LEVEL_NAMES if level_data[lv]["active"]]

    # Per-threshold ROI-masked F1 (Steps 3b–3d)
    per_level = {}
    for level in LEVEL_NAMES:
        if not level_data[level]["active"]:
            # Inactive level — edge-case: |∂C(τ_k)|=0, weights auto-excluded
            per_level[level] = {
                "precision": 0.0, "recall": 0.0, "f1": 0.0, "active": False,
                "dilated_cells": None, "dilated_slic": None, "roi_mask": None,
            }
            continue

        p, r, f1, dil_c, dil_s, roi = boundary_f1(
            level_data[level]["boundary"],
            boundary_slic,
            level_data[level]["cell_mask"],
            tolerance_px=tolerance_px,
            margin_px=margin_px,
        )
        per_level[level] = {
            "precision": p, "recall": r, "f1": f1, "active": True,
            "dilated_cells": dil_c, "dilated_slic": dil_s, "roi_mask": roi,
        }

    if not active_levels:
        log.warning("No active threshold levels — event has no cells above any threshold.")
        return {
            "per_level": per_level, "active_levels": [],
            "f1_multi_G": 0.0, "f1_multi_P": 0.0, "f1_multi_F": 0.0,
            "w_G": {}, "w_P": {}, "w_F": {}, "g_vals": {},
        }

    # Step 3e — Scheme G, P, F weights and composite scores
    grad_mag = compute_gradient_magnitude(vil_smooth)
    w_G, w_P, w_F, g_vals = compute_scheme_weights(level_data, grad_mag, active_levels)

    f1_multi_G = sum(w_G[lv] * per_level[lv]["f1"] for lv in active_levels)
    f1_multi_P = sum(w_P[lv] * per_level[lv]["f1"] for lv in active_levels)
    f1_multi_F = sum(w_F[lv] * per_level[lv]["f1"] for lv in active_levels)

    return {
        "per_level": per_level,
        "active_levels": active_levels,
        "f1_multi_G": f1_multi_G,
        "f1_multi_P": f1_multi_P,
        "f1_multi_F": f1_multi_F,
        "w_G": w_G, "w_P": w_P, "w_F": w_F,
        "g_vals": g_vals,
    }

# Scheme F (fixed-weight baseline): Step 3e, blocks_technical_v2.md
FIXED_WEIGHTS_F = {"low": 0.20, "med": 0.30, "high": 0.50}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(event_id, n_list, sigma_cell, tolerance_px,
        split_merged, n_max_sweep, use_dem, sigma_sweep,
        margin_px, margin_sweep, out_dir):

    if not os.path.exists(CATALOG_PATH):
        log.error("Catalog not found!")
        return
    catalog = pd.read_csv(CATALOG_PATH, low_memory=False)

    log.info(f"Loading data for {event_id} ...")
    data, extent = load_channels(event_id, catalog)
    if "vil" not in data:
        log.error("VIL channel not found for this event.")
        return

    vil_phys     = decode_vil_physical(data["vil"][0])
    fused_frame0 = build_fused_cube_frame0(data)

    dem_norm = None
    if use_dem:
        log.info("Fetching DEM ...")
        dem_norm = fetch_dem_norm(extent)

    # σ_cell calibration sweep (at τ_med only, per spec §1a)
    if sigma_sweep:
        sigma_stabilisation_sweep(vil_phys, TAU_MED, n_max_sweep)

    # Build multi-threshold reference cells (Steps 1a–1g)
    log.info(
        f"Building multi-threshold reference (Steps 1a–1g): "
        f"σ_cell={sigma_cell}, split={split_merged}, A_min=H×W/{n_max_sweep}"
    )
    vil_smooth, level_data = build_all_threshold_cells(
        vil_phys, sigma_cell=sigma_cell, n_max_sweep=n_max_sweep,
        split_merged=split_merged,
    )

    active_levels = [lv for lv in LEVEL_NAMES if level_data[lv]["active"]]
    log.info(f"Active threshold levels after Step 1e: {active_levels}")
    for lv in LEVEL_NAMES:
        ld = level_data[lv]
        log.info(
            f"  τ_{lv}={LEVEL_TAU[lv]:.1f} kg/m²: M={ld['M']}, "
            f"areas={sorted(ld['areas'].values()) if ld['areas'] else []}"
        )

    # ROI margin sensitivity sweep (robustness check, δ_margin ∈ {5,7,10} px)
    if margin_sweep:
        log.info(f"ROI margin sensitivity sweep (δ_margin ∈ {MARGIN_SWEEP_VALUES}) ...")
        # Precompute SLIC once per N to avoid redundant calls in the sweep
        slic_by_n = {
            n: (lbl := compute_slic_t0(fused_frame0, n, dem_norm=dem_norm),
                extract_boundary(lbl, require_positive=False))
            for n in n_list
        }
        print("\n  ROI margin sensitivity (spec §3b robustness check):")
        print("  " + "-" * 70)
        print(f"  {'δ_margin':>8}  {'N':>6}  {'F1^G':>8}  {'F1^P':>8}  {'F1^F':>8}")
        print("  " + "-" * 70)
        for sweep_margin in MARGIN_SWEEP_VALUES:
            for n_req in n_list:
                _, boundary_slic = slic_by_n[n_req]
                mt = compute_multi_threshold_f1(
                    level_data, vil_smooth, boundary_slic,
                    tolerance_px=tolerance_px, margin_px=sweep_margin,
                )
                print(
                    f"  {sweep_margin:>8}  {n_req:>6}  "
                    f"{mt['f1_multi_G']:>8.3f}  "
                    f"{mt['f1_multi_P']:>8.3f}  "
                    f"{mt['f1_multi_F']:>8.3f}"
                )
        print()

    # Main visualization
    results_per_n = visualize_event(
        event_id=event_id,
        vil_phys=vil_phys,
        vil_smooth=vil_smooth,
        level_data=level_data,
        fused_frame0=fused_frame0,
        dem_norm=dem_norm,
        n_list=n_list,
        sigma_cell=sigma_cell,
        tolerance_px=tolerance_px,
        split_merged=split_merged,
        margin_px=margin_px,
        out_dir=out_dir,
    )

    # Summary table
    print("\n  B1 Multi-Threshold Results")
    print("  " + "=" * 88)
    print(f"  {'N':>6}  {'actual':>7}  "
          f"{'F1_low':>8}  {'F1_med':>8}  {'F1_high':>9}  "
          f"{'F1^G':>8}  {'F1^P':>8}  {'F1^F':>8}")
    print("  " + "-" * 88)
    for n_req, actual_n, mt, _ in results_per_n:
        pl = mt["per_level"]
        print(
            f"  {n_req:>6}  {actual_n:>7}  "
            f"{pl['low']['f1']:>8.3f}  {pl['med']['f1']:>8.3f}  {pl['high']['f1']:>9.3f}  "
            f"{mt['f1_multi_G']:>8.3f}  {mt['f1_multi_P']:>8.3f}  {mt['f1_multi_F']:>8.3f}"
        )

    # Convergence check
    if len(results_per_n) > 1:
        ns = [r[0] for r in results_per_n]
        print("\n  Convergence check (spec §3e): argmax per composite scheme")
        for key, label in [("f1_multi_G", "^G"), ("f1_multi_P", "^P"), ("f1_multi_F", "^F")]:
            vals = [r[2][key] for r in results_per_n]
            idx  = int(np.argmax(vals))
            print(f"    argmax F1{label} = N={ns[idx]}  (F1={vals[idx]:.3f})")
        print("    -> If all three argmax values agree within ±50, "
              "the multi-threshold mechanism is scheme-robust.")

    # Scheme G gradient weights (class-characterising statistic)
    if active_levels:
        print("\n  Scheme G: mean |∇VIL| at ∂C per threshold (g_k) and event-mean weights:")
        for lv in active_levels:
            g_list = [r[2]["g_vals"].get(lv, 0.0) for r in results_per_n
                      if lv in r[2].get("g_vals", {})]
            w_list = [r[2]["w_G"].get(lv, 0.0)    for r in results_per_n
                      if lv in r[2].get("w_G", {})]
            if g_list:
                print(f"    τ_{lv}: mean g_k={np.mean(g_list):.4f}  "
                      f"mean w_k^G={np.mean(w_list):.3f}")
        print("    (Expected for convective events: g_high > g_med > g_low)")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "B1 Diagnostic: SCIT-inspired multi-threshold CC vs TemporalSLIC "
            "(blocks_technical_v2.md §B1, Steps 1a–1g, 3a–3e)"
        )
    )
    parser.add_argument("--event_id",    default=None)
    parser.add_argument("--n_list",      type=int, nargs="+", default=[950],
                        help="One or more superpixel counts to evaluate")
    parser.add_argument("--sigma_cell",  type=float, default=1.0,
                        help="Gaussian pre-smoothing σ calibrated at τ_med, "
                             "shared across all threshold levels (default: 1.0 px)")
    parser.add_argument("--tolerance",   type=int, default=DEFAULT_TOLERANCE_PX,
                        help=f"Boundary-match tolerance δ in pixels (default: {DEFAULT_TOLERANCE_PX})")
    parser.add_argument("--margin_px",   type=int, default=DEFAULT_MARGIN_PX,
                        help=f"ROI dilation δ_margin (default: {DEFAULT_MARGIN_PX} px; "
                             "midpoint of {5,7,10}; must be > tolerance)")
    parser.add_argument("--split_merged", action="store_true",
                        help="Enable Step 1f watershed-from-maxima cell splitting")
    parser.add_argument("--n_max_sweep", type=int, default=DEFAULT_N_MAX_SWEEP,
                        help=f"Finest N in the sweep, for A_min derivation (default: {DEFAULT_N_MAX_SWEEP})")
    parser.add_argument("--use_dem",     action="store_true",
                        help="Fetch and append DEM channel (requires optional libs)")
    parser.add_argument("--sigma_sweep", action="store_true",
                        help="Print σ_cell stabilisation diagnostic at τ_med (Step 1a calibration)")
    parser.add_argument("--margin_sweep", action="store_true",
                        help="Sensitivity sweep over δ_margin ∈ {5,7,10} px (spec §3b robustness check)")
    parser.add_argument("--out_dir",     default=OUT_DIR)
    args = parser.parse_args()

    event_id = args.event_id
    if event_id is None:
        event_id = input("Enter Event ID: ").strip()

    run(
        event_id=event_id,
        n_list=args.n_list,
        sigma_cell=args.sigma_cell,
        tolerance_px=args.tolerance,
        margin_px=args.margin_px,
        split_merged=args.split_merged,
        n_max_sweep=args.n_max_sweep,
        use_dem=args.use_dem,
        sigma_sweep=args.sigma_sweep,
        margin_sweep=args.margin_sweep,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()