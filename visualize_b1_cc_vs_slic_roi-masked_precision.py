"""
visualize_b1_cc_vs_slic.py — Block B1: Connected Components vs TemporalSLIC
=============================================================================
Visual diagnostic for Block B1 (Convective Cell Boundary Alignment).

For a single SEVIR event, this script:
  1. Builds the absolute-physical-threshold connected-component reference
     cell map exactly per Section B1, Steps 1a–1f of blocks_technical_v2.md
     (Gaussian pre-smoothing → absolute VIL threshold → hole-filling →
     8-connected labelling → minimum-area discard tied to the finest
     superpixel resolution tested → optional watershed-from-maxima split
     of merged cells).
  2. Runs standard SLIC (the t=0 branch of TemporalSLIC) on the fused
     VIL+IR107+IR069[+DEM] feature cube, for one or more requested N values.
  3. Computes boundary ROI-Precision / Recall / F1_bound (eq. B1, Step 3)
     between the two segmentations.  Precision uses the ROI-masked denominator
     (∂SLIC ∩ Ω_C) so that clear-air Lagrangian tiling is not penalised; only
     SLIC edges inside the dilated convective zone Ω_C = dilate(C, δ_margin)
     are counted in the denominator.
  4. Renders a comparison figure: reference cells, SLIC boundaries, a
     combined boundary overlay, and a match/miss diagnostic panel that
     visually attributes precision and recall failures to specific pixels.
  5. If multiple N values are given, also plots F1_bound vs N for this event,
     mirroring (at single-event scale) the Figure B-1 aggregate plot.

This reuses the data-loading and SLIC conventions from
Visualize_sevir_with_superpixel_fused.py (BASE_PATH, CATALOG_PATH,
load_fused_channels, the t=0 branch of TemporalSLIC_DEM) but is otherwise
self-contained — it does not import that file, so it can be run independently.

Usage
-----
    python visualize_b1_cc_vs_slic.py --event_id <ID> --n_list 500 850 1150
    python visualize_b1_cc_vs_slic.py                      # interactive prompt
    python visualize_b1_cc_vs_slic.py --event_id <ID> --n_list 1150 --split_merged
    python visualize_b1_cc_vs_slic.py --event_id <ID> --n_list 1150 --sigma_sweep

Dependencies
------------
    pip install h5py pandas numpy scipy scikit-image opencv-python matplotlib tqdm
    (DEM channel is optional; requires pystac_client, planetary_computer,
     rioxarray — gracefully skipped if unavailable, exactly as in the
     original visualization script.)
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

FUSION_CHANNELS = ["vil", "ir107", "ir069"]
COMPACTNESS     = 10.0
ELEVATION_LAMBDA = 0.5

# Absolute physical VIL thresholds (Preliminary section, blocks_technical_v2.md)
VIL_MAX_PHYS = 70.0     # kg/m^2, SEVIR clip value
TAU_LOW      = 4.4      # kg/m^2  -- onset of detectable precipitation
TAU_MED      = 20.3     # kg/m^2  -- moderate deep convection (default cell threshold)
TAU_HIGH     = 36.5     # kg/m^2  -- severe convection (seed threshold for splitting)
TAU_TABLE    = {"low": TAU_LOW, "med": TAU_MED, "high": TAU_HIGH}

# Default sweep grid (used only to derive A_min = H*W/N_max(sweep))
DEFAULT_N_MAX_SWEEP = 1500

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
# B1 STEP 1 — Connected-component reference cell construction
#   (Steps 1a–1f exactly per blocks_technical_v2.md, Section B1)
# ─────────────────────────────────────────────────────────────────────────────

def decode_vil_physical(vil_raw_frame0: np.ndarray) -> np.ndarray:
    """raw float32 (already cast from uint8) -> physical kg/m^2."""
    return np.clip(vil_raw_frame0, 0, 255) / 255.0 * VIL_MAX_PHYS


def sigma_stabilisation_sweep(vil_phys, tau_cell, n_max_sweep,
                              sigmas=(0.5, 0.75, 1.0, 1.5, 2.0)):
    """
    Diagnostic for Step 1a: prints the number of retained connected
    components M(sigma) across a small grid, so sigma_cell can be chosen
    as the smallest value at which M stabilises (changes < 5% per 0.25px
    step), per the calibration procedure in Section B1.
    """
    print("\n  sigma_cell stabilisation sweep:")
    print("  " + "-" * 40)
    prev_M = None
    for s in sigmas:
        C, M, _ = build_reference_cells(vil_phys, tau_cell, sigma_cell=s,
                                        n_max_sweep=n_max_sweep, split_merged=False)
        delta = "" if prev_M is None else f"  (Δ={100*(M-prev_M)/max(prev_M,1):+.1f}%)"
        print(f"    sigma={s:4.2f} px  ->  M={M:3d} cells{delta}")
        prev_M = M
    print("  " + "-" * 40)
    print("  Choose the smallest sigma where successive M values differ by < 5%.\n")


def build_reference_cells(
    vil_phys: np.ndarray,
    tau_cell: float,
    sigma_cell: float = 1.0,
    n_max_sweep: int = DEFAULT_N_MAX_SWEEP,
    split_merged: bool = False,
    seed_min_distance: int = 5,
    tau_seed: float = TAU_HIGH,
):
    """
    Full B1 Step 1 pipeline (1a-1f). Returns (C, M, areas).

    C      : (H, W) int32 label map, 0=background, 1..M = cell id
    M      : number of retained cells
    areas  : dict {cell_id: area_in_pixels}
    """
    H, W = vil_phys.shape

    # 1a — Gaussian pre-smoothing
    vil_smooth = gaussian_filter(vil_phys, sigma=sigma_cell)

    # 1b — Absolute physical threshold (NOT relative to event dynamic range)
    B_VIL = vil_smooth > tau_cell

    # 1c — Hole filling
    B_filled = binary_fill_holes(B_VIL)

    # 1d — Connected component labelling, 8-connectivity
    structure_8 = np.ones((3, 3), dtype=np.int32)
    C_raw, M_raw = cc_label(B_filled, structure=structure_8)

    # 1e — Minimum area: tied to the finest superpixel resolution tested,
    #      NOT an arbitrary pixel count.
    A_min = (H * W) / n_max_sweep

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

    # 1f — Optional watershed-from-maxima split of merged cells
    if split_merged and M > 0:
        C, areas = _split_merged_cells(C, vil_smooth, tau_seed, seed_min_distance)
        M = int(C.max())

    return C, M, areas


def _split_merged_cells(C, vil_smooth, tau_seed, min_distance):
    """
    Step 1f: for each retained component, detect local maxima exceeding
    tau_seed; if more than one is found, split via watershed on the
    inverted smoothed field restricted to that component.
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
# SLIC (t=0 branch only — standard multi-channel SLIC, no flow/watershed needed
# since B1 evaluates exclusively at the first frame)
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
# Boundary extraction and F1 (B1, Steps 2-3)
# ─────────────────────────────────────────────────────────────────────────────

def extract_boundary(label_map, require_positive=False):
    """
    4-connected boundary pixel mask.

    require_positive=True  : used for the cell map C — boundary pixels must
                              themselves be foreground (C>0); a pixel is a
                              boundary pixel if any 4-neighbour has a
                              different label (including label 0 = background).
    require_positive=False : used for SLIC — every pixel is foreground, so
                              simply test neighbour-label inequality.
    """
    H, W = label_map.shape
    boundary = np.zeros((H, W), dtype=bool)

    up    = np.roll(label_map,  1, axis=0)
    down  = np.roll(label_map, -1, axis=0)
    left  = np.roll(label_map,  1, axis=1)
    right = np.roll(label_map, -1, axis=1)

    diff = ((label_map != up) | (label_map != down) |
            (label_map != left) | (label_map != right))

    # Edge pixels: np.roll wraps around, which is wrong at the image border.
    # Mask out the outer ring from "different" by comparing against itself there.
    diff[0, :] = (label_map[0, :] != down[0, :]) | (label_map[0, :] != left[0, :]) | (label_map[0, :] != right[0, :])
    diff[-1, :] = (label_map[-1, :] != up[-1, :]) | (label_map[-1, :] != left[-1, :]) | (label_map[-1, :] != right[-1, :])
    diff[:, 0] = (label_map[:, 0] != up[:, 0]) | (label_map[:, 0] != down[:, 0]) | (label_map[:, 0] != right[:, 0])
    diff[:, -1] = (label_map[:, -1] != up[:, -1]) | (label_map[:, -1] != down[:, -1]) | (label_map[:, -1] != left[:, -1])

    boundary = diff
    if require_positive:
        boundary = boundary & (label_map > 0)
    return boundary


def boundary_f1(boundary_cells, boundary_slic, cell_mask,
                tolerance_px=2, margin_px=8):
    """
    ROI-masked Precision / Recall / F1 at a given pixel tolerance (B1 Step 3).

    Motivation
    ----------
    TemporalSLIC tiles the *entire* domain so it can advect a Lagrangian graph
    forward in time.  Evaluating global precision — counting every SLIC edge in
    the denominator — crushes the score because the vast majority of superpixel
    boundaries fall in clear-air regions with no reference cell boundary to
    match.  The ROI-masked variant fixes this by restricting the precision
    denominator to edges that lie inside the convective evaluation zone Ω_C.

    Mathematical definition
    -----------------------
    Ω_C        = dilate(cell_mask, δ_margin)          — evaluation zone
    ∂SLIC_ROI  = ∂SLIC ∩ Ω_C                         — denominator
    Precision_ROI = |∂SLIC ∩ dilate(∂C, δ)| / |∂SLIC_ROI|

    Recall is unchanged: it measures how much of ∂C is covered by ∂SLIC,
    which is already restricted to storm pixels by construction.

    Parameters
    ----------
    boundary_cells : (H,W) bool  — ∂C, reference cell boundary pixels
    boundary_slic  : (H,W) bool  — ∂SLIC, superpixel boundary pixels
    cell_mask      : (H,W) bool  — binary foreground mask (cell_map > 0)
    tolerance_px   : int         — dilation radius δ for boundary match
    margin_px      : int         — dilation radius δ_margin that defines Ω_C
                                   (recommended 5–10 px; default 8)

    Returns
    -------
    precision, recall, f1  : float
    dilated_cells          : (H,W) bool  — dilate(∂C, δ), used by the panel
    dilated_slic           : (H,W) bool  — dilate(∂SLIC, δ), used by the panel
    roi_mask               : (H,W) bool  — Ω_C, used by the panel
    """
    se_tol    = disk(tolerance_px)
    se_margin = disk(margin_px)

    dilated_cells = binary_dilation(boundary_cells, structure=se_tol)
    dilated_slic  = binary_dilation(boundary_slic,  structure=se_tol)

    # Ω_C : convective evaluation zone — a spatial buffer around every cell
    roi_mask = binary_dilation(cell_mask, structure=se_margin)

    # ∂SLIC_ROI : only the SLIC edges inside the evaluation zone
    boundary_slic_roi = boundary_slic & roi_mask

    n_slic_roi = boundary_slic_roi.sum()   # ROI-masked denominator
    n_cells    = boundary_cells.sum()

    # Numerator: SLIC edges (anywhere) that land within tolerance of ∂C
    precision = (boundary_slic & dilated_cells).sum() / (n_slic_roi + 1e-8)
    recall    = (boundary_cells & dilated_slic).sum()  / (n_cells    + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return float(precision), float(recall), float(f1), dilated_cells, dilated_slic, roi_mask


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def _cell_colormap(M):
    """Distinct colours per cell id, with id=0 forced to black (background)."""
    if M == 0:
        return ListedColormap(["black"])
    base = plt.get_cmap("tab20")(np.linspace(0, 1, max(M, 1)))
    colors = np.vstack([[0, 0, 0, 1], base])
    return ListedColormap(colors)


def render_match_miss_panel(vil_phys, boundary_cells, boundary_slic,
                            dilated_cells, dilated_slic, roi_mask):
    """
    Diagnostic overlay attributing precision/recall failures to pixels.

    Colour key
    ----------
    green      — TP: SLIC boundary within tolerance of a cell boundary
    orange     — FP inside Ω_C: SLIC boundary in the evaluation zone but not
                 near any cell boundary → counted in denominator, hurts P_ROI
    dim ochre  — FP outside Ω_C: SLIC boundary in clear-air region → ignored
                 by the ROI-masked precision metric (Lagrangian tiling artefact)
    magenta    — FN: cell boundary pixel not covered by any SLIC boundary
                 (hurts recall)
    white      — outline of the ROI evaluation zone Ω_C (for spatial context)
    """
    H, W = vil_phys.shape
    panel = np.tile((vil_phys / (vil_phys.max() + 1e-6))[:, :, None], (1, 1, 3)) * 0.35

    tp_mask         = boundary_slic & dilated_cells
    fp_counted_mask = boundary_slic & ~dilated_cells & roi_mask    # inside Ω_C → penalises P_ROI
    fp_ignored_mask = boundary_slic & ~dilated_cells & ~roi_mask   # outside Ω_C → not penalised
    fn_mask         = boundary_cells & ~dilated_slic

    # Ω_C outline — extract edge pixels of the ROI zone for spatial context
    roi_boundary = extract_boundary(roi_mask.astype(np.int32), require_positive=True)

    # Paint in order: ignored FPs first (lowest priority), then active masks on top
    panel[fp_ignored_mask] = [0.55, 0.45, 0.10]   # dim ochre  — present but not counted
    panel[tp_mask]         = [0.00, 0.90, 0.00]   # green      — true positives
    panel[fp_counted_mask] = [1.00, 0.55, 0.00]   # orange     — counted false positives
    panel[fn_mask]         = [0.95, 0.00, 0.85]   # magenta    — false negatives
    panel[roi_boundary]    = [0.90, 0.90, 0.90]   # white      — Ω_C zone outline
    return np.clip(panel, 0, 1)


def visualize_event(event_id, vil_phys, cell_map, M, fused_frame0,
                    dem_norm, n_list, tau_cell, sigma_cell, tolerance_px,
                    split_merged, margin_px, out_dir):
    n_rows = 1 + len(n_list)
    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 4.2 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, 3)

    boundary_cells = extract_boundary(cell_map, require_positive=True)

    # ---- Row 0: shared reference (VIL, connected components, cell boundaries on VIL) ----
    vmax_vil = max(vil_phys.max(), 1.0)
    axes[0, 0].imshow(vil_phys, cmap="jet", vmin=0, vmax=vmax_vil, origin="lower")
    axes[0, 0].set_title(f"VIL (physical, kg/m²)\nτ_cell={tau_cell} kg/m²", fontsize=10)
    axes[0, 0].axis("off")

    cmap_cells = _cell_colormap(M)
    axes[0, 1].imshow(cell_map, cmap=cmap_cells, origin="lower",
                      vmin=0, vmax=max(M, 1))
    axes[0, 1].set_title(
        f"Connected-component cells\nM={M}  (σ={sigma_cell}px, split={split_merged})",
        fontsize=10,
    )
    axes[0, 1].axis("off")

    rgb_vil = plt.get_cmap("jet")(np.clip(vil_phys / vmax_vil, 0, 1))[:, :, :3]
    cell_overlay = mark_boundaries(rgb_vil, cell_map, color=(1, 0, 0), mode="thick")
    axes[0, 2].imshow(cell_overlay, origin="lower")
    axes[0, 2].set_title("Cell boundaries (red) on VIL", fontsize=10)
    axes[0, 2].axis("off")

    f1_per_n = []

    # Binary foreground mask — used by boundary_f1 to define the ROI zone Ω_C
    cell_mask = (cell_map > 0)

    for i, n_req in enumerate(n_list):
        row = i + 1
        slic_labels = compute_slic_t0(fused_frame0, n_req, dem_norm=dem_norm)
        actual_n = len(np.unique(slic_labels))
        boundary_slic = extract_boundary(slic_labels, require_positive=False)

        precision, recall, f1, dil_cells, dil_slic, roi_mask = boundary_f1(
            boundary_cells, boundary_slic, cell_mask,
            tolerance_px=tolerance_px, margin_px=margin_px,
        )
        f1_per_n.append((n_req, actual_n, precision, recall, f1))

        slic_overlay = mark_boundaries(rgb_vil, slic_labels, color=(0, 0.9, 1.0), mode="thick")
        axes[row, 0].imshow(slic_overlay, origin="lower")
        axes[row, 0].set_title(f"SLIC N={n_req} (actual={actual_n})", fontsize=10)
        axes[row, 0].axis("off")

        combined = mark_boundaries(rgb_vil, cell_map, color=(1, 0, 0), mode="thick")
        combined = mark_boundaries(combined, slic_labels, color=(0, 0.9, 1.0), mode="thick")
        axes[row, 1].imshow(combined, origin="lower")
        axes[row, 1].set_title("Red=cells  Cyan=SLIC", fontsize=10)
        axes[row, 1].axis("off")

        match_panel = render_match_miss_panel(
            vil_phys, boundary_cells, boundary_slic, dil_cells, dil_slic, roi_mask
        )
        axes[row, 2].imshow(match_panel, origin="lower")
        axes[row, 2].set_title(
            f"P_ROI={precision:.3f}  R={recall:.3f}  F1={f1:.3f}"
            f"  (tol={tolerance_px}px, margin={margin_px}px)",
            fontsize=10,
        )
        axes[row, 2].axis("off")

    legend_text = (
        "Match/miss panel:  green = TP (SLIC matches cell boundary)   "
        "orange = FP inside Ω_C (hurts ROI precision)   "
        "dim ochre = FP outside Ω_C (ignored by metric — Lagrangian tiling)   "
        "magenta = FN (missed cell boundary)   "
        "white = ROI zone Ω_C boundary"
    )
    fig.text(0.5, 0.005, legend_text, ha="center", fontsize=8, style="italic")

    fig.suptitle(
        f"B1 Diagnostic — Event {event_id}\n"
        f"M={M} reference cells  |  τ_cell={tau_cell} kg/m²  |  σ_cell={sigma_cell}px  "
        f"|  split_merged={split_merged}  |  ROI margin={margin_px}px",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    out_path = os.path.join(out_dir, f"{event_id}_b1_diagnostic.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved figure -> {out_path}")

    # ---- Optional: F1_bound vs N line plot (single-event mirror of Fig B-1) ----
    if len(n_list) > 1:
        ns       = [row[0] for row in f1_per_n]
        f1s      = [row[4] for row in f1_per_n]
        precs    = [row[2] for row in f1_per_n]
        recs     = [row[3] for row in f1_per_n]

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(ns, f1s, "ko-", label="F1_bound", lw=2)
        ax2.plot(ns, precs, "b--", label="Precision_ROI", alpha=0.7)
        ax2.plot(ns, recs, "r--", label="Recall", alpha=0.7)
        ax2.set_xlabel("N (superpixels)")
        ax2.set_ylabel("Score")
        ax2.set_title(
            f"F1_bound vs N — Event {event_id}\n"
            f"(Precision = ROI-masked, margin={margin_px}px)"
        )
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
        fig2.tight_layout()
        out_path2 = os.path.join(out_dir, f"{event_id}_b1_f1_vs_N.png")
        fig2.savefig(out_path2, dpi=130, bbox_inches="tight")
        plt.close(fig2)
        log.info(f"Saved F1-vs-N plot -> {out_path2}")

    return f1_per_n


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(event_id, n_list, tau_key, sigma_cell, tolerance_px,
       split_merged, n_max_sweep, use_dem, sigma_sweep, margin_px, out_dir):
    if not os.path.exists(CATALOG_PATH):
        log.error("Catalog not found!")
        return
    catalog = pd.read_csv(CATALOG_PATH, low_memory=False)

    log.info(f"Loading data for {event_id} ...")
    data, extent = load_channels(event_id, catalog)
    if "vil" not in data:
        log.error("VIL channel not found for this event.")
        return

    vil_phys = decode_vil_physical(data["vil"][0])
    fused_frame0 = build_fused_cube_frame0(data)

    dem_norm = None
    if use_dem:
        log.info("Fetching DEM ...")
        dem_norm = fetch_dem_norm(extent)

    tau_cell = TAU_TABLE[tau_key]

    if sigma_sweep:
        sigma_stabilisation_sweep(vil_phys, tau_cell, n_max_sweep)

    log.info(f"Building reference cells: tau_cell={tau_cell} kg/m², sigma_cell={sigma_cell}, "
             f"split_merged={split_merged}")
    cell_map, M, areas = build_reference_cells(
        vil_phys, tau_cell, sigma_cell=sigma_cell,
        n_max_sweep=n_max_sweep, split_merged=split_merged,
    )
    log.info(f"Reference cells: M={M}, areas={sorted(areas.values())}")

    f1_per_n = visualize_event(
        event_id, vil_phys, cell_map, M, fused_frame0, dem_norm,
        n_list, tau_cell, sigma_cell, tolerance_px, split_merged, margin_px, out_dir,
    )

    print("\n  N      actual_N   Precision   Recall   F1_bound")
    print("  " + "-" * 50)
    for n_req, actual_n, p, r, f1 in f1_per_n:
        print(f"  {n_req:<7d}{actual_n:<11d}{p:<12.3f}{r:<9.3f}{f1:.3f}")


def main():
    parser = argparse.ArgumentParser(description="B1: Connected components vs TemporalSLIC")
    parser.add_argument("--event_id", default=None)
    parser.add_argument("--n_list", type=int, nargs="+", default=[950])
    parser.add_argument("--tau", choices=["low", "med", "high"], default="med",
                        help="Absolute physical threshold for cell identification (default: med=20.3 kg/m^2)")
    parser.add_argument("--sigma_cell", type=float, default=1.0)
    parser.add_argument("--tolerance", type=int, default=2, help="Boundary-match tolerance in pixels")
    parser.add_argument("--margin_px", type=int, default=8,
                        help="ROI dilation radius delta_margin defining the convective evaluation zone (recommended 5-10px, default 8)")
    parser.add_argument("--split_merged", action="store_true",
                        help="Enable Step 1f watershed-from-maxima splitting of merged cells")
    parser.add_argument("--n_max_sweep", type=int, default=DEFAULT_N_MAX_SWEEP,
                        help="Finest N in the sweep, used to derive A_min (default: 1500)")
    parser.add_argument("--use_dem", action="store_true",
                        help="Fetch and append DEM channel (requires optional DEM libs)")
    parser.add_argument("--sigma_sweep", action="store_true",
                        help="Print the sigma_cell stabilisation diagnostic before running")
    parser.add_argument("--out_dir", default=OUT_DIR)
    args = parser.parse_args()

    event_id = args.event_id
    if event_id is None:
        event_id = input("Enter Event ID: ").strip()

    run(
        event_id=event_id,
        n_list=args.n_list,
        tau_key=args.tau,
        sigma_cell=args.sigma_cell,
        tolerance_px=args.tolerance,
        margin_px=args.margin_px,
        split_merged=args.split_merged,
        n_max_sweep=args.n_max_sweep,
        use_dem=args.use_dem,
        sigma_sweep=args.sigma_sweep,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()