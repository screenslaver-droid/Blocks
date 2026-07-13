"""
visualize_slic_sevir.py
=======================
Visualises TemporalSLIC_DEM superpixel segmentation across every frame of a
SEVIR event and saves the result as an animated GIF (or MP4 if ffmpeg is on PATH).

Layout — 2 × 2 grid:
  ┌──────────────────────┬──────────────────────┐
  │  VIL                 │  IR107               │
  │  Temporal SLIC       │  Temporal SLIC       │   ← flow-guided watershed
  ├──────────────────────┼──────────────────────┤
  │  VIL                 │  IR107               │
  │  Standard SLIC       │  Standard SLIC       │   ← static SLIC each frame
  └──────────────────────┴──────────────────────┘

Each panel shows:
  • Radar/IR data rendered with its native colourmap.
  • Superpixel boundaries drawn on top (gold = Temporal, cyan = Static).
  • Centroid dots (red = Temporal / white = Static).
  • Subsampled optical flow vectors on Temporal panels only.

All segmentation is pre-computed before the animation starts so that:
  (a) TemporalSLIC's stateful flow-tracking stays correct (must run sequentially).
  (b) The animation loop is just im.set_data() — no heavy work per frame.
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from skimage.segmentation import slic, watershed, mark_boundaries
from skimage.filters import sobel
from skimage.feature import peak_local_max
from scipy.ndimage import label, center_of_mass
from skimage.morphology import disk, dilation
import cv2
import os
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import cartopy.crs as ccrs
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# LOGGING
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# CONFIGURATION
# --------------------------------------------------------------------------- #
BASE_PATH    = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\SEVIR\2019"
CATALOG_PATH = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\SEVIR\CATALOG.csv"
EVENTS_DIR   = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\Events"

N_SEGMENTS        = 1500
COMPACTNESS       = 10.0
ELEVATION_LAMBDA  = 0.5
SECONDS_PER_FRAME = 300       # 5 min per SEVIR frame
QUIVER_STRIDE     = 24        # show 1 flow arrow every N pixels

os.makedirs(EVENTS_DIR, exist_ok=True)

SEVIR_PROJ = ccrs.LambertConformal(
    central_longitude=-98.0, central_latitude=38.0,
    standard_parallels=(30.0, 60.0),
    globe=ccrs.Globe(semimajor_axis=6370000, semiminor_axis=6370000)
)

CHANNEL_CFG = {
    "vil":   {"cmap": plt.get_cmap("jet"),    "vmin": 0,    "vmax": 255},
    "ir107": {"cmap": plt.get_cmap("gray_r"), "vmin": None, "vmax": None},
}

MODE_STYLE = {
    True:  {"boundary": (1.0, 0.75, 0.0), "centroid": "red",   "label": "Temporal SLIC"},
    False: {"boundary": (0.0, 0.90, 1.0), "centroid": "white", "label": "Standard SLIC"},
}

# --------------------------------------------------------------------------- #
# DEM PIPELINE
# --------------------------------------------------------------------------- #
try:
    import pystac_client
    import planetary_computer
    import rioxarray
    from rioxarray.merge import merge_arrays
    HAS_DEM_LIBS = True
except ImportError:
    HAS_DEM_LIBS = False
    log.warning("DEM libraries not found — using flat terrain (DEM = zeros).")

_dem_cache: dict = {}

def get_sevir_grid(extent, nx: int = 384, ny: int = 384):
    proj = SEVIR_PROJ
    src  = ccrs.PlateCarree()
    x0, y0 = proj.transform_point(extent[0], extent[2], src)
    x1, y1 = proj.transform_point(extent[1], extent[3], src)
    xv, yv = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    grid   = src.transform_points(proj, xv, yv)
    return grid[..., 0], grid[..., 1]

def _load_and_clip_tile(item, extent, buf=0.1):
    da = rioxarray.open_rasterio(item.assets["data"].href, lock=False).squeeze()
    da = da.rio.clip_box(minx=extent[0]-buf, miny=extent[2]-buf,
                         maxx=extent[1]+buf, maxy=extent[3]+buf)
    da = da.coarsen(x=3, y=3, boundary="trim").mean()
    return da

def fetch_and_regrid_dem(extent, nx: int = 384, ny: int = 384) -> np.ndarray:
    cache_key = tuple(np.round(extent, 4))
    if cache_key in _dem_cache:
        return _dem_cache[cache_key]
    if not HAS_DEM_LIBS:
        return np.zeros((ny, nx))

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    items = list(catalog.search(
        collections=["cop-dem-glo-30"],
        bbox=[extent[0], extent[2], extent[1], extent[3]],
    ).item_collection())
    if not items:
        return np.zeros((ny, nx))

    datasets = [None] * len(items)
    with ThreadPoolExecutor(max_workers=min(len(items), 8)) as pool:
        fmap = {pool.submit(_load_and_clip_tile, it, extent): i
                for i, it in enumerate(items)}
        for fut in tqdm(as_completed(fmap), total=len(items), desc="DEM tiles"):
            try:    datasets[fmap[fut]] = fut.result()
            except Exception as e: log.warning(f"DEM tile failed: {e}")

    datasets = [d for d in datasets if d is not None and d.size > 0]
    merged = merge_arrays(datasets) if len(datasets) > 1 else datasets[0]
    lons, lats, vals = merged.x.values, merged.y.values, merged.values
    if lats[0] > lats[-1]:
        lats = lats[::-1]; vals = vals[::-1, :]

    interp = RegularGridInterpolator(
        (lats, lons), vals.astype(np.float32),
        method="linear", bounds_error=False, fill_value=0.0,
    )
    tgt_lons, tgt_lats = get_sevir_grid(extent, nx, ny)
    result = interp(np.column_stack((tgt_lats.ravel(), tgt_lons.ravel()))
                    ).reshape(ny, nx).astype(np.float32)
    _dem_cache[cache_key] = result
    return result

# --------------------------------------------------------------------------- #
# TEMPORAL SLIC
# --------------------------------------------------------------------------- #
'''class TemporalSLIC_DEM:
    def __init__(self, dem_norm: np.ndarray, n_segments=150, 
                 compactness=10.0, lambda_z=0.5):
        """
        Args:
            dem_norm: Normalized DEM (0-1).
            n_segments: Target number of superpixels.
            compactness: Balances color proximity vs. space proximity.
            lambda_z: Weight of elevation in boundary detection.
        """
        self.n_segments   = n_segments
        self.compactness  = compactness
        self.lambda_z     = lambda_z
        
        # Pre-weight DEM for gradient calculation
        self.dem_weighted = dem_norm * lambda_z
        
        # State tracking for Lagrangian Advection
        self.prev_labels    = None
        self.prev_centroids = None # Format: {label: [y, x]}
        self.prev_gray      = None

    def segment(self, frame: np.ndarray, use_flow: bool = True):
        """
        Performs Lagrangian segmentation.
        1. Frame 0: Standard SLIC.
        2. Frame t->t+1: Advect centroids via Optical Flow -> Compact Watershed.
        """
        H, W = frame.shape
        
        # 1. Normalize Frame (0-1) for features, keep (0-255) for Flow
        mn, mx = frame.min(), frame.max()
        if mx - mn > 0:
            frame_norm = (frame - mn) / (mx - mn)
        else:
            frame_norm = np.zeros_like(frame, dtype=np.float32)
            
        curr_gray = (frame_norm * 255).astype(np.uint8)
        
        # Initialize Flow container
        flow = np.zeros((H, W, 2), dtype=np.float32)

        # ---------------------------------------------------------
        # CASE A: First Frame (Use Standard SLIC)
        # ---------------------------------------------------------
        if self.prev_labels is None:
            # Stack Frame + DEM for multichannel clustering if desired, 
            # or just use frame for standard SLIC. 
            # Here we use frame_norm and enforce DEM via post-gradient if needed, 
            # but standard SLIC on the image is usually sufficient for init.
            labels = slic(frame_norm.astype(np.float64), 
                          n_segments=self.n_segments,
                          compactness=self.compactness,
                          start_label=1,
                          enforce_connectivity=True, 
                          channel_axis=None)
            
            # Initialize Centroids
            self.prev_centroids = self._calculate_centroids(labels)

        # ---------------------------------------------------------
        # CASE B: Temporal Update (Lagrangian Advection)
        # ---------------------------------------------------------
        else:
            # 1. Calculate Dense Optical Flow (Farneback)
            if use_flow:
                flow = cv2.calcOpticalFlowFarneback(
                    self.prev_gray, curr_gray, None, 
                    pyr_scale=0.5, levels=3, winsize=15, 
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
            
            # 2. Advect Centroids (The "Lagrangian" Step)
            advected_markers = np.zeros((H, W), dtype=np.int32)
            valid_centroids = {}
            
            for lbl, centroid in self.prev_centroids.items():
                y_old, x_old = int(centroid[0]), int(centroid[1])
                
                # Robustness check: Ensure old centroid is within bounds
                if 0 <= y_old < H and 0 <= x_old < W:
                    # KEY FIX: Use MEAN flow of the superpixel, not point flow
                    # (Simplified here to local window average for speed)
                    window = 5
                    y1, y2 = max(0, y_old-window), min(H, y_old+window)
                    x1, x2 = max(0, x_old-window), min(W, x_old+window)
                    
                    flow_y = np.mean(flow[y1:y2, x1:x2, 1])
                    flow_x = np.mean(flow[y1:y2, x1:x2, 0])
                    
                    # Advect
                    y_new = int(np.clip(y_old + flow_y, 0, H-1))
                    x_new = int(np.clip(x_old + flow_x, 0, W-1))
                    
                    # Place Marker
                    advected_markers[y_new, x_new] = lbl
                    valid_centroids[lbl] = [y_new, x_new]

            # 3. Refinement (Compact Watershed)
            # Combine Image Gradient + DEM Gradient for boundaries
            grad_img = sobel(frame_norm)
            grad_dem = sobel(self.dem_weighted)
            combined_gradient = grad_img + grad_dem
            
            # Run Watershed starting from Advected Markers
            # "compactness" param here mimics SLIC behavior
            labels = watershed(combined_gradient, 
                               markers=advected_markers, 
                               compactness=self.compactness / 100.0) # Scale down for watershed

            # Update centroids for next step
            self.prev_centroids = self._calculate_centroids(labels)

        # Update State
        self.prev_labels = labels
        self.prev_gray   = curr_gray
        
        return labels, flow

    def _calculate_centroids(self, labels):
        """
        Fast centroid calculation using scipy.ndimage.
        Returns: dict {label: [y, x]}
        """
        # Get unique labels (excluding 0 if any)
        unique_labels = np.unique(labels)
        if unique_labels[0] == 0:
            unique_labels = unique_labels[1:]
            
        # center_of_mass returns list of tuples [(y,x), (y,x)...]
        centroids = center_of_mass(np.ones_like(labels), labels, unique_labels)
        
        # Map label -> centroid
        return {lbl: np.array(c) for lbl, c in zip(unique_labels, centroids)}'''

class TemporalSLIC_DEM:
    def __init__(self, dem_norm: np.ndarray, n_segments=150, 
                 compactness=10.0, lambda_z=0.5):
        """
        Args:
            dem_norm: Normalized DEM (0-1).
            n_segments: Target number of superpixels.
            compactness: Balances color proximity vs. space proximity.
            lambda_z: Weight of elevation (Z-dimension) relative to XY space.
        """
        self.n_segments   = n_segments
        self.compactness  = compactness
        self.lambda_z     = lambda_z
        self.dem_norm     = dem_norm
        
        # State tracking
        self.prev_labels    = None
        self.prev_centroids = None 
        self.prev_gray      = None

    def segment(self, frame: np.ndarray, use_flow: bool = True):
        H, W = frame.shape
        
        # Normalize Frame (0-1)
        mn, mx = frame.min(), frame.max()
        frame_norm = (frame - mn) / (mx - mn) if mx - mn > 0 else np.zeros_like(frame)
        curr_gray = (frame_norm * 255).astype(np.uint8)
        
        # ---------------------------------------------------------
        # FEATURE FUSION: Create a 3D Coordinate Space (X, Y, Z)
        # ---------------------------------------------------------
        # We scale the DEM by lambda_z to treat it as a physical dimension.
        # This prevents pixels on opposite sides of a ridge from being "close".
        elevation_layer = self.dem_norm * self.lambda_z

        if self.prev_labels is None:
            # Multi-channel SLIC: [Intensity, Elevation]
            # By treating elevation as a channel, SLIC uses it in the distance metric.
            features = np.dstack([frame_norm, elevation_layer])
            
            labels = slic(features, 
                          n_segments=self.n_segments,
                          compactness=self.compactness,
                          start_label=1,
                          enforce_connectivity=True,
                          channel_axis=-1)
            
            self.prev_centroids = self._calculate_centroids(labels)

        else:
            if use_flow:
                flow = cv2.calcOpticalFlowFarneback(
                    self.prev_gray, curr_gray, None, 
                    pyr_scale=0.5, levels=3, winsize=15, 
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
            else:
                flow = np.zeros((H, W, 2), dtype=np.float32)
            
            # Advect Centroids
            advected_markers = np.zeros((H, W), dtype=np.int32)
            for lbl, centroid in self.prev_centroids.items():
                y_old, x_old = int(centroid[0]), int(centroid[1])
                if 0 <= y_old < H and 0 <= x_old < W:
                    # Using local window mean flow for stability
                    window = 5
                    y1, y2 = max(0, y_old-window), min(H, y_old+window)
                    x1, x2 = max(0, x_old-window), min(W, x_old+window)
                    
                    flow_y = np.mean(flow[y1:y2, x1:x2, 1])
                    flow_x = np.mean(flow[y1:y2, x1:x2, 0])
                    
                    y_new = int(np.clip(y_old + flow_y, 0, H-1))
                    x_new = int(np.clip(x_old + flow_x, 0, W-1))
                    
                    advected_markers[y_new, x_new] = lbl

            # ---------------------------------------------------------
            # 3D COMPACT WATERSHED
            # ---------------------------------------------------------
            # Instead of just adding gradients, we use the image gradient
            # but preserve the "compactness" which acts like a spatial anchor.
            grad_img = sobel(frame_norm)
            
            # We treat the elevation as a "cost" surface. 
            # High elevation gradients act as walls that Watershed cannot easily cross.
            grad_dem = sobel(elevation_layer)
            combined_gradient = np.hypot(grad_img, grad_dem)
            
            labels = watershed(combined_gradient, 
                               markers=advected_markers, 
                               compactness=self.compactness / 100.0)

            self.prev_centroids = self._calculate_centroids(labels)

        self.prev_labels = labels
        self.prev_gray = curr_gray
        return labels, (flow if 'flow' in locals() else np.zeros((H, W, 2)))

    def _calculate_centroids(self, labels):
        unique_labels = np.unique(labels)
        if unique_labels[0] == 0: unique_labels = unique_labels[1:]
        centroids = center_of_mass(np.ones_like(labels), labels, unique_labels)
        return {lbl: np.array(c) for lbl, c in zip(unique_labels, centroids)}
    
# --------------------------------------------------------------------------- #
# DATA LOADING  — FIX: filter catalog by BOTH event id AND img_type per channel
# --------------------------------------------------------------------------- #
def _get_channel_path(event_id: str, img_type: str,
                      catalog: pd.DataFrame) -> str | None:
    """
    Look up the file path for a specific (event_id, img_type) pair.

    SEVIR filenames embed the channel name:
        SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5
        SEVIR_IR107_STORMEVENTS_2019_0101_0630.h5

    The catalog has one row per (event, channel). Using only
    catalog[id == event_id].iloc[0] for all channels returns the VIL row's
    filename for every channel, so BASE_PATH/ir107/<VIL_filename>.h5
    does not exist on disk.

    Fix: filter by img_type as well so each channel gets its own filename.
    """
    rows = catalog[(catalog["id"] == event_id) & (catalog["img_type"] == img_type)]
    if rows.empty:
        return None
    return os.path.join(BASE_PATH, img_type, os.path.basename(rows.iloc[0]["file_name"]))


def _read_hdf5(path: str, event_id: str, img_type: str) -> np.ndarray | None:
    """Extract this event's data array from an HDF5 file."""
    if not os.path.exists(path):
        log.warning(f"  File not found: {path}")
        return None
    with h5py.File(path, "r") as f:
        # Case 1: event stored as a top-level key
        if event_id in f:
            return f[event_id][:]
        # Case 2: flat array layout — find row index from 'id' dataset
        if "id" in f and img_type in f:
            ids = f["id"][:]
            if isinstance(ids[0], bytes):
                ids = [x.decode() for x in ids]
            hit = np.where(np.array(ids) == event_id)[0]
            if len(hit):
                return f[img_type][hit[0]]
    log.warning(f"  Event {event_id} not found inside {path}")
    return None


def load_channels(event_id: str, catalog: pd.DataFrame):
    """
    Load VIL and IR107 for the given event.

    Returns:
        data   : {ch: ndarray shape (T, H, W), float32}
        extent : [llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat]
    """
    # Use VIL row for the geographic extent (it's identical across channels)
    meta_rows = catalog[catalog["id"] == event_id]
    if meta_rows.empty:
        log.error(f"Event {event_id} not found in catalog.")
        return {}, []

    extent_row = meta_rows.iloc[0]
    extent = [
        extent_row["llcrnrlon"], extent_row["urcrnrlon"],
        extent_row["llcrnrlat"], extent_row["urcrnrlat"],
    ]

    data = {}

    for ch in ["vil", "ir107"]:
        # Per-channel filename lookup — the key fix
        path = _get_channel_path(event_id, ch, catalog)
        if path is None:
            log.warning(f"  {ch}: no catalog entry for this event — skipping.")
            continue

        log.info(f"  {ch}: resolved path → {path}")
        raw = _read_hdf5(path, event_id, ch)
        if raw is None:
            continue

        log.info(f"  {ch}: raw shape = {raw.shape}, dtype = {raw.dtype}")

        if ch == "vil":
            # SEVIR VIL raw shape: (H, W, T) — reorder to (T, H, W)
            if raw.ndim == 3:
                # Axis 2 is T when T < H (T=49, H=384)
                arr = raw.transpose(2, 0, 1) if raw.shape[2] < raw.shape[0] else raw
            else:
                arr = raw
        else:
            # IR107 raw shape: (H_ir, W_ir, T) or (T, H_ir, W_ir)
            # Normalise to (T, H, W) then upsample to match VIL spatial size
            if raw.ndim == 3:
                arr_tfw = raw.transpose(2, 0, 1) if raw.shape[2] < raw.shape[0] else raw
            else:
                arr_tfw = raw

            T_ir, H_ir, W_ir = arr_tfw.shape
            tgt_h, tgt_w = (data["vil"].shape[1:] if "vil" in data else (384, 384))

            if H_ir != tgt_h or W_ir != tgt_w:
                log.info(f"  {ch}: upsampling {H_ir}×{W_ir} → {tgt_h}×{tgt_w}")
                arr = np.empty((T_ir, tgt_h, tgt_w), dtype=np.float32)
                for t in range(T_ir):
                    arr[t] = cv2.resize(arr_tfw[t].astype(np.float32),
                                        (tgt_w, tgt_h), interpolation=cv2.INTER_CUBIC)
            else:
                arr = arr_tfw

        data[ch] = arr.astype(np.float32)
        log.info(f"  {ch}: loaded — shape={data[ch].shape}, "
                 f"range=[{data[ch].min():.1f}, {data[ch].max():.1f}]")

    return data, extent

# --------------------------------------------------------------------------- #
# COLOUR SCALE HELPER
# --------------------------------------------------------------------------- #
def resolve_vmin_vmax(ch: str, arr: np.ndarray):
    cfg = CHANNEL_CFG[ch]
    if cfg["vmin"] is not None:
        return cfg["vmin"], cfg["vmax"]
    valid = arr[arr != 0]
    if valid.size == 0: return 0.0, 1.0
    return float(valid.min()), float(valid.max())

# --------------------------------------------------------------------------- #
# FRAME → RGB  +  boundary burn
# --------------------------------------------------------------------------- #
def frame_to_rgb(frame: np.ndarray, ch: str,
                 vmin: float, vmax: float) -> np.ndarray:
    norm = np.clip((frame - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
    rgba = CHANNEL_CFG[ch]["cmap"](norm)
    return rgba[:, :, :3].astype(np.float32)

def render_boundary(rgb: np.ndarray, labels: np.ndarray,
                    color: tuple) -> np.ndarray:
    out = mark_boundaries(rgb, labels, color=color, mode="thick")
    return (out * 255).clip(0, 255).astype(np.uint8)

# --------------------------------------------------------------------------- #
# PRE-COMPUTATION
# --------------------------------------------------------------------------- #
def precompute(data: dict, dem_norm: np.ndarray):
    """
    Run TemporalSLIC_DEM sequentially for both modes and channels.
    All boundary images pre-rendered to uint8 RGB — animation just calls set_data().

    Returns:
        centroids_store : {ch: {use_flow: list[dict]}}
        flows_store     : {ch: list[ndarray(H,W,2)]}   — Temporal mode only
        rendered_store  : {ch: {use_flow: ndarray(T,H,W,3) uint8}}
    """
    centroids_store = {ch: {} for ch in data}
    flows_store     = {ch: [] for ch in data}
    rendered_store  = {ch: {} for ch in data}

    for ch, arr in data.items():
        T = arr.shape[0]
        vmin, vmax = resolve_vmin_vmax(ch, arr)
        log.info(f"\n  {ch.upper()}: {T} frames | vmin={vmin:.1f} vmax={vmax:.1f}")

        for use_flow in [True, False]:
            style = MODE_STYLE[use_flow]
            tslic = TemporalSLIC_DEM(dem_norm, n_segments=N_SEGMENTS,
                                     compactness=COMPACTNESS, lambda_z=ELEVATION_LAMBDA)

            centroids_list = []
            flows_list     = []
            rendered_list  = []

            for t in tqdm(range(T), desc=f"  {ch.upper()} [{style['label']}]"):
                labels, flow = tslic.segment(arr[t], use_flow=use_flow)
                centroids_list.append(dict(tslic.prev_centroids))
                flows_list.append(flow)

                rgb      = frame_to_rgb(arr[t], ch, vmin, vmax)
                rendered = render_boundary(rgb, labels, style["boundary"])
                rendered_list.append(rendered)

            centroids_store[ch][use_flow] = centroids_list
            rendered_store[ch][use_flow]  = np.stack(rendered_list, axis=0)

            if use_flow:
                flows_store[ch] = flows_list

    return centroids_store, flows_store, rendered_store

# --------------------------------------------------------------------------- #
# WRITER
# --------------------------------------------------------------------------- #
def get_writer():
    if shutil.which("ffmpeg"):
        log.info("ffmpeg found — saving as .mp4")
        return animation.FFMpegWriter(fps=5, bitrate=2000), ".mp4"
    log.info("ffmpeg not found — saving as .gif (PillowWriter)")
    return animation.PillowWriter(fps=5), ".gif"

# --------------------------------------------------------------------------- #
# ANIMATION
# --------------------------------------------------------------------------- #
def animate_slic(event_id: str, data: dict,
                 rendered_store: dict, centroids_store: dict,
                 flows_store: dict):
    channels = list(data.keys())
    n_cols   = len(channels)
    T        = max(data[ch].shape[0] for ch in channels)
    total_min = T * SECONDS_PER_FRAME // 60

    writer, ext = get_writer()
    dpi      = 80 if ext == ".gif" else 120
    out_path = os.path.join(EVENTS_DIR, f"{event_id}_slic_{N_SEGMENTS}{ext}")

    log.info(f"\nAnimating {T} frames ({total_min} min) → {out_path}")

    fig, axes = plt.subplots(
        nrows=2, ncols=n_cols,
        figsize=(6 * n_cols, 10),
        gridspec_kw={"hspace": 0.08, "wspace": 0.06}
    )
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for col_idx, ch in enumerate(channels):
        axes[0, col_idx].set_title(ch.upper(), fontsize=13, fontweight="bold", pad=8)
    for row_idx, use_flow in enumerate([True, False]):
        axes[row_idx, 0].set_ylabel(MODE_STYLE[use_flow]["label"], fontsize=11, labelpad=8)

    time_text = fig.text(
        0.5, 0.01, '', ha='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )
    fig.suptitle(
        f"Superpixel Segmentation — Event {event_id}  ({total_min} min)\n"
        f"Gold = Temporal SLIC   |   Cyan = Standard SLIC",
        fontsize=12, y=0.99
    )

    # Initialise artists with frame 0
    im_artists = {}
    sc_artists = {}
    qv_artists = {}

    for row_idx, use_flow in enumerate([True, False]):
        for col_idx, ch in enumerate(channels):
            ax  = axes[row_idx, col_idx]
            im  = ax.imshow(rendered_store[ch][use_flow][0],
                            origin="lower", animated=True, aspect="auto")
            ax.axis("off")
            im_artists[(row_idx, col_idx)] = im
            sc_artists[(row_idx, col_idx)] = None
            qv_artists[(row_idx, col_idx)] = None

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    def _update(t):
        updated = []
        for row_idx, use_flow in enumerate([True, False]):
            for col_idx, ch in enumerate(channels):
                ax  = axes[row_idx, col_idx]
                key = (row_idx, col_idx)

                T_ch   = rendered_store[ch][use_flow].shape[0]
                t_safe = min(t, T_ch - 1)

                # 1. Boundary image — fast pixel swap
                im_artists[key].set_data(rendered_store[ch][use_flow][t_safe])
                updated.append(im_artists[key])

                H, W = data[ch].shape[1], data[ch].shape[2]

                # 2. Centroid scatter — remove old, place new
                if sc_artists[key] is not None:
                    sc_artists[key].remove()
                cents = centroids_store[ch][use_flow][t_safe]
                if cents:
                    cy = np.array([v[0] for v in cents.values()])
                    cx = np.array([v[1] for v in cents.values()])
                    sc = ax.scatter(cx, cy,
                                    c=MODE_STYLE[use_flow]["centroid"],
                                    s=6, linewidths=0, zorder=5)
                    sc_artists[key] = sc
                    updated.append(sc)
                else:
                    sc_artists[key] = None

                # 3. Flow quiver — Temporal panels only
                if use_flow:
                    if qv_artists[key] is not None:
                        qv_artists[key].remove()
                    flow = flows_store[ch][t_safe]
                    ys   = np.arange(0, H, QUIVER_STRIDE)
                    xs   = np.arange(0, W, QUIVER_STRIDE)
                    xg, yg = np.meshgrid(xs, ys)
                    u = flow[yg, xg, 0]
                    v = flow[yg, xg, 1]
                    valid = np.hypot(u, v) > 0.5
                    if valid.any():
                        qv = ax.quiver(xg[valid], yg[valid], u[valid], v[valid],
                                       color="white", alpha=0.55,
                                       scale=120, width=0.003, zorder=4)
                        qv_artists[key] = qv
                        updated.append(qv)
                    else:
                        qv_artists[key] = None

        elapsed_min = t * SECONDS_PER_FRAME // 60
        time_text.set_text(
            f"T+{elapsed_min:03d} min   frame {t+1}/{T}   ({SECONDS_PER_FRAME}s/frame)"
        )
        updated.append(time_text)
        return updated

    anim = animation.FuncAnimation(
        fig, _update, frames=T, interval=200, blit=False
    )

    log.info(f"Writing animation at dpi={dpi} ...")
    anim.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    log.info(f"Saved: {out_path}")
    return out_path

# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if not os.path.exists(CATALOG_PATH):
        print("Catalog not found!"); exit()

    print("Loading catalog ...")
    catalog = pd.read_csv(CATALOG_PATH, parse_dates=["time_utc"], low_memory=False)

    while True:
        print("\n" + "=" * 50)
        event_id = input("Enter Event ID (or 'q' to quit): ").strip()
        if event_id.lower() == "q": break
        if not event_id: continue

        print(f"\nLoading data for {event_id} ...")
        data, extent = load_channels(event_id, catalog)
        if not data:
            print("  [Error] No data found. Check event ID and file paths.")
            continue

        T = max(arr.shape[0] for arr in data.values())
        print(f"  Event duration: {T} frames = {T * SECONDS_PER_FRAME // 60} min")

        print("\nFetching DEM ...")
        dem_raw  = fetch_and_regrid_dem(extent)
        dem_norm = (dem_raw - dem_raw.min()) / (dem_raw.max() - dem_raw.min() + 1e-6)

        print("\nPre-computing segmentation (both modes × both channels) ...")
        centroids_store, flows_store, rendered_store = precompute(data, dem_norm)

        out = animate_slic(event_id, data, rendered_store, centroids_store, flows_store)
        print(f"\nDone → {out}")