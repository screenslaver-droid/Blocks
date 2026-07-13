"""
visualize_slic_sevir.py  —  Fused Multi-Channel Temporal SLIC on SEVIR
=======================================================================
A SINGLE "Master Superpixel" segmentation is computed on the fused
(VIL + IR107 + IR069 + DEM) feature cube and overlaid on every display panel.

Layout — 2 × N grid  (N = number of available display channels):
  ┌──────────┬──────────┬──────────┐
  │  VIL     │  IR107   │  IR069   │   ← Temporal SLIC (flow-guided watershed)
  ├──────────┼──────────┼──────────┤
  │  VIL     │  IR107   │  IR069   │   ← Standard SLIC (stationary watershed)
  └──────────┴──────────┴──────────┘

Key changes from the channel-independent version
-------------------------------------------------
1. `load_fused_channels` normalises VIL / IR107 / IR069 to [0, 1] and stacks
   them into a (T, H, W, C) feature cube before segmentation.
2. `TemporalSLIC_DEM.segment` now accepts (H, W, C) input:
     • Frame 0  : SLIC in (C+1)-D spectral space with DEM as the final channel.
     • Frame t+ : Optical flow computed on a VIL grayscale proxy; watershed
                  gradient is the pixel-wise MAX of per-channel Sobel magnitudes
                  plus the DEM Sobel — so boundaries snap to the sharpest edge
                  found in ANY spectral band.
3. `precompute` runs ONE segmentation per mode and renders the resulting label
   map onto every display channel, giving a consistent "Master Boundary" set.
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from skimage.segmentation import slic, watershed, mark_boundaries
from skimage.filters import sobel
from scipy.ndimage import center_of_mass
import cv2
import os
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.interpolate import RegularGridInterpolator
import cartopy.crs as ccrs
from tqdm import tqdm

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
BASE_PATH    = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\SEVIR\2019"
CATALOG_PATH = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\SEVIR\CATALOG.csv"
EVENTS_DIR   = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\Events"

N_SEGMENTS        = 950
COMPACTNESS       = 10
ELEVATION_LAMBDA  = 0.5
SECONDS_PER_FRAME = 300      # 5 min per SEVIR frame
QUIVER_STRIDE     = 24       # show 1 flow arrow every N pixels

os.makedirs(EVENTS_DIR, exist_ok=True)

SEVIR_PROJ = ccrs.LambertConformal(
    central_longitude=-98.0, central_latitude=38.0,
    standard_parallels=(30.0, 60.0),
    globe=ccrs.Globe(semimajor_axis=6370000, semiminor_axis=6370000),
)

# Channels rendered in the visualization (one column each)
DISPLAY_CHANNELS = ["vil", "ir107", "ir069"]

# Channels stacked into the fused feature cube for segmentation (order matters)
FUSION_CHANNELS = ["vil", "ir107", "ir069"]

CHANNEL_CFG = {
    "vil":   {"cmap": plt.get_cmap("jet"),    "vmin": 0,    "vmax": 255},
    "ir107": {"cmap": plt.get_cmap("gray_r"), "vmin": None, "vmax": None},
    "ir069": {"cmap": plt.get_cmap("gray_r"), "vmin": None, "vmax": None},
}

MODE_STYLE = {
    True:  {"boundary": (1.0, 0.75, 0.0), "centroid": "red",   "label": "Temporal SLIC"},
    False: {"boundary": (0.0, 0.90, 1.0), "centroid": "white", "label": "Standard SLIC"},
}

# --------------------------------------------------------------------------- #
# DEM PIPELINE  (unchanged)
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
    da = da.rio.clip_box(minx=extent[0] - buf, miny=extent[2] - buf,
                         maxx=extent[1] + buf, maxy=extent[3] + buf)
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
        lats = lats[::-1]
        vals = vals[::-1, :]

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
# TEMPORAL SLIC — Fused Multi-Channel Version
# --------------------------------------------------------------------------- #
class TemporalSLIC_DEM:
    def __init__(self, dem_norm: np.ndarray, n_segments: int = 150,
                 compactness: float = 10.0, lambda_z: float = 0.5):
        """
        Args:
            dem_norm   : Normalized DEM (H, W), values in [0, 1].
            n_segments : Target number of superpixels.
            compactness: Balances spectral vs. spatial proximity in SLIC.
            lambda_z   : Weight of the elevation channel in SLIC + gradient.
        """
        self.n_segments  = n_segments
        self.compactness = compactness
        self.lambda_z    = lambda_z

        # DEM kept for both SLIC channel appending and gradient computation
        self.dem_norm     = dem_norm                     # (H, W)  raw, for SLIC
        self.dem_weighted = dem_norm * lambda_z          # (H, W)  for gradient

        # Lagrangian state
        self.prev_labels    = None
        self.prev_centroids = None   # {label: [y, x]}
        self.prev_gray      = None   # uint8 grayscale for Farneback

    # ---------------------------------------------------------------------- #
    def segment(self, fused_frame: np.ndarray, use_flow: bool = True):
        """
        Segment one time-step of the fused feature cube.

        Args:
            fused_frame : (H, W, C) float32, each channel already in [0, 1].
                          Channels correspond to FUSION_CHANNELS (VIL, IR107, IR069…).
                          DEM is appended internally — do NOT include it here.
            use_flow    : If True (and not frame 0) advect centroids with
                          Farneback optical flow; otherwise centroids are frozen.

        Returns:
            labels : (H, W) int32  superpixel label map.
            flow   : (H, W, 2) float32  optical-flow field
                     (zeros on frame 0 or when use_flow=False).
        """
        # Gracefully handle a plain 2-D (single-channel) input
        if fused_frame.ndim == 2:
            fused_frame = fused_frame[:, :, np.newaxis]

        H, W, C = fused_frame.shape

        # ---- Grayscale proxy for optical flow --------------------------------
        # VIL (channel 0) carries the dominant storm motion signal.
        gray_norm = fused_frame[:, :, 0]
        curr_gray = (gray_norm * 255).astype(np.uint8)

        # ---- Build the full feature cube: satellite channels + DEM -----------
        # DEM is scaled by lambda_z so its influence matches the user weight.
        dem_ch       = (self.dem_norm * self.lambda_z)[:H, :W, np.newaxis]
        feature_cube = np.concatenate([fused_frame, dem_ch], axis=-1).astype(np.float64)
        # Shape: (H, W, C+1)

        flow = np.zeros((H, W, 2), dtype=np.float32)

        # ================================================================== #
        # CASE A: First Frame — initialize with multi-channel SLIC           #
        # ================================================================== #
        if self.prev_labels is None:
            # SLIC clusters in (C+1)-dimensional spectral space plus 2-D spatial.
            # channel_axis=-1 tells skimage the last axis is the channel axis.
            labels = slic(
                feature_cube,
                n_segments=self.n_segments,
                compactness=self.compactness,
                start_label=1,
                enforce_connectivity=True,
                channel_axis=-1,
            )
            self.prev_centroids = self._calculate_centroids(labels)

        # ================================================================== #
        # CASE B: Subsequent Frames — Lagrangian advection + Watershed       #
        # ================================================================== #
        else:
            # 1. Dense Optical Flow (Farneback on VIL grayscale proxy)
            if use_flow:
                flow = cv2.calcOpticalFlowFarneback(
                    self.prev_gray, curr_gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )

            # 2. Advect centroids (Lagrangian step)
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

            # 3. Fused gradient: pixel-wise MAX of per-channel Sobel magnitudes
            #    plus the DEM Sobel.  Boundaries snap to the sharpest edge in
            #    ANY spectral band — not just the primary display channel.
            grad_layers = [sobel(fused_frame[:, :, c]) for c in range(C)]
            grad_layers.append(sobel(self.dem_weighted))     # DEM contribution
            combined_gradient = np.max(np.stack(grad_layers, axis=0), axis=0)

            # 4. Compact Watershed from advected markers
            labels = watershed(
                combined_gradient,
                markers=advected_markers,
                compactness=self.compactness / 100.0,
            )

            self.prev_centroids = self._calculate_centroids(labels)

        # Update Lagrangian state
        self.prev_labels = labels
        self.prev_gray   = curr_gray
        return labels, flow

    # ---------------------------------------------------------------------- #
    def _calculate_centroids(self, labels: np.ndarray) -> dict:
        """Return {label: np.array([y, x])} using scipy.ndimage for speed."""
        unique_labels = np.unique(labels)
        if len(unique_labels) and unique_labels[0] == 0:
            unique_labels = unique_labels[1:]
        centroids = center_of_mass(np.ones_like(labels), labels, unique_labels)
        return {lbl: np.array(c) for lbl, c in zip(unique_labels, centroids)}


# --------------------------------------------------------------------------- #
# DATA LOADING
# --------------------------------------------------------------------------- #
def _get_channel_path(event_id: str, img_type: str,
                      catalog: pd.DataFrame) -> str | None:
    """
    Return the HDF5 file path for a specific (event_id, img_type) pair.
    Filtering by img_type prevents returning the VIL filename for IR channels.
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


def _load_single_channel(event_id: str, ch: str,
                          catalog: pd.DataFrame,
                          tgt_shape: tuple) -> np.ndarray | None:
    """
    Load one channel, reorder axes to (T, H, W), and upsample to tgt_shape
    (H, W) if the native resolution differs.
    """
    path = _get_channel_path(event_id, ch, catalog)
    if path is None:
        log.warning(f"  {ch}: no catalog entry — skipping.")
        return None

    log.info(f"  {ch}: resolved path → {path}")
    raw = _read_hdf5(path, event_id, ch)
    if raw is None:
        return None

    log.info(f"  {ch}: raw shape={raw.shape}, dtype={raw.dtype}")

    # Normalise axis order to (T, H, W)
    if raw.ndim == 3:
        arr = raw.transpose(2, 0, 1) if raw.shape[2] < raw.shape[0] else raw
    else:
        arr = raw

    T_ch, H_ch, W_ch = arr.shape
    tgt_h, tgt_w = tgt_shape

    if H_ch != tgt_h or W_ch != tgt_w:
        log.info(f"  {ch}: upsampling {H_ch}×{W_ch} → {tgt_h}×{tgt_w}")
        out = np.empty((T_ch, tgt_h, tgt_w), dtype=np.float32)
        for t in range(T_ch):
            out[t] = cv2.resize(arr[t].astype(np.float32),
                                (tgt_w, tgt_h), interpolation=cv2.INTER_CUBIC)
        arr = out

    result = arr.astype(np.float32)
    log.info(f"  {ch}: loaded — shape={result.shape}, "
             f"range=[{result.min():.1f}, {result.max():.1f}]")
    return result


def load_channels(event_id: str, catalog: pd.DataFrame):
    """
    Load VIL, IR107, and IR069 for the given event.

    Returns:
        data   : {ch: ndarray (T, H, W) float32}
        extent : [llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat]
    """
    meta_rows = catalog[catalog["id"] == event_id]
    if meta_rows.empty:
        log.error(f"Event {event_id} not found in catalog.")
        return {}, []

    extent_row = meta_rows.iloc[0]
    extent = [
        extent_row["llcrnrlon"], extent_row["urcrnrlon"],
        extent_row["llcrnrlat"], extent_row["urcrnrlat"],
    ]

    data      = {}
    tgt_shape = (384, 384)

    # VIL loaded first so remaining channels can upsample to its exact size
    vil = _load_single_channel(event_id, "vil", catalog, tgt_shape)
    if vil is not None:
        data["vil"] = vil
        tgt_shape   = (vil.shape[1], vil.shape[2])

    for ch in ["ir107", "ir069"]:
        arr = _load_single_channel(event_id, ch, catalog, tgt_shape)
        if arr is not None:
            data[ch] = arr

    return data, extent


def load_fused_channels(event_id: str, catalog: pd.DataFrame):
    """
    Load all satellite channels, normalise each to [0, 1], and stack into a
    fused (T, H, W, C) feature cube ready for multi-channel SLIC.

    Returns:
        fused  : ndarray (T, H, W, C) float32  — normalised feature cube
        data   : {ch: ndarray (T, H, W) float32} — raw arrays for display
        extent : list[float]
    """
    data, extent = load_channels(event_id, catalog)
    if not data:
        return None, {}, []

    fused_stack = []
    for ch in FUSION_CHANNELS:
        if ch in data:
            arr = data[ch]
            mn, mx = arr.min(), arr.max()
            norm = (arr - mn) / (mx - mn + 1e-6)   # (T, H, W) → [0, 1]
            fused_stack.append(norm)
        else:
            log.warning(f"  Fusion channel '{ch}' unavailable — omitting from cube.")

    if not fused_stack:
        log.error("  No fusion channels available — cannot build feature cube.")
        return None, data, extent

    # Shape: (T, H, W, C)
    fused = np.stack(fused_stack, axis=-1).astype(np.float32)
    log.info(f"  Fused feature cube: {fused.shape}  ({len(fused_stack)} channels: "
             f"{[ch for ch in FUSION_CHANNELS if ch in data]})")
    return fused, data, extent


# --------------------------------------------------------------------------- #
# COLOUR SCALE HELPER
# --------------------------------------------------------------------------- #
def resolve_vmin_vmax(ch: str, arr: np.ndarray):
    cfg = CHANNEL_CFG[ch]
    if cfg["vmin"] is not None:
        return cfg["vmin"], cfg["vmax"]
    valid = arr[arr != 0]
    if valid.size == 0:
        return 0.0, 1.0
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
# PRE-COMPUTATION  —  Fused single-segmentation
# --------------------------------------------------------------------------- #
def precompute(fused: np.ndarray, data: dict, dem_norm: np.ndarray):
    """
    Run ONE TemporalSLIC_DEM per mode on the fused (T, H, W, C) feature cube.
    The resulting label maps are rendered onto each display channel separately,
    producing consistent "Master Boundaries" across all panels.

    Returns:
        centroids_store : {use_flow: list[dict]}           — shared across channels
        flows_store     : list[ndarray(H,W,2)]             — Temporal mode only
        rendered_store  : {ch: {use_flow: ndarray(T,H,W,3) uint8}}
    """
    T           = fused.shape[0]
    display_chs = [ch for ch in DISPLAY_CHANNELS if ch in data]

    # Pre-resolve colour ranges for every display channel
    vranges = {ch: resolve_vmin_vmax(ch, data[ch]) for ch in display_chs}

    centroids_store: dict  = {}
    flows_store:     list  = []
    rendered_store:  dict  = {ch: {} for ch in display_chs}

    for use_flow in [True, False]:
        style = MODE_STYLE[use_flow]
        tslic = TemporalSLIC_DEM(
            dem_norm,
            n_segments=N_SEGMENTS,
            compactness=COMPACTNESS,
            lambda_z=ELEVATION_LAMBDA,
        )

        centroids_list: list = []
        labels_list:    list = []
        flow_list:      list = []

        log.info(f"\n  [{style['label']}] segmenting {T} frames on fused cube …")
        for t in tqdm(range(T), desc=f"  Fused [{style['label']}]"):
            labels, flow = tslic.segment(fused[t], use_flow=use_flow)
            centroids_list.append(dict(tslic.prev_centroids))
            labels_list.append(labels)
            flow_list.append(flow)

        centroids_store[use_flow] = centroids_list
        if use_flow:
            flows_store = flow_list   # only Temporal mode needs quiver arrows

        # Render the SAME label map onto each display channel
        for ch in display_chs:
            vmin, vmax    = vranges[ch]
            rendered_list = []
            for t in range(T):
                rgb      = frame_to_rgb(data[ch][t], ch, vmin, vmax)
                rendered = render_boundary(rgb, labels_list[t], style["boundary"])
                rendered_list.append(rendered)
            rendered_store[ch][use_flow] = np.stack(rendered_list, axis=0)

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
                 flows_store: list):
    """
    Build and save the animated GIF / MP4.

    centroids_store : {use_flow: list[dict]}  — single set shared by all columns
    flows_store     : list[ndarray(H,W,2)]    — Temporal mode only
    rendered_store  : {ch: {use_flow: ndarray(T,H,W,3)}}
    """
    display_chs = [ch for ch in DISPLAY_CHANNELS if ch in data]
    n_cols      = len(display_chs)
    T           = max(data[ch].shape[0] for ch in display_chs)
    total_min   = T * SECONDS_PER_FRAME // 60

    writer, ext = get_writer()
    dpi      = 80 if ext == ".gif" else 120
    out_path = os.path.join(EVENTS_DIR, f"{event_id}_slic_{N_SEGMENTS}{ext}")

    log.info(f"\nAnimating {T} frames ({total_min} min) → {out_path}")

    fig, axes = plt.subplots(
        nrows=2, ncols=n_cols,
        figsize=(6 * n_cols, 10),
        gridspec_kw={"hspace": 0.08, "wspace": 0.06},
    )
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for col_idx, ch in enumerate(display_chs):
        axes[0, col_idx].set_title(ch.upper(), fontsize=13, fontweight="bold", pad=8)
    for row_idx, use_flow in enumerate([True, False]):
        axes[row_idx, 0].set_ylabel(MODE_STYLE[use_flow]["label"], fontsize=11, labelpad=8)

    time_text = fig.text(
        0.5, 0.01, "", ha="center", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )
    fig.suptitle(
        f"Fused Superpixel Segmentation — Event {event_id}  ({total_min} min)\n"
        f"Gold = Temporal SLIC   |   Cyan = Standard SLIC   |   Master Boundaries",
        fontsize=12, y=0.99,
    )

    im_artists: dict = {}
    sc_artists: dict = {}
    qv_artists: dict = {}

    for row_idx, use_flow in enumerate([True, False]):
        for col_idx, ch in enumerate(display_chs):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(rendered_store[ch][use_flow][0],
                           origin="lower", animated=True, aspect="auto")
            ax.axis("off")
            im_artists[(row_idx, col_idx)] = im
            sc_artists[(row_idx, col_idx)] = None
            qv_artists[(row_idx, col_idx)] = None

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    def _update(t):
        updated = []
        for row_idx, use_flow in enumerate([True, False]):
            for col_idx, ch in enumerate(display_chs):
                ax  = axes[row_idx, col_idx]
                key = (row_idx, col_idx)

                T_ch   = rendered_store[ch][use_flow].shape[0]
                t_safe = min(t, T_ch - 1)

                # 1. Boundary image — fast pixel swap
                im_artists[key].set_data(rendered_store[ch][use_flow][t_safe])
                updated.append(im_artists[key])

                H_ax, W_ax = data[ch].shape[1], data[ch].shape[2]

                # 2. Centroid scatter — single set shared across columns
                if sc_artists[key] is not None:
                    sc_artists[key].remove()
                cents = centroids_store[use_flow][t_safe]
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

                # 3. Flow quiver — Temporal row only, single flow field
                if use_flow:
                    if qv_artists[key] is not None:
                        qv_artists[key].remove()
                    flow = flows_store[t_safe]
                    ys   = np.arange(0, H_ax, QUIVER_STRIDE)
                    xs   = np.arange(0, W_ax, QUIVER_STRIDE)
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
            f"T+{elapsed_min:03d} min   frame {t + 1}/{T}   ({SECONDS_PER_FRAME}s/frame)"
        )
        updated.append(time_text)
        return updated

    anim = animation.FuncAnimation(
        fig, _update, frames=T, interval=200, blit=False,
    )

    log.info(f"Writing animation at dpi={dpi} …")
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

    print("Loading catalog …")
    catalog = pd.read_csv(CATALOG_PATH, parse_dates=["time_utc"], low_memory=False)

    while True:
        print("\n" + "=" * 50)
        event_id = input("Enter Event ID (or 'q' to quit): ").strip()
        if event_id.lower() == "q":
            break
        if not event_id:
            continue

        print(f"\nLoading data for {event_id} …")
        fused, data, extent = load_fused_channels(event_id, catalog)
        if fused is None or not data:
            print("  [Error] No data found. Check event ID and file paths.")
            continue

        T = fused.shape[0]
        print(f"  Event duration: {T} frames = {T * SECONDS_PER_FRAME // 60} min")
        print(f"  Feature cube  : {fused.shape}  (T × H × W × C)")

        print("\nFetching DEM …")
        dem_raw  = fetch_and_regrid_dem(extent)
        dem_norm = (dem_raw - dem_raw.min()) / (dem_raw.max() - dem_raw.min() + 1e-6)

        print("\nPre-computing fused segmentation (both modes) …")
        centroids_store, flows_store, rendered_store = precompute(fused, data, dem_norm)

        out = animate_slic(event_id, data, rendered_store, centroids_store, flows_store)
        print(f"\nDone → {out}")