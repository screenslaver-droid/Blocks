"""
visualize_slic_sevir.py  —  Adaptive Residual Temporal SLIC on Fused SEVIR Channels
=====================================================================================
Implements `AdaptiveResidualTSLIC` from the pseudocode, extended with:
  • Fused (VIL + IR107 + IR069 + DEM) feature cube for all segmentation steps.
  • Superpixel Birth / Death adaptive to storm lifecycle.
  • Residual-based error detection using optical-flow frame warping.
  • Region Adjacency Graph (RAG) construction per frame.
  • RAG edges overlaid on Temporal SLIC panels in the animation.

Layout — 2 × N grid  (N = available display channels: VIL, IR107, IR069):
  ┌──────────┬──────────┬──────────┐
  │  VIL     │  IR107   │  IR069   │   ← Adaptive Residual SLIC  (gold + RAG edges)
  ├──────────┼──────────┼──────────┤
  │  VIL     │  IR107   │  IR069   │   ← Standard SLIC  (cyan boundaries)
  └──────────┴──────────┴──────────┘

Key concepts
------------
Advection   : Each center moves by the *mean flow of all its member pixels*,
              not just the single-pixel flow at the centroid.
Death       : Centers whose fused intensity (max across channels) falls below
              THRESH_DEATH are removed — they are in empty / dissipated space.
Birth       : Where the per-pixel optical-flow prediction has high residual AND
              no existing center is within MIN_SPACING pixels, a new center is
              seeded at the blob centroid.
Refinement  : Adaptive centers are used as watershed markers on the fused
              gradient (pixel-wise max of per-channel Sobel + DEM Sobel),
              implementing "constrained SLIC" without a pixel-loop K-means.
RAG         : Built by scanning horizontal and vertical pixel adjacencies to
              find label-boundary pairs; edges carry mean fused-feature distance.
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from skimage.segmentation import slic, watershed, mark_boundaries
from skimage.filters import sobel
from skimage.measure import label as sk_label, regionprops
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

N_SEGMENTS        = 1500
COMPACTNESS       = 10.0
ELEVATION_LAMBDA  = 0.5
SECONDS_PER_FRAME = 300        # 5 min per SEVIR frame
QUIVER_STRIDE     = 24         # show 1 flow arrow every N pixels

# --- Adaptive Residual SLIC thresholds (all in normalised [0,1] space) ------
THRESH_DEATH   = 0.05    # superpixel dies if max fused intensity < this
THRESH_BIRTH   = 0.15    # residual pixel qualifies for birth if > this
MIN_SPACING    = 20      # min pixel distance between any two centers (birth guard)
SHOW_RAG_EDGES = True    # draw RAG adjacency edges on Temporal panels

os.makedirs(EVENTS_DIR, exist_ok=True)

SEVIR_PROJ = ccrs.LambertConformal(
    central_longitude=-98.0, central_latitude=38.0,
    standard_parallels=(30.0, 60.0),
    globe=ccrs.Globe(semimajor_axis=6370000, semiminor_axis=6370000),
)

# Display channels -> one column each in the animation
DISPLAY_CHANNELS = ["vil", "ir107", "ir069"]
# Channels stacked into the fused feature cube for segmentation
FUSION_CHANNELS  = ["vil", "ir107", "ir069"]

CHANNEL_CFG = {
    "vil":   {"cmap": plt.get_cmap("jet"),    "vmin": 0,    "vmax": 255},
    "ir107": {"cmap": plt.get_cmap("gray_r"), "vmin": None, "vmax": None},
    "ir069": {"cmap": plt.get_cmap("gray_r"), "vmin": None, "vmax": None},
}

MODE_STYLE = {
    True:  {"boundary": (1.0, 0.75, 0.0), "centroid": "red",   "label": "Adaptive Residual SLIC"},
    False: {"boundary": (0.0, 0.90, 1.0), "centroid": "white", "label": "Standard SLIC"},
}

# --------------------------------------------------------------------------- #
# DEM PIPELINE  (unchanged from previous version)
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
    merged   = merge_arrays(datasets) if len(datasets) > 1 else datasets[0]
    lons, lats, vals = merged.x.values, merged.y.values, merged.values
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        vals = vals[::-1, :]

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


# --------------------------------------------------------------------------- #
# ADAPTIVE RESIDUAL TEMPORAL SLIC  — Fused Multi-Channel + DEM
# --------------------------------------------------------------------------- #
class AdaptiveResidualTSLIC_DEM:
    """
    Implements AdaptiveResidualTSLIC on a fused (H, W, C) feature cube
    (VIL + IR107 + IR069) with DEM appended as an additional spectral anchor.

    Frame 0  : Grid-sampled centers -> standard multi-channel SLIC.
    Frame t+ : (A) Advect centers by mean-flow of member pixels.
               (B) Death: kill centers in dissipated regions.
                   Birth: seed centers in high-residual holes.
               (C) Refine boundaries via fused-gradient watershed.
               (D) Build Region Adjacency Graph (RAG).
    """

    def __init__(self, dem_norm: np.ndarray, n_segments: int = 1500,
                 compactness: float = 10.0, lambda_z: float = 0.5,
                 thresh_death: float = THRESH_DEATH,
                 thresh_birth: float = THRESH_BIRTH,
                 min_spacing:  int   = MIN_SPACING):
        self.n_segments   = n_segments
        self.compactness  = compactness
        self.lambda_z     = lambda_z
        self.thresh_death = thresh_death
        self.thresh_birth = thresh_birth
        self.min_spacing  = min_spacing

        self.dem_norm     = dem_norm              # (H, W) normalised, for SLIC
        self.dem_weighted = dem_norm * lambda_z   # (H, W) for gradient

        # Lagrangian state
        self.prev_labels  = None   # (H, W) int32
        self.prev_centers = None   # np.ndarray (K, 2) — [y, x] per row
        self.prev_fused   = None   # (H, W, C) float32 — previous fused frame
        self.prev_gray    = None   # (H, W)    uint8   — VIL proxy for Farneback

    # ------------------------------------------------------------------ #
    # PUBLIC
    # ------------------------------------------------------------------ #
    def segment(self, fused_frame: np.ndarray, use_flow: bool = True):
        """
        Segment one time-step of the fused feature cube.

        Args:
            fused_frame : (H, W, C) float32, each channel in [0, 1].
                          DEM is appended internally.
            use_flow    : If True (and not frame 0), advect via Farneback
                          optical flow and compute warp residual.

        Returns:
            labels : (H, W) int32        superpixel label map.
            flow   : (H, W, 2) float32   optical-flow field.
            rag    : dict  {'nodes': {lbl: feature_vec},
                            'edges': [(lbl_a, lbl_b, weight), ...]}
        """
        if fused_frame.ndim == 2:
            fused_frame = fused_frame[:, :, np.newaxis]

        H, W, C = fused_frame.shape

        # VIL (channel 0) as grayscale proxy for Farneback
        curr_gray = (fused_frame[:, :, 0] * 255).astype(np.uint8)

        # (C+1)-channel feature cube: satellite channels + DEM
        dem_ch       = self.dem_norm[:H, :W, np.newaxis] * self.lambda_z
        feature_cube = np.concatenate([fused_frame, dem_ch], axis=-1).astype(np.float64)

        flow = np.zeros((H, W, 2), dtype=np.float32)

        # ============================================================== #
        # CASE A: Frame 0 — multi-channel SLIC initialisation            #
        # ============================================================== #
        if self.prev_labels is None:
            labels = slic(
                feature_cube,
                n_segments=self.n_segments,
                compactness=self.compactness,
                start_label=1,
                enforce_connectivity=True,
                channel_axis=-1,
            )
            self.prev_centers = self._labels_to_centers(labels)
            self.prev_labels  = labels
            self.prev_fused   = fused_frame
            self.prev_gray    = curr_gray
            rag = self._build_rag(labels, fused_frame)
            return labels, flow, rag

        # ============================================================== #
        # CASE B+: Adaptive Residual Temporal Update                     #
        # ============================================================== #

        # --- A. Dense Optical Flow (Farneback on VIL proxy) ------------
        if use_flow:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )

        # --- A. Advect each center by the MEAN flow of its member pixels
        #        This is the key difference from single-centroid-point flow.
        new_centers = self._advect_centers(flow, H, W)

        # --- B. Residual: warp prev fused frame, diff with current -----
        pred_fused = self._warp_fused(self.prev_fused, flow)           # (H,W,C)
        # Max residual across channels: high if ANY band shows surprise
        residual   = np.max(np.abs(fused_frame - pred_fused), axis=-1) # (H,W)

        # --- B. Death: remove centers in dissipated / empty space ------
        new_centers = self._apply_death(new_centers, fused_frame, H, W)

        # --- B. Birth: seed centers in high-residual uncovered holes ---
        new_centers = self._apply_birth(new_centers, fused_frame, residual, H, W)

        # --- C. Refinement via Compact Watershed (Constrained SLIC) ----
        labels = self._refine_watershed(fused_frame, new_centers, H, W)

        # Recompute centers from refined labels so they sit at true centroids
        self.prev_centers = self._labels_to_centers(labels)

        # --- D. Build Region Adjacency Graph ----------------------------
        rag = self._build_rag(labels, fused_frame)

        # Update Lagrangian state
        self.prev_labels = labels
        self.prev_fused  = fused_frame
        self.prev_gray   = curr_gray

        return labels, flow, rag

    # ------------------------------------------------------------------ #
    # INTERNAL HELPERS
    # ------------------------------------------------------------------ #

    def _labels_to_centers(self, labels: np.ndarray) -> np.ndarray:
        """Return (K, 2) array of [y, x] centroids for all non-zero labels."""
        unique = np.unique(labels)
        unique = unique[unique != 0]
        if len(unique) == 0:
            return np.empty((0, 2), dtype=np.float32)
        cms = center_of_mass(np.ones_like(labels), labels, unique)
        return np.array(cms, dtype=np.float32)   # (K, 2)

    def _advect_centers(self, flow: np.ndarray, H: int, W: int) -> np.ndarray:
        """
        Move each center by the MEAN optical flow of ALL its member pixels.

        This is the defining characteristic of AdaptiveResidualTSLIC vs. the
        simpler approach of sampling flow only at the centroid pixel.
        """
        labels  = self.prev_labels
        centers = self.prev_centers
        if centers is None or len(centers) == 0:
            return np.empty((0, 2), dtype=np.float32)

        unique  = np.unique(labels)
        unique  = unique[unique != 0]
        flow_fy = flow[:, :, 1]   # vertical component
        flow_fx = flow[:, :, 0]   # horizontal component

        new_centers = []
        for k, lbl in enumerate(unique):
            if k >= len(centers):
                break
            mask      = (labels == lbl)
            mean_fy   = float(flow_fy[mask].mean()) if mask.any() else 0.0
            mean_fx   = float(flow_fx[mask].mean()) if mask.any() else 0.0
            y_old, x_old = centers[k]
            new_centers.append([
                float(np.clip(y_old + mean_fy, 0, H - 1)),
                float(np.clip(x_old + mean_fx, 0, W - 1)),
            ])
        return np.array(new_centers, dtype=np.float32) if new_centers \
               else np.empty((0, 2), dtype=np.float32)

    def _warp_fused(self, fused: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """
        Inverse-warp every channel of fused using the flow field.
        Each output pixel samples from the previous frame at (position - flow).
        """
        H, W  = flow.shape[:2]
        map_y = (np.arange(H, dtype=np.float32)[:, None]
                 * np.ones((1, W), dtype=np.float32)) - flow[:, :, 1]
        map_x = (np.arange(W, dtype=np.float32)[None, :]
                 * np.ones((H, 1), dtype=np.float32)) - flow[:, :, 0]
        C     = fused.shape[2]
        warped = np.empty_like(fused)
        for c in range(C):
            warped[:, :, c] = cv2.remap(
                fused[:, :, c].astype(np.float32),
                map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        return warped

    def _apply_death(self, centers: np.ndarray,
                     fused_curr: np.ndarray, H: int, W: int) -> np.ndarray:
        """
        Kill any center whose fused-channel maximum intensity at its position
        falls below THRESH_DEATH (the region has dissipated / is empty space).
        """
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

    def _apply_birth(self, centers: np.ndarray,
                     fused_curr: np.ndarray,
                     residual:   np.ndarray,
                     H: int, W: int) -> np.ndarray:
        """
        Seed new centers in high-residual regions not already covered.

        Algorithm:
          1. Threshold residual map -> binary mask of high-error pixels.
          2. Connected-component label the mask to find "blobs".
          3. For each blob: add a center only if its centroid is at least
             MIN_SPACING pixels away from every existing center.
        """
        birth_mask = (residual > self.thresh_birth)
        if not birth_mask.any():
            return centers

        blob_labels, n_blobs = sk_label(birth_mask, return_num=True)
        if n_blobs == 0:
            return centers

        new_centers = list(centers)
        existing    = np.array(new_centers) if len(new_centers) > 0 else None

        for blob_id in range(1, n_blobs + 1):
            blob_mask = (blob_labels == blob_id)
            # Discard blobs that are themselves in empty / dissipated space
            if fused_curr[blob_mask].max() < self.thresh_death:
                continue
            props = regionprops(blob_mask.astype(np.uint8))
            if not props:
                continue
            by, bx = props[0].centroid

            # Distance guard — respect MIN_SPACING between any two centers
            if existing is not None and len(existing) > 0:
                dists = np.hypot(existing[:, 0] - by, existing[:, 1] - bx)
                if dists.min() < self.min_spacing:
                    continue

            new_center = [
                float(np.clip(by, 0, H - 1)),
                float(np.clip(bx, 0, W - 1)),
            ]
            new_centers.append(new_center)
            existing = np.array(new_centers)

        return np.array(new_centers, dtype=np.float32)

    def _refine_watershed(self, fused_curr: np.ndarray,
                          centers: np.ndarray, H: int, W: int) -> np.ndarray:
        """
        "Constrained SLIC" via compact watershed:
          - Place integer markers at adaptive center positions.
          - Combined gradient = pixel-wise MAX of per-channel Sobel + DEM Sobel.
          - Watershed grows each marker to the nearest gradient ridge.

        This is equivalent to running K-means restricted to local windows
        around each center, but is vectorised and uses scikit-image's
        optimised C implementation.
        """
        markers = np.zeros((H, W), dtype=np.int32)
        for lbl_idx, (cy, cx) in enumerate(centers, start=1):
            iy = int(np.clip(cy, 0, H - 1))
            ix = int(np.clip(cx, 0, W - 1))
            markers[iy, ix] = lbl_idx

        C            = fused_curr.shape[2]
        grad_layers  = [sobel(fused_curr[:, :, c]) for c in range(C)]
        grad_layers.append(sobel(self.dem_weighted[:H, :W]))
        combined_grad = np.max(np.stack(grad_layers, axis=0), axis=0)

        labels = watershed(
            combined_grad,
            markers=markers,
            compactness=self.compactness / 100.0,
        )
        return labels

    def _build_rag(self, labels: np.ndarray,
                   fused_curr: np.ndarray) -> dict:
        """
        Build a Region Adjacency Graph by scanning horizontal + vertical
        pixel adjacencies.

        Returns:
            {
              'nodes': {lbl: mean_feature_vec (C,)},
              'edges': [(lbl_a, lbl_b, euclidean_feature_distance), ...]
            }
        """
        H, W, C = fused_curr.shape
        unique   = np.unique(labels)
        unique   = unique[unique != 0]

        # Node features: mean fused vector per superpixel
        nodes = {int(lbl): fused_curr[labels == lbl].mean(axis=0) for lbl in unique}

        # Adjacency: scan horizontal then vertical pixel pairs
        adj_set = set()
        for left, right in [(labels[:, :-1], labels[:, 1:]),
                             (labels[:-1, :], labels[1:, :])]:
            diff = left != right
            pa   = left[diff].ravel()
            pb   = right[diff].ravel()
            for a, b in zip(pa, pb):
                if a != 0 and b != 0:
                    adj_set.add((int(min(a, b)), int(max(a, b))))

        edges = []
        for a, b in adj_set:
            if a in nodes and b in nodes:
                dist = float(np.linalg.norm(nodes[a] - nodes[b]))
                edges.append((a, b, dist))

        return {"nodes": nodes, "edges": edges}


# --------------------------------------------------------------------------- #
# DATA LOADING
# --------------------------------------------------------------------------- #
def _get_channel_path(event_id: str, img_type: str,
                      catalog: pd.DataFrame) -> str | None:
    rows = catalog[(catalog["id"] == event_id) & (catalog["img_type"] == img_type)]
    if rows.empty:
        return None
    return os.path.join(BASE_PATH, img_type, os.path.basename(rows.iloc[0]["file_name"]))


def _read_hdf5(path: str, event_id: str, img_type: str) -> np.ndarray | None:
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
    path = _get_channel_path(event_id, ch, catalog)
    if path is None:
        log.warning(f"  {ch}: no catalog entry — skipping.")
        return None
    log.info(f"  {ch}: resolved path -> {path}")
    raw = _read_hdf5(path, event_id, ch)
    if raw is None:
        return None
    log.info(f"  {ch}: raw shape={raw.shape}, dtype={raw.dtype}")
    if raw.ndim == 3:
        arr = raw.transpose(2, 0, 1) if raw.shape[2] < raw.shape[0] else raw
    else:
        arr = raw
    T_ch, H_ch, W_ch = arr.shape
    tgt_h, tgt_w = tgt_shape
    if H_ch != tgt_h or W_ch != tgt_w:
        log.info(f"  {ch}: upsampling {H_ch}x{W_ch} -> {tgt_h}x{tgt_w}")
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
    """Load VIL, IR107, IR069; return raw arrays + geographic extent."""
    meta_rows = catalog[catalog["id"] == event_id]
    if meta_rows.empty:
        log.error(f"Event {event_id} not found in catalog.")
        return {}, []
    extent_row = meta_rows.iloc[0]
    extent = [extent_row["llcrnrlon"], extent_row["urcrnrlon"],
              extent_row["llcrnrlat"], extent_row["urcrnrlat"]]
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
    return data, extent


def load_fused_channels(event_id: str, catalog: pd.DataFrame):
    """
    Load all satellite channels, normalise each to [0, 1], and stack into
    a fused (T, H, W, C) feature cube.

    Returns:
        fused  : (T, H, W, C) float32   normalised feature cube for segmentation
        data   : {ch: (T, H, W) float32} raw arrays for display colourmap
        extent : list[float]
    """
    data, extent = load_channels(event_id, catalog)
    if not data:
        return None, {}, []

    fused_stack = []
    for ch in FUSION_CHANNELS:
        if ch in data:
            arr    = data[ch]
            mn, mx = arr.min(), arr.max()
            fused_stack.append((arr - mn) / (mx - mn + 1e-6))
        else:
            log.warning(f"  Fusion channel '{ch}' unavailable — omitting.")

    if not fused_stack:
        log.error("  No fusion channels available — cannot build feature cube.")
        return None, data, extent

    fused = np.stack(fused_stack, axis=-1).astype(np.float32)  # (T, H, W, C)
    log.info(f"  Fused feature cube: {fused.shape}  "
             f"({len(fused_stack)} channels: {[ch for ch in FUSION_CHANNELS if ch in data]})")
    return fused, data, extent


# --------------------------------------------------------------------------- #
# COLOUR + BOUNDARY HELPERS
# --------------------------------------------------------------------------- #
def resolve_vmin_vmax(ch: str, arr: np.ndarray):
    cfg = CHANNEL_CFG[ch]
    if cfg["vmin"] is not None:
        return cfg["vmin"], cfg["vmax"]
    valid = arr[arr != 0]
    return (float(valid.min()), float(valid.max())) if valid.size else (0.0, 1.0)


def frame_to_rgb(frame: np.ndarray, ch: str,
                 vmin: float, vmax: float) -> np.ndarray:
    norm = np.clip((frame - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
    rgba = CHANNEL_CFG[ch]["cmap"](norm)
    return rgba[:, :, :3].astype(np.float32)


def render_boundary(rgb: np.ndarray, labels: np.ndarray,
                    color: tuple) -> np.ndarray:
    out = mark_boundaries(rgb, labels, color=color, mode="thick")
    return (out * 255).clip(0, 255).astype(np.uint8)


def rag_to_line_segments(rag: dict, centers: np.ndarray) -> np.ndarray:
    """
    Convert the RAG edge list into an (E, 2, 2) array of [[x0,y0],[x1,y1]]
    segments suitable for matplotlib LineCollection.
    Centers are indexed by (label - 1) since watershed markers start at 1.
    """
    if centers is None or len(centers) == 0 or not rag["edges"]:
        return np.empty((0, 2, 2), dtype=np.float32)
    # centers[k] = [y, x] for marker label (k+1)
    lbl_to_center = {k + 1: centers[k] for k in range(len(centers))}
    segs = []
    for a, b, _ in rag["edges"]:
        if a in lbl_to_center and b in lbl_to_center:
            ya, xa = lbl_to_center[a]
            yb, xb = lbl_to_center[b]
            segs.append([[xa, ya], [xb, yb]])   # LineCollection uses (x, y) order
    return np.array(segs, dtype=np.float32) if segs else np.empty((0, 2, 2))


# --------------------------------------------------------------------------- #
# PRE-COMPUTATION
# --------------------------------------------------------------------------- #
def precompute(fused: np.ndarray, data: dict, dem_norm: np.ndarray):
    """
    Run ONE AdaptiveResidualTSLIC_DEM per mode on the fused (T, H, W, C) cube.
    The resulting Master Boundary label map is rendered onto every display channel.

    Returns:
        centroids_store : {use_flow: list[np.ndarray(K, 2)]}
        flows_store     : list[ndarray(H,W,2)]       — Temporal mode
        rendered_store  : {ch: {use_flow: ndarray(T,H,W,3) uint8}}
        rag_store       : list[dict]                 — Temporal mode
    """
    T           = fused.shape[0]
    display_chs = [ch for ch in DISPLAY_CHANNELS if ch in data]
    vranges     = {ch: resolve_vmin_vmax(ch, data[ch]) for ch in display_chs}

    centroids_store: dict = {}
    flows_store:     list = []
    rendered_store:  dict = {ch: {} for ch in display_chs}
    rag_store:       list = []

    for use_flow in [True, False]:
        style = MODE_STYLE[use_flow]
        tslic = AdaptiveResidualTSLIC_DEM(
            dem_norm,
            n_segments=N_SEGMENTS,
            compactness=COMPACTNESS,
            lambda_z=ELEVATION_LAMBDA,
            thresh_death=THRESH_DEATH,
            thresh_birth=THRESH_BIRTH,
            min_spacing=MIN_SPACING,
        )

        centroids_list: list = []
        labels_list:    list = []
        flow_list:      list = []
        rag_list:       list = []

        log.info(f"\n  [{style['label']}] segmenting {T} frames on fused cube ...")
        for t in tqdm(range(T), desc=f"  [{style['label']}]"):
            labels, flow, rag = tslic.segment(fused[t], use_flow=use_flow)
            centroids_list.append(
                tslic.prev_centers.copy()
                if tslic.prev_centers is not None
                else np.empty((0, 2), dtype=np.float32)
            )
            labels_list.append(labels)
            flow_list.append(flow)
            rag_list.append(rag)

        centroids_store[use_flow] = centroids_list
        if use_flow:
            flows_store = flow_list
            rag_store   = rag_list

        # Render the same Master Boundary label map onto each display channel
        for ch in display_chs:
            vmin, vmax    = vranges[ch]
            rendered_list = []
            for t in range(T):
                rgb      = frame_to_rgb(data[ch][t], ch, vmin, vmax)
                rendered = render_boundary(rgb, labels_list[t], style["boundary"])
                rendered_list.append(rendered)
            rendered_store[ch][use_flow] = np.stack(rendered_list, axis=0)

    return centroids_store, flows_store, rendered_store, rag_store


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
                 flows_store: list, rag_store: list):
    """
    Build and save the animated GIF / MP4.

    Temporal SLIC panels additionally show:
      • Subsampled optical flow quiver (white arrows).
      • RAG adjacency edges as a semi-transparent gold LineCollection.
    """
    display_chs = [ch for ch in DISPLAY_CHANNELS if ch in data]
    n_cols      = len(display_chs)
    T           = max(data[ch].shape[0] for ch in display_chs)
    total_min   = T * SECONDS_PER_FRAME // 60

    writer, ext = get_writer()
    dpi      = 80 if ext == ".gif" else 120
    out_path = os.path.join(EVENTS_DIR, f"{event_id}_slic_{N_SEGMENTS}{ext}")
    log.info(f"\nAnimating {T} frames ({total_min} min) -> {out_path}")

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
        axes[row_idx, 0].set_ylabel(MODE_STYLE[use_flow]["label"],
                                    fontsize=10, labelpad=8)

    time_text = fig.text(
        0.5, 0.01, "", ha="center", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )
    fig.suptitle(
        f"Adaptive Residual Fused Superpixel  —  Event {event_id}  ({total_min} min)\n"
        f"Gold = Adaptive SLIC  (Birth/Death + RAG)   |   Cyan = Standard SLIC",
        fontsize=11, y=0.99,
    )

    im_artists:  dict = {}
    sc_artists:  dict = {}
    qv_artists:  dict = {}
    rag_artists: dict = {}

    H_ax = data[display_chs[0]].shape[1]
    W_ax = data[display_chs[0]].shape[2]

    for row_idx, use_flow in enumerate([True, False]):
        for col_idx, ch in enumerate(display_chs):
            ax  = axes[row_idx, col_idx]
            im  = ax.imshow(rendered_store[ch][use_flow][0],
                            origin="lower", animated=True, aspect="auto")
            ax.axis("off")
            key = (row_idx, col_idx)
            im_artists[key]  = im
            sc_artists[key]  = None
            qv_artists[key]  = None
            rag_artists[key] = None

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    def _update(t):
        updated = []
        for row_idx, use_flow in enumerate([True, False]):
            for col_idx, ch in enumerate(display_chs):
                ax     = axes[row_idx, col_idx]
                key    = (row_idx, col_idx)
                t_safe = min(t, rendered_store[ch][use_flow].shape[0] - 1)

                # 1. Boundary image — fast pixel swap
                im_artists[key].set_data(rendered_store[ch][use_flow][t_safe])
                updated.append(im_artists[key])

                # 2. Centroid scatter — single shared set across all columns
                if sc_artists[key] is not None:
                    sc_artists[key].remove()
                    sc_artists[key] = None
                cents = centroids_store[use_flow][t_safe]
                if cents is not None and len(cents) > 0:
                    sc = ax.scatter(cents[:, 1], cents[:, 0],
                                    c=MODE_STYLE[use_flow]["centroid"],
                                    s=6, linewidths=0, zorder=5)
                    sc_artists[key] = sc
                    updated.append(sc)

                # 3. Flow quiver — Temporal row only
                if use_flow:
                    if qv_artists[key] is not None:
                        qv_artists[key].remove()
                        qv_artists[key] = None
                    flow = flows_store[t_safe]
                    ys   = np.arange(0, H_ax, QUIVER_STRIDE)
                    xs   = np.arange(0, W_ax, QUIVER_STRIDE)
                    xg, yg = np.meshgrid(xs, ys)
                    u = flow[yg, xg, 0]
                    v = flow[yg, xg, 1]
                    valid = np.hypot(u, v) > 0.5
                    if valid.any():
                        qv = ax.quiver(xg[valid], yg[valid],
                                       u[valid], v[valid],
                                       color="white", alpha=0.55,
                                       scale=120, width=0.003, zorder=4)
                        qv_artists[key] = qv
                        updated.append(qv)

                # 4. RAG edges — Temporal row only, when enabled
                if use_flow and SHOW_RAG_EDGES and rag_store:
                    if rag_artists[key] is not None:
                        rag_artists[key].remove()
                        rag_artists[key] = None
                    segs = rag_to_line_segments(
                        rag_store[t_safe],
                        centroids_store[use_flow][t_safe],
                    )
                    if len(segs) > 0:
                        lc = LineCollection(segs,
                                            colors=[(1.0, 0.75, 0.0, 0.30)],
                                            linewidths=0.5, zorder=3)
                        ax.add_collection(lc)
                        rag_artists[key] = lc
                        updated.append(lc)

        elapsed_min = t * SECONDS_PER_FRAME // 60
        time_text.set_text(
            f"T+{elapsed_min:03d} min   frame {t + 1}/{T}   ({SECONDS_PER_FRAME}s/frame)"
        )
        updated.append(time_text)
        return updated

    anim = animation.FuncAnimation(
        fig, _update, frames=T, interval=200, blit=False,
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
        if event_id.lower() == "q":
            break
        if not event_id:
            continue

        print(f"\nLoading data for {event_id} ...")
        fused, data, extent = load_fused_channels(event_id, catalog)
        if fused is None or not data:
            print("  [Error] No data found. Check event ID and file paths.")
            continue

        T = fused.shape[0]
        print(f"  Event duration : {T} frames = {T * SECONDS_PER_FRAME // 60} min")
        print(f"  Feature cube   : {fused.shape}  (T x H x W x C)")

        print("\nFetching DEM ...")
        dem_raw  = fetch_and_regrid_dem(extent)
        dem_norm = (dem_raw - dem_raw.min()) / (dem_raw.max() - dem_raw.min() + 1e-6)

        print("\nPre-computing fused adaptive segmentation (both modes) ...")
        centroids_store, flows_store, rendered_store, rag_store = precompute(
            fused, data, dem_norm
        )

        out = animate_slic(event_id, data, rendered_store,
                           centroids_store, flows_store, rag_store)
        print(f"\nDone -> {out}")