"""
visualize_hierarchical_graph.py  —  Bi-Level Hierarchical Lagrangian Graph on SEVIR
======================================================================================
Visualises the proposed hierarchical graph (fine → middle) against the original
flat Lagrangian DRIFT superpixel graph, side-by-side for each display channel.

NOTE ON ARCHITECTURE (per mathematical spec §1.1 and §13):
  The system is a TWO-LEVEL hierarchy: fine (parcel-scale) and middle (DRIFT
  cell-scale).  The coarse level has been REPLACED by a Global Context Readout:
    g(t) = MLP_gcr( mean / std / max pool over {h_i^m(t)} )
  g(t) has no spatial position and no ODE — it is a broadcast conditioning signal
  only.  A true spatial coarse level is deferred to future work (§2.3 of spec).

Layout — 2 × N grid  (N = number of display channels):
  ┌──────────┬──────────┬──────────┐
  │  VIL     │  IR107   │  IR069   │   ← TOP ROW:  Hierarchical graph
  │          │          │          │       • Fine boundaries (magenta, ~3–6 km)
  │          │          │          │       • Middle boundaries (red, ~11 km)
  │          │          │          │       • E_fm edges (cyan)   fine → middle
  │          │          │          │       • E_mm edges (yellow) middle adjacency
  │          │          │          │       • GCR g(t) stats annotation (top-right)
  ├──────────┼──────────┼──────────┤
  │  VIL     │  IR107   │  IR069   │   ← BOTTOM ROW: Original flat Lagrangian DRIFT
  │          │          │          │       • Middle SLIC superpixels (gold/cyan)
  │          │          │          │       • Optical-flow quivers
  └──────────┴──────────┴──────────┘

Hierarchy construction (all from the mathematical spec):
  • Middle nodes  — Topography-Aware Temporal SLIC on fused (VIL,IR107,IR069) cube
                    N^m = N* ∈ {750,…,1150}; global optimum N*=1150 per §5 of thesis
  • Fine nodes    — per-superpixel watershed sub-segmentation seeded by local VIL maxima
                    K_i ∈ [1, K_max=8] via rate-distortion elbow in (x,y,VIL) space
                    (§2.2.1, eq. 2.2a–c); K_avg ≈ 2.2 → N^f ≈ 2,530 typical
  • E_mm edges    — topological adjacency of middle superpixels from TemporalSLIC
                    label maps L_t (§4.1, eq. 4.3a); k_active_topo ≈ 6 per node
  • E_fm edges    — fine → parent middle cross-level edges via soft assignment A^{fm}
                    (§4.2, eq. 4.6); threshold θ_fm = 1/(2 N^m)
  • GCR g(t)      — domain-wide scene statistics pooled from middle states (§13)
"""

# --------------------------------------------------------------------------- #
# IMPORTS
# --------------------------------------------------------------------------- #
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from skimage.segmentation import slic, watershed, mark_boundaries
from skimage.filters import sobel, gaussian
from skimage.feature import peak_local_max
from scipy.ndimage import center_of_mass, label as ndlabel
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import cv2
import os
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.interpolate import RegularGridInterpolator


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
# SEVIR PROJECTION & DEM IMPORTS
# --------------------------------------------------------------------------- #
try:
    import cartopy.crs as ccrs
    import pystac_client
    import planetary_computer
    import rioxarray
    from rioxarray.merge import merge_arrays
    from tqdm import tqdm
    SEVIR_PROJ = ccrs.LambertConformal(
        central_longitude=-98.0, central_latitude=38.0,
        standard_parallels=(30.0, 60.0),
        globe=ccrs.Globe(semimajor_axis=6370000, semiminor_axis=6370000),
    )
    HAS_DEM_LIBS = True
except ImportError:
    HAS_DEM_LIBS = False
    log.warning("DEM or Cartopy libraries not found — using flat terrain (DEM = zeros).")


# --------------------------------------------------------------------------- #
# CONFIGURATION  (mirrors original script; only paths need adjustment)
# --------------------------------------------------------------------------- #
BASE_PATH    = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\SEVIR\2019"
CATALOG_PATH = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\SEVIR\CATALOG.csv"
EVENTS_DIR   = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\Events"

os.makedirs(EVENTS_DIR, exist_ok=True)

# =========================================================================== #
#  DEM PIPELINE                                                                #
# =========================================================================== #
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
    if not HAS_DEM_LIBS:
        log.info("  Using flat DEM (zeros) due to missing libraries.")
        return np.zeros((ny, nx), dtype=np.float32)

    cache_key = tuple(np.round(extent, 4))
    if cache_key in _dem_cache:
        return _dem_cache[cache_key]

    log.info("  Fetching Cop-DEM-GLO-30 via Planetary Computer …")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    items = list(catalog.search(
        collections=["cop-dem-glo-30"],
        bbox=[extent[0], extent[2], extent[1], extent[3]],
    ).item_collection())
    
    if not items:
        log.warning("  No DEM tiles found, returning flat terrain.")
        return np.zeros((ny, nx), dtype=np.float32)

    datasets = [None] * len(items)
    with ThreadPoolExecutor(max_workers=min(len(items), 8)) as pool:
        fmap = {pool.submit(_load_and_clip_tile, it, extent): i
                for i, it in enumerate(items)}
        for fut in tqdm(as_completed(fmap), total=len(items), desc="DEM tiles"):
            try:    
                datasets[fmap[fut]] = fut.result()
            except Exception as e: 
                log.warning(f"DEM tile failed: {e}")

    datasets = [d for d in datasets if d is not None and d.size > 0]
    merged = merge_arrays(datasets) if len(datasets) > 1 else datasets[0]
    lons, lats, vals = merged.x.values, merged.y.values, merged.values
    
    # Ensure correct latitudinal orientation
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

def normalise_dem(dem_raw):
    rng = dem_raw.max() - dem_raw.min()
    return (dem_raw - dem_raw.min()) / (rng + 1e-6)

# --------------------------------------------------------------------------- #
# SEVIR PROJECTION  (unchanged from original)
# --------------------------------------------------------------------------- #
try:
    import cartopy.crs as ccrs
    SEVIR_PROJ = ccrs.LambertConformal(
        central_longitude=-98.0, central_latitude=38.0,
        standard_parallels=(30.0, 60.0),
        globe=ccrs.Globe(semimajor_axis=6370000, semiminor_axis=6370000),
    )
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    log.warning("cartopy not found — geographic projection disabled.")

# --------------------------------------------------------------------------- #
# DISPLAY CHANNELS & COLOUR CONFIG
# --------------------------------------------------------------------------- #
DISPLAY_CHANNELS = ["vil", "ir107", "ir069"]
FUSION_CHANNELS  = ["vil", "ir107", "ir069"]

CHANNEL_CFG = {
    "vil":   {"cmap": plt.get_cmap("jet"),    "vmin": 0,    "vmax": 255},
    "ir107": {"cmap": plt.get_cmap("gray_r"), "vmin": None, "vmax": None},
    "ir069": {"cmap": plt.get_cmap("gray_r"), "vmin": None, "vmax": None},
}

SECONDS_PER_FRAME = 300
QUIVER_STRIDE     = 24

# --------------------------------------------------------------------------- #
# HIERARCHY VISUAL STYLE                                                       #
# --------------------------------------------------------------------------- #
#   Fine nodes    ─ magenta fill, magenta boundary
#   Middle nodes  ─ red boundary (SLIC superpixels)
#   (Coarse level removed; see GCR section below)
FINE_COLOR   = (1.00, 0.00, 1.00)   # magenta  — fine-level boundaries / centroids
MIDDLE_COLOR = (0.95, 0.15, 0.15)   # red      — middle-level boundaries / centroids
FLAT_TEMPORAL= (1.00, 0.75, 0.00)   # gold     — flat Temporal SLIC (bottom row)
COARSE_COLOR = (1.00, 0.85, 0.00)   # gold     — reserved for future coarse level

# ── Edge visualisation toggles (pre-baked into rendered images for speed) ── #
SHOW_FM_EDGES = True    # E_fm: fine→middle cross-level edges (cyan)
SHOW_MM_EDGES = True    # E_mm: middle topological adjacency edges (yellow)
FM_EDGE_COLOR = (0,   200, 200)     # BGR for cv2: cyan
MM_EDGE_COLOR = (0,   200, 220)     # BGR for cv2: yellow-green
FM_EDGE_ALPHA = 0.45
MM_EDGE_ALPHA = 0.35

N_SEGMENTS_MIDDLE = 950             # N^m visualisation default; paper optimum = 1150
COMPACTNESS       = 10
ELEVATION_LAMBDA  = 0.5
K_MAX             = 8          # hard upper bound on fine nodes per superpixel (§2.2.1)
K_MIN             = 1
TAU_VIL           = 10         # min raw VIL to count as precipitation feature (dBZ proxy)
R_CELL_PX         = 5          # min separation between fine-node seeds (px)


# =========================================================================== #
#  DATA LOADING  (identical pipeline to original script)                       #
# =========================================================================== #

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
    meta_rows = catalog[catalog["id"] == event_id]
    if meta_rows.empty:
        return {}, []
    er = meta_rows.iloc[0]
    extent = [er["llcrnrlon"], er["urcrnrlon"], er["llcrnrlat"], er["urcrnrlat"]]
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


def load_fused_channels(event_id, catalog):
    data, extent = load_channels(event_id, catalog)
    if not data:
        return None, {}, []
    fused_stack = []
    for ch in FUSION_CHANNELS:
        if ch in data:
            arr = data[ch]
            mn, mx = arr.min(), arr.max()
            norm = (arr - mn) / (mx - mn + 1e-6)
            fused_stack.append(norm)
    if not fused_stack:
        return None, data, extent
    fused = np.stack(fused_stack, axis=-1).astype(np.float32)
    return fused, data, extent


# =========================================================================== #
#  SYNTHETIC DATA GENERATOR (used when SEVIR files are absent)                 #
# =========================================================================== #

def generate_synthetic_event(H=384, W=384, T=13, seed=42):
    """
    Generate a realistic-looking synthetic VIL/IR107/IR069 event with
    multiple storm cells at different intensities and lifecycle stages.
    """
    rng = np.random.default_rng(seed)
    log.info("Generating synthetic SEVIR event …")

    def gauss2d(cy, cx, sigma, H, W):
        y, x = np.ogrid[:H, :W]
        return np.exp(-((y - cy)**2 + (x - cx)**2) / (2 * sigma**2))

    # Storm cell parameters [cy, cx, sigma_px, peak_vil, drift_dy, drift_dx]
    cells = [
        [120, 100, 28, 220, -0.4,  0.8],   # large MCS, moderate drift
        [200, 220, 18, 200, -0.2,  1.1],   # medium cell, fast eastward
        [280, 160, 12, 180, -0.5,  0.5],   # small intense cell
        [ 90, 300, 22, 150, -0.3,  0.7],   # diffuse stratiform
        [320,  80, 15, 160,  0.1,  0.9],   # decaying system
    ]

    vil_stack   = np.zeros((T, H, W), dtype=np.float32)
    ir107_stack = np.zeros((T, H, W), dtype=np.float32)
    ir069_stack = np.zeros((T, H, W), dtype=np.float32)

    base_ir107 = 250.0
    base_ir069 = 240.0

    for t in range(T):
        life = t / (T - 1)  # 0→1
        for cy0, cx0, sig, peak, ddy, ddx in cells:
            cy = cy0 + ddy * t * SECONDS_PER_FRAME / 1000 * 10
            cx = cx0 + ddx * t * SECONDS_PER_FRAME / 1000 * 10
            cy = float(np.clip(cy, 0, H - 1))
            cx = float(np.clip(cx, 0, W - 1))

            # Life cycle: grow, plateau, decay
            if life < 0.4:
                intensity = life / 0.4
            elif life < 0.7:
                intensity = 1.0
            else:
                intensity = 1.0 - (life - 0.7) / 0.3

            g = gauss2d(cy, cx, sig, H, W)
            vil_val = peak * intensity + rng.normal(0, 5, (H, W))
            vil_stack[t] += g * vil_val

            ir_offset = -60 * intensity * g
            ir107_stack[t] += ir_offset
            ir069_stack[t] += ir_offset * 0.85

        vil_stack[t] = np.clip(vil_stack[t], 0, 255)
        ir107_stack[t] = base_ir107 + ir107_stack[t] + rng.normal(0, 2, (H, W))
        ir069_stack[t] = base_ir069 + ir069_stack[t] + rng.normal(0, 2, (H, W))

    data = {
        "vil":   vil_stack.clip(0, 255),
        "ir107": ir107_stack,
        "ir069": ir069_stack,
    }
    extent = [-104.0, -92.0, 33.0, 42.0]
    fused_stack = []
    for ch in FUSION_CHANNELS:
        arr = data[ch]
        mn, mx = arr.min(), arr.max()
        fused_stack.append((arr - mn) / (mx - mn + 1e-6))
    fused = np.stack(fused_stack, axis=-1).astype(np.float32)
    log.info(f"  Synthetic event: shape={fused.shape}")
    return fused, data, extent


# =========================================================================== #
#  DEM (flat stub — identical behaviour to original when DEM libs absent)      #
# =========================================================================== #

def fetch_dem_or_zeros(extent, H=384, W=384):
    log.info("  Using flat DEM (zeros).")
    return np.zeros((H, W), dtype=np.float32)


def normalise_dem(dem_raw):
    rng = dem_raw.max() - dem_raw.min()
    return (dem_raw - dem_raw.min()) / (rng + 1e-6)


# =========================================================================== #
#  TEMPORAL SLIC (same as original, used for BOTH flat and hierarchical rows)  #
# =========================================================================== #

class TemporalSLIC_DEM:
    def __init__(self, dem_norm, n_segments=150, compactness=10.0, lambda_z=0.5):
        self.n_segments  = n_segments
        self.compactness = compactness
        self.lambda_z    = lambda_z
        self.dem_norm     = dem_norm
        self.dem_weighted = dem_norm * lambda_z
        self.prev_labels    = None
        self.prev_centroids = None
        self.prev_gray      = None

    def segment(self, fused_frame, use_flow=True):
        if fused_frame.ndim == 2:
            fused_frame = fused_frame[:, :, np.newaxis]
        H, W, C = fused_frame.shape
        gray_norm = fused_frame[:, :, 0]
        curr_gray = (gray_norm * 255).astype(np.uint8)
        dem_ch       = (self.dem_norm * self.lambda_z)[:H, :W, np.newaxis]
        feature_cube = np.concatenate([fused_frame, dem_ch], axis=-1).astype(np.float64)
        flow = np.zeros((H, W, 2), dtype=np.float32)

        if self.prev_labels is None:
            labels = slic(
                feature_cube,
                n_segments=self.n_segments,
                compactness=self.compactness,
                start_label=1,
                enforce_connectivity=True,
                channel_axis=-1,
            )
            self.prev_centroids = self._calculate_centroids(labels)
        else:
            if use_flow and self.prev_gray is not None:
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
                    y_new = int(np.clip(y_old + flow_y, 0, H - 1))
                    x_new = int(np.clip(x_old + flow_x, 0, W - 1))
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

    def _calculate_centroids(self, labels):
        unique_labels = np.unique(labels)
        if len(unique_labels) and unique_labels[0] == 0:
            unique_labels = unique_labels[1:]
        centroids = center_of_mass(np.ones_like(labels), labels, unique_labels)
        return {lbl: np.array(c) for lbl, c in zip(unique_labels, centroids)}


# =========================================================================== #
#  FINE-LEVEL SUB-SEGMENTATION  (§2.2 of the .md)                              #
# =========================================================================== #

'''def _rate_distortion_elbow(values_1d, k_max=K_MAX):
    """
    Compute the normalised distortion gain G(k) for k=1..k_max on a 1-D
    VIL value array.  Returns K_i = argmax G(k), clamped to [K_MIN, K_MAX].
    """
    values = values_1d.ravel().astype(np.float64)
    if len(values) < 2:
        return K_MIN
    distortions = []
    for k in range(1, k_max + 1):
        if k >= len(values):
            # Degenerate: more clusters than pixels
            distortions.append(0.0)
            continue
        # k-means on 1-D (Lloyd's)
        idx = np.round(np.linspace(0, len(values) - 1, k)).astype(int)
        centers = np.sort(values)[idx]
        for _ in range(20):
            dists  = np.abs(values[:, None] - centers[None, :])
            assign = np.argmin(dists, axis=1)
            new_c  = np.array([values[assign == c].mean() if (assign == c).any()
                               else centers[c] for c in range(k)])
            if np.allclose(new_c, centers, atol=1e-6):
                break
            centers = new_c
        dists  = np.abs(values[:, None] - centers[None, :])
        wcss   = np.mean(np.min(dists, axis=1) ** 2)
        distortions.append(wcss)

    distortions = np.array(distortions)
    D1 = distortions[0]
    DK = distortions[-1]
    span = D1 - DK + 1e-12
    gains = np.array([(distortions[k - 1] - distortions[k]) / span
                      for k in range(1, len(distortions))])
    k_i = int(np.argmax(gains)) + 1          # gains[0] is G(1)
    return int(np.clip(k_i, K_MIN, K_MAX))'''

def _rate_distortion_elbow(ys, xs, vil_vals, k_max=K_MAX, L_sp=11.32):
    """
    Compute the elbow using the true Kneedle algorithm (maximum distance 
    from the secant line) on a 3-D (x, y, VIL) feature space.
    """
    if len(vil_vals) < 2:
        return K_MIN

    cy, cx = ys.mean(), xs.mean()
    vil_max = vil_vals.max() + 1e-6
    
    f_y = (ys - cy) / L_sp
    f_x = (xs - cx) / L_sp
    f_v = vil_vals / vil_max
    
    features = np.column_stack((f_y, f_x, f_v))
    N_px = len(features)
    
    distortions = []
    for k in range(1, k_max + 1):
        if k >= N_px:
            distortions.append(0.0)
            continue
            
        sort_idx = np.argsort(f_v)
        init_idx = np.round(np.linspace(0, N_px - 1, k)).astype(int)
        centers = features[sort_idx[init_idx]].copy()
        
        for _ in range(20):
            dists = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
            assign = np.argmin(dists, axis=1)
            
            new_c = np.array([
                features[assign == c].mean(axis=0) if (assign == c).any() else centers[c] 
                for c in range(k)
            ])
            
            if np.allclose(new_c, centers, atol=1e-6):
                break
            centers = new_c
            
        dists = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
        wcss = np.mean(np.min(dists, axis=1) ** 2)
        distortions.append(wcss)

    distortions = np.array(distortions)
    if len(distortions) < 2:
        return K_MIN
        
    D1 = distortions[0]
    DK = distortions[-1]
    K_len = len(distortions)
    
    # True Kneedle algorithm: expected linear drop vs actual drop
    expected = np.array([D1 + i * (DK - D1) / (K_len - 1) for i in range(K_len)])
    distances = expected - distortions
    
    k_i = int(np.argmax(distances)) + 1
    return int(np.clip(k_i, K_MIN, K_MAX))

from skimage.segmentation import find_boundaries

def render_hierarchical_boundaries(rgb, fine_labels, mid_labels, fine_color, mid_color):
    """
    Explicitly isolates internal fine boundaries from middle boundaries
    to guarantee high visibility without overlapping colors.
    """
    out = rgb.copy()
    if out.max() > 1.0:
        out = out / 255.0
        
    # Find boolean boundary maps
    fine_b = find_boundaries(fine_labels, mode='inner')
    mid_b  = find_boundaries(mid_labels, mode='thick')
    
    # Pure internal fine boundaries (fine lines that don't touch the parent edge)
    internal_fine_b = fine_b & (~mid_b)
    
    # Paint pixels directly
    out[internal_fine_b] = fine_color
    out[mid_b] = mid_color
    
    return (out * 255).clip(0, 255).astype(np.uint8)


def build_fine_nodes(middle_labels, vil_frame_norm):
    """
    For each middle superpixel, run the §2.2 algorithm:
      1. Gaussian smooth VIL
      2. Find K_i via rate-distortion elbow in (x,y,VIL) space
      3. Locate K_i peak seeds with min separation R_CELL_PX
      4. Watershed within mask
      5. Return fine_labels (H,W), fine_centroids {global_j: [y,x]},
             parent_map {global_j: middle_lbl}
    """
    H, W = vil_frame_norm.shape
    vil_smooth = gaussian(vil_frame_norm, sigma=1.5)

    fine_labels   = np.zeros((H, W), dtype=np.int32)
    fine_centroids = {}
    parent_map     = {}
    global_j       = 0

    unique_middle = np.unique(middle_labels)
    if unique_middle[0] == 0:
        unique_middle = unique_middle[1:]

    for mid_lbl in unique_middle:
        mask = (middle_labels == mid_lbl)
        px_ys, px_xs = np.where(mask)
        if len(px_ys) < 2:
            # Degenerate superpixel → one fine node = parent
            cy, cx = float(px_ys.mean()), float(px_xs.mean())
            global_j += 1
            fine_labels[mask] = global_j
            fine_centroids[global_j] = np.array([cy, cx])
            parent_map[global_j] = mid_lbl
            continue

        vil_vals = vil_smooth[px_ys, px_xs]
        #k_i = _rate_distortion_elbow(vil_vals)

        # NEW
        # L_sp = sqrt(H*W / N^m) per the mathematical spec
        L_sp = np.sqrt((H * W) / len(unique_middle))
        
        k_i = _rate_distortion_elbow(px_ys, px_xs, vil_vals, k_max=K_MAX, L_sp=L_sp)

        # Compute per-pixel gradient for watershed barrier
        sp_gradient = sobel(vil_smooth) * mask.astype(float)

        # Build seed markers using peak_local_max on the smoothed VIL within mask
        local_vil = vil_smooth * mask.astype(float)
        min_distance = max(R_CELL_PX, 3)
        peaks = peak_local_max(
            local_vil,
            min_distance=min_distance,
            num_peaks=k_i,
            threshold_abs=TAU_VIL / 255.0,
            labels=mask.astype(np.uint8),
        )

        if len(peaks) == 0:
            # No peaks above threshold → single fine node
            peaks = np.array([[int(px_ys.mean()), int(px_xs.mean())]])

        # Build marker image
        markers = np.zeros((H, W), dtype=np.int32)
        for local_idx, (py, px) in enumerate(peaks, start=1):
            markers[py, px] = local_idx

        # Watershed within mask
        if markers.sum() > 0:
            ws = watershed(-local_vil + sp_gradient, markers=markers, mask=mask)
        else:
            ws = mask.astype(np.int32)

        local_unique = np.unique(ws)
        local_unique = local_unique[local_unique > 0]

        for loc_lbl in local_unique:
            sub_mask = (ws == loc_lbl)
            ys, xs   = np.where(sub_mask)
            if len(ys) == 0:
                continue
            w_intensity = vil_smooth[ys, xs] + 1e-6
            cy = float((w_intensity * ys).sum() / w_intensity.sum())
            cx = float((w_intensity * xs).sum() / w_intensity.sum())
            global_j += 1
            fine_labels[sub_mask] = global_j
            fine_centroids[global_j] = np.array([cy, cx])
            parent_map[global_j] = mid_lbl

    return fine_labels, fine_centroids, parent_map


# =========================================================================== #
#  GLOBAL CONTEXT READOUT  (replaces coarse level — §13 of mathematical spec)  #
#  g(t) = MLP_gcr( mean/std/max pool over {h_i^m(t)} )                         #
#  No spatial position, no ODE.  Visualised as a text annotation only.          #
# =========================================================================== #

def compute_gcr_stats(mid_labels, vil_norm):
    """
    Compute domain-wide pooled statistics over all middle superpixels.
    Proxies g(t) = MLP_gcr(mean/std/max pool {h_i^m(t)}) from spec §13.

    Returns dict with keys: mean_vil, std_vil, max_vil, n_mid.
    """
    unique_mid = [int(lbl) for lbl in np.unique(mid_labels) if lbl > 0]
    if not unique_mid:
        return {"mean_vil": 0.0, "std_vil": 0.0, "max_vil": 0.0, "n_mid": 0}
    vil_means = np.array([float(vil_norm[mid_labels == lbl].mean())
                          for lbl in unique_mid])
    return {
        "mean_vil": float(vil_means.mean()),
        "std_vil":  float(vil_means.std()),
        "max_vil":  float(vil_means.max()),
        "n_mid":    len(unique_mid),
    }


# =========================================================================== #
#  GRAPH EDGE COMPUTATION  (topology-driven, per spec §4.1–§4.2)               #
# =========================================================================== #

def compute_middle_adjacency(mid_labels):
    """
    Compute E_mm^{topo}(t): topological adjacency of middle superpixels.
    Two superpixels are adjacent if their pixel regions share a 4-connected
    boundary in the label map L_t.  Implements eq. (4.3a) of the spec.

    Returns a frozenset of (i, j) int pairs with i < j.
    """
    adj = set()
    for a_flat, b_flat in [
        (mid_labels[:, :-1].ravel(), mid_labels[:, 1:].ravel()),   # horizontal
        (mid_labels[:-1, :].ravel(), mid_labels[1:, :].ravel()),   # vertical
    ]:
        diff_mask     = a_flat != b_flat
        a_sel, b_sel  = a_flat[diff_mask], b_flat[diff_mask]
        valid         = (a_sel > 0) & (b_sel > 0)
        if not valid.any():
            continue
        lo = np.minimum(a_sel[valid], b_sel[valid])
        hi = np.maximum(a_sel[valid], b_sel[valid])
        pairs = np.unique(np.column_stack([lo, hi]), axis=0)
        for i, j in pairs:
            adj.add((int(i), int(j)))
    return adj


# =========================================================================== #
#  EDGE RENDERERS  (pre-baked into RGB images; avoids per-frame artist churn)  #
# =========================================================================== #

def render_fm_edges(rgb_uint8, fine_centroids, mid_centroids, parent_map):
    """
    Draw E_fm cross-level edges (fine → parent middle centroid) on rgb_uint8.
    Implements §4.2 eq. (4.6): E_fm(t) = {(j,i) : A^{fm}_{ji}(t) > θ_fm}.
    Uses hard parent assignment (π(j)=i) as the top-1 approximation.
    """
    if not SHOW_FM_EDGES or not fine_centroids or not mid_centroids:
        return rgb_uint8
    overlay = rgb_uint8.copy()
    for j, fc in fine_centroids.items():
        mi = parent_map.get(j)
        if mi is None:
            continue
        mc = mid_centroids.get(mi)
        if mc is None:
            continue
        cv2.line(overlay,
                 (int(fc[1]),  int(fc[0])),
                 (int(mc[1]),  int(mc[0])),
                 FM_EDGE_COLOR, 1, lineType=cv2.LINE_AA)
    return cv2.addWeighted(overlay, FM_EDGE_ALPHA, rgb_uint8, 1 - FM_EDGE_ALPHA, 0)


def render_mm_edges(rgb_uint8, mm_adj, mid_centroids):
    """
    Draw E_mm topological adjacency edges between adjacent middle node centroids.
    Implements §4.1 eq. (4.3a–c): E_mm^{topo}(t) from TemporalSLIC label maps.
    """
    if not SHOW_MM_EDGES or not mm_adj or not mid_centroids:
        return rgb_uint8
    overlay = rgb_uint8.copy()
    for (i, j) in mm_adj:
        ci = mid_centroids.get(i)
        cj = mid_centroids.get(j)
        if ci is None or cj is None:
            continue
        cv2.line(overlay,
                 (int(ci[1]), int(ci[0])),
                 (int(cj[1]), int(cj[0])),
                 MM_EDGE_COLOR, 1, lineType=cv2.LINE_AA)
    return cv2.addWeighted(overlay, MM_EDGE_ALPHA, rgb_uint8, 1 - MM_EDGE_ALPHA, 0)



# =========================================================================== #
#  COLOUR SCALE & RENDERING HELPERS                                            #
# =========================================================================== #

def resolve_vmin_vmax(ch, arr):
    cfg = CHANNEL_CFG[ch]
    if cfg["vmin"] is not None:
        return cfg["vmin"], cfg["vmax"]
    valid = arr[arr != 0]
    if valid.size == 0:
        return 0.0, 1.0
    return float(valid.min()), float(valid.max())


def frame_to_rgb(frame, ch, vmin, vmax):
    norm = np.clip((frame - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
    rgba = CHANNEL_CFG[ch]["cmap"](norm)
    return rgba[:, :, :3].astype(np.float32)


def render_boundary(rgb, labels, color, mode="thick"):
    # background_label=-1 ensures all internal sub-segmentations are drawn
    out = mark_boundaries(rgb, labels, color=color, mode=mode, background_label=-1)
    return (out * 255).clip(0, 255).astype(np.uint8)


# =========================================================================== #
#  PRE-COMPUTATION                                                              #
# =========================================================================== #

def precompute_hierarchical(fused, data, dem_norm):
    T = fused.shape[0]
    H, W = fused.shape[1], fused.shape[2]
    display_chs = [ch for ch in DISPLAY_CHANNELS if ch in data]

    vranges = {ch: resolve_vmin_vmax(ch, data[ch]) for ch in display_chs}

    tslic = TemporalSLIC_DEM(dem_norm, n_segments=N_SEGMENTS_MIDDLE,
                             compactness=COMPACTNESS, lambda_z=ELEVATION_LAMBDA)

    hier_rendered  = {ch: [] for ch in display_chs}
    flat_rendered  = {ch: [] for ch in display_chs}
    hier_store     = []
    flows_store    = []

    for t in range(T):
        log.info(f"  Frame {t + 1}/{T} …")

        # ── Middle-level segmentation ────────────────────────────────────── #
        mid_labels, flow = tslic.segment(fused[t], use_flow=True)
        mid_centroids    = dict(tslic.prev_centroids)
        flows_store.append(flow)

        # ── Fine-level sub-segmentation ──────────────────────────────────── #
        vil_norm = fused[t, :, :, 0]
        fine_labels, fine_centroids, parent_map = build_fine_nodes(mid_labels, vil_norm)
        log.info(f"    Fine nodes: N^f = {len(fine_centroids)}  |  "
                 f"Middle nodes: N^m = {len(mid_centroids)}")

        # ── E_mm topological adjacency (§4.1, eq. 4.3a) ─────────────────── #
        mm_adj = compute_middle_adjacency(mid_labels)
        log.info(f"    E_mm edges: {len(mm_adj)}  (k_avg ≈ "
                 f"{2*len(mm_adj)/max(len(mid_centroids),1):.1f} per node)")

        # ── Global Context Readout g(t) stats (§13) ──────────────────────── #
        gcr_stats = compute_gcr_stats(mid_labels, vil_norm)

        hier_store.append({
            "mid_labels":     mid_labels,
            "mid_centroids":  mid_centroids,
            "fine_labels":    fine_labels,
            "fine_centroids": fine_centroids,
            "parent_map":     parent_map,
            "mm_adj":         mm_adj,
            "gcr_stats":      gcr_stats,
        })

        # ── Render channels ──────────────────────────────────────────────── #
        for ch in display_chs:
            vmin, vmax = vranges[ch]
            rgb = frame_to_rgb(data[ch][t], ch, vmin, vmax)

            # TOP ROW: hierarchical boundaries → E_mm edges → E_fm edges
            h_rgb = render_hierarchical_boundaries(
                rgb, fine_labels, mid_labels, FINE_COLOR, MIDDLE_COLOR)
            h_rgb = render_mm_edges(h_rgb, mm_adj, mid_centroids)
            h_rgb = render_fm_edges(h_rgb, fine_centroids, mid_centroids, parent_map)
            hier_rendered[ch].append(h_rgb)

            # BOTTOM ROW: flat Temporal SLIC boundaries in gold
            f_rgb = render_boundary(rgb, mid_labels, FLAT_TEMPORAL, mode="thick")
            flat_rendered[ch].append(f_rgb)

    return hier_rendered, flat_rendered, hier_store, flows_store


# =========================================================================== #
#  ANIMATION                                                                    #
# =========================================================================== #

def get_writer():
    if shutil.which("ffmpeg"):
        return animation.FFMpegWriter(fps=5, bitrate=2000), ".mp4"
    return animation.PillowWriter(fps=5), ".gif"


def animate_hierarchical(event_id, data, hier_rendered, flat_rendered,
                          hier_store, flows_store, out_dir=None):
    if out_dir is None:
        out_dir = EVENTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    display_chs = [ch for ch in DISPLAY_CHANNELS if ch in data]
    n_cols = len(display_chs)
    T = max(data[ch].shape[0] for ch in display_chs)
    total_min = T * SECONDS_PER_FRAME // 60

    writer, ext = get_writer()
    dpi = 90
    out_path = os.path.join(out_dir, f"{event_id}_hierarchical_graph{ext}")
    log.info(f"\nAnimating {T} frames ({total_min} min) → {out_path}")

    fig_h = 10
    fig_w = 6 * n_cols
    fig, axes = plt.subplots(
        nrows=2, ncols=n_cols,
        figsize=(fig_w, fig_h),
        gridspec_kw={"hspace": 0.12, "wspace": 0.06},
    )
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    fig.patch.set_facecolor("#0d0d0d")
    for ax in axes.flat:
        ax.set_facecolor("#0d0d0d")
        ax.axis("off")

    for col_idx, ch in enumerate(display_chs):
        axes[0, col_idx].set_title(ch.upper(), fontsize=14, fontweight="bold",
                                   pad=10, color="white")

    row_labels = ["Hierarchical Graph\n(Fine → Middle)",
                  "Flat Lagrangian DRIFT\n(Temporal SLIC)"]
    for row_idx, lbl in enumerate(row_labels):
        axes[row_idx, 0].set_ylabel(lbl, fontsize=10, labelpad=10, color="white",
                                    fontweight="bold")
        axes[row_idx, 0].yaxis.label.set_color("white")

    legend_patches = [
        mpatches.Patch(facecolor=FINE_COLOR,   edgecolor="white",
                       label="Fine nodes  (~3–6 km)"),
        mpatches.Patch(facecolor=MIDDLE_COLOR, edgecolor="white",
                       label="Middle nodes (~11 km)"),
        mpatches.Patch(color=(FM_EDGE_COLOR[0]/255, FM_EDGE_COLOR[1]/255, FM_EDGE_COLOR[2]/255),
                       label="E_fm edges (fine→mid)") if SHOW_FM_EDGES else None,
        mpatches.Patch(color=(MM_EDGE_COLOR[0]/255, MM_EDGE_COLOR[1]/255, MM_EDGE_COLOR[2]/255),
                       label="E_mm edges (mid adj.)") if SHOW_MM_EDGES else None,
        mpatches.Patch(facecolor=FLAT_TEMPORAL, edgecolor="white",
                       label="Flat DRIFT (Temporal SLIC)"),
    ]
    legend_patches = [p for p in legend_patches if p is not None]
    fig.legend(handles=legend_patches, loc="lower center", ncol=len(legend_patches),
               fontsize=9, framealpha=0.3, facecolor="#222222", edgecolor="gray",
               labelcolor="white", bbox_to_anchor=(0.5, 0.0))

    time_text = fig.text(
        0.5, 0.975, "", ha="center", fontsize=11, color="white",
        bbox=dict(boxstyle="round", facecolor="#111111", alpha=0.7),
    )
    fig.suptitle(
        f"DRIFT Hierarchical Graph vs Flat Lagrangian — Event {event_id}  ({total_min} min)",
        fontsize=13, y=0.998, color="white", fontweight="bold",
    )

    im_artists  = {}
    sc_artists  = {}
    qv_artists  = {}

    for row_idx in range(2):
        for col_idx, ch in enumerate(display_chs):
            ax = axes[row_idx, col_idx]
            init_img = (hier_rendered[ch][0] if row_idx == 0 else flat_rendered[ch][0])
            im = ax.imshow(init_img, origin="lower", animated=True, aspect="auto")
            ax.axis("off")
            im_artists[(row_idx, col_idx)]  = im
            sc_artists[(row_idx, col_idx)]  = []
            qv_artists[(row_idx, col_idx)]  = None

    plt.tight_layout(rect=[0, 0.05, 1, 0.975])

    def _update(t):
        updated = []
        h_info = hier_store[min(t, len(hier_store) - 1)]
        mid_c  = h_info["mid_centroids"]
        fine_c = h_info["fine_centroids"]
        flow   = flows_store[min(t, len(flows_store) - 1)]

        for row_idx in range(2):
            for col_idx, ch in enumerate(display_chs):
                ax  = axes[row_idx, col_idx]
                key = (row_idx, col_idx)

                T_ch   = (hier_rendered[ch] if row_idx == 0 else flat_rendered[ch])
                t_safe = min(t, len(T_ch) - 1)

                im_artists[key].set_data(T_ch[t_safe])
                updated.append(im_artists[key])

                H_ax = data[ch].shape[1]
                W_ax = data[ch].shape[2]

                for artist in sc_artists[key]:
                    artist.remove()
                sc_artists[key].clear()

                if row_idx == 0:
                    if fine_c:
                        fy = np.array([v[0] for v in fine_c.values()])
                        fx = np.array([v[1] for v in fine_c.values()])
                        sc = ax.scatter(fx, fy, c=[FINE_COLOR], s=4,
                                        linewidths=0, zorder=6, alpha=0.85)
                        sc_artists[key].append(sc)
                        updated.append(sc)

                    if mid_c:
                        my = np.array([v[0] for v in mid_c.values()])
                        mx = np.array([v[1] for v in mid_c.values()])
                        sc2 = ax.scatter(mx, my, c=[MIDDLE_COLOR], s=10,
                                         linewidths=0.3, zorder=7,
                                         edgecolors="white", alpha=0.9)
                        sc_artists[key].append(sc2)
                        updated.append(sc2)
                else:
                    if mid_c:
                        my = np.array([v[0] for v in mid_c.values()])
                        mx = np.array([v[1] for v in mid_c.values()])
                        sc = ax.scatter(mx, my, c="red", s=6,
                                        linewidths=0, zorder=5)
                        sc_artists[key].append(sc)
                        updated.append(sc)

                    if qv_artists[key] is not None:
                        qv_artists[key].remove()
                        qv_artists[key] = None
                    ys = np.arange(0, H_ax, QUIVER_STRIDE)
                    xs = np.arange(0, W_ax, QUIVER_STRIDE)
                    xg, yg = np.meshgrid(xs, ys)
                    u = flow[yg, xg, 0]
                    v_flow = flow[yg, xg, 1]
                    valid = np.hypot(u, v_flow) > 0.5
                    if valid.any():
                        qv = ax.quiver(xg[valid], yg[valid],
                                       u[valid], v_flow[valid],
                                       color="white", alpha=0.55,
                                       scale=120, width=0.003, zorder=4)
                        qv_artists[key] = qv
                        updated.append(qv)

        elapsed_min = t * SECONDS_PER_FRAME // 60
        h_info_t = hier_store[min(t, len(hier_store) - 1)]
        gcr      = h_info_t.get("gcr_stats", {})
        N_f      = len(h_info_t["fine_centroids"])
        N_m      = len(h_info_t["mid_centroids"])
        gcr_str  = (f"  |  GCR g(t): μ={gcr.get('mean_vil',0):.3f} "
                    f"σ={gcr.get('std_vil',0):.3f} "
                    f"max={gcr.get('max_vil',0):.3f}")
        time_text.set_text(
            f"T+{elapsed_min:03d} min   frame {t + 1}/{T}   "
            f"[N^f={N_f}, N^m={N_m}]{gcr_str}"
        )
        updated.append(time_text)
        return updated

    anim = animation.FuncAnimation(fig, _update, frames=T, interval=200, blit=False)
    log.info(f"Writing animation at dpi={dpi} …")
    anim.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    log.info(f"Saved: {out_path}")
    return out_path


# =========================================================================== #
#  SINGLE STATIC FRAME PREVIEW  (useful for quick inspection)                  #
# =========================================================================== #

def save_static_preview(event_id, data, hier_rendered, flat_rendered,
                         hier_store, t_idx=0, out_dir=None):
    """
    Save a single-frame PNG showing both rows side-by-side.

    TOP ROW  — Hierarchical graph (Fine → Middle) with E_fm and E_mm edges
               pre-baked into the rendered image, plus GCR g(t) annotation.
    BOTTOM ROW — Flat Lagrangian DRIFT (Temporal SLIC boundaries).
    """
    if out_dir is None:
        out_dir = EVENTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    display_chs = [ch for ch in DISPLAY_CHANNELS if ch in data]
    n_cols = len(display_chs)
    h_info = hier_store[min(t_idx, len(hier_store) - 1)]
    mid_c    = h_info["mid_centroids"]
    fine_c   = h_info["fine_centroids"]
    gcr      = h_info.get("gcr_stats", {})
    mm_adj   = h_info.get("mm_adj", set())

    fig, axes = plt.subplots(
        nrows=2, ncols=n_cols,
        figsize=(6 * n_cols, 10),
        gridspec_kw={"hspace": 0.12, "wspace": 0.06},
    )
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    fig.patch.set_facecolor("#0d0d0d")

    row_labels = ["Hierarchical Graph  (Fine → Middle)",
                  "Flat Lagrangian DRIFT  (Temporal SLIC)"]
    t_safe = min(t_idx, len(hier_rendered[display_chs[0]]) - 1)

    for row_idx in range(2):
        for col_idx, ch in enumerate(display_chs):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("#0d0d0d")
            ax.axis("off")

            img = (hier_rendered[ch][t_safe] if row_idx == 0
                   else flat_rendered[ch][t_safe])
            ax.imshow(img, origin="lower", aspect="auto")

            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx], fontsize=10, color="white",
                              fontweight="bold", labelpad=8)
                ax.yaxis.label.set_color("white")

            if row_idx == 0:
                ax.set_title(ch.upper(), fontsize=13, fontweight="bold",
                             color="white", pad=8)

                # Fine centroids (magenta dots)
                if fine_c:
                    fy = [v[0] for v in fine_c.values()]
                    fx = [v[1] for v in fine_c.values()]
                    ax.scatter(fx, fy, c=[FINE_COLOR], s=4, zorder=6,
                               linewidths=0, alpha=0.85)

                # Middle centroids (red dots with white outline)
                if mid_c:
                    my = [v[0] for v in mid_c.values()]
                    mx = [v[1] for v in mid_c.values()]
                    ax.scatter(mx, my, c=[MIDDLE_COLOR], s=10,
                               edgecolors="white", linewidths=0.3, zorder=7)

                # GCR g(t) annotation — top-right corner (only first column)
                if gcr and col_idx == 0:
                    K_avg = len(fine_c) / max(len(mid_c), 1)
                    gcr_txt = (
                        f"GCR  g(t)  [§13]\n"
                        f"μ_VIL = {gcr.get('mean_vil', 0):.3f}\n"
                        f"σ_VIL = {gcr.get('std_vil',  0):.3f}\n"
                        f"max   = {gcr.get('max_vil',   0):.3f}\n"
                        f"N^f={len(fine_c)}  N^m={len(mid_c)}\n"
                        f"K_avg={K_avg:.1f}  |E_mm|={len(mm_adj)}"
                    )
                    ax.text(
                        0.99, 0.99, gcr_txt,
                        transform=ax.transAxes, fontsize=6.5,
                        color="#FFD700", va="top", ha="right",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="#111111", alpha=0.72,
                                  edgecolor="#555555"),
                        zorder=10,
                    )

            else:
                # Middle centroids (red) for the flat DRIFT row
                if mid_c:
                    my = [v[0] for v in mid_c.values()]
                    mx = [v[1] for v in mid_c.values()]
                    ax.scatter(mx, my, c="red", s=6, zorder=5, linewidths=0)

    # ── Legend ───────────────────────────────────────────────────────────── #
    def _rgb01(bgr_255):
        return (bgr_255[2]/255, bgr_255[1]/255, bgr_255[0]/255)

    legend_patches = [
        mpatches.Patch(facecolor=FINE_COLOR,   edgecolor="white",
                       label=f"Fine nodes  N^f≈{len(fine_c)}  (~3–6 km)"),
        mpatches.Patch(facecolor=MIDDLE_COLOR, edgecolor="white",
                       label=f"Middle nodes  N^m≈{len(mid_c)}  (~11 km)"),
    ]
    if SHOW_FM_EDGES:
        legend_patches.append(
            mpatches.Patch(color=_rgb01(FM_EDGE_COLOR),
                           label="E_fm  fine→mid  (§4.2 eq.4.6)"))
    if SHOW_MM_EDGES:
        legend_patches.append(
            mpatches.Patch(color=_rgb01(MM_EDGE_COLOR),
                           label=f"E_mm  mid adj.  |E|={len(mm_adj)}  (§4.1 eq.4.3a)"))
    legend_patches.append(
        mpatches.Patch(facecolor=FLAT_TEMPORAL, edgecolor="white",
                       label="Flat DRIFT (Temporal SLIC)"))

    fig.legend(handles=legend_patches, loc="lower center",
               ncol=min(len(legend_patches), 5),
               fontsize=8.5, framealpha=0.35,
               facecolor="#222222", edgecolor="gray",
               labelcolor="white", bbox_to_anchor=(0.5, 0.0))

    elapsed_min = t_idx * SECONDS_PER_FRAME // 60
    fig.suptitle(
        f"DRIFT Hierarchical Graph  ←→  Flat Lagrangian   |   Event {event_id}   "
        f"T+{elapsed_min:03d} min   [N^f={len(fine_c)}, N^m={len(mid_c)}]",
        fontsize=13, y=0.998, color="white", fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.975])
    png_path = os.path.join(out_dir, f"{event_id}_hierarchical_preview_t{t_idx}.png")
    fig.savefig(png_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"Static preview saved: {png_path}")
    return png_path


# =========================================================================== #
#  MAIN                                                                         #
# =========================================================================== #

if __name__ == "__main__":
    use_synthetic = False

    # ── Try to load the real catalog ──────────────────────────────────────── #
    if os.path.exists(CATALOG_PATH):
        print("Loading catalog …")
        catalog = pd.read_csv(CATALOG_PATH, parse_dates=["time_utc"],
                              low_memory=False)

        while True:
            print("\n" + "=" * 60)
            event_id = input("Enter Event ID (or 's' for synthetic, 'q' to quit): ").strip()
            if event_id.lower() == "q":
                break
            if event_id.lower() == "s":
                use_synthetic = True
                event_id = "SYNTHETIC_001"
            elif not event_id:
                continue

            if use_synthetic:
                fused, data, extent = generate_synthetic_event()
            else:
                print(f"\nLoading data for {event_id} …")
                fused, data, extent = load_fused_channels(event_id, catalog)
                if fused is None or not data:
                    print("  [Error] No data found. Check event ID and file paths.")
                    use_synthetic = False
                    continue

            T = fused.shape[0]
            print(f"  Duration: {T} frames = {T * SECONDS_PER_FRAME // 60} min")
            print(f"  Feature cube: {fused.shape}  (T × H × W × C)")

            H, W = fused.shape[1], fused.shape[2]
            dem_raw  = fetch_and_regrid_dem(extent, nx=W, ny=H)
            dem_norm = normalise_dem(dem_raw)

            print("\nPre-computing hierarchical + flat segmentations …")
            hier_rendered, flat_rendered, hier_store, flows_store = \
                precompute_hierarchical(fused, data, dem_norm)

            # Static preview (frame 0)
            #png = save_static_preview(event_id, data, hier_rendered, flat_rendered, hier_store, t_idx=0, out_dir=EVENTS_DIR)
            #print(f"\nStatic preview → {png}")

            # Full animation
            out = animate_hierarchical(event_id, data, hier_rendered, flat_rendered,
                                       hier_store, flows_store, out_dir=EVENTS_DIR)
            print(f"\nDone → {out}")
            use_synthetic = False

    else:
        # No catalog → run directly on synthetic data
        print("Catalog not found — running on synthetic SEVIR data.")
        event_id = "SYNTHETIC_001"
        fused, data, extent = generate_synthetic_event()
        T = fused.shape[0]
        H, W = fused.shape[1], fused.shape[2]
        dem_raw  = fetch_dem_or_zeros(extent, H, W)
        dem_norm = normalise_dem(dem_raw)

        print("Pre-computing hierarchical + flat segmentations …")
        hier_rendered, flat_rendered, hier_store, flows_store = \
            precompute_hierarchical(fused, data, dem_norm)

        #png = save_static_preview(event_id, data, hier_rendered, flat_rendered, hier_store, t_idx=0, out_dir=".")
        #print(f"\nStatic preview → {png}")

        out = animate_hierarchical(event_id, data, hier_rendered, flat_rendered,
                                   hier_store, flows_store, out_dir=".")
        print(f"\nDone → {out}")