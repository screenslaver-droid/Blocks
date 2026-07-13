import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.segmentation import slic, watershed
from skimage.filters import sobel
from skimage.morphology import disk, dilation
from skimage.feature import peak_local_max
from scipy.ndimage import label, center_of_mass
import cv2
import os
import logging
import time
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import cartopy.crs as ccrs
from tqdm import tqdm

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("sevir_debug.log", mode="w")
    ]
)
log = logging.getLogger(__name__)

for noisy in ("rasterio", "urllib3", "botocore", "pystac_client", "planetary_computer"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

def checkpoint(label: str):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            log.debug(f"[CHECKPOINT START] {label}")
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            log.debug(f"[CHECKPOINT END  ] {label} — {elapsed:.3f}s")
            return result
        return wrapper
    return decorator

# --- DEM DEPENDENCIES ---
try:
    import pystac_client
    import planetary_computer
    import rioxarray
    from rioxarray.merge import merge_arrays
    HAS_DEM_LIBS = True
    log.info("DEM libraries loaded successfully.")
except ImportError:
    HAS_DEM_LIBS = False
    log.warning("Missing DEM libraries (pystac_client / rioxarray). Falling back to flat terrain.")

# --- CONFIGURATION ---
BASE_PATH   = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\SEVIR\2019"
CATALOG_PATH = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\SEVIR\CATALOG.csv"

USE_OPTICAL_FLOW  = True  # True = Temporal SLIC, False = Standard SLIC (Static Grid)
ELEVATION_LAMBDA  = 0.5    
N_SEGMENTS        = 1500

SEVIR_PROJ = ccrs.LambertConformal(
    central_longitude=-98.0, central_latitude=38.0,
    standard_parallels=(30.0, 60.0),
    globe=ccrs.Globe(semimajor_axis=6370000, semiminor_axis=6370000)
)

# --------------------------------------------------------------------------- #
# 1. DEM PIPELINE (FIXED MEMORY LEAK)
# --------------------------------------------------------------------------- #
@checkpoint("get_sevir_grid")
def get_sevir_grid(extent, nx: int = 384, ny: int = 384):
    proj = SEVIR_PROJ
    src  = ccrs.PlateCarree()
    x0, y0 = proj.transform_point(extent[0], extent[2], src)
    x1, y1 = proj.transform_point(extent[1], extent[3], src)
    xv, yv = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    grid   = src.transform_points(proj, xv, yv)
    return grid[..., 0], grid[..., 1]

_dem_cache: dict = {}

def _load_and_clip_tile(item, extent, buf=0.1):
    da = rioxarray.open_rasterio(item.assets["data"].href, lock=False).squeeze()
    # 1. Clip bounding box
    da = da.rio.clip_box(
        minx=extent[0] - buf, miny=extent[2] - buf,
        maxx=extent[1] + buf, maxy=extent[3] + buf
    )
    # 2. THE FIX: Coarsen 30m to 90m. Reduces RAM by 9x and speeds up interpolation 100x.
    da = da.coarsen(x=3, y=3, boundary="trim").mean()
    return da

@checkpoint("fetch_and_regrid_dem")
def fetch_and_regrid_dem(extent, nx: int = 384, ny: int = 384) -> np.ndarray:
    cache_key = tuple(np.round(extent, 4))
    if cache_key in _dem_cache:
        return _dem_cache[cache_key]

    if not HAS_DEM_LIBS: return np.zeros((ny, nx))

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=["cop-dem-glo-30"],
        bbox=[extent[0], extent[2], extent[1], extent[3]],
    )
    items = list(search.item_collection())

    if len(items) == 0: return np.zeros((ny, nx))

    datasets = [None] * len(items)
    with ThreadPoolExecutor(max_workers=min(len(items), 8)) as pool:
        future_to_idx = {pool.submit(_load_and_clip_tile, item, extent): i for i, item in enumerate(items)}
        for future in tqdm(as_completed(future_to_idx), total=len(items), desc="  DEM tiles"):
            idx = future_to_idx[future]
            try: datasets[idx] = future.result()
            except Exception as exc: log.warning(f"  Tile failed: {exc}")
            
    datasets = [d for d in datasets if d is not None and d.size > 0]
    full_dem = merge_arrays(datasets) if len(datasets) > 1 else datasets[0]

    clipped_lons, clipped_lats = full_dem.x.values, full_dem.y.values
    clipped_vals = full_dem.values

    if clipped_lats[0] > clipped_lats[-1]:
        clipped_lats = clipped_lats[::-1]
        clipped_vals = clipped_vals[::-1, :]

    interp = RegularGridInterpolator(
        (clipped_lats, clipped_lons), clipped_vals.astype(np.float32),
        method="linear", bounds_error=False, fill_value=0.0,
    )
    target_lons, target_lats = get_sevir_grid(extent, nx, ny)
    query_pts = np.column_stack((target_lats.ravel(), target_lons.ravel()))
    result = interp(query_pts).reshape(ny, nx).astype(np.float32)

    _dem_cache[cache_key] = result
    return result

# --------------------------------------------------------------------------- #
# 2. DATA FUSION
# --------------------------------------------------------------------------- #
class SevirFusion:
    FRAME_COUNT = 49
    def __init__(self, vil_shape=(384, 384)):
        self.target_shape = vil_shape

    @checkpoint("SevirFusion.process_ir")
    def process_ir(self, ir_data: np.ndarray) -> np.ndarray:
        if ir_data.ndim == 3 and ir_data.shape[2] == self.FRAME_COUNT:
            ir_data = ir_data.transpose(2, 0, 1)
        th, tw = self.target_shape
        ir_upsampled = np.empty((self.FRAME_COUNT, th, tw), dtype=np.float32)
        for t in range(self.FRAME_COUNT):
            ir_upsampled[t] = cv2.resize(ir_data[t].astype(np.float32), (tw, th), interpolation=cv2.INTER_CUBIC)
        return ir_upsampled

# --------------------------------------------------------------------------- #
# 3. FLOW ADHERENCE ERROR (FAE)
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 3. FLOW ADHERENCE ERROR (FAE)
# --------------------------------------------------------------------------- #
def calculate_fae(centroids_t: dict, centroids_t1: dict, optical_flow: np.ndarray, labels_t: np.ndarray) -> float:
    """
    Vectorised Flow Adherence Error.
    Extracts strictly the labels that survived from t to t+1 and calculates error.
    """
    # 1. Find strictly common labels (Intersection of keys)
    common_labels = sorted(list(set(centroids_t.keys()) & set(centroids_t1.keys())))
    if not common_labels: 
        return 0.0

    # 2. Extract coordinates safely guaranteed to match in shape
    # centroids are in (y, x) format
    coords_t  = np.array([centroids_t[l] for l in common_labels], dtype=np.float32)
    coords_t1 = np.array([centroids_t1[l] for l in common_labels], dtype=np.float32)

    # 3. Actual displacement (Reverse to x, y)
    vel_actual = coords_t1[:, ::-1] - coords_t[:, ::-1]  # (N, 2) -> [dx, dy]

    # 4. Expected displacement (Optical Flow)
    flow_u = optical_flow[..., 0]
    flow_v = optical_flow[..., 1]

    flat_labels = labels_t.ravel()
    flat_u      = flow_u.ravel()
    flat_v      = flow_v.ravel()

    # Use bincount for O(pixels) aggregation
    max_label = int(max(common_labels)) + 1
    sum_u     = np.bincount(flat_labels, weights=flat_u, minlength=max_label)
    sum_v     = np.bincount(flat_labels, weights=flat_v, minlength=max_label)
    count     = np.bincount(flat_labels, minlength=max_label).clip(min=1)

    mean_flow_u = sum_u[common_labels] / count[common_labels]
    mean_flow_v = sum_v[common_labels] / count[common_labels]
    mean_flow   = np.stack([mean_flow_u, mean_flow_v], axis=1)  # (N, 2) -> [fu, fv]

    # 5. Calculate L2 Norm (Euclidean distance error in pixels)
    errors = np.linalg.norm(vel_actual - mean_flow, axis=1)
    return float(errors.mean())

# --------------------------------------------------------------------------- #
# 4. TEMPORAL SLIC
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
# 5. MAIN EXECUTION (FIXED INDEX LOOKUP)
# --------------------------------------------------------------------------- #
@checkpoint("analyze_event_fae")
def analyze_event_fae(event_id: str, catalog: pd.DataFrame, use_optical_flow: bool) -> dict:
    rows = catalog[catalog["id"] == event_id]
    if rows.empty: return {}
    row = rows.iloc[0]
    extent = [row["llcrnrlon"], row["urcrnrlon"], row["llcrnrlat"], row["urcrnrlat"]]
    
    dem_raw  = fetch_and_regrid_dem(extent)
    dem_norm = (dem_raw - dem_raw.min()) / ((dem_raw.max() - dem_raw.min()) + 1e-6)

    fusion  = SevirFusion()
    results = {}

    for ch in ["vil", "ir107"]:
        fname = os.path.basename(row["file_name"])
        path  = os.path.join(BASE_PATH, ch, fname)
        if not os.path.exists(path): continue

        t_load = time.perf_counter()
        
        # THE FIX: Only extract the single event from the massive file
        with h5py.File(path, "r") as f:
            if 'id' in f and ch in f:
                ids = f['id'][:]
                if len(ids) > 0 and isinstance(ids[0], bytes):
                    ids = [x.decode('utf-8') for x in ids]
                idx = np.where(np.array(ids) == event_id)[0][0]
                raw = f[ch][idx]  # ONLY GRAB THE ONE EVENT
            elif event_id in f:
                raw = f[event_id][:]
            else:
                raw = f[list(f.keys())[0]][0] 
                
        log.debug(f"  HDF5 load: {path} | raw shape={raw.shape} | {time.perf_counter()-t_load:.3f}s")

        if ch == "vil":
            data = raw.transpose(2, 0, 1) if raw.shape[2] == 49 else raw
        else:
            data = fusion.process_ir(raw)

        T, H, W = data.shape
        tslic = TemporalSLIC_DEM(dem_norm, n_segments=N_SEGMENTS, lambda_z=ELEVATION_LAMBDA)
        fae_scores = []
        prev_labels, prev_centroids = None, None

        for t in tqdm(range(T), desc=f"  {ch.upper()}"):
            labels, flow = tslic.segment(data[t], use_flow=use_optical_flow)
            curr_centroids = tslic.prev_centroids

            if prev_labels is not None:
                fae = calculate_fae(prev_centroids, curr_centroids, flow, prev_labels)
                fae_scores.append(fae)

            prev_labels, prev_centroids = labels, curr_centroids

        results[ch] = fae_scores
        
    return results

if __name__ == "__main__":
    event_id = input("Enter Event ID: ").strip() or "S805933"
    catalog = pd.read_csv(CATALOG_PATH, parse_dates=["time_utc"], low_memory=False)

    # --- Run both modes ---
    MODE_CONFIG = {
        True:  {"label": "Temporal SLIC (Flow-Guided)", "color_vil": "#1f77b4", "color_ir": "#aec7e8", "ls": "-"},
        False: {"label": "Standard SLIC (Static)",      "color_vil": "#d62728", "color_ir": "#f7b6b6", "ls": "--"},
    }

    all_results: dict[bool, dict] = {}
    for use_flow in [True, False]:
        log.info(f"\n{'='*60}\nRunning with use_optical_flow={use_flow}\n{'='*60}")
        all_results[use_flow] = analyze_event_fae(event_id, catalog, use_flow)

    # --- Determine channels present in either run ---
    channels = sorted(
        set().union(*[set(r.keys()) for r in all_results.values()])
    )

    if not channels:
        log.error("No FAE results to plot — check event ID and file paths.")
    else:
        n_ch  = len(channels)
        fig, axes = plt.subplots(1, n_ch, figsize=(7 * n_ch, 5), sharey=False)
        if n_ch == 1:
            axes = [axes]

        for ax, ch in zip(axes, channels):
            for use_flow, cfg in MODE_CONFIG.items():
                scores = all_results[use_flow].get(ch)
                if scores is None or len(scores) == 0:
                    continue
                mu = np.mean(scores)
                ax.plot(
                    scores,
                    marker="o", markersize=4,
                    linestyle=cfg["ls"],
                    color=cfg["color_vil"] if ch == "vil" else cfg["color_ir"],
                    label=f"{cfg['label']}  (μ={mu:.2f} px)",
                    alpha=0.85,
                )

            ax.set_title(f"{ch.upper()} — FAE Comparison", fontsize=12, fontweight="bold")
            ax.set_xlabel("Frame Transition")
            ax.set_ylabel("Tracking Error (Pixels)")
            ax.legend(fontsize=9, loc="upper right")
            ax.grid(True, linestyle="--", alpha=0.5)

        fig.suptitle(
            f"Flow Adherence Error — Temporal SLIC vs Standard SLIC\n"
            f"Event: {event_id}  |  Lower = Superpixels track motion better",
            fontsize=13, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        plt.savefig(f"fae_comparison_{event_id}.png", dpi=150, bbox_inches="tight")
        log.info(f"Saved: fae_comparison_{event_id}.png")
        plt.show()