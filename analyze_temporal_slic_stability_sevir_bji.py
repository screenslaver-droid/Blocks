import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.segmentation import slic, find_boundaries, watershed
from skimage.filters import sobel
from skimage.morphology import disk, dilation
import cv2
import os
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from multiprocessing import Pool
import functools

# --- CONFIGURATION ---
BASE_PATH = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\SEVIR\2019"
CATALOG_PATH = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\SEVIR\CATALOG.csv"

# Performance tuning parameters
USE_PARALLEL = True       # Use multiprocessing for SLIC
FRAME_SKIP = 1            # Analyze every Nth frame (1=all frames for flow consistency)
N_SEGMENTS = 150          # Number of superpixels
USE_OPTICAL_FLOW = True   # Enable optical flow tracking (set False for standard SLIC)

# --- FUSION HELPER ---
class SevirFusion:
    def __init__(self, vil_shape=(384, 384)):
        self.target_shape = vil_shape
        self.frame_count = 49 

    def process_ir(self, ir_data):
        """Upsample IR data (49, 192, 192) -> (49, 384, 384)"""
        if ir_data.ndim == 3 and ir_data.shape[2] == 49:
            ir_data = ir_data.transpose(2, 0, 1)
        
        ir_upsampled = np.zeros((self.frame_count, *self.target_shape), dtype=np.float32)
        for t in range(self.frame_count):
            ir_upsampled[t] = cv2.resize(
                ir_data[t].astype(np.float32), 
                (self.target_shape[1], self.target_shape[0]), 
                interpolation=cv2.INTER_CUBIC
            )
        return ir_upsampled

    def process_lightning(self, lght_data, extent):
        """Rasterize Lightning (N, 4) -> (49, 384, 384) grid"""
        grid = np.zeros((self.frame_count, *self.target_shape), dtype=np.float32)
        
        # Handle pre-gridded lightning data
        if lght_data.ndim == 3:
            if lght_data.shape[1] != self.target_shape[0]:
                return self.process_ir(lght_data)
            return lght_data.astype(np.float32)

        if lght_data.shape[0] == 0:  # No strikes
            return grid

        # Raw point data
        lats, lons = lght_data[:, 0], lght_data[:, 1]
        min_lon, max_lon, min_lat, max_lat = extent[0], extent[1], extent[2], extent[3]

        x = ((lons - min_lon) / (max_lon - min_lon) * self.target_shape[1]).astype(int)
        y = ((lats - min_lat) / (max_lat - min_lat) * self.target_shape[0]).astype(int)
        
        # Extract time information if available
        if lght_data.shape[1] > 2:
            possible_time = lght_data[:, -1]
            if np.max(possible_time) > 48:
                t_idx = (possible_time / np.max(possible_time) * 48).astype(int)
            else:
                t_idx = possible_time.astype(int)
            t_idx = np.clip(t_idx, 0, 48)
        else:
            # Distribute randomly if no time info
            t_idx = np.random.randint(0, 49, size=len(lats))
        
        valid = (x >= 0) & (x < self.target_shape[1]) & (y >= 0) & (y < self.target_shape[0])
        np.add.at(grid, (t_idx[valid], y[valid], x[valid]), 1)
        
        # Smooth each frame
        for t in range(self.frame_count):
            if np.any(grid[t] > 0):
                grid[t] = gaussian_filter(grid[t], sigma=1.0)
                
        return grid


# --- TEMPORAL SLIC CLASS ---
class TemporalSLIC:
    """
    Temporal superpixel segmentation with optional optical flow tracking.
    
    When use_flow=True: Tracks superpixel centroids across frames using optical flow
                        and uses watershed to update segmentation.
    When use_flow=False: Runs standard SLIC independently on each frame.
    """
    def __init__(self, n_segments=200, compactness=10, sigma=1, max_iter=5):
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
        self.max_iter = max_iter
        self.prev_labels = None
        self.prev_centroids = None
        self.prev_frame = None

    def get_centroids(self, labels):
        """Extract (label_id, y, x) centroids from a label map."""
        centroids = []
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == 0:  # Skip background
                continue
            coords = np.argwhere(labels == label)
            if len(coords) > 0:
                center = coords.mean(axis=0)
                centroids.append([label, center[0], center[1]])
                
        return np.array(centroids) if centroids else np.array([]).reshape(0, 3)

    def segment(self, frame, use_flow=True):
        """
        Segment a single frame using either standard SLIC or flow-guided watershed.
        
        Args:
            frame: 2D numpy array (H, W)
            use_flow: If True and not first frame, use optical flow tracking
            
        Returns:
            labels: Segmentation label map (same shape as frame)
        """
        # Normalize frame to [0, 1]
        frame_min, frame_max = np.min(frame), np.max(frame)
        if frame_max > frame_min:
            frame_norm = (frame - frame_min) / (frame_max - frame_min)
        else:
            # Constant frame - return empty segmentation
            return np.zeros_like(frame, dtype=int)

        # CASE 1: First frame or flow disabled - use standard SLIC
        if self.prev_labels is None or not use_flow:
            try:
                labels = slic(
                    frame_norm.astype(np.float64), 
                    n_segments=self.n_segments, 
                    compactness=self.compactness, 
                    sigma=self.sigma,
                    max_num_iter=self.max_iter,
                    start_label=1,
                    channel_axis=None
                )
            except Exception as e:
                print(f"      [Warning] SLIC failed: {e}")
                labels = np.zeros_like(frame, dtype=int)
        
        # CASE 2: Subsequent frames with flow - use optical flow + watershed
        else:
            # A. Calculate optical flow between previous and current frame
            prev_gray = (self.prev_frame * 255).astype(np.uint8)
            curr_gray = (frame_norm * 255).astype(np.uint8)
            
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray, None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0
                )
            except Exception as e:
                print(f"      [Warning] Optical flow failed: {e}, using standard SLIC")
                return self.segment(frame, use_flow=False)
            
            # B. Advect previous centroids using flow vectors
            markers = np.zeros_like(frame, dtype=int)
            
            if len(self.prev_centroids) > 0:
                for (label_id, y, x) in self.prev_centroids:
                    iy, ix = int(y), int(x)
                    
                    # Get flow vector at old centroid position
                    if 0 <= iy < flow.shape[0] and 0 <= ix < flow.shape[1]:
                        dx, dy = flow[iy, ix]
                        
                        # Calculate new position
                        new_x = int(np.clip(x + dx, 0, frame.shape[1] - 1))
                        new_y = int(np.clip(y + dy, 0, frame.shape[0] - 1))
                        
                        # Place marker with original label ID (tracking continuity)
                        markers[new_y, new_x] = int(label_id)
            
            # C. Dilate markers so watershed has seed regions
            if np.any(markers > 0):
                markers = dilation(markers, disk(2))
            else:
                # No valid markers - fall back to standard SLIC
                return self.segment(frame, use_flow=False)
            
            # D. Watershed segmentation using gradient as elevation map
            gradient = sobel(gaussian_filter(frame_norm, sigma=1))
            
            try:
                labels = watershed(
                    gradient, 
                    markers=markers, 
                    mask=(frame_norm > 0.01)  # Exclude very low values
                )
            except Exception as e:
                print(f"      [Warning] Watershed failed: {e}, using standard SLIC")
                return self.segment(frame, use_flow=False)

        # Update state for next iteration
        self.prev_labels = labels
        self.prev_frame = frame_norm
        self.prev_centroids = self.get_centroids(labels)
        
        return labels

    def reset(self):
        """Reset temporal state (use when starting new sequence)."""
        self.prev_labels = None
        self.prev_centroids = None
        self.prev_frame = None


def calculate_boundary_jaccard_index(labels1, labels2):
    """
    Calculate Boundary Jaccard Index (BJI) between two segmentations.
    
    BJI = |B1 ∩ B2| / |B1 ∪ B2|
    
    where B1 and B2 are the boundary pixel sets.
    
    Higher values indicate more stable/consistent boundaries.
    """
    # Extract boundaries
    boundaries1 = find_boundaries(labels1, mode='thick')
    boundaries2 = find_boundaries(labels2, mode='thick')
    
    # Calculate intersection and union
    intersection = np.logical_and(boundaries1, boundaries2).sum()
    union = np.logical_or(boundaries1, boundaries2).sum()
    
    # Calculate Jaccard index
    if union > 0:
        bji = intersection / union
    else:
        bji = 1.0  # Perfect match if both have no boundaries
    
    return bji


def read_single_event(filepath, event_id, img_type):
    """
    Load ONLY the requested event from HDF5 file.
    Critical optimization: avoids loading entire multi-event files.
    """
    if filepath is None or not os.path.exists(filepath):
        return None
        
    with h5py.File(filepath, 'r') as f:
        # Case 1: Event stored directly (common for lightning)
        if event_id in f:
            return f[event_id][:]
        
        # Case 2: Standard indexed format (common for VIL/IR)
        if 'id' in f and img_type in f:
            ids = f['id'][:]
            if len(ids) > 0 and isinstance(ids[0], bytes):
                ids = [x.decode('utf-8') for x in ids]
            
            if event_id in ids:
                idx = np.where(np.array(ids) == event_id)[0][0]
                return f[img_type][idx]  # Load only this index
        
        # Case 3: Fallback - try img_type key directly
        if img_type in f:
            return f[img_type][0]
    
    return None


def analyze_event(event_id, catalog, use_flow=True, frame_skip=1, n_segments=150):
    """
    Analyze temporal stability of an event using Boundary Jaccard Index.
    
    Args:
        event_id: SEVIR event identifier
        catalog: Pandas DataFrame with SEVIR catalog
        use_flow: Enable optical flow tracking
        frame_skip: Process every Nth frame
        n_segments: Number of superpixels
        
    Returns:
        results: Dict mapping channel names to lists of BJI values
    """
    # Get event metadata
    event_rows = catalog[catalog['id'] == event_id]
    if event_rows.empty:
        print(f"ERROR: Event {event_id} not found in catalog")
        return {}
        
    row = event_rows.iloc[0]
    extent = [row['llcrnrlon'], row['urcrnrlon'], row['llcrnrlat'], row['urcrnrlat']]
    
    print(f"\nEvent: {event_id}")
    print(f"Time: {row['time_utc']}")
    print(f"Location: [{extent[2]:.2f}°N, {extent[0]:.2f}°W] to [{extent[3]:.2f}°N, {extent[1]:.2f}°W]")
    print("-" * 70)
    
    fusion = SevirFusion()
    results = {}
    
    # Process each channel
    channels = ['vil', 'ir107', 'ir069', 'lght']
    
    for ch in channels:
        print(f"\n[{ch.upper()}] Processing...")
        
        # Get file path
        filepath = get_filename(event_id, ch, catalog)
        if not filepath:
            print(f"  [Warning] File path not found in catalog")
            continue
            
        if not os.path.exists(filepath):
            print(f"  [Warning] File does not exist: {os.path.basename(filepath)}")
            continue
        
        # Load data (optimized - only this event)
        print(f"  Loading from {os.path.basename(filepath)}...")
        raw_data = read_single_event(filepath, event_id, ch)
        
        if raw_data is None:
            print(f"  [Warning] Could not read data")
            continue
        
        print(f"  Raw shape: {raw_data.shape}, dtype: {raw_data.dtype}")
        
        # Process/fuse data to common format
        if ch == 'vil':
            # VIL is usually already (49, 384, 384)
            if raw_data.ndim == 3 and raw_data.shape[2] == 49:
                data = raw_data.transpose(2, 0, 1)
            else:
                data = raw_data
            data = data.astype(np.float32)
            
        elif 'ir' in ch:
            print(f"  Upsampling IR data...")
            data = fusion.process_ir(raw_data)
            
        elif ch == 'lght':
            print(f"  Rasterizing lightning data...")
            data = fusion.process_lightning(raw_data, extent)
            # Disable flow for lightning (discrete events, no continuity)
            use_flow_ch = False
        else:
            data = raw_data.astype(np.float32)
            use_flow_ch = use_flow
            
        # For other channels, use the global flow setting
        if ch != 'lght':
            use_flow_ch = use_flow
            
        print(f"  Processed shape: {data.shape}")
        
        # Run Temporal SLIC with Boundary Jaccard Index metric
        tslic = TemporalSLIC(
            n_segments=n_segments,
            compactness=10,
            sigma=1,
            max_iter=5
        )
        
        bji_scores = []
        prev_labels = None
        
        # Select frames to process
        frame_indices = range(0, data.shape[0], frame_skip)
        
        flow_status = "Flow-Guided" if use_flow_ch else "Standard SLIC"
        print(f"  Running {flow_status} segmentation on {len(list(frame_indices))} frames...")
        
        for t in tqdm(frame_indices, desc=f"  {ch.upper()}", leave=False):
            # Segment current frame
            current_labels = tslic.segment(data[t], use_flow=use_flow_ch)
            
            # Calculate BJI with previous frame
            if prev_labels is not None:
                bji = calculate_boundary_jaccard_index(prev_labels, current_labels)
                bji_scores.append(bji)
            
            prev_labels = current_labels
        
        results[ch] = bji_scores
        
        if bji_scores:
            mean_bji = np.mean(bji_scores)
            std_bji = np.std(bji_scores)
            print(f"  Boundary Jaccard Index: {mean_bji:.4f} ± {std_bji:.4f}")
            print(f"  Range: [{np.min(bji_scores):.4f}, {np.max(bji_scores):.4f}]")
        else:
            print(f"  [Warning] No BJI scores computed")

    return results


def get_filename(event_id, img_type, catalog):
    """Find the filepath for a given event and image type."""
    row = catalog[(catalog['id'] == event_id) & (catalog['img_type'] == img_type)]
    if row.empty:
        return None
    return os.path.join(BASE_PATH, img_type, os.path.basename(row.iloc[0]['file_name']))


# --- MAIN ---
if __name__ == "__main__":
    print("=" * 70)
    print("SEVIR TEMPORAL STABILITY ANALYZER")
    print("Metric: Boundary Jaccard Index (BJI)")
    print("=" * 70)
    
    # Get event ID from user
    event_id = input("\nEnter Event ID (or press Enter for 'S834603'): ").strip()
    if not event_id:
        event_id = "S834603"
    
    # Configuration
    print(f"\nConfiguration:")
    print(f"  Optical Flow: {USE_OPTICAL_FLOW}")
    print(f"  Frame Skip: {FRAME_SKIP}")
    print(f"  Superpixels: {N_SEGMENTS}")
    print(f"  Parallel Processing: {USE_PARALLEL}")
    
    # Load catalog
    if not os.path.exists(CATALOG_PATH):
        print(f"\nERROR: Catalog not found at {CATALOG_PATH}")
        exit(1)
        
    catalog = pd.read_csv(CATALOG_PATH, parse_dates=['time_utc'], low_memory=False)
    
    # Analyze event
    results = analyze_event(
        event_id, 
        catalog, 
        use_flow=USE_OPTICAL_FLOW,
        frame_skip=FRAME_SKIP,
        n_segments=N_SEGMENTS
    )
    
    # Plot results
    if results:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        
        plt.figure(figsize=(12, 7))
        
        colors = {'vil': '#1f77b4', 'ir107': '#ff7f0e', 'ir069': '#2ca02c', 'lght': '#d62728'}
        
        for ch, bji_scores in results.items():
            if len(bji_scores) > 0:
                mean_bji = np.mean(bji_scores)
                std_bji = np.std(bji_scores)
                
                print(f"\n{ch.upper()}:")
                print(f"  Mean BJI: {mean_bji:.4f} ± {std_bji:.4f}")
                print(f"  Range: [{np.min(bji_scores):.4f}, {np.max(bji_scores):.4f}]")
                print(f"  Median: {np.median(bji_scores):.4f}")
                
                # Plot with frame indices adjusted for skipping
                x_values = np.arange(len(bji_scores)) * FRAME_SKIP + 1
                
                plt.plot(
                    x_values, 
                    bji_scores, 
                    marker='o', 
                    markersize=4,
                    label=f"{ch.upper()} (μ={mean_bji:.3f}, σ={std_bji:.3f})",
                    color=colors.get(ch, None),
                    linewidth=2,
                    alpha=0.8
                )
        
        plt.xlabel('Frame Transition', fontsize=12)
        plt.ylabel('Boundary Jaccard Index (BJI)', fontsize=12)
        
        flow_method = "Optical Flow-Guided" if USE_OPTICAL_FLOW else "Standard SLIC"
        plt.title(f'Temporal Stability Analysis - Event {event_id}\n{flow_method}', fontsize=14)
        
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.ylim([0, 1])  # BJI is between 0 and 1
        plt.tight_layout()
        
        # Save plot
        output_filename = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\SEVIR\bji_stability_True_S805933.png"
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved as: {output_filename}")
        
        plt.show()
    else:
        print("\nNo results to plot - no data was successfully processed")