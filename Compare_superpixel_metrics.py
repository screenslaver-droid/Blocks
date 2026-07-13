import numpy as np
import cv2
from scipy.spatial.distance import directed_hausdorff, cdist
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import find_boundaries
from tqdm import tqdm
from Visualize_sevir_with_superpixel_fused import TemporalSLIC_DEM, load_fused_channels, fetch_and_regrid_dem, SECONDS_PER_FRAME
from Visualize_sevir_with_superpixel_fused_adaptive_residual_tslic import AdaptiveResidualTSLIC_DEM

class SuperpixelEvaluator:
    """
    Evaluates temporal superpixel segmentations using physical fluid dynamics
    and graph theory metrics on a fused satellite/DEM feature cube.
    """
    def __init__(self, fused_cube: np.ndarray, flow_fields: list):
        """
        Args:
            fused_cube: (T, H, W, C) float32 array of normalized fused features.
            flow_fields: list of (H, W, 2) float32 optical flow arrays.
        """
        self.fused_cube = fused_cube
        self.flow_fields = flow_fields
        self.T, self.H, self.W, self.C = fused_cube.shape

    # ----------------------------------------------------------------------- #
    # 1. Flow Adherence Error (FAE)
    # ----------------------------------------------------------------------- #
    def calculate_fae(self, centroids_t: dict, centroids_t1: dict, 
                      optical_flow: np.ndarray, labels_t: np.ndarray) -> float:
        """Vectorised Flow Adherence Error for surviving labels."""
        common_labels = sorted(list(set(centroids_t.keys()) & set(centroids_t1.keys())))
        if not common_labels: 
            return 0.0

        # (N, 2) -> [y, x]
        coords_t  = np.array([centroids_t[l] for l in common_labels], dtype=np.float32)
        coords_t1 = np.array([centroids_t1[l] for l in common_labels], dtype=np.float32)

        # Actual displacement (Reverse to [dx, dy])
        vel_actual = coords_t1[:, ::-1] - coords_t[:, ::-1]  

        flow_u = optical_flow[..., 0]
        flow_v = optical_flow[..., 1]

        flat_labels = labels_t.ravel()
        max_label = int(max(common_labels)) + 1
        
        sum_u = np.bincount(flat_labels, weights=flow_u.ravel(), minlength=max_label)
        sum_v = np.bincount(flat_labels, weights=flow_v.ravel(), minlength=max_label)
        count = np.bincount(flat_labels, minlength=max_label).clip(min=1)

        mean_flow_u = sum_u[common_labels] / count[common_labels]
        mean_flow_v = sum_v[common_labels] / count[common_labels]
        mean_flow   = np.stack([mean_flow_u, mean_flow_v], axis=1)

        errors = np.linalg.norm(vel_actual - mean_flow, axis=1)
        return float(errors.mean())

    # ----------------------------------------------------------------------- #
    # 2. Boundary Displacement Variance (BDV / Hausdorff)
    # ----------------------------------------------------------------------- #
    def calculate_bdv(self, labels_t: np.ndarray, labels_t1: np.ndarray, 
                      optical_flow: np.ndarray) -> float:
        """
        Measures 'Bad Instability' by calculating the Hausdorff distance 
        between the flow-warped boundary of t and actual boundary of t+1.
        """
        bound_t = find_boundaries(labels_t, mode='inner')
        bound_t1 = find_boundaries(labels_t1, mode='inner')

        # Warp boundary_t using optical flow
        map_y = (np.arange(self.H, dtype=np.float32)[:, None] * np.ones((1, self.W), dtype=np.float32)) - optical_flow[:, :, 1]
        map_x = (np.arange(self.W, dtype=np.float32)[None, :] * np.ones((self.H, 1), dtype=np.float32)) - optical_flow[:, :, 0]
        
        bound_t_warped = cv2.remap(bound_t.astype(np.float32), map_x, map_y, 
                                   interpolation=cv2.INTER_NEAREST) > 0.5

        pts_warp = np.argwhere(bound_t_warped)
        pts_t1   = np.argwhere(bound_t1)

        if len(pts_warp) == 0 or len(pts_t1) == 0:
            return 0.0

        # Hausdorff distance
        d1 = directed_hausdorff(pts_warp, pts_t1)[0]
        d2 = directed_hausdorff(pts_t1, pts_warp)[0]
        return max(d1, d2)

    # ----------------------------------------------------------------------- #
    # 3. Graph Edit Distance (GED) Rate
    # ----------------------------------------------------------------------- #
    def calculate_ged_rate(self, tracked_t: set, tracked_t1: set) -> float:
        """Measures 'Good Instability' (Birth/Death of storm cores)."""
        born = len(tracked_t1 - tracked_t)
        dead = len(tracked_t - tracked_t1)
        total = len(tracked_t | tracked_t1)
        return (born + dead) / total if total > 0 else 0.0

    # ----------------------------------------------------------------------- #
    # 4. Intra-Superpixel Material Derivative Variance (MDV)
    # ----------------------------------------------------------------------- #
    def calculate_mdv(self, fused_t: np.ndarray, fused_t1: np.ndarray, 
                      labels_t1: np.ndarray, optical_flow: np.ndarray) -> float:
        """
        Penalizes superpixels if pixels inside them are changing in 
        physically conflicting ways (e.g. half growing, half decaying).
        """
        # Warp t to align with t+1
        map_y = (np.arange(self.H, dtype=np.float32)[:, None] * np.ones((1, self.W), dtype=np.float32)) - optical_flow[:, :, 1]
        map_x = (np.arange(self.W, dtype=np.float32)[None, :] * np.ones((self.H, 1), dtype=np.float32)) - optical_flow[:, :, 0]
        
        fused_t_warped = np.empty_like(fused_t)
        for c in range(self.C):
            fused_t_warped[..., c] = cv2.remap(fused_t[..., c], map_x, map_y, 
                                               interpolation=cv2.INTER_LINEAR, 
                                               borderMode=cv2.BORDER_REPLICATE)

        # R_pixel: The discrete material derivative (L2 norm across channels)
        R_pixel = np.linalg.norm(fused_t1 - fused_t_warped, axis=-1)

        unique_labels = np.unique(labels_t1)
        unique_labels = unique_labels[unique_labels != 0]
        
        mdv_weighted_sum = 0.0
        total_pixels = 0
        
        for lbl in unique_labels:
            mask = (labels_t1 == lbl)
            px_count = np.sum(mask)
            if px_count > 1:
                variance = np.var(R_pixel[mask])
                mdv_weighted_sum += variance * px_count
                total_pixels += px_count

        return mdv_weighted_sum / total_pixels if total_pixels > 0 else 0.0


    # ----------------------------------------------------------------------- #
    # Track Labels & Run Complete Evaluation
    # ----------------------------------------------------------------------- #
    def _track_labels(self, centroids_t: dict, centroids_t1_raw: dict, 
                      flow: np.ndarray, dist_thresh=15.0):
        """
        Bipartite matching to track IDs across frames (crucial for Adaptive SLIC
        which resets IDs to 1..N every frame).
        """
        if not centroids_t or not centroids_t1_raw:
            return centroids_t1_raw, {}

        # 1. Advect t centroids by flow
        advected_t = {}
        for lbl, (y, x) in centroids_t.items():
            iy, ix = int(np.clip(y, 0, self.H-1)), int(np.clip(x, 0, self.W-1))
            fy, fx = flow[iy, ix, 1], flow[iy, ix, 0]
            advected_t[lbl] = np.array([y + fy, x + fx])

        t_keys = list(advected_t.keys())
        t_pts  = np.array([advected_t[k] for k in t_keys])
        
        t1_keys_raw = list(centroids_t1_raw.keys())
        t1_pts = np.array([centroids_t1_raw[k] for k in t1_keys_raw])

        # 2. Cost matrix (Euclidean distance)
        cost_matrix = cdist(t_pts, t1_pts)
        
        # 3. Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        tracked_centroids_t1 = {}
        next_new_id = max(t_keys) + 1 if t_keys else 1
        
        # Assign matches within threshold
        matched_t1_indices = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < dist_thresh:
                tracked_centroids_t1[t_keys[r]] = t1_pts[c]
                matched_t1_indices.add(c)
        
        # Assign new IDs to unassigned nodes (Births)
        for c in range(len(t1_pts)):
            if c not in matched_t1_indices:
                tracked_centroids_t1[next_new_id] = t1_pts[c]
                next_new_id += 1
                
        return tracked_centroids_t1

    def evaluate(self, labels_list: list, centroids_raw_list: list):
        """Runs the full metric suite over the video."""
        metrics = {"FAE": [], "BDV": [], "GED": [], "MDV": []}
        
        # Setup LFS Tracker
        track_features = {} 
        
        # Initialize Frame 0 Tracking
        tracked_centroids = [{}] * self.T
        tracked_centroids[0] = centroids_raw_list[0]
        
        for lbl in tracked_centroids[0]:
            mask = labels_list[0] == lbl
            track_features[lbl] = [self.fused_cube[0][mask].mean(axis=0)]

        for t in tqdm(range(self.T - 1), desc="Evaluating Frames"):
            lbl_t, lbl_t1 = labels_list[t], labels_list[t+1]
            flow = self.flow_fields[t]
            fused_t, fused_t1 = self.fused_cube[t], self.fused_cube[t+1]
            
            # Track centroids for time t+1
            tracked_c_t1 = self._track_labels(tracked_centroids[t], 
                                              centroids_raw_list[t+1], flow)
            tracked_centroids[t+1] = tracked_c_t1
            
            # Calculate Frame-to-Frame Metrics
            metrics["FAE"].append(self.calculate_fae(tracked_centroids[t], tracked_c_t1, flow, lbl_t))
            metrics["BDV"].append(self.calculate_bdv(lbl_t, lbl_t1, flow))
            metrics["GED"].append(self.calculate_ged_rate(set(tracked_centroids[t].keys()), set(tracked_c_t1.keys())))
            metrics["MDV"].append(self.calculate_mdv(fused_t, fused_t1, lbl_t1, flow))
            
            # Update LFS history
            for lbl, c_pos in tracked_c_t1.items():
                # Remap physical coordinates back to raw label map for masking
                raw_lbl_matches = [k for k, v in centroids_raw_list[t+1].items() if np.allclose(v, c_pos)]
                if raw_lbl_matches:
                    raw_lbl = raw_lbl_matches[0]
                    mask = (lbl_t1 == raw_lbl)
                    mean_feat = fused_t1[mask].mean(axis=0)
                    if lbl not in track_features:
                        track_features[lbl] = []
                    track_features[lbl].append(mean_feat)

        # Calculate Final Lagrangian Feature Smoothness (LFS)
        lfs_scores = []
        for lbl, history in track_features.items():
            if len(history) > 1:
                hist_arr = np.array(history) # (T_k, C)
                # Mean absolute difference of successive frames
                diffs = np.linalg.norm(hist_arr[1:] - hist_arr[:-1], axis=1)
                lfs_scores.append(diffs.mean())
                
        metrics["LFS"] = np.mean(lfs_scores) if lfs_scores else 0.0

        # Average out temporal metrics
        for key in ["FAE", "BDV", "GED", "MDV"]:
            metrics[key] = np.mean(metrics[key])
            
        return metrics

def run_segmentation(fused_cube, dem_norm, method_type="standard"):
    """
    Runs the selected temporal SLIC method over the fused feature cube.
    Returns the temporal lists of labels, centroids (formatted as dicts), and optical flows.
    """
    T = fused_cube.shape[0]
    labels_list = []
    centroids_list = []
    flows_list = []
    
    if method_type == "standard":
        tslic = TemporalSLIC_DEM(dem_norm, n_segments=1500, compactness=10.0, lambda_z=0.5)
        desc = "  Running Standard Fused T-SLIC"
    else:
        tslic = AdaptiveResidualTSLIC_DEM(dem_norm, n_segments=1500, compactness=10.0, lambda_z=0.5)
        desc = "  Running Adaptive Residual T-SLIC"

    for t in tqdm(range(T), desc=desc):
        # 1. Segment
        if method_type == "standard":
            labels, flow = tslic.segment(fused_cube[t], use_flow=True)
            # Standard method returns a dict of centroids
            cents_dict = tslic.prev_centroids.copy() if tslic.prev_centroids else {}
        else:
            labels, flow, rag = tslic.segment(fused_cube[t], use_flow=True)
            # Adaptive method returns an ndarray of shape (K, 2).
            # Convert to dict {1: [y, x], 2: [y, x], ...} so the evaluator can process it.
            if tslic.prev_centers is not None:
                cents_dict = {i + 1: c for i, c in enumerate(tslic.prev_centers)}
            else:
                cents_dict = {}
                
        # 2. Store
        labels_list.append(labels)
        flows_list.append(flow)
        centroids_list.append(cents_dict)

    return labels_list, centroids_list, flows_list

def main_evaluation_pipeline():
    CATALOG_PATH = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\SEVIR\CATALOG.csv"
    
    if not os.path.exists(CATALOG_PATH):
        print(f"Catalog not found at {CATALOG_PATH}")
        return

    print("Loading catalog...")
    catalog = pd.read_csv(CATALOG_PATH, parse_dates=["time_utc"], low_memory=False)

    while True:
        print("\n" + "=" * 60)
        event_id = input("Enter Event ID to evaluate (or 'q' to quit): ").strip()
        if event_id.lower() == "q":
            break
        if not event_id:
            continue

        # 1. Load Data
        print(f"\nLoading fused feature cube for {event_id}...")
        fused, data, extent = load_fused_channels(event_id, catalog)
        if fused is None:
            print("  [Error] Data not found. Skipping.")
            continue
            
        T = fused.shape[0]
        print(f"  Loaded {T} frames. Fetching DEM...")
        dem_raw = fetch_and_regrid_dem(extent)
        dem_norm = (dem_raw - dem_raw.min()) / (dem_raw.max() - dem_raw.min() + 1e-6)

        # 2. Run Standard Temporal SLIC
        print("\n--- Processing Method 1: Standard Temporal SLIC ---")
        lbl_std, cnt_std, flw_std = run_segmentation(fused, dem_norm, method_type="standard")

        # 3. Run Adaptive Residual Temporal SLIC
        print("\n--- Processing Method 2: Adaptive Residual T-SLIC ---")
        lbl_adp, cnt_adp, flw_adp = run_segmentation(fused, dem_norm, method_type="adaptive")

        # 4. Evaluate Both Methods
        print("\n--- Running Physical & Graph Metrics ---")
        
        # Note: We pass the flow fields into the evaluate method so it has access to the 
        # specific optical flow generated during that run.
        evaluator = SuperpixelEvaluator(fused)
        
        print("  Evaluating Standard Method...")
        results_std = evaluator.evaluate(lbl_std, cnt_std, flw_std)
        
        print("  Evaluating Adaptive Method...")
        results_adp = evaluator.evaluate(lbl_adp, cnt_adp, flw_adp)

        # 5. Print Comparison Table
        print("\n" + "=" * 60)
        print(f" EVALUATION RESULTS: EVENT {event_id}")
        print("=" * 60)
        print(f"{'Metric':<40} | {'Standard':<10} | {'Adaptive':<10}")
        print("-" * 64)
        print(f"{'Flow Adherence Error (FAE) ↓':<40} | {results_std['FAE']:<10.4f} | {results_adp['FAE']:<10.4f}")
        print(f"{'Boundary Displacement Variance (BDV) ↓':<40} | {results_std['BDV']:<10.4f} | {results_adp['BDV']:<10.4f}")
        print(f"{'Graph Edit Distance Rate (GED) ↑':<40} | {results_std['GED']:<10.4f} | {results_adp['GED']:<10.4f}")
        print(f"{'Lagrangian Feature Smoothness (LFS) ↓':<40} | {results_std['LFS']:<10.4f} | {results_adp['LFS']:<10.4f}")
        print(f"{'Material Derivative Variance (MDV) ↓':<40} | {results_std['MDV']:<10.4f} | {results_adp['MDV']:<10.4f}")
        print("=" * 60)
        print("* Arrows indicate whether a lower (↓) or higher (↑) score is physically better.")

if __name__ == "__main__":
    main_evaluation_pipeline()