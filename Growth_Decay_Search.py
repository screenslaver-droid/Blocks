import h5py
import numpy as np
import os
import pandas as pd
from scipy.stats import linregress
from tqdm import tqdm  # pip install tqdm (optional, for progress bar)

# --- CONFIGURATION ---
DATA_ROOT = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\SEVIR"
CATALOG_PATH = os.path.join(DATA_ROOT, "CATALOG.csv")

# Thresholds
NOISE_THRESH = 20      # Pixel value below which is ignored
GROWTH_THRESH = 1000   # Slope > 1000: Rapid Growth
DECAY_THRESH = -1000   # Slope < -1000: Rapid Decay

def get_local_path(catalog_filename):
    """
    Resolves the catalog's relative path to the local filesystem.
    Input: 'vil/2018/SEVIR_VIL_STORMEVENTS_2018_0101_0630.h5'
    """
    # The catalog path usually includes the year folder (e.g. vil/2018/...)
    # We just need to join it with DATA_ROOT.
    # However, based on your previous script structure, we'll try to be robust.
    
    # Try direct join first (Standard SEVIR structure)
    # DATA_ROOT/vil/2018/filename.h5
    p1 = os.path.join(DATA_ROOT, catalog_filename)
    if os.path.exists(p1):
        return p1
        
    # Try the structure from your inspect_sevir.py (DATA_ROOT/2018/vil/filename.h5)
    parts = catalog_filename.split('/') # [vil, 2018, filename]
    if len(parts) == 3:
        p2 = os.path.join(DATA_ROOT, parts[1], parts[0], parts[2])
        if os.path.exists(p2):
            return p2
            
    return None

def calculate_slope(vil_frames):
    """Computes linear regression slope of VIL mass over time."""
    # 1. Filter Noise
    mask = vil_frames > NOISE_THRESH
    
    # 2. Sum Mass per Frame (Time, H, W) -> (Time,)
    # We use valid pixels only
    mass_series = np.sum(np.where(mask, vil_frames, 0), axis=(1, 2))
    
    # 3. Linear Regression
    x = np.arange(len(mass_series))
    slope, _, _, _, _ = linregress(x, mass_series)
    return slope

def main():
    if not os.path.exists(CATALOG_PATH):
        print("Catalog not found.")
        return

    print("--- SEVIR BULK EVENT SEARCH ---")
    print(f"1. Loading Catalog from {CATALOG_PATH}...")
    df = pd.read_csv(CATALOG_PATH, low_memory=False)
    
    # Filter for VIL only
    vil_df = df[df['img_type'] == 'vil'].copy()
    
    # Group by File Name to minimize disk I/O
    grouped = vil_df.groupby('file_name')
    
    significant_growth = []
    significant_decay = []
    
    print(f"2. Scanning {len(grouped)} files for significant events...")
    print(f"   (Thresholds: Growth > {GROWTH_THRESH}, Decay < {DECAY_THRESH})\n")

    # Iterate through each physical file
    for relative_path, group in tqdm(grouped, total=len(grouped), unit="file"):
        
        full_path = get_local_path(relative_path)
        if not full_path:
            continue
            
        target_ids = set(group['id'].values)
        
        try:
            with h5py.File(full_path, 'r') as f:
                # Load all IDs in this file once
                if 'id' not in f or 'vil' not in f:
                    continue
                    
                file_ids = f['id'][:]
                # Decode bytes to string if necessary
                file_ids = [x.decode('utf-8') if isinstance(x, bytes) else x for x in file_ids]
                
                # Find indices of the events we want
                for i, event_id in enumerate(file_ids):
                    if event_id in target_ids:
                        # Load VIL data for this specific event
                        data = f['vil'][i] 
                        
                        # Compute Slope
                        slope = calculate_slope(data)
                        
                        if slope > GROWTH_THRESH:
                            significant_growth.append((event_id, slope))
                        elif slope < DECAY_THRESH:
                            significant_decay.append((event_id, slope))
                            
        except Exception as e:
            print(f"Error reading {relative_path}: {e}")

    # --- OUTPUT RESULTS ---
    print("\n" + "="*40)
    print(f"SCAN COMPLETE")
    print("="*40)
    
    # Sort by intensity of change (highest slope first)
    significant_growth.sort(key=lambda x: x[1], reverse=True)
    significant_decay.sort(key=lambda x: x[1]) # Lowest negative first

    print(f"\n[TOP 10 GROWTH EVENTS] (Total found: {len(significant_growth)})")
    print(f"{'Event ID':<15} | {'Slope':<10}")
    print("-" * 28)
    for eid, slope in significant_growth[:10]:
        print(f"{eid:<15} | {slope:.2f}")

    print(f"\n[TOP 10 DECAY EVENTS] (Total found: {len(significant_decay)})")
    print(f"{'Event ID':<15} | {'Slope':<10}")
    print("-" * 28)
    for eid, slope in significant_decay[:10]:
        print(f"{eid:<15} | {slope:.2f}")

    # Optional: Save to CSV
    save_q = input("\nSave full lists to CSV? (y/n): ")
    if save_q.lower() == 'y':
        res_df = pd.DataFrame(significant_growth + significant_decay, columns=['id', 'slope'])
        res_df['type'] = res_df['slope'].apply(lambda x: 'Growth' if x > 0 else 'Decay')
        res_df.to_csv("significant_events.csv", index=False)
        print("Saved to significant_events.csv")

if __name__ == "__main__":
    main()