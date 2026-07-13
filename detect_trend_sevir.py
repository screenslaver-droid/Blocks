import h5py
import numpy as np
import os
import pandas as pd
from scipy.stats import linregress

# --- CONFIGURATION (Match your previous script) ---
DATA_ROOT = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\SEVIR"
CATALOG_PATH = os.path.join(DATA_ROOT, "CATALOG.csv")

# Thresholds for classification
# NOISE_THRESH: Pixel value (0-255) below which we ignore (light rain/clouds)
NOISE_THRESH = 20  
# SLOPE_THRESH: How steep the line must be to count as Growth/Decay
SLOPE_THRESH = 500.0 

def get_vil_data(event_id, catalog):
    """
    Locates and loads the VIL numpy array for a specific event ID.
    Adapted from inspect_sevir.py
    """
    img_type = 'vil'
    
    # 1. Find file path from catalog
    subset = catalog[(catalog['id'] == event_id) & (catalog['img_type'] == img_type)]
    if subset.empty:
        print(f"[Error] Event {event_id} not found in catalog or has no VIL data.")
        return None
        
    row = subset.iloc[0]
    filename = os.path.basename(row['file_name'])
    
    # Check all year folders
    filepath = None
    for year in ['2017', '2018', '2019']:
        p = os.path.join(DATA_ROOT, year, img_type, filename)
        if os.path.exists(p):
            filepath = p
            break
            
    if not filepath:
        print(f"[Error] File {filename} not found locally.")
        return None

    # 2. Extract Data
    try:
        with h5py.File(filepath, 'r') as f:
            # SEVIR VIL is stored with an 'id' index lookup
            if 'vil' in f and 'id' in f:
                # Decode IDs from bytes to strings
                ids = [x.decode('utf-8') for x in f['id'][:]]
                
                if event_id in ids:
                    idx = ids.index(event_id)
                    data = f['vil'][idx] # Shape: (49, 384, 384)
                    return data
                else:
                    print(f"[Error] ID {event_id} exists in catalog but not in file index.")
    except Exception as e:
        print(f"[Error] Failed to read HDF5: {e}")
        
    return None

def analyze_growth_decay(event_id, data):
    """
    Computes the linear trend of the storm intensity.
    """
    # 1. Filter Noise
    # Create a mask where VIL is significant (> NOISE_THRESH)
    # We use uint8 values directly (0-255). 20 is approx low-to-mid intensity rain.
    clean_data = np.where(data > NOISE_THRESH, data, 0)
    
    # 2. Calculate Metric (Sum of VIL Mass per frame)
    # Axis (1,2) sums the height and width, leaving the Time dimension
    time_series = np.sum(clean_data, axis=(1, 2))
    
    # 3. Linear Regression
    # We just want the slope over the 49 frames (4 hours)
    x = np.arange(len(time_series))
    slope, intercept, r_value, p_value, std_err = linregress(x, time_series)
    
    # 4. Determine Label
    if slope > SLOPE_THRESH:
        label = "GROWTH"
        confidence = "High" if slope > (SLOPE_THRESH * 2) else "Moderate"
    elif slope < -SLOPE_THRESH:
        label = "DECAY"
        confidence = "High" if slope < -(SLOPE_THRESH * 2) else "Moderate"
    else:
        label = "STEADY"
        confidence = "N/A"

    return {
        "id": event_id,
        "label": label,
        "slope": slope,
        "r2": r_value**2,
        "confidence": confidence,
        "start_mass": time_series[0],
        "end_mass": time_series[-1]
    }

def main():
    # Load Catalog
    if not os.path.exists(CATALOG_PATH):
        print("Catalog not found.")
        return
    df = pd.read_csv(CATALOG_PATH, low_memory=False)
    
    # --- INPUT YOUR EVENT ID HERE ---
    # 2. Ask for Input
    # This keeps asking until you type 'exit'
    while True:
        target_id = input("\nEnter Event ID (or 'exit' to quit): ").strip()
        
        if target_id.lower() == 'exit':
            break
            
        if not target_id:
            continue

        # --------------------------------
    
        print(f"--- Analyzing Event: {target_id} ---")
        
        # 1. Get Data
        vil_data = get_vil_data(target_id, df)
        
        if vil_data is not None:
            # 2. Detect Trend
            result = analyze_growth_decay(target_id, vil_data)
            
            # 3. Report
            print(f"\nResults:")
            print(f"  Classification:  ** {result['label']} **")
            print(f"  Slope:           {result['slope']:.2f} (pixel_mass / frame)")
            print(f"  R-Squared:       {result['r2']:.4f} (Linearity)")
            print(f"  Intensity Change: {result['start_mass']:.0f} -> {result['end_mass']:.0f}")
            
            # Interpret slope
            if result['label'] == "GROWTH":
                print(f"  -> The storm is intensifying significantly over the 4-hour window.")
            elif result['label'] == "DECAY":
                print(f"  -> The storm is dissipating.")
            else:
                print(f"  -> The storm structure is maintaining relatively constant intensity.")
    

if __name__ == "__main__":
    main()