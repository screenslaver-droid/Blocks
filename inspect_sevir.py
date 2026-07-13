import h5py
import numpy as np
import os
import pandas as pd

'''
# SEVIR Dataset Inventory and Analysis Tool

This code performs a comprehensive audit of your local SEVIR dataset to identify complete weather events and analyze their data characteristics.

## Main Purpose

The script solves a common problem when working with large datasets: **finding which events have ALL required data channels available locally**, then performing a deep statistical inspection of one complete event.

## How It Works

### Phase 1: Inventory Check (`get_available_events`)

For each data type (vil, ir107, ir069, lght), the function:
1. Scans your local filesystem across multiple year folders (2017-2019)
2. Lists all actual HDF5 files present on disk
3. Cross-references these files against the catalog CSV
4. Returns a set of event IDs that physically exist for that channel

**Why this matters:** The catalog might list thousands of events, but you may have only downloaded a subset. This efficiently identifies what you actually have.

### Phase 2: Find Complete Events

The script computes the **intersection** of all four channels:
```
common_ids = vil_ids ∩ ir107_ids ∩ ir069_ids ∩ lght_ids
```

Only events that have all four data types are considered "complete" and usable for multi-channel analysis or machine learning.

### Phase 3: Deep Statistical Analysis (`analyze_channel`)

For one complete event, the script opens each of the four HDF5 files and performs detailed inspection:

**For Image Data (VIL, IR channels):**
- Shape and data type
- Value range (min/max)
- Statistical moments (mean, standard deviation)
- Sparsity percentage (how much is zeros/missing)

**For Lightning Data:**
- Total number of strikes
- Geographic bounds (latitude/longitude ranges)
- Data format confirmation

The function handles two HDF5 storage formats:
1. **Standard indexed format**: Data stored under channel names ('vil', 'ir107') with a separate 'id' index
2. **Raw event format**: Data stored directly under the event ID key (common for lightning)

## Output Example

```
Events found locally:
  - VIL:   450
  - IR107: 480
  - IR069: 465
  - LGHT:  420
  - COMMON: 380 (Complete datasets)

Inspecting Event: S834603
  Time: 2019-05-20 18:00:00
  
  [VIL] Analysis
    - Shape: (49, 384, 384)
    - Range: [0 - 255]
    - Mean: 12.34
    - Sparsity: 78.5%
  
  [LGHT] Analysis
    - Total Strikes: 1,247
    - Lat Range: 35.2 to 38.9
    - Lon Range: -98.3 to -94.1
```

## Practical Use Case

Before training a machine learning model or performing analysis, you need to know:
- How many complete samples do I have?
- What's the data quality and distribution?
- Are there missing or corrupted files?

This script answers all these questions efficiently without manually checking hundreds of files.

'''

# --- CONFIGURATION ---
# Path to your SEVIR root folder
DATA_ROOT = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\SEVIR"
CATALOG_PATH = os.path.join(DATA_ROOT, "CATALOG.csv")

def get_available_events(catalog, img_type):
    """
    Finds which Event IDs for a specific image type actually exist on your disk.
    Returns a set of Event IDs.
    """
    print(f"Scanning local files for '{img_type}'...")
    
    # 1. Get all file names for this type from catalog
    subset = catalog[catalog['img_type'] == img_type].copy()
    
    # 2. Scan local directories (2017, 2018, 2019)
    present_files = set()
    for year in ['2017', '2018', '2019']:
        # Check path: SEVIR/2018/vil
        path = os.path.join(DATA_ROOT, year, img_type)
        if os.path.exists(path):
            present_files.update(os.listdir(path))

    # 3. Filter catalog to find IDs inside these files
    # The catalog 'file_name' is like 'vil/2018/SEVIR_VIL_...h5'
    # We extract the basename 'SEVIR_VIL_...h5' to match our scan
    subset['short_filename'] = subset['file_name'].apply(os.path.basename)
    
    # Only keep rows where we have the file
    found_rows = subset[subset['short_filename'].isin(present_files)]
    
    unique_ids = set(found_rows['id'].unique())
    print(f"  -> Found {len(unique_ids)} valid IDs locally.")
    return unique_ids

def show_sample_data(name, data):
    """Shows a small sample of the actual data values."""
    print(f"    - Sample Data:")
    
    if data.ndim == 3:
        # For 3D image data, show a small corner of the first and last frame
        print(f"      First Frame [0:3, 0:5]:")
        print(f"        {data[0, 0:3, 0:5]}")
        print(f"      Last Frame [-1, 0:3, 0:5]:")
        print(f"        {data[-1, 0:3, 0:5]}")
        
    elif data.ndim == 2:
        # For 2D data (lightning), show first few rows
        num_samples = min(5, data.shape[0])
        print(f"      First {num_samples} entries:")
        for i in range(num_samples):
            print(f"        Row {i}: {data[i]}")


def analyze_dataset(name, data):
    """Calculates and prints statistics for a dataset."""
    print(f"    - Type:  {data.dtype}")
    print(f"    - Shape: {data.shape}")
    
    # Check dimensions to decide how to analyze
    if data.ndim == 3:
        # IMAGE VIDEO (Time, Height, Width)
        print("    - Format: 3D Video Sequence")
        
        # Convert to float for stats to avoid overflow
        # Reading data into memory for stats
        sample = data[:] 
        
        min_v = np.min(sample)
        max_v = np.max(sample)
        mean_v = np.mean(sample)
        std_v = np.std(sample)
        
        # Sparsity (Crucial for Graph construction)
        zeros = np.sum(sample == 0)
        sparsity = (zeros / sample.size) * 100
        
        print(f"    - Range: [{min_v} to {max_v}]")
        print(f"    - Mean:  {mean_v:.2f} (Std: {std_v:.2f})")
        print(f"    - Sparsity (Zeros): {sparsity:.2f}%")
        
        # Show sample data
        show_sample_data(name, sample)
        
    elif data.ndim == 2:
        # RAW DATA TABLE (Rows, Columns)
        print("    - Format: 2D Raw Table (Lightning/Metadata)")
        print(f"    - Entries: {data.shape[0]}")
        
        # If it looks like lightning (N, 4) or (N, 3)
        if data.shape[1] >= 2:
            # Usually Col 0 is Lat, Col 1 is Lon
            lat_min, lat_max = np.min(data[:,0]), np.max(data[:,0])
            lon_min, lon_max = np.min(data[:,1]), np.max(data[:,1])
            print(f"    - Lat Bounds: {lat_min:.2f} to {lat_max:.2f}")
            print(f"    - Lon Bounds: {lon_min:.2f} to {lon_max:.2f}")
            
            # Show column statistics for all columns
            print(f"    - Column Statistics:")
            for col in range(data.shape[1]):
                col_min, col_max = np.min(data[:, col]), np.max(data[:, col])
                col_mean = np.mean(data[:, col])
                print(f"      Col {col}: Range [{col_min:.2f} to {col_max:.2f}], Mean: {col_mean:.2f}")
        
        # Show sample data
        show_sample_data(name, data)
        
def inspect_channel(event_id, img_type, catalog):
    """Locates the file for an ID and analyzes the data inside."""
    
    # 1. Find the file path using the catalog
    row = catalog[(catalog['id'] == event_id) & (catalog['img_type'] == img_type)].iloc[0]
    filename = os.path.basename(row['file_name'])
    
    # We don't know which year folder it's in, so we check all 3
    filepath = None
    for year in ['2017', '2018', '2019']:
        p = os.path.join(DATA_ROOT, year, img_type, filename)
        if os.path.exists(p):
            filepath = p
            break
            
    if not filepath:
        print(f"  [ERROR] Could not locate file for {img_type} on disk.")
        return

    print(f"\n  [{img_type.upper()}] File: {filename}")
    
    # 2. Open and Analyze
    try:
        with h5py.File(filepath, 'r') as f:
            # Case A: Standard Key (e.g., 'vil')
            if img_type in f:
                # Need to find index of event_id
                if 'id' in f:
                    ids = f['id'][:]
                    if len(ids) > 0 and isinstance(ids[0], bytes):
                        ids = [x.decode('utf-8') for x in ids]
                    
                    if event_id in ids:
                        idx = np.where(np.array(ids) == event_id)[0][0]
                        analyze_dataset(img_type, f[img_type][idx])
                    else:
                        print("    [Error] ID not found in file index.")
            
            # Case B: Event ID Key (Common for Lightning)
            elif event_id in f:
                analyze_dataset(event_id, f[event_id][:])
            
            else:
                print(f"    [Error] Could not find data key. Keys: {list(f.keys())[:5]}")
                
    except Exception as e:
        print(f"    [Error] Reading HDF5: {e}")

def main():
    print(f"--- SEVIR DATA INSPECTOR ---")
    if not os.path.exists(CATALOG_PATH):
        print("Catalog not found. Please download CATALOG.csv first.")
        return

    # 1. Load Catalog
    df = pd.read_csv(CATALOG_PATH, parse_dates=['time_utc'], low_memory=False)
    
    # 2. Find Common Events
    print("\nSTEP 1: Cross-referencing local files with catalog...")
    vil_ids = get_available_events(df, 'vil')
    ir1_ids = get_available_events(df, 'ir107')
    ir2_ids = get_available_events(df, 'ir069')
    lght_ids = get_available_events(df, 'lght')
    
    # Intersection: IDs present in ALL sets
    common_ids = sorted(list(vil_ids & ir1_ids & ir2_ids & lght_ids))
    
    count = len(common_ids)
    print(f"\nSTEP 2: Found {count} COMMON EVENTS (Complete 4-channel sets).")
    
    if count == 0:
        print("No complete sets found. You might be missing Lightning or IR data for your VIL events.")
        return

    # 3. LIST COMMON EVENTS
    print("\n--- LIST OF AVAILABLE COMPLETE EVENTS ---")
    # Print up to 50 IDs, formatted nicely
    for i in range(0, min(count, 500), 5):
        print("  " + ", ".join(common_ids[i:i+5]))
    
    if count > 500:
        print(f"  ... and {count - 50} more.")

    # 4. INSPECT ONE EVENT
    target_id = common_ids[0]
    print(f"\nSTEP 3: Deep Inspection of Event: {target_id}")
    print("-" * 50)
    
    # Metadata
    meta = df[df['id'] == target_id].iloc[0]
    print(f"Time (UTC): {meta['time_utc']}")
    print(f"Location:   {meta['llcrnrlat']:.2f}, {meta['llcrnrlon']:.2f} (Lat, Lon)")
    print(f"Duration:   4 Hours (49 frames)")
    print("-" * 50)
    
    # Analyze Channels
    for t in ['vil', 'ir107', 'ir069', 'lght']:
        inspect_channel(target_id, t, df)

if __name__ == "__main__":
    main()