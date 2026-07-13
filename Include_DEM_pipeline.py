import pystac_client
import planetary_computer
import rioxarray
import numpy as np

def fetch_dynamic_dem(extent, nx=384, ny=384):
    """
    Fetches a 30m DEM crop for a specific bounding box from the Cloud.
    extent: [min_lon, max_lon, min_lat, max_lat]
    """
    # 1. Connect to Microsoft Planetary Computer (Free STAC API)
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )

    # 2. Define the Search Box (Reformat extent to [min_lon, min_lat, max_lon, max_lat])
    # Note: Incoming 'extent' from your code is usually [min_lon, max_lon, min_lat, max_lat]
    # STAC expects [min_lon, min_lat, max_lon, max_lat]
    bbox = [extent[0], extent[2], extent[1], extent[3]]

    # 3. Search for the "Copernicus Global 30m" Dataset
    search = catalog.search(
        collections=["cop-dem-glo-30"],
        bbox=bbox
    )
    
    # 4. Get the tiles that cover this area
    items = search.item_collection()
    print(f"  [DEM] Found {len(items)} tiles covering this event.")

    # 5. Load and Merge the tiles (Stream directly into memory)
    # This might take 2-5 seconds depending on internet
    dem_data = None
    try:
        # Load the data using rioxarray
        # We merge them in case the event sits on the border of two tiles
        datasets = []
        for item in items:
            signed_asset = item.assets["data"]
            ds = rioxarray.open_rasterio(signed_asset.href).squeeze()
            datasets.append(ds)
        
        # Merge if multiple tiles, otherwise just take the first
        if len(datasets) > 1:
            from rioxarray.merge import merge_arrays
            full_dem = merge_arrays(datasets)
        else:
            full_dem = datasets[0]

        # 6. Crop and Reproject to your exact 384x384 grid
        # (This is a simplified reproject; for best results, match your LCC grid exactly)
        # For now, let's return the raw data and let your 'griddata' function interpolate it
        return full_dem

    except Exception as e:
        print(f"  [DEM Error] Could not fetch data: {e}")
        return None

# --- UPDATE YOUR PLOTTING CODE ---
# Inside plot_event_with_dem:
# dem_ds = fetch_dynamic_dem(extent)
# Use griddata to sample 'dem_ds' at your 'grid_lons', 'grid_lats'