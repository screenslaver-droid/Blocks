"""
prefetch_dem.py  —  Login-node DEM prefetch for PARAM Rudra N-sweep (OPTIMIZED)
=====================================================================
Run this ONCE on the login node before submitting the diagnostic sweep job.
It fetches and caches DEM tiles for every event bounding box that the sweep
will need, so the compute nodes never need outbound internet access.
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

import cartopy.crs as ccrs

# ── Optional DEM libraries (must be installed on login node) ────────────────
try:
    import pystac_client
    import planetary_computer
    import rioxarray
    from rioxarray.merge import merge_arrays
    HAS_DEM_LIBS = True
except ImportError:
    HAS_DEM_LIBS = False
    print("ERROR: DEM libraries not found.")
    print("Install them with:")
    print("  conda install -c conda-forge rioxarray pystac-client planetary-computer")
    sys.exit(1)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ── SEVIR projection (identical to sweep script) ─────────────────────────────
SEVIR_PROJ = ccrs.LambertConformal(
    central_longitude=-98.0, central_latitude=38.0,
    standard_parallels=(30.0, 60.0),
    globe=ccrs.Globe(semimajor_axis=6370000, semiminor_axis=6370000),
)


# ============================================================================
# Geometry helpers (copied verbatim from sweep script — do NOT change)
# ============================================================================

def _dem_cache_path(dem_cache_dir: str, key: tuple) -> str:
    """Produces the EXACT filename the sweep script expects."""
    os.makedirs(dem_cache_dir, exist_ok=True)
    name = "_".join(f"{v:.4f}" for v in key).replace("-", "m")
    return os.path.join(dem_cache_dir, f"dem_{name}.npy")


def get_sevir_grid(extent, nx=384, ny=384):
    src = ccrs.PlateCarree()
    x0, y0 = SEVIR_PROJ.transform_point(extent[0], extent[2], src)
    x1, y1 = SEVIR_PROJ.transform_point(extent[1], extent[3], src)
    xv, yv = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    grid = src.transform_points(SEVIR_PROJ, xv, yv)
    return grid[..., 0], grid[..., 1]


def _load_and_clip_tile(item, extent, buf=0.1):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            da = rioxarray.open_rasterio(
                item.assets["data"].href, lock=False).squeeze()
            da = da.rio.clip_box(
                minx=extent[0] - buf, miny=extent[2] - buf,
                maxx=extent[1] + buf, maxy=extent[3] + buf)
            return da.coarsen(x=3, y=3, boundary="trim").mean()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise


def fetch_and_regrid_dem(extent, dem_cache_dir: str, pc_catalog, nx=384, ny=384):
    """
    Fetch DEM for a bounding box and save to disk cache.
    Returns (path, status) where status is 'fetched', 'cached', or 'failed'.
    """
    key       = tuple(np.round(extent, 4))
    disk_path = _dem_cache_path(dem_cache_dir, key)

    # Already cached?
    if os.path.exists(disk_path):
        try:
            existing = np.load(disk_path)
            if existing.shape == (ny, nx) and existing.nbytes > 0:
                return disk_path, "cached"
            os.remove(disk_path)
        except Exception:
            pass

    try:
        # Retry logic for STAC API search to handle concurrent connection drops
        items = None
        for attempt in range(3):
            try:
                items = list(pc_catalog.search(
                    collections=["cop-dem-glo-30"],
                    bbox=[extent[0], extent[2], extent[1], extent[3]],
                ).item_collection())
                break
            except Exception:
                if attempt == 2: raise
                time.sleep(2)

        if not items:
            result = np.zeros((ny, nx), dtype=np.float32)
        else:
            datasets = [None] * len(items)
            with ThreadPoolExecutor(max_workers=min(len(items), 4)) as pool:
                fmap = {pool.submit(_load_and_clip_tile, it, extent): i
                        for i, it in enumerate(items)}
                for fut in as_completed(fmap):
                    try:
                        datasets[fmap[fut]] = fut.result()
                    except Exception:
                        pass # Ignore individual tile failures, rely on merge

            datasets = [d for d in datasets if d is not None and d.size > 0]

            if not datasets:
                result = np.zeros((ny, nx), dtype=np.float32)
            else:
                merged = merge_arrays(datasets) if len(datasets) > 1 else datasets[0]
                lons, lats = merged.x.values, merged.y.values
                vals = merged.values
                if lats[0] > lats[-1]:
                    lats, vals = lats[::-1], vals[::-1, :]
                interp = RegularGridInterpolator(
                    (lats, lons), vals.astype(np.float32),
                    method="linear", bounds_error=False, fill_value=0.0)
                tg_lons, tg_lats = get_sevir_grid(extent, nx, ny)
                result = interp(
                    np.column_stack((tg_lats.ravel(), tg_lons.ravel()))
                ).reshape(ny, nx).astype(np.float32)

        np.save(disk_path, result)
        return disk_path, "fetched"

    except Exception as e:
        log.error(f"  DEM fetch failed permanently for {extent}: {e}")
        return disk_path, "failed"


# ============================================================================
# Event selection (mirrors select_from_catalogue in sweep script)
# ============================================================================

def select_events(catalogue_path: str, per_class: int) -> list[str]:
    df = pd.read_csv(catalogue_path)
    if "lifecycle_class" not in df.columns or "id" not in df.columns:
        log.error("Catalogue missing 'id' or 'lifecycle_class'.")
        sys.exit(1)
    out = []
    for cls, grp in df.groupby("lifecycle_class"):
        sample = grp.sample(min(per_class, len(grp)), random_state=42)
        out.extend(sample["id"].tolist())
    log.info(f"  Selected {len(out)} events from "
             f"{df['lifecycle_class'].nunique()} lifecycle classes "
             f"(per_class={per_class}).")
    return out


def get_extents_for_events(event_ids: list[str],
                           catalog_path: str) -> dict[str, list]:
    catalog = pd.read_csv(catalog_path, low_memory=False)
    catalog = catalog.drop_duplicates(subset="id")
    catalog = catalog.set_index("id")
    extents = {}
    missing = []
    for eid in event_ids:
        if eid in catalog.index:
            row = catalog.loc[eid]
            extents[eid] = [
                float(row["llcrnrlon"]), float(row["urcrnrlon"]),
                float(row["llcrnrlat"]), float(row["urcrnrlat"]),
            ]
        else:
            missing.append(eid)
    if missing:
        log.warning(f"  {len(missing)} events not found in CATALOG.csv.")
    return extents


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prefetch DEM tiles for PARAM Rudra N-sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--catalogue", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--dem_cache", required=True)
    parser.add_argument("--per_class", type=int, default=3)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--workers", type=int, default=8, help="Parallel download streams")
    args = parser.parse_args()

    os.makedirs(args.dem_cache, exist_ok=True)

    log.info("=" * 60)
    log.info("PARAM Rudra DEM Prefetch (Optimized)")
    log.info("=" * 60)

    event_ids = select_events(args.catalogue, args.per_class)
    extents = get_extents_for_events(event_ids, args.catalog)

    unique_extents: dict[tuple, list[str]] = {}
    for eid, ext in extents.items():
        key = tuple(np.round(ext, 4))
        unique_extents.setdefault(key, []).append(eid)

    log.info(f"  {len(extents)} events map to {len(unique_extents)} unique DEM tiles.")

    if args.dry_run:
        log.info("DRY RUN — exiting.")
        return

    # 4. Open API Client ONCE for the entire script
    log.info("Opening connection to Planetary Computer STAC API...")
    pc_catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    results = {"fetched": [], "cached": [], "failed": []}
    t0 = time.perf_counter()

    # 5. Parallel Execution of unique extents
    def process_extent(item):
        key, _ = item
        extent = list(key)
        _, status = fetch_and_regrid_dem(extent, args.dem_cache, pc_catalog)
        return key, status

    log.info(f"Starting parallel fetch with {args.workers} workers...")

    # Mute standard logging temporarily so it doesn't break the progress bar UI
    logging.getLogger().setLevel(logging.WARNING)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_extent, item): item for item in unique_extents.items()}
        for fut in tqdm(as_completed(futures), total=len(unique_extents), desc="Fetching DEM tiles", unit="tile"):
            key, status = fut.result()
            results[status].append(key)

    # Restore logging
    logging.getLogger().setLevel(logging.INFO)

    elapsed = time.perf_counter() - t0
    log.info("\n" + "=" * 60)
    log.info(f"DEM prefetch complete in {elapsed:.0f}s")
    log.info(f"  Fetched : {len(results['fetched'])} tiles")
    log.info(f"  Cached  : {len(results['cached'])} tiles (already present)")
    log.info(f"  Failed  : {len(results['failed'])} tiles")
    log.info(f"  Total .npy files in cache: {len(list(Path(args.dem_cache).glob('dem_*.npy')))}")
    log.info("=" * 60)

if __name__ == "__main__":
    main()