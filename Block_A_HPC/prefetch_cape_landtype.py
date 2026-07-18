"""
prefetch_offline_caches.py
===========================
Standalone LOGIN-NODE (or any machine with internet) script that performs
the two one-time, whole-CONUS precomputes the main sweep only ever READS
from — it never fetches these itself mid-sweep:

    cape_cache/<DATA_ROOT>/cape_clim_<year>_<month:02d>.npy   (12 files)
        <- build_era5_cape_climatology()  (needs cdsapi + ~/.cdsapirc)

    landtype_cache/<DATA_ROOT>/landtype_<year>.npy            (1 file)
        <- build_modis_landtype_grid()    (needs Planetary Computer STAC libs)

It imports and calls those two functions directly from
block_a_solver_benchmark.py (rather than re-implementing them), so the
cache files this produces are guaranteed identical to what the main
script itself would build, with zero risk of the two drifting apart.

WHEN TO RUN THIS
----------------
Once, BEFORE submitting calibrate.cmd / sweep_array.cmd, on a machine
that actually has outbound internet. Aqua's compute nodes very likely
don't (see calibrate.cmd's header comment) — so in practice this usually
means: run it on your own laptop / workstation / a login node (check
Aqua's policy on running network-heavy jobs on the login node — keep it
short, or use whatever data-transfer node HPCE provides), then rsync the
two output folders to DATA_ROOT on the cluster. If you happen to run this
directly ON the cluster with the same DATA_ROOT the sweep will use, no
rsync is needed at all.

REQUIREMENTS (only on the machine THIS script runs on — sweep nodes don't
need any of this since they only read the finished cache):
    pip install cdsapi xarray netCDF4                       # CAPE
    pip install pystac-client planetary-computer rioxarray   # land-type
    ~/.cdsapirc must exist and be valid:
        https://cds.climate.copernicus.eu/how-to-api

USAGE
-----
    python prefetch_offline_caches.py
    python prefetch_offline_caches.py --skip-cape
    python prefetch_offline_caches.py --skip-landtype
    python prefetch_offline_caches.py --year 2019
    python prefetch_offline_caches.py --data-root /path/to/blockA_data

CAPE fetches month-by-month and skips months whose cache file already
exists (see build_era5_cape_climatology's docstring), so if the CDS API
queue times you out partway through, just re-run the same command — it
resumes rather than restarting.
"""
import argparse
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=str, default=None,
                     help="Override DATA_ROOT from block_a_solver_benchmark.py "
                          "(default: whatever's hardcoded there — set that "
                          "correctly first, or pass it here)")
    ap.add_argument("--year", type=int, default=None,
                     help="Override CAPE_YEAR (default: the script's own "
                          "CAPE_YEAR constant, currently used for both the "
                          "CAPE climatology and the land-type grid)")
    ap.add_argument("--skip-cape", action="store_true",
                     help="Skip the ERA5 CAPE climatology (needs cdsapi + ~/.cdsapirc)")
    ap.add_argument("--skip-landtype", action="store_true",
                     help="Skip the Esri 10m LULC land-type grid (needs STAC libs)")
    ap.add_argument("--nx", type=int, default=384)
    ap.add_argument("--ny", type=int, default=384)
    args = ap.parse_args()

    # Import AFTER argparse so `--help` works even without cdsapi / STAC libs
    # installed, and add this script's own directory to sys.path so it finds
    # block_a_solver_benchmark.py sitting alongside it.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import block_a_solver_benchmark as bench

    data_root = args.data_root or bench.DATA_ROOT
    year      = args.year or bench.CAPE_YEAR
    extent    = bench.CONUS_EXTENT
    cape_dir      = os.path.join(data_root, "cape_cache")
    landtype_dir  = os.path.join(data_root, "landtype_cache")

    print(f"DATA_ROOT      = {data_root}")
    print(f"year           = {year}")
    print(f"extent         = {extent}  (lon_min, lon_max, lat_min, lat_max)")
    print(f"cape_cache     -> {cape_dir}")
    print(f"landtype_cache -> {landtype_dir}")
    print()

    if not args.skip_cape:
        print(f"=== ERA5 CAPE climatology ({year}) ===")
        t0 = time.time()
        try:
            bench.build_era5_cape_climatology(
                year=year, extent=extent, out_dir=cape_dir, nx=args.nx, ny=args.ny)
        except Exception as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            print("(check ~/.cdsapirc and `pip install cdsapi xarray netCDF4`)",
                  file=sys.stderr)
            sys.exit(1)
        n_done = len([f for f in os.listdir(cape_dir) if f.startswith("cape_clim_")])
        print(f"cape_cache: {n_done}/12 monthly files present "
              f"({time.time() - t0:.0f}s)\n")
        if n_done < 12:
            print("WARNING: fewer than 12 monthly files — re-run this same "
                  "command to resume the missing months.\n")
    else:
        print("Skipping CAPE (--skip-cape)\n")

    if not args.skip_landtype:
        print(f"=== Land-type grid ({year}) ===")
        t0 = time.time()
        try:
            bench.build_modis_landtype_grid(
                year=year, extent=extent, out_dir=landtype_dir, nx=args.nx, ny=args.ny)
        except Exception as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            print("(check `pip install pystac-client planetary-computer rioxarray`)",
                  file=sys.stderr)
            sys.exit(1)
        expected = os.path.join(landtype_dir, f"landtype_{year}.npy")
        ok = os.path.exists(expected)
        print(f"landtype_cache: {'OK — ' + expected if ok else 'MISSING file after run!'} "
              f"({time.time() - t0:.0f}s)\n")
    else:
        print("Skipping land-type (--skip-landtype)\n")

    if data_root != bench.DATA_ROOT:
        print("You ran this with a --data-root different from the one "
              "hardcoded in block_a_solver_benchmark.py — remember to point "
              "DATA_ROOT at this same path (or rsync these two folders into "
              "wherever DATA_ROOT already points) before submitting the "
              "cluster jobs.\n")

    print("Done. If this ran somewhere other than the cluster itself, copy "
          "both folders over, e.g.:")
    print(f"  rsync -av {cape_dir}/     aqua:{data_root}/cape_cache/")
    print(f"  rsync -av {landtype_dir}/ aqua:{data_root}/landtype_cache/")


if __name__ == "__main__":
    main()