"""
block_a_solver_benchmark.py
============================
Block A: Empirical solver benchmark -- IMEX Strang-split (Tsit5 + Kvaerno5)
vs. DOPRI5 (unsplit) on the Lagrangian superpixel graph built from SEVIR
events, following the "Block A -- Solver Feasibility Benchmark" section of
blocks.tex (Sections A.1-A.8), and the "Preliminary: SEVIR Data Encoding and
Normalisation" section that both Block A and Block B must share.

This version differs from the original benchmark script in the following
spec-driven ways (see inline comments tagged with the relevant section):

  * Preliminary   -- Per-event, per-channel MIN-MAX normalisation
                      (arr - arr.min()) / (arr.max() - arr.min() + 1e-6),
                      identical to load_fused_channels. Replaces the old
                      VIL/255 + IR-max-only normalisation.
  * Preliminary   -- Channel order fixed to (VIL, IR107, IR069).
  * A.5 step 3    -- 4th input channel added: normalised, lambda_z-weighted
                      DEM, concatenated for the SLIC feature cube.
  * A.5 step 4    -- build_graph_topological() now EXPLICITLY instantiates
                      TemporalSLIC_DEM (ported verbatim, see its class
                      docstring, from Visualize_sevir_with_superpixel_fused.py)
                      and calls its .segment() method, rather than
                      reimplementing an equivalent skimage.slic() call by
                      hand. A.5 requires this exact class specifically so
                      that Block A's graphs correspond 1:1 to the ones the
                      Chapter 5 N-sweep's quality metrics (and therefore N*
                      per class) were computed from -- a parameter-matched
                      but independent call risks silent drift from the
                      real pipeline the next time either one changes.
                      dem_norm (raw, [0,1]) is now passed through
                      end-to-end instead of a pre-weighted dem_weighted,
                      since the class applies lambda_z to its own 4th
                      feature-cube channel internally, exactly as it does
                      in the source script.
  * A.5 step 6    -- Middle-level edges are TOPOLOGICAL adjacency of the
                      t=0 SLIC label map (4-connected boundary scan), NOT
                      the pre-allocated radius superset (old R_MAX_PX/
                      cdist logic has been removed).
  * A.2 / A.5 s.7 -- sigma_rbf(N) = c_sigma * sqrt(H*W/N) (N-dependent
                      bandwidth), with c_sigma fixed once via the A.6
                      calibration procedure -- replaces the old fixed
                      R_MAX_PX/SIGMA_COEFF bandwidth.
  * A.2           -- Diffusion term is D * sum_j ew_ij (h_j - h_i) with a
                      SCALAR, once-calibrated D (Section A.2's CFL-based
                      calibration) -- the old learned W_flux matrix has
                      been removed, since Block A's explicit term has no
                      learned parameters.
  * A.2           -- Reaction MLPs (src/snk) use SiLU activations (not
                      tanh), and sigma_init^c is calibrated per class via
                      the per-node Jacobian spectral-radius procedure
                      given in A.2's pseudocode.
  * A.4           -- NFE is computed as accepted_steps x known stage count
                      (Tsit5=5, Kvaerno5=6, Dopri5=6), not raw diffrax
                      num_steps. Rejected-step counts are logged separately.
  * A.5           -- lambda_max(L) is logged per (event, N) for Figure A-4,
                      both under unit weights (lambda_max_topo) and under
                      the calibrated D * RBF weights (lambda_max_weighted).
  * A.6           -- N sweep, events-per-class, and seed count restored to
                      the spec values: N in {250,500,750,1000,1150,1150,1500},
                      5 events/class x 7 classes, 5 seeds.
  * A.8           -- Summary now reports reaction_fraction / diffusion_
                      fraction directly, for the Joint Score weight-ordering
                      validation.

KNOWN GAPS relative to the spec (flagged rather than silently guessed):
  * TARGET_RHO below are ORDER-OF-MAGNITUDE placeholders derived from the
    SRbase comments in the original script's STIFFNESS_SCALE table, since
    Table 5.6 of the thesis was not available in this context. Replace with
    the exact SRbase(c)/Delta_t_macro values before publishing Block A
    figures.
  * load_dem_from_cache() / prefetch_dem_for_events() now source real
    Copernicus DEM GLO-30 tiles via the Planetary Computer STAC API (same
    pipeline as visualize_hierarchical_graph.py's fetch_and_regrid_dem),
    fronted by a disk cache KEYED BY (ROUNDED) EXTENT rather than by
    event_id -- ported from the standalone login-node prefetch_dem.py
    script's unique_extents / _dem_cache_path logic, so events that share
    a bounding box hit one cached tile instead of each storing a redundant
    copy, and a DEM_CACHE_DIR warmed by running prefetch_dem.py directly
    is picked up here with no extra work. STAC search and per-tile reads
    also now retry up to 3x with a short backoff (also ported from
    prefetch_dem.py) before giving up. Requires
    cartopy/pystac_client/planetary_computer/rioxarray; falls back to an
    all-zero DEM (with a warning) if those aren't installed or the fetch
    fails for a given event/extent.
  * nfe_reaction uses the nominal Kvaerno5 stage count (accepted_steps x 6),
    not the true Newton-iteration count. A.4 asks for
    "total_implicit_evals_reaction (includes Newton iterations)", which
    requires enabling diffrax's nonlinear-solver stats collection
    (e.g. a custom NewtonNonlinearSolver with stat tracking) -- not exposed
    by the default sol.stats dict. Extend estimate_reaction_nfe() if the
    exact count is required for the paper.
  * The A.3 tolerance-selection sweep and c_sigma/D calibration are
    genuine numerical procedures (not hand-picked constants) but run on
    whatever sample event(s) are available locally; on the real 851-event
    catalogue / HPC run, re-run calibrate_c_sigma / calibrate_D with the
    designated STEADY-class calibration event from Section A.6.

Output: block_a_results.csv        (one row per event x N x seed)
        block_a_summary.csv        (per-class x N mean +/- std, for figures)
        block_a_calibration.json   (c_sigma, D, r_stab, per-class sigma_init)

Dependencies
------------
  pip install jax[cpu] diffrax equinox
  pip install scikit-image scipy tqdm h5py pandas numpy opencv-python
  (For GPU: pip install jax[cuda12] instead of jax[cpu])

Usage
-----
  python block_a_solver_benchmark.py
  python block_a_solver_benchmark.py --n_events 5 --seeds 2   # quick test
  python block_a_solver_benchmark.py --calibrate-tolerance    # also run A.3
"""

from __future__ import annotations
import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Optional

import cv2
import h5py
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse.linalg import eigsh
from skimage.segmentation import slic as skimage_slic
from skimage.segmentation import watershed
from skimage.filters import sobel
from scipy.ndimage import center_of_mass
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  — adjust DATA_ROOT / DEM_CACHE_DIR to match your environment
# ─────────────────────────────────────────────────────────────────────────────
DATA_ROOT      = r"/media/sid_nair/OS/Users/Siddharth Nair/BTP-2/GAT-ODE/SEVIR"
CATALOGUE_PATH = os.path.join(DATA_ROOT, "event_catalogue.csv")
OUTPUT_CSV     = os.path.join(DATA_ROOT, "block_a_results.csv")
SUMMARY_CSV    = os.path.join(DATA_ROOT, "block_a_summary.csv")
CALIB_JSON     = os.path.join(DATA_ROOT, "block_a_calibration.json")
DEM_CACHE_DIR  = os.path.join(DATA_ROOT, "dem_cache")   # see load_dem_from_cache()

# N sweep — A.6: must include 500 (ISV elbow) and 1150 (global N*)
N_VALUES           = [250, 500, 750, 1000, 1150, 1500]
N_EVENTS_PER_CLASS = 5      # A.6: 5 events/class x 7 classes = 35 events total
N_SEEDS            = 5      # A.6: 5 random weight seeds per (event, N)
N_STAR             = 1150   # global N* (calibration anchor for c_sigma and D)

LAMBDA_Z = 0.5   # Preliminary: DEM weighting, dem_weighted = dem_norm * lambda_z

# ODE integration
DT_MACRO      = 300.0   # 5-minute macro step [seconds]
N_MACRO_STEPS = 12      # 12 x 5 min = 60 min forecast horizon
NODE_DIM      = 3       # [VIL, IR107, IR069]  — Preliminary channel order
ENV_DIM       = 5       # [x/W, y/H, VIL, IR107, DEM]  — A.5 step 8
HIDDEN_DIM    = 64      # reaction MLP hidden size

RTOL_DEFAULT = 1e-2   # blocks.tex A.3, line 222: "start at rtol=1e-2" — this
ATOL_DEFAULT = 1e-3   # is the SPEC'S OWN prescribed starting/loosest point for
# the tolerance-selection halving search, not an arbitrary default. It also
# matches the "training tolerance" convention used elsewhere in blocks.tex
# (Section B3, rtol=1e-2/atol=1e-3) for the same reaction architecture.
# Previously this was hardcoded to 1e-3/1e-4 (TIGHTER than the spec's own
# starting point) and was never actually derived via calibrate_tolerance()
# by default — see calibrate_tolerance()'s docstring for why that silently
# starved Kvaerno5 of step-size room on genuinely stiff classes.
MAX_STEPS_EXPLICIT = 4000
MAX_STEPS_IMPLICIT = 4000
MAX_STEPS_DOPRI5   = 50_000  # high ceiling — may be hit for active classes

# A.4: known RK/ESDIRK stage counts used to convert accepted steps -> NFE
STAGE_COUNT_TSIT5    = 5   # 5-stage, FSAL
STAGE_COUNT_KVAERNO5 = 6   # 6-stage ESDIRK (nominal; see module docstring)
STAGE_COUNT_DOPRI5   = 6   # 7-stage, FSAL -> 6 new evals/step

# A.6: c_sigma sweep candidates for the once-only calibration
C_SIGMA_CANDIDATES = [0.5, 1.0, 1.5, 2.0, 3.0]

# A.2: CFL target — diffusion-only step size at N* should be ~= this
# fraction of the macro step.
DT_FRAC_CFL = 0.1

# A.2: rho_target^c = SRbase(c) / Delta_t_macro.
# TODO(Siddharth): replace with the exact SRbase(c) values from Table 5.6
# of the thesis. These are order-of-magnitude placeholders derived from the
# SRbase comments in the original STIFFNESS_SCALE table (SRbase / 300 s).
TARGET_RHO: dict[str, float] = {
    "RAPID_GROWTH": 33.3,   # SRbase ~ 1e4
    "GROWTH_DECAY": 10.0,   # SRbase ~ 3e3 (1e3-1e4)
    "EPISODIC":      3.33,  # SRbase ~ 1e3
    "PLATEAU":       0.333, # SRbase ~ 1e2
    "RAPID_DECAY":   1.0,   # SRbase ~ 3e2 (1e2-1e3)
    "STEADY":        0.333, # SRbase ~ 1e2
    "QUIESCENT":     0.1,   # SRbase ~ 3e1 (1e1-1e2)
}

# ─────────────────────────────────────────────────────────────────────────────
# LAZY JAX IMPORT  (keeps startup fast; fails clearly if JAX not installed)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import jax
    import jax.numpy as jnp
    import diffrax
    JAX_OK = True
except ImportError as e:
    log.error(f"JAX / Diffrax not found: {e}")
    log.error("Install: pip install jax[cpu] diffrax")
    JAX_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# LAZY DEM-PIPELINE IMPORT  (Copernicus DEM GLO-30 via Planetary Computer,
# matching the fetch_and_regrid_dem pipeline in visualize_hierarchical_graph.py)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import cartopy.crs as ccrs
    import pystac_client
    import planetary_computer
    import rioxarray
    from rioxarray.merge import merge_arrays
    from scipy.interpolate import RegularGridInterpolator

    # Bottleneck/hang fix: GDAL's VSICURL layer has NO timeout by default,
    # so a single stalled tile request can block a worker thread forever
    # while other threads pile up behind it. These must be set before any
    # rasterio/GDAL HTTP read happens.
    os.environ.setdefault("GDAL_HTTP_TIMEOUT", "30")           # seconds per request
    os.environ.setdefault("GDAL_HTTP_CONNECTTIMEOUT", "10")    # seconds to connect
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "2")
    os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "1")
    os.environ.setdefault("CPL_VSIL_CURL_CACHE_SIZE", "16000000")  # cap curl cache (~16MB)
    os.environ.setdefault("GDAL_CACHEMAX", "256")               # MB, cap GDAL block cache

    SEVIR_PROJ = ccrs.LambertConformal(
        central_longitude=-98.0, central_latitude=38.0,
        standard_parallels=(30.0, 60.0),
        globe=ccrs.Globe(semimajor_axis=6370000, semiminor_axis=6370000),
    )

    # Bottleneck fix: a SINGLE, size-capped pool for all tile-level fetches,
    # shared across every event. Previously each call to
    # fetch_and_regrid_dem() created its OWN ThreadPoolExecutor(max_workers
    # =min(len(items), 8)); once events themselves were also fetched
    # concurrently (prefetch_dem_for_events), the two pools nested and
    # multiplied — e.g. 6 concurrent events x 8 tiles/event = up to 48
    # simultaneous full-resolution GeoTIFF reads in flight, each holding
    # its own GDAL memory buffers. That's what was blowing up memory /
    # getting the process OOM-killed. Routing ALL tile fetches (from any
    # number of events, fetched concurrently or not) through one bounded
    # pool caps total concurrent network+memory usage regardless of how
    # many events are being prefetched at once.
    _TILE_FETCH_MAX_WORKERS = 4   # conservative default; raise once you've
                                  # confirmed memory headroom for your machine
    _tile_fetch_pool = ThreadPoolExecutor(max_workers=_TILE_FETCH_MAX_WORKERS)

    HAS_DEM_LIBS = True
except ImportError as e:
    HAS_DEM_LIBS = False
    log.warning(f"DEM/Cartopy libraries not found ({e}) — DEM channel will be all-zeros "
                f"unless a cached tile already exists under DEM_CACHE_DIR.")


# ─────────────────────────────────────────────────────────────────────────────
# SEVIR DATA LOADING  (mirrors Growth_Decay_Classify.py conventions)
# ─────────────────────────────────────────────────────────────────────────────

def get_local_path(catalog_filename: str) -> Optional[str]:
    """Resolve a catalog-relative filename to an absolute path."""
    p1 = os.path.join(DATA_ROOT, catalog_filename)
    if os.path.exists(p1):
        return p1
    parts = catalog_filename.replace("\\", "/").split("/")
    if len(parts) == 3:
        p2 = os.path.join(DATA_ROOT, parts[1], parts[0], parts[2])
        if os.path.exists(p2):
            return p2
    return None


def _read_channel_from_hdf5(file_path: str, img_type: str,
                              event_id: str) -> Optional[np.ndarray]:
    """
    Open one HDF5 file and extract the (T, H, W) array for event_id.
    Returns float32 raw values, or None if the event is absent.
    """
    try:
        with h5py.File(file_path, "r") as f:
            if img_type not in f or "id" not in f:
                return None
            file_ids = [
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in f["id"][:]
            ]
            if event_id not in file_ids:
                return None
            idx  = file_ids.index(event_id)
            data = f[img_type][idx].astype(np.float32)
            # Normalise axis order to (T, H, W)
            if data.ndim == 3 and data.shape[2] < data.shape[0]:
                data = data.transpose(2, 0, 1)
            return data
    except Exception as exc:
        log.debug(f"HDF5 read error [{img_type}] {file_path}: {exc}")
        return None


def _normalise_channel_minmax(data: np.ndarray) -> np.ndarray:
    """
    Preliminary section: per-event, per-channel GLOBAL min-max scaling to
    [0, 1], where min/max are taken over ALL pixels and ALL T frames of the
    event — identical to load_fused_channels:

        arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)

    This replaces the old asymmetric handling (VIL/255, IR clip-then-
    max-only) so that Blocks A and B are built on identical graphs to the
    N-sweep. VIL_min/VIL_max are NOT stored here since the Preliminary
    section states physical-unit recovery is not needed in Block A.
    """
    arr_min = float(data.min())
    arr_max = float(data.max())
    return (data - arr_min) / (arr_max - arr_min + 1e-6)


def load_event_multichannel(
        event_id: str,
        catalog_df: pd.DataFrame,
) -> Optional[dict[str, np.ndarray]]:
    """
    Load VIL, IR069, IR107 for a single event from the SEVIR HDF5 files.

    Returns a dict {img_type: (T, H, W) float32 in [0,1]} for all three
    channels, or None if VIL is unavailable.

    If IR069 / IR107 are missing but VIL is present, fills with zeros
    so the benchmark can still run (VIL drives stiffness, IR channels
    contribute to node state richness).
    """
    T_ref, H_ref, W_ref = None, None, None
    channels: dict[str, np.ndarray] = {}

    for img_type in ["vil", "ir069", "ir107"]:
        rows = catalog_df[
            (catalog_df["id"] == event_id) &
            (catalog_df["img_type"] == img_type)
        ]
        if rows.empty:
            log.debug(f"  {event_id}: no catalog row for {img_type}")
            channels[img_type] = None
            continue

        file_path = get_local_path(rows.iloc[0]["file_name"])
        if file_path is None:
            channels[img_type] = None
            continue

        raw = _read_channel_from_hdf5(file_path, img_type, event_id)
        if raw is None:
            channels[img_type] = None
            continue

        # Preliminary: identical per-event, per-channel min-max normalisation
        # for VIL, IR107, and IR069 — no physical decoding in Block A.
        channels[img_type] = _normalise_channel_minmax(raw)
        if T_ref is None:
            T_ref, H_ref, W_ref = channels[img_type].shape

    # VIL is mandatory
    if channels.get("vil") is None:
        return None

    T_ref, H_ref, W_ref = channels["vil"].shape

    # Fill missing IR channels with zeros (preserves graph shape)
    for k in ["ir069", "ir107"]:
        if channels[k] is None:
            log.debug(f"  {event_id}: {k} missing, using zeros")
            channels[k] = np.zeros((T_ref, H_ref, W_ref), dtype=np.float32)

    return channels


# ─────────────────────────────────────────────────────────────────────────────
# DEM LOADING  (Preliminary / A.5 step 3 — 4th SLIC feature-cube channel)
#
# Sourced from the real DEM pipeline in visualize_hierarchical_graph.py:
# Copernicus DEM GLO-30 tiles fetched from Microsoft's Planetary Computer
# STAC catalog, reprojected onto the SEVIR Lambert-Conformal grid, and
# min-max normalised.
#
# Disk cache keying (ported from prefetch_dem.py): the on-disk cache is
# keyed by the ROUNDED EXTENT (dem_<lon0>_<lon1>_<lat0>_<lat1>.npy), not by
# event_id. Several sampled events (e.g. different lifecycle-class draws
# that happen to reuse the same SEVIR patch, or repeated events across N
# values / seeds) can share an IDENTICAL bounding box, so keying by extent
# means they hit the exact same cached tile instead of each fetching and
# storing their own redundant copy under a different filename. This also
# means a cache warmed by running prefetch_dem.py directly on the login
# node is picked up here with no extra work.
# ─────────────────────────────────────────────────────────────────────────────
_dem_cache: dict = {}   # in-memory cache, keyed by rounded extent tuple


def _dem_extent_cache_path(dem_cache_dir: str, key: tuple) -> str:
    """
    Disk cache filename for a (rounded) extent tuple, matching the naming
    scheme used by prefetch_dem.py's _dem_cache_path() EXACTLY, so this
    script and the standalone login-node prefetch script always agree on
    where a given extent's tile lives on disk.
    """
    os.makedirs(dem_cache_dir, exist_ok=True)
    name = "_".join(f"{v:.4f}" for v in key).replace("-", "m")
    return os.path.join(dem_cache_dir, f"dem_{name}.npy")


def get_sevir_grid(extent, nx: int = 384, ny: int = 384):
    """Target (lon, lat) grid for an event's SEVIR Lambert-Conformal extent."""
    proj = SEVIR_PROJ
    src  = ccrs.PlateCarree()
    x0, y0 = proj.transform_point(extent[0], extent[2], src)
    x1, y1 = proj.transform_point(extent[1], extent[3], src)
    xv, yv = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    grid   = src.transform_points(proj, xv, yv)
    return grid[..., 0], grid[..., 1]


def _load_and_clip_tile(item, extent, buf: float = 0.1, max_retries: int = 3):
    """
    Retry logic ported from prefetch_dem.py's _load_and_clip_tile: a single
    tile read occasionally drops (transient connection reset / VSICURL
    hiccup) even with the GDAL_HTTP_* timeout/retry env vars set, so wrap
    the whole open+clip+coarsen in its own retry loop with a short backoff
    before giving up on this tile (the caller already tolerates individual
    tile failures via n_failed / the batch-fold logic below).
    """
    for attempt in range(max_retries):
        try:
            da = rioxarray.open_rasterio(item.assets["data"].href, lock=False).squeeze()
            da = da.rio.clip_box(minx=extent[0] - buf, miny=extent[2] - buf,
                                  maxx=extent[1] + buf, maxy=extent[3] + buf)
            da = da.coarsen(x=3, y=3, boundary="trim").mean().load()
            return da
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise


# Per-tile network timeout — belt-and-suspenders on top of the GDAL_HTTP_*
# env vars above, in case a request wedges at a layer GDAL's own timeout
# doesn't cover (e.g. STAC catalog search, DNS, or a hung thread).
_TILE_FETCH_TIMEOUT_SEC = 45


_STAC_SEARCH_TIMEOUT_SEC = 30
_stac_search_pool = ThreadPoolExecutor(max_workers=4) if HAS_DEM_LIBS else None


def _stac_search_items(extent, max_retries: int = 3):
    """STAC catalog search wrapped with a hard timeout. GDAL_HTTP_TIMEOUT
    (set above) only covers GDAL/rasterio's own HTTP reads — pystac_client's
    catalog.search() is a plain `requests` call underneath and can hang
    independently of those env vars if the STAC API itself stalls.

    Retry loop ported from prefetch_dem.py's search-retry logic, to absorb
    the concurrent-connection drops that same script was written to guard
    against when many events' extents are searched at once."""
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    for attempt in range(max_retries):
        try:
            return list(catalog.search(
                collections=["cop-dem-glo-30"],
                bbox=[extent[0], extent[2], extent[1], extent[3]],
            ).item_collection())
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2)


def fetch_and_regrid_dem(extent, nx: int = 384, ny: int = 384) -> np.ndarray:
    """
    Fetch Copernicus DEM GLO-30 tiles overlapping `extent`
    (=[llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat]) from the Planetary
    Computer STAC API, merge, and regrid onto the SEVIR projection grid.
    Returns RAW elevation in metres (NOT normalised) — callers normalise.

    NOTE: a single SEVIR event's extent (~700+ km per side) typically
    overlaps MANY 1-degree GLO-30 tiles (often 30-50+), not just one or
    two — this is the dominant cost of a cache-miss fetch, not per-tile
    overhead. Tile fetches for THIS event are parallelised via the shared,
    size-capped _tile_fetch_pool (see module import block) rather than a
    fresh per-call pool, so concurrent fetches across MULTIPLE events
    (see prefetch_dem_for_events) don't multiply into an unbounded number
    of simultaneous full-resolution GeoTIFF reads.
    """
    if not HAS_DEM_LIBS:
        return np.zeros((ny, nx), dtype=np.float32)

    cache_key = tuple(np.round(extent, 4))
    if cache_key in _dem_cache:
        return _dem_cache[cache_key]

    # Sanity check: a normal SEVIR event extent should span a few degrees
    # at most. If this fires, either the catalog's llcrnrlon/urcrnrlon/
    # llcrnrlat/urcrnrlat are NOT in degrees (e.g. projected metres leaking
    # through), or something upstream computed a bad bbox — either way the
    # STAC search below could return hundreds/thousands of tiles instead
    # of the expected few dozen, which is a much more direct route to an
    # OOM kill than fetch concurrency ever was.
    lon_span = abs(extent[1] - extent[0])
    lat_span = abs(extent[3] - extent[2])
    if lon_span > 15 or lat_span > 15:
        log.warning(
            f"    Event extent spans {lon_span:.2f}deg lon x {lat_span:.2f}deg lat "
            f"— abnormally large for a SEVIR patch. This will likely fetch a very "
            f"large number of DEM tiles. Double check llcrnrlon/urcrnrlon/"
            f"llcrnrlat/urcrnrlat in CATALOG.csv are in degrees, not projected metres."
        )

    log.info("    Fetching Cop-DEM-GLO-30 via Planetary Computer …")
    try:
        items = _stac_search_pool.submit(_stac_search_items, extent) \
                                  .result(timeout=_STAC_SEARCH_TIMEOUT_SEC)

        if not items:
            log.warning("    No DEM tiles found for this extent — using zeros.")
            return np.zeros((ny, nx), dtype=np.float32)

        log.info(f"    {len(items)} DEM tile(s) overlap this event's extent "
                  f"(pool cap: {_TILE_FETCH_MAX_WORKERS} concurrent, shared "
                  f"across all events being fetched right now).")
        if len(items) > 80:
            log.warning(
                f"    {len(items)} tiles is a LOT for one event — this alone can "
                f"exhaust memory even with batched merging. Check the extent "
                f"sanity warning above."
            )

        # Merge tiles INCREMENTALLY in small batches as they complete,
        # instead of collecting every tile into a list and merging once at
        # the end. Concurrency capping (_tile_fetch_pool) only bounds how
        # many tiles are being FETCHED at once — it does nothing to bound
        # how many completed tiles sit in memory waiting for the others,
        # which is what was actually driving peak memory up with events
        # that overlap dozens of tiles. This caps retained tile memory to
        # roughly one batch + one running merged accumulator at any time.
        fmap = {_tile_fetch_pool.submit(_load_and_clip_tile, it, extent): i
                for i, it in enumerate(items)}

        running_merged = None
        batch: list = []
        BATCH_SIZE = _TILE_FETCH_MAX_WORKERS
        n_failed = 0

        def _fold_batch():
            nonlocal running_merged, batch
            if not batch:
                return
            to_merge = ([running_merged] if running_merged is not None else []) + batch
            running_merged = merge_arrays(to_merge) if len(to_merge) > 1 else to_merge[0]
            batch = []

        for fut in as_completed(fmap):
            i = fmap[fut]
            try:
                da = fut.result(timeout=_TILE_FETCH_TIMEOUT_SEC)
                if da is not None and da.size > 0:
                    batch.append(da)
                if len(batch) >= BATCH_SIZE:
                    _fold_batch()
            except Exception as e:
                n_failed += 1
                log.warning(f"    DEM tile {i} failed/timed out: {e}")
        _fold_batch()

        if n_failed:
            log.warning(f"    {n_failed}/{len(items)} DEM tiles failed or timed out.")

        if running_merged is None:
            return np.zeros((ny, nx), dtype=np.float32)

        merged = running_merged
        lons, lats, vals = merged.x.values, merged.y.values, merged.values.copy()

        # Free the (potentially large) tile arrays as soon as we've pulled
        # out the plain numpy values we actually need.
        del merged, running_merged, batch

        if lats[0] > lats[-1]:
            lats = lats[::-1]
            vals = vals[::-1, :]

        interp = RegularGridInterpolator(
            (lats, lons), vals.astype(np.float32),
            method="linear", bounds_error=False, fill_value=0.0,
        )
        del vals
        tgt_lons, tgt_lats = get_sevir_grid(extent, nx, ny)
        result = interp(
            np.column_stack((tgt_lats.ravel(), tgt_lons.ravel()))
        ).reshape(ny, nx).astype(np.float32)
        _dem_cache[cache_key] = result
        return result
    except Exception as exc:
        log.warning(f"    DEM fetch failed ({exc}) — using zeros.")
        return np.zeros((ny, nx), dtype=np.float32)


def get_event_extent(event_id: str, sevir_catalog: pd.DataFrame) -> Optional[list]:
    """[llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat] for an event, from the
    SEVIR catalog (same fields used in load_channels)."""
    rows = sevir_catalog[sevir_catalog["id"] == event_id]
    if rows.empty:
        return None
    er = rows.iloc[0]
    try:
        return [float(er["llcrnrlon"]), float(er["urcrnrlon"]),
                float(er["llcrnrlat"]), float(er["urcrnrlat"])]
    except KeyError:
        return None


def _get_or_fetch_raw_dem(extent, nx: int = 384, ny: int = 384) -> np.ndarray:
    """
    Return the RAW (un-normalised, metres) DEM tile for `extent`, at a
    fixed canonical resolution, using the on-disk EXTENT-keyed cache
    (_dem_extent_cache_path — matches prefetch_dem.py's naming) in front of
    fetch_and_regrid_dem(). Fetching at one canonical (ny, nx) regardless of
    the calling event's actual (H, W) is what lets a single cached tile be
    shared across every event with this extent, even if those events'
    channel grids differ in size — callers resize the cheap, already-merged
    (ny, nx) array to their own (H, W) afterwards.
    """
    key = tuple(np.round(extent, 4))
    cache_path = _dem_extent_cache_path(DEM_CACHE_DIR, key)

    if os.path.exists(cache_path):
        try:
            cached = np.load(cache_path).astype(np.float32)
            if cached.shape == (ny, nx) and cached.size > 0:
                return cached
        except Exception as exc:
            log.warning(f"    Could not read cached DEM tile {cache_path} "
                        f"({exc}) — refetching.")

    dem_raw = fetch_and_regrid_dem(extent, nx=nx, ny=ny)
    if np.any(dem_raw):   # only cache genuine (non-all-zero) fetches
        try:
            np.save(cache_path, dem_raw)
        except Exception as exc:
            log.warning(f"    Could not write DEM cache {cache_path}: {exc}")

    return dem_raw


def load_dem_from_cache(event_id: str, sevir_catalog: pd.DataFrame,
                         H: int, W: int) -> np.ndarray:
    """
    Load the DEM tile for an event, using the extent-keyed disk cache
    (see _get_or_fetch_raw_dem / _dem_extent_cache_path, ported from
    prefetch_dem.py) in front of the real fetch_and_regrid_dem() pipeline
    (Copernicus DEM GLO-30 / Planetary Computer). Returns dem_norm in
    [0, 1] (NOT yet multiplied by LAMBDA_Z — callers apply the lambda_z
    weighting).
    """
    extent = get_event_extent(event_id, sevir_catalog)
    if extent is None:
        log.warning(f"    No extent found for {event_id} — DEM = zeros.")
        return np.zeros((H, W), dtype=np.float32)

    dem_raw = _get_or_fetch_raw_dem(extent)
    if dem_raw.shape != (H, W):
        dem_raw = cv2.resize(dem_raw, (W, H), interpolation=cv2.INTER_CUBIC)
    return _normalise_channel_minmax(dem_raw)


def prefetch_dem_for_events(
        event_ids:      list,
        sevir_catalog:  pd.DataFrame,
        channel_shapes: dict,
        max_workers:    int = 3,
) -> dict:
    """
    Bottleneck fix: fetch (or load from disk cache) the DEM tile for EVERY
    sampled event up front, in parallel, before any graph-build /
    calibration / ODE-integration work starts.

    Previously, load_dem_from_cache() was called one event at a time,
    inline in the same loop as the (fast, local) HDF5 channel load — so a
    cold DEM_CACHE_DIR meant N_EVENTS_PER_CLASS x n_classes sequential
    Planetary Computer STAC round-trips (network-latency-bound) sitting
    in front of the CPU-bound benchmark sweep, one event at a time.

    Unique-extent dedup (ported from prefetch_dem.py): events are first
    grouped by their ROUNDED (llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat)
    extent, mirroring prefetch_dem.py's `unique_extents` dict. Only ONE
    fetch (network + disk cache) is issued per UNIQUE extent, and its
    result is reused for every event sharing that extent — the same
    dedup prefetch_dem.py performs before submitting the login-node sweep.
    Since each fetch is I/O-bound (network request + tile download),
    running the unique extents concurrently collapses wall-clock cost from
    O(n_events x latency) down to roughly O(n_unique_extents x latency)
    for whichever extents are still cache misses.

    NOTE on concurrency: `max_workers` here only bounds how many UNIQUE
    EXTENTS' fetches are orchestrated at once (mostly idle threads waiting
    on their own extent's tile downloads/STAC search). The actual
    tile-level network + memory load is capped separately by the shared,
    module-level _tile_fetch_pool (see fetch_and_regrid_dem / the
    HAS_DEM_LIBS import block), so raising this number does NOT multiply
    into more concurrent full-resolution GeoTIFF reads — that used to
    happen when each event had its own private tile-fetch
    ThreadPoolExecutor(max_workers=8), which is what caused OOM kills once
    events were also parallelised.

    Cache HITS (the common case on repeat runs, or a login node warmed by
    running prefetch_dem.py first) are essentially free disk reads, so
    this also short-circuits near-instantly once DEM_CACHE_DIR is warm.

    Returns {event_id: dem_norm} — normalised DEM in [0, 1], resized to
    that event's own (H, W) and NOT yet multiplied by LAMBDA_Z (callers
    apply that weighting), for every event_id in `channel_shapes`. Events
    for which the fetch ultimately fails fall back to an all-zero DEM of
    the correct shape.
    """
    os.makedirs(DEM_CACHE_DIR, exist_ok=True)
    todo = [eid for eid in event_ids if eid in channel_shapes]
    results: dict = {}
    if not todo:
        return results

    # Group events by unique (rounded) extent — same logic as
    # prefetch_dem.py's main(): build a {extent_key: [event_ids]} map so
    # each distinct DEM tile is fetched exactly once regardless of how
    # many sampled events happen to share it.
    extent_to_events: dict[tuple, list] = {}
    for eid in todo:
        extent = get_event_extent(eid, sevir_catalog)
        if extent is None:
            log.warning(f"    No extent found for {eid} — DEM = zeros.")
            H, W = channel_shapes[eid]
            results[eid] = np.zeros((H, W), dtype=np.float32)
            continue
        key = tuple(np.round(extent, 4))
        extent_to_events.setdefault(key, (extent, []))
        extent_to_events[key][1].append(eid)

    n_unique = len(extent_to_events)
    n_cached = sum(
        1 for key in extent_to_events
        if os.path.exists(_dem_extent_cache_path(DEM_CACHE_DIR, key))
    )
    log.info(
        f"Prefetching DEM for {len(todo)} events -> {n_unique} unique "
        f"extent(s) ({n_cached} already cached, {n_unique - n_cached} to "
        f"fetch) with up to {max_workers} concurrent workers …"
    )

    def _fetch_one(key: tuple):
        extent, _eids = extent_to_events[key]
        return key, _get_or_fetch_raw_dem(extent)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, key): key for key in extent_to_events}
        for fut in tqdm(as_completed(futures), total=len(futures),
                         desc="DEM prefetch (unique extents)"):
            key = futures[fut]
            extent, eids = extent_to_events[key]
            try:
                _, dem_raw_canonical = fut.result()
            except Exception as exc:
                log.warning(f"  extent {key} ({len(eids)} event(s)): "
                            f"DEM prefetch failed ({exc}) — using zeros.")
                dem_raw_canonical = np.zeros((384, 384), dtype=np.float32)

            # Broadcast this ONE fetched/cached tile to every event that
            # shares the extent, resizing + normalising per event's own
            # (H, W) — the fetch itself already happened only once above.
            for eid in eids:
                H, W = channel_shapes[eid]
                dem_raw = dem_raw_canonical
                if dem_raw.shape != (H, W):
                    dem_raw = cv2.resize(dem_raw, (W, H),
                                          interpolation=cv2.INTER_CUBIC)
                results[eid] = _normalise_channel_minmax(dem_raw)

            # Diagnostic: log peak RSS after each unique extent finishes.
            # If the process gets killed again, the last line of this log
            # tells us exactly which extent was in flight and how much
            # memory had been used up to that point — instead of guessing.
            try:
                import resource
                peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                log.info(f"  extent {key} ({len(eids)} event(s)): done  |  "
                         f"peak RSS so far: {peak_mb:,.0f} MB")
            except Exception:
                pass
    log.info(f"DEM prefetch complete in {time.time() - t0:.1f}s "
             f"({len(results)}/{len(todo)} events, {n_unique} unique "
             f"extent(s) fetched).")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# OPTICAL FLOW  (A.5 step 4 — logged only; positions static in Block A)
# ─────────────────────────────────────────────────────────────────────────────

def compute_precomputed_flows(vil_channel: np.ndarray) -> list[np.ndarray]:
    """
    Farneback optical flow between consecutive VIL frames, matching the
    N-sweep's precomputed_flows construction (A.5 step 4):
    prev_gray = F_VIL[t-1]*255, curr_gray = F_VIL[t]*255, cast to uint8.
    """
    flows = []
    T = vil_channel.shape[0]
    for t in range(1, T):
        prev_gray = np.clip(vil_channel[t - 1] * 255.0, 0, 255).astype(np.uint8)
        curr_gray = np.clip(vil_channel[t] * 255.0, 0, 255).astype(np.uint8)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0,
        )
        flows.append(flow)
    return flows


def compute_superpixel_velocities(flows: list[np.ndarray],
                                   label_dense: np.ndarray,
                                   N_act: int) -> np.ndarray:
    """
    A.5 step 5: v_i = area-weighted mean of precomputed_flows[1] over each
    superpixel. Logged only — Block A holds positions static (x_i(t)=x_i(0)),
    so velocities are not used in the integration.
    """
    v = np.zeros((N_act, 2), dtype=np.float32)
    if len(flows) < 2:
        return v
    flow = flows[1]
    for k in range(N_act):
        mask = (label_dense == k)
        if mask.any():
            v[k] = flow[mask].mean(axis=0)
    return v


# ─────────────────────────────────────────────────────────────────────────────
# TOPOLOGICAL ADJACENCY  (A.5 step 6 — replaces the radius superset)
# ─────────────────────────────────────────────────────────────────────────────

def extract_topological_edges(label_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    A.5 step 6: topological adjacency E_mm^topo(t) from a SLIC label map.
    Two superpixels i != j are adjacent iff some pixel labelled i has a
    4-connected neighbour labelled j. This replaces the pre-allocated
    radius superset (E_super, R_max=108 px) used in the original script.

    Returns bidirectional (senders, receivers), each of length 2*|pairs|,
    exactly as specified ("Make bidirectional: senders and receivers each
    of length 2|pairs|").
    """
    right_diff = label_map[:, :-1] != label_map[:, 1:]
    left_pairs = np.stack(
        [label_map[:, :-1][right_diff], label_map[:, 1:][right_diff]], axis=1
    )

    down_diff = label_map[:-1, :] != label_map[1:, :]
    down_pairs = np.stack(
        [label_map[:-1, :][down_diff], label_map[1:, :][down_diff]], axis=1
    )

    pairs = np.concatenate([left_pairs, down_pairs], axis=0)
    if pairs.size == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    pairs_sorted = np.sort(pairs, axis=1)
    pairs_unique = np.unique(pairs_sorted, axis=0)

    senders   = np.concatenate([pairs_unique[:, 0], pairs_unique[:, 1]]).astype(np.int32)
    receivers = np.concatenate([pairs_unique[:, 1], pairs_unique[:, 0]]).astype(np.int32)
    return senders, receivers


def sigma_rbf(N: int, H: int, W: int, c_sigma: float) -> float:
    """A.2: N-dependent RBF bandwidth, sigma_rbf(N) = c_sigma * L_sp(N)."""
    return c_sigma * float(np.sqrt(H * W / N))


def compute_rbf_edge_weights(positions: np.ndarray, senders: np.ndarray,
                              receivers: np.ndarray, sigma: float) -> np.ndarray:
    """A.5 step 7: RBF edge weights on the topological edge set."""
    d = np.linalg.norm(positions[senders] - positions[receivers], axis=1)
    return np.exp(-(d ** 2) / (2.0 * sigma ** 2)).astype(np.float32)


def compute_lambda_max(n_nodes: int, senders: np.ndarray, receivers: np.ndarray,
                        edge_weights: np.ndarray) -> float:
    """
    Largest-magnitude eigenvalue of the (weighted) graph Laplacian
    L = degree_matrix - adjacency_matrix. Used for the CFL diagnostic
    (A.5's required lambda_max logging, and Figure A-4).
    """
    if n_nodes == 0 or len(senders) == 0:
        return 0.0
    A = scipy.sparse.coo_matrix(
        (edge_weights, (senders, receivers)), shape=(n_nodes, n_nodes)
    ).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    L = scipy.sparse.diags(deg) - A
    try:
        val = eigsh(L, k=1, which="LM", return_eigenvectors=False)
        return float(abs(val[0]))
    except Exception:
        # Fallback: power iteration for the largest-magnitude eigenvalue.
        rng = np.random.default_rng(0)
        v = rng.normal(size=n_nodes)
        v /= (np.linalg.norm(v) + 1e-12)
        for _ in range(50):
            v = L @ v
            v = v / (np.linalg.norm(v) + 1e-12)
        return float(abs(v @ (L @ v)))


def _weighted_k_max(graph: dict) -> float:
    """Maximum weighted degree k_max(N), used by the D calibration (A.2)."""
    n = graph["N_actual"]
    if n == 0 or len(graph["senders"]) == 0:
        return 0.0
    A = scipy.sparse.coo_matrix(
        (graph["edge_weights"], (graph["senders"], graph["receivers"])),
        shape=(n, n),
    ).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    return float(deg.max())


# ─────────────────────────────────────────────────────────────────────────────
# SUPERPIXEL GRAPH BUILDER  (A.5: full procedure)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL SLIC  (ported verbatim, save for docstring trimming, from
# TemporalSLIC_DEM in Visualize_sevir_with_superpixel_fused.py — the actual
# class used to produce the Chapter 5 N-sweep's label maps. A.5 REQUIRES the
# Lagrangian graph be built by this exact class, not by an independently
# parameterised skimage.slic() call, precisely so that Block A's stiffness
# measurements correspond to the graphs whose quality metrics determined N*
# per lifecycle class (blocks.tex, Preliminary section).
# ─────────────────────────────────────────────────────────────────────────────
class TemporalSLIC_DEM:
    def __init__(self, dem_norm: np.ndarray, n_segments: int = 150,
                 compactness: float = 10.0, lambda_z: float = 0.5):
        """
        Args:
            dem_norm   : Normalized DEM (H, W), values in [0, 1].
            n_segments : Target number of superpixels.
            compactness: Balances spectral vs. spatial proximity in SLIC.
            lambda_z   : Weight of the elevation channel in SLIC + gradient.
        """
        self.n_segments  = n_segments
        self.compactness = compactness
        self.lambda_z    = lambda_z

        # DEM kept for both SLIC channel appending and gradient computation
        self.dem_norm     = dem_norm                     # (H, W)  raw, for SLIC
        self.dem_weighted = dem_norm * lambda_z          # (H, W)  for gradient

        # Lagrangian state
        self.prev_labels    = None
        self.prev_centroids = None   # {label: [y, x]}
        self.prev_gray      = None   # uint8 grayscale for Farneback

    # ---------------------------------------------------------------------- #
    def segment(self, fused_frame: np.ndarray, use_flow: bool = True):
        """
        Segment one time-step of the fused feature cube.

        Args:
            fused_frame : (H, W, C) float32, each channel already in [0, 1].
                          Channels correspond to FUSION_CHANNELS (VIL, IR107, IR069…).
                          DEM is appended internally — do NOT include it here.
            use_flow    : If True (and not frame 0) advect centroids with
                          Farneback optical flow; otherwise centroids are frozen.

        Returns:
            labels : (H, W) int32  superpixel label map.
            flow   : (H, W, 2) float32  optical-flow field
                     (zeros on frame 0 or when use_flow=False).
        """
        # Gracefully handle a plain 2-D (single-channel) input
        if fused_frame.ndim == 2:
            fused_frame = fused_frame[:, :, np.newaxis]

        H, W, C = fused_frame.shape

        # ---- Grayscale proxy for optical flow --------------------------------
        # VIL (channel 0) carries the dominant storm motion signal.
        gray_norm = fused_frame[:, :, 0]
        curr_gray = (gray_norm * 255).astype(np.uint8)

        # ---- Build the full feature cube: satellite channels + DEM -----------
        # DEM is scaled by lambda_z so its influence matches the user weight.
        dem_ch       = (self.dem_norm * self.lambda_z)[:H, :W, np.newaxis]
        feature_cube = np.concatenate([fused_frame, dem_ch], axis=-1).astype(np.float64)
        # Shape: (H, W, C+1)

        flow = np.zeros((H, W, 2), dtype=np.float32)

        # ================================================================== #
        # CASE A: First Frame — initialize with multi-channel SLIC           #
        # ================================================================== #
        if self.prev_labels is None:
            # SLIC clusters in (C+1)-dimensional spectral space plus 2-D spatial.
            # channel_axis=-1 tells skimage the last axis is the channel axis.
            labels = skimage_slic(
                feature_cube,
                n_segments=self.n_segments,
                compactness=self.compactness,
                start_label=1,
                enforce_connectivity=True,
                channel_axis=-1,
            )
            self.prev_centroids = self._calculate_centroids(labels)

        # ================================================================== #
        # CASE B: Subsequent Frames — Lagrangian advection + Watershed       #
        # ================================================================== #
        else:
            # 1. Dense Optical Flow (Farneback on VIL grayscale proxy)
            if use_flow:
                flow = cv2.calcOpticalFlowFarneback(
                    self.prev_gray, curr_gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )

            # 2. Advect centroids (Lagrangian step)
            advected_markers = np.zeros((H, W), dtype=np.int32)

            for lbl, centroid in self.prev_centroids.items():
                y_old, x_old = int(centroid[0]), int(centroid[1])
                if 0 <= y_old < H and 0 <= x_old < W:
                    window = 5
                    y1 = max(0, y_old - window); y2 = min(H, y_old + window)
                    x1 = max(0, x_old - window); x2 = min(W, x_old + window)
                    flow_y = np.mean(flow[y1:y2, x1:x2, 1])
                    flow_x = np.mean(flow[y1:y2, x1:x2, 0])
                    y_new  = int(np.clip(y_old + flow_y, 0, H - 1))
                    x_new  = int(np.clip(x_old + flow_x, 0, W - 1))
                    advected_markers[y_new, x_new] = lbl

            # 3. Fused gradient: pixel-wise MAX of per-channel Sobel magnitudes
            #    plus the DEM Sobel.  Boundaries snap to the sharpest edge in
            #    ANY spectral band — not just the primary display channel.
            grad_layers = [sobel(fused_frame[:, :, c]) for c in range(C)]
            grad_layers.append(sobel(self.dem_weighted))     # DEM contribution
            combined_gradient = np.max(np.stack(grad_layers, axis=0), axis=0)

            # 4. Compact Watershed from advected markers
            labels = watershed(
                combined_gradient,
                markers=advected_markers,
                compactness=self.compactness / 100.0,
            )

            self.prev_centroids = self._calculate_centroids(labels)

        # Update Lagrangian state
        self.prev_labels = labels
        self.prev_gray   = curr_gray
        return labels, flow

    # ---------------------------------------------------------------------- #
    def _calculate_centroids(self, labels: np.ndarray) -> dict:
        """Return {label: np.array([y, x])} using scipy.ndimage for speed."""
        unique_labels = np.unique(labels)
        if len(unique_labels) and unique_labels[0] == 0:
            unique_labels = unique_labels[1:]
        centroids = center_of_mass(np.ones_like(labels), labels, unique_labels)
        return {lbl: np.array(c) for lbl, c in zip(unique_labels, centroids)}


def build_graph_topological(
        channels: dict[str, np.ndarray],
        dem_norm: np.ndarray,
        N: int,
        c_sigma: float,
        t_idx: int = 0,
        flows: Optional[list[np.ndarray]] = None,
) -> Optional[dict]:
    """
    A.5: build the Lagrangian superpixel graph, using TemporalSLIC_DEM
    (ported verbatim above from Visualize_sevir_with_superpixel_fused.py)
    run across the FULL T-frame window — not a single .segment() call.

    blocks.tex A.5 step 4 says to "Run Temporal SLIC ... on the 4-channel
    feature cube", and step 6 explicitly refers to "the label maps L_t for
    t=1,...,12" as "the propagated SLIC maps from the Temporal SLIC
    pipeline (optical-flow watershed propagation)" before then saying only
    L_0's topology is Block A's ACTIVE edge set. That means the propagated
    sequence must actually be produced — TemporalSLIC_DEM's CASE B
    (optical-flow advection + watershed re-labelling) is not optional
    machinery to skip past. Calling .segment() exactly once only ever
    exercises the class's "CASE A: First Frame" branch, which is no
    different from a bare skimage.slic() call — it is NOT "running Temporal
    SLIC", it's running plain SLIC once and calling it Temporal SLIC.

    Procedure:
      1. Instantiate ONE TemporalSLIC_DEM(dem_norm, n_segments=N,
         compactness=10, lambda_z=LAMBDA_Z) per (event, N) and call
         .segment() once per frame t=0..T-1, in order, on the (H, W, 3)
         [VIL, IR107, IR069] cube for that frame, with use_flow=True for
         t>=1 (Farneback optical flow between consecutive VIL frames,
         exactly the N-sweep's precomputed_flows construction) — this is
         the actual Lagrangian propagation: t=0 is SLIC (CASE A), each
         subsequent frame advects t-1's centroids by the flow and
         re-segments via compact watershed from those advected markers
         (CASE B), so labels are tracked (same integer id = same physical
         parcel) across the whole window, not independently reseeded.
      2. Extract node states / positions / the ACTIVE edge set from L_0 =
         label_maps[0] specifically (blocks.tex: "At Block A (no ODE,
         static positions), only L_0 is used since positions do not
         change" -> E_mm^active = E_mm^topo(t=0)). Positions/h0 come from
         frame t_idx (0 by default), matching L_0.
      3. Also compute E_mm^super = union_{t=0}^{T-1} E_mm^topo(t) (step 6)
         over the FULL propagated sequence, returned as a diagnostic /
         for reuse by Block B & C (blocks.tex: B3/B4 "integrate the actual
         DRIFT ODE and therefore require a fully specified Lagrangian
         graph -- this is exactly the graph constructed in Block A,
         Section A.5"). NOT used in Block A's own ODE integration, which
         is driven only by the L_0 active edge set per step 6.
      4. RBF edge weights on the ACTIVE (L_0) edge set using
         sigma_rbf(N) = c_sigma * sqrt(H*W/N).
      5. env_i = [x_i/W, y_i/H, h_i[VIL], h_i[IR107]].

    Returns a dict with keys:
      positions, h0, env, senders, receivers, edge_weights, velocities,
      N_actual, E, H, W, sigma_rbf, label_maps (all T propagated L_t),
      senders_super, receivers_super, E_super (diagnostic edge superset),
      lambda_max_topo   (unit-weight eigenvalue — topology-only stiffness)
      lambda_max_ew     (RBF-weight eigenvalue, NOT yet scaled by D)
    """
    vil_seq   = channels["vil"]     # (T, H, W)
    ir107_seq = channels["ir107"]
    ir069_seq = channels["ir069"]
    T, H, W   = vil_seq.shape

    def _resized(frame: np.ndarray) -> np.ndarray:
        return (frame if frame.shape == (H, W)
                else cv2.resize(frame, (W, H), interpolation=cv2.INTER_CUBIC))

    dem = dem_norm
    if dem.shape != (H, W):
        dem = cv2.resize(dem, (W, H), interpolation=cv2.INTER_CUBIC)

    # ── A.5 step 4: run TemporalSLIC_DEM across the FULL window ──────────────
    # ONE instance carries the Lagrangian state (prev_labels/prev_centroids/
    # prev_gray) across all T frames, exactly as
    # Visualize_sevir_with_superpixel_fused.py's precompute() loop does:
    #     tslic = TemporalSLIC_DEM(dem_norm, n_segments=N, ...)
    #     for t in range(T): labels, flow = tslic.segment(fused[t], use_flow=True)
    # t=0 takes CASE A (fresh SLIC, since prev_labels is None); t=1..T-1
    # take CASE B (Farneback advection of t-1's centroids + compact
    # watershed from those advected markers) — this is what makes it
    # "Temporal" rather than a one-shot SLIC call.
    try:
        tslic = TemporalSLIC_DEM(
            dem_norm=dem, n_segments=N, compactness=10.0, lambda_z=LAMBDA_Z,
        )
        label_maps: list[np.ndarray] = []
        for t in range(T):
            fused_t = np.stack([
                _resized(vil_seq[t]), _resized(ir107_seq[t]), _resized(ir069_seq[t]),
            ], axis=-1)
            labels_t, _flow_t = tslic.segment(fused_t, use_flow=(t > 0))
            label_maps.append(labels_t)
    except Exception as exc:
        log.warning(f"    TemporalSLIC_DEM propagation failed for N={N}: {exc}")
        return None

    segments = label_maps[t_idx]   # L_0 by default — the ACTIVE edge set (A.5 step 6)
    vil, ir107, ir069 = _resized(vil_seq[t_idx]), _resized(ir107_seq[t_idx]), _resized(ir069_seq[t_idx])

    unique_ids = np.unique(segments)
    N_act      = len(unique_ids)
    label_dense = np.searchsorted(unique_ids, segments).astype(np.int32)

    # ── A.5 step 5: centroids and node states (VIL, IR107, IR069 order) ─────
    positions = np.zeros((N_act, 2), dtype=np.float32)
    h0        = np.zeros((N_act, 3), dtype=np.float32)
    for k in range(N_act):
        mask   = (label_dense == k)
        ys, xs = np.where(mask)
        positions[k] = [xs.mean(), ys.mean()]
        h0[k, 0] = vil[mask].mean()
        h0[k, 1] = ir107[mask].mean()
        h0[k, 2] = ir069[mask].mean()

    velocities = (
        compute_superpixel_velocities(flows, label_dense, N_act)
        if flows is not None else np.zeros((N_act, 2), dtype=np.float32)
    )

    # ── A.5 step 6: topological adjacency of L_0 — Block A's ACTIVE edge set ─
    senders, receivers = extract_topological_edges(label_dense)

    # ── A.5 step 6 (superset, diagnostic / Block B & C reuse only): union of
    #     topological adjacency across ALL propagated L_t, re-expressed in
    #     the SAME dense node indexing derived from L_0 (labels are tracked
    #     across frames by TemporalSLIC_DEM's Lagrangian design, so this is
    #     coherent — a label id that survives to frame t refers to the same
    #     physical parcel it named at t=0). NOT consumed by Block A's own
    #     IMEX/DOPRI5 integration below, which uses `senders`/`receivers`
    #     (L_0 only) per step 6's explicit "E_mm^active = E_mm^topo(t=0)".
    super_pairs: set[tuple[int, int]] = set()
    for labels_t in label_maps:
        dense_t = np.searchsorted(unique_ids, np.clip(labels_t, unique_ids[0], unique_ids[-1]))
        # Guard against a label at t>0 that isn't one of L_0's ids (can
        # happen only in pathological watershed edge cases) by masking it
        # out rather than mis-mapping it onto a neighbouring L_0 id.
        valid = unique_ids[np.clip(dense_t, 0, N_act - 1)] == labels_t
        dense_t = np.where(valid, dense_t, -1).astype(np.int32)
        s_t, r_t = extract_topological_edges(dense_t)
        keep = (s_t >= 0) & (r_t >= 0)
        for a, b in zip(s_t[keep], r_t[keep]):
            super_pairs.add((int(a), int(b)) if a <= b else (int(b), int(a)))
    if super_pairs:
        pairs_arr = np.array(sorted(super_pairs), dtype=np.int32)
        senders_super   = np.concatenate([pairs_arr[:, 0], pairs_arr[:, 1]])
        receivers_super = np.concatenate([pairs_arr[:, 1], pairs_arr[:, 0]])
    else:
        senders_super = receivers_super = np.array([], dtype=np.int32)

    # ── A.5 step 7: RBF edge weights, N-dependent bandwidth (on L_0 only) ────
    sigma = sigma_rbf(N, H, W, c_sigma)
    edge_weights = compute_rbf_edge_weights(positions, senders, receivers, sigma)

    lambda_max_topo = compute_lambda_max(
        N_act, senders, receivers, np.ones_like(edge_weights)
    )
    lambda_max_ew = compute_lambda_max(N_act, senders, receivers, edge_weights)

    # ── A.5 step 8: environment features ─────────────────────────────────────
    env = np.zeros((N_act, ENV_DIM), dtype=np.float32)
    env[:, 0] = positions[:, 0] / W
    env[:, 1] = positions[:, 1] / H
    env[:, 2] = h0[:, 0]   # VIL
    env[:, 3] = h0[:, 1]   # IR107

    # Sample the (H, W)-resized DEM at the superpixel centroids. Use `dem`
    # (not the raw `dem_norm` argument) since `dem` is the version already
    # resized to match this event's (H, W) grid — the same grid `positions`
    # (in pixel x/y) and `h0` were derived from a few lines above.
    xs = np.clip(positions[:, 0].astype(int), 0, W - 1)
    ys = np.clip(positions[:, 1].astype(int), 0, H - 1)
    env[:, 4] = dem[ys, xs]   # DEM

    return {
        "positions":       positions,
        "h0":              h0,
        "env":             env,
        "velocities":      velocities,
        "senders":         senders,
        "receivers":       receivers,
        "edge_weights":    edge_weights,
        "N_actual":        N_act,
        "E":               len(senders),
        "H":               H,
        "W":               W,
        "sigma_rbf":       sigma,
        "lambda_max_topo": lambda_max_topo,
        "lambda_max_ew":   lambda_max_ew,
        # A.5 step 4/6 diagnostics: the full propagated label sequence and
        # the resulting edge superset E_mm^super = union_t E_mm^topo(t).
        # NOT used by Block A's own IMEX/DOPRI5 integration (which runs on
        # `senders`/`receivers` = E_mm^topo(t=0) only) — kept for the
        # lambda_max(L_t) diagnostic blocks.tex asks for, and because
        # Block B/C are specified to reuse exactly this graph construction.
        "label_maps":      label_maps,
        "senders_super":   senders_super,
        "receivers_super": receivers_super,
        "E_super":         len(senders_super),
    }


# ─────────────────────────────────────────────────────────────────────────────
# A.6 — c_sigma CALIBRATION  (run once, single STEADY event at N*)
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_c_sigma(
        channels: dict[str, np.ndarray],
        dem_norm: np.ndarray,
        N: int = N_STAR,
        candidates: list[float] = C_SIGMA_CANDIDATES,
) -> tuple[float, float, dict[float, float]]:
    """
    A.6: using a single STEADY-class event at N*=1150, compute
    lambda_max_topo (topological edges, ew=1). Sweep c_sigma, recompute
    lambda_max with RBF weights on the SAME topological edges, and select
    the c_sigma whose lambda_max is closest to lambda_max_topo (within 20%
    per the spec's intent). Topology is built once (independent of
    c_sigma); only edge weights are re-derived per candidate.

    Returns (best_c_sigma, lambda_max_topo, {c_sigma: lambda_max}).
    """
    base = build_graph_topological(channels, dem_norm, N=N, c_sigma=1.0)
    if base is None:
        raise RuntimeError("c_sigma calibration failed: could not build base graph.")

    n = base["N_actual"]
    lambda_max_topo = compute_lambda_max(
        n, base["senders"], base["receivers"], np.ones_like(base["edge_weights"])
    )

    H, W = base["H"], base["W"]
    results: dict[float, float] = {}
    for c in candidates:
        sigma = sigma_rbf(N, H, W, c)
        ew = compute_rbf_edge_weights(base["positions"], base["senders"],
                                       base["receivers"], sigma)
        results[c] = compute_lambda_max(n, base["senders"], base["receivers"], ew)

    best_c = min(
        candidates,
        key=lambda c: abs(results[c] - lambda_max_topo) / (lambda_max_topo + 1e-12),
    )
    return best_c, lambda_max_topo, results


# ─────────────────────────────────────────────────────────────────────────────
# A.2 — Tsit5 STABILITY RADIUS  (numerically probed, "not assumed")
# ─────────────────────────────────────────────────────────────────────────────

def estimate_tsit5_stability_radius(dt_test: float = 1.0) -> float:
    """
    A.2 requires r_stab to be "determined from its Butcher tableau, not
    assumed". Rather than hand-deriving the stability polynomial, this
    numerically probes Tsit5's actual step function on the scalar linear
    test problem dy/dt = lambda*y (lambda <= 0): the real-axis stability
    boundary IS defined by where |y1/y0| transitions from <=1 to >1 under a
    single fixed-size Tsit5 step, so probing the solver directly is
    equivalent to reading the boundary off the tableau's stability function.
    """
    if not JAX_OK:
        raise RuntimeError("JAX/Diffrax required for stability-radius estimation.")

    solver = diffrax.Tsit5()
    term = diffrax.ODETerm(lambda t, y, args: -args * y)

    # JIT once (lam traced), reused for every one of the ~50 bisection/
    # doubling evaluations below — without this, each call to
    # diffrax.diffeqsolve traces from scratch (no @jax.jit at all), which is
    # ~50x more tracing work than necessary for what is otherwise a single
    # cheap scalar ODE step.
    @jax.jit
    def _solve(lam):
        sol = diffrax.diffeqsolve(
            term, solver, t0=0.0, t1=dt_test, dt0=dt_test,
            y0=jnp.array(1.0), args=lam,
            stepsize_controller=diffrax.ConstantStepSize(),
            max_steps=4, throw=False,
        )
        return sol.ys[-1]

    def is_stable(z: float) -> bool:
        lam = z / dt_test
        y1 = float(_solve(jnp.asarray(lam)))
        return abs(y1) <= 1.0

    lo, hi = 0.0, 10.0
    tries = 0
    while is_stable(hi) and tries < 10:
        hi *= 2.0
        tries += 1
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if is_stable(mid):
            lo = mid
        else:
            hi = mid
    return lo   # r_stab


def calibrate_D(graph_at_nstar: dict, r_stab: float,
                 dt_frac: float = DT_FRAC_CFL, dt_macro: float = DT_MACRO) -> float:
    """
    A.2: set D such that the diffusion-only CFL step size at N*=1150 is
    approximately dt_frac (1/10) of the macro step:

        D = r_stab / (k_max(N*) * dt_frac * dt_macro)
    """
    k_max = _weighted_k_max(graph_at_nstar)
    if k_max <= 0:
        raise RuntimeError("D calibration failed: k_max(N*) is zero.")
    return r_stab / (k_max * dt_frac * dt_macro)


# ─────────────────────────────────────────────────────────────────────────────
# JAX WEIGHT INITIALISATION  (A.2 — reaction term only; diffusion is D*L*h)
# ─────────────────────────────────────────────────────────────────────────────

def make_weights(sigma_init: float, seed: int) -> dict:
    """
    Randomly initialise the reaction MLP weights. Diffusion has no learned
    parameters in Block A (A.2: f_E = D * sum_j ew_ij (h_j - h_i) with a
    single calibrated scalar D) — the old learned W_flux matrix has been
    removed.
    """
    key  = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 5)
    d, e, h = NODE_DIM, ENV_DIM, HIDDEN_DIM

    return {
        "gate_w": jax.random.normal(keys[0], (d + e, 1)) * 0.1,
        "gate_b": jnp.zeros(1),
        "src_w1": jax.random.normal(keys[1], (d + e, h)) * sigma_init,
        "src_w2": jax.random.normal(keys[2], (h, d))     * sigma_init,
        "snk_w1": jax.random.normal(keys[3], (d + e, h)) * sigma_init,
        "snk_w2": jax.random.normal(keys[4], (h, d))     * sigma_init,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ODE RIGHT-HAND SIDES  (A.2)
# ─────────────────────────────────────────────────────────────────────────────

def make_diffusion_rhs(senders_np, receivers_np, edge_weights_np, D: float):
    """
    A.2 explicit term — isotropic graph diffusion with a single calibrated
    scalar D (no learned parameters):
        dh_i = D * sum_j ew_ij(t) * (h_j - h_i)
    The antisymmetric (h_j - h_i) structure guarantees sum_i dh_i = 0
    (mass conservation).
    """
    s  = jnp.array(senders_np)
    r  = jnp.array(receivers_np)
    ew = jnp.array(edge_weights_np)   # (E,)
    D_arr = jnp.asarray(D)

    def diffusion_rhs(t, h, args):
        diff = h[r] - h[s]                              # (E, d) antisymmetric
        flux = D_arr * ew[:, None] * diff                 # (E, d)
        return jnp.zeros_like(h).at[r].add(flux)          # (N_act, d) scatter-add

    # Perf hook only (see the "FAST PATH" block above _diffrax_result_name):
    # exposes the same underlying arrays as a pytree so imex_strang_integrate
    # / dopri5_integrate can route the actual solve through a stable, cached
    # JIT program instead of baking these values into a fresh closure each
    # call. Does not change diffusion_rhs's own (t, h, args) behaviour.
    diffusion_rhs._rhs_args = {
        "senders": s, "receivers": r, "edge_weights": ew, "D": D_arr,
    }
    return diffusion_rhs


def _local_reaction(h_i, e_i, weights):
    """A.2 implicit term, per node (no inter-node coupling)."""
    h_env = jnp.concatenate([h_i, e_i])
    s     = jax.nn.sigmoid(h_env @ weights["gate_w"][:, 0] + weights["gate_b"][0])
    f_src = jax.nn.silu(h_env @ weights["src_w1"]) @ weights["src_w2"]
    f_snk = jax.nn.silu(h_env @ weights["snk_w1"]) @ weights["snk_w2"]
    return s * f_src - (1.0 - s) * f_snk


def make_reaction_rhs(env_jax, weights):
    """
    A.2 implicit term — competitive gated MLP reaction (SiLU activations,
    exactly as specified, replacing the old tanh MLPs):
        s_i   = sigmoid([h_i || e_i] . gate_w + gate_b)
        dh_i  = s_i * MLP_src(h_i, e_i) - (1 - s_i) * MLP_snk(h_i, e_i)
    """
    gw, gb   = weights["gate_w"], weights["gate_b"]
    sw1, sw2 = weights["src_w1"], weights["src_w2"]
    kw1, kw2 = weights["snk_w1"], weights["snk_w2"]
    env = env_jax   # (N_act, e)

    def reaction_rhs(t, h, args):
        h_env = jnp.concatenate([h, env], axis=-1)      # (N_act, d+e)
        s     = jax.nn.sigmoid(h_env @ gw + gb)          # (N_act, 1)
        f_src = jax.nn.silu(h_env @ sw1) @ sw2           # (N_act, d)
        f_snk = jax.nn.silu(h_env @ kw1) @ kw2           # (N_act, d)
        return s * f_src - (1.0 - s) * f_snk             # (N_act, d)

    # Perf hook only (see the "FAST PATH" block above _diffrax_result_name) —
    # does not change reaction_rhs's own (t, h, args) behaviour.
    reaction_rhs._rhs_args = {"env": env, "weights": weights}
    return reaction_rhs


def measure_reaction_spectral_radius(weights, h_sample, env_sample) -> float:
    """
    A.2 pseudocode: per-node d_m x d_m Jacobian-block spectral norm (2-norm),
    NOT the global spectral radius (which would be dominated by the
    diffusion graph Laplacian). Because f_I has no inter-node coupling, the
    full N*d_m x N*d_m Jacobian is already block-diagonal, so jacobian on
    the per-node function directly gives each block.
    """
    def per_node(h_i, e_i):
        J = jax.jacobian(_local_reaction)(h_i, e_i, weights)
        return jnp.linalg.norm(J, ord=2)

    rhos = jax.vmap(per_node)(h_sample, env_sample)
    return float(jnp.median(rhos))


def assess_reaction_stability_away_from_h0(
        weights, h0_jax, env_jax,
        perturbation_scales: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0),
        seed: int = 0,
) -> dict[str, float]:
    """
    DIAGNOSTIC (not part of the calibrated ODE itself — read-only, doesn't
    change sigma_init, weights, or the reaction architecture): calibrate_
    class_sigma()/measure_reaction_spectral_radius() only ever evaluate the
    reaction Jacobian AT h0 (A.2's own pseudocode: "J = jax.jacobian(f_I)
    (t=0, h_sample, env_sample)"). That's a LOCAL linearisation at a single
    point, not a guarantee about the Jacobian's behaviour once the
    trajectory moves away from h0 -- and it can move a lot, fast, for a
    class calibrated to a large target rho.

    Concretely: SiLU(x) = x * sigmoid(x) is UNBOUNDED and its derivative
    approaches 1 (not 0) as x -> +inf, unlike tanh/sigmoid activations
    which saturate. So MLP_src/MLP_snk's effective local Jacobian norm can
    keep GROWING as h grows past the small, near-zero values it started at
    -- the h0-calibrated rho is only a lower bound on what Kvaerno5 will
    actually encounter mid-integration for a class whose reaction term
    pushes h away from h0 quickly. This function makes that check
    empirical instead of asserting it: it re-measures the SAME
    measure_reaction_spectral_radius() Jacobian-norm at h0 perturbed by
    increasing multiples of its own scale (using the weights' own src/snk
    output as the perturbation direction — i.e. "one Euler-step's worth of
    reaction, scaled" rather than an arbitrary random direction), and
    reports how rho grows.

    Returns {f"scale_{s}": rho} for each requested perturbation scale.
    A roughly FLAT profile across scales means the h0 calibration is a
    trustworthy proxy for the whole trajectory (SiLU is behaving near-
    linearly/saturating in the range visited). A profile that grows
    sharply with scale is direct evidence that the h0-only calibration
    under-states the true stiffness Kvaerno5 will face once integration
    gets underway -- i.e. that random weight init calibrated only at h0 is
    NOT, by itself, a reliable proxy for the class's true dynamical
    stiffness, and the tolerance/max_steps budget needs to account for
    that gap rather than trusting rho_measured at face value.
    """
    rxn_rhs = make_reaction_rhs(env_jax, weights)
    # Direction: the reaction RHS's own output at h0 — i.e. perturb along
    # the direction the ODE itself is already pushing h, which is the
    # direction most likely to be visited early in the integration.
    direction = rxn_rhs(0.0, h0_jax, None)
    dir_norm = jnp.linalg.norm(direction) + 1e-12

    out: dict[str, float] = {}
    for scale in perturbation_scales:
        h_pert = h0_jax + scale * direction / dir_norm
        rho = measure_reaction_spectral_radius(weights, h_pert, env_jax)
        out[f"scale_{scale}"] = rho
    return out


def calibrate_class_sigma(
        lifecycle_class: str,
        calib_graphs: list[dict],
        seed: int = 0,
        max_iter: int = 15,
        tol: float = 0.1,
) -> tuple[float, float]:
    """
    A.2 "Stiffness calibration for Block A": calibrate sigma_init^c on up
    to 5 events at N*=1150 so the measured spectral radius matches
    rho_target^c = SRbase(c)/Delta_t_macro (TARGET_RHO), following the
    spec's "set and verify... adjust sigma^c_init and repeat" procedure.
    Since rho ~ sigma_init^2 * sqrt(d_hidden) (A.2), each iteration applies
    a direct multiplicative correction sqrt(target/measured).
    """
    target = TARGET_RHO.get(lifecycle_class, 1.0)
    sigma = 1.0
    rho_measured = float("nan")

    for _ in range(max_iter):
        rhos = []
        for g in calib_graphs:
            weights = make_weights(sigma, seed)
            rho = measure_reaction_spectral_radius(
                weights, jnp.array(g["h0"]), jnp.array(g["env"])
            )
            rhos.append(rho)
        if not rhos:
            break
        rho_measured = float(np.median(rhos))
        if abs(rho_measured - target) <= tol * target:
            break
        sigma *= math.sqrt(target / max(rho_measured, 1e-8))

    return sigma, rho_measured


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATORS  (A.3 IMEX Strang splitting, A.4 NFE bookkeeping)
# ─────────────────────────────────────────────────────────────────────────────

def _make_substep_fn(term, solver, max_steps: int, pid):
    """
    Build a JIT-compiled sub-step runner for a fixed (term, solver,
    max_steps, pid). Compiling once and reusing across the N_MACRO_STEPS
    loop (rather than calling diffrax.diffeqsolve eagerly per step) avoids
    per-iteration Python/XLA dispatch overhead — without this, each
    Newton/RK iteration inside diffrax's internal while_loop pays
    interpreter overhead, which otherwise dominates the reported wall-clock
    NFE timings (Section A.4) and makes even small graphs impractically slow.

    Returns a function (t0, t1, dt0, y0) -> (y_out, accepted, rejected, result)
    where accepted/rejected are still-traced JAX scalars and `result` is the
    RAW diffrax.RESULTS code (not yet reduced to a bool) — callers convert
    with int()/bool() once outside the jit boundary, and can additionally
    decode `result` via _diffrax_result_name() for diagnostics when a solve
    doesn't come back `successful` (e.g. max_steps_reached vs an implicit
    solver's Newton iteration diverging are very different problems that a
    single "ok=False" collapses together).
    """
    @jax.jit
    def _step(t0, t1, dt0, y0):
        sol = diffrax.diffeqsolve(
            term, solver,
            t0=t0, t1=t1, dt0=dt0,
            y0=y0, args=None,
            stepsize_controller=pid,
            saveat=diffrax.SaveAt(t1=True),
            max_steps=max_steps,
            throw=False,
        )
        y_out = sol.ys[-1]
        accepted = sol.stats.get("num_accepted_steps", sol.stats.get("num_steps", 0))
        rejected = sol.stats.get("num_rejected_steps", 0)
        return y_out, accepted, rejected, sol.result

    return _step


# ─────────────────────────────────────────────────────────────────────────────
# FAST PATH: cached, reusable JIT-compiled step functions
# ─────────────────────────────────────────────────────────────────────────────
# WHY THIS EXISTS: _make_substep_fn (above) builds a BRAND-NEW @jax.jit
# closure every time it's called, with rtol/atol/D/edge_weights/weights all
# baked in as Python-level constants captured by the closure. jax.jit's
# compilation cache is keyed on FUNCTION-OBJECT IDENTITY, so a fresh closure
# is always a cache miss — every seed in benchmark_event's seed loop, every
# candidate rtol in calibrate_tolerance's halving search, and every event at
# a given N was independently paying a full XLA trace+compile of Tsit5
# and/or Kvaerno5 and/or Dopri5, even though nothing about the PROBLEM
# STRUCTURE (only numeric values) had changed between calls.
#
# The fix: make the RHS functions actually fed to diffrax.ODETerm STABLE,
# MODULE-LEVEL functions that read all their numeric inputs (D, edge
# weights, senders/receivers, env, reaction MLP weights) from diffrax's own
# `args` mechanism instead of from a Python closure. Since `args` is a JAX
# pytree, jax.jit treats its leaves as traced values — different rtol/atol/
# weights/D/edge_weight VALUES (same shapes/dtypes) hit the SAME compiled
# program instead of retriggering a trace. Empirically (see the accompanying
# benchmarking), this took a synthetic case from ~2.2s per call (first-call
# compile) to ~0.0003s per call for every subsequent call with different
# numeric args — i.e. compilation happens ONCE per (solver, max_steps,
# graph-shape) combination for the entire process lifetime, not once per
# call.
#
# make_diffusion_rhs / make_reaction_rhs (below) are UNCHANGED in their
# public signature and math — they still return an ordinary (t, h, args)
# closure exactly as before, for any caller (this module's own
# benchmark_event, or test_block_a_single_event.py's run_one_event, which
# calls ba.make_diffusion_rhs/ba.make_reaction_rhs/ba.imex_strang_integrate
# directly) that wants to use them standalone. They are ADDITIONALLY tagged
# with a `._rhs_args` attribute exposing the same underlying arrays as a
# pytree, purely as a perf hook: imex_strang_integrate/dopri5_integrate
# check for this attribute and, if present, route the actual computation
# through the fast cached path below; if absent (a hand-written custom RHS
# not built via make_diffusion_rhs/make_reaction_rhs), they transparently
# fall back to the original _make_substep_fn behaviour, so nothing breaks.

def _fast_diffusion_rhs(t, h, args):
    """Stable, closure-free version of make_diffusion_rhs's math — reads
    (senders, receivers, edge_weights, D) from args["diff"] instead of a
    Python closure, so this exact function object (and its diffrax.ODETerm
    wrapper) can be built ONCE and reused for every call regardless of
    which graph/D is actually being solved."""
    d = args["diff"]
    diff = h[d["receivers"]] - h[d["senders"]]
    flux = d["D"] * d["edge_weights"][:, None] * diff
    return jnp.zeros_like(h).at[d["receivers"]].add(flux)


def _fast_reaction_rhs(t, h, args):
    """Stable, closure-free version of make_reaction_rhs's math — reads
    (env, weights) from args["rxn"] instead of a Python closure."""
    r = args["rxn"]
    env, weights = r["env"], r["weights"]
    h_env = jnp.concatenate([h, env], axis=-1)
    gw, gb   = weights["gate_w"], weights["gate_b"]
    sw1, sw2 = weights["src_w1"], weights["src_w2"]
    kw1, kw2 = weights["snk_w1"], weights["snk_w2"]
    s     = jax.nn.sigmoid(h_env @ gw + gb)
    f_src = jax.nn.silu(h_env @ sw1) @ sw2
    f_snk = jax.nn.silu(h_env @ kw1) @ kw2
    return s * f_src - (1.0 - s) * f_snk


def _fast_combined_rhs(t, h, args):
    """f_E + f_I on the shared {"diff":..., "rxn":...} args pytree — used
    for the unsplit DOPRI5 integration."""
    return _fast_diffusion_rhs(t, h, args) + _fast_reaction_rhs(t, h, args)


_FAST_STEP_CACHE: dict = {}   # (kind, max_steps) -> jitted step function, built once


def _get_cached_step(kind: str, max_steps: int):
    """
    Returns a JIT-compiled (t0, t1, dt0, y0, args, rtol, atol) -> (y_out,
    accepted, rejected, result) step function for `kind` in {"diffusion",
    "reaction", "dopri5"}. Built and compiled ONCE per (kind, max_steps)
    pair for the whole process (module-level cache keyed on that pair, not
    on shape — jax.jit's own internal cache already handles multiple
    y0/args shapes for the SAME function object, so different graph sizes
    across events/N-values still compile once each, exactly as many times
    as there are genuinely distinct shapes, and no more).
    """
    key = (kind, max_steps)
    cached = _FAST_STEP_CACHE.get(key)
    if cached is not None:
        return cached

    if kind == "diffusion":
        term, solver = diffrax.ODETerm(_fast_diffusion_rhs), diffrax.Tsit5()
    elif kind == "reaction":
        term, solver = diffrax.ODETerm(_fast_reaction_rhs), diffrax.Kvaerno5()
    elif kind == "dopri5":
        term, solver = diffrax.ODETerm(_fast_combined_rhs), diffrax.Dopri5()
    else:
        raise ValueError(f"unknown step kind: {kind!r}")

    @jax.jit
    def _step(t0, t1, dt0, y0, args, rtol, atol):
        pid = diffrax.PIDController(rtol=rtol, atol=atol)
        sol = diffrax.diffeqsolve(
            term, solver,
            t0=t0, t1=t1, dt0=dt0,
            y0=y0, args=args,
            stepsize_controller=pid,
            saveat=diffrax.SaveAt(t1=True),
            max_steps=max_steps,
            throw=False,
        )
        y_out = sol.ys[-1]
        accepted = sol.stats.get("num_accepted_steps", sol.stats.get("num_steps", 0))
        rejected = sol.stats.get("num_rejected_steps", 0)
        return y_out, accepted, rejected, sol.result

    _FAST_STEP_CACHE[key] = _step
    return _step


def _extract_fast_args(diff_rhs_fn, rxn_rhs_fn):
    """
    Pull the {"diff":..., "rxn":...} pytree out of diff_rhs_fn/rxn_rhs_fn if
    they were built via make_diffusion_rhs/make_reaction_rhs (tagged with
    ._rhs_args). Returns None if either callable lacks the tag, signalling
    callers to fall back to the original always-recompiles behaviour — this
    keeps arbitrary hand-written (t, h, args) callables working exactly as
    before, since the fast path requires knowing the concrete pytree layout.
    """
    diff_args = getattr(diff_rhs_fn, "_rhs_args", None)
    rxn_args  = getattr(rxn_rhs_fn, "_rhs_args", None)
    if diff_args is None or rxn_args is None:
        return None
    return {"diff": diff_args, "rxn": rxn_args}


def _diffrax_result_name(result) -> str:
    """
    Best-effort human-readable name for a diffrax.RESULTS code (e.g.
    'successful', 'max_steps_reached', 'implicit_divergence',
    'dt_min_reached') — used purely for diagnostic logging when a
    macro-step sub-solve doesn't come back successful, so a failure can be
    triaged (step-budget problem vs. genuine numerical divergence) without
    re-instrumenting the solver by hand each time. Falls back gracefully to
    the raw integer code if diffrax's internal RESULTS representation ever
    changes shape.
    """
    for accessor in (
        lambda r: diffrax.RESULTS[int(r)],   # if RESULTS supports int indexing
        lambda r: str(r),                     # Enumeration's own __str__/__repr__
    ):
        try:
            name = accessor(result)
            if name:
                return str(name)
        except Exception:
            continue
    return f"<diffrax result code {result}>"


def imex_strang_integrate(
        h0_jax,
        diff_rhs_fn,
        rxn_rhs_fn,
        dt:      float = DT_MACRO,
        n_steps: int   = N_MACRO_STEPS,
        rtol:    float = RTOL_DEFAULT,
        atol:    float = ATOL_DEFAULT,
        event_id:        str = "",
        lifecycle_class: str = "",
):
    """
    A.3: 60-minute IMEX Strang-split integration.
        L_D(dt/2)  ->  L_R(dt)  ->  L_D(dt/2)
    L_D uses Tsit5 (explicit), L_R uses Kvaerno5 (implicit, L-stable).

    `event_id` / `lifecycle_class` are optional and used ONLY to label the
    diagnostic warning logged when a sub-step doesn't come back
    `successful` — pass them from the caller so a failure in a large sweep
    can be traced back to which event/class/macro-step/stage it happened
    at (and via _diffrax_result_name(), WHICH failure mode: hitting
    max_steps vs. e.g. Kvaerno5's Newton iteration diverging are different
    problems requiring different fixes) instead of only surfacing as a
    single top-line `converged=False` with no further detail.

    Returns
    -------
    h_final                    : (N_act, d) JAX array
    nfe_diffusion, nfe_reaction : A.4 stage-count-scaled NFE
    rejected_diff, rejected_rxn : rejected-step diagnostics
    converged                   : bool
    """
    # Fast path: if diff_rhs_fn/rxn_rhs_fn came from make_diffusion_rhs/
    # make_reaction_rhs, route the solve through the cached, reusable JIT
    # step functions (see the "FAST PATH" block above _diffrax_result_name)
    # so compilation happens once per (max_steps, graph shape) instead of
    # once per call — this is what makes repeated calls across seeds and
    # calibrate_tolerance's halving search cheap. Falls back to the
    # original always-recompiles path for any custom RHS callable that
    # doesn't carry the ._rhs_args tag.
    fast_args = _extract_fast_args(diff_rhs_fn, rxn_rhs_fn)
    rtol_j, atol_j = jnp.asarray(rtol), jnp.asarray(atol)

    if fast_args is not None:
        diff_step_fn = _get_cached_step("diffusion", MAX_STEPS_EXPLICIT)
        rxn_step_fn  = _get_cached_step("reaction", MAX_STEPS_IMPLICIT)

        def diff_step(t0, t1, dt0, y0):
            return diff_step_fn(t0, t1, dt0, y0, fast_args, rtol_j, atol_j)

        def rxn_step(t0, t1, dt0, y0):
            return rxn_step_fn(t0, t1, dt0, y0, fast_args, rtol_j, atol_j)
    else:
        diff_term  = diffrax.ODETerm(diff_rhs_fn)
        rxn_term   = diffrax.ODETerm(rxn_rhs_fn)
        exp_solver = diffrax.Tsit5()
        imp_solver = diffrax.Kvaerno5()
        pid        = diffrax.PIDController(rtol=rtol, atol=atol)
        # Compiled once, reused for all n_steps macro iterations (see
        # _make_substep_fn docstring for why this matters for wall-clock NFE).
        diff_step = _make_substep_fn(diff_term, exp_solver, MAX_STEPS_EXPLICIT, pid)
        rxn_step  = _make_substep_fn(rxn_term, imp_solver, MAX_STEPS_IMPLICIT, pid)

    tag = f"[{lifecycle_class or '?'}/{event_id or '?'}]"

    def _check(stage: str, k: int, t0: float, t1: float, a, rj, result) -> bool:
        ok = bool(result == diffrax.RESULTS.successful)
        if not ok:
            log.warning(
                f"  {tag} IMEX {stage} sub-step FAILED at macro-step "
                f"{k + 1}/{n_steps} (t=[{t0:.0f}, {t1:.0f}]s): "
                f"diffrax result={_diffrax_result_name(result)}  "
                f"(accepted={int(a)}, rejected={int(rj)})"
            )
        return ok

    h = h0_jax
    acc_diff = acc_rxn = rej_diff = rej_rxn = 0
    converged = True

    for k in range(n_steps):
        t_s    = float(k) * dt
        t_half = t_s + dt / 2.0
        t_e    = t_s + dt

        h, a, rj, result = diff_step(t_s, t_half, dt / 20.0, h)
        acc_diff += int(a); rej_diff += int(rj)
        converged = converged and _check("diffusion (1st half-step)", k, t_s, t_half, a, rj, result)

        h, a, rj, result = rxn_step(t_s, t_e, dt / 2.0, h)
        acc_rxn += int(a); rej_rxn += int(rj)
        converged = converged and _check("reaction", k, t_s, t_e, a, rj, result)

        h, a, rj, result = diff_step(t_half, t_e, dt / 20.0, h)
        acc_diff += int(a); rej_diff += int(rj)
        converged = converged and _check("diffusion (2nd half-step)", k, t_half, t_e, a, rj, result)

    nfe_diff = acc_diff * STAGE_COUNT_TSIT5
    nfe_rxn  = acc_rxn * STAGE_COUNT_KVAERNO5  # nominal; see module docstring

    return h, nfe_diff, nfe_rxn, rej_diff, rej_rxn, converged


def dopri5_integrate(
        h0_jax,
        diff_rhs_fn,
        rxn_rhs_fn,
        dt:      float = DT_MACRO,
        n_steps: int   = N_MACRO_STEPS,
        rtol:    float = RTOL_DEFAULT,
        atol:    float = ATOL_DEFAULT,
        event_id:        str = "",
        lifecycle_class: str = "",
):
    """
    A.3/A.4: 60-minute unsplit DOPRI5 integration on the combined RHS
    f_E + f_I. NFE = accepted_steps x 6 (FSAL reuse).

    `event_id` / `lifecycle_class` are optional and used ONLY to label the
    diagnostic warning logged when a macro-step doesn't come back
    `successful` (see imex_strang_integrate's docstring for the rationale).
    """
    # See imex_strang_integrate's docstring for why the fast path exists.
    fast_args = _extract_fast_args(diff_rhs_fn, rxn_rhs_fn)
    rtol_j, atol_j = jnp.asarray(rtol), jnp.asarray(atol)

    if fast_args is not None:
        step_fn_cached = _get_cached_step("dopri5", MAX_STEPS_DOPRI5)

        def step_fn(t0, t1, dt0, y0):
            return step_fn_cached(t0, t1, dt0, y0, fast_args, rtol_j, atol_j)
    else:
        def combined_rhs(t, h, args):
            return diff_rhs_fn(t, h, args) + rxn_rhs_fn(t, h, args)

        term   = diffrax.ODETerm(combined_rhs)
        solver = diffrax.Dopri5()
        pid    = diffrax.PIDController(rtol=rtol, atol=atol)
        step_fn = _make_substep_fn(term, solver, MAX_STEPS_DOPRI5, pid)

    tag = f"[{lifecycle_class or '?'}/{event_id or '?'}]"

    h = h0_jax
    acc = rej = 0
    converged = True

    for k in range(n_steps):
        t_s = float(k) * dt
        t_e = t_s + dt
        h, a, rj, result = step_fn(t_s, t_e, dt / 10.0, h)
        ok = bool(result == diffrax.RESULTS.successful)
        acc += int(a); rej += int(rj); converged = converged and ok
        if not ok:
            log.warning(
                f"  {tag} DOPRI5 step FAILED at macro-step {k + 1}/{n_steps} "
                f"(t=[{t_s:.0f}, {t_e:.0f}]s): "
                f"diffrax result={_diffrax_result_name(result)}  "
                f"(accepted={int(a)}, rejected={int(rj)})"
            )

    nfe = acc * STAGE_COUNT_DOPRI5
    return h, nfe, rej, converged


# ─────────────────────────────────────────────────────────────────────────────
# DIVERGENCE-VS-STIFFNESS DIAGNOSTIC
# ─────────────────────────────────────────────────────────────────────────────
# "Newton can't find a root on near-vertical gradients" (true stiffness) and
# "the state is genuinely blowing up toward infinity" (divergence) both show
# up identically as max_steps_reached with lots of rejections — but they call
# for completely different fixes. Kvaerno5 is L-stable specifically so it
# does NOT need to fight arbitrarily large-but-finite stiffness: if the true
# solution stays bounded, a good adaptive controller should eventually find a
# workable step size no matter how stiff the local Jacobian is. What it
# can't do is integrate through an actual finite-time blow-up, where the
# controller is CORRECTLY shrinking dt because the solution really is
# diverging — in that case raising max_steps just delays the failure at the
# cost of more compute, it doesn't fix anything.
#
# The cheap diagnostic (re-run a failing sub-solver with every accepted
# step's state retained, rather than just the final one) distinguishes them:
#   - max(|h|) climbing toward large values while the accepted dt
#     monotonically collapses  -> DIVERGENCE (more steps/tighter tolerance
#     won't help; the reaction/DOPRI5 output itself needs bounding)
#   - max(|h|) stays bounded while dt is small/oscillating -> STIFFNESS (a
#     smaller step budget or tighter tolerance is the correct, sufficient fix)
#
# Two additional diagnostics, both added because the growth/dt-shrink
# heuristic above can itself go quiet exactly when it matters most:
#
#   1. Overflow trap. Once |h| actually overflows to inf/nan, diffrax's PID
#      error estimate becomes nan too, and "is error > 1" silently evaluates
#      False for nan in most implementations — so the controller doesn't
#      necessarily keep shrinking dt as it approaches the blow-up, it can
#      just accept degenerate steps with a flat dt. That produces exactly
#      the "huge h_growth, small dt_shrink" pattern that falls through the
#      is_diverging condition below and gets mislabelled AMBIGUOUS, even
#      though a growth factor of 1e30+ in a state meant to live in [0, 1]^3
#      is not actually ambiguous. Checking directly for non-finite/blown-up
#      |h| sidesteps the heuristic entirely.
#
#   2. Dominant-eigenvalue sign at h0. measure_reaction_spectral_radius()
#      only ever reports ||J||_2 — an unsigned magnitude — so calibrate_
#      class_sigma() can happily match a target rho without knowing whether
#      the calibrated linearisation is actually DISSIPATIVE (negative real
#      eigenvalue — genuine "stiffness" in the classical sense Kvaerno5 is
#      built to exploit efficiently) or locally UNSTABLE (positive real
#      eigenvalue — a genuinely growing mode that no solver, implicit or
#      explicit, can integrate efficiently, because the true solution really
#      is diverging). A.2's calibration has no reason to prefer one sign
#      over the other under random init, so this is reported alongside the
#      growth/shrink numbers rather than folded silently into rho_measured.

def _dominant_eig_real_part(weights, h_sample, env_sample) -> float:
    """
    Median (across nodes) real part of the DOMINANT eigenvalue of the
    per-node reaction Jacobian at (h_sample, env_sample) — signed, unlike
    measure_reaction_spectral_radius()'s ||J||_2. Because f_I has no
    inter-node coupling, the full Jacobian is block-diagonal, so this is
    exactly the per-node eigenvalue problem (small, d_m x d_m).

    Negative  -> locally dissipative at this point (classical stiffness:
                 fast relaxation, which implicit solvers handle gracefully).
    Positive  -> locally unstable at this point (a genuinely growing mode;
                 the reaction ODE's true solution diverges here regardless
                 of solver choice — this is NOT something max_steps or a
                 tighter tolerance can fix).
    """
    def per_node(h_i, e_i):
        J = jax.jacobian(_local_reaction)(h_i, e_i, weights)
        eigs = jnp.linalg.eigvals(J)
        return jnp.max(jnp.real(eigs))

    dom = jax.vmap(per_node)(h_sample, env_sample)
    return float(jnp.median(dom))


def diagnose_stiffness_vs_divergence(
        h0_jax,
        diff_rhs_fn,
        rxn_rhs_fn,
        rtol: float = RTOL_DEFAULT,
        atol: float = ATOL_DEFAULT,
        dt: float = DT_MACRO,
        n_steps: int = N_MACRO_STEPS,
        kind: str = "reaction",
        event_id: str = "",
        lifecycle_class: str = "",
        h_growth_threshold: float = 5.0,
        dt_shrink_threshold: float = 0.1,
        overflow_threshold: float = 1e6,
) -> dict:
    """
    Re-runs ONE sub-solver (`kind` in {"diffusion", "reaction", "dopri5"})
    across the SAME macro-step schedule as imex_strang_integrate /
    dopri5_integrate, but with SaveAt(steps=True) instead of SaveAt(t1=True)
    so every accepted step's (t, h) is kept, not just the final state. From
    that trajectory it builds max(|h|) and accepted-dt histories (both
    concatenated across all n_steps macro-steps) and classifies the failure
    per the rule above, then sharpens that classification with two direct
    checks (overflow trap + dominant-eigenvalue sign) rather than relying on
    the growth/shrink heuristic alone.

    NOT part of the calibrated ODE integration and NOT on the fast cached-
    JIT path used by the main sweep (see the "FAST PATH" block above
    _diffrax_result_name): SaveAt(steps=True) has to allocate and return a
    full max_steps-length trajectory, and this function isn't JIT-compiled/
    cached across calls, so it is meaningfully more expensive per call. It's
    meant for triaging a HANDFUL of already-identified failing
    (event, N, seed) triples, not for folding into the main NFE sweep.

    Returns a dict:
      verdict             : "DIVERGENCE" | "STIFFNESS" | "AMBIGUOUS"
                            ("DIVERGENCE" now also fires directly off the
                            overflow trap, independent of h_growth/dt_shrink)
      t_history           : concatenated accepted-step end times (np.ndarray)
      h_max_history       : max(|h|) at each accepted step (np.ndarray)
      dt_history          : accepted step size for each of those steps
      h_growth            : final/initial max(|h|) ratio used for the verdict
      dt_shrink           : final/initial accepted-dt ratio used for the verdict
      dt_trend_corr       : correlation of dt with step index (very negative
                            => steadily collapsing dt)
      n_accepted_total    : total accepted steps across the whole schedule
      overflow_detected   : True if any accepted step has non-finite |h| or
                            |h| > overflow_threshold — a direct, heuristic-
                            free divergence signal (see module comment above)
      first_overflow_step : index (into h_max_history) of the first such
                            step, or None if overflow_detected is False
      first_overflow_t    : simulated time of that step, or None
      dominant_eig_real   : median (across nodes) SIGNED real part of the
                            reaction Jacobian's dominant eigenvalue at h0 —
                            negative = locally dissipative (classical
                            stiffness), positive = locally unstable (a
                            genuinely growing mode). None for kind=
                            "diffusion" (no reaction weights involved) or if
                            rxn_rhs_fn doesn't carry the `_rhs_args` hook
                            make_reaction_rhs() attaches.
    """
    if kind == "diffusion":
        term, solver, max_steps = diffrax.ODETerm(diff_rhs_fn), diffrax.Tsit5(), MAX_STEPS_EXPLICIT
        sub_dt0 = dt / 20.0
    elif kind == "reaction":
        term, solver, max_steps = diffrax.ODETerm(rxn_rhs_fn), diffrax.Kvaerno5(), MAX_STEPS_IMPLICIT
        sub_dt0 = dt / 2.0
    elif kind == "dopri5":
        def combined_rhs(t, h, args):
            return diff_rhs_fn(t, h, args) + rxn_rhs_fn(t, h, args)
        term, solver, max_steps = diffrax.ODETerm(combined_rhs), diffrax.Dopri5(), MAX_STEPS_DOPRI5
        sub_dt0 = dt / 10.0
    else:
        raise ValueError(f"unknown kind: {kind!r} (expected diffusion/reaction/dopri5)")

    pid = diffrax.PIDController(rtol=rtol, atol=atol)
    tag = f"[{lifecycle_class or '?'}/{event_id or '?'}]"

    # Diagnostic 2: dominant-eigenvalue sign at h0. Cheap (one jacobian +
    # eigvals call over the node dimension), so just always compute it when
    # we have the reaction weights/env available, regardless of `kind` —
    # a "diffusion"-only run has no reaction weights to check.
    dominant_eig_real = None
    rxn_args = getattr(rxn_rhs_fn, "_rhs_args", None)
    if kind in ("reaction", "dopri5") and rxn_args is not None:
        dominant_eig_real = _dominant_eig_real_part(
            rxn_args["weights"], h0_jax, rxn_args["env"],
        )
        sign_note = "UNSTABLE (positive)" if dominant_eig_real > 0 else "dissipative (negative)"
        log.info(
            f"  {tag} [diagnose_stiffness_vs_divergence:{kind}] dominant_eig_real="
            f"{dominant_eig_real:.3f} at h0 -> {sign_note}"
        )

    t_hist:  list = []
    h_hist:  list = []
    dt_hist: list = []
    total_rejected = 0
    total_max_steps_budget = max_steps * n_steps

    h = h0_jax
    for k in range(n_steps):
        t_s = float(k) * dt
        t_e = t_s + dt
        sol = diffrax.diffeqsolve(
            term, solver, t0=t_s, t1=t_e, dt0=sub_dt0, y0=h, args=None,
            stepsize_controller=pid, saveat=diffrax.SaveAt(steps=True),
            max_steps=max_steps, throw=False,
        )
        n_acc = int(sol.stats["num_accepted_steps"])
        total_rejected += int(sol.stats.get("num_rejected_steps", 0))
        if n_acc > 0:
            ts = np.asarray(sol.ts)[:n_acc]
            ys = np.asarray(sol.ys)[:n_acc]
            step_starts = np.concatenate([[t_s], ts[:-1]]) if n_acc > 1 else np.array([t_s])
            t_hist.extend(ts.tolist())
            h_hist.extend(np.max(np.abs(ys.reshape(n_acc, -1)), axis=1).tolist())
            dt_hist.extend((ts - step_starts).tolist())
            h = jnp.asarray(ys[-1])
        if not bool(sol.result == diffrax.RESULTS.successful):
            log.warning(
                f"  {tag} [diagnose_stiffness_vs_divergence:{kind}] macro-step "
                f"{k + 1}/{n_steps} did not converge "
                f"(diffrax result={_diffrax_result_name(sol.result)}); "
                f"continuing from its last accepted state to keep building "
                f"the trajectory for diagnosis."
            )

    t_arr  = np.asarray(t_hist, dtype=np.float64)
    h_arr  = np.asarray(h_hist, dtype=np.float64)
    dt_arr = np.asarray(dt_hist, dtype=np.float64)

    # Diagnostic 1: overflow trap. np.abs(nan) > threshold is False (nan
    # comparisons are always False), so the isfinite check is required
    # alongside the magnitude check — a step that's already gone to nan
    # would otherwise slip past a magnitude-only test.
    overflow_mask = ~np.isfinite(h_arr) | (np.abs(h_arr) > overflow_threshold)
    overflow_detected = bool(np.any(overflow_mask))
    if overflow_detected:
        first_overflow_step = int(np.argmax(overflow_mask))
        first_overflow_t = float(t_arr[first_overflow_step])
        log.warning(
            f"  {tag} [diagnose_stiffness_vs_divergence:{kind}] OVERFLOW: "
            f"|h| non-finite or > {overflow_threshold:.0e} at accepted step "
            f"{first_overflow_step} (t={first_overflow_t:.4g}s) — treating "
            f"as DIVERGENCE regardless of the growth/dt-shrink heuristic."
        )
    else:
        first_overflow_step = None
        first_overflow_t = None

    if len(h_arr) < 2:
        # Too little accepted-step data to fit a growth/dt trend — this
        # itself usually means Newton (or the PID controller) rejected
        # almost every attempted step, which is its own divergence signal:
        # a genuinely blowing-up RHS makes the local Jacobian/derivative so
        # large that even the FIRST step candidates get rejected outright,
        # long before 2 could ever be accepted. Treat "almost the entire
        # step budget burned on rejections, ~0 accepted" as low-confidence
        # DIVERGENCE rather than punting to AMBIGUOUS, since that pattern
        # is not what ordinary (bounded) stiffness looks like — a stiff but
        # stable problem still accepts a steady trickle of tiny steps.
        reject_frac = total_rejected / max(1, total_rejected + len(h_arr))
        if overflow_detected:
            # Direct evidence trumps the low-data heuristic entirely — no
            # need to reason about reject_frac if |h| has already overflowed.
            verdict = "DIVERGENCE"
            log.warning(
                f"  {tag} [diagnose_stiffness_vs_divergence:{kind}] "
                f"only {len(h_arr)} step(s) accepted, but overflow was "
                f"directly observed at step {first_overflow_step} — DIVERGENCE, "
                f"not low-confidence."
            )
        elif total_rejected > 0.5 * total_max_steps_budget and reject_frac > 0.95:
            verdict = "DIVERGENCE (low data — see note)"
            log.warning(
                f"  {tag} [diagnose_stiffness_vs_divergence:{kind}] "
                f"only {len(h_arr)} step(s) ever accepted out of "
                f"{total_rejected} rejected (budget {total_max_steps_budget}) "
                f"— treating as likely DIVERGENCE: even the earliest step "
                f"candidates were rejected, which bounded stiffness alone "
                f"does not typically do."
            )
        else:
            verdict = "AMBIGUOUS"
            log.warning(f"  {tag} [diagnose_stiffness_vs_divergence:{kind}] "
                        f"fewer than 2 accepted steps total — cannot classify.")
        return {
            "verdict": verdict, "t_history": t_arr, "h_max_history": h_arr,
            "dt_history": dt_arr, "h_growth": float("nan"), "dt_shrink": float("nan"),
            "dt_trend_corr": float("nan"), "n_accepted_total": len(h_arr),
            "n_rejected_total": total_rejected,
            "overflow_detected": overflow_detected,
            "first_overflow_step": first_overflow_step,
            "first_overflow_t": first_overflow_t,
            "dominant_eig_real": dominant_eig_real,
        }

    n_edge = max(1, min(5, len(h_arr) // 4))
    h_growth  = float(np.mean(h_arr[-n_edge:])) / (float(np.mean(h_arr[:n_edge])) + 1e-12)
    dt_shrink = float(np.mean(dt_arr[-n_edge:])) / (float(np.mean(dt_arr[:n_edge])) + 1e-12)

    step_idx = np.arange(len(dt_arr))
    dt_trend_corr = (
        float(np.corrcoef(step_idx, dt_arr)[0, 1])
        if len(dt_arr) >= 3 and np.std(dt_arr) > 0 else 0.0
    )

    is_diverging     = (h_growth > h_growth_threshold and dt_shrink < dt_shrink_threshold
                         and dt_trend_corr < -0.3)
    is_bounded_stiff = (not overflow_detected and h_growth <= h_growth_threshold and
                        (dt_shrink < dt_shrink_threshold or dt_arr.min() < sub_dt0 * dt_shrink_threshold))

    if overflow_detected:
        # Direct evidence overrides the growth/dt-shrink heuristic — this is
        # precisely the case that heuristic can miss (huge h_growth alongside
        # a flat dt_shrink, because a nan error estimate doesn't necessarily
        # keep shrinking dt the way a "clean" approach to blow-up would).
        verdict = "DIVERGENCE"
    elif is_diverging:
        verdict = "DIVERGENCE"
    elif is_bounded_stiff:
        verdict = "STIFFNESS"
    else:
        verdict = "AMBIGUOUS"

    eig_note = (
        f"  dominant_eig_real={dominant_eig_real:.3f}" if dominant_eig_real is not None else ""
    )
    log.info(
        f"  {tag} [diagnose_stiffness_vs_divergence:{kind}] verdict={verdict}  "
        f"h_growth={h_growth:.2f}x  dt_shrink={dt_shrink:.2e}x  "
        f"dt_trend_corr={dt_trend_corr:.2f}  overflow_detected={overflow_detected}"
        f"{eig_note}  (n_accepted_total={len(h_arr)})"
    )

    return {
        "verdict": verdict, "t_history": t_arr, "h_max_history": h_arr,
        "dt_history": dt_arr, "h_growth": h_growth, "dt_shrink": dt_shrink,
        "dt_trend_corr": dt_trend_corr, "n_accepted_total": len(h_arr),
        "n_rejected_total": total_rejected,
        "overflow_detected": overflow_detected,
        "first_overflow_step": first_overflow_step,
        "first_overflow_t": first_overflow_t,
        "dominant_eig_real": dominant_eig_real,
    }


# ─────────────────────────────────────────────────────────────────────────────
# A.3 — TOLERANCE SELECTION  (optional; expensive, off by default)
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_tolerance(h0_jax, diff_rhs_fn, rxn_rhs_fn,
                         rtol_start: float = 1e-2, min_rtol: float = 1e-8,
                         label: str = ""):
    """
    A.3: start at rtol=1e-2, halve iteratively, and stop when the solution's
    relative change between consecutive halvings drops below 1%. atol is
    kept at rtol/10 throughout.

    IMPORTANT — which (h0, diff_rhs_fn, rxn_rhs_fn) to pass in: blocks.tex
    A.3 (line 218) requires ONE SHARED (rtol, atol) used across every class,
    so whichever class's dynamics this function is calibrated against
    determines what the WHOLE benchmark's tolerance ends up being. Calling
    this with a MILD class's weights/graph (e.g. STEADY, target rho~0.3)
    will happily converge at a comfortably tight tolerance and tell you
    NOTHING about whether that same shared tolerance is remotely workable
    for a genuinely stiff class (e.g. RAPID_GROWTH, target rho~33 — ~100x
    stiffer). Per A.3 condition (ii) ("loose enough that NFE is not
    dominated by accuracy requirements... otherwise the stiffness
    difference is hidden"), the calibration needs to be SENSITIVE to the
    stiff end of the class spectrum, not just the calm end — so callers
    should pass in the STIFFEST class's (h0, diff_rhs, rxn_rhs), not an
    arbitrary/convenient one. See run_benchmark()'s call site.

    Each candidate tolerance's solve is checked for actual convergence
    (`ok`) before its h_final is trusted for the rel_change comparison —
    an EARLIER version of this function discarded the convergence flag
    entirely (`h_final, *_ = imex_strang_integrate(...)`), so a candidate
    that hit max_steps_reached could silently be treated as a valid,
    stable sample. If even rtol_start itself fails to converge, that's a
    genuine finding (the reaction dynamics are too stiff for ANY tolerance
    in the tested range at the current max_steps budget) and is now
    surfaced as a warning rather than silently swallowed.
    """
    rtol = rtol_start
    prev_h = None
    last_good: Optional[tuple[float, float]] = None

    while rtol >= min_rtol:
        atol = rtol * 0.1
        h_final, _nfe_diff, _nfe_rxn, _rej_diff, _rej_rxn, ok = imex_strang_integrate(
            h0_jax, diff_rhs_fn, rxn_rhs_fn, rtol=rtol, atol=atol,
            lifecycle_class=label,
        )
        if not ok:
            log.warning(
                f"    [tolerance calib{f' {label}' if label else ''}] "
                f"rtol={rtol:.1e} atol={atol:.1e}: IMEX did NOT converge "
                f"(max_steps_reached or similar) — this candidate tolerance "
                f"is untrustworthy, not just imprecise; stopping the "
                f"halving search here rather than tightening further."
            )
            break

        if prev_h is not None:
            rel_change = float(jnp.linalg.norm(h_final - prev_h)) / (
                float(jnp.linalg.norm(prev_h)) + 1e-12
            )
            if rel_change < 0.01:
                return rtol, atol

        last_good = (rtol, atol)
        prev_h = h_final
        rtol /= 2.0

    if last_good is not None:
        return last_good
    # Not even rtol_start converged — return it anyway (it's still the
    # spec's prescribed starting point) but the warning above is the real
    # signal here: this class's dynamics need either a looser starting
    # point, a larger max_steps budget, or a shorter macro step, not a
    # smaller rtol.
    return rtol_start, rtol_start * 0.1


# ─────────────────────────────────────────────────────────────────────────────
# PER-EVENT BENCHMARK  (sweep over N for one event)
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_event(
        event_id:        str,
        lifecycle_class:  str,
        channels:        dict,
        dem_norm:        np.ndarray,
        n_values:        list[int],
        seeds:           list[int],
        c_sigma:         float,
        D:               float,
        sigma_init:      float,
        rtol:            float = RTOL_DEFAULT,
        atol:            float = ATOL_DEFAULT,
) -> list[dict]:
    """
    For a single event, sweep N and return a list of result rows.
    Each row covers one (N, seed) combination.
    """
    rows = []

    for N in n_values:
        log.debug(f"    N={N} …")

        graph = build_graph_topological(channels, dem_norm, N=N, c_sigma=c_sigma)
        if graph is None:
            log.warning(f"    N={N}: graph build failed, skipping.")
            continue

        N_act = graph["N_actual"]
        E     = graph["E"]

        env_jax = jnp.array(graph["env"])
        h0_jax  = jnp.array(graph["h0"])

        # Figure A-4: lambda_max(L) exactly = D * lambda_max(L_ew), since
        # f_E = D * L_ew * h is linear in D.
        lambda_max_weighted = D * graph["lambda_max_ew"]

        for seed in seeds:
            weights  = make_weights(sigma_init=sigma_init, seed=seed)
            diff_rhs = make_diffusion_rhs(
                graph["senders"], graph["receivers"], graph["edge_weights"], D,
            )
            rxn_rhs  = make_reaction_rhs(env_jax, weights)

            rho = float("nan")
            if seed == seeds[0]:
                try:
                    rho = measure_reaction_spectral_radius(weights, h0_jax, env_jax)
                except Exception:
                    pass

            # ── IMEX Strang ──────────────────────────────────────────────
            t0_imex = time.perf_counter()
            h_imex, nfe_diff, nfe_rxn, rej_diff, rej_rxn, ok_imex = \
                imex_strang_integrate(h0_jax, diff_rhs, rxn_rhs, rtol=rtol, atol=atol,
                                       event_id=event_id, lifecycle_class=lifecycle_class)
            wall_imex = time.perf_counter() - t0_imex
            nfe_imex  = nfe_diff + nfe_rxn

            # ── DOPRI5 unsplit ───────────────────────────────────────────
            t0_dop = time.perf_counter()
            h_dop, nfe_dop, rej_dop, ok_dop = \
                dopri5_integrate(h0_jax, diff_rhs, rxn_rhs, rtol=rtol, atol=atol,
                                  event_id=event_id, lifecycle_class=lifecycle_class)
            wall_dop = time.perf_counter() - t0_dop

            # ── Solution agreement ───────────────────────────────────────
            denom  = float(jnp.linalg.norm(h_dop) + 1e-12)
            l2_err = float(jnp.linalg.norm(h_imex - h_dop)) / denom

            rows.append({
                "event_id":            event_id,
                "lifecycle_class":     lifecycle_class,
                "N":                   N,
                "N_actual":            N_act,
                "E":                   E,
                "seed":                seed,
                "c_sigma":             c_sigma,
                "sigma_rbf":           graph["sigma_rbf"],
                "D":                   D,
                "sigma_init":          sigma_init,
                "spectral_radius":     rho,
                "lambda_max_topo":     graph["lambda_max_topo"],
                "lambda_max_weighted": lambda_max_weighted,
                # ── IMEX breakdown ──────────────────────────────────────
                "nfe_diffusion":       nfe_diff,
                "nfe_reaction":        nfe_rxn,
                "nfe_imex_total":      nfe_imex,
                "rejected_diff":       rej_diff,
                "rejected_rxn":        rej_rxn,
                "wall_imex_sec":       round(wall_imex, 4),
                "imex_converged":      ok_imex,
                # ── DOPRI5 ──────────────────────────────────────────────
                "nfe_dopri5":          nfe_dop,
                "rejected_dopri5":     rej_dop,
                "wall_dopri5_sec":     round(wall_dop, 4),
                "dopri5_converged":    ok_dop,
                # ── Derived ─────────────────────────────────────────────
                "nfe_ratio":           round(nfe_dop / (nfe_imex + 1), 4),
                "solution_l2_error":   round(l2_err, 6),
            })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def _save_calibration_json(calib: dict, path: str = CALIB_JSON) -> None:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(calib, f, indent=2, default=float)
        log.info(f"Calibration constants saved → {path}")
    except Exception as exc:
        log.warning(f"Could not save calibration JSON: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BENCHMARK LOOP
# ─────────────────────────────────────────────────────────────────────────────

def _benchmark_event_worker(payload: dict) -> dict:
    """
    Top-level (picklable) unit of work for the ProcessPoolExecutor below.
    Must be module-scope, not nested, so the `spawn` start method (required
    because JAX + fork don't mix — a forked child inherits a half-initialised
    XLA runtime) can pickle a reference to it and re-import this module
    fresh in the child process. Each event is fully independent once
    c_sigma/D/class_sigma/rtol/atol are known (all computed up front in
    run_benchmark before this stage), so running events on separate CPU
    cores is safe and turns ~n_events sequential IMEX/DOPRI5 sweeps into
    ~1 sweep's worth of wall time, bounded by whichever event needs the
    most solver steps.
    """
    rows = benchmark_event(
        event_id=payload["event_id"],
        lifecycle_class=payload["cls"],
        channels=payload["channels"],
        dem_norm=payload["dem_norm"],
        n_values=payload["n_values"],
        seeds=payload["seeds"],
        c_sigma=payload["c_sigma"],
        D=payload["D"],
        sigma_init=payload["sigma_init"],
        rtol=payload["rtol"],
        atol=payload["atol"],
    )
    return {"event_id": payload["event_id"], "cls": payload["cls"], "rows": rows}


def _log_event_progress(out: dict) -> None:
    """Shared per-event progress line for both the sequential and
    process-pool branches of the main sweep, so log output looks the same
    regardless of how many workers are running it."""
    rows = out["rows"]
    if not rows:
        return
    last = rows[-1]
    log.info(
        f"  {out['event_id']} [{out['cls']}] N={last['N']} "
        f"IMEX({last['nfe_diffusion']}D+{last['nfe_reaction']}R="
        f"{last['nfe_imex_total']}) "
        f"DOPRI5({last['nfe_dopri5']}) "
        f"ratio={last['nfe_ratio']:.2f}x "
        f"ok_imex={last['imex_converged']} "
        f"ok_dop={last['dopri5_converged']}"
    )


def run_benchmark(
        catalogue_path:      str  = CATALOGUE_PATH,
        n_events_per_class:  int  = N_EVENTS_PER_CLASS,
        n_seeds:             int  = N_SEEDS,
        n_values:            list = N_VALUES,
        run_tolerance_calibration: bool = False,
        n_workers:           Optional[int] = None,
) -> pd.DataFrame:
    """
    Main benchmark loop.

    1. Load event_catalogue.csv and the SEVIR raw catalog.
    2. Stratified sample: n_events_per_class per lifecycle class (A.6).
    3. Load channels + DEM once per sampled event.
    4. A.6 c_sigma calibration (single STEADY event @ N*).
    5. A.2 D calibration (same event @ N*).
    6. A.2 per-class sigma_init calibration (<=5 events/class @ N*).
    7. (Optional) A.3 tolerance calibration.
    8. For each event x N x seed: run IMEX and DOPRI5, record NFE.

    `n_workers`: number of OS processes to spread step 8 (the dominant
    cost — every event independently runs its own N x seed IMEX/DOPRI5
    sweep once c_sigma/D/class_sigma/rtol/atol are known) across. Defaults
    to min(cpu_count, n_events) when None. Each event is fully independent
    once calibration is done, so this turns ~n_events x T_event wall time
    into ~T_slowest_event, bounded by however many cores are available.
    Pass n_workers=1 to force the original sequential behaviour.
    """
    if not os.path.exists(catalogue_path):
        raise FileNotFoundError(
            f"event_catalogue.csv not found at {catalogue_path}.\n"
            f"Run Growth_Decay_Classify.py first."
        )

    catalogue = pd.read_csv(catalogue_path, low_memory=False)
    log.info(f"Loaded event catalogue: {len(catalogue):,} events")

    sevir_catalog_path = os.path.join(DATA_ROOT, "CATALOG.csv")
    if not os.path.exists(sevir_catalog_path):
        raise FileNotFoundError(f"SEVIR CATALOG.csv not found at {sevir_catalog_path}")

    sevir_catalog = pd.read_csv(sevir_catalog_path, low_memory=False)
    log.info(f"Loaded SEVIR catalog: {len(sevir_catalog):,} rows")

    # ── A.6: stratified sample ───────────────────────────────────────────
    sampled = (
        catalogue
        .groupby("lifecycle_class", group_keys=False)
        .apply(lambda g: g.sample(min(n_events_per_class, len(g)), random_state=42))
        .reset_index(drop=True)
    )
    log.info(f"Sampled {len(sampled)} events across "
             f"{sampled['lifecycle_class'].nunique()} classes")
    for cls, cnt in sampled["lifecycle_class"].value_counts().items():
        log.info(f"  {cls:<18} {cnt}")

    # ── A.5 steps 1-2: load channels (local HDF5, fast/CPU-bound) for every
    #     sampled event first ────────────────────────────────────────────
    channels_by_event: dict[str, dict] = {}
    class_by_event:    dict[str, str]  = {}
    for _, row in sampled.iterrows():
        event_id = str(row["id"])
        channels = load_event_multichannel(event_id, sevir_catalog)
        if channels is None:
            log.warning(f"  {event_id}: could not load channels, skipping.")
            continue
        channels_by_event[event_id] = channels
        class_by_event[event_id]    = str(row["lifecycle_class"])

    # ── A.5 step 3: fetch DEM for ALL events up front, in parallel, BEFORE
    #     any graph-build / calibration / ODE work starts. This is the fix
    #     for the DEM-fetch bottleneck — previously each event's (network-
    #     bound) DEM fetch was interleaved one-at-a-time with the channel
    #     loop above, so the whole sweep sat behind N sequential STAC
    #     round-trips before any ODE comparison could run. ──────────────
    channel_shapes = {
        eid: ch["vil"].shape[1:] for eid, ch in channels_by_event.items()
    }
    dem_by_event = prefetch_dem_for_events(
        list(channels_by_event.keys()), sevir_catalog, channel_shapes,
    )

    event_data: dict[str, dict] = {}
    for event_id, channels in channels_by_event.items():
        dem_norm = dem_by_event.get(
            event_id, np.zeros(channel_shapes[event_id], dtype=np.float32)
        )
        event_data[event_id] = {
            "lifecycle_class": class_by_event[event_id],
            "channels":        channels,
            # RAW normalised DEM in [0, 1] — NOT pre-multiplied by LAMBDA_Z.
            # TemporalSLIC_DEM (see build_graph_topological) applies
            # lambda_z internally when it builds its own 4th feature-cube
            # channel, exactly as it does in
            # Visualize_sevir_with_superpixel_fused.py, so pre-weighting
            # here would double-apply lambda_z.
            "dem_norm":        dem_norm,
        }
    log.info(f"Loaded channel+DEM data for {len(event_data)}/{len(sampled)} sampled events")
    if not event_data:
        return pd.DataFrame()

    # ── From here on, everything is CPU-bound (graph build, calibration,
    #     IMEX/DOPRI5 ODE comparison) — no further network calls. ───────

    # ── A.6: c_sigma calibration (single STEADY event @ N*=1150) ──────────
    steady_ids = [eid for eid, d in event_data.items()
                  if d["lifecycle_class"] == "STEADY"]
    calib_event_id = steady_ids[0] if steady_ids else next(iter(event_data))
    c_sigma, lambda_max_topo_ref, c_sigma_sweep = calibrate_c_sigma(
        event_data[calib_event_id]["channels"],
        event_data[calib_event_id]["dem_norm"],
    )
    log.info(f"[calibration] c_sigma = {c_sigma}  "
             f"(lambda_max_topo = {lambda_max_topo_ref:.3f}, event={calib_event_id})")
    for c, lam in c_sigma_sweep.items():
        log.info(f"    c_sigma={c:<4}  lambda_max={lam:.3f}")

    # ── A.2: D calibration (same event @ N*=1150) ──────────────────────────
    graph_nstar = build_graph_topological(
        event_data[calib_event_id]["channels"],
        event_data[calib_event_id]["dem_norm"],
        N=N_STAR, c_sigma=c_sigma,
    )
    r_stab = estimate_tsit5_stability_radius()
    k_max_nstar = _weighted_k_max(graph_nstar)
    D = calibrate_D(graph_nstar, r_stab)
    log.info(f"[calibration] r_stab(Tsit5) = {r_stab:.4f}  "
             f"k_max(N*={N_STAR}) = {k_max_nstar:.4f}  ->  D = {D:.6g}")

    # ── A.2: per-class sigma_init calibration (<=5 events/class @ N*) ──────
    class_calib_graphs: dict[str, list[dict]] = {}
    for eid, d in event_data.items():
        cls = d["lifecycle_class"]
        bucket = class_calib_graphs.setdefault(cls, [])
        if len(bucket) >= 5:
            continue
        g = build_graph_topological(d["channels"], d["dem_norm"],
                                     N=N_STAR, c_sigma=c_sigma)
        if g is not None:
            bucket.append(g)

    class_sigma: dict[str, float] = {}
    for cls, graphs in class_calib_graphs.items():
        sigma, rho = calibrate_class_sigma(cls, graphs)
        class_sigma[cls] = sigma
        target = TARGET_RHO.get(cls, float("nan"))
        log.info(f"[calibration] {cls:<15} sigma_init={sigma:.4f}  "
                 f"rho_measured={rho:.3f}  (target={target:.3f})")

        # DIAGNOSTIC (see assess_reaction_stability_away_from_h0's docstring):
        # is the h0-only calibrated rho still representative once h moves
        # away from h0, or does SiLU's non-saturating growth mean the
        # effective local stiffness keeps climbing as integration proceeds?
        # A flat profile across scales says the calibration is trustworthy
        # for the whole trajectory; a sharply rising one says Kvaerno5 will
        # face much worse stiffness mid-integration than rho_measured
        # implies — independent of tolerance/max_steps tuning.
        if graphs:
            g0 = graphs[0]
            w = make_weights(sigma_init=sigma, seed=0)
            profile = assess_reaction_stability_away_from_h0(
                w, jnp.array(g0["h0"]), jnp.array(g0["env"]),
            )
            profile_str = "  ".join(f"{k}={v:.2f}" for k, v in profile.items())
            log.info(f"    away-from-h0 rho profile: {profile_str}")

    # ── A.3 (optional, expensive): tolerance calibration ────────────────────
    # Calibrated against the STIFFEST class actually present (by TARGET_RHO),
    # not the STEADY event used for c_sigma/D above. A.3's own condition (ii)
    # ties the tolerance choice to preserving visibility of the stiffness
    # DIFFERENCE across classes ("otherwise both solvers take similar
    # numbers of steps and the stiffness difference is hidden") — a shared
    # tolerance calibrated only against a mild class (STEADY, rho~0.3) will
    # converge easily and tell you nothing about whether that same
    # tolerance is workable for e.g. RAPID_GROWTH (rho~33, ~100x stiffer).
    # Since blocks.tex A.3 mandates ONE shared tolerance for the whole
    # benchmark, the binding constraint is whichever class is hardest, so
    # that is what must drive the calibration.
    rtol, atol = RTOL_DEFAULT, ATOL_DEFAULT
    if run_tolerance_calibration:
        stiffest_cls = max(class_sigma, key=lambda c: TARGET_RHO.get(c, 0.0))
        stiffest_graphs = class_calib_graphs.get(stiffest_cls, [])
        calib_graph = stiffest_graphs[0] if stiffest_graphs else graph_nstar
        h0_jax  = jnp.array(calib_graph["h0"])
        env_jax = jnp.array(calib_graph["env"])
        weights = make_weights(class_sigma.get(stiffest_cls, 1.0), seed=0)
        diff_rhs = make_diffusion_rhs(calib_graph["senders"], calib_graph["receivers"],
                                       calib_graph["edge_weights"], D)
        rxn_rhs = make_reaction_rhs(env_jax, weights)
        log.info(f"[calibration] A.3 tolerance: calibrating against "
                 f"{stiffest_cls} (stiffest class present, "
                 f"target_rho={TARGET_RHO.get(stiffest_cls, float('nan')):.3f}) …")
        rtol, atol = calibrate_tolerance(h0_jax, diff_rhs, rxn_rhs, label=stiffest_cls)
        log.info(f"[calibration] A.3 tolerance -> rtol={rtol:.2e}  atol={atol:.2e}")

    _save_calibration_json({
        "c_sigma":              c_sigma,
        "lambda_max_topo_ref":  lambda_max_topo_ref,
        "c_sigma_sweep":        c_sigma_sweep,
        "r_stab_tsit5":         r_stab,
        "k_max_at_N_star":      k_max_nstar,
        "D":                    D,
        "N_star":                N_STAR,
        "class_sigma_init":     class_sigma,
        "rtol":                 rtol,
        "atol":                 atol,
        "calibration_event_id": calib_event_id,
    })

    # ── A.6 main sweep: event x N x seed ─────────────────────────────────
    # This is the dominant cost of the whole benchmark — every event
    # independently runs len(n_values) x n_seeds IMEX-vs-DOPRI5 comparisons
    # — and events are fully independent of each other once c_sigma/D/
    # class_sigma/rtol/atol are fixed above, so it's the stage worth
    # spreading across CPU cores rather than running ~n_events x T_event
    # sequentially.
    seeds = list(range(n_seeds))
    payloads = [
        {
            "event_id":   event_id,
            "cls":        d["lifecycle_class"],
            "channels":   d["channels"],
            "dem_norm":   d["dem_norm"],
            "n_values":   n_values,
            "seeds":      seeds,
            "c_sigma":    c_sigma,
            "D":          D,
            "sigma_init": class_sigma.get(d["lifecycle_class"], 1.0),
            "rtol":       rtol,
            "atol":       atol,
        }
        for event_id, d in event_data.items()
    ]

    resolved_workers = max(1, min(n_workers or (os.cpu_count() or 1), len(payloads)))
    outputs: dict[str, dict] = {}

    if resolved_workers <= 1:
        log.info("Running main sweep sequentially (n_workers=1) …")
        for payload in tqdm(payloads, desc="Events"):
            out = _benchmark_event_worker(payload)
            outputs[out["event_id"]] = out
            _log_event_progress(out)
    else:
        log.info(f"Running main sweep across {resolved_workers} worker "
                 f"process(es) for {len(payloads)} event(s) …")
        ctx = mp.get_context("spawn")   # JAX + fork don't mix; spawn is required
        with ProcessPoolExecutor(max_workers=resolved_workers, mp_context=ctx) as pool:
            futures = {pool.submit(_benchmark_event_worker, p): p["event_id"]
                       for p in payloads}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Events"):
                event_id = futures[fut]
                try:
                    out = fut.result()
                except Exception as exc:
                    log.error(f"  [ERROR] {event_id}: worker failed: {exc}")
                    continue
                outputs[out["event_id"]] = out
                _log_event_progress(out)

    # Preserve event_data's original iteration order in the final rows,
    # regardless of which worker happened to finish first.
    all_rows = []
    for event_id in event_data:
        if event_id in outputs:
            all_rows.extend(outputs[event_id]["rows"])

    return pd.DataFrame(all_rows)


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY AGGREGATION  (for paper figures)
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate results to per-class x N statistics for Figures A-1..A-4 and
    the A.8 Joint Score weight-ordering validation.
    """
    agg = (
        df.groupby(["lifecycle_class", "N"])
        .agg(
            nfe_diffusion_mean    = ("nfe_diffusion",       "mean"),
            nfe_diffusion_std     = ("nfe_diffusion",       "std"),
            nfe_reaction_mean     = ("nfe_reaction",        "mean"),
            nfe_reaction_std      = ("nfe_reaction",        "std"),
            nfe_imex_total_mean   = ("nfe_imex_total",      "mean"),
            nfe_imex_total_std    = ("nfe_imex_total",      "std"),
            nfe_dopri5_mean       = ("nfe_dopri5",          "mean"),
            nfe_dopri5_std        = ("nfe_dopri5",          "std"),
            nfe_ratio_mean        = ("nfe_ratio",           "mean"),
            nfe_ratio_std         = ("nfe_ratio",           "std"),
            wall_imex_mean        = ("wall_imex_sec",       "mean"),
            wall_imex_std         = ("wall_imex_sec",       "std"),
            wall_dopri5_mean      = ("wall_dopri5_sec",     "mean"),
            wall_dopri5_std       = ("wall_dopri5_sec",     "std"),
            dopri5_converged_frac = ("dopri5_converged",    "mean"),
            imex_converged_frac   = ("imex_converged",      "mean"),
            l2_error_mean         = ("solution_l2_error",   "mean"),
            lambda_max_topo_mean  = ("lambda_max_topo",     "mean"),
            lambda_max_topo_std   = ("lambda_max_topo",     "std"),
            lambda_max_w_mean     = ("lambda_max_weighted", "mean"),
            lambda_max_w_std      = ("lambda_max_weighted", "std"),
            rejected_diff_mean    = ("rejected_diff",       "mean"),
            rejected_rxn_mean     = ("rejected_rxn",        "mean"),
            n_trials              = ("nfe_imex_total",      "count"),
        )
        .reset_index()
    )
    # A.8: Joint Score weight-ordering validation fractions.
    agg["reaction_fraction"]  = agg["nfe_reaction_mean"] / agg["nfe_imex_total_mean"]
    agg["diffusion_fraction"] = agg["nfe_diffusion_mean"] / agg["nfe_imex_total_mean"]
    return agg


def print_summary_table(summary: pd.DataFrame) -> None:
    """Print a compact table of key results: IMEX vs DOPRI5 at N=500 and N=N*."""
    key_Ns = [500, N_STAR]
    print("\n" + "=" * 90)
    print(f"  Block A Summary: IMEX vs DOPRI5 NFE at N=500 (ISV elbow) "
          f"and N={N_STAR} (global N*)")
    print("=" * 90)
    header = (f"{'Class':<18}  {'N':>5}  "
              f"{'NFE_D':>7}  {'NFE_R':>7}  {'NFE_IMEX':>9}  "
              f"{'NFE_DOP':>9}  {'ratio':>6}  "
              f"{'DOP_cvg':>8}  {'IMEX_cvg':>9}")
    print(header)
    print("-" * 90)

    for cls in sorted(summary["lifecycle_class"].unique()):
        for N in key_Ns:
            row = summary[
                (summary["lifecycle_class"] == cls) & (summary["N"] == N)
            ]
            if row.empty:
                continue
            r = row.iloc[0]
            print(
                f"{cls:<18}  {N:>5}  "
                f"{r['nfe_diffusion_mean']:>7.0f}  "
                f"{r['nfe_reaction_mean']:>7.0f}  "
                f"{r['nfe_imex_total_mean']:>9.0f}  "
                f"{r['nfe_dopri5_mean']:>9.0f}  "
                f"{r['nfe_ratio_mean']:>6.2f}x  "
                f"{r['dopri5_converged_frac']:>8.2%}  "
                f"{r['imex_converged_frac']:>9.2%}"
            )
        print()
    print("=" * 90 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Block A: IMEX vs DOPRI5 solver benchmark on SEVIR superpixel graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--n_events", type=int, default=N_EVENTS_PER_CLASS,
                        help=f"Events per class (default: {N_EVENTS_PER_CLASS})")
    parser.add_argument("--seeds",    type=int, default=N_SEEDS,
                        help=f"Random seeds per (event, N) (default: {N_SEEDS})")
    parser.add_argument("--n_values", type=int, nargs="+", default=N_VALUES,
                        help=f"N sweep values (default: {N_VALUES})")
    parser.add_argument("--out",      type=str, default=OUTPUT_CSV,
                        help="Output CSV path for raw results")
    parser.add_argument("--summary",  type=str, default=SUMMARY_CSV,
                        help="Output CSV path for aggregated summary")
    parser.add_argument("--calibrate-tolerance", action="store_true",
                        help="Also run the (expensive) A.3 tolerance-selection sweep")
    parser.add_argument("--workers", type=int, default=None,
                        help="Worker processes for the main event sweep "
                             "(default: min(cpu_count, n_events); pass 1 "
                             "to force sequential execution)")
    args = parser.parse_args()

    if not JAX_OK:
        sys.exit(1)

    log.info(f"JAX devices: {jax.devices()}")
    log.info(f"N sweep: {args.n_values}")
    log.info(f"Events per class: {args.n_events}  |  Seeds: {args.seeds}")

    # ── Run benchmark ─────────────────────────────────────────────────────────
    results_df = run_benchmark(
        catalogue_path=CATALOGUE_PATH,
        n_events_per_class=args.n_events,
        n_seeds=args.seeds,
        n_values=args.n_values,
        run_tolerance_calibration=args.calibrate_tolerance,
        n_workers=args.workers,
    )

    if results_df.empty:
        log.error("No results produced — check data paths and HDF5 files.")
        sys.exit(1)

    # ── Save raw results ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    results_df.to_csv(args.out, index=False)
    log.info(f"Raw results saved → {args.out}  ({len(results_df):,} rows)")

    # ── Aggregate and summarise ───────────────────────────────────────────────
    summary_df = build_summary(results_df)
    summary_df.to_csv(args.summary, index=False)
    log.info(f"Summary saved  → {args.summary}")

    print_summary_table(summary_df)

    # ── A.8: Joint Score weight-ordering validation ────────────────────────────
    # alpha_R=0.10, alpha_D=0.60, alpha_T=0.30 under IMEX-ARK.
    for cls in sorted(summary_df["lifecycle_class"].unique()):
        for N in [N_STAR, 1500]:
            row = summary_df[
                (summary_df["lifecycle_class"] == cls) & (summary_df["N"] == N)
            ]
            if row.empty:
                continue
            r = row.iloc[0]
            if cls == "RAPID_GROWTH" and N >= 1000:
                ok = r["reaction_fraction"] < 0.20
                tag = "OK" if ok else "CHECK"
                log.info(f"  [A.8 {tag}] {cls} N={N}: "
                         f"reaction_fraction={r['reaction_fraction']:.2f} "
                         f"(want < 0.20, consistent with alpha_R=0.10)")
            if N >= N_STAR:
                ok = r["diffusion_fraction"] > 0.60
                tag = "OK" if ok else "CHECK"
                log.info(f"  [A.8 {tag}] {cls} N={N}: "
                         f"diffusion_fraction={r['diffusion_fraction']:.2f} "
                         f"(want > 0.60, consistent with alpha_D=0.60)")

    # ── Key convergence checks ──────────────────────────────────────────────
    for cls in ["RAPID_GROWTH", "GROWTH_DECAY", "STEADY", "QUIESCENT"]:
        for N in [500, N_STAR]:
            row = summary_df[
                (summary_df["lifecycle_class"] == cls) & (summary_df["N"] == N)
            ]
            if row.empty:
                continue
            r = row.iloc[0]
            dop_cvg = r["dopri5_converged_frac"]
            imex_cvg = r["imex_converged_frac"]
            if cls in ["RAPID_GROWTH", "GROWTH_DECAY"] and N >= 750:
                if dop_cvg < 0.5:
                    log.info(f"  [EXPECTED] {cls} N={N}: "
                             f"DOPRI5 converged only {dop_cvg:.0%} — reaction stiffness.")
                else:
                    log.warning(f"  [CHECK] {cls} N={N}: DOPRI5 converged {dop_cvg:.0%} "
                                f"— sigma_init calibration may need revisiting.")
            if imex_cvg < 1.0:
                log.warning(f"  [UNEXPECTED] {cls} N={N}: "
                            f"IMEX only {imex_cvg:.0%} convergence — check max_steps.")

    log.info("Block A benchmark complete.")


if __name__ == "__main__":
    main()