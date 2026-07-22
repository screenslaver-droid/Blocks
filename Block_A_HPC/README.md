# Block A solver benchmark — Aqua (IITM HPCE) full sweep

Runs `block_a_solver_benchmark.py` over **every SEVIR event** in
`event_catalogue.csv`, for **N ∈ {50, 100, …, 2000}** (40 values), on the
Aqua CPU cluster, via a 3-stage PBS job chain:

| stage | script | what it does | cost |
|---|---|---|---|
| 1 | `calibrate.cmd` | computes `c_sigma`, `D`, per-class `sigma_init`, `rtol`/`atol` **once**, on `--all-events`, and warms the DEM/CAPE/land-type disk caches | 1 job |
| 2 | `sweep_array.cmd` | PBS array job, 40 subjobs (`-J 0-39`), one per N value, each running the full all-events sweep for that single N, reusing stage 1's calibration | 40 jobs |
| 3 | `merge.cmd` | concatenates the 40 per-N CSVs into one `block_a_results.csv` / `block_a_summary.csv` | 1 job |

`submit_all.sh` submits all three with PBS dependencies so the chain runs
unattended (`afterok`, `afterokarray`).

## Why split this way, not by event

Calibration (`c_sigma`/`D`/`sigma_init`) only depends on `N*=1150` and a
fixed calibration event — it's the same regardless of which N you're
sweeping. Recomputing it inside every job would be wasted, purely-serial
wall time. Splitting by **N** instead of by event chunk means:
- each of the 40 array subjobs is `~1/40` of the total wall time,
- within each subjob, the script's existing `ProcessPoolExecutor` (see
  `run_benchmark`'s `n_workers`/`--workers`) still parallelises across
  **events** on that node's cores — the two parallelism axes (PBS array
  over N, process pool over events) don't conflict.

I added a thread-pinning guard (`OMP_NUM_THREADS=1` etc., set both in the
script and the PBS scripts) so the `--workers` process pool is the *only*
source of parallelism per node — otherwise each worker process's own
BLAS/XLA backend would also try to grab every core and you'd oversubscribe
the node 40×.

## Required edits before submitting (do these first)

1. **`DATA_ROOT`** at the top of `block_a_solver_benchmark.py` is currently
   a local path (`/media/sid_nair/...`). Change it to wherever you rsync
   your SEVIR data + `event_catalogue.csv` + `CATALOG.csv` on Aqua.
2. **`<QUEUE_NAME>` / `<PROJECT_NAME>`** placeholders in `calibrate.cmd`,
   `sweep_array.cmd`, `merge.cmd` — get the real values from
   `qstat -Q` and your HPCE account page (I couldn't fetch the
   login-gated `usingintelcompilers` / `clusterfaq` pages myself, so
   double check queue name, walltime ceilings, and `#PBS -l select`
   syntax against those pages or `man qsub` once you're on Aqua).
3. **`ncpus` per node** — I used 40 in `sweep_array.cmd` as a placeholder;
   set it to Aqua's actual cores-per-node (check `pbsnodes -a` or the
   Software Stack page) and pass the same number via `--workers`.
4. **Conda env** (`blockA_env`) — create once on Aqua:
   ```bash
   module load anaconda3          # exact name via `module avail anaconda`
   conda create -n blockA_env python=3.11 -y
   conda activate blockA_env
   pip install "jax[cpu]" diffrax equinox
   pip install scikit-image scipy tqdm h5py pandas numpy opencv-python
   pip install cartopy pystac_client planetary_computer rioxarray   # DEM fetch
   ```
5. **Internet access for DEM/CAPE/land-type fetch** — `calibrate.cmd` runs
   with `--all-events`, so on a cold cache it will try to fetch
   Copernicus DEM tiles for every event via the Planetary Computer STAC
   API. Aqua's compute nodes very likely have no outbound internet.
   Either point `calibrate.cmd` at a queue that does have it, or warm
   `dem_cache/` / `cape_cache/` / `landtype_cache/` yourself beforehand
   (locally, or via whatever data-transfer node Aqua provides) and
   `rsync` them alongside the SEVIR data — every job after that only
   reads from disk. See `calibrate.cmd`'s header comment.

## Layout (flat, per HPCE's "avoid nested subdirectories" guidance)

```
$HOME/blockA_sweep/
├── block_a_solver_benchmark.py
├── block_b3_integration_accuracy.py
├── calibrate.cmd
├── sweep_array.cmd
├── merge.cmd
├── merge_results.py
├── plot_results.py
├── prefetch_offline_caches.py
├── submit_all.sh
├── sweep_array_b3.cmd
├── merge_b3.cmd
├── merge_results_b3.py
├── submit_all_b3.sh
└── README.md
```

## Path reference — what each config variable expects on disk

Everything hangs off one root, `DATA_ROOT` (top of `block_a_solver_benchmark.py`).
Nothing else needs editing if this one is set correctly — every other path
below is `os.path.join(DATA_ROOT, ...)`.

```
$SCRATCH/blockA_data/                       <- set DATA_ROOT to this
├── CATALOG.csv                             <- official SEVIR catalog (unmodified)
├── event_catalogue.csv                     <- YOUR catalogue (id, lifecycle_class, ...)
├── vil/
│   ├── 2018/
│   │   └── SEVIR_VIL_STORMEVENTS_2018_0101_0630.h5
│   │   └── SEVIR_VIL_STORMEVENTS_2018_0701_1231.h5
│   └── 2019/ ...
├── ir069/
│   └── 2018/ SEVIR_IR069_STORMEVENTS_2018_....h5  ...
├── ir107/
│   └── 2018/ SEVIR_IR107_STORMEVENTS_2018_....h5  ...
├── dem_cache/                              <- auto-created, see below
├── cape_cache/                             <- must be pre-populated once, see below
├── landtype_cache/                         <- must be pre-populated once, see below
├── block_a_calibration.json                <- written by calibrate.cmd
└── block_a_results.csv / block_a_summary.csv   <- final merged output
```

### `CATALOGUE_PATH` → `event_catalogue.csv`

**Your** project-specific catalogue, not part of the official SEVIR
release — it's what `Growth_Decay_Classify.py` (mentioned in the script's
docstring, not part of this deliverable) is expected to produce. The
script only reads two columns from it directly:
- `id` — SEVIR event id, must match `CATALOG.csv`'s `id` column
- `lifecycle_class` — one of `RAPID_GROWTH / GROWTH_DECAY / EPISODIC /
  PLATEAU / RAPID_DECAY / STEADY / QUIESCENT` (the keys of `TARGET_RHO`
  near the top of the script)

If this file doesn't exist yet, `run_benchmark()` raises
`FileNotFoundError` immediately with the exact path it looked for — you
need this in place (however you generate it) before submitting anything.

### `CATALOG.csv` (hardcoded as `DATA_ROOT/CATALOG.csv`, not configurable via CLI)

The **official, unmodified SEVIR catalog** — download it from the SEVIR
release, don't hand-build it. Columns the script actually reads:
`id`, `img_type` (`vil`/`ir069`/`ir107`), `file_name`, and
`llcrnrlon`/`urcrnrlon`/`llcrnrlat`/`urcrnrlat` (event bounding box in
degrees — used for the DEM/CAPE/land-type extent lookups). If your copy
has projected-metre extents instead of degrees here, the DEM-fetch code
detects the resulting absurd tile count and errors out with a hint to
check this.

### SEVIR `.h5` files — where `file_name` actually resolves to

`CATALOG.csv`'s `file_name` column holds a **catalog-relative** path like
`vil/2018/SEVIR_VIL_STORMEVENTS_2018_0101_0630.h5`. `get_local_path()`
tries, in order:
1. `DATA_ROOT/<file_name as-is>` → i.e. `DATA_ROOT/vil/2018/....h5`
   (the standard SEVIR release layout — use this one)
2. if that's missing AND the path has exactly 3 segments, it tries the
   first two segments **swapped**: `DATA_ROOT/2018/vil/....h5` — a
   fallback for people who reorganised by year first. You don't need to
   pick one on purpose; just don't mix both layouts for the same file.

If an event's `vil` file can't be resolved this way, that event is
skipped with a warning (`could not load channels, skipping`) — worth a
`grep WARNING` over the calibration job's log before trusting a full
sweep ran on the event count you expected.

### `DEM_CACHE_DIR` → `DATA_ROOT/dem_cache/`

Fully auto-managed — `os.makedirs(..., exist_ok=True)` creates it, and
`prefetch_dem_for_events()` populates it. One `.npy` file per **unique
rounded bounding box**, not per event (many events share an extent), named
`dem_<lon_min>_<lon_max>_<lat_min>_<lat_max>.npy` (minus signs become
`m`, e.g. `dem_m100.1234_m95.0000_30.0000_35.5000.npy`). This naming is
deliberately identical to the standalone `prefetch_dem.py` script
mentioned in the docstring, so if you warm this folder some other way
(e.g. running that script on a machine with internet, then `rsync`-ing
`dem_cache/` to Aqua), this pipeline will pick it up with zero extra
config — it just needs the directory to exist at this exact path with
files named this way.

### `CAPE_CACHE_DIR` → `DATA_ROOT/cape_cache/`

**Not auto-fetched during the sweep** — `load_cape_for_event()` only
*reads* `cape_clim_<year>_<month:02d>.npy` (e.g. `cape_clim_2019_07.npy`),
built once, beforehand, by `prefetch_offline_caches.py` (see below). If
this directory is empty when the sweep runs, CAPE defaults to zeros
rather than failing, so a silently-empty cache is easy to miss — check it
has 12 files (`cape_clim_2019_01.npy` ... `_12.npy`) before trusting
results that depend on the CAPE channel.

### `LANDTYPE_CACHE_DIR` → `DATA_ROOT/landtype_cache/`

Same pattern as CAPE: a one-time, whole-CONUS precompute
(`prefetch_offline_caches.py` again), producing a single
`landtype_<year>.npy`, which `prefetch_landtype_for_events()` then reads
from at sweep time as a fast path. Unlike CAPE, land-type also has a
genuine per-event fallback fetch if this cache is cold (see
`prefetch_landtype_for_events`'s docstring) — slower, but it won't
silently zero out like CAPE does.

### Populating `cape_cache/` and `landtype_cache/`: `prefetch_offline_caches.py`

Run this **once**, on whatever machine actually has outbound internet
(almost certainly not Aqua's compute nodes — see `calibrate.cmd`'s header
comment):

```bash
pip install cdsapi xarray netCDF4                       # for CAPE
pip install pystac-client planetary-computer rioxarray   # for land-type
# ~/.cdsapirc must exist first: https://cds.climate.copernicus.eu/how-to-api

python prefetch_offline_caches.py                        # both caches, CAPE_YEAR
python prefetch_offline_caches.py --skip-cape             # land-type only
python prefetch_offline_caches.py --skip-landtype         # CAPE only
python prefetch_offline_caches.py --year 2019 --data-root /path/to/blockA_data
```

It imports `build_era5_cape_climatology` / `build_modis_landtype_grid`
straight from `block_a_solver_benchmark.py` (no reimplementation, so the
cache it produces can't drift from what the main script itself would
build), reports how many of the 12 monthly CAPE files landed, and prints
the exact `rsync` commands to copy both folders onto the cluster
afterward. CAPE resumes cleanly if interrupted — already-cached months
are skipped on re-run, so just re-run the same command if the CDS API
times out partway through.



### Outputs

- `CALIB_JSON` (`block_a_calibration.json`) — written once by
  `calibrate.cmd`, read by all 40 `sweep_array.cmd` subjobs via
  `--calib-json`.
- `OUTPUT_CSV`/`SUMMARY_CSV` — I overrode these per-N in `sweep_array.cmd`
  (`block_a_results_N####.csv`) precisely so 40 concurrent subjobs never
  write the same file; `merge.cmd` reassembles the canonical
  `block_a_results.csv`/`block_a_summary.csv` names at the end.

### Practical setup order on Aqua

1. `rsync` `CATALOG.csv`, `event_catalogue.csv`, and the `vil/ir069/ir107`
   HDF5 tree to `DATA_ROOT` on Aqua's `/scratch` (not `/home` — SEVIR is
   large; check Aqua's storage quota policy on the Cluster FAQ page).
2. Build `cape_cache/` and `landtype_cache/` **once**, anywhere with
   internet, via `prefetch_offline_caches.py`, then `rsync` them over too
   (or warm them from a queue that has internet, if Aqua has one — see
   the note in `calibrate.cmd`).
3. Point `DATA_ROOT` at that Aqua path, then run `submit_all.sh`.



## Plots

`merge.cmd` only produces the two CSVs. To turn `block_a_summary.csv` into
figures, run `plot_results.py` afterward (anywhere — it only needs
pandas + matplotlib, not the jax/diffrax/DEM stack, so this is fine to
run on your laptop against a copied-down `block_a_summary.csv` rather
than on the cluster):

```bash
python plot_results.py --summary block_a_summary.csv --out-dir figures
python plot_results.py --log-y                          # NFE/wall-time span
                                                          # orders of magnitude
                                                          # across N=50..2000
python plot_results.py --classes RAPID_GROWTH STEADY     # subset of classes
```

Produces four small-multiple panels (one subplot per `lifecycle_class`),
mean +/- std shaded band vs N:
- `fig_nfe_vs_N.png` — IMEX total vs DOPRI5 NFE (the main comparison)
- `fig_wall_time_vs_N.png` — wall-clock seconds, both solvers
- `fig_nfe_ratio_vs_N.png` — DOPRI5/IMEX NFE ratio
- `fig_lambda_max_vs_N.png` — lambda_max(L), unit-weighted vs D-weighted
  (blocks.tex Figure A-4)

A note on reading `fig_nfe_ratio_vs_N.png` and `fig_nfe_vs_N.png` in
particular: `nfe_reaction` (and therefore `nfe_imex_total`/`nfe_ratio`)
is a *nominal stage-count proxy* for Kvaerno5, not a measured count —
see the "How accurate is the NFE proxy" discussion earlier in this
conversation. Rejected steps also aren't folded into the NFE columns at
all, only logged separately (`rejected_diff`/`rejected_rxn` /
`rejected_dopri5`). Treat NFE trends as directional, and cross-check
against `fig_wall_time_vs_N.png` (a real measurement) where the two
disagree.

## Running it

```bash
cd $HOME/blockA_sweep
bash submit_all.sh
qstat -an                 # watch progress
```

Cancel everything this run submitted:
```bash
qdel $(cat .blockA_jobids)
```

Outputs, once `merge.cmd` finishes: `block_a_results.csv` (one row per
event × N × seed, ~851 events × 40 N × 5 seeds if the full catalogue is
that size) and `block_a_summary.csv` (per-class × N aggregates).

## Block B3 (integration accuracy vs N) — run after Block A finishes

`block_b3_integration_accuracy.py` imports `block_a_solver_benchmark.py`
as a library rather than reimplementing anything, so "same randomly
initialised weight matrices and graph construction as Block A" isn't a
job-config choice — it's already structural in the script itself:

- Graph: `bA.build_graph_topological(..., c_sigma=...)` called with the
  **same `c_sigma`/`D`** read from `block_a_calibration.json` — the exact
  file Block A's `calibrate.cmd` produces. B3 never recalibrates.
- Weights: `bA.make_weights(sigma_init, seed)` called with the **same
  `sigma_init`** (per class, from that same JSON) and the **same seed
  convention** (`range(n_seeds)`) Block A used — `make_weights` is a pure
  function of `(sigma_init, seed)`, so this is bit-identical, not
  "reproduced."

What I *did* have to add for the cluster sweep specifically:
- `--all-events` (mirroring Block A's own flag) — without it, B3 defaults
  to a 5/class stratified subsample, which wouldn't cover the same event
  set Block A's `--all-events` sweep did.
- The same thread-pinning guard as Block A (`OMP_NUM_THREADS=1` etc.) —
  duplicated at the top of `block_b3_integration_accuracy.py` itself
  rather than relying on Block A's copy, since this file imports `numpy`
  directly before it imports `block_a_solver_benchmark`, so Block A's
  guard would fire one import too late to affect numpy's own BLAS backend
  otherwise.

**Files**: `sweep_array_b3.cmd` (same 40-way `N = (index+1)*50` array as
Block A's `sweep_array.cmd`, `--calibration-json` pointed at Block A's
output), `merge_b3.cmd` + `merge_results_b3.py` (concatenates the 40
per-N CSVs, rebuilds `block_b3_summary.csv` via `build_summary_b3()`
over the *full* merged table, and re-runs the elbow heuristic into
`block_b3_elbow.csv`), and `submit_all_b3.sh` (checks
`block_a_calibration.json` exists before submitting anything).

```bash
cd $HOME/blockA_sweep
bash submit_all.sh      # Block A — wait for this to fully finish
bash submit_all_b3.sh   # Block B3 — only after block_a_calibration.json exists
```

One asymmetry worth knowing about: unlike `merge_results.py` (Block A),
`merge_results_b3.py` imports `block_b3_integration_accuracy.py`, which
hard-exits at import time if JAX/Diffrax aren't importable (that's
`block_b3_integration_accuracy.py`'s own existing behavior, not something
I added) — so `merge_b3.cmd` needs the same `blockA_env` conda
environment as the sweep itself, not a bare pandas environment.

Outputs, once `merge_b3.cmd` finishes: `block_b3_results.csv` (one row
per event × N × seed: `eps_int`, `nfe_ref`, `nfe_train`,
`nfe_ratio_train_over_ref`, wall times, convergence flags),
`block_b3_summary.csv` (per-class × N mean/std), and `block_b3_elbow.csv`
(the class-relative elbow heuristic — flagged in that script's own
docstring as a stand-in for Table 5.6's real per-class N_LFS^c ceiling,
not a substitute for it).



The docs you linked also cover GPU (`cudagpu`) job scripts, but I didn't
build a GPU path here: this workload's parallelism is *already* structured
as many independent small per-event ODE solves distributed across CPU
processes (`ProcessPoolExecutor`), not one big batched array op. Running
it on a single GPU would mean one JAX process per GPU with no equivalent
of `--workers` to spread events across — you'd need to rewrite the event
loop as a `jax.vmap`/`pmap` over events to actually benefit, which is a
real (and separate) restructuring, not a config change. The CPU array-job
approach above is the better fit for the code as it stands.

## Estimating whether 40 array jobs is affordable

Before submitting the full chain, it's worth timing one (event, N, seed)
IMEX+DOPRI5 pair locally, e.g.:
```bash
python block_a_solver_benchmark.py --n_events 1 --seeds 1 --n_values 2000 --workers 1
```
and multiplying by (n_events_total × 5 seeds), divided by your `--workers`
count, to sanity check `walltime` in `sweep_array.cmd` before the cluster
burns a full array on a timeout.