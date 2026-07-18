#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────
# blockA_sweep / calibrate.cmd
# One-time calibration job. Runs steps 1-7 of run_benchmark() ONLY
# (c_sigma, D, per-class sigma_init, tolerance) on the full --all-events
# catalogue and writes block_a_calibration.json. Every N-value job in
# sweep_array.cmd loads this same JSON with --calib-json instead of
# recomputing it, so calibration constants are identical for every N.
#
# IMPORTANT — network access: on first run (empty dem_cache/), this job
# also fetches Copernicus DEM tiles + land-cover via the Planetary
# Computer STAC API for EVERY event in event_catalogue.csv (see
# prefetch_dem_for_events / prefetch_landtype_for_events), which needs
# internet access. Aqua's compute nodes are very likely firewalled off
# from the open internet. Two ways to handle this — confirm which one
# applies on Aqua via the Cluster FAQ / HPCE support before submitting:
#   (a) if there is a designated "data transfer" / internet-enabled queue,
#       point #PBS -q at it for THIS job only; sweep_array.cmd and
#       merge.cmd don't need internet and can stay on the compute queue.
#   (b) otherwise, warm DEM_CACHE_DIR / CAPE_CACHE_DIR / LANDTYPE_CACHE_DIR
#       locally (or on the login node, briefly, per its usage policy)
#       BEFORE submitting, then rsync those three folders to the same
#       DATA_ROOT-relative paths on Aqua — this job (and every array task)
#       will then only ever read from local disk cache.
# ─────────────────────────────────────────────────────────────────────────

#PBS -N blockA_calib
#PBS -q <QUEUE_NAME>
#PBS -P <PROJECT_NAME>
#PBS -l select=1:ncpus=20
#PBS -l walltime=06:00:00
#PBS -j oe

cd "$PBS_O_WORKDIR" || exit 1
exec > "blockA_calib.log" 2>&1

echo "=== blockA_calib starting on $(hostname) at $(date) ==="

module purge
# Confirm exact module name/version with: module avail anaconda
module load anaconda3
source activate blockA_env   # see README: conda env w/ jax[cpu] diffrax equinox ...

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

NCPUS_REQ=${NCPUS:-20}

python block_a_solver_benchmark.py \
    --all-events \
    --calibrate-only \
    --calib-json "$PBS_O_WORKDIR/block_a_calibration.json" \
    --workers "$NCPUS_REQ"

echo "=== blockA_calib finished at $(date) ==="
