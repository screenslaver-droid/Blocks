#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────
# blockA_sweep / sweep_array_b3.cmd
# Block B3 (integration accuracy vs N). Same 40-way array as Block A's
# sweep_array.cmd -- N = (index + 1) * 50, index 0..39 -- and --all-events,
# so B3 covers the IDENTICAL event set Block A did.
#
# "Same randomly initialised weight matrices and graph construction as
# Block A" is not something this job configures -- it's structural in
# block_b3_integration_accuracy.py itself: B3 imports block_a_solver_
# benchmark.py and calls bA.build_graph_topological(..., c_sigma=...) and
# bA.make_weights(sigma_init, seed) with the SAME c_sigma/D/sigma_init read
# from block_a_calibration.json and the SAME seed convention (range(n_seeds))
# Block A used -- see that script's module docstring. What THIS job has to
# get right is just: point --calibration-json at the exact
# block_a_calibration.json Block A's calibrate.cmd produced, and sweep the
# same N values -- both handled below.
#
# Depends on Block A's full chain (calibrate.cmd -> sweep_array.cmd ->
# merge.cmd) having already finished successfully -- submit_all_b3.sh wires
# this dependency in for you; don't run this by hand until
# block_a_calibration.json exists.
# ─────────────────────────────────────────────────────────────────────────

#PBS -N blockB3_sweep
#PBS -q <QUEUE_NAME>
#PBS -P <PROJECT_NAME>
#PBS -l select=1:ncpus=40
#PBS -l walltime=24:00:00
#PBS -J 0-39
#PBS -j oe

cd "$PBS_O_WORKDIR" || exit 1

N=$(( (PBS_ARRAY_INDEX + 1) * 50 ))
TAG=$(printf "N%04d" "$N")

exec > "blockB3_sweep_${TAG}.log" 2>&1
echo "=== blockB3_sweep ${TAG} (array index ${PBS_ARRAY_INDEX}) starting on $(hostname) at $(date) ==="

module purge
module load anaconda3
source activate blockA_env   # same env as Block A -- B3 imports it directly

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

NCPUS_REQ=${NCPUS:-40}

CALIB_JSON="$PBS_O_WORKDIR/block_a_calibration.json"
if [ ! -f "$CALIB_JSON" ]; then
    echo "ERROR: $CALIB_JSON not found — Block A must finish (calibrate.cmd) first." >&2
    exit 1
fi

python block_b3_integration_accuracy.py \
    --all-events \
    --n_values "$N" \
    --seeds 5 \
    --calibration-json "$CALIB_JSON" \
    --output         "$PBS_O_WORKDIR/block_b3_results_${TAG}.csv" \
    --summary-output "$PBS_O_WORKDIR/block_b3_summary_${TAG}.csv" \
    --n_workers "$NCPUS_REQ"

echo "=== blockB3_sweep ${TAG} finished at $(date) ==="
