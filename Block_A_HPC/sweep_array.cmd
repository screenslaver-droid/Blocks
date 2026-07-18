#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────
# blockA_sweep / sweep_array.cmd
# PBS array job: 40 subjobs, one per N in {50, 100, ..., 2000}.
# PBS_ARRAY_INDEX runs 0..39 -> N = (index + 1) * 50.
# Each subjob runs the FULL --all-events sweep for its single N value,
# reusing the calibration produced by calibrate.cmd (--calib-json), and
# writes its own block_a_results_N####.csv / block_a_summary_N####.csv so
# subjobs never touch the same output file. merge.cmd concatenates them
# afterward.
#
# Submit only after calibrate.cmd has finished successfully — see
# submit_all.sh, which wires this up with -W depend=afterok:<calib_jobid>.
# ─────────────────────────────────────────────────────────────────────────

#PBS -N blockA_sweep
#PBS -q workq
#PBS -l select=1:ncpus=40
#PBS -l walltime=24:00:00
#PBS -J 0-39
#PBS -j oe

cd "$PBS_O_WORKDIR" || exit 1

N=$(( (PBS_ARRAY_INDEX + 1) * 50 ))
TAG=$(printf "N%04d" "$N")

exec > "blockA_sweep_${TAG}.log" 2>&1
echo "=== blockA_sweep ${TAG} (array index ${PBS_ARRAY_INDEX}) starting on $(hostname) at $(date) ==="

module purge
module load anaconda3
source activate blockA_env

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

NCPUS_REQ=${NCPUS:-40}

CALIB_JSON="$PBS_O_WORKDIR/block_a_calibration.json"
if [ ! -f "$CALIB_JSON" ]; then
    echo "ERROR: $CALIB_JSON not found — run calibrate.cmd first." >&2
    exit 1
fi

python block_a_solver_benchmark.py \
    --all-events \
    --n_values "$N" \
    --seeds 5 \
    --calib-json "$CALIB_JSON" \
    --out     "$PBS_O_WORKDIR/block_a_results_${TAG}.csv" \
    --summary "$PBS_O_WORKDIR/block_a_summary_${TAG}.csv" \
    --workers "$NCPUS_REQ"

echo "=== blockA_sweep ${TAG} finished at $(date) ==="
