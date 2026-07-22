#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────
# blockA_sweep / merge_b3.cmd
# Concatenates the 40 per-N CSVs from sweep_array_b3.cmd into one final
# block_b3_results.csv + block_b3_summary.csv + block_b3_elbow.csv.
# Submit with -W depend=afterokarray:<sweep_array_b3_jobid> (see
# submit_all_b3.sh).
# ─────────────────────────────────────────────────────────────────────────

#PBS -N blockB3_merge
#PBS -q <QUEUE_NAME>
#PBS -P <PROJECT_NAME>
#PBS -l select=1:ncpus=1
#PBS -l walltime=00:30:00
#PBS -j oe

cd "$PBS_O_WORKDIR" || exit 1
exec > "blockB3_merge.log" 2>&1

echo "=== blockB3_merge starting on $(hostname) at $(date) ==="

module purge
module load anaconda3
source activate blockA_env
# NOTE: unlike merge.cmd (Block A), this needs jax importable -- merging
# imports block_b3_integration_accuracy.py, which imports block_a_solver_
# benchmark.py and hard-exits at import time if JAX/Diffrax aren't
# available (see that script's own guard). Same env as the sweep, so this
# is fine as long as you haven't pointed `module load`/`source activate`
# at something different here.

python merge_results_b3.py \
    --dir "$PBS_O_WORKDIR" \
    --out block_b3_results.csv \
    --summary block_b3_summary.csv \
    --elbow-out block_b3_elbow.csv

echo "=== blockB3_merge finished at $(date) ==="
