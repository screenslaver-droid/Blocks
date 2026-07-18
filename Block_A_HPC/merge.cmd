#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────
# blockA_sweep / merge.cmd
# Concatenates the 40 per-N CSVs from sweep_array.cmd into one final
# block_a_results.csv + block_a_summary.csv. Cheap, single-core job.
# Submit with -W depend=afterokarray:<sweep_array_jobid> (see submit_all.sh).
# ─────────────────────────────────────────────────────────────────────────

#PBS -N blockA_merge
#PBS -q <QUEUE_NAME>
#PBS -P <PROJECT_NAME>
#PBS -l select=1:ncpus=1
#PBS -l walltime=00:30:00
#PBS -j oe

cd "$PBS_O_WORKDIR" || exit 1
exec > "blockA_merge.log" 2>&1

echo "=== blockA_merge starting on $(hostname) at $(date) ==="

module purge
module load anaconda3
source activate blockA_env

python merge_results.py \
    --dir "$PBS_O_WORKDIR" \
    --out block_a_results.csv \
    --summary block_a_summary.csv

echo "=== blockA_merge finished at $(date) ==="
