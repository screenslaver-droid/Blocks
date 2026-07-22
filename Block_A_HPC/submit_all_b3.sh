#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────
# blockA_sweep / submit_all_b3.sh
# Submits sweep_array_b3.cmd (40-way array) -> merge_b3.cmd as a dependency
# chain. Run this AFTER Block A's own chain (submit_all.sh) has finished
# successfully -- it needs block_a_calibration.json to already exist.
#
# Usage:  bash submit_all_b3.sh
# Track:  qstat -an
# Cancel: qdel $(cat .blockB3_jobids)
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f block_a_calibration.json ]; then
    echo "ERROR: block_a_calibration.json not found in $(pwd)." >&2
    echo "       Block A's calibrate.cmd (part of submit_all.sh) must finish" >&2
    echo "       successfully before B3 can run -- see README.md." >&2
    exit 1
fi

echo "[1/2] Submitting 40-way N-value B3 array sweep..."
SWEEP_JOBID=$(qsub sweep_array_b3.cmd)
echo "      sweep_array_b3.cmd -> $SWEEP_JOBID"

echo "[2/2] Submitting B3 merge job (depends on the full array finishing)..."
# See sweep_array.cmd/submit_all.sh's equivalent note: if your PBS Pro
# version/site rejects 'afterokarray', submit merge_b3.cmd by hand once
# `qstat -an | grep blockB3_sweep` shows no subjobs left.
MERGE_JOBID=$(qsub -W depend=afterokarray:"${SWEEP_JOBID}" merge_b3.cmd)
echo "      merge_b3.cmd -> $MERGE_JOBID"

echo "$SWEEP_JOBID $MERGE_JOBID" > .blockB3_jobids

echo
echo "Submitted. Job IDs saved to .blockB3_jobids."
echo "Check status with: qstat -an"
echo "Final outputs (once merge_b3.cmd completes): block_b3_results.csv,"
echo "block_b3_summary.csv, block_b3_elbow.csv"
