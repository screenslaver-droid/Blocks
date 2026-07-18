#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────
# blockA_sweep / submit_all.sh
# Submits calibrate.cmd -> sweep_array.cmd (40-way array) -> merge.cmd as
# a dependency chain, so you only run this once and can walk away.
#
# Usage:  bash submit_all.sh
# Track:  qstat -an
# Cancel: qdel <jobid>            (cancels one subjob or the whole chain)
#         qdel $(cat .blockA_jobids)   (cancels everything this run submitted)
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/3] Submitting calibration job..."
CALIB_JOBID=$(qsub calibrate.cmd)
echo "      calibrate.cmd -> $CALIB_JOBID"

echo "[2/3] Submitting 40-way N-value array sweep (depends on calibration)..."
SWEEP_JOBID=$(qsub -W depend=afterok:"${CALIB_JOBID}" sweep_array.cmd)
echo "      sweep_array.cmd -> $SWEEP_JOBID"

echo "[3/3] Submitting merge job (depends on the full array finishing)..."
# NOTE: 'afterokarray' waits for every subjob in the array to finish
# successfully. If your PBS Pro version/site rejects this dependency
# keyword, fall back to submitting merge.cmd by hand once
# `qstat -an | grep blockA_sweep` shows no subjobs left.
MERGE_JOBID=$(qsub -W depend=afterokarray:"${SWEEP_JOBID}" merge.cmd)
echo "      merge.cmd -> $MERGE_JOBID"

echo "$CALIB_JOBID $SWEEP_JOBID $MERGE_JOBID" > .blockA_jobids

echo
echo "Submitted. Job IDs saved to .blockA_jobids."
echo "Check status with: qstat -an"
echo "Final outputs (once merge.cmd completes): block_a_results.csv, block_a_summary.csv"
