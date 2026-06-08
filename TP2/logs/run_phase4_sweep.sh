#!/usr/bin/env bash
# Phase-4 re-run driver for the GPU machine (jmrc-ASUS).
#
# Runs the improved closed-loop refinement (CLIP-token cap, per-image early-stop,
# diversity via temperature escalation + axis rotation + per-call seeds, 7836
# parse logging) once per optimiser seed, satisfying the statement's Reporting
# Protocol: "if the prompt search is stochastic, repeat with at least 5 seeds;
# keep the LCM rendering seed fixed across all repetitions."
#
# The LCM *render* seed is fixed per image (from the filename) and NEVER changes.
# Only the proposal-sampling seed (P4_SEED) varies. Each repetition writes to its
# own tag (outputs/phase4_s<seed>/, phase4_s<seed>_*.json/csv) so nothing clobbers
# the previous run, and a crash just means re-running the same line to resume.
#
# Usage:
#   ./run_phase4_sweep.sh              # full 5-seed sweep, 12 iters each
#   P4_ITERS=6 ./run_phase4_sweep.sh   # shorter repetitions if time is tight
#   SEEDS="1234" ./run_phase4_sweep.sh # single clean run only (headline deliverable)
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
SEEDS="${SEEDS:-1234 5678 9012 3456 7890}"   # >= 5 per the Reporting Protocol

for SEED in $SEEDS; do
  TAG="_s${SEED}"
  echo "============================================================"
  echo ">>> Phase 4 repetition: optimiser seed=${SEED}  tag=${TAG}"
  echo ">>> outputs -> outputs/phase4${TAG}/ , phase4${TAG}_*.json/csv"
  echo "============================================================"
  # P4_TAG isolates outputs+checkpoint; P4_SEED varies only the prompt search.
  # Re-running this exact line after a crash resumes from that tag's checkpoint.
  P4_TAG="${TAG}" P4_SEED="${SEED}" P4_ITERS="${P4_ITERS:-12}" \
    "${PY}" -u phase4_refine.py 2>&1 | tee "phase4${TAG}.log"
done

echo
echo ">>> All repetitions done. Aggregating across seeds ..."
"${PY}" phase4_aggregate_seeds.py
echo ">>> See phase4_seed_aggregate.json for mean +/- std across repetitions."
