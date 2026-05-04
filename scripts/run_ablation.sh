#!/usr/bin/env bash
# Run BIRD + Ambrosia benchmarks on a given branch and archive results.
# Usage: ./scripts/run_ablation.sh <branch-name>
# Prereq: SSH tunnel to remote DB on localhost:5444; .env populated.

set -euo pipefail

# Force unbuffered Python so progress shows up live.
# tee turns stdout into a pipe → Python switches to block-buffering by default,
# which can look like the process is hung. Setting this in the env propagates
# into the python child spawned by `uv run`.
export PYTHONUNBUFFERED=1

BRANCH="${1:?branch name required, e.g. feat/02_column_statistics}"

echo "[$(date +%H:%M:%S)] === checkout $BRANCH ==="
git checkout "$BRANCH"
echo "[$(date +%H:%M:%S)] HEAD = $(git rev-parse --short HEAD)"

OUT="ablation_results/${BRANCH//\//_}"
mkdir -p "$OUT"

echo "[$(date +%H:%M:%S)] === BIRD on $BRANCH ==="
uv run --env-file .env bird_benchmark.py 2>&1 | tee "$OUT/bird.log"
[[ -f query_results.json ]] && cp query_results.json "$OUT/bird_query_results.json"

echo "[$(date +%H:%M:%S)] === AMBROSIA on $BRANCH ==="
uv run --env-file .env ambrosia_benchmark.py 2>&1 | tee "$OUT/ambrosia.log"
[[ -f query_results.json ]] && cp query_results.json "$OUT/ambrosia_query_results.json"

echo "[$(date +%H:%M:%S)] === Summary ==="
{
  echo "Branch: $BRANCH"
  echo "Commit: $(git rev-parse --short HEAD)"
  echo
  grep -E "(Overall accuracy|^Accuracy):" "$OUT"/bird.log "$OUT"/ambrosia.log || true
} | tee "$OUT/summary.txt"
