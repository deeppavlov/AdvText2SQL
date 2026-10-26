#!/usr/bin/env bash
# Baseline, then independent features; a single supervised SSH tunnel for the suite.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" == "--feature-combinations" || "${1:-}" == "--hybrid-relationships" ]]; then
    exec bash local/run_openrouter_baseline.sh "$@"
fi
exec bash local/run_openrouter_baseline.sh --feature-ablation "$@"
