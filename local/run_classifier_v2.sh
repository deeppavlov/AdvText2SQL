#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
if [[ -x .venv/Scripts/python.exe ]]; then
    PYTHON=.venv/Scripts/python.exe
elif [[ -x .venv/bin/python ]]; then
    PYTHON=.venv/bin/python
else
    PYTHON=python
fi
run=false
for argument in "$@"; do
    case "$argument" in
        --prepare) exec "$PYTHON" -u scripts/prepare_classifier_v2.py "$@" ;;
        --dry-run|--export|--help) exec "$PYTHON" -u scripts/collect_classifier_v2.py "$@" ;;
        --run) run=true ;;
    esac
done
if [[ "$run" != true ]]; then
    echo 'Choose --prepare, --dry-run, --export, or explicit --run.' >&2
    exit 2
fi
arguments=()
for argument in "$@"; do
    [[ "$argument" == --run ]] || arguments+=("$argument")
done
"$PYTHON" -u scripts/collect_classifier_v2.py --dry-run "${arguments[@]}"
TUNNEL_PID=""
cleanup() {
    if [[ -n "$TUNNEL_PID" ]]; then
        kill "$TUNNEL_PID" 2>/dev/null || true
        wait "$TUNNEL_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM
mkdir -p research_results/runs
bash local/start_tunnel.sh >> research_results/runs/classifier_v2_tunnel.log 2>&1 &
TUNNEL_PID=$!
ready=false
for ((i=0; i<45; i++)); do
    if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
        echo 'SSH tunnel failed; collection was not started.' >&2
        exit 1
    fi
    if (echo >/dev/tcp/127.0.0.1/5444) 2>/dev/null; then
        ready=true
        break
    fi
    sleep 1
done
[[ "$ready" == true ]] || { echo 'SSH tunnel is not ready.' >&2; exit 1; }
"$PYTHON" -u scripts/collect_classifier_v2.py "$@"
