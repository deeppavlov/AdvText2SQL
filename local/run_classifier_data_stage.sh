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
connect=false
for argument in "$@"; do
    case "$argument" in
        --prepare) exec "$PYTHON" -u scripts/prepare_classifier_data_stage.py "$@" ;;
        --dry-run|--export|--help) exec "$PYTHON" -u scripts/classifier_data_stage.py "$@" ;;
        --run|--snapshot) connect=true ;;
    esac
done
if [[ "$connect" != true ]]; then
    echo 'Choose --prepare, --dry-run, --export, --snapshot, or explicit --run.' >&2
    exit 2
fi
arguments=()
for argument in "$@"; do
    [[ "$argument" == --run || "$argument" == --snapshot ]] || arguments+=("$argument")
done
"$PYTHON" -u scripts/classifier_data_stage.py --dry-run "${arguments[@]}"
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
bash local/start_tunnel.sh >> research_results/runs/classifier_data_stage_tunnel.log 2>&1 &
TUNNEL_PID=$!
ready=false
for ((i=0; i<45; i++)); do
    kill -0 "$TUNNEL_PID" 2>/dev/null || { echo 'SSH tunnel failed.' >&2; exit 1; }
    if (echo >/dev/tcp/127.0.0.1/5444) 2>/dev/null; then
        ready=true
        break
    fi
    sleep 1
done
[[ "$ready" == true ]] || { echo 'SSH tunnel is not ready.' >&2; exit 1; }
"$PYTHON" -u scripts/classifier_data_stage.py "$@"
