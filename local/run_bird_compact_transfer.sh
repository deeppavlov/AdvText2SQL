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
for argument in "$@"; do
    case "$argument" in
        --dry-run|--prepare|--status|--report|--help)
            exec "$PYTHON" -u scripts/bird_compact_transfer.py "$@" ;;
    esac
done
"$PYTHON" -u scripts/bird_compact_transfer.py "$@" --dry-run
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
echo '[TUNNEL] Log: research_results/runs/bird_compact_transfer_tunnel.log'
bash local/start_tunnel.sh >> research_results/runs/bird_compact_transfer_tunnel.log 2>&1 &
TUNNEL_PID=$!
ready=false
for ((i=0; i<45; i++)); do
    if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
        echo 'SSH tunnel failed; compact transfer was not started.' >&2
        exit 1
    fi
    if (echo >/dev/tcp/127.0.0.1/5444) 2>/dev/null; then
        ready=true
        break
    fi
    sleep 1
done
if [[ "$ready" != true ]]; then
    echo 'SSH tunnel not ready; compact transfer was not started.' >&2
    exit 1
fi
"$PYTHON" -u scripts/bird_compact_transfer.py "$@"
