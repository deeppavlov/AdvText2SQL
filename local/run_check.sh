#!/usr/bin/env bash
set -euo pipefail

SSH_USER="onik110"
SSH_HOST="lnsigo.mipt.ru"
SSH_PORT="2278"

export PATH="/c/Users/718/anaconda3/Scripts:/mnt/c/Users/718/anaconda3/Scripts:$PATH"
UV=$(command -v uv 2>/dev/null || \
     ls /c/Users/718/anaconda3/Scripts/uv* 2>/dev/null | head -1 || \
     ls /mnt/c/Users/718/anaconda3/Scripts/uv* 2>/dev/null | head -1) || true
if [[ -z "$UV" ]]; then echo "ERROR: uv not found"; exit 1; fi

echo "[TUNNEL] Starting..."
ssh -N -o ExitOnForwardFailure=yes -L 5444:10.11.1.6:5444 -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" &
TUNNEL_PID=$!

for i in $(seq 1 30); do
    if (echo >/dev/tcp/localhost/5444) 2>/dev/null; then
        echo "[TUNNEL] Ready after ${i}s"
        break
    fi
    sleep 1
done

"$UV" run --env-file .env check_ambrosia_dbs.py

kill "$TUNNEL_PID" 2>/dev/null || true
