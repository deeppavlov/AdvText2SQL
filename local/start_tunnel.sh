#!/usr/bin/env bash
# Start a persistent SSH tunnel once. Run this in a separate terminal/background
# BEFORE running run_single.sh, ablation_large_bird.sh, or ablation_large_ambrosia.sh.
# All of those scripts detect an already-open port 5444 and skip their own tunnel,
# so you only enter the SSH password here, once.
#
# Usage:
#   bash local/start_tunnel.sh          # foreground — keep this terminal open
#   bash local/start_tunnel.sh &        # background in current shell
#
# To stop: kill the ssh process, or close the terminal it's running in.

set -uo pipefail

SSH_USER="onik110"
SSH_HOST="lnsigo.mipt.ru"
SSH_PORT="2278"
CONTROL_SOCKET="/tmp/ssh_tunnel_persistent"

if (echo >/dev/tcp/localhost/5444) 2>/dev/null; then
    echo "[TUNNEL] Port 5444 already open — nothing to do."
    exit 0
fi

echo "Connecting SSH tunnel (enter password once)..."
ssh -N -M -S "$CONTROL_SOCKET" \
    -o ControlPersist=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ConnectTimeout=15 \
    -L 5444:10.11.1.6:5444 \
    -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" &
SSH_PID=$!

for i in $(seq 1 30); do
    if (echo >/dev/tcp/localhost/5444) 2>/dev/null; then
        echo "[TUNNEL] Ready after ${i}s (PID=$SSH_PID)"
        echo "Leave this running. Other ablation scripts will reuse this connection."
        # Watchdog: reconnect automatically if the tunnel drops (Ambrosia builds ~90+ DBs per run)
        while true; do
            wait "$SSH_PID" 2>/dev/null
            echo "[WATCHDOG] Tunnel dropped — reconnecting..."
            ssh -N -M -S "$CONTROL_SOCKET" \
                -o ControlPersist=yes \
                -o ServerAliveInterval=30 \
                -o ServerAliveCountMax=3 \
                -o ConnectTimeout=15 \
                -L 5444:10.11.1.6:5444 \
                -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" &
            SSH_PID=$!
        done
        exit 0
    fi
    sleep 1
done

echo "[TUNNEL] ERROR: port 5444 not reachable after 30s"
kill "$SSH_PID" 2>/dev/null || true
exit 1
