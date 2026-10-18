#!/usr/bin/env bash
# Bring up nnInteractive on a Colab L4 and forward it to 127.0.0.1:1527 (design doc §7.4).
# Idempotent: reuses an existing `nnint` runtime / server / forward when present.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S=nnint; PORT=1527; LOG="$HERE/logs/server_exec.log"
mkdir -p "$HERE/logs" ~/.ssh/cm

if curl -sf --max-time 5 "http://127.0.0.1:$PORT/healthz" >/dev/null; then
  echo "already up"; curl -s "http://127.0.0.1:$PORT/healthz"; echo; exit 0
fi

if ! colab ls 2>/dev/null | grep -qw "$S"; then
  colab new -s "$S" --gpu L4
fi
colab install -s "$S" nnInteractive

if ! pgrep -f "colab exec -s $S -f" >/dev/null; then
  : > "$LOG"
  nohup colab exec -s "$S" -f "$HERE/start_server.py" > "$LOG" 2>&1 &
fi
echo "waiting for server..."
for i in $(seq 1 60); do
  grep -q "Uvicorn running on" "$LOG" 2>/dev/null && break
  grep -q "server already listening" "$LOG" 2>/dev/null && break
  sleep 5
done
grep -qE "Uvicorn running on|server already listening" "$LOG" || { echo "server did not start; see $LOG"; tail -20 "$LOG"; exit 1; }

ssh -O check colab-nnint 2>/dev/null || ssh -fN colab-nnint
ssh -O forward -L "$PORT:127.0.0.1:$PORT" colab-nnint 2>/dev/null || true
for i in $(seq 1 12); do
  curl -sf --max-time 5 "http://127.0.0.1:$PORT/healthz" && { echo; exit 0; }
  sleep 5
done
echo "healthz failed"; exit 1
