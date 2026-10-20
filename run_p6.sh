#!/usr/bin/env bash
# Phase 6 full run (169 test cases), resumable; waits on provider limits. Does not touch the main segd/nni.
# Everything experiment-defining (model, thinking, split, budgets, concurrency 6, segmenter sammed3d, prompts) is in
# experiments/presets/p6-sammed3d-high.json; only the machine's service URLs are passed here.
# Needs: a second segd on :8766 (this script starts it) and the sammed3d service on :8095.
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT/harness/runner"
while :; do
  curl -sf -m 10 localhost:8766/healthz >/dev/null || { tmux kill-session -t segd_p6 2>/dev/null; tmux new-session -d -s segd_p6 -c "$ROOT/harness/server" "uv run uvicorn segd.app:app --host 127.0.0.1 --port 8766 2>&1 | tee -a /tmp/segd_p6.log"; sleep 30; }
  curl -sf -m 10 localhost:8095/healthz >/dev/null || { echo "[p6] sammed3d down"; sleep 600; continue; }
  out=$(mktemp)
  npx tsx run_cases.ts --preset p6-sammed3d-high --resume --segd-url http://127.0.0.1:8766 \
    --segmenter-url http://127.0.0.1:8095 2>&1 | tee "$out"
  code=${PIPESTATUS[0]}
  if rg -q "resume: nothing to do" "$out"; then rm -f "$out"; touch "$ROOT/results/p6-sammed3d-high/ALL_DONE"; echo "[p6] ALL DONE"; exit 0; fi
  rm -f "$out"
  if [ "$code" = 3 ]; then sleep 1200; elif [ "$code" != 0 ]; then sleep 120; fi
done
