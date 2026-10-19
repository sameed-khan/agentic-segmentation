#!/usr/bin/env bash
# Run a split to completion across provider limits: repeat run_cases.ts --resume until every case is complete.
#   harness/runner/run_until_done.sh <run_id> <thinking> <split> "<note>" [extra run_cases args...]
# Exit code 3 from the runner = stopped on a provider/API error (rate/session limit) -> wait, then resume.
# Log: /tmp/<run_id>.log. State lives entirely in results/<run_id>/ (safe to kill and restart this script).
set -u
RUN=$1; THINK=$2; SPLIT=$3; NOTE=$4; shift 4
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WAIT_S=${WAIT_S:-1200}
cd "$ROOT/harness/runner"
attempt=0
while :; do
  attempt=$((attempt + 1))
  if ! curl -sf -m 10 localhost:8765/healthz >/dev/null; then
    echo "[until_done] $(date -Is) segd down -> restarting"
    tmux kill-session -t segd 2>/dev/null
    tmux new-session -d -s segd "cd $ROOT/harness/server && uv run segd 2>&1 | tee -a /tmp/segd.log"
    sleep 30
  fi
  if ! curl -sf -m 10 localhost:${NNI_PORT:-1527}/healthz >/dev/null; then
    echo "[until_done] $(date -Is) nnInteractive (${NNI_PORT:-1527}) down -> waiting ${WAIT_S}s (restart manually: tmux nni, ~/nni/start.sh)"
    sleep "$WAIT_S"; continue
  fi
  echo "[until_done] $(date -Is) attempt $attempt run=$RUN thinking=$THINK split=$SPLIT"
  out=$(mktemp)
  npx tsx run_cases.ts --split "$SPLIT" --run-id "$RUN" --thinking "$THINK" --concurrency "${CONC:-4}" --resume \
    --note "$NOTE" --nni-url "http://127.0.0.1:${NNI_PORT:-1527}" "$@" 2>&1 | tee "$out"
  code=${PIPESTATUS[0]}
  if rg -q "resume: nothing to do" "$out"; then rm -f "$out"; echo "[until_done] $(date -Is) ALL CASES COMPLETE"; touch "$ROOT/results/$RUN/ALL_DONE"; exit 0; fi
  rm -f "$out"
  if [ "$code" = 3 ]; then echo "[until_done] $(date -Is) provider error -> sleep ${WAIT_S}s"; sleep "$WAIT_S"
  elif [ "$code" != 0 ]; then echo "[until_done] $(date -Is) runner exit $code -> sleep 120s"; sleep 120; fi
done
