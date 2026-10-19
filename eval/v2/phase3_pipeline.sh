#!/usr/bin/env bash
# Phase 3 pipeline (test split). Idempotent: each finished step writes results/phase3_state/<step>.done.
# Run in tmux p3; see research/HANDOFF_phase3.md.
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
ST=results/phase3_state; mkdir -p "$ST"
PY="uv run --project harness/server python"
log() { echo "[p3] $(date -Is) $*"; }
step() {  # step <name> <command...>; retries a failing step every 10 min
  local name=$1; shift
  [ -f "$ST/$name.done" ] && { log "skip $name (done)"; return 0; }
  while :; do
    log "start $name"
    if "$@"; then touch "$ST/$name.done"; log "done $name"; return 0; fi
    log "FAILED $name (exit $?) -> retry in 600s"; sleep 600
  done
}

phase2_loop() {  # ask.ts exits 3 while error rows remain -> wait and retry
  while :; do
    (cd harness/study && npx tsx ask.ts --run p2-test-high --results-root "$ROOT/results/phase2_test" \
      --prepare-dir "$ROOT/results/phase2_test/prepare" --conditions R,S,S-gen --reps 3 --thinking high --concurrency 4)
    code=$?
    [ $code = 0 ] && break
    log "phase2 ask exit $code -> sleep 1200"; sleep 1200
  done
  $PY harness/study/score.py --run p2-test-high --results-root results/phase2_test --prepare-dir results/phase2_test/prepare
}

NOTE_H='Phase 3 primary: harness-v2, test split (169), thinking high (preregistered, experiment_log §6)'
NOTE_X='Phase 3 second pass: harness-v2, test split (169), thinking xhigh (preregistered, experiment_log §6)'

step download   $PY eval/v2/download_mstt.py test
step datacheck  $PY eval/v2/data_check.py test
step prepare    $PY harness/study/prepare.py --split test --jobs 6
step oneclick   $PY eval/v2/oneclick_baseline.py test
step agent_high harness/runner/run_until_done.sh phase3-high high test "$NOTE_H"
step score_high $PY eval/v2/score.py results/phase3-high
step phase2     phase2_loop
step agent_xhigh harness/runner/run_until_done.sh phase3-xhigh xhigh test "$NOTE_X"
step score_xhigh $PY eval/v2/score.py results/phase3-xhigh
log "PIPELINE COMPLETE"
