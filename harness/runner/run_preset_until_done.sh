#!/usr/bin/env bash
# Run an experiment preset to completion across provider limits: repeat `run_cases.ts --preset <name> --resume`
# until every case is complete (like run_until_done.sh, but model/thinking/split/budgets/concurrency/prompts all come
# from experiments/presets/<name>.json).
#   harness/runner/run_preset_until_done.sh <preset> [extra run_cases args, e.g. --concurrency 4 --nni-url ...]
# Exit code 3 from the runner = stopped on a provider/API error (rate/session limit) -> wait, then resume.
# State lives entirely in results/<run_id>/ (safe to kill and restart this script). Log: stdout.
set -u
PRESET=$1; shift
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WAIT_S=${WAIT_S:-1200}
RUN=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['runner']['run_id'])" "$ROOT/experiments/presets/$PRESET.json") || exit 2
cd "$ROOT/harness/runner"
attempt=0
while :; do
  attempt=$((attempt + 1))
  if ! curl -sf -m 10 localhost:8765/healthz >/dev/null; then
    echo "[until_done] $(date -Is) segd (8765) down: start it (cd harness/server && uv run segd); waiting ${WAIT_S}s"
    sleep "$WAIT_S"; continue
  fi
  echo "[until_done] $(date -Is) attempt $attempt preset=$PRESET"
  out=$(mktemp)
  npx tsx run_cases.ts --preset "$PRESET" --resume "$@" 2>&1 | tee "$out"
  code=${PIPESTATUS[0]}
  if rg -q "resume: nothing to do" "$out"; then rm -f "$out"; echo "[until_done] $(date -Is) ALL CASES COMPLETE"; touch "$ROOT/results/$RUN/ALL_DONE"; exit 0; fi
  rm -f "$out"
  if [ "$code" = 3 ]; then echo "[until_done] $(date -Is) provider error -> sleep ${WAIT_S}s"; sleep "$WAIT_S"
  elif [ "$code" != 0 ]; then echo "[until_done] $(date -Is) runner exit $code -> sleep 120s"; sleep 120; fi
done
