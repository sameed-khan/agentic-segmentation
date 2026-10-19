#!/usr/bin/env bash
# Phase 3b model sweep on the test split (169 cases), harness frozen at harness-v2.0.1 (agent-facing).
# Idempotent: results/phase3_state/sweep_<run>.done per finished run; runs/ resume via run_until_done.sh.
# Order = Sameed's order (Anthropic first, Opus first). Run in tmux p3b; see research/HANDOFF_phase3.md.
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
ST=results/phase3_state; mkdir -p "$ST"
PY="uv run --project harness/server python"
log() { echo "[sweep] $(date -Is) $*"; }
# run_id | model | thinking
RUNS=$(cat <<'E'
p3-opus55-medium anthropic/claude-opus-5-5 medium
p3-opus55-high anthropic/claude-opus-5-5 high
p3-opus55-xhigh anthropic/claude-opus-5-5 xhigh
p3-haiku55-xhigh anthropic/claude-haiku-5-5 xhigh
p3-haiku55-max anthropic/claude-haiku-5-5 max
p3-gpt61sol-high openai/gpt-6.1-sol high
p3-gpt61sol-xhigh openai/gpt-6.1-sol xhigh
p3-gpt6luna-xhigh openai/gpt-6-luna xhigh
p3-gpt6luna-max openai/gpt-6-luna max
p3-dsv41flash-xhigh opencode-go/deepseek-v4.1-flash xhigh
p3-dsv41flash-max opencode-go/deepseek-v4.1-flash max
p3-muse13-high opencode-go/muse-spark-1.3-contributor high
p3-muse13-xhigh opencode-go/muse-spark-1.3-contributor xhigh
E
)
SKIP=${SKIP:-}   # space-separated run ids to skip (e.g. a model that failed its smoke test)
while read -r RUN MODEL THINK; do
  [ -z "$RUN" ] && continue
  [ -f "$ST/sweep_$RUN.done" ] && { log "skip $RUN (done)"; continue; }
  case " $SKIP " in *" $RUN "*) log "skip $RUN (SKIP)"; continue;; esac
  log "start $RUN $MODEL $THINK"
  harness/runner/run_until_done.sh "$RUN" "$THINK" test "Phase 3b model sweep: $MODEL thinking $THINK, harness-v2.0.1, test split" --model "$MODEL"
  $PY eval/v2/score.py "results/$RUN" >/dev/null 2>&1 && touch "$ST/sweep_$RUN.done" && log "done $RUN"
done <<< "$RUNS"
log "SWEEP COMPLETE"
