#!/usr/bin/env bash
# (Re)start the Phase 3b sweep scheduler in tmux p3a (kills any running lane/scheduler first).
cd "$(dirname "$0")/../.."
for s in p3a p3b p3c; do tmux kill-session -t $s 2>/dev/null; done
sleep 2; pkill -f "[r]un_cases.ts --split test" 2>/dev/null; sleep 3
tmux new-session -d -s p3a "cd $PWD && uv run --project harness/server python eval/v2/sweep_scheduler.py 2>&1 | tee -a /tmp/sweep_sched.log"
