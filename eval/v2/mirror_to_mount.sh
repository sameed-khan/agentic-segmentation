#!/usr/bin/env bash
# cron (every 30 min): push DVC artifacts (per-run) and git main to the network mount. (Raw results mirror retired
# 2026-10-20: results are per-run DVC artifacts in <mount>/dvc-remote; DVC remote 'mount' is the default.)
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
cd /home/ubuntu/agentic-segmentation || exit 0
pgrep -f "[d]vc push" >/dev/null && exit 0
uv run --project harness/server dvc push -r mount -q 2>/dev/null || dvc push -r mount -q
git push -q mount main 2>/dev/null
date -Is > /tmp/mirror_last
