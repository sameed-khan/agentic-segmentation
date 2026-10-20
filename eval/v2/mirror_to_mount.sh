#!/usr/bin/env bash
# cron (every 30 min): mirror A10 results + DVC store + git to the network mount for the A100 orchestrator
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
M=/home/ubuntu/diaphragmatic-endometriosis/agentic-segmentation-data
pgrep -f "[r]sync -a /home/ubuntu/agentic-segmentation/results/" >/dev/null && exit 0
rsync -a /home/ubuntu/agentic-segmentation/results/ $M/results_a10/
rsync -a /home/ubuntu/dvc-remote/ $M/dvc-remote/
cd /home/ubuntu/agentic-segmentation && git push -q mount main 2>/dev/null
date -Is > $M/results_a10/.last_mirror
