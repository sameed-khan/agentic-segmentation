#!/usr/bin/env bash
# wait for the iso conversion chain, then repoint manifests and rerun the data check (streamed log)
cd /home/ubuntu/agseg-p4
while kill -0 1447827 2>/dev/null; do sleep 60; done
echo "[p4_finish] $(date -Is) iso chain done" >> eval/v4/logs_finish.txt
uv run --project harness/server python -u eval/v4/repoint_manifests_to_iso.py >> eval/v4/logs_finish.txt 2>&1
for d in beamster spinal_mm_seg longitudinal_ct; do uv run --project harness/server python -u eval/v4/data_check_phase4.py $d 6 --jobs 8 >> eval/v4/logs_finish.txt 2>&1; done
echo "[p4_finish] $(date -Is) DONE" >> eval/v4/logs_finish.txt
