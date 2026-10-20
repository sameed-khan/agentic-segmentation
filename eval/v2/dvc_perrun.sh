#!/usr/bin/env bash
# Restructure DVC: one artifact per run (was: whole results/ as one artifact). Active runs are added later.
set -e
cd /home/ubuntu/agentic-segmentation
ACTIVE="p3-gpt6luna-max p3-gpt61sol-xhigh phase3_state"
RUN_GROUPS="phase2_dev phase2_test phase5 candidate_replay"
paths=()
for d in results/*/; do n=$(basename $d)
  case " $ACTIVE " in *" $n "*) continue;; esac
  case " $RUN_GROUPS " in *" $n "*) for c in results/$n/*; do paths+=("$c"); done; continue;; esac
  paths+=("results/$n")
done
echo "[dvc] ${#paths[@]} artifacts"
dvc remove results.dvc
dvc add "${paths[@]}"
echo "[dvc] added; pushing"
dvc push
echo "[dvc] DONE"
