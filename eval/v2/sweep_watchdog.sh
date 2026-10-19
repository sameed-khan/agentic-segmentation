#!/usr/bin/env bash
# cron (every 10 min): keep the Phase 3b sweep scheduler alive without an orchestrator session.
# Disabled by: touch results/phase3_state/STOPPED   (remove the file to re-enable)
export PATH=/home/ubuntu/.pi/agent/bin:/home/ubuntu/.local/share/pi-node/current/bin:/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin
cd /home/ubuntu/agentic-segmentation || exit 0
ST=results/phase3_state
[ -f $ST/STOPPED ] && exit 0
[ -f $ST/SWEEP_COMPLETE ] && exit 0
if ! pgrep -f "[s]weep_scheduler.py" >/dev/null; then
  echo "[watchdog] $(date -Is) scheduler not running -> start" >> /tmp/sweep_watchdog.log
  eval/v2/start_sweep.sh >> /tmp/sweep_watchdog.log 2>&1
  exit 0
fi
# stale heartbeat (> 40 min) = hung scheduler -> restart
if [ -f $ST/sched_heartbeat ] && [ $(( $(date +%s) - $(date -d "$(cat $ST/sched_heartbeat)" +%s) )) -gt 2400 ]; then
  echo "[watchdog] $(date -Is) heartbeat stale -> restart" >> /tmp/sweep_watchdog.log
  eval/v2/start_sweep.sh >> /tmp/sweep_watchdog.log 2>&1
fi
