"""Phase 3b model-sweep scheduler (single lane, concurrency 8, one model configuration at a time).

Order = Sameed's order. Normally runs strictly in order. Only when a provider hits its quota/rate limit
(runner exit 3 = a case ended on a provider error) is that provider blocked for BLOCK_S and the scheduler moves on
to the first pending run of an unblocked provider (interleaving providers across 5-hour windows). After every
runner invocation it re-picks from the top of the order, so earlier runs resume as soon as their provider is free.
State: results/phase3_state/sweep_<run>.done (finished + scored), results/phase3_state/sched.json (blocks).
Run in tmux p3a:  uv run --project harness/server python eval/v2/sweep_scheduler.py 2>&1 | tee -a /tmp/sweep_sched.log
Safe to kill and restart at any time (runner --resume).
"""
from __future__ import annotations

import csv
import json
import os
import statistics as st
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ST = ROOT / "results" / "phase3_state"
CONC = int(os.environ.get("CONC", "8"))
BLOCK_S = int(os.environ.get("BLOCK_S", "900"))
RUNS = [  # run_id, model, thinking
    ("p3-opus55-medium", "anthropic/claude-opus-5-5", "medium"),
    ("p3-opus55-high", "anthropic/claude-opus-5-5", "high"),
    ("p3-opus55-xhigh", "anthropic/claude-opus-5-5", "xhigh"),
    ("p3-haiku55-xhigh", "anthropic/claude-haiku-5-5", "xhigh"),
    ("p3-haiku55-max", "anthropic/claude-haiku-5-5", "max"),
    ("p3-dsv41flash-xhigh", "opencode-go/deepseek-v4.1-flash", "xhigh"),
    ("p3-dsv41flash-max", "opencode-go/deepseek-v4.1-flash", "max"),
    ("p3-muse13-high", "opencode-go/muse-spark-1.3-contributor", "high"),
    ("p3-muse13-xhigh", "opencode-go/muse-spark-1.3-contributor", "xhigh"),
    ("p3-gpt61sol-high", "openai/gpt-6.1-sol", "high"),
    ("p3-gpt61sol-xhigh", "openai/gpt-6.1-sol", "xhigh"),
    ("p3-gpt6luna-xhigh", "openai/gpt-6-luna", "xhigh"),
    ("p3-gpt6luna-max", "openai/gpt-6-luna", "max"),
    ("p3-argon4-high", "opencode-go/gemini-4-argon-contributor", "high"),
    ("p3-opus5-best", "anthropic/claude-opus-5", "BEST_OPUS55"),
]
SKIP = set(os.environ.get("SKIP", "").split())


def log(msg: str) -> None:
    print(f"[sched] {datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}", flush=True)


def load_state() -> dict:
    p = ST / "sched.json"
    return json.loads(p.read_text()) if p.exists() else {"blocked_until": {}, "fails": {}}


def save_state(s: dict) -> None:
    (ST / "sched.json").write_text(json.dumps(s, indent=1))


def healthy(url: str) -> bool:
    return subprocess.run(["curl", "-sf", "-m", "10", url], capture_output=True).returncode == 0


def ensure_services() -> None:
    if not healthy("localhost:8765/healthz"):
        log("segd down -> restart")
        subprocess.run("tmux kill-session -t segd 2>/dev/null; tmux new-session -d -s segd "
                       f"\"cd {ROOT}/harness/server && uv run segd 2>&1 | tee -a /tmp/segd.log\"", shell=True)
        time.sleep(30)
    while not healthy("localhost:1527/healthz"):
        log("nnInteractive down -> restart (~/nni/start.sh)")
        subprocess.run("tmux kill-session -t nni 2>/dev/null; ~/nni/start.sh", shell=True)
        time.sleep(60)


def reset_services() -> None:
    """Restart segd and nnInteractive: releases nnInteractive sessions leaked by killed runners
    (segd keeps a case and its GPU session until DELETE / idle timeout)."""
    log("reset segd + nnInteractive")
    subprocess.run("tmux kill-session -t segd 2>/dev/null; tmux kill-session -t nni 2>/dev/null", shell=True)
    time.sleep(5)
    subprocess.run("~/nni/start.sh", shell=True)
    subprocess.run(f"tmux new-session -d -s segd \"cd {ROOT}/harness/server && uv run segd 2>&1 | tee -a /tmp/segd.log\"",
                   shell=True)
    for _ in range(30):
        time.sleep(10)
        if healthy("localhost:8765/healthz") and healthy("localhost:1527/healthz"):
            return
    log("services not healthy after reset")


def best_opus55() -> str | None:
    best = None
    for t in ("medium", "high", "xhigh"):
        f = ROOT / "results" / f"p3-opus55-{t}" / "scores.csv"
        if not (ST / f"sweep_p3-opus55-{t}.done").exists():
            return None
        r = list(csv.DictReader(open(f)))
        m = st.mean(st.mean(float(x["dice"]) for x in r if x["histotype"] == h) for h in "FLMNV")
        if best is None or m > best[0]:
            best = (m, t)
    return best[1]


def higher_priority_free(current_run: str, s: dict) -> str | None:
    """First pending run ahead of current_run whose provider is unblocked (and differs from the current one)."""
    cur_prov = next(m for r, m, _ in RUNS if r == current_run).split("/")[0]
    for run, model, think in RUNS:
        if run == current_run:
            return None
        prov = model.split("/")[0]
        if run in SKIP or (ST / f"sweep_{run}.done").exists() or prov == cur_prov:
            continue
        if think == "BEST_OPUS55" and best_opus55() is None:
            continue
        if s["blocked_until"].get(prov, 0) <= time.time():
            return run
    return None


def run_once(run: str, model: str, think: str, s: dict) -> tuple[int, bool]:
    import threading
    cmd = ["npx", "tsx", "run_cases.ts", "--split", "test", "--run-id", run, "--model", model, "--thinking", think,
           "--concurrency", str(CONC), "--resume", "--nni-url", "http://127.0.0.1:1527",
           "--note", f"Phase 3b model sweep: {model} thinking {think}, harness-v2.0.1, test split"]
    p = subprocess.Popen(cmd, cwd=ROOT / "harness" / "runner", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, start_new_session=True)
    flag = {"nothing": False, "infra": 0}

    def reader():
        for line in p.stdout:
            sys.stdout.write(line); sys.stdout.flush()
            if "resume: nothing to do" in line:
                flag["nothing"] = True
            if "failed: non-JSON reply" in line or "at capacity" in line:
                flag["infra"] += 1
    t = threading.Thread(target=reader, daemon=True); t.start()
    while p.poll() is None:
        time.sleep(60)
        hp = higher_priority_free(run, s)
        disk = load_state()
        for k, v in disk.get("blocked_until", {}).items():
            s["blocked_until"][k] = max(v, s["blocked_until"].get(k, 0))
        hp = higher_priority_free(run, s)
        if hp and p.poll() is None:
            # preempt: a run earlier in Sameed's order is runnable again (its provider's window refreshed).
            # Graceful: the runner finishes in-flight cases and exits (STOP file).
            log(f"preempt {run}: {hp} is runnable again -> STOP file, waiting for in-flight cases")
            stop = ROOT / "results" / run / "STOP"
            stop.touch()
            try:
                p.wait(900)          # wall limit per case is 10 min
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, 15)
                p.wait(60)
            stop.unlink(missing_ok=True)
            t.join(10)
            return -16, False
    t.join(10)
    if flag["infra"]:
        log(f"{flag['infra']} case(s) failed on segd/nnInteractive -> reset services")
        reset_services()
    return p.returncode, flag["nothing"]


def main() -> None:
    ST.mkdir(parents=True, exist_ok=True)
    s = load_state()
    reset_services()   # a previous scheduler/runner may have been killed with cases open
    while True:
        # manual blocks (e.g. Sameed: "no Anthropic until the reset") can be added to sched.json while running
        disk = load_state()
        for k, v in disk.get("blocked_until", {}).items():
            s["blocked_until"][k] = max(v, s["blocked_until"].get(k, 0))
        pending = [r for r in RUNS if r[0] not in SKIP and not (ST / f"sweep_{r[0]}.done").exists()]
        if not pending:
            log("SWEEP COMPLETE")
            return
        now = time.time()
        pick = None
        for run, model, think in pending:
            prov = model.split("/")[0]
            if s["blocked_until"].get(prov, 0) > now:
                continue
            if think == "BEST_OPUS55":
                think = best_opus55()
                if think is None:
                    continue
            pick = (run, model, think, prov)
            break
        if pick is None:
            wake = min(v for v in s["blocked_until"].values()) if s["blocked_until"] else now + 300
            wait = max(60, min(wake - now, 1800))
            log(f"all pending providers blocked -> sleep {wait:.0f}s")
            time.sleep(wait)
            continue
        run, model, think, prov = pick
        ensure_services()
        log(f"run {run} ({model}, {think})")
        code, nothing = run_once(run, model, think, s)
        if nothing:
            r = subprocess.run(["uv", "run", "--project", "harness/server", "python", "eval/v2/score.py", f"results/{run}"],
                               cwd=ROOT, capture_output=True, text=True)
            if r.returncode == 0:
                (ST / f"sweep_{run}.done").touch()
                s.setdefault("limit_streak", {})[prov] = 0
                log(f"done {run} (scored)")
            else:
                log(f"score failed for {run}: {r.stderr[-500:]}")
                time.sleep(300)
        elif code == 3:
            n = s.setdefault("limit_streak", {}).get(prov, 0) + 1
            s["limit_streak"][prov] = n
            block = min(BLOCK_S * 2 ** (n - 1), 4 * 3600)   # 15, 30, 60, 120, 240 min while it stays limited
            s["blocked_until"][prov] = time.time() + block
            save_state(s)
            log(f"provider {prov} limited on {run} (streak {n}) -> blocked {block}s, switching")
        elif code == -15:
            reset_services()
        elif code == -16:
            pass   # graceful preemption: no sessions leaked
        elif code != 0:
            reset_services()
            s["fails"][run] = s["fails"].get(run, 0) + 1
            save_state(s)
            log(f"runner exit {code} on {run} (fail #{s['fails'][run]}) -> sleep 120s")
            time.sleep(120)
        # code 0 without 'nothing to do': loop -> same run resumes (picks up incomplete cases)


if __name__ == "__main__":
    main()
