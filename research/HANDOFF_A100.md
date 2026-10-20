# Handoff: Phase 4 + Phase 5 on the A100 (in parallel), while Phase 3 finishes on the A10

You are the orchestrator on the A100 machine. Read this fully, then `research/experiment_log.md` (E1–E19 + §6
preregistration), `research/harness_v3_methodology.md`, `research/phase5_implementation.md`,
`research/phase4_datasets.md`, `research/phase4_harness_changes.md`, `harness/server/CONTRACTS.md`.

**Task (Sameed, 2026-10-20):** run **Phase 4** (generalization to BEAMSTER) and **Phase 5** (specialist pointers,
MedGemma, harness v3) **simultaneously** on the A100, while the rest of Phase 3 finishes unattended on the A10.
For a 7-minute talk; priorities below. Sameed's preferences are in §9.

---

## 1. Where everything is (network mount)
Mount (both machines): on the A10 `/home/ubuntu/diaphragmatic-endometriosis` = `/lambda/nfs/diaphragmatic-endometriosis`.
Everything for this project lives in `<mount>/agentic-segmentation-data/`:

| path | what |
|---|---|
| `agentic-segmentation.git` | bare git repo = shared remote. Branches: `main` (A10 Phase 3 line, harness-v2.0.1 + logs), `phase5-v3` (**the A100 code line**: harness v3 + Phase 4 harness changes + Phase 5 tooling), `phase4-data` (merged into phase5-v3). Tags harness-v1, harness-v2, harness-v2.0.1. |
| `mstt199/` | MSTT-199 images + masks (58 GB) — Phase 5 uses MSTT |
| `phase4/beamster/` | BEAMSTER raw + `nifti_iso/` (1 mm iso) + `check/` previews; manifest in repo `eval/v4/beamster_cases.csv` |
| `dvc-remote/` | copy of the DVC store (all results up to the last mirror) |
| `results_a10/` | live mirror of the A10 `results/` (cron every 30 min; `.last_mirror` = timestamp) — Phase 3 sweep outputs land here |

Longitudinal-CT and Spinal-MM-SEG are **skipped** (Sameed); UCSD-BMETS blocked (Aspera token). Phase 4 = BEAMSTER only.

## 2. Setup on the A100
1. `git clone <mount>/agentic-segmentation-data/agentic-segmentation.git ~/agentic-segmentation && cd ~/agentic-segmentation && git checkout phase5-v3`
2. Data: `export SEGD_DATA_ROOT=<mount>/agentic-segmentation-data` (segd remaps manifest paths `/home/ubuntu/agentic-segmentation/data/...` to this root; MSTT `DATA_DIR` honours it too). If network reads are slow for nnInteractive uploads, copy `mstt199/` and `phase4/beamster/nifti_iso` to local NVMe and point `SEGD_DATA_ROOT` there.
3. Results for Study A/B inputs (Phase 2 prepare renders, design-run trajectories): `dvc remote modify local url <mount>/agentic-segmentation-data/dvc-remote/agentic-segmentation && dvc pull` (or read `results_a10/` directly; Study A needs `results/phase2_test/prepare`, `results/phase2_dev/prepare`; Study B needs `results/phase1-0*`, `results/candidate_replay`).
4. Python: `uv` (0.13+); `cd harness/server && uv sync`. Node 22 + `cd harness/runner && npm ci`, `cd harness/study && npm ci`, `cd harness/extension && npm ci` (if present).
5. pi CLI 1.3.0 (`npm i -g` as on the A10) and **credentials: Sameed logs in on the A100** (`pi` → login for Anthropic, OpenAI, opencode-go). Do NOT copy `~/.pi/agent/auth.json` via the mount.
6. nnInteractive server: `uv venv ~/nni/.venv && uv pip install --python ~/nni/.venv "nninteractive[server]==2.6.0"` (A10 had nninteractive 2.6.0, nninteractive_client 2.6.0, torch 2.14.1 cu13); `~/nni/start.sh`:
   `tmux new-session -d -s nni "cd ~/nni && .venv/bin/nninteractive-server --model nnInteractive_v1.0 --host 127.0.0.1 --port 1527 --device cuda:0 --max-sessions 16 --torch-n-threads 8 --idle-timeout-seconds 3600 2>&1 | tee -a ~/nni/server.log"`
   (16 sessions: Phase 4 and Phase 5 agent runs in parallel at concurrency 8 each.)
7. segd: `tmux new-session -d -s segd "cd ~/agentic-segmentation/harness/server && SEGD_DATA_ROOT=... uv run segd 2>&1 | tee -a /tmp/segd.log"`; health: `curl localhost:8765/healthz; curl localhost:1527/healthz`.
8. Local models (Phase 5): `harness/local_models/setup_a100.sh` (venvs, pinned weights, llama.cpp sm_80 build, tmux services). **MedGemma needs `HF_TOKEN` with the HAI-DEF terms accepted** — ask Sameed. Ports: MedGemma Q4 :8091, GUI-Actor :8092, Molmo2 :8093, MedGemma BF16 :8094. Then `python harness/local_models/smoke_test.py`. VRAM plan (80 GB): nni ≤ 5 GB, Molmo2 BF16 ~19, GUI-Actor ~7, MedGemma BF16 ~10, Q4 ~5.
9. Run `cd harness/server && NNI_URL= uv run pytest -q` (expect all pass) before any run.

## 3. Phase 3 (A10, unattended) — what you must do at the end
The A10 keeps running `eval/v2/sweep_scheduler.py` (cron watchdog). Remaining queue: Muse Spark 1.3 high/xhigh
(opencode-go), GPT-6.1-sol high/xhigh, GPT-6-luna xhigh/max (openai), Gemini 4 Argon high (opencode-go). Opus 5 dropped.
Completion: `results_a10/phase3_state/SWEEP_COMPLETE`. Per finished run: `results_a10/phase3_state/sweep_<run>.done`.
At the end (on the A100): copy the new `results_a10/p3-*` into your `results/`, run `eval/v2/export_run.py results/<run>`
for each, `eval/v2/build_log_tables.py`, `eval/v2/sweep_summary.py > research/experiments/sweep_summary.md`,
write E20 (model sweep) in the experiment log, `dvc add results && dvc push`, commit, push to the mount remote.
Done so far (exported on main): Opus 5.5 medium 0.746 / high 0.745 / xhigh 0.766; Haiku 5.5 xhigh 0.661 / max 0.686;
DeepSeek v4.1 flash xhigh 0.629 / max 0.633 (many wall-time forced finishes); Sonnet 5.5 high 0.688 / xhigh 0.737.
**Quota coordination:** the A10 sweep now only uses opencode-go and OpenAI. Anthropic is free for the A100. Do not run
OpenAI/opencode-go agent configurations on the A100 until the sweep is done (they would compete and trip quota blocks).

## 4. Priorities (talk-driven; Sameed agreed)
1. **Phase 5 Study A** — standalone localization by GUI-Actor-2B, Molmo2-8B, MedGemma (Q4 + BF16) on the 169 test
   slices vs frontier (Sonnet raw 0.704, marks 0.826; add Opus 5.5 medium marks condition). Cheap, local. First.
2. **Phase 4** — BEAMSTER generalization with a model panel.
3. **Phase 5 Study C** — harness v3 arms with Sonnet 5.5 high, confirm with Opus 5.5 medium.
4. Study B (decision sets) — supporting.

Run 1 and 2 in parallel (Study A uses only local GPU models; Phase 4 uses API + nnInteractive).

## 5. Phase 4 protocol (BEAMSTER)
- Code: branch `phase5-v3`, runner `--dataset beamster` (selects `system_multi.md`/`task_multi.md`, beamster budget
  defaults, multi-lesion observation text). Scorer: see CONTRACTS.md 'Phase 4' and the multi-lesion scorer
  (global Dice + lesion-wise detection/F1 + per-lesion Dice + size breakdown).
- Design (30 cases) only for tuning: prompts (≤ 8 variants), budgets, mark density; use Haiku 5.5 max or Sonnet 5.5 high
  (cheap). Freeze (tag `harness-v3-p4`), preregister in the experiment log (endpoint: mean global Dice + lesion F1 on
  the 110 test cases; secondary per-size sensitivity), then test.
- Panel (axes: size, vendor, cost, optional effort): **Opus 5.5 medium**, **Haiku 5.5 max**, **best non-Anthropic**
  (pick from the finished sweep: GPT-6.1-sol / GPT-6-luna / DeepSeek max — run after the sweep ends), optional
  **Opus 5.5 xhigh** (effort axis) if budget allows.
- No published BEAMSTER baseline was verifiable; report against a one-click nnInteractive oracle per lesion
  (`eval/v2/oneclick_baseline.py` pattern, adapted to multi-lesion) and inter-run variance.

## 6. Phase 5 protocol
Exactly `research/harness_v3_methodology.md` §3 + `research/phase5_implementation.md`:
Study A (design tuning ≤ 8 prompt variants/model → freeze → test 169), Study B (decision sets from design runs; build
with `harness/study/build_decision_sets.py` — note the 'done' set has only ~10 positives: report CIs, consider a
lenient threshold variant declared in advance), Study C (arms v2 / v3-point / v3-med / v3-both on design with Sonnet
5.5 high, then one frozen v3 on test vs the Sonnet high v2 test run `phase3-high`; adherence via `eval/v2/v3_adherence.py`).
All flags are off by default; v3-off is byte-identical to harness v2 (cross-revision test in tests/test_v3_off_identity.py).

## 7. Running things in parallel safely
- One nnInteractive server, 16 sessions; segd handles many cases. Each runner invocation uses its own `--run-id`.
- Use `harness/runner/run_until_done.sh` (resume + provider-limit loop) or adapt `eval/v2/sweep_scheduler.py` (queue,
  provider blocks, graceful preemption via `<run>/STOP`, watchdog via cron) for long queues.
- Killing runners leaks nnInteractive sessions; restart segd + nni after kills (the scheduler's `reset_services`).
- The 10-minute wall cap per case (15 for beamster, if changed) is a budget: report forced-finish rates.

## 8. Logging / provenance rules
Experiment log entries E21+ (`research/experiment_log.md`), `eval/v2/export_run.py` → `research/experiments/runs/`,
`build_log_tables.py`; git tags for every frozen configuration; preregister before any test-split run; never tune on test.
Commit with thorough messages; push to the mount remote (`git push origin phase5-v3`). Workers never commit; back up
files before edits; review critical code with `openai/gpt-6.1-sol:high` before results are collected.

## 9. Sameed's preferences
Subagent models: well-specified features `opencode-go/glm-5.3-flash:high`; complex `opencode-go/glm-5.3-flash:max`;
running scripts `opencode-go/deepseek-v4.1-flash:high`; reviewer `openai/gpt-6.1-sol:high`; research
`opencode-go/gpt-6-luna`; never Kimi; Sonnet/Opus subagents only when necessary. Use `rg`, never `grep`. Explain codes
in plain words. Concise reports. Reviewers cannot run commands: prepare diff/test artifacts for them in /tmp.
GLM workers sometimes stop mid-task or emit corrupted text: check the worktree and tests yourself, resume or re-spawn.

## 10. Status at handoff
(see bottom; updated by the A10 orchestrator)
