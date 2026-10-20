# Handoff: Phase 4 + Phase 5 on the A100 (in parallel), while Phase 3 finishes on the A10

**You are the Phase 4/5 orchestrator on the A100.** Your pi session runs in `~/agentic-segmentation` (the repo,
branch `phase5-v3`). This file is the canonical handoff (`~/HANDOFF_A100.md` is a symlink to it; the A10's original
is `research/HANDOFF_A100.md` on `main`). Read this fully, then `research/experiment_log.md` (E1–E19 + §6
preregistration; read `origin/main`'s copy for the latest A10 entries), `research/harness_v3_methodology.md`,
`research/phase5_implementation.md`, `research/phase4_datasets.md`, `research/phase4_harness_changes.md`,
`harness/server/CONTRACTS.md`.

**Task (Sameed, 2026-10-20):** run **Phase 4** (generalization to BEAMSTER) and **Phase 5** (specialist pointers,
MedGemma, harness v3) **simultaneously** on the A100. **Phase 3 is not yours:** the A10 orchestrator (a separate
agent on the other Lambda machine) finishes Phase 3 completely (sweep, exports, log entries). Do not run, export,
rescore or edit anything Phase 3.
For a 7-minute talk; priorities in §4. Sameed's preferences in §9.

---

## 0. A100 machine state (set up 2026-10-20 ~04:20 UTC by the setup session — read first)

**Hardware (differs from the plan!):** 1× **NVIDIA A100-SXM4-40GB** (40 GB, *not* 80 GB), driver 580 / CUDA 13.0,
`nvcc` 12.8, 30 CPUs, 216 GB RAM, 472 GB free on `/` (NVMe), passwordless `sudo` works.
→ The 80 GB "everything coexists" VRAM plan in `research/phase5_implementation.md` §3 and `setup_a100.sh` does
**not** fit (nni ~5 + Molmo2 BF16 ~19 + GUI-Actor ~7 (+verifier ~4) + MedGemma BF16 ~10 + Q4 ~5 ≈ 46–50 GB +
activations). See §2a for the revised plan.

**Done (folder structure only; nothing installed, nothing started, nothing run):**

| path | what |
|---|---|
| `~/agentic-segmentation/` | clone of the mount bare repo, branch `phase5-v3` @ `145cd2d` (tracks `origin/phase5-v3`), clean tree. Repo-local git identity set to `Sameed Khan <sameedgkhan@gmail.com>` (same as A10 commits). |
| `~/agentic-segmentation/.env` | **secrets file, mode 600, excluded from git via `.git/info/exclude`.** Holds `HF_TOKEN=` (Sameed fills it in) + path vars (`MODELS_DIR`, `LLAMA_CPP_DIR`, `GUI_ACTOR_DIR`). Nothing reads it automatically: `set -a; source ~/agentic-segmentation/.env; set +a` before `setup_a100.sh` / any MedGemma download. Never print the token, never commit or copy it to the mount. Consider also adding `.env` to `.gitignore` in a commit. |
| `~/agentic-segmentation/data/` | local dir (git-ignored) with symlinks: `data/mstt199 → <mount>/mstt199`, `data/phase4/beamster → <mount>/phase4/beamster`. segd's default `data_root()` is `<repo>/data`, which equals the manifests' recorded root `/home/ubuntu/agentic-segmentation/data`, so **`SEGD_DATA_ROOT` does not need to be set** on this box. If NFS reads are slow for nnInteractive uploads, replace a symlink with a local copy (`rsync -a <mount>/phase4/beamster/nifti_iso/ …` 3.2 GB; mstt199 58 GB) — paths stay the same. |
| `~/agentic-segmentation/results/` | empty; fill with `dvc pull` (remote already configured, below) or copy from `results_a10/`. |
| `~/agentic-segmentation/.dvc/config.local` | overrides remote `local` url → `<mount>/dvc-remote/agentic-segmentation` (git-ignored by `.dvc/.gitignore`). **DVC itself is not installed yet.** |
| `~/models/` | empty; `MODELS_DIR` for `setup_a100.sh` weights. |
| `/opt/llama.cpp`, `/opt/GUI-Actor` | empty dirs chowned to `ubuntu` (script defaults; `git clone` into them works). |
| `~/nni/start.sh` | nnInteractive tmux launcher (16 sessions, port 1527, idempotent). `~/nni/.venv` **not created yet**. |

**Already present on the box:** pi CLI 1.3.0 (`~/.pi/agent/bin/pi`) with packages pi-subagents, pi-anthropic-auth,
pi-web-access; **Sameed is already logged in** (auth.json has anthropic, openai, opencode-go — do not touch/copy it).
Node v22.23.3 + npm (pi's bundled node at `~/.local/share/pi-node/current/bin`), git, tmux, cmake, nvcc 12.8,
Python 3.12.3. **Missing:** `uv`, `dvc`, all venvs / `node_modules`, model weights, llama.cpp build.

**Mount on the A100:** `~/diaphragmatic-endometriosis` → `/lambda/nfs/diaphragmatic-endometriosis`; below,
`<mount>` = `/home/ubuntu/diaphragmatic-endometriosis/agentic-segmentation-data`.

**Phase 3 status seen at setup (04:20 UTC):** `results_a10/.last_mirror` = 04:01; no `SWEEP_COMPLETE` yet;
scheduler heartbeat 04:00; opencode-go was rate-blocked until 04:06. `origin/main` @ `27b8e41` adds (E-log §6
"Phase 3b sweep changes"): **Opus 5 and Gemini 4 Argon cancelled**; order after GPT-6.1-sol high: GPT-6-luna
xhigh, GPT-6-luna max, then a preregistered 30-case GPT-6.1-sol xhigh probe (full run only if ≥ +0.03 vs Opus 5.5
xhigh and Wilcoxon p < 0.05). So the sweep now mostly uses **OpenAI**; opencode-go use is winding down (Muse done
through xhigh). Re-check `git log origin/main` and `results_a10/phase3_state/` when you start.

## 1. Where everything is (network mount)
Everything for this project lives in `<mount>/`:

| path | what |
|---|---|
| `agentic-segmentation.git` | bare git repo = shared remote (`origin` of the A100 clone). Branches: `main` (A10 Phase 3 line, harness-v2.0.1 + logs), `phase5-v3` (**the A100 code line**: harness v3 + Phase 4 harness changes + Phase 5 tooling), `phase4-data` (merged into phase5-v3). Tags harness-v1, harness-v2, harness-v2-rc1, harness-v2.0.1. (Clone warns "remote HEAD refers to nonexistent ref" — harmless.) |
| `mstt199/` | MSTT-199 images + masks (58 GB) — Phase 5 uses MSTT |
| `phase4/beamster/` | BEAMSTER raw + `nifti_iso/` (1 mm iso, 3.2 GB) + `check/` previews + clinical tables; manifest in repo `eval/v4/beamster_cases.csv` |
| `dvc-remote/agentic-segmentation` | copy of the DVC store (all results up to the last mirror) |
| `results_a10/` | live mirror of the A10 `results/` (cron every 30 min; `.last_mirror` = timestamp) — Phase 3 sweep outputs land here. **Read-only for you.** |
| `HANDOFF_A100.md` | the A10's original copy of this handoff (outdated; this file supersedes it) |

Longitudinal-CT and Spinal-MM-SEG are **skipped** (Sameed); UCSD-BMETS blocked (Aspera token). Phase 4 = BEAMSTER only.

## 2. Remaining setup on the A100 (do in order; steps 1–3 of the original list are done, see §0)
1. **Tools:** install `uv` (0.13+; `curl -LsSf https://astral.sh/uv/install.sh | sh`) and DVC (`uv tool install dvc`
   or `pip install --user dvc`). Make sure pi's node is on PATH for npm/tsc (`~/.local/share/pi-node/current/bin`).
2. **Results inputs:** `cd ~/agentic-segmentation && dvc pull` (remote preconfigured via `.dvc/config.local`;
   ~6.9 GB / 16k files) — or copy just what is needed from `<mount>/results_a10/`: Study A needs
   `results/phase2_test/prepare`, `results/phase2_dev/prepare`; Study B needs `results/phase1-0*`,
   `results/candidate_replay`; Study C's paired baseline is `results/phase3-high`.
3. **Python/Node deps:** `cd harness/server && uv sync`; `cd harness/runner && npm ci`; `cd harness/study && npm ci`;
   `cd harness/extension && npm ci` (present).
4. **nnInteractive:** `uv venv ~/nni/.venv && uv pip install --python ~/nni/.venv "nninteractive[server]==2.6.0"`
   (A10 had nninteractive 2.6.0, nninteractive_client 2.6.0, torch 2.14.1 cu13 — driver here is CUDA 13.0, OK).
   Start: `~/nni/start.sh` (16 sessions: Phase 4 and Phase 5 runs at concurrency 8 each).
5. **segd:** `tmux new-session -d -s segd "cd ~/agentic-segmentation/harness/server && uv run segd 2>&1 | tee -a /tmp/segd.log"`
   (no `SEGD_DATA_ROOT` needed, §0). Health: `curl localhost:8765/healthz; curl localhost:1527/healthz`.
6. **Local models (Phase 5):** check `HF_TOKEN` is filled in `.env` (ask Sameed if empty; his HF account must have
   accepted the HAI-DEF terms for `google/medgemma-1.5-4b-it`), then
   `set -a; source .env; set +a; bash harness/local_models/setup_a100.sh --weights-only` (venvs, pinned weights,
   llama.cpp sm_80 build at a pinned commit — needs network; GUI-Actor repo clone). **Do not use the script's
   default "start all services" step on this 40 GB card** — start services selectively per §2a (copy the tmux
   commands from step 5/5 of the script, or run `--services-only` only after editing it to start a subset).
   Ports: MedGemma Q4 :8091, GUI-Actor :8092, Molmo2 :8093, MedGemma BF16 :8094. Then
   `python harness/local_models/smoke_test.py --prepare-dir results/phase2_dev/prepare` (per running service).
   Record llama-server sha256 + `~/models/weights_manifest.json` in run metadata.
7. **Tests before any run:** `cd harness/server && NNI_URL= uv run pytest -q` (expect 187 passed / 2 skipped at
   145cd2d); local_models / study / adherence / multi-scorer suites (152 passed) — run `eval/v2/test_score.py` and
   `harness/study/tests/test_score.py` in **separate** pytest invocations (basename clash); runner `tsc` + node tests.

### 2a. Revised VRAM plan (40 GB) — decide/measure, then log the decision (E30)
- Always on: nni (~1 GB idle, ≤ 5 GB peak with 16 sessions — **measure peak** under Phase 4 load).
- **Study A (standalone localization):** run models **one at a time** next to nni: GUI-Actor (+verifier) ~11 GB,
  MedGemma Q4 ~5, MedGemma BF16 ~10, Molmo2 BF16 ~19 (BF16 for Molmo2 fits alone next to nni; Q4+BF16 MedGemma
  together also fit). Stop a service (`tmux kill-session -t …`) before starting the next; check `nvidia-smi`.
- **Study C (in-harness):** one pointer service (+ optionally one MedGemma) next to nni, as the methodology already
  says ("Study C runs one local model at a time next to nnInteractive"). Molmo2 in-harness: BF16 if it fits with
  nni peak, otherwise the bnb 8-bit variant (~9 GB) — that would be a *change to the arm*, so declare it before runs.
- Phase 4 runs (API models + nni) can proceed concurrently with any of the above as long as nni has headroom.
- Update `research/phase5_implementation.md` §3 and the `setup_a100.sh` header with the 40 GB reality in a commit.

## 3. Division of labour with the A10 orchestrator (Phase 3)
- **A10 agent owns Phase 3** and branch `main` (harness-v2.0.1 + Phase 3 logs, exports, E20–E29). It pushes `main` to
  the mount remote (cron every 30 min + after each export). You never push `main` and never touch Phase 3 results.
- **You own Phases 4–5** and branch `phase5-v3`. Push only that branch (`git push origin phase5-v3`).
- **Experiment log:** to avoid merge conflicts, write your entries in `research/experiment_log_p45.md` (E30, E31, …;
  create it), not in `research/experiment_log.md` (A10 writes E20–E29 there). Same for tables:
  `eval/v2/build_log_tables.py` output can be regenerated after the merge. Commit this handoff
  (`research/HANDOFF_A100_p45.md`, currently untracked) in your first commit; the name avoids an add/add conflict
  with `main`'s `research/HANDOFF_A100.md`.
- **Final merge (you, once `results_a10/phase3_state/SWEEP_COMPLETE` exists and the A10 agent has pushed its last
  commit):** `git fetch origin && git merge origin/main` into `phase5-v3` (expect only research-doc conflicts; append
  `experiment_log_p45.md` into the main log), run the full test suites, then push and fast-forward `main`
  (`git push origin phase5-v3:main`). Until then, merging `origin/main` into your branch is optional (read-only).
- `results_a10/` is for reference only (e.g. a Phase 3 baseline run needed for a paired comparison such as Study C vs
  `phase3-high`); copy what you need into your own `results/`, never modify the mirror.
- **Shared quotas:** both machines use Sameed's same provider accounts. The A10 sweep now uses mainly **OpenAI**
  (GPT-6-luna xhigh/max, GPT-6.1-sol xhigh probe; see §0) and possibly a little opencode-go. Anthropic is not used by
  the sweep → **your agent runs: prefer Anthropic models until the sweep ends.** Your subagents (GLM/DeepSeek via
  opencode-go, gpt-6.1-sol via OpenAI) share those quotas: keep them light; the A10 scheduler switches providers on
  quota limits, so you cannot break it, only slow it.

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
  (pick from the finished sweep: Muse Spark 1.3 high/xhigh / GPT-6.1-sol / GPT-6-luna / DeepSeek max — run after the
  sweep ends, or Muse earlier since opencode-go is now mostly free of the sweep), optional **Opus 5.5 xhigh**
  (effort axis) if budget allows.
- No published BEAMSTER baseline was verifiable; report against a one-click nnInteractive oracle per lesion
  (`eval/v2/oneclick_baseline.py` pattern, adapted to multi-lesion) and inter-run variance.

## 6. Phase 5 protocol
Exactly `research/harness_v3_methodology.md` §3 + `research/phase5_implementation.md`:
Study A (design tuning ≤ 8 prompt variants/model → freeze → test 169), Study B (decision sets from design runs; build
with `harness/study/build_decision_sets.py` — note the 'done' set has only ~10 positives: report CIs, consider a
lenient threshold variant declared in advance), Study C (arms v2 / v3-point / v3-med / v3-both on design with Sonnet
5.5 high, then one frozen v3 on test vs the Sonnet high v2 test run `phase3-high`; adherence via `eval/v2/v3_adherence.py`).
All flags are off by default; v3-off is byte-identical to harness v2 (cross-revision test in tests/test_v3_off_identity.py).
Hardware deviation (40 GB instead of 80 GB → sequential local services, possibly Molmo2 8-bit in-harness) must be
logged in E30 before any Study A/C run.

## 7. Running things in parallel safely
- One nnInteractive server, 16 sessions; segd handles many cases. Each runner invocation uses its own `--run-id`.
- Use `harness/runner/run_until_done.sh` (resume + provider-limit loop) or adapt `eval/v2/sweep_scheduler.py` (queue,
  provider blocks, graceful preemption via `<run>/STOP`, watchdog via cron) for long queues.
- Killing runners leaks nnInteractive sessions; restart segd + nni after kills (the scheduler's `reset_services`).
- The 10-minute wall cap per case (12 min for beamster by default) is a budget: report forced-finish rates.
- GPU contention: local-model services and nni share one 40 GB card — watch `nvidia-smi` for OOM; an nni OOM
  mid-run corrupts Phase 4 cases (re-run them, note in the log).

## 8. Logging / provenance rules
Experiment log entries E30+ in `research/experiment_log_p45.md`, `eval/v2/export_run.py` → `research/experiments/runs/`,
`build_log_tables.py`; git tags for every frozen configuration; preregister before any test-split run; never tune on test.
Commit with thorough messages; push to the mount remote (`git push origin phase5-v3` only). Workers never commit; back up
files before edits; review critical code with `openai/gpt-6.1-sol:high` before results are collected.
Results on the A100 are in `~/agentic-segmentation/results/` (local NVMe). Nothing mirrors them yet: set up a
`dvc add results && dvc push` routine (to `<mount>/dvc-remote`) or an rsync cron to `<mount>/results_a100/` so results
survive the instance.

## 9. Sameed's preferences
Subagent models: well-specified features `opencode-go/glm-5.3-flash:high`; complex `opencode-go/glm-5.3-flash:max`;
running scripts `opencode-go/deepseek-v4.1-flash:high`; reviewer `openai/gpt-6.1-sol:high`; research
`opencode-go/gpt-6-luna`; never Kimi; Sonnet/Opus subagents only when necessary. Use `rg`, never `grep`. Explain codes
in plain words. Concise reports. Reviewers cannot run commands: prepare diff/test artifacts for them in /tmp.
GLM workers sometimes stop mid-task or emit corrupted text: check the worktree and tests yourself, resume or re-spawn.
Secrets: never echo `HF_TOKEN` or auth files into logs, commits, subagent tasks or the mount.

## 10. Status at handoff (2026-10-20, A10 orchestrator; still valid)
- Branch `phase5-v3` @ `145cd2d` on the mount remote = **ready**:
  - Phase 5 tooling (local model services with --fake modes, Study A/B code) — reviewed twice by gpt-6.1-sol, fixed.
  - Harness v3 in segd (seg_locate, MedGemma checkpoints, adherence) — reviewed twice, fixed; v3-off byte-identical
    to main's harness-v2.0.1 segd (cross-revision test `tests/test_v3_off_identity.py`).
  - Phase 4 BEAMSTER changes — reviewed once (6 findings), all fixed with regression tests (no re-review; optional).
  - Tests at 145cd2d: segd 187 passed / 2 skipped; local_models+study+adherence+multi-scorer 152 passed;
    runner tsc + node tests pass. Nothing has run against real local models or real BEAMSTER agent runs yet.
- Phase 4 specifics (CONTRACTS.md 'Phase 4'): `--dataset beamster` (prompts system_multi.md/task_multi.md, budgets
  50 tool calls / 20 prompts / 12 min unless flags given, dense mark profile), `SEGD_DATA_ROOT`, multi-lesion scorer
  `eval/v4/score_multi.py` (union Dice/HD95/NSD, lesion detection ≥ 10 % overlap, FP components, lesion F1, per-lesion
  Dice, size buckets <0.5 / 0.5–2 / >2 mL, `none_detected` tag). Mark coverage (30 design cases, 72 lesions): full-view
  fine marks hit <0.5 mL lesions 0.24, 0.5–2 mL 0.71, >2 mL 0.96; zoomed crop <0.5 mL 0.62; 10/72 lesions have no mark
  even zoomed (a mark-generator change would need a decision). v3 + beamster is refused (no combined prompt) by design.
- Known: pytest basename clash `eval/v2/test_score.py` vs `harness/study/tests/test_score.py` in a single combined run
  (run them separately). Study B 'done' set has only ~10 positives.
- Phase 3 model ranking so far (for the Phase 4 panel): Opus 5.5 xhigh 0.766, Opus medium 0.746, Opus high 0.745,
  Sonnet xhigh 0.737, **Muse Spark 1.3 high 0.718 ($1.7 for 169 cases — strong non-Anthropic candidate)**, Sonnet high
  0.688, Haiku max 0.686, Haiku xhigh 0.661, DeepSeek 0.63. Muse xhigh exported since (see `origin/main` log);
  GPT-6.1-sol high / GPT-6-luna pending (A10); Argon and Opus 5 cancelled.

## 11. First actions for you (suggested)
1. Read the docs listed at the top; `git fetch origin && git log origin/main -5`; check `results_a10/phase3_state/`.
2. Confirm with Sameed that `HF_TOKEN` is in `.env` (check only that it is non-empty; do not print it).
3. §2 steps 1–7 (delegate long installs/builds to a script-running subagent; verify yourself).
4. Write E30 (setup + 40 GB deviation + revised VRAM plan), commit with this handoff, push `phase5-v3`.
5. Start Study A design tuning (local) and Phase 4 design tuning (API + nni) in parallel.
