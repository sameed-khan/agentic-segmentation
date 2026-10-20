# Handoff: Phase 3 (test split) — read this first if you are a revived session

**Task (Sameed, 2026-10-19):** freeze harness as `harness-v2` (done), then execute Phase 3 on the 169 test cases at
thinking **high** (primary), then a second full pass at **xhigh** for comparison. No design-set confirmation pass.
Decision models (Jev / OpenAI Decisions) are skipped for now. Fable is not to be tested (research only, done:
`research/literature/fable_vs_sonnet.md`). Session/rate limits are expected: everything long runs in tmux and resumes.

Preregistration (endpoint, gate, what may not change): `research/experiment_log.md` §6. Do not change `harness/`
(prompts, tools, server, extension) between the high and xhigh passes. Infrastructure fixes in runner/eval are OK;
log them.

## How the work survives a dead session
- `results/phase3_state/` holds one `<step>.done` marker per finished pipeline step.
- The whole pipeline runs in tmux **`p3`**: `eval/v2/phase3_pipeline.sh` (idempotent; skips done steps).
  Log: `/tmp/phase3_pipeline.log`. Restart it any time:
  `tmux kill-session -t p3; tmux new-session -d -s p3 "cd ~/agentic-segmentation && eval/v2/phase3_pipeline.sh 2>&1 | tee -a /tmp/phase3_pipeline.log"`
- Agent passes use `harness/runner/run_until_done.sh` (runner `--resume`: complete cases are skipped, incomplete ones
  moved to `results/<run>/_incomplete/`; on provider error the runner exits 3 and the loop sleeps 20 min, then resumes).
  When a pass is complete, `results/<run>/ALL_DONE` exists.
- Services: segd tmux `segd` (:8765; the loop restarts it if down), nnInteractive tmux `nni` (:1527, `~/nni/start.sh`;
  NOT auto-restarted). Check: `curl localhost:8765/healthz; curl localhost:1527/healthz`.

## On revival: check state
```
ls results/phase3_state/; tail -30 /tmp/phase3_pipeline.log; tmux ls
ls results/phase3-high | wc -l; ls results/phase3-xhigh 2>/dev/null | wc -l   # ~169 case dirs + files when complete
```
If `p3` is not running and steps remain, restart it (command above). If it is running, let it run.

## Pipeline steps (in `eval/v2/phase3_pipeline.sh`)
| step | what | output |
|---|---|---|
| download | `eval/v2/download_mstt.py test` (368 files, ~24 GB) | `data/mstt199/` |
| datacheck | `eval/v2/data_check.py test` (shapes the runner uses) | `eval/v2/data_check/test.jsonl` |
| prepare | Phase 2 inputs + marks cache for all test cases | `results/phase2_test/prepare/`, `results/cache/marks/` |
| oneclick | one click at GT centroid, nnInteractive only | `results/oneclick_test/` |
| agent_high | agent pass, thinking high | `results/phase3-high/` |
| score_high | `eval/v2/score.py` | `results/phase3-high/scores.csv` |
| phase2 | localization study R,S,S-gen ×3, thinking high | `results/phase2_test/p2-test-high/` |
| agent_xhigh | agent pass, thinking xhigh | `results/phase3-xhigh/` |
| score_xhigh | scorer | `results/phase3-xhigh/scores.csv` |

## Manual work after the pipeline (orchestrator, not in the script)
1. After `score_high`: per-histotype mean Dice vs U-Net (F 0.507, L 0.744, M 0.823, N 0.748, V 0.584; mean 0.681),
   gate check (unweighted mean ≥ 0.681 and > U-Net on ≥ 3/5), medians, localization failures, candidate use, cost,
   one-click comparison, 15 multi-rater cases vs inter-rater Dice. Write E17 in `research/experiment_log.md`.
2. `eval/v2/build_log_tables.py` only tables runs with exactly 30 cases (line ~45): add a Phase 3 section for 169-case
   runs (this is a known TODO), then export: `uv run --project harness/server python eval/v2/export_run.py results/phase3-high`.
3. `dvc add results && dvc push`, `git add -A && git commit` (orchestrator commits; workers never commit).
4. Same after `score_xhigh` (E18: paired per-case xhigh vs high, Wilcoxon), and Phase 2 test (E19).
5. Report to Sameed. Phase 4 is not authorised.

## Preferences (from Sameed)
Subagent models: well-specified features `opencode-go/glm-5.3-flash:high`; complex `opencode-go/glm-5.3-flash:max`;
running scripts `opencode-go/deepseek-v4.1-flash:high`; reviewer `openai/gpt-6.1-sol:high`; never Kimi; Sonnet/Opus
subagents only when necessary. Use `rg`, never `grep`. Explain codes in plain words. Concise reports.

## Log of this phase (append as you go)
- 2026-10-19: tag `harness-v2` (runner resume infra on top of f3dfede); preregistration f99604f; test download started
  (tmux `dl`, `/tmp/dl_test.log`); worker generalising prepare/ask/score/oneclick to the test split.
- 2026-10-19: worker finished test-split infra; committed; `harness-v2` re-tagged on that commit (agent-facing files
  unchanged from f3dfede). Killed tmux `dl`; pipeline started in tmux `p3` (download step resumes the download).
- 2026-10-19 06:40: V25 blocked segd 11 min (scipy opening r=14 mm). Fixed with bit-identical EDT morphology
  (`segd.ops.ball_morphology`, test `tests/test_ball_morphology.py`), tag `harness-v2.0.1`, see experiment_log §6
  deviation log. segd restarted. V25 parked in `results/phase3-high/_incomplete/V25_api_*` and rerun. Pipeline restarted.
- 2026-10-19 ~07:10: high pass complete and scored; E17 written (gate NOT met: 2/5 histotypes; unweighted 0.688);
  exported; committed. Pipeline continues: phase2 (test localization study) then agent_xhigh.
- ~07:35: phase2 test done (E18; path bug fixed, marker set by hand). Pipeline restarted -> agent_xhigh running.
- ~09:10: xhigh pass complete; E19 written (xhigh 0.737, 5/5 above U-Net; high 0.688, 2/5). TABLES §6 added. Phase 3 DONE; reported to Sameed. Phase 4 not authorised.
- 2026-10-19T12:22:42+00:00: model sweep started (tmux p3b, eval/v2/model_sweep.sh, /tmp/model_sweep.log). Dry-stub smokes OK for haiku max, gpt-6.1-sol, gpt-6-luna max, deepseek-v4.1-flash max; muse-spark-1.3 BLOCKED (opencode-go: model trains on request data; needs workspace opt-in from Sameed) -> restart sweep with SKIP='p3-muse13-high p3-muse13-xhigh' if not resolved. Researcher (gpt-6-luna) writing research/literature/local_visuospatial_vlms.md.

## Phase 3b: model sweep (Sameed 2026-10-19 ~12:30 UTC; full autonomy while he is away)
Runs (order): Opus 5.5 medium/high/xhigh, Haiku 5.5 xhigh/max, DeepSeek v4.1 flash xhigh/max, Muse Spark 1.3
contributor high/xhigh (opencode-go opt-in enabled by Sameed), GPT-6.1-sol high/xhigh, GPT-6-luna xhigh/max,
Gemini 4 Argon contributor high (opencode-go; ~440 requests / 5 h, so last of opencode-go), Opus 5 at the best
Opus 5.5 thinking level. One configuration at a time, concurrency 8 (nnInteractive `--max-sessions 8`, ~/nni/start.sh).
Directive: follow the order; ONLY when a provider hits its quota, switch to the next run of another provider and
come back when its window refreshes (interleave to maximise throughput).
- Scheduler: `eval/v2/sweep_scheduler.py` in tmux **p3a**, log `/tmp/sweep_sched.log`, (re)start:
  `eval/v2/start_sweep.sh`. State: `results/phase3_state/sweep_<run>.done`, `sched.json` (provider blocks, 15 min).
  Provider limit (runner exit 3) -> block provider, run next; every 60 s it preempts a lower-priority run when an
  earlier run's provider is free again. Every kill/preempt/infra failure -> restart segd + nnInteractive (killed
  runners leak nnInteractive sessions -> "server at capacity (8 sessions)" -> segd HTTP 500; happened 12:42).
- Runner resume: complete = done, no errors, no provider errors, final mask. Give up on a case only after 3 parked
  genuine agent failures (killed/infra attempts never count; fixed 12:45 after kills were counted).
- After each run is done: analyse vs phase3-high/xhigh (per histotype, gate, loc. fails, tools, tokens/cost), write
  E-entry, export_run.py, build_log_tables.py, dvc add/push, commit. Report to Sameed when he is back.
- 13:14 UTC: p3-opus55-medium done (unweighted 0.746, 5/5 > U-Net, 19 loc fails, $28), exported. Narrative E20 (whole sweep) to write at the end; use eval/v2/sweep_summary.py.
- 14:40 UTC: p3-opus55-high done (0.745, 4/5), exported. Sameed: no more Anthropic runs until the reset 17:40 UTC (1:40 PM EDT) -> manual block in sched.json; scheduler now re-reads sched.json each loop. DeepSeek v4.1 flash xhigh running (40/169).
- 15:17 UTC: harness v3 methodology drafted (research/harness_v3_methodology.md) + literature notes; awaiting Sameed's go on deployment/studies.
- 20:11 UTC: Sameed: sweep STOPPED (scheduler + runner killed; segd/nni left up). Do not restart without his go. Restart = eval/v2/start_sweep.sh.

## 2026-10-19 ~20:20 UTC: Sameed's directives (full autonomy, he stepped away)
- Sweep restarted (Phase 3 stays on this A10 machine). Order: opus55 m/h/x > deepseek x/max > haiku x/max > muse
  h/x > gpt-6.1-sol h/x > gpt-6-luna x/max > argon high > opus5 (best opus55 level). On a quota limit: retry twice
  after 5 min, then switch providers; back off 15->240 min; graceful preemption (STOP file) when an earlier run's
  provider is free. **Self-healing:** cron `*/10 * * * * eval/v2/sweep_watchdog.sh` restarts the scheduler if dead or
  heartbeat > 40 min. To stop the sweep: `touch results/phase3_state/STOPPED` then kill tmux p3a.
  When everything is done: `results/phase3_state/SWEEP_COMPLETE`.
- Phase 4 data prep: worker in worktree `/home/ubuntu/agseg-p4` (branch phase4-data): Spinal-Multiple-Myeloma-SEG,
  BEAMSTER (+1–2 more post-2026-06-30 3D lesion datasets, non-MSK). Data -> data/phase4/.
- Phase 5 (harness v3, runs on an A100 later): worker in worktree `/home/ubuntu/agseg-p5` (branch phase5-v3): local
  model services (GUI-Actor-2B, Molmo2-8B, MedGemma 1.5 Q4_K_M + BF16), Study A/B code with fake backends.
  Skip BiomedParse v2 and Qwen3-VL-8B. Study C agent = Sonnet 5.5 high, confirm with Opus 5.5 medium.
  Next after the worker: review (gpt-6.1-sol), merge branch, then segd v3 integration (seg_locate + MedGemma checkpoints).
- Never let workers edit the main tree while the sweep runs (segd restarts load main-tree code).
- 22:05 UTC: Phase 5 tooling reviewed twice by gpt-6.1-sol; fixes committed on branch phase5-v3 (c025c27, 123 tests). NOT merged into main while the sweep runs (would change harness hash mid-sweep; Phase 5 runs on the A100 anyway). segd v3 integration worker started in agseg-p5.
- 22:21 UTC: Phase 4 worker 1 timed out (2 h). State: BEAMSTER downloaded + manifest (140; 30 design/110 test); Longitudinal-CT (autoPET/CT-IV, melanoma whole-body CT, FDAT 2026-09-18, CC BY-NC) downloaded 99 GB + manifest (334); Spinal-MM-SEG: DICOM 304 GB not fetched — but NBIA also serves 144 DICOM SEG (6.6 GB) -> new worker downloads conventional CT (37 GB) + SEG and converts; UCSD-BMETS blocked (Aspera token). Worker 2 running in agseg-p4.
- 00:22 UTC: Phase 5 v3 integration reviewed x2 + fixed (branch phase5-v3 f25d687; 178+142 tests; cross-revision v3-off byte identity verified vs main segd). Phase 4 worker 2 timed out; branch phase4-data committed (WIP); iso conversion chain running (nohup), then /tmp/p4_finish.sh repoints manifests + data checks (log agseg-p4/eval/v4/logs_finish.txt; 'DONE' line when finished). Then: write Phase 4 summary table into research/phase4_datasets.md.
- 01:11 UTC: haiku55-max done (0.686). Phase 4 data IS downloaded (local data/phase4, ~340 GB incl. raw); copying phase4 + mstt199 to network mount ~/diaphragmatic-endometriosis/agentic-segmentation-data/ (tmux sync, /tmp/sync_mount.log, waits for p4_finish data checks). Manifests use absolute /home/ubuntu/agentic-segmentation/data paths -> need a DATA_ROOT remap on the A100.
- 01:36 UTC: Sameed: drop Opus 5 -> marker sweep_p3-opus5-best.done contains 'skipped' (no scores exist; not a result).
- 03:14 UTC: Phase 4 harness changes committed on phase5-v3 (1c8de32; 186+148 tests; manifest image/mask column swap fixed); pushed to mount; gpt-6.1-sol review running. A10 owns Phase 3 + main; A100 agent owns phase5-v3 (Phases 4-5).
- 03:29 UTC: phase5-v3 @145cd2d pushed (Phase 4 fixes). HANDOFF_A100.md finalized (also at mount root). A10 now: Phase 3 only.
- 05:16 UTC: gpt61sol-high done (0.777, best; +0.011 vs Opus xhigh p=0.10), exported. Scheduler: sol-xhigh probe/full stop the sweep on quota (no waiting) -> then Phase 6 = harness v2 + SAM-Med3D-turbo instead of nnInteractive (Sonnet 5.5 high, 169 test, paired vs phase3-high; large tumours: overlapping 128^3 patches, option a). Build worker in worktree /home/ubuntu/agseg-p6 (branch phase6-sammed3d, from main).
