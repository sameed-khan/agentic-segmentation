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
