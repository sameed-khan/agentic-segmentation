# agentic-segmentation

A general-purpose coding agent (pi + Claude) segments soft-tissue tumors in 3D MRI (MSTT-199) by operating the
interactive segmentation model nnInteractive through a small set of `seg_*` tools.

## Where things are

| Path | What | Versioned by |
|---|---|---|
| `research/experiment_log.md` | **Master experiment log**: every run, result, and the decision it informed | git |
| `research/experiments/TABLES.md` | Generated tables (per case / pass / condition) | git (generated) |
| `research/experiments/runs.csv`, `runs/<run_id>/` | Run registry and per-run text records (scores, meta, trajectories, transcripts without images, Phase 2 answers) | git |
| `research/experiments/phase1_design_log.md` | Notes written pass by pass during Phase 1 | git |
| `research/harness_design_v2.md` | Plan, phases, gates (thresholds fixed before data) | git |
| `harness/` | segd daemon (`server/`), pi extension, SDK runner, prompts, Phase 2 study code | git (tag `harness-v1`) |
| `eval/v2/` | Data split, download, data checks, one-click baseline, scorer, mark coverage, export/table scripts | git |
| `results/` | Heavy run artifacts (renders the agent saw, mask versions, final masks, full sessions, mark caches) | **DVC** (`results.dvc`) |
| `data/` | MSTT-199 volumes (`uv run eval/v2/download_mstt.py design data/mstt199`) | neither |
| `archive/` | Retired material (first run, May 2026) | neither |

## Recording a run

```bash
# 1. run (agent passes record git commit/describe + dirty files in batch_summary.json / meta.json)
cd harness/runner && npx tsx run_cases.ts --split design --run-id phase1-06 --thinking high --note "..."
# 2. score
uv run --project harness/server python eval/v2/score.py results/phase1-06
# 3. export text records to git, rebuild registry and tables
uv run --project harness/server python eval/v2/export_run.py results/phase1-06
uv run --project harness/server python eval/v2/build_log_tables.py
# 4. heavy artifacts to DVC, then commit
dvc add results && dvc push && git add -A && git commit -m "run phase1-06: ..."
```

DVC remote `local` = `~/dvc-remote/agentic-segmentation` on the dev server (same disk; add an off-machine remote,
e.g. S3/GCS/SSH, with `dvc remote add` for a real backup). Restore artifacts with `dvc pull`.

## Services (dev server)
- nnInteractive: `~/nni/start.sh` (tmux `nni`, port 1527).
- segd: `cd harness/server && uv run segd` (tmux `segd`, port 8765).
