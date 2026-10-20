# agentic-segmentation

A general-purpose coding agent (pi + a frontier LLM) segments tumors in 3D MRI (MSTT-199 soft-tissue tumors; BEAMSTER
brain metastases) by operating an interactive segmentation model through a small set of `seg_*` tools (nnInteractive;
SAM-Med3D-turbo as a swap-in segmenter).

**One main line.** Every experiment since Phase 3 is reproduced by *configuration*, not by checking out a branch:

| Optional feature (all independent flags) | Default (= frozen Phase 3 harness, tag `harness-v2.0.1`) | Switch |
|---|---|---|
| harness v3: `seg_locate` pointer tool, MedGemma second-reader checkpoints | off | runner `--v3-pointer/--v3-medgemma/--v3-checkpoints` (case-open `v3`) |
| dataset: MSTT-199 or BEAMSTER (multi-lesion prompts, dense marks, component review) | `mstt` | runner `--dataset beamster` |
| segmenter: nnInteractive or SAM-Med3D-turbo | `nninteractive` | runner `--segmenter sammed3d` |

`harness/server/CONTRACTS.md` §7 is the flag reference; `harness/server/tests/test_identity.py` proves the defaults are
byte-identical to `harness-v2.0.1` and that each flag path equals the branch it came from. A logged experiment is a **preset**
(`experiments/presets/<name>.json`): `npx tsx run_cases.ts --preset p3-opus55-high` (explicit flags override it).
`research/REPRODUCE.md` maps every experiment → preset → git tag → `results/<run>.dvc` → log entry.

## Where things are

| Path | What | Versioned by |
|---|---|---|
| `research/experiment_log.md` | **Master experiment log** (E1–E22 + preregistrations); E30–E31 (Phases 4–5, A100) in `research/experiment_log_p45.md` | git |
| `research/REPRODUCE.md` | Experiment → preset → tag → results artifact → log entry | git |
| `experiments/presets/*.json` | Agent-run presets: Phase 3 (`phase3-*`), model sweep (`p3-*`), Phase 6 (`p6-*`) — model, thinking, cases, budgets, concurrency, dataset, segmenter, v3, prompts, provenance | git |
| `experiments/presets/studies/*.json` | Command presets of the Phase 2/5 study scripts (pointing, Study A/B, 3D probes) | git |
| `experiments/run_preset.py` | List/show/run presets (`run <name> [--dry-run]`) | git |
| `research/experiments/TABLES.md`, `runs.csv`, `runs/<run_id>/` | Generated tables, run registry, per-run text records (scores, meta, trajectories, transcripts without images) | git |
| `research/harness_design_v2.md`, `harness_v3_methodology.md`, `phase4_*.md`, `phase5_implementation.md`, `phase6_sammed3d.md` | Plans, phases, gates, implementation notes; `HANDOFF_*.md` = historical session handoffs | git |
| `harness/server/` | segd daemon (src layout `segd`), `CONTRACTS.md` (interface + flag reference), tests incl. cross-revision identity | git |
| `harness/extension/`, `harness/runner/` | pi extension (`seg_*` tools; `seg_locate` only with a v3 pointer) and SDK batch runner (`run_cases.ts`, `cli_args.ts`, `presets.ts`) | git |
| `harness/prompts/` | `system.md`/`task.md` (default), `system_v3.md`, `system_multi.md`/`task_multi.md` (BEAMSTER), `medgemma/*.md` (v3 second reader) | git |
| `harness/local_models/` | Local model services: `pointd_*` (GUI-Actor, Molmo2, MolmoPoint, Qwen3-VL), `medgemma`, `sammed3d` (segmenter service), `probe3d` (SAT, BiomedParse) | git |
| `harness/study/` | Phase 2/5 single-turn studies: pointing (`ask.ts`), Study A (`local_point_study.py`), Study B (decision sets/replays), scorers | git |
| `eval/v2/`, `eval/v4/` | Splits, data checks, scorer (`score.py`), one-click baseline, sweep tooling; BEAMSTER manifests/scorer (`score_multi.py`) | git |
| `results/` | Heavy run artifacts (renders, mask versions, masks, sessions) — one `results/<run>.dvc` per run | **DVC** |
| `data/` | MSTT-199 (`uv run eval/v2/download_mstt.py design data/mstt199`) and Phase 4 data (`eval/v4/download_*.py`); or any tree via `SEGD_DATA_ROOT` | neither |
| git tags | `harness-v1`, `harness-v2-rc1`, `harness-v2`, `harness-v2.0.1`, `harness-v2-p6` | git |

**Earlier phases (E1–E16, harness v1 and its development) are reproducible via git tags only** (`harness-v1` →
`harness-v2-rc1` → `harness-v2`); they predate the flag surface. See `research/REPRODUCE.md` for the per-run tag table.

## Reproducing an experiment

```bash
# services: nnInteractive (:1527) and segd (:8765); SAM-Med3D (:8095) only for segmenter sammed3d
cd harness/server && uv sync && uv run segd &            # tmux segd
cd harness/runner && npm ci && cd ../extension && npm ci
# data: download scripts (or symlink data/ / SEGD_DATA_ROOT=/path/to/data); run outputs from DVC: dvc pull results/<run>.dvc
cd harness/runner
npx tsx run_cases.ts --list-presets
npx tsx run_cases.ts --preset phase3-high                         # run id = preset name, results/phase3-high/
npx tsx run_cases.ts --preset p3-opus55-high --cases F02,F06 --run-id try-01   # explicit flags override the preset
npx tsx run_cases.ts --preset p6-sammed3d-high --segd-url http://127.0.0.1:8766   # machine settings are never in a preset
# study presets (Phase 2/5 scripts):  python experiments/run_preset.py run dA-molmo2-marks-p4 [--dry-run]
```

Environment variables: `SEGD_DATA_ROOT` (data tree, default `<repo>/data`; BEAMSTER manifest paths are remapped onto it),
`NNI_URL` (live nnInteractive for the segd tests; empty = FakeNNI), `SAMED3D_SRC` (sammed3d `src/` if not the sibling
`harness/local_models/sammed3d`), `AUTH_SHIM` (pi Anthropic auth shim package dir). Provider credentials: `~/.pi/agent/auth.json`.
Local-model services (v3 pointers/MedGemma, Studies) take their ports from `harness/local_models/*` (8091–8096; note the
Study-A-only MolmoPoint server and the SAM-Med3D service both default to :8095 — do not run them together without `--port`).

## Recording a run

```bash
# 1. run (agent passes record git commit/describe + dirty files + preset name in batch_summary.json / meta.json)
cd harness/runner && npx tsx run_cases.ts --preset <name>           # or explicit flags: --split design --run-id phase1-06 --thinking high --note "..."
# 2. score
uv run --project harness/server python eval/v2/score.py results/<run>        # BEAMSTER: eval/v4/score_multi.py
# 3. export text records to git, rebuild registry and tables
uv run --project harness/server python eval/v2/export_run.py results/<run>
uv run --project harness/server python eval/v2/build_log_tables.py
# 4. heavy artifacts to DVC (one .dvc per run: eval/v2/dvc_perrun.sh), then commit
```

A new experiment = a new preset JSON (copy a similar one; fill `provenance`), a log entry, and a row in `research/REPRODUCE.md`.
DVC default remote = the network mount; restore artifacts with `dvc pull`.

## Tests

```bash
(cd harness/server && NNI_URL= uv run pytest -q)                  # segd incl. cross-revision identity (needs git history/tags)
uv run --project harness/server python -m pytest harness/local_models/tests harness/study/tests eval/v2 eval/v4 experiments/tests -q
PYTHONPATH=harness/local_models/sammed3d/src <python with torchio+pytest> -m pytest harness/local_models/sammed3d/tests   # service project env
(cd harness/runner && npx tsc --noEmit -p . && node --test --import tsx test/*.test.mts)
(cd harness/extension && ../runner/node_modules/.bin/tsx test_v3_tools.ts)
```

`eval/v2/test_score.py::test_score_case_end_to_end` needs the MSTT-199 data (skips otherwise).

## Services (dev server)
- nnInteractive: `~/nni/start.sh` (tmux `nni`, port 1527).
- segd: `cd harness/server && uv run segd` (tmux `segd`, port 8765).
- SAM-Med3D (segmenter swap only): `harness/local_models/sammed3d` (`sammed3d-service`, port 8095).


## Starting from scratch (after the GPU instances are gone)
Everything lives in two places: **code + logs** on GitHub (`git@github.com:sameed-khan/agentic-segmentation.git`),
**data + results** in Backblaze B2 bucket `sk-research-data`:

| B2 prefix | content |
|---|---|
| `agentic-segmentation-data/dvc-remote/agentic-segmentation/` | DVC store: every run as its own artifact (`results/<run>.dvc`) |
| `agentic-segmentation-data/data/` | MSTT-199 (`mstt199/`) and BEAMSTER (`phase4/beamster/`) images + masks |
| `agentic-segmentation-data/agentic-segmentation.git/` | bare git mirror (all branches + tags) — backup of GitHub |
| `agentic-segmentation-data/archive/` | A100 + A10 session transcripts, logs, machine rebuild info |

1. `git clone git@github.com:sameed-khan/agentic-segmentation.git && cd agentic-segmentation`
2. Install DVC with S3 support (`uv tool install "dvc[s3]"` or `pip install "dvc[s3]"`). DVC remote `b2` is in `.dvc/config` (S3 endpoint `https://s3.us-east-005.backblazeb2.com`). Credentials: a B2
   application key → `dvc remote modify --local b2 access_key_id <keyID>` and
   `dvc remote modify --local b2 secret_access_key <applicationKey>` (never commit them).
3. Pull only what you need, e.g. `dvc pull -r b2 results/phase3-high.dvc results/p3-gpt61sol-high.dvc`
   (or everything: `dvc pull -r b2`). Text records for most analyses are already in git:
   `research/experiment_log.md` (E1–E22, master narrative), `research/experiment_log_p45.md` (E30–E31),
   `research/experiments/TABLES.md`, `research/experiments/sweep_summary.md`, `research/experiments/runs/<run>/`
   (scores.csv, per-case records), `research/REPRODUCE.md` (experiment → preset → tag → artifact).
4. Images for re-scoring/figures: `rclone copy b2:sk-research-data/agentic-segmentation-data/data ./data` (or
   re-download via `eval/v2/download_mstt.py`), then `export SEGD_DATA_ROOT=$PWD/data`.
Re-scoring (`eval/v2/score.py results/<run>`) and figure making run on CPU.
