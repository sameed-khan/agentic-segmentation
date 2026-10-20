# Reproducing the logged experiments

One main line, behaviour by flags: every experiment from Phase 3 on is a **preset** (`experiments/presets/<name>.json`)
that records model, thinking level, case selection, budgets, concurrency, dataset, segmenter, v3 flags, system/task
prompt files and the original git tag/commit. Run it by name; explicit flags override it.

```bash
(cd harness/runner && npx tsx run_cases.ts --list-presets)
(cd harness/runner && npx tsx run_cases.ts --preset p3-opus55-high)        # the logged configuration, run id = preset name
(cd harness/runner && npx tsx run_cases.ts --preset p3-opus55-high --run-id retry-1 --cases F02,F06)   # override explicitly
harness/runner/run_preset_until_done.sh p3-opus55-high              # resumable across provider limits (--resume)
python experiments/run_preset.py run phase3-high --dry-run          # launcher: prints the command (also runs study presets)
```

Setup: `uv sync` in `harness/server`, `npm ci` in `harness/runner` + `harness/extension` (+ `harness/study` for the study
presets); nnInteractive on `:1527` and segd on `:8765` (`cd harness/server && uv run segd`); data volumes via the download scripts
(`eval/v2/download_mstt.py`, `eval/v4/download_beamster.py`) into `data/` (or any existing tree: symlink or `SEGD_DATA_ROOT`),
run outputs via DVC (`dvc pull results/<run>.dvc`); provider credentials
in `~/.pi/agent/auth.json` (+ the `@gotgenes/pi-anthropic-auth` shim for Anthropic OAuth). Phase 6 additionally needs
the SAM-Med3D service on `:8095` (`harness/local_models/sammed3d`; checkpoint via `scripts/fetch_checkpoint.py`).

**Check a reproduction** against the original: `uv run --project harness/server python eval/v2/score.py results/<run>`
and compare with `research/experiments/runs/<run>/scores.csv` (exported text records; the heavy artifacts are the
`results/<run>.dvc` files). LLM runs are stochastic: expect the same distribution, not identical masks.

Why the presets are trustworthy: the default flags reproduce the frozen Phase 3 harness byte-for-byte
(`harness/server/tests/test_identity.py`, tag `harness-v2.0.1`), the segmenter and v3/BEAMSTER paths are checked against the
branches they came from, and `experiments/tests/test_presets.py` + `harness/runner/test/presets.test.mts` check every
preset against the committed run exports (`research/experiments/runs/<run>/batch_summary.json`) and the repo files.

## Agent runs (preset → code → results → log)

Code column: the git tag the original run used (plus the exact commit its `batch_summary.json` recorded; sweep runs
were launched from a working tree that was `harness-v2.0.1` plus runner/scheduler commits only — the agent-facing
prompts, tools and segd were identical). All MSTT-199, budgets 40 tool calls / 15 model prompts / 10 min wall / 3 nudges,
`dataset=mstt`, v3 off. Log entries: E17 Phase 3 primary, E19 second pass, E20 model sweep, E21/E22 Phase 6.

| Preset | Model | Thinking | Cases | Conc. | Segmenter | Code | Results artifact | Log |
|---|---|---|---|---|---|---|---|---|
| `phase3-high` | claude-sonnet-5-5 | high | test split (169) | 4 | nninteractive | `harness-v2.0.1` @`0b22d7a` | `results/phase3-high.dvc` | E17 (+ §6 preregistration) |
| `phase3-xhigh` | claude-sonnet-5-5 | xhigh | test split (169) | 4 | nninteractive | `harness-v2.0.1` @`2bcff01` | `results/phase3-xhigh.dvc` | E19 |
| `p3-argon4-high` | gemini-4-argon-contributor | high | test split (169) | 8 | nninteractive | `harness-v2.0.1` @`0153494` | `results/p3-argon4-high.dvc` | E20 |
| `p3-dsv41flash-max` | deepseek-v4.1-flash | max | test split (169) | 8 | nninteractive | `harness-v2.0.1` @`0b2d65c` | `results/p3-dsv41flash-max.dvc` | E20 |
| `p3-dsv41flash-xhigh` | deepseek-v4.1-flash | xhigh | test split (169) | 8 | nninteractive | `harness-v2.0.1` @`dcb840c` | `results/p3-dsv41flash-xhigh.dvc` | E20 |
| `p3-gpt61sol-high` | gpt-6.1-sol | high | test split (169) | 8 | nninteractive | `harness-v2.0.1` @`27b8e41` | `results/p3-gpt61sol-high.dvc` | E20 |
| `p3-gpt61sol-xhigh` | gpt-6.1-sol | xhigh | 30 cases | 8 | nninteractive | `harness-v2.0.1` @`06fe6ba` | `results/p3-gpt61sol-xhigh.dvc` | E20 |
| `p3-gpt6luna-max` | gpt-6-luna | max | test split (169) | 8 | nninteractive | `harness-v2.0.1` @`5017fd0` | `results/p3-gpt6luna-max.dvc` | E20 |
| `p3-gpt6luna-xhigh` | gpt-6-luna | xhigh | test split (169) | 8 | nninteractive | `harness-v2.0.1` @`1dcc622` | `results/p3-gpt6luna-xhigh.dvc` | E20 |
| `p3-haiku55-max` | claude-haiku-5-5 | max | test split (169) | 8 | nninteractive | `harness-v2.0.1` @`8ee9f4f` | `results/p3-haiku55-max.dvc` | E20 |
| `p3-haiku55-xhigh` | claude-haiku-5-5 | xhigh | test split (169) | 8 | nninteractive | `harness-v2.0.1` @`0b2d65c` | `results/p3-haiku55-xhigh.dvc` | E20 |
| `p3-muse13-high` | muse-spark-1.3-contributor | high | test split (169) | 8 | nninteractive | `harness-v2.0.1` @`a9bc910` | `results/p3-muse13-high.dvc` | E20 |
| `p3-muse13-xhigh` | muse-spark-1.3-contributor | xhigh | test split (169) | 8 | nninteractive | `harness-v2.0.1` @`e1a2859` | `results/p3-muse13-xhigh.dvc` | E20 |
| `p3-opus55-high` | claude-opus-5-5 | high | test split (169) | 8 | nninteractive | `harness-v2.0.1` @`bb23453` | `results/p3-opus55-high.dvc` | E20 |
| `p3-opus55-medium` | claude-opus-5-5 | medium | test split (169) | 8 | nninteractive | `harness-v2.0.1` @`15c519a` | `results/p3-opus55-medium.dvc` | E20 |
| `p3-opus55-xhigh` | claude-opus-5-5 | xhigh | test split (169) | 8 | nninteractive | `harness-v2.0.1` @`0efa25b` | `results/p3-opus55-xhigh.dvc` | E20 |
| `p6-sammed3d-high` | claude-sonnet-5-5 | high | test split (169) | 6 | sammed3d | `harness-v2-p6` @`f7b40e8` | `results/p6-sammed3d-high.dvc` | E21 (preregistration), E22 (result) |
| `p6-sammed3d-pilot5b` | claude-sonnet-5-5 | high | 5 cases | 2 | sammed3d | `harness-v2-p6` @`f7b40e8` | `results/p6-sammed3d-pilot5b.dvc` | E22 |

Notes:
- `phase3-high` ran on tag `harness-v2`; V25 was rerun on `harness-v2.0.1` after the morphology fix (bit-identical
  operations, E17 deviation log), so `harness-v2.0.1` reproduces the table.
- `p3-gpt61sol-xhigh` is the preregistered **30-case probe** (6 test cases per histotype, `probe_cases.json`); it failed
  its gate and the full run was skipped (E20). `p3-argon4-high` aborted at once (provider API unsupported); kept for
  completeness. The scheduler `eval/v2/sweep_scheduler.py` that launched the sweep is historical; presets replace its run list.
- `p6-sammed3d-high` is `--segmenter sammed3d` (concurrency 6); the original run used a second segd on `:8766` — pass
  `--segd-url` for your machine. Its tag `harness-v2-p6` is the Phase 6 branch tip; the integrated code is
  identity-tested against it (`test_identity.py::test_sammed3d_path_equals_phase6_branch`).
- Flags available but not yet used by any logged experiment: `--v3-*` (Study C was designed, not run) and
  `--dataset beamster` (Phase 4 was skipped); both are covered by the equivalence tests against `phase5-v3`.
  `v3` + `beamster` together is refused.

## Phase 2 / Phase 5 studies (command presets, `experiments/presets/studies/`)

Presets of kind `study` hold the exact command lines (script, flags, prompt variants) of the single-turn studies and are run
with `python experiments/run_preset.py run <name>` (`--dry-run` prints them). Provenance = git commit of the original run
(all on this line after the merge). Local-model services must be up (`harness/local_models/`, ports in their READMEs).

| Experiment | Presets | Script | Results artifact | Log |
|---|---|---|---|---|
| Phase 2 localization on test, Sonnet 5.5 high (R/S/S-gen, 3 reps) | `p2-test-high` | `harness/study/ask.ts` | `results/phase2_test.dvc` | E18 |
| Frontier pointing reference for Study A: Opus 5.5 medium (design / test) | `dev-opus55-medium`, `p2-test-opus55-medium` | `ask.ts` | `results/phase2_dev.dvc`, `results/phase2_test.dvc` | E31 |
| Study A: standalone localization by local models, per model × condition × prompt variant (46 runs) | `dA-<model>-<condition>-<variant>` (gui_actor, molmo2, medgemma_q4, medgemma_bf16) | `harness/study/local_point_study.py` + `score_local_point.py` | `results/phase5.dvc` (`studyA/`) | E31 |
| Study B: decision sets | `studyB-build-sets` | `build_decision_sets.py` | `results/phase5.dvc` (`studyB/*/manifest.jsonl`) | E31 |
| Study B: replays (frontier Sonnet high / Opus medium; MedGemma Q4/BF16) × {candidate_judgement, done_check, wrong_structure} (12 runs) | `studyB-<set>-<run>` | `replay_decisions.ts` / `replay_local.py` + `score_decision.py` | `results/phase5.dvc` (`studyB/`) | E31 |
| 3D probes: SAT-Nano, BiomedParse v2 and scorer | `probe3d-sat`, `probe3d-biomedparse`, `probe3d-score` | `harness/local_models/probe3d/` | `results/phase5.dvc` (`probe3d/`) | E31 |

MolmoPoint / Qwen3-VL Study A runs: run files were lost with the workers' worktrees (E31); scores are recorded in the log
and `results/phase5/SUMMARY.md`, and the scripts support them (`--models` via the service ports), but no preset exists.

## Earlier phases: git tags only (harness v1, Phase 1–2 development)

Experiments E1–E16 (design split, harness v1 and its development) are **not** preset-reproducible: they predate the flag
surface and were run on code states that differ in agent-facing prompts and tools. Reproduce them by checking out the
tag (`git worktree add ../agseg-<tag> <tag>`), following that tag's `README.md`/runner usage, and comparing with the
exports in `research/experiments/runs/<run>/`.

| Experiment(s) | Runs (`results/<run>.dvc`) | Code | Log |
|---|---|---|---|
| E1 one-click baseline (no LLM) | `oneclick_design`, `oneclick_design_colab`, `oneclick_test` | `eval/v2/oneclick_baseline.py` (`harness-v1`) | E1, E17 |
| E3–E9 integration + Phase 1 passes 1–5 | `integ-02`, `phase1-01` … `phase1-05` | pre-tag history; `harness-v1` (`6fb08a5`) = state after pass 5; `harness_sha256` in `runs/<run>/batch_summary.json` identifies the tree | E3, E5–E9 |
| E4, E10 Phase 2 localization (design) | `phase2_dev/dev-01`, `dev-02-*`, `prepare`, `smoke2` | `harness-v1` (+1 commit `51b7064`) | E4, E10 |
| E11 thinking levels (harness v1) | `phase1-06-high`, `phase1-07-xhigh`, `phase1-08-max` | `harness-v1` @`51b7064` | E11 |
| E12 example image in task | `phase1-09-xhigh-ex` | `harness-v1` @`da0ce0b` (`--examples`) | E12 |
| E13 oracle refinement ceiling | `oracle_refine` | `eval/v2/oracle_refine.py` (script is on the main line) | E13 |
| E14 candidate generators (offline) | `candidate_bench` | `eval/v2/candidate_bench.py`, `segd/candidates.py` (on the main line) | E14 |
| E15 agent with candidate corrections | `phase1-10-high-cand` | tag `harness-v2-rc1` (`dd8e851`) | E15 |
| E16 candidate-decision replay | `candidate_replay` | `harness/study/replay_candidates.ts` @`harness-v2-rc1` | E16 |
| Smoke / lane runs | `smoke-*`, `laneA_*`, `_smoke_phase4`, `cache` | — (infrastructure checks; `cache` = mark caches) | — |

The default flags of the integrated line are the Phase 3 harness (`harness-v2.0.1`), not harness v1: `harness-v1` →
`harness-v2-rc1` → `harness-v2` → `harness-v2.0.1` added candidate corrections (`seg_candidates`/`seg_accept`), the lesion
line, `--resume` and the fast morphology — those are in every preset above.
