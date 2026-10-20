# Handoff: Phase 4/5 A100 line PAUSED (2026-10-20 ~08:20 UTC)

Read first: `research/experiment_log_p45.md` (E30 setup, E31 results + decision), `results/phase5/SUMMARY.md`.

## State at pause
- Decision (Sameed): no Phase 5 candidate oracle passed the Study C gate → Study A test run, Study C and Phase 4 are
  NOT run; project paused. Branch `phase5-v3` (mount bare repo) holds all code + logs; NOT merged into `main`
  (the A10 owns `main`; it was running the Phase 3 sweep tail and started a Phase 6 = SAM-Med3D-turbo build).
- All jobs and services were stopped before shutdown.

## Where everything is on the mount (`<mount>` = /home/ubuntu/diaphragmatic-endometriosis/agentic-segmentation-data)
| what | where |
|---|---|
| code + logs (git) | `<mount>/agentic-segmentation.git` branch `phase5-v3`; full bundle `<mount>/a100_backup/agentic-segmentation-phase5-v3.bundle` |
| A100 results (Phase 5, Phase 2 frontier runs made here) | `<mount>/results_a100/` (`phase5/`, `phase2_dev/`, `phase2_test/`) |
| complete A100 `results/` tree (incl. DVC-pulled inputs) | `<mount>/results_a100/results_full/` |
| pi session transcripts (orchestrator + subagents) | `<mount>/results_a100/pi_sessions/` |
| subagent async run dirs (patches, logs) | `results/phase5/_session_artifacts/subagent_runs/` |
| machine notes (crontab, tmux launchers, package versions) | `<mount>/a100_backup/machine/` |

Not copied (re-creatable): model weights (`~/models`, pinned revisions in
`results/phase5/_session_artifacts/weights_manifest*.json`), uv venvs (uv.lock committed), llama.cpp build (pinned
commit + sha256 recorded), BiomedParse/SAT clones (commits in `probe3d_repo_commits.txt`, no local edits),
nnInteractive venv (`nninteractive==2.6.0`). Secrets (`.env` HF token, pi auth) intentionally NOT copied.

## To resume on a new box
1. `git clone <mount>/agentic-segmentation.git -b phase5-v3`; `data/` symlinks as in `HANDOFF_A100_p45.md` §0;
   copy `<mount>/results_a100/results_full/` → `results/` (or `dvc pull` + copy `results_a100/phase5`).
2. Install per `HANDOFF_A100_p45.md` §2 (uv, dvc, `uv sync`, `npm ci`, nni venv); recreate `.env` with an HF token
   whose account accepted HAI-DEF (MedGemma) and the BiomedParse gate; `setup_a100.sh --weights-only`;
   `harness/local_models/download_new_pointers.sh` (MolmoPoint, Qwen3-VL); probe3d README for SAT/BiomedParse.
3. 40 GB card: start services in subsets (E30). All model servers serialize inference (v1.0.1+, BF16 MedGemma 1.0.2).

## Open leads (if work resumes)
- Study C was designed but not run: oracles only as location hints (ranked marks / slice ranges; never masks;
  nnInteractive does all delineation), as a tool (`seg_locate`) or as injected telemetry; controls: no-agent
  "oracle top-1 → one click" pipeline + adherence analysis (`eval/v2/v3_adherence.py`).
- Agent localization failures are mostly recognition (E31 table); a search oracle matters more on BEAMSTER (many
  small metastases) — BiomedParse v2 has a T1-Gd enhancing-tumour head (Phase 4 idea, skipped).
- Final merge of `phase5-v3` into `main` per `HANDOFF_A100_p45.md` §3 once the A10 line is settled.
