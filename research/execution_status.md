# Execution status (validation phase), updated 2026-10-18 late evening (GPU dev server)

Companion to `harness_design_v2.md`, `HANDOFF_2026-10-18.md`. Detailed pass log: `eval/v2/design_log.md`.
**State: validation phase complete; stopped before Phase 3 for discussion with Sameed.**

## Environment (dev server: A10 24 GB, 30 cores, 220 GB RAM)
- nnInteractive 2.6.0: `~/nni/.venv`, `~/nni/start.sh` (tmux `nni`, port 1527, `--max-sessions 4 --idle-timeout-seconds 3600`).
- segd: tmux `segd` (`cd harness/server && uv run segd`, port 8765). Restart after segd code changes.
- Auth shim 3.4.3; pi CLI 1.3.0; runner/study pin SDK 1.2.0. pi `externalEditor` set to vim.

## Results
- One-click oracle baseline (A10): mean 0.697, median 0.739.
- Phase 2 dev (`results/phase2_dev/dev-01/report.md`): R 0.722, S 0.889, S-gen 0.900 → gate PASS; S ≈ S-gen.
- Phase 1: 5 passes (`results/phase1-0{1..5}`); pass 5 (freeze candidate, harness sha `3c9154…`): mean 0.597, median 0.750,
  freeze criteria hold. Agent uses ~1 model prompt per case and never iterates; 6–7/30 localization failures.

## Open items for discussion
See the final report in the chat / design_log pass 5 entry.
