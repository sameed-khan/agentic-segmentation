# Experiment log, Phases 4–5 (A100 line, branch `phase5-v3`)

Continuation of `research/experiment_log.md` (E1–E19 + §6 preregistration of Phase 3; E20–E29 are written by the A10
orchestrator on `main`). Entries here are E30+. This file is appended to the main log at the final merge.
Same conventions: every number points to a file under `results/` (DVC / mirrored to `<mount>/results_a100/`) or
`research/experiments/runs/`.

---

## E30. A100 setup, 40 GB hardware deviation, revised VRAM plan, order of work (2026-10-20)

**Machine.** 1× NVIDIA A100-SXM4-40GB (driver 580.105.08 / CUDA 13.0, nvcc 12.8), 30 CPUs, 216 GB RAM, NVMe.
The Phase 5 plan (`research/phase5_implementation.md` §3, `setup_a100.sh`) assumed an **A100 80 GB** where all
services coexist. This is a **hardware deviation**, logged before any Study A/C run.

**Software (installed 2026-10-20 04:28–04:40 UTC).** uv 0.13.3; DVC (uv tool) + `dvc pull` from the mount remote
(14,789 files); `harness/server` uv sync; runner/study/extension `npm ci`; nnInteractive 2.6.0 server in `~/nni/.venv`
(torch 2.14.1+cu130, 16 sessions, port 1527); segd :8765. Local models via `harness/local_models/setup_a100.sh
--weights-only` with the pinned HF revisions (`~/models/weights_manifest.json`), llama.cpp built at the pinned commit
`1b29621b` (sm_80), GUI-Actor repo at `d98d1bb`. Fixes needed on the box (commit `bafd5f1`): GUI-Actor venv lacked
`torchvision` (Qwen2-VL processor imports it; pinned 0.22.1 to match torch 2.7.1), and `uv sync` tried to build the
optional `flash-attn` sdist (static metadata added; the `[flash]` extra stays opt-in, SDPA is the default).
Second fix (commit after `bafd5f1`): the three Python model servers had no inference lock; GUI-Actor failed one request
under 2 concurrent calls (`RuntimeError: Already borrowed`, HF tokenizers). All servers now serialize inference;
`latency_ms` is measured inside the lock (pure inference time). Runs made before this fix are discarded and repeated.

**Tests at `bafd5f1`.** segd 187 passed / 2 skipped; local_models + study + adherence + multi-scorer 152 passed;
`eval/v2/test_score.py` 20 passed (separate invocation); runner `tsc` OK + node tests 2/2.

**Smoke tests (`harness/local_models/smoke_test.py`, design case F14).** GUI-Actor: MRI /point OK (1.3 s), UI-screenshot
sanity point INSIDE the target button. MedGemma Q4 and BF16: strict JSON OK (0.6 s / 2.1 s). Molmo2: the smoke query
("the fibrous soft-tissue tumor", no verb) made Molmo2 describe instead of point (0 points); with the study prompt
("Point to the …") it emits the official `<points coords="1 1 277 598">` grammar and `molmo_points.py` parses it
correctly (parser validated on real output, as the build notes required).

**Measured VRAM (nvidia-smi, per process, idle after load).**

| service | precision | VRAM |
|---|---|---|
| nnInteractive (16 sessions, idle) | — | 1.0 GB (peak under load: to measure in Study C / Phase 4) |
| MedGemma Q4_K_M (llama-server, ctx 8192) :8091 | Q4_K_M + mmproj F16 | 4.3 GB |
| MedGemma BF16 (Transformers) :8094 | BF16 | 8.9 GB |
| GUI-Actor-2B + Verifier-2B :8092 | BF16 | 10.0 GB |
| Molmo2-8B :8093 | BF16 | 20.6 GB |

All five together ≈ 44.8 GB > 40 GB. **Revised plan (decision):**
- Study A: services run in subsets that fit (Molmo2 + both MedGemma arms + nni ≈ 35 GB; GUI-Actor + both MedGemma +
  nni ≈ 24 GB); never GUI-Actor and Molmo2 together. Inference is serialized per service, so co-residence affects
  only VRAM, not results.
- Study C: one pointer + one MedGemma arm next to nni. Worst case Molmo2 BF16 + MedGemma BF16 + nni (≤ 5 GB peak)
  ≈ 35 GB → **fits in BF16; no 8-bit Molmo2 arm is needed** (the arm stays as planned). If nni peak + activations
  exceed the card, the fallback is MedGemma Q4 instead of BF16 in that arm, declared before the run.

**Order of work (Sameed, 2026-10-20, overrides handoff §4):** finish **Phase 5 completely first** (Study A → Study C
→ Study B), **then Phase 4**. Subagent policy (Sameed): code writing GLM-5.3-flash (max effort), script execution /
research DeepSeek, code review by a fresh-context Opus 5.5 (medium) agent instead of gpt-6.1-sol.

**Results backup.** `~/mirror_a100.sh` (cron every 30 min) rsyncs `results/phase5`, `results/phase4`,
`results/phase2_*` and `results/p5-*`/`p4-*` runs to `<mount>/results_a100/` (`.last_mirror`).
