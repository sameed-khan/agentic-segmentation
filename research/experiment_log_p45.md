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

---

## E31. Phase 5 component studies (design split): no open model passes the Study C gate → Phase 5 paused (2026-10-20)

All numbers on the 30 MSTT design cases; nothing was run on the test split. Code: commits `440196a`…`9afcc3b`.
Reviews: two fresh-context Opus 5.5 (medium) reviews of the Study A code (first: OK with notes, M1–M4 fixed in
`483364d`; second: blocker B1 llama-server /healthz → fixed in `a594309`).

**Harness flaws found and fixed during design (all before any reported number).**
1. *Two-panel marks image:* pointer models point at the tumour on the clean (left) panel 100 % of the time (Molmo2
   fold rate 1.0), but marks exist only on the right panel → snapping failed. Declared rule: clean-panel points are
   translated onto the marked panel before snapping (`--clean-panel-fold`, default on; fold-off sensitivity reported in
   the same pass). Molmo2 marks hit@1 0.10 → 0.57.
2. *MedGemma box scale:* MedGemma answers boxes on 0–1000 (Gemma convention); scorer accepted only 0–1 → 100 % parse
   failures. Fixed (strict rule: per-1000 only integer-valued; nested `[[box]]` unwrapped).
3. *MedGemma 1.5 thinking mode:* on longer prompts it answers `<unused94>thought … <unused95>` + answer; with the
   original 256-token cap nearly all Study B replies and all Q4 box replies were truncated. Fix: max_tokens 2048, parse
   only the answer after the thought, truncated thoughts = parse failures. All earlier MedGemma runs invalidated
   (`results/phase5/_invalid/`).
4. *Model servers not thread-safe* (GUI-Actor tokenizer race) → inference lock, SERVICE_VERSION 1.0.1, runner refuses
   to resume across service versions; pre-lock Molmo2 v0 run discarded.

**Study A — standalone localization (GT's largest axial slice given; hit@1 = point inside GT; best of ≤ 6 prompt
variants per model × condition, so design numbers are optimistic).**

| model | marks hit@1 | raw hit@1 | notes |
|---|---|---|---|
| Sonnet 5.5 high (frontier, dev-02-high) | **0.911** | 0.700 | baseline |
| Opus 5.5 medium (frontier, `phase2_dev/dev-opus55-medium`) | 0.889 | – | test split S: 0.840 (`phase2_test/p2-test-opus55-medium`; Sonnet high 0.826) |
| Molmo2-8B | 0.667 | 0.600 | McNemar vs Sonnet p = 0.016; **0 cases hit that Sonnet misses** (7 the reverse) |
| Qwen3-VL-8B-Instruct | 0.533 | 0.667 | abstains (prose) on 5–6/30 |
| MolmoPoint-8B | 0.533 | 0.533 | |
| GUI-Actor-2B | 0.333 | 0.233 | |
| MedGemma 1.5 4B BF16 | 0.467 (mark id) | 0.200 (box) | |
| MedGemma 1.5 4B Q4_K_M | 0.300 (mark id) | 0.333 (box) | |

Local-model misses are wrong structures (mostly > 25 mm), on the same hard cases the frontier misses (F14, F15, L33).
Snapping is lossless (raw-point hit = snapped hit). hit@3 ≈ hit@1 for all pointers (one location emitted).

**3D probes (whole volume + text; mask → deepest point; `results/phase5/probe3d/score_all.log`).** SAT-Nano "tumor":
3D hit 0.50, k*-slice hit 0.47, Dice 0.16, empty 7 %; other SAT prompts 0.30–0.33. BiomedParse v2 (gated weights,
accepted by Sameed): empty prediction on 77–100 % of cases for all 12 prompts (no soft-tissue-tumour class); best
3D hit 0.13.

**Study B — decision sets (single replays; MedGemma v1 = after the thinking fix).**

| set | Sonnet high AUROC | Opus medium AUROC | MedGemma Q4 AUROC | MedGemma BF16 AUROC |
|---|---|---|---|---|
| candidate_judgement (100; 18 lesion) | 0.756 (sens 0.11, spec 0.99) | 0.832 | 0.378 (says "lesion" to ~all) | 0.503 (sens 0.83, spec 0.10) |
| done_check (132; 12 not done) | 0.556 | 0.507 | 0.535 | 0.403 |
| wrong_structure (140; 21 wrong) | 0.866 | 0.887 | 0.484 | 0.632 |

Frontier candidate sensitivity reproduces E16 (0.11). "Are we done?" is at chance for every model.

**Decomposition of the agent's localization failures (test, existing Phase 2/3 runs, no new test runs).** Of the
agent's localization failures, most are *recognition* failures (the same model also misses on the correct given slice
in Phase 2), few are *search* failures: Sonnet high 20 recognition / 7 search (of 27), Sonnet xhigh 15/3, Opus xhigh
13/2, GPT-6.1-sol high 10/3. Lipomatous cases on fat-suppressed T2 (7 test cases) are not a drag (Sonnet high 0.855
vs 0.783 on T1 L cases); only design case L33 is a persistent miss.

**Gate (agreed with Sameed before Study C):** an oracle enters the harness only if, on design, it localizes ≥ 3 of the
agent's 7 design localization failures (F14, F15, L33, L43, N40, V46, V57) without being wrong where the agent is
right; MedGemma as second reader only if its Study B judgement beats or complements the frontier. Result: **no
candidate passes** (best: SAT "tumor" finds 1/7 — L33 — and misses half of all tumours; 2D pointers 0 complementary
cases; MedGemma at chance in Study B). Oracles would never segment (location hints only, nnInteractive does all
delineation) — design recorded for a future attempt.

**Decision (Sameed, 2026-10-20): Study A is not run on the test split; Study C is not run; Phase 4 is skipped; the
project pauses here.** Interpretation for the talk: the agent's remaining localization errors are recognition errors
on atypical anatomy that are hard for *every* available model — small specialists (pointing-trained or
medical-trained, 2D or 3D) fail on the same cases and below the generalist; the judgement gap (accepting true tumour
rim) is not closed by a 4B medical VLM either. Neither gap is a pointing-precision or tooling problem.

Run files of the MolmoPoint/Qwen3-VL design runs and the 3D-probe prediction masks were lost when the workers'
temporary worktrees were removed; scores are preserved (worker reports, `results/phase5/probe3d/score_all.log`,
`results/phase5/_session_artifacts/logs/probe_*`). Pause handoff: `research/HANDOFF_A100_p45_pause.md`.
