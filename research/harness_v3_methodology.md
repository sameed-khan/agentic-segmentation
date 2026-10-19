# Harness v3 and specialist models: methodology (draft for Sameed, 2026-10-19)

Sources: `research/literature/pointer_models_gui_actor_molmo2.md`, `research/literature/medgemma_1_5.md`,
`research/literature/local_visuospatial_vlms.md` (primary-source research, claims cited there). Nothing below has been
run yet. Facts about our own system come from E1–E19 (`research/experiment_log.md`).

## 1. Why: the questions this answers

Our harness already shows that a general frontier reasoner, driving an interactive segmenter through tools, beats a
trained task model (U-Net) on MSTT-199 (E19: Sonnet 5.5 xhigh 0.737, Opus 5.5 medium 0.746 vs 0.681). The remaining
error is concentrated in two **decisions**, not in delineation mechanics (E13, E15–E19):
1. **Localization**: picking the wrong structure (11–16 % of test cases; F and V worst). Localized cases average 0.81.
2. **Judgement**: deciding the mask is complete / which candidate tissue belongs to the lesion (sensitivity 0.06–0.13
   for GT-positive candidates at specificity 0.99, E16; oracle acceptance would add +0.03–0.04 Dice).

This maps onto open questions that matter to radiology segmentation research:

| RQ | Question | Radiology parallel |
|---|---|---|
| RQ1 | Does a small model **trained for the task skill** (visual pointing/grounding) localize lesions better than a large **general reasoner** given identical inputs? | Task-specific vs foundation models; "is scale or specialization the lever?" |
| RQ2 | Does adding **medical-domain intelligence** to the harness (a medical VLM consulted at decision points) improve the decisions where we lose Dice? | Domain-specific vs general AI; AI as a second reader |
| RQ3 | Is the gain a **component** effect (the model is better at the decision) or a **system** effect (the agent uses the advice well)? | Standalone AI accuracy vs human-AI / AI-AI team performance (automation bias, under-reliance) |
| RQ4 | What do **small local models** (4-bit on one 24 GB GPU) cost in accuracy, latency and money? | On-premise deployment where data cannot leave the hospital |

Two framings we must not over-claim: (a) these are **off-the-shelf** models applied out of domain (none was trained on
soft-tissue tumour MRI; MedGemma 1.5's MRI data are internal head/abdomen/knee), so results speak to transfer, not to
what a fine-tuned specialist could do; (b) one dataset (MSTT-199), so generalization needs Phase 4's second dataset.

## 2. Model choice: a factorial over "task skill" × "medical knowledge"

| | no medical training | medical training |
|---|---|---|
| **trained for pointing/grounding** | **GUI-Actor-2B** (GUI screenshots; coordinate-free attention head over patches) · **Molmo2-8B** (natural images; emits `<points>` 0–1000 coords) | (optional fill) BiomedParse v2: text-prompted biomedical segmentation, 3D MRI support |
| **general reasoning, no pointing head** | frontier: Sonnet 5.5, Opus 5.5 (+ sweep models) | **MedGemma 1.5 4B** (Gemma 3 + MedSigLIP; boxes only shown for chest X-ray) |

Why these and not others:
- **GUI-Actor-2B**: the purest "pointing specialist": its output *is* a spatial attention map, not text, so it
  isolates spatial grounding from language reasoning. Small (2B), MIT code, published ScreenSpot 86.5 / ScreenSpot-Pro
  36.7 (41.8 with verifier). Domain gap (UI widgets vs greyscale anatomy) is the hypothesis under test, not a flaw.
- **Molmo2-8B**: pointing specialist trained on natural images (PixMo-style point annotations), multi-point and
  counting; a second, independent pointing lineage so a result is not one model's quirk. Apache-2.0 (training data
  research-only terms).
- **MedGemma 1.5 4B**: the strongest openly available *medical* VLM that fits locally; covers CT/MRI as ordered 2D
  slices. Tests "domain knowledge" separately from "pointing skill". HAI-DEF terms (research use is fine).
- **Frontier generalists**: our existing baselines; same inputs, already measured (pointing study and agent runs).
- Not chosen as primary: UI-TARS-1.5-7B (larger, weaker public checkpoint), M3D-LaMed (needs 32×256×256 volume input,
  not comparable inputs), Qwen3-VL-8B (generalist; could be a "small generalist" control — see §7).

The factorial lets us state results as contrasts: pointing-trained vs not (GUI-Actor/Molmo2 vs MedGemma/frontier),
medical vs not (MedGemma vs similar-size generalist), small-local vs frontier.

## 3. Experiments

All tuning on the 30 design cases only; each configuration frozen (git tag) and preregistered in the experiment log
before any test-split run. Test split: 169 cases. Paired per-case statistics throughout (McNemar for hit rates,
Wilcoxon for Dice, case-bootstrap 95 % CIs); per-histotype results as secondary (n = 33–34 each).

### Study A: standalone localization (RQ1, RQ4) — component level, cheap
Identical inputs to the existing Phase 2 study (`results/phase2_test/prepare`): the axial slice through the GT's
largest cross-section, same PNG renders; target text = histotype.
- Conditions: **raw** (clean image; model outputs a point) and **marks** (set-of-marks image; point snapped to the
  nearest drawn mark, and, for text models, "answer a mark id").
- Outputs per model: GUI-Actor top-k region points with scores (k = 1, 3); Molmo2 first point (and all points);
  MedGemma box `[y0,x0,y1,x1]` in [0,1] → centre, plus "answer a mark id".
- Metrics: hit@1 (point inside GT), hit@3 (any of top 3), distance to GT (mm), parse/abstain rate, latency.
- Baselines (same cases): Sonnet 5.5 raw 0.704, marks 0.826 (E18); add Opus 5.5 medium for the marks condition
  (~1,500 calls, cheap) so the best agent model is a baseline too.
- Decision rule (preregistered): a specialist "wins" RQ1 if hit@1 ≥ the frontier marks baseline with McNemar p < 0.05;
  hit@3 ≥ 0.90 makes it useful as a *proposal* tool even if hit@1 is lower.
- Limitation stated up front: the slice is chosen with GT (in-plane pointing only); slice search is tested in Study C.

### Study B: decision sets for judgement (RQ2, RQ3 component level)
Fixed, GT-labelled decision states replayed as single calls (the E16 method, which reproduced in-agent behaviour):
1. **Candidate judgement** — existing E16 set (27 reviews, 100 candidates, 18 GT-positive; design). Baseline Sonnet
   sensitivity 0.056/0.130 at specificity 0.988. Metric: sensitivity, specificity, Dice gain vs oracle (+0.037).
2. **"Are we done?"** — from design-run review grids (phase1-06…10): label = the state is *not* done if another
   reachable version or the oracle-accept improves Dice by ≥ 0.05 (E13/E14 machinery). Metric: AUROC of the
   model's done/not-done probability, accuracy at the frozen threshold.
3. **Localization check** — states right after the first prompt: label = mask on the wrong structure (Dice < 0.1).
   Metric: sensitivity for wrong-structure states at ≥ 0.95 specificity (a false alarm costs budget, a miss costs the case).
For each set: MedGemma (Q4 and BF16), frontier models (Sonnet high, Opus medium), and a combination rule
(e.g. accept a candidate if MedGemma says lesion with p ≥ τ). Test-split decision sets are built from Phase 3 runs
only *after* v3 is frozen, for evaluation.

### Study C: harness v3, system level (RQ2, RQ3)
New, optional components, each switchable by flag so every combination is an ablation of the same frozen code:
- `seg_locate` (pointer tool): given the current view and the target text, returns ranked marks proposed by the
  specialist (GUI-Actor or Molmo2, whichever wins Study A) with scores. Agent decides whether to use them.
- **Medical second reader** (MedGemma) at three fixed checkpoints, injected into tool results (not a tool the agent
  may skip, so exposure is controlled): (1) after `seg_open`/first sheets: "where is the lesion?" (box → nearest
  marks); (2) at the first finish review: "is the outline complete; where is it wrong?"; (3) per candidate: lesion /
  normal / uncertain with confidence.
- Arms on the design set (Sonnet 5.5 high, the cheapest strong agent; repeat the winner with Opus 5.5 medium):
  v2 (control) · v3-point · v3-med · v3-both. Primary metric mean Dice; mechanism metrics: localization failures,
  agreement/override rates (how often the agent follows advice, and whether following it was right — RQ3),
  candidate accept sensitivity, tool calls, cost, latency.
- One frozen v3 configuration goes to the test split (169), paired against the same agent's v2 run from the sweep.

## 4. Deployment on this server (one NVIDIA A10, 23 GB; CUDA 12.8; 222 GB RAM)

Measured: nnInteractive with 8 sessions shows ~0.9 GB resident when idle (peak during inference to be measured;
plan for ≤ 5 GB). Each model runs as its own local HTTP service in its own `uv` environment (incompatible
`transformers` pins), started in tmux like segd/nni, loopback only, pinned versions recorded in run metadata.

| model | runtime | precision | est. VRAM | notes |
|---|---|---|---|---|
| MedGemma 1.5 4B | llama.cpp `llama-server` (OpenAI-compatible, images via base64, `--jinja`) | **Q4_K_M** (`unsloth/medgemma-1.5-4b-it-GGUF`, 2.49 GB) + `mmproj-F16.gguf` (0.85 GB) | ~4–5 GB at 8k context | Community conversion, not Google's. Reference: Transformers BF16 (~9 GB) on the design set to measure the Q4 effect. |
| GUI-Actor-2B | Transformers 4.51.3 + official repo code (custom pointer head) | **BF16** (~5 GB) | ~6–7 GB | Q4_K_M is not possible: llama.cpp runs plain Qwen2-VL but not GUI-Actor's attention head; at 2B, quantization is unnecessary. |
| Molmo2-8B | Transformers 4.57.1 (`trust_remote_code`) or vLLM ≥ 0.15 | BF16 (~17.3 GB) for Study A run when nnInteractive is idle; bnb 8-bit (~9 GB) for in-harness use | 9–19 GB | No llama.cpp/GGUF support, so no Q4_K_M. Load with explicit `torch_dtype=bfloat16` (checkpoint is FP32; `dtype="auto"` would need ~35 GB). 8-bit vs BF16 measured on the design set. |

Services (proposed): `llama-server` :8091 (MedGemma), `pointd` :8092 (FastAPI wrapping GUI-Actor or Molmo2:
`POST /point {image_png_b64, query, k}` → `{points:[{x,y,score}], raw_output, latency_ms}`). segd calls them with the
exact PNGs the agent sees; every request/response is logged (image hash, prompt, raw output) into the trajectory.
Decoding deterministic (temperature 0) so local results are reproducible; frontier baselines use their usual
settings and repeated calls. Sequencing: Study A runs need no nnInteractive, so they can run between sweep runs or
alongside it if VRAM allows; Study C runs one local model at a time next to nnInteractive.

Setup order: (1) llama.cpp CUDA build (pinned commit) + MedGemma GGUF — smoke test: one image, multiple images,
JSON output, VRAM, latency; (2) GUI-Actor venv + official example on a UI screenshot (verifies the pointer head),
then one MRI slice; (3) Molmo2 BF16 smoke test, then 8-bit.

## 5. Prompt engineering protocol (design split only)
- Per model × role: at most 8 prompt variants, selected by the role's primary metric on design data, then frozen.
  The same budget applies to the frontier comparison prompts where they are new (Study B "done?" and localization
  check), so no arm gets more tuning than another.
- Inputs fixed per role: renders identical to the agent's; MedGemma images at its native 896 px (the 1000 px panels
  are resized by the processor; we test a crop-to-panel variant within the 8).
- Output contracts: strict JSON with an explicit `uncertain` option; parse failures counted as errors, not retried.
- Hold-out discipline: with 30 design cases, use 5-fold case-level cross-validation for threshold/prompt choice and
  report the CV estimate; nothing from the test split is viewed until the v3 tag and preregistration are committed.

## 6. Threats to validity and how the design handles them
- **Domain gap** (GUI/natural images → MRI): the object of RQ1; reported as transfer results. Raw and marks conditions
  separate "can it see the lesion" from "can it use the marks".
- **Quantization confound**: BF16 vs Q4/8-bit measured on the design set for MedGemma and Molmo2, reported.
- **Tuning asymmetry**: equal prompt budgets, design-only tuning, preregistration.
- **Leakage**: MedGemma's MRI data are internal (Google); MSTT-199 is public since 2024 and could be in web-scale
  pretraining of any model — unknowable; Phase 4 uses a dataset released after 2026-06-30.
- **Advice exposure / automation bias**: fixed checkpoints (not optional tools) for the second reader; log
  follow/override and correctness (RQ3).
- **Run-to-run variance** of agent runs (±0.03–0.05 on 30 cases): design-set decisions use component sets first;
  system comparisons paired on the same cases; test run once per arm, plus a repeat of the control if budget allows.
- **Multiple comparisons**: one preregistered primary contrast per RQ; everything else labelled exploratory.
- **Licences**: HAI-DEF (MedGemma) and Molmo2 training-data terms permit research use; no clinical use claims.

## 7. Options and open decisions for Sameed
1. Fill the "medical + pointing" cell with **BiomedParse v2** (text-prompted biomedical segmentation incl. 3D MRI;
   research-only licence) so the factorial is complete. Recommended as Study A extension only.
2. Add **Qwen3-VL-8B** (Q4_K_M via llama.cpp; supported) as a "small general VLM" control: separates "small and
   local" from "specialized". Recommended for Study A (cheap).
3. Fine-tuning (e.g. LoRA of MedGemma or GUI-Actor on design-case slices) is out of scope here: 30 cases are too few;
   it would be a Phase 4 study with a training split.
4. Agent model for Study C: Sonnet 5.5 high (cheap, weakest localization → most room) is proposed; Opus 5.5 medium
   for the confirmatory run.

## 8. Order of work
1. Deploy + smoke-test MedGemma (Q4_K_M), GUI-Actor (BF16), Molmo2 (BF16/8-bit) — while the sweep runs.
2. Study A on design (prompt tuning), freeze, then test (169) — local, no API cost.
3. Study B on design sets; freeze roles.
4. Implement v3 components in segd (flags), tests + code review (gpt-6.1-sol), design-set arms, freeze `harness-v3`.
5. Test-split v3 run(s), paired against v2; write up.
