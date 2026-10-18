# Experiment log: agentic 3D segmentation on MSTT-199 (development set)

This is the master record of what was run, what came out, and which design decision each result informed.
It is written for later reuse (presentation of the development process), so every number points to a file.

- **Generated tables** (per-case, per-pass, per-condition): `research/experiments/TABLES.md`
  (regenerate with `uv run --project harness/server python eval/v2/build_log_tables.py`).
- **Run registry** (one row per run): `research/experiments/runs.csv`.
- **Per-run text records** (scores, per-case meta, every segd operation, the agent's transcript without image
  payloads, Phase 2 raw answers): `research/experiments/runs/<run_id>/` (git).
- **Heavy artifacts** (rendered images the agent saw, mask versions, final masks, full pi sessions with images,
  Phase 2 input images, mark caches): `results/` tracked by **DVC** (`results.dvc`; `dvc pull` restores them).
- **Pass-by-pass Phase 1 notes** written while iterating: `research/experiments/phase1_design_log.md`.
- **Plan and gates**: `research/harness_design_v2.md` (§3 phases and thresholds fixed before data).

---

## 0. Glossary (codes used below)

| Code | Meaning |
|---|---|
| **design split** | 30 MSTT-199 cases (6 per histotype) used only for development. Never reported as results. Test split = the other 169 cases. |
| **F / L / M / N / V** | MSTT-199 histotype groups: Fibrous (desmoid, undifferentiated pleomorphic sarcoma), Lipomatous (well-differentiated / dedifferentiated liposarcoma), Myxoid (myxoma, myxofibrosarcoma), Nerve sheath (schwannoma, MPNST), Vascular (hemangioma/AVM, angiosarcoma). |
| **GT** | Ground-truth mask from the dataset. The agent never sees GT; only offline scoring reads it. |
| **one-click baseline** | No LLM. One positive nnInteractive click at the centroid of the GT's largest component (an *oracle* seed). Measures what nnInteractive does when the location is known. |
| **marks / set-of-marks** | Numbered labels drawn on rendered slices. The agent points by choosing a mark number instead of typing pixel coordinates. Global marks (integers) are fixed 3D points; edge marks (`E<n>`) lie on image edges of the current view. |
| **Phase 2 localization study** | Single model call per image (no agent loop): show the axial slice where the GT is largest, ask the model to point at the tumor. Tests pointing/knowledge separately from segmentation. |
| **R** | Phase 2 condition "raw": image with pixel rulers, model answers `(x, y)` coordinates. |
| **S** | Phase 2 condition "set-of-marks": same slice with numbered marks, model answers a mark number. Target named by histotype (e.g. "fibrous soft-tissue tumor"). |
| **S-gen** | Like S but the target is only "soft-tissue mass" (no histotype). S vs S-gen tests whether the model uses histotype knowledge. |
| **S-desc** | Like S plus a 1–3 sentence literature description of the histotype's MRI appearance (`harness/study/descriptions.json`). |
| **S-ex** | Like S plus one *example image*: a different design case of the same histotype and sequence, shown in three planes with its GT outline in green ("this is what such a tumor looks like"). Leave-one-out: never the case itself. |
| **hit rate** | Fraction of answers whose point lies inside GT on that slice. |
| **Phase 1 pass** | The full agent (pi + seg_* tools + nnInteractive) run on all 30 design cases without any seed. |
| **tags** | Scorer failure tags: `localization` (no positive prompt landed in GT and Dice < 0.5), `delineation` (landed, Dice < 0.5), `harness` (tool/infra error), `budget` (forced finish), `none`. |
| **Dice / HD95 / NSD** | Overlap (0–1); 95th-percentile Hausdorff distance (mm); normalized surface Dice at 2 mm tolerance. |
| **thinking level** | pi's reasoning-effort setting for the model (medium in all runs so far). |

---

## 1. Prior work: first run (May 2026, Sonnet 4.6) — archived

Material in `archive/first_run_2026-05/` (not versioned). Summary (details: `research/harness_design_v2.md` §1):
pilot without a seed gave median Dice 0.29; the reported "median 0.80" came from runs seeded with the GT centroid and a
prompt telling the agent not to localize. No one-click baseline existed, so the agent's contribution was unknown. The
harness was rebuilt from scratch for this phase.

---

## 2. Environment and provenance

| Item | Value |
|---|---|
| Machine | GPU dev server: NVIDIA A10 24 GB, 30 cores, 220 GB RAM, Ubuntu 24.04 (moved here 2026-10-18 from desktop + Colab L4) |
| nnInteractive | 2.6.0, model `nnInteractive_v1.0`, local server (`~/nni/start.sh`, `--max-sessions 4 --idle-timeout-seconds 3600`) |
| Agent | pi SDK 1.2.0 (pinned in `harness/runner/package.json`), auth shim `@gotgenes/pi-anthropic-auth` 3.4.3, model `anthropic/claude-sonnet-5-5`, thinking `medium`, concurrency 4 |
| Data | MSTT-199 isotropic 1 mm, design split `eval/v2/mstt_cases.csv` (seed 2026). Data checks: `eval/v2/data_check/design.jsonl` |
| Code version | Before git: `harness_sha256` in each run's `batch_summary.json`. From 2026-10-18: git commit + `git describe` in every run's meta (`harness-v1` tag = Phase 1 pass 5 code) |

---

## 3. Chronological record

### E1. One-click nnInteractive baseline (oracle seed) — runs `oneclick_design_colab`, `oneclick_design`
- Colab L4 (first) and A10 rerun: mean Dice **0.696 / 0.697**, median 0.739; per-case max |Δ| 0.016 (GPU nondeterminism).
- Per histotype (A10): F 0.559, L 0.720, M 0.831, N 0.651, V 0.723 (published U-Net: 0.507/0.744/0.823/0.748/0.584, mean 0.681).
- Failure cases: F32 over-segmentation (841 vs 220 mL), F15 (2 mL lesion, 0.197), F50 (76 components).
- **Decision:** most of the old "0.80" was nnInteractive with an oracle seed; the agent must be compared with ~0.70,
  not only with the published U-Net. Timings on the A10: set_image 0.17 s (Colab 8 s), point 0.3–0.8 s.

### E2. Harness build and reviews (no runs scored; 2026-10-18)
Components: segd daemon (`harness/server`), marks + rendering, pi extension, SDK runner, scorer, Phase 2 study code.
Independent reviews by a separate model (Opus 5.5), findings and decisions:
1. **segd core review** — 1 blocker (lasso/scribble near the body edge crashed on cropped uploads), 4 major
   (nnInteractive session silently expired after 10 min of viewing; failed calls not counted in budgets; FakeNNI/tests
   axial-only; leaked sessions on failed open). All fixed. **Budget rule decided:** every tool call counts (also views and
   failed calls); with the tool budget used up only `seg_finish` works.
2. **marks/render review** — no blockers; geometry verified independently (72/72 planted-voxel checks; drawn marks =
   resolved click positions). Majors: edge-mark labels hid their own glyph; edge marks were restricted to the current mask
   (so the agent could not outline missed tumor). Fixed; edge marks now cover the body with priority near the mask edge.
   Marks cache v7 validated by a parameter fingerprint; computation deterministic, ≤ 27 s per volume on CPU.
3. **Mark coverage** (offline diagnostic, `eval/v2/marks_dev/coverage.md`): at least one mark inside GT on the largest
   axial GT slice in 30/30 cases (full-slice coarse view, the Phase 2 S input), 3D coverage 30/30, small lesions
   F15/V55/V57 covered.

### E3. Integration runs — `integ-02` (3 cases)
- V55 0.845, F14 0 (wrong structure), M33 0.156 (172 vs 2034 mL). Found and fixed: runner/segd output-path mismatch,
  task message showed "unknown" volume size.
- **Observation that drove Phase 1:** the agent clicks once, looks at one axial slice, and finishes (4–6 tool calls).

### E4. Phase 2 localization study, design cases — run `phase2_dev__dev-01`
- Setup: largest-GT axial slice per case, full slice, frozen marks v7 / render; 3 repetitions × 30 cases × {R, S, S-gen};
  Sonnet 5.5, thinking medium, single turn, no tools. Prompts: `harness/study/prompts/`. Strict answer parsing (0 parse failures).
- **Results:** R **0.722**, S **0.889**, S-gen **0.900**; mark coverage 30/30.
  Per histotype S: F 0.667, L 0.833, M 1.000, N 1.000, V 0.944. R vs S: N 0.500 → 1.000, V 0.667 → 0.944.
- **Gate** (fixed before data: S ≥ 0.80 and S − R ≥ 0.10): **PASS** (S − R = +0.167). S-desc / S-ex were not run
  (the plan runs them only if S fails).
- **Interpretation:** R misses are near misses (3–6 mm; N34, N40, V37, V55) → coordinate/grounding error that marks remove.
  S misses are wrong structures (F14, F15, L33). S ≈ S-gen → naming the histotype does not change where the model points.
- **Decision:** use set-of-marks pointing; no histotype description in the agent prompt (at that point).
- Caveats: dev cases = the cases the marks were tuned on (coverage optimistic); n = 6 per histotype.

### E5. Phase 1 pass 1 — `phase1-01`
Changes before the pass: 3-plane feedback image after every edit (axial/coronal/sagittal through the mask, wide crop);
`seg_view crop:"marks"` with mark ids; sheets cropped to the body; system prompt with an explicit localization step.
- 30/30 finished, 0 harness errors. **Mean 0.639, median 0.750**; 5.0 tool calls and **1.03 model prompts** per case.
- Localized cases ≈ one-click Dice; 6 localization failures (F15, F32, L33, L43, N40, V46).
- **Finding:** the agent does not use the check/correct loop. Example (M33): "638 mL seems reasonable … I'll just go with
  this and wrap it up" with 12 components. **Decision:** add a structural verification step (two-stage finish).

### E6. Phase 1 pass 2 — `phase1-02`
Change: `seg_finish` first returns a 3×3 review grid (3 slices per plane); a second call confirms.
- **Mean 0.612, median 0.750**; 6.3 tool calls, 1.07 prompts. Agent confirms without editing.
- **Error analysis** (localized, Dice < 0.8): systematic under-segmentation, precision 0.97–1.0 and recall 0.3–0.6
  (M33, N34, L35, V65, N01, M08, F44); F32 is the only over-segmentation. GT-vs-prediction renders (M33, N34): masks stop
  early at the cranio-caudal ends; GT is blocky along z (annotated on thicker slices) and more inclusive.
- Localization varies between passes by 2–3 cases (F14/V57 failed here, correct in pass 1; V46 the reverse).
- **Decision:** make continuation past the mask visible in the review.

### E7. Phase 1 pass 3 — `phase1-03`
Change: review grid 3×5 adds slices 8 mm past each mask end (orange frames) + a checklist in the review text.
- **Mean 0.573, median 0.750**; 5.7 tool calls, 1.1 prompts.
- **Harness bug found:** the model emits two `seg_finish` calls in one turn (parallel tool calls), confirming the review
  before seeing it. **Decision:** confirmation must quote a 4-digit `review_code` printed in the review.

### E8. Phase 1 pass 4 — `phase1-04`
Change: `review_code`.
- **Mean 0.613, median 0.750**; 6.0 tool calls; **exactly 1 model prompt in all 30 cases**. The agent reads the review
  (quotes the code) and confirms; its reasoning says the outline follows the mass. Under-segmentation relative to GT
  remains (M33 739 vs 2034 mL) → interpreted as annotation convention + model judgement, not a harness bug.
- Cosmetic: slice chip overlapped the orientation letter in narrow review cells → fixed (no behavioural change).

### E9. Phase 1 pass 5 — `phase1-05` → **harness v1** (git tag `harness-v1`, commit `6fb08a5`)
- 30/30 finished, 0 harness/internal/runner errors, no forced finishes, mark coverage 30/30 → **freeze criteria hold**
  (`harness_design_v2.md` §3).
- **Mean 0.597, median 0.750**, HD95 median 19.8 mm; 6.0 tool calls, 1.03 prompts; ≈ $1.2 for 30 cases.
- Tags: none 21, localization 7 (F14, F15, L33, L43, N40, V46, V57), delineation 2 (F32, N01).
- **Localized cases (23): agent 0.779 vs one-click oracle 0.756 on the same cases** → the whole gap to the oracle
  (0.597 vs 0.697) is localization.
- Confidence: high 3 (all correct), medium 22 (4 localization failures), low 5 (3 failures).
- Open usability item: 7× first view sent `crop:"marks"` with empty ids (agent recovers next call).

#### Phase 1 summary across passes (medium thinking)

| pass | run | change tested | mean | median | localized mean | loc. fails | tool calls | prompts |
|---|---|---|---|---|---|---|---|---|
| 1 | phase1-01 | 3-plane feedback, localization prompt | 0.639 | 0.750 | 0.781 | 6 | 5.03 | 1.03 |
| 2 | phase1-02 | two-stage finish (3×3 review) | 0.612 | 0.750 | 0.765 | 6 | 6.30 | 1.07 |
| 3 | phase1-03 | 3×5 review with past-the-mask slices | 0.573 | 0.750 | 0.748 | 7 | 5.73 | 1.10 |
| 4 | phase1-04 | review_code | 0.613 | 0.750 | 0.766 | 6 | 6.03 | 1.00 |
| 5 | phase1-05 | chip fix (= harness v1) | 0.597 | 0.750 | 0.779 | 7 | 6.03 | 1.03 |
| – | oneclick_design | oracle one click (no LLM) | 0.697 | 0.739 | – | – | – | – |

Pass-to-pass differences in the mean are within localization noise (±1–2 cases × ~0.8 Dice / 30 ≈ ±0.03–0.05);
the harness changes did not change behaviour because the agent never iterated.

---

## 4. Decisions with Sameed after the validation phase (2026-10-18 evening)

1. Raise the agent's thinking level: test `high`, then `xhigh`, then `max` on the design cases.
2. Expand the Phase 2 study with reference examples (S-ex: another case's GT outline as "what this anatomy looks like")
   and, if it helps localization, give such examples to the agent.
3. Start version control (git; DVC for heavy artifacts); harness v1 = tag `harness-v1`.
4. Keep iterating in the development phase; record everything here.

## 5. Development phase continued (after harness v1)

### E10. Phase 2 `dev-02`: reference examples, descriptions, thinking level — runs `phase2_dev__dev-02-*`
Same images and scoring as E4 (`dev-01`); code `harness-v1-1-g51b7064`; 3 reps × 30 cases per condition.

| run | thinking | R | S | S-gen | S-desc | S-ex |
|---|---|---|---|---|---|---|
| dev-01 | medium | 0.722 | 0.889 | 0.900 | – | – |
| dev-02-medium | medium | – | – | – | 0.889 | 0.900 |
| dev-02-high | high | 0.700 | 0.911 | – | – | 0.900 |
| dev-02-xhigh | xhigh | 0.733 | 0.900 | – | – | 0.900 |
| dev-02-max | max | (running) | | | | |

- **No condition and no thinking level changes single-slice pointing**: every S-family condition sits at 0.89–0.91.
  The misses are the same three cases in every condition and repetition: **F14, F15, L33** (S, S-gen, S-desc, S-ex).
  R misses stay the near-miss set (N01, N34, N40, V37, V55; 3–6 mm).
- Offline look at the three (GT outline on the largest slice): F14 = subtle heterogeneous posterolateral pelvic lesion
  next to a large dark anterior mass that the model picks; F15 = 2 mL lesion beside a bright structure; L33 = T2
  fat-suppressed lipomatous tumor (dark, wrapped around the bone). Showing another patient's outlined tumor (S-ex) or a
  literature description (S-desc) does not move these.
- **Interpretation:** on a given correct slice, localization is at a ceiling of ~0.90 for this model; S-ex/S-desc add
  nothing on the dev set. The agent's localization failures therefore come mostly from *search* (which slice/region it
  looks at), which the single-slice study does not test.

### E11. Phase 1 at higher thinking levels (harness v1 unchanged) — runs `phase1-06-high`, `phase1-07-xhigh`, `phase1-08-max`
| run | thinking | mean | median | localized mean | loc. fails | tool calls | prompts | cost USD |
|---|---|---|---|---|---|---|---|---|
| phase1-05 | medium | 0.597 | 0.750 | 0.779 | 7 | 6.0 | 1.03 | 1.18 |
| phase1-06-high | high | 0.659 | 0.806 | 0.791 | 5 | 10.4 | 1.27 | 2.50 |
| phase1-07-xhigh | xhigh | **0.706** | **0.832** | 0.795 | 4 | 21.6 | 1.67 | 7.58 |
| phase1-08-max | max | (running) | | | | | | |

- Higher thinking makes the agent **search more** (more sheets/views) and localize more cases: L43 and V57 found at high
  and xhigh, V46 found at xhigh; L35 extended to the full tumor (0.53 → 0.90 at high). xhigh mean 0.706 is the first
  pass above the one-click oracle mean (0.697).
- Model prompts per case rise only to 1.3–1.7: the agent still rarely refines with nnInteractive; gains are localization.
- Persistent failures across all levels: F15 (2 mL), L33 (T2 lipomatous), N40; delineation: M33, N01, N34 (under-segmentation
  relative to the inclusive GT), F32 (over-segmentation).
- max thinking: single Phase 2 calls on hard cases take 6–8 min and 47–66 k output tokens; Phase 1 cases hit the 10-min
  wall-time limit (forced finishes) — results pending.

## 6. Next experiments (planned)

| ID | Question | Runs |
|---|---|---|
| E10 | Do reference examples / descriptions improve localization? Does thinking level change pointing? | Phase 2 `dev-02`: S-ex, S-desc (medium) + S, S-ex at high/xhigh/max |
| E11 | Does a higher thinking level make the agent iterate and localize better? | Phase 1 harness v1 at high, xhigh, max |
| E12 | Does an example image in the agent's task improve localization end to end? | Phase 1 with S-ex-style example (if E10 positive) |
