# Phase 1 design log (MSTT design split, 30 cases)

Each pass: date, harness sha256 (runner `meta.json` → `harness_sha256`), per-case Dice, first positive click inside
lesion, failure tag (`harness` | `localization` | `grounding` | `delineation` | `budget` | `none`), notes.
Every prompt/harness change between passes is listed under "Changes" with the reason.

Reference: one-click nnInteractive at the GT centroid (oracle seed, no LLM), `results/oneclick_design/`:
mean Dice 0.697 (A10 rerun 2026-10-18; Colab run 0.696, max per-case |Δ| 0.016).

## Environment changes
- 2026-10-18: moved from desktop + Colab L4 to the GPU dev server (A10 24 GB, 30 cores, 220 GB RAM).
  nnInteractive 2.6.0 server local (`~/nni/.venv`, tmux session `nni`, `--max-sessions 4 --torch-n-threads 8`).
  set_image 0.17 s mean (Colab: 8 s), point prompt ~0.3–0.8 s. Auth shim on the server is 3.4.3 (desktop: 3.4.2).

## Pre-pass changes (before Phase 1 pass 1)
- 2026-10-18 review of segd core (Opus): budget semantics decided: every seg_* call counts (incl. failures, views);
  tool budget exhausted → only `seg_finish`; prompt budget exhausted → model prompts error, other ops work.
  system.md updated accordingly. Extension: `seg_paint.slice_range` documented as inclusive; `seg_lasso` ids documented
  as ordered around the outline. nnInteractive server restarted with `--idle-timeout-seconds 3600` (default 600 s
  would reap a session while the agent only views).

## Phase 2 (dev localization study), run `dev-01`, 2026-10-18
Frozen marks v7 / render after the marks-render review fix round. 30 design cases × 3 reps, Sonnet 5.5 medium, single turn,
largest GT axial slice, full slice. Report: `results/phase2_dev/dev-01/report.md`.
- R 0.722, S 0.889, S-gen 0.900; coverage 30/30. Gate (S ≥ 0.80, S − R ≥ 0.10): **PASS** (+0.167). S ≈ S-gen.
- S misses are semantic (F14, F15, L33: wrong structure, d 7–90 mm); R misses are mostly near misses (d 3–6 mm: N34, N40, V37, V55) = grounding error.
- Decision: proceed with set-of-marks; S-desc/S-ex not run (gate passed); no histotype description added to the agent prompt.
- R prompt clarified before the run: rulers now sit outside the image, coordinates are ruler (image-area) px.

## Changes before Phase 1 pass 1
- Feedback image after every edit: three planes (axial/coronal/sagittal) through the mask centre, crop = mask bbox + max(30 mm, 50 %) (was: one tightly cropped slice). Motivation: integ-02 M33 (172 mL predicted vs 2034 mL GT) looked perfect on the single axial slice.
- `seg_view` `crop: "marks"` takes `ids` (zoom on named marks + 25 mm, reveals fine marks). Edge marks over the whole body (were restricted to the mask). Sheets cropped to the body.
- system.md: localization step (scan ≥ 2 planes, normal-anatomy confounders, compare limbs), check the three-plane image especially at the lesion ends, "use the budget to check and correct".

## Pass 1 — `phase1-01`, 2026-10-18, harness_sha256 `5cc4409b96847386fb828d684e642c797660c4c93c67d8b5aadda38f8788abd1`
30/30 finished, 0 harness errors (6 agent-input errors: 5× `crop:"marks"` with empty ids, 1× mark not in view), no forced finishes.
**Mean Dice 0.639, median 0.750** (one-click oracle: 0.697 / 0.739). Mean 5.0 tool calls, 1.03 model prompts; confidence 23 medium / 4 low / 3 high.

| case | Dice | one-click | first click inside | tag | notes |
|---|---|---|---|---|---|
| F14 | 0.645 | 0.752 | yes | none | |
| F15 | 0 | 0.197 | no (10 mm) | localization | picked dark region near condyle; confidence low |
| F32 | 0.423 | 0.407 | no (3 mm) | localization | |
| F44 | 0.725 | 0.682 | yes | none | |
| F49 | 0.905 | 0.866 | yes | none | |
| F50 | 0.889 | 0.448 | yes | none | |
| L19 | 0.835 | 0.825 | yes | none | |
| L20 | 0.940 | 0.941 | yes | none | |
| L33 | 0.000 | 0.597 | no (38 mm) | localization | T2 lipomatous, wrong structure |
| L35 | 0.923 | 0.633 | yes | none | |
| L38 | 0.967 | 0.965 | yes | none | |
| L43 | 0 | 0.361 | no (78 mm) | localization | pelvic fat region |
| M01 | 0.907 | 0.899 | yes | none | |
| M07 | 0.947 | 0.946 | yes | none | |
| M08 | 0.736 | 0.727 | yes | none | |
| M09 | 0.954 | 0.950 | yes | none | |
| M21 | 0.834 | 0.815 | yes | none | |
| M33 | 0.473 | 0.650 | yes | delineation | 12 components, agent: "I'll just go with this and wrap it up" |
| N01 | 0.481 | 0.484 | yes | delineation | |
| N18 | 0.888 | 0.883 | yes | none | |
| N21 | 0.872 | 0.837 | yes | none | |
| N26 | 0.923 | 0.932 | yes | none | |
| N34 | 0.442 | 0.492 | yes | delineation | |
| N40 | 0 | 0.280 | no (61 mm) | localization | |
| V37 | 0.765 | 0.770 | yes | none | |
| V46 | 0 | 0.608 | no (68 mm) | localization | agent suspected bladder, finished anyway |
| V54 | 0.707 | 0.785 | yes | none | |
| V55 | 0.843 | 0.844 | yes | none | |
| V57 | 0.715 | 0.717 | yes | none | |
| V65 | 0.434 | 0.613 | yes | delineation | |

Diagnosis: the agent does not use the check/correct loop at all (1 prompt, finish). When localized, Dice ≈ one-click.
Freeze criteria (harness): 30/30 finished, coverage ≥ 0.90, no open `harness` bugs → technically met, but the loop is not
exercised, so the correction tools are untested in practice. Not frozen.

## Changes before pass 2
- Two-stage `seg_finish`: the first call returns a final review (3×3 grid: 3 slices per plane through the mask extent);
  a second call confirms. Editing after the review requires a new review. Runner forced finish bypasses it.
- Components summary shows the remainder ("largest 99.6%; 11 others total 2.3 mL").

## Pass 2 — `phase1-02`, 2026-10-18 (two-stage finish with 3×3 review)
30/30 finished, 0 harness/internal errors, no forced finishes. **Mean 0.612, median 0.750.** Mean 6.3 tool calls, 1.07 prompts.
The agent confirms the review without editing. Localization failures: F14, F15, L33, L43, N40, V57 (F14/V57 were correct in
pass 1, V46 failed in pass 1 and is correct now → run-to-run variance in localization of ~2–3 cases).
Error analysis of localized cases with Dice < 0.8: **systematic under-segmentation** (precision 0.97–1.0, recall 0.3–0.6:
M33, N34, L35, V65, N01, M08, F44); F32 is the only over-segmentation (817 vs 220 mL). Offline GT-vs-prediction renders
(M33, N34): the mask stops early at the cranio-caudal ends; GT is also blocky along z (annotated on thicker slices).

## Changes before pass 3
- Review grid 3×5: adds slices 8 mm past each end of the mask per axis (orange frame) + a checklist in the review text.

## Pass 3 — `phase1-03`, 2026-10-18 (3×5 review with past-the-mask slices)
30/30 finished, 0 harness errors. **Mean 0.573, median 0.750.** 5.7 tool calls, 1.1 prompts.
**Harness bug found (review bypass):** the agent emits two `seg_finish` calls in one assistant turn (parallel tool calls),
so the second confirms the review before the agent has seen it (M33, L35 transcripts). Localization failures: F14, F15, L33,
L43, N34, N40, V57 (variance again: N34 localized in passes 1–2).

## Changes before pass 4
- Review confirmation requires the 4-digit `review_code` printed in the review (deterministic hash of case/version/call count);
  a confirmation without it returns "Not saved" (counts as a call). `seg_finish` schema gets optional `review_code`.

## Pass 4 — `phase1-04`, 2026-10-18 (review_code)
30/30 finished, 0 harness/internal errors, no forced finishes. **Mean 0.613, median 0.750.** 6.0 tool calls,
**exactly 1 model prompt in all 30 cases**. The agent now reads the review (quotes the code) and still confirms; in its
reasoning it judges the outline correct. Localization failures: F14, F15, L33, L43, N40, V57 (6/30).
Delineation errors remain under-segmentation relative to GT (M33 739 vs 2034 mL; on the review the outline follows the
bright mass, GT is more inclusive) → largely an annotation-convention / model-judgement issue, not a harness bug.
Cosmetic bug: in narrow review cells the slice chip overlapped the top orientation letter ("cor 21 S").

## Changes before pass 5 (freeze candidate)
- `_draw_frame`: top (or bottom) orientation letter shifted right of the title chip in narrow panels. No behavioural change.

## Pass 5 — `phase1-05`, 2026-10-18, harness_sha256 `3c915449843a9088674884fab22e3535f68b0693cbfa6a9e096a5e2011520782` (freeze candidate)
30/30 finished, 0 harness/internal/runner errors, no forced finishes, mark coverage 30/30 → **freeze criteria hold**.
**Mean 0.597, median 0.750**; HD95 median 19.8 mm. 6.0 tool calls, 1.03 model prompts, cost ≈ $1.2 for 30 cases.
Tags: none 21, localization 7 (F14, F15, L33, L43, N40, V46, V57), delineation 2 (F32 over-seg, N01).
Localized cases (23): agent 0.779 vs one-click oracle 0.756 on the same cases.
Confidence: low 5 (3 are localization failures), medium 22 (4 localization failures), high 3 (all correct).
Agent-input errors: 7× `seg_view crop:"marks"` with empty `ids` on the first view (recovered next call) — open usability item.
Per-case Dice: F14 0, F15 0.002, F32 0.420, F44 0.885, F49 0.905, F50 0.889, L19 0.838, L20 0.941, L33 0.000, L35 0.529,
L38 0.967, L43 0, M01 0.907, M07 0.947, M08 0.736, M09 0.954, M21 0.815, M33 0.531, N01 0.473, N18 0.876, N21 0.856,
N26 0.930, N34 0.586, N40 0, V37 0.765, V46 0, V54 0.707, V55 0.845, V57 0, V65 0.618.

Status: harness v1 frozen candidate pending discussion with Sameed (agent behaviour issues are model/protocol questions, not harness bugs).
