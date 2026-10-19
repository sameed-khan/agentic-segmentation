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
| dev-02-max | max | abandoned (33/270 calls valid; rest 429 rate limit) | | | | |

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
| phase1-08-max | max | abandoned (see below) | | | | | | |

- Higher thinking makes the agent **search more** (more sheets/views) and localize more cases: L43 and V57 found at high
  and xhigh, V46 found at xhigh; L35 extended to the full tumor (0.53 → 0.90 at high). xhigh mean 0.706 is the first
  pass above the one-click oracle mean (0.697).
- Model prompts per case rise only to 1.3–1.7: the agent still rarely refines with nnInteractive; gains are localization.
- Persistent failures across all levels: F15 (2 mL), L33 (T2 lipomatous), N40; delineation: M33, N01, N34 (under-segmentation
  relative to the inclusive GT), F32 (over-segmentation).
- max thinking: single Phase 2 calls on hard cases take 6–8 min and 47–66 k output tokens; Phase 1 cases hit the 10-min
  wall-time limit (forced finishes) — results pending.

### E12. Example image in the agent's task — run `phase1-09-xhigh-ex` (code `harness-v1-3-gda0ce0b`)
Harness v1 at xhigh; the task message carries one leave-one-out example (another design case of the same histotype and
sequence, GT outline in axial/coronal/sagittal; L33 gets a T1 lipomatous donor), images from the Phase 2 prepare step.

| run | mean | median | loc. fails | localized mean | tool calls | prompts | cost USD |
|---|---|---|---|---|---|---|---|
| phase1-07-xhigh (no example) | 0.706 | 0.832 | 4 | 0.795 | 21.6 | 1.67 | 7.58 |
| phase1-09-xhigh-ex (example) | 0.700 | 0.805 | 4 | 0.789 | 20.5 | 1.73 | 7.31 |

- Per-case changes > 0.05 go both ways: better M33 0.38→0.53, N01 0.44→0.53, F32 0.49→0.56, L35 0.87→0.93;
  worse F14 0.67→0.49 (localization lost), L43 0.90→0.74, V57 0.70→0.60. Net effect ≈ 0, within run-to-run variance.
- **Decision:** do not add example images to the agent (consistent with E10, where S-ex did not change pointing).

### E13. Oracle iterative-refinement ceiling (no LLM) — run `oracle_refine` (`eval/v2/oracle_refine.py`)
Click 1 = GT centroid (same as one-click); clicks 2..10 = interior point (EDT max) of the largest error region,
positive for a false-negative region, negative for a false-positive region. Simulated "perfect user".

| clicks | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|
| mean Dice | 0.696 | 0.788 | 0.844 | 0.864 | 0.875 | 0.884 | 0.889 | 0.898 |
| median Dice | 0.739 | 0.844 | 0.872 | 0.893 | 0.902 | 0.909 | 0.915 | 0.922 |

- **Point prompts are sufficient to reach the GT convention:** three well-placed clicks give +0.15 mean Dice over one
  click; five give +0.18. The agent at xhigh (0.706, ~1.7 prompts) captures almost none of this.
- The under-segmented cases need **positive** clicks (missed tumor), not boundary polishing: M33 0.63→0.82 (5 clicks,
  7 of the first 8 positive), N01 0.48→0.79, N34 0.49→0.78 (3), N40 0.27→0.74, L43 0.36→0.87 (3), V65 0.61→0.85 (3).
  F32 needs negative clicks (over-segmentation 0.40→0.80). Small lesions saturate lower (F15 0.75, V57 0.79 at 10 clicks).
- **Interpretation:** the delineation gap is a *decision* problem (the agent does not see or believe that tumor is
  missing), not a precision/tool problem: precise pointing at boundaries is not what the oracle needed.

### Failure analysis of never-found / persistently low cases (all 8 passes; script `/tmp/fail.py`, numbers below)
Contrast = (mean GT intensity − mean of a 5-voxel ring around GT) / ring SD.

| case | GT mL | max axial area mm² | contrast | what the agent does | failure type |
|---|---|---|---|---|---|
| F15 | 2.2 | 137 | 0.27 | clicks a brighter structure 7–10 mm away in every pass | tiny **and** iso-intense |
| N40 | 13.6 | 180 (thin, 150 mm long along the nerve) | 3.08 | always picks a round mass at z≈230 outside the GT z-range (34–182); xhigh reasoning sees the bright posterior structure at z 67–167 and dismisses it as vessels/nerve | elongated fusiform lesion mistaken for normal neurovascular bundle (knowledge/expectation of a round "mass") |
| L33 | 155 | 2044 | 0.17 | picks other structures 21–41 mm away | fat-suppressed lipomatous tumor ≈ suppressed fat (no contrast) |
| F14 | 64 | 2012 | 0.73 | picks the large dark anterior pelvic mass (correct in 2/8 passes) | conspicuous distractor |
| F32 | 220 (2 components, 124 + 96 mL) | 4188 | 0.62 | clicks within 3 mm of / inside GT, then over-segments (≈ 820 mL) | delineation (over-seg), not localization |

- Not primarily a size/downsampling problem: slices are rendered *up*-sampled (≥ 768 px per panel), marks cover every
  lesion (30/30), and in the single-slice study (E4/E10) the model misses F14, F15 and L33 even on the correct slice at
  768 px. Contact sheets are mildly downscaled by the API (1.0–1.6 MP → ~1.15 MP), so a 13 mm lesion is ~15–20 px per
  sheet cell; this could matter for F15 during search but F15 is also missed when shown directly.

### E14. Candidate-correction generators, offline benchmark — `candidate_bench` (`eval/v2/candidate_bench.py`, `segd/candidates.py`)
Idea (discussion with Sameed, 2026-10-19): turn "find your own error" into "accept/reject proposed corrections".
Generators are GT-free (image + current mask + nnInteractive): **intensity** (grow into adjacent voxels within the
mask's intensity band, ≤ 30 mm, split into 10 mm distance shells), **supervoxel** (complete partially covered SLIC
supervoxels + similar neighbours), **ends** (continue the mask past its end slices while the footprint signal
persists), **probe** (nnInteractive what-if positive/negative clicks around the mask in a scratch session), combo.
Starting states: the agent's 240 final masks from 8 passes; 33 with zero GT overlap skipped; 207 evaluated.

| method | cand/state | FN recall @1/@3/@8 | add precision | oracle-accept Dice (start 0.739) | accept-all | time/state |
|---|---|---|---|---|---|---|
| intensity | 3.9 | .30/.54/.55 | .30 | .786 (+.047) | .641 | 1.6 s |
| supervoxel | 6.1 | .36/.56/.57 | .20 | .777 (+.038) | .626 | 5.2 s |
| ends | 3.3 | .04/.06/.07 | .24 | .750 (+.011) | .644 | 2.7 s |
| probe | 4.3 | .47/.60/.63 | .30 | .784 (+.045) | .602 | 28.9 s |

Under-segmented states (84; recall < 0.7; start 0.555): intensity oracle-accept 0.663 (+.108, add precision 0.50),
probe 0.657 (+.102; best on L +0.13), supervoxel +0.09, ends +0.02. By histotype (intensity): N +0.22, M +0.14, V +0.07, L +0.07, F +0.05.
- The candidate ceiling (+0.05 overall / +0.11 under-segmented) is well below the click oracle (E13, 0.875 with 5 clicks):
  about half of the missed GT is never proposed; the GT in e.g. N34 is a wide rim around the bright core plus thin
  extensions that intensity continuation does not reach.
- Precision is the bottleneck: accepting everything is *worse* than accepting nothing. The tool only helps if the agent
  judges candidates well.
- **Decision:** integrate `intensity` (max 5 candidates) as `seg_candidates` / `seg_accept`, and show candidates in the
  first `seg_finish` review (one required candidate round). Probe kept as a possible second source (L/V cases).

### E15. Agent with candidate corrections, thinking high — run `phase1-10-high-cand` (tag `harness-v2-rc1`)
Integration (code `dd8e851`): tools `seg_candidates`, `seg_accept`; the first `seg_finish` review of every version also
shows ≤ 5 intensity candidates (second image, magenta outlines) and asks the agent to accept/reject each; ≤ 3 accept rounds.
Reviewed by gpt-6.1-sol (3 majors: stale authorization after undo/checkout, authorization before the sheet rendered,
non-transactional persistence after commit) → fixed and regression-tested (151 tests).

| run | mean | median | loc. fails | localized mean | tool calls | prompts | cost USD |
|---|---|---|---|---|---|---|---|
| phase1-06-high (no candidates) | 0.659 | 0.806 | 5 | 0.791 | 10.4 | 1.27 | 2.50 |
| phase1-10-high-cand | 0.684 | 0.801 | 6 | 0.819 | 9.7 | 1.30 | 2.53 |

- The +0.025 mean is **not from candidates**: the agent accepted a candidate in **1 of 30 cases** (V57, a wrong one).
  Differences come from localization/run variance (F14 0 → 0.72, V46 0.60 → 0.85, N01 0.00, F44 0.89 → 0.73).
- **Candidate judgement** (offline, `eval/v2/candidate_judgement.py`: the shown candidates are recomputed deterministically
  and scored against GT): 32 reviews showed 100 candidates; **18 were mostly tumour** (> 50 % inside GT); the agent
  accepted **0 of these 18** and 1 of the 82 bad ones. Rejected good candidates include M08 C1–C3 (precision 0.97–0.99),
  M33 C1 (0.83), N34 C1/C4/C5 (0.73–0.93), L43 C2 (0.77), F50 C4 (0.98), V37 C1 (0.89).
  Accepting exactly the good ones would have added +0.032 mean Dice over the reviewed states (M33 0.63 → 0.82,
  N34 0.44 → 0.70, L43 0.67 → 0.82, F44 0.73 → 0.85, M08 0.74 → 0.83).
- **Interpretation:** the agent rejects almost everything (specificity 81/82, sensitivity 0/18). The bottleneck is
  judgement of "is this rim/extension part of the lesion" (annotation convention), not candidate availability or tooling.
  These 100 labelled decisions (images + GT label) are a ready evaluation set for decision models (step 4) or prompt optimisation.

### E16. Candidate-decision replay: does a "what counts as the lesion" line change judgement? — `candidate_replay__e16`
Prompt line (agreed with Sameed, added to `harness/prompts/system.md` as the first Rules bullet): *"What counts as the
lesion. Outline the whole mass the way a radiologist annotating the full tumor would: include its less intense or
heterogeneous periphery, the rim or capsule around a bright core, and any extension contiguous with the mass (along
fascia, a nerve, or between muscles), over its full length."*
Replay: the 27 candidate reviews of E15 (100 candidates, 18 GT-positive; `harness/study/prepare_candidate_replay.py`)
sent as single tool-less calls (Sonnet 5.5, thinking high, both review images + review text) with the E15 system
prompt (base) or with the line (lesion); 3 reps; `harness/study/replay_candidates.ts`, `score_candidate_replay.py`.

| condition | sensitivity | specificity | accepted / review | mean Dice Δ | oracle Δ |
|---|---|---|---|---|---|
| base | 0.056 | 0.988 | 0.07 | −0.003 | +0.037 |
| lesion | 0.130 | 0.988 | 0.12 | −0.000 | +0.037 |

- The line roughly doubles sensitivity (only N21 C1 and M08 C1 flip) at unchanged specificity; Dice impact ≈ 0.
  The large rims (M33 C1, precision 0.83; N34 C1, 0.73) are still rejected in every call: the model describes them as
  "a thin shell of muscle/fascia/edema around the mask … would add far too much". One consistent false positive
  (V57_v2 C4, both conditions).
- Single-turn replay reproduces the in-agent behaviour (E15: 0/18), so the replay set is a cheap proxy for judgement.
- Side check (`eval/v2/dilation_check.py`): uniform dilation of final masks does not substitute: +1 voxel gives
  +0.006 mean Dice, ≥ 2 voxels lowers it (xhigh: r0 0.706, r1 0.712, r2 0.699, r3 0.680). The GT rims are case-specific.
- **Interpretation:** the remaining delineation gap is an annotation-convention disagreement that the model actively
  resists on visual grounds; instruction alone does not move it. Kept the line (harmless, slightly positive).

## 6. Phase 3: test split (169 cases) — preregistered 2026-10-19, before any test data was downloaded or viewed
Agreed with Sameed: skip the design-set confirmation pass of harness v2; run everything on the test split at thinking
**high** (primary), then a second full pass at **xhigh** for comparison.

**Frozen configuration:** git tag `harness-v2` (agent-facing prompts/tools identical to commit `f3dfede`: harness v1 +
`seg_candidates`/`seg_accept` + candidate round in the finish review + lesion line; the tag adds runner `--resume`
infrastructure only). Model `anthropic/claude-sonnet-5-5`, budgets 40 tool calls / 15 model prompts / 10 min wall,
3 nudges then forced finish, concurrency 4, target text = histotype text, sequence = the annotated one, no seed.
Nothing in `harness/` changes between the high and xhigh passes. Cases that end through a provider error
(rate/session limit) are rerun from scratch (`run_until_done.sh`); this is logged per case (`_incomplete/`).

**Reference masks:** single-rater cases: the one mask; 15 three-rater cases: voxel-wise majority vote
(`segd.data.load_mask`); pairwise inter-rater Dice reported alongside.

**Primary endpoint:** per-histotype **mean** Dice of the high pass vs the published U-Net (F 0.507, L 0.744, M 0.823,
N 0.748, V 0.584; mean 0.681). **Gate:** overall mean (unweighted over the five histotypes) ≥ 0.681 **and** higher than
the U-Net on ≥ 3 of 5 histotypes. Unpaired comparison against group means (no per-case U-Net predictions).
**Secondary:** case-weighted mean, medians, HD95, NSD (scorer defaults), localization failures (first positive click
outside GT and Dice < 0.5), tool calls, prompts, candidate use, cost; xhigh vs high (paired per case, Wilcoxon);
one-click nnInteractive at the GT centroid (no LLM) on the same cases; Phase 2 localization study on test
(conditions R, S, S-gen, 3 reps, thinking high) for hit rate and mark coverage.
**Not changed after seeing test results:** prompts, tools, budgets, scorer, endpoint. Design-set results (E1–E16) are
development; only test-split numbers are reported as results.

**Deviation log (Phase 3):**
- 2026-10-19 06:40, high pass, after 168/169 cases: V25's `seg_edit open radius_mm=14` on a 503 mL mask took 659 s in
  `scipy.ndimage.binary_opening` (cost ∝ ball volume) and blocked segd; the case hit the wall-time limit and the forced
  finish timed out (harness failure). Fix (tag `harness-v2.0.1`): `segd.ops.ball_morphology`, exact-EDT implementation
  of the same operations, **bit-identical** to scipy (randomised test incl. border cases; the V25 opening reproduces
  the logged 377.2 mL exactly), 2.9 s instead of 659 s. No other case logged an error or used a large radius.
  V25 is rerun from scratch on the fixed code; all other high-pass cases ran on `harness-v2`. Runner: wall-time
  aborts are no longer classified as provider errors. The xhigh pass runs entirely on `harness-v2.0.1`.

### E17. Phase 3 primary result: test split, thinking high — run `phase3-high` (harness-v2; V25 on v2.0.1)
169 cases, 0 forced finishes, 0 harness-tagged cases after the V25 rerun, mean 9.7 tool calls / 1.41 prompts, $13.88.

| histotype | n | agent mean | agent median | published U-Net mean | one-click oracle (GT centroid) | loc. fails | mean if localized |
|---|---|---|---|---|---|---|---|
| F | 34 | 0.504 | 0.666 | 0.507 | 0.625 | 9 | 0.678 |
| L | 34 | **0.798** | 0.869 | 0.744 | 0.626 | 3 | 0.860 |
| M | 34 | 0.792 | 0.875 | 0.823 | 0.811 | 2 | 0.841 |
| N | 34 | 0.746 | 0.873 | 0.748 | 0.781 | 4 | 0.831 |
| V | 33 | **0.602** | 0.799 | 0.584 | 0.733 | 9 | 0.828 |
| all | 169 | 0.688 (unweighted) / 0.689 (cases) | 0.837 | 0.681 | 0.715 | 27 | 0.812 |

- **Gate (preregistered): not met.** Unweighted mean 0.688 ≥ 0.681 ✓, but above the U-Net on only **2/5** histotypes
  (L +0.054, V +0.018); F −0.003 and N −0.002 are ties within noise, M −0.031.
- **Localization is the dominant loss:** 27/169 (16 %) wrong structure (first positive click outside GT, Dice < 0.5),
  9 each in F and V. Localized cases average 0.812, above every U-Net histotype mean except M.
- Agent beats the one-click oracle only on L (+0.17, the lipomatous tumours where one click captures too little);
  elsewhere the oracle (which is given the location) is higher, consistent with localization being the gap.
- **Confidence is informative:** high 0.927 (n 16), medium 0.750 (n 127), low 0.242 (n 26).
- Candidates: accepted in 7 cases (M34, N36, V42, V48, V50, V56, V61); explicit `seg_candidates` in 22.
- Multi-rater cases (15): mean pairwise inter-rater Dice 0.991 (the three masks are near-identical, so they give no
  useful variability ceiling); agent vs majority 0.693, vs individual raters 0.692 (`eval/v2/interrater.py`).
- One-click oracle on test: 0.715 mean (design 0.697), `results/oneclick_test/`.

## 7. Next experiments (planned)

| ID | Question | Runs |
|---|---|---|
| E10 | Do reference examples / descriptions improve localization? Does thinking level change pointing? | Phase 2 `dev-02`: S-ex, S-desc (medium) + S, S-ex at high/xhigh/max |
| E11 | Does a higher thinking level make the agent iterate and localize better? | Phase 1 harness v1 at high, xhigh, max |
| E12 | Does an example image in the agent's task improve localization or extent end to end? | `phase1-09-xhigh-ex`: harness v1 at xhigh + leave-one-out example (another design case, same histotype/sequence, GT outline in 3 planes; L33 gets a T1 lipomatous donor) attached to the task message (`run_cases.ts --examples`). Compare with `phase1-07-xhigh`. |
