# Phase 4 harness changes — specification (NOT implemented)

What the segd harness must change to run agents on the Phase 4 datasets
(BEAMSTER, Spinal-Multiple-Myeloma-SEG, UCSD-BMETS-Longitudinal, Longitudinal-CT).
MSTT paths and behaviour stay untouched; everything below is additive.

## 1. Case loading: multi-dataset manifests
Today `segd/data.py` hard-wires MSTT: `DATA_DIR = PROJECT_ROOT / "data" / "mstt199"`
and `CASES_CSV = PROJECT_ROOT / "eval" / "v2" / "mstt_cases.csv"` (data.py:19-20);
`read_cases()` maps the MSTT-specific `image_kaggle`/`masks_kaggle` columns (data.py:70-80).
- Add a manifest-driven reader: `read_cases_phase4(dataset)` consuming
  `eval/v4/<dataset>_cases.csv` (columns `case, target_text, modality, sequence,
  image_path, mask_paths, n_lesions, lesion_volume_ml, split` — absolute paths already).
- `CaseRow` gains `modality: str` and `n_lesions` (precomputed GT component count —
  loading from the manifest avoids loading the GT mask in agent-reachable paths).
- `CaseState.__init__` (case_state.py:126) already takes `target_text`/`sequence`; pass
  the manifest values through `app.py` `POST /cases` (open) — no schema change needed,
  only a dataset argument or per-run env var `SEGD_DATASET`.

## 2. CT windowing in renders (BEAMSTER is MRI; spinal_mm_seg and longitudinal_ct are CT)
`segd/render.py` windows **per-volume percentiles (0.5–99.5) inside the body mask**:
`_window_for()` (render.py:154) + `segd.marks.window_params()` (marks.py:475). For CT that
shows arbitrary HU percentiles — HU must be windowed by **fixed clinical windows**:
- specify a `Modality`-aware window: CT → soft-tissue (lo≈0, hi≈300 HU) or lung
  (lo≈-1000, hi≈100) selectable per case; MRI → current percentile behaviour unchanged.
- cleanest seam: `window_params(image, body)` already receives only arrays; add
  `window_params(image, body, modality=None, fixed_window=None)` and have
  `_window_for()` pass `CaseState`/manifest modality. `render_overview`/`render_sheet`/
  `render_view` thread it via the existing `window` cache on `MarkSet` (render.py:161).
- Whole-body CT also breaks `body_mask` (Otsu, marks.py:416) assumptions? Otsu on HU
  volume lumps air+lung+bone fine, but verify; may need an HU-threshold pre-filter
  (> -300 HU) before Otsu for CT.

## 3. Multi-lesion targets (the biggest change)
MSTT cases have exactly one target lesion per case; nnInteractive interaction state is a
single target buffer, and observation text reports `components N (largest X%)`
(observe.py:13-34) tuned for one object. Phase 4 targets are "ALL lesions":
- Target text (manifest `target_text`) must say e.g. "brain metastases (all enhancing
  lesions) — segment EVERY lesion, not just the largest".
- Segmentation ops (`ops.py`): every model prompt (`point`/`lasso`/`scribble`/`box`/
  `refine`) currently accumulates into ONE object. Options: (a) keep one multi-component
  object and let agents click each lesion (nnInteractive supports multi-component
  targets via repeated positive prompts into the same buffer — no API change), or
  (b) per-lesion object switching. Recommend (a) + explicit `seg_edit keep_component`
  semantics fix: today `keep_component` keeps the largest component (observe/ops) — for
  multi-lesion targets it must keep ALL components above a small size floor or take a
  component id list, else every `edit` nukes the other lesions.
- Observation text: report lesion COUNT and per-lesion volumes (manifest has GT counts;
  the observation must derive counts from the current mask only) + flag "expected
  N lesions (from the case card)" so the agent can self-check detection completeness
  (n_lesions is GT — publishing it in the observation is a scoring leak; instead put
  "expected lesion count" in the CASE CARD given to the agent at open — decision needed).
- Scorer: `eval/v2/score.py` computes global Dice/HD95/NSD per version
  (score.py:166-211, `dice_from_counts`, `surface_metrics`). Add per Phase 4:
  global Dice over all lesions (union — same formula, already fine), **per-lesion
  detection F1** (GT component is "detected" if Dice(pred∩gt_component /
  gt_component) ≥ 0.1 or an overlap-fraction threshold), mean per-lesion Dice,
  and small-lesion (< 1 mL) breakdown. Surface metrics per lesion on the matched pair.
- Budgets: multi-lesion + tiny lesions need more prompts; suggest max_model_prompts
  scaled by manifest n_lesions (e.g. 10 + 2·n_lesions, cap 30) — CaseState budgets
  (case_state.py:126, CONTRACTS §4) are already configurable per case via `POST /cases`.

## 4. Very small lesions (BEAMSTER Dataset_B, < 1 mL)
- Render: at 1 mm iso a 0.3 mL lesion is a ~8 mm blob — visible, but mark density must
  rise: `compute_marks` (marks.py:603) thinning constants (10 mm coarse / 6 mm fine
  clearance) assume few large lesions; for small lesions raise per-panel mark caps
  and shrink the clearance near the mask bbox.
- `render_view` crop="mask" already zooms to mask bbox + 15 mm (CONTRACTS §3) — good
  for < 1 mL targets; keep.
- Mark the case density in the manifest (n_lesions, lesion_volume_ml present) and let
  the runner choose budgets per case instead of per run.

## 5. Data layout / spacing
- Phase 4 cases are converted to 1 mm isotropic canonical RAS by `eval/v4/convert_to_iso.py`
  into `data/phase4/<dataset>/nifti_iso/` — same conventions as `segd.data.load_image`
  expects (canonical, float32, spacing from header). Whole-body CT at 3 mm converted to
  1 mm is ~2-6 TB for all patients: either convert per split on demand (implemented:
  `convert_to_iso.py <dataset> design|test`) or make the harness spacing-aware — the
  hardcoded "1 mm iso" appears in `segd/marks.py` `MarkSet.visible` (slice-window in
  voxels) and CONTRACTS §2; the mm↔voxel conversions elsewhere already use `spacing`.
  If we keep nifti_iso for everything, no harness change is needed for spacing.
- `load_mask` (data.py:119) majority-votes multiple masks; Phase 4 manifests have a
  single mask per case (integer labels for Longitudinal-CT) — mask loading must map
  "any label > 0" → binary (already the behaviour) and keep integer labels for
  per-lesion scoring (add `load_mask_labels`).

## 6. Runner / logging
- `harness/runner/run_cases.ts` selects cases from `eval/v2/mstt_cases.csv`; point it at
  `eval/v4/<dataset>_cases.csv` (split filter) and record `dataset` in
  `batch_summary.json`.
- `eval/v2/export_run.py`/`build_log_tables.py` assume the MSTT `histotype` column;
  make the CSV column set pluggable or emit `dataset=<name>, sequence` in tables.
- DVC: `results.dvc` unaffected; phase-4 data stays outside git at
  `data/phase4/` (never `dvc add` it).

## Explicitly out of scope for Phase 4 (approved direction)
No harness edits: this document is the spec. Items needing a decision before
implementation: expected-lesion-count disclosure at `seg_open` (scoring-leak tradeoff),
per-lesion vs single-object nnInteractive state, CT window presets.
