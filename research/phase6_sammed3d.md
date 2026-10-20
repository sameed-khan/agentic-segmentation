# Phase 6 — SAM-Med3D-turbo as the interactive segmenter (harness v2 swap)

Branch `phase6-sammed3d` (worktree `/home/ubuntu/agseg-p6`). Question: how much of the agent
result is the agent vs the segmenter? Same agent, same tools/prompts/renders/marks/budgets/
observation formats; only the interactive segmenter changes from nnInteractive to
SAM-Med3D-turbo. Literature context: `research/literature/segmenter_swap_medsam2.md` (SAM-Med3D is the
"native-3D point/mask-prompted sensitivity arm"; boxes/lasso/scribbles are emulated, see mapping).

## 1. What was built

| Piece | Where |
|---|---|
| Local-model project (own uv env, torch 2.6.0+cu124, torchio 1.2.1, medim 0.1.2) | `harness/local_models/sammed3d/` |
| Engine (official checkpoint via `medim.create_model("SAM-Med3D", ...)`, one patch forward = official `sam_model_infer` step) | `src/sammed3d/engine.py` |
| Pure geometry: resample, patches/tiling, z-norm, mask prompts, stitching, prompt maps | `src/sammed3d/geometry.py` |
| Service: session API mirroring the nnInteractive remote-session methods; `--fake` backend | `src/sammed3d/service.py` |
| Client `SamMed3DSession` (call-signature drop-in; no torch) | `src/sammed3d/client.py` |
| segd integration: `segmenter`/`segmenter_url` at case open, trajectory mapping log | `harness/server/src/segd/{app,case_state,config,sammed3d}.py` |
| Runner `--segmenter`, `--segmenter-url` (→ `meta.json`, `batch_summary.json`) | `harness/runner/run_cases.ts` |

Service: `cd harness/local_models/sammed3d && uv run sammed3d-service --port 8095` (tmux `sammed3d`).

## 2. Checkpoint

Official `sam_med3d_turbo.pth` (HuggingFace `blueyo0/SAM-Med3D`, "SAM-Med3D-turbo": fine-tuned on 44
datasets), downloaded with `scripts/fetch_checkpoint.py`,
`sha256 = 899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c` (pinned in the script, in
`checkpoints/sam_med3d_turbo.sha256` and in the engine; the engine refuses to load on mismatch and
reports it in `/healthz`). Official code reference: `uni-medical/SAM-Med3D` @ `f3de1fa` (read-only
clone; `utils/infer_utils.py`). Code licence Apache-2.0; no separate weights licence is stated in the
repo README — verify on the HF page before redistributing anything.

## 3. Preprocessing and inference (identical to the official `infer_utils.py` except patch placement)

1. **Resample** the session image (the crop segd uploads, 1 mm isotropic MSTT) to 1.5 mm, linear,
   torchio's centred reference grid (`new_size = floor(size·spacing/1.5)`). Reimplemented with scipy;
   `tests/test_geometry.py::test_resample_matches_torchio_exactly` asserts equality with `tio.Resample`.
2. **Patch** of 128³ model voxels (192 mm), zero-padded where it leaves the volume (as official
   `CropOrPad`); **ZNormalization over x>0** per patch (`tio.ZNormalization(masking_method=x>0)`, applied
   to all voxels; tested against torchio). A constant/empty patch is mapped to 0 (torchio would raise).
3. **Forward** exactly as `sam_model_infer`: image encoder → prompt encoder (points + 32³ low-res mask
   prompt, zeros on the first call) → mask decoder → trilinear ×4 to 128³ (align_corners=False) →
   logits. Mask = sigmoid(logits) > 0.5.
4. **Back to the session grid** by nearest-neighbour resampling of the binary mask (official
   `data_postprocess`).
5. Iterative clicking: every prediction re-runs with ALL accumulated points and the previous
   prediction's low-res mask (the same window reuses the raw low-res logits as the official carry-over).

**Validation against the official code path** (F14, official `tio.Resample → ToCanonical →
CropOrPad(label) → ZNormalization`, click at the GT centre voxel): our engine on the official ROI gives
logits with max abs diff **0.0**; our resample+crop+z-norm of the same window differs from the official
ROI by 5e-7 (resample 1e-4 before z-norm); the official click gives Dice 0.667 vs GT.

### Patch placement (no GT)

* First prediction: one window centred on the first positive prompt (model space), zero-padded at edges.
* Later: bbox(positive points ∪ previous mask). Fits in 128 voxels on every axis → one window, still
  starting at the first positive and shifted minimally to cover the box.
* A box > 128 voxels (192 mm) on some axis → tiled per axis: tiles at stride ≤ 96 voxels (≥25 % overlap,
  end tile aligned), box padded by 16 voxels on tiled axes; axes that fit keep a single tile centred on
  the first positive. Cap 27 patches: the per-axis tile count is coarsened (stride up to 128, i.e. zero
  overlap) before any tile is dropped; if even that exceeds 27, the tiles farthest from the first positive
  are dropped; both are written to `patch_note` in the log.
* Each patch gets the prompt points inside it + its 32³ slice of the previous mask. A tile with neither
  (no point inside, no high-res previous-mask foreground in the window) is skipped and logged. Logits of overlapping patches are averaged;
  threshold at 0.5 after averaging.

## 4. Prompt mapping (harness → SAM-Med3D)

| Harness call | SAM-Med3D prompt | Fidelity |
|---|---|---|
| `seg_point` add/erase (n marks) | n native points, label 1/0; one prediction after the last | native |
| `seg_box` (2D box on the active slice, include flag) | centre + 4 quarter points (0.25/0.75 of each in-plane extent) on the slice voxel, all with the include label (3D box would be centre + 8) | emulated, lossy |
| `seg_lasso` | filled contour → ≤5 deterministic farthest-point samples (start at the EDT maximum), label per polarity | emulated, lossy |
| `seg_scribble` | ≤5 deterministic evenly spaced samples along the 1-voxel stroke (BFS order from the lexicographically smallest endpoint) | emulated, lossy |
| `seg_refine` / sync rule (initial seg) | resets all earlier points and the first-positive anchor (nnInteractive semantics); the mask → low-res mask prompt for the patches; `seg_refine` (predicting) adds one positive point at the EDT maximum of the mask as window seed, the state-only sync call adds none | emulated |
| `seg_paint`, `seg_edit`, undo/checkout, accept | no model call; mask changes set `nni_dirty`, the next prompt does `reset_interactions()` + initial seg as for nnInteractive | external |

The service returns, and segd stores in the trajectory record (`segmenter_interactions`), the real
mapping of every call: each prompt call since the previous prediction with its `mapped_points`
(`[i, j, k, label]`, session voxels), then `patches_used / patches_skipped / windows_model_vox /
n_points_total / patch_note`. All sampling is deterministic and independent of the image content and
of GT.

## 5. Tests

* `harness/local_models/sammed3d`: `uv run pytest -q` — 33 passed (geometry, resample = torchio, z-norm =
  torchio, patch placement/tiling incl. a 400 mm synthetic mask, stitching, mask prompts, prompt maps,
  service API via the fake backend, client signature parity against the real nnInteractive client).
* `harness/server`: `NNI_URL= uv run pytest -q` — all pass, incl. `tests/test_segmenter_swap.py` (swap path
  with the fake sammed3d service over real HTTP: point/lasso/box/paint+sync/refine, mapping log in the
  trajectory, session expiry → reconnect, HTTP open with `segmenter`) and
  `tests/test_p6_default_identity.py` (identical observations and PNG bytes between this worktree's
  default segmenter and the main tree's segd source for open/sheet/view/point/box/edit/undo/finish-review/
  finish-confirm with FakeNNI).
* `harness/runner`: `npx tsc --noEmit -p .` clean.

Bugs found and fixed while reviewing the earlier worker's code (all with regression tests):
(1) the first-positive window seed was in session space but used as model space (window 1.5× off for 1 mm
data — the first real call returned an empty mask); (2) the previous-mask bbox used `prob > 0` instead of
`> 0.5`; (3) the stitched union crop was mapped back as if it started at the model origin; (4) window
placement for non-tiled axes of a tiled box started at the box edge instead of centring on the prompt;
(5) the mapping log only kept the last call of a multi-mark `seg_point`; (6) the single window was clamped
into the volume (moved the prompt off-centre; official pads).

## 6. GPU smoke (real service, A10 shared; swap path through `CaseState`)

Service process VRAM **1.8 GB** (nvidia-smi per process; torch 2.6, fp32, one 128³ patch at a time);
latency per call (HTTP incl. 1.5 mm resample cache, forward ×patches, grid mapping): 0.9–1.5 s with one
patch, 1.8–2.9 s with 3–4 patches, refine 2–3.6 s (7 s on M33; includes the render). Positive point at
the GT voxel nearest the centroid, then a positive point in the largest false-negative blob, then a
negative point in the largest false-positive blob, then `seg_refine` (GT only used for choosing the smoke
clicks and for Dice):

| case (GT extent mm) | step | Dice | patches | latency s |
|---|---|---|---|---|
| F14 (83×183×64) | point+ GT centroid | 0.681 | 1 | 1.0–1.3 |
| | point+ largest FN | 0.688 | 1 | 1.0 |
| | point− largest FP | 0.704 | 1 | 1.0 |
| | refine | 0.680 | 1 | 2.1 |
| M21 (89×101×333; tiling) | point+ GT centroid | 0.399 | 1 | 1.5 |
| | point+ far-high end | 0.685 | 1 | 1.5 |
| | point+ far-low end | 0.700 | 3 | 1.8 |
| | point+ largest FN | 0.779 | 3 | 1.8 |
| | point− largest FP | 0.759 | 3 | 1.9 |
| | refine | 0.753 | 3 | 3.6 |
| M33 (154×126×435; tiling, 2.0 L tumour) | point+ GT centroid | 0.179 | 1 | 2.3 |
| | + far-high / far-low ends | 0.207 / 0.203 | 2 / 3 | 2.5 / 2.7 |
| | point+ FN / point− FP | 0.181 / 0.228 | 4 / 3 | 2.9 / 2.8 |
| | refine | 0.275 | 3 | 6.7 |

Reading: a single click at the GT centroid reproduces the official-pipeline Dice (0.68 vs 0.667 for the
official GT-centred window). On very large tumours a single 192 mm patch cannot cover the lesion; extra
positives at the extremes trigger tiling (up to 4 patches observed) and help on M21 (0.40 → 0.75) but not
on M33 (the model under-segments the 2 L tumour: 0.2–0.28). Early failed attempt for the record:
before fix (1) the first call returned an empty mask (Dice 0). These are single-case, GT-click-oracle
smoke numbers, not a benchmark; they say nothing yet about agent performance with SAM-Med3D.

End-to-end check without an LLM (F14, real marks/renders): `open → view → point → paint → refine → point
erase` through `ops.run_op` on the swap path; observation grammar, renders and trajectory (with
`segmenter_interactions`) as for nnInteractive.

## 7. Caveats for the comparison

* The agent's system prompt (`harness/prompts/system.md`) still says "nnInteractive"; kept verbatim so
  the prompts are identical across arms. Report this when interpreting the swap.
* Box/lasso/scribble are emulated by ≤9 points (information loss relative to nnInteractive's native
  2D box/lasso/scribble); report "native" (point, refine) and "emulated" calls separately in analysis.
* SAM-Med3D-turbo is a point/mask model trained on CT-heavy 3D data (44 datasets); MRI soft-tissue
  transfer is unverified by the authors. Patch size 192 mm: tumours larger than that need tiling and
  more clicks (M33 above).
* `candidates.propose(method="probe")` would still use nnInteractive; the configured method is
  `intensity`, so the swap arm does not touch it.
* Calls on one service are serialised by a model lock; the service accepts up to 8 sessions.

## 8. 5-case design pilot (agent: Sonnet 5.5, thinking high) — NOT run yet

Separate segd instance (port 8766; the sweep's 8765 is untouched), the sammed3d service on 8095 (running):

```bash
# 1. segd of this worktree on its own port (no nnInteractive involved)
tmux new-session -d -s segd_p6 -c /home/ubuntu/agseg-p6/harness/server \
  "uv run uvicorn segd.app:app --host 127.0.0.1 --port 8766 2>&1 | tee /tmp/segd_p6.log"
curl -s localhost:8766/healthz; curl -s localhost:8095/healthz

# 2. the pilot: one case per histotype (F14 F, L20 L, M07 M, N26 N, V37 V), results in /home/ubuntu/agseg-p6/results/
cd /home/ubuntu/agseg-p6/harness/runner
npx tsx run_cases.ts --cases F14,L20,M07,N26,V37 --run-id p6-sammed3d-pilot5 \
  --model anthropic/claude-sonnet-5-5 --thinking high --concurrency 2 \
  --segd-url http://127.0.0.1:8766 --segmenter sammed3d --segmenter-url http://127.0.0.1:8095 \
  --note "phase6 pilot: SAM-Med3D-turbo swap, design 5 cases, sonnet-5-5 high"
```

For the nnInteractive control on the same 5 cases use the same command with `--segmenter nninteractive
--nni-url http://127.0.0.1:1527 --run-id p6-nni-pilot5` (needs the nnInteractive server up).

## 9. Review round 1 (commit b93e04d) — findings, fixes, tests

| # | Finding | Fix | Regression test |
|---|---|---|---|
| 1 | `add_initial_seg` kept old clicks/first positive (nnInteractive resets: `inference_session.py` "WARNING THIS WILL RESET INTERACTIONS") | points + first_positive cleared; interior seed only for predicting calls; `points_reset` logged | `test_service_fake.py::test_initial_seg_resets_clicks_like_nninteractive` (history session ≡ fresh session: windows, points, prompt, mask) |
| 2 | tile skipped when the pooled mask prompt had no positive logit (thin masks) | skip decided by `geometry.prev_mask_in_window` (high-res prob>0.5) | `test_thin_mask_only_tiles_are_not_skipped` (400 mm 1-voxel tube, spy predictor: every tile runs) |
| 3 | back-mapping zeroed the outer half voxel (scipy centre-interval support); forward resample too for spacing>1.5 mm | ITK rules: support [-0.5, n-0.5), clamp inside, round half up; back-maps are separable (no dense coordinate grids) | `test_geometry.py::test_back_mapping_matches_torchio_nearest_*`, `test_session_mask_to_model_matches_torchio_nearest`, `test_resample_matches_torchio_for_coarse_spacing_and_edge_voxels` (TorchIO equality; exact-half tie voxels excluded: ITK's tie rounding is per-axis floating-point noise, ≤1 model voxel) |
| 4 | 2D-box points at slice+0.5 | singleton axis → the slice voxel index | `test_bbox_points_of_a_size_one_slice_box_sit_on_the_slice_voxel`, `test_box_on_a_slice_registers_points_on_that_slice` |
| 5 | identity test could self-compare / used a mutable path | pinned fixture (`git archive 1dcc622`, sha256 manifest) or `SEGD_BASELINE_SRC`; explicit skip; baseline asserted pre-Phase-6 | `test_p6_default_identity.py` |
| 6 | abandoned sessions pinned capacity/memory | idle TTL reaper with in-flight protection; `/healthz` session count/age | `tests/test_session_expiry.py` (capacity recovery, activity refresh, in-flight survives, healthz) |

**Host memory, 8 concurrent opens** (real service, 8 largest test-split images uploaded in full as float16, e.g.
V33 441×441×372, V25 501×501×283; each opened concurrently + one point prediction; GPU process peak 2.2 GB):
service peak RSS **8.6 GB** with the b93e04d code, **6.8 GB** after the fixes (idle 0.6–1.3 GB, 1.7 GB after the
sessions are closed). Both far below 16 GB, so the dense-grid forward resampling was kept (back-mapping is
already separable); segd uploads a body crop above 60 MB packed, so real sweeps are lighter still.
