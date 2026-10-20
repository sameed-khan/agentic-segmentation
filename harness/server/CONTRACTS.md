# segd interface contract (v1)

Binding contract between the segd modules, the pi extension, `segctl`, and the runner. Read
`research/harness_design_v2.md` §5 first; this file pins down what §5 leaves open. If you must deviate,
write the deviation at the bottom of this file under "Deviations" with a reason.

Project: `harness/server` is a uv project (`uv run ...` from that dir; src layout, package `segd`).
Dependencies are already added (numpy, nibabel, scipy, scikit-image, pillow, blosc2, fastapi, uvicorn,
httpx, nninteractive-client, pandas; dev: pytest). Do not add heavy deps without noting it here.

## 0. Hard rules
- **GT never reaches the agent.** Only `segd.data.load_mask` reads masks, and only offline code (scoring,
  diagnostics, baseline, tests) may call it. `app.py`, `case_state.py`, `ops.py`, `render.py`, `marks.py`
  must never import or call `load_mask`.
- No raw coordinates in tool arguments or tool outputs shown to the agent (no voxel indices, no mm positions).
  Slice indices ARE allowed (they are navigation, not pointing).
- Every image shown to the agent is saved to `renders/` and every operation logged to `trajectory.jsonl`.

## 1. Coordinates and display
- Arrays: `[i, j, k]` voxel indices after `nib.as_closest_canonical` (see `segd/data.py`). MSTT affines are
  identity, 1 mm isotropic, nominal RAS+: +i = patient Right, +j = Anterior, +k = Superior. Visually checked on
  design cases: A/P and S/I plausible; R/L cannot be verified for single limbs (say so in the case card).
- Axis names: `axial` = fixed k (shows i,j), `coronal` = fixed j (shows i,k), `sagittal` = fixed i (shows j,k).
- Display convention (radiological): axial: patient right on image LEFT (column 0 = max i), anterior at top
  (row 0 = max j). coronal: columns as axial (column 0 = max i), superior at top (row 0 = max k). sagittal:
  anterior on image LEFT (column 0 = max j), superior at top (row 0 = max k). Panels print orientation letters at
  the four edges and a 10 mm scale bar, and the slice index (e.g. `axial 143/342`) in a corner.
- `segd.render` must expose a pure pair of functions `voxel_to_panel(axis, slice_idx, ijk, view_geom) -> (x, y)`
  and `panel_to_voxel(axis, slice_idx, xy, view_geom) -> ijk` (view_geom includes crop box and scale), used by
  both rendering and the Phase 2 raw-pointing study. Round-trip tests required.

## 2. Marks (`segd/marks.py`)
```python
@dataclass
class Mark:
    id: int                 # global, stable, 1..N; coarse marks get the lowest IDs
    kind: str               # "blob+" | "blob-" | "region"
    ijk: tuple[int,int,int] # voxel (never shown to agent)
    scale_mm: float
    level: str              # "coarse" | "fine"  (coarse ⊂ displayed at overview; fine revealed in crops)

class MarkSet:
    marks: dict[int, Mark]
    def visible(self, axis: str, slice_idx: int, box2d: tuple|None, level: str) -> list[Mark]
        # mark is visible on a slice if |mark_slice_coord - slice_idx| <= max(scale_mm/2, 2) (in voxels; 1 mm iso)
        # level "coarse" -> coarse marks only; "fine" -> coarse + fine; cap ~50 labels per panel (thin by priority)
    def save(path) / load(path)

def compute_marks(image: np.ndarray, spacing) -> MarkSet   # deterministic; params as module constants
def body_mask(image) -> np.ndarray                          # Otsu, largest component, holes filled
def edge_marks_2d(slice2d, mask2d_body, box2d, max_n=40) -> list[tuple[int,int]]  # per-render, in-plane (row/col of the 2D slice array, NOT panel px)
```
Edge marks get IDs `"E1".."En"` valid only for the render that produced them.
Agents refer to global marks by integer and edge marks by string `E<n>`.

## 3. Rendering (`segd/render.py`)
Functions return `(PIL.Image, RenderInfo)`. `RenderInfo` records axis, slice, view_geom, and the list of marks
drawn with their panel positions (and edge marks `E<n>` → in-plane coordinates) so `CaseState` can resolve IDs.
- `render_overview(image, spacing, mask=None, marks=None) -> ...`: three orthogonal panels through the volume centre
  (or mask centroid if a mask exists), orientation letters, scale bar.
- `render_sheet(image, axis, start, stop, n, mask=None) -> ...`: contact sheet of n slices labelled with slice index.
- `render_view(image, axis, slice_idx, crop, overlay, marks_level, markset, mask=None, mark_boxes=None) -> ...`:
  clean panel and marked panel side by side; each panel ≤ 768 px long side; total ≤ 1568 px wide.
  crop ∈ {"full", "mask", "marks"} ("mask" = mask bbox + 15 mm margin on that slice/axis; "marks" handled by caller
  passing a box). overlay ∈ {"none","outline","fill"} (outline default). marks_level ∈ {"none","coarse","fine","edges"}.
  Labels: numbered, contrasting halo, glyph per kind (blob+ ○, blob- □, region △, edge ◇ with `E` prefix).
- `render_outline_change(image, prev_mask, new_mask, axis) -> ...`: the slice (on given axis) with the largest
  changed area, cropped to the new-mask bbox + margin, outlines: new mask (green), removed region (red).
- Window: per-volume percentiles (0.5–99.5) inside the body mask.

## 4. CaseState and ops (`segd/case_state.py`, `segd/ops.py`, `segd/observe.py`, `segd/log.py`)
- `CaseState(case_id, image, affine, spacing, target_text, sequence, out_dir, nni_url|None)`.
  Holds: versions (list of uint8 arrays, id `v0` = empty, then `v1`, ...; parent pointer per version; immutable),
  `current` version id, MarkSet (cached at `<cache>/marks/<case>.npz`), the **active view** (axis, slice, view
  geometry, RenderInfo of the last `seg_view`), counters (tool calls, model prompts), nnInteractive session.
- **Active view rule**: every spatial argument (mark IDs) is resolved in the active view = the last `view`
  render. A global mark ID must be visible in that view (else error "mark 37 is not shown in the current view").
  Resolution projects the mark onto the active slice (in-plane position = mark's in-plane coords, slice = active slice).
  Edge IDs come only from the active view.
- nnInteractive sync rule: the server keeps its own interaction state. After any non-model change to the current
  mask (paint, edit, checkout, undo), set `nni_dirty=True`; before the next model prompt, call
  `reset_interactions()` then `add_initial_seg_interaction(current_mask, run_prediction=False)` if mask non-empty.
  Positive model prompts add to the current object; negative prompts remove. After every model prompt, copy the
  target buffer into a new version.
- Image upload: float16, shape `(1, I, J, K)`; crop to body bbox (+5 vox) and keep offset if the packed size > 60 MB.
  `nni_url=None` → a `FakeNNI` (in tests) that grows a sphere/flood-fill region around points so ops are testable offline.
- Ops (each returns an `Observation`, appends a trajectory record, may create a version):
  `open`, `sheet(axis, start, stop, n)`, `view(axis, slice|"mask", crop, overlay, marks)`, `point(ids, polarity)`,
  `lasso(ids, polarity)` (≥3 same-view marks → filled polygon on active slice → `add_lasso_interaction` with
  `interaction_bbox`), `scribble(ids, polarity)` (polyline 1-voxel wide, rasterized on active slice),
  `box(id_a, id_b)` (2D box on active slice), `paint(ids, mode add|erase, slice_range)` (polygon raster edit applied
  to slices in range, no model), `refine()` (`add_initial_seg_interaction(current, run_prediction=True)`),
  `edit(op, mark=None, radius_mm=None)` with op ∈ keep_component|remove_component|fill_holes|close|open|dilate|erode,
  `undo()` (checkout parent), `checkout(version)`, `finish(confidence, reason)` (writes final_mask.nii.gz in original
  file geometry with the original affine; ends the case; further ops error).
- seg_open text must include: shape, spacing, sequence, display convention, target text, budgets, mark convention.
- Budgets enforced by CaseState (configurable): max_tool_calls=40, max_model_prompts=15. When exceeded, mutating ops
  return an error telling the agent to call `seg_finish`. The runner enforces wall time.
- `Observation` (text for the agent; compact, ≤ ~12 lines):
  `version v5 (parent v4) | volume 123.4 mL | components 2 (largest 98%) | extent axial 101–163, coronal 80–131, sagittal 55–120 | change +12.3 mL / −0.4 mL | prompts used 4/15, tool calls 9/40`
  plus an image (`render_outline_change`) for mutating ops; views return their own images.
- Logging (`results/<run_id>/<case>/`): `trajectory.jsonl` (one JSON per op: ts, op, args, resolved voxel coords,
  timings render/nni/total, parent/child version, observation text, render filenames, error), `renders/NNN_<op>.png`,
  `versions.npz` (written at finish and every 5 versions), `final_mask.nii.gz`, `case_meta.json`.

## 5. HTTP API (`segd/app.py`, FastAPI on 127.0.0.1:8765)
All endpoints JSON. Responses: `{"ok": true, "text": "...", "images": ["<base64 png>", ...], "done": false}`
or `{"ok": false, "error": "...", "text": "<same as error>"}` (HTTP 200 for agent-facing errors so the tool can
show them; 4xx/5xx only for programmer errors).
- `POST /cases` body `{case_id, run_id, out_dir, nni_url|null, budgets?, dataset?, segmenter?, segmenter_url?,
  v3?}` (the optional keys are the feature flags of §7; absent = harness-v2.0.1 behaviour) → opens state (loads image via
  `segd.data.load_image`, computes/loads marks, uploads to nnInteractive). Returns `{ok, handle}`. handle = case_id.
- `POST /cases/{handle}/op` body `{op: "open"|"sheet"|"view"|"point"|..., args: {...}}` → agent-facing response.
- `GET /cases/{handle}/status` → counters, current version, done flag.
- `DELETE /cases/{handle}` → flush logs, close nnInteractive session.
- `GET /healthz` → `{ok:true}`.
Concurrency: multiple cases at once (one CaseState each); ops on one case are serialized with a lock.
nnInteractive calls are blocking → run ops in a threadpool (`run_in_threadpool`) so cases do not block each other.

## 6. Tool ↔ op mapping (extension)
`seg_open`→open, `seg_sheet`→sheet, `seg_view`→view, `seg_point`→point, `seg_lasso`→lasso, `seg_scribble`→scribble,
`seg_box`→box, `seg_paint`→paint, `seg_refine`→refine, `seg_edit`→edit, `seg_undo`→undo, `seg_checkout`→checkout,
`seg_finish`→finish. The extension knows its case handle from env `SEGD_CASE` (pi CLI) or from the runner.
Arg names in tools = arg names in ops (snake_case). Mark IDs: integers for global marks, strings "E<n>" for edges;
the tool schema accepts `ids: array of (integer|string)`.

## 7. Feature flags (one main line; every experiment is a flag combination)

The integrated segd carries three optional feature sets that were developed on separate branches
(`phase5-v3`, `phase6-sammed3d`). They are **orthogonal case-open flags**; with none of them set the behaviour is
exactly the frozen Phase 3 harness (tag `harness-v2.0.1`): same tools, prompts, observations, PNG bytes,
trajectory fields. The runner sends a flag only when it is not at its default (`dataset`, `v3`), or records it in
`meta.json`/`batch_summary.json` (`segmenter`, `segmenter_url`, `dataset`, `v3`, `preset`).

| Flag (case-open key) | Runner flag | Values (default first) | Selects | Details |
|---|---|---|---|---|
| `v3` | `--v3-pointer`, `--v3-pointer-url`, `--v3-medgemma`, `--v3-medgemma-url`, `--v3-checkpoints` | `null` (off) / `{pointer: gui_actor\|molmo2, medgemma: q4\|bf16, checkpoints}` | `seg_locate` pointer tool, MedGemma second-reader checkpoints | "Deviations/v3" below; `segd/v3.py` |
| `dataset` | `--dataset` | `mstt` / `beamster` | manifest (`eval/v2/mstt_cases.csv` / `eval/v4/beamster_cases.csv`), multi-lesion prompts/observations/review sheet, dense marks profile, budgets 50/20/12 min | "Deviations/Phase 4" below; data root `$SEGD_DATA_ROOT` |
| `segmenter` (+ `segmenter_url`) | `--segmenter`, `--segmenter-url` | `nninteractive` / `sammed3d` | the interactive segmenter behind `CaseState.nni` (identical call surface) | "Phase 6" below; service `harness/local_models/sammed3d` on `:8095` |

Environment: `NNI_URL` (live nnInteractive; unset/empty = `FakeNNI`), `SEGD_DATA_ROOT` (data tree, default
`<repo>/data`), `SAMED3D_SRC` (sammed3d project `src/`, default the sibling `harness/local_models/sammed3d/src`).
Invalid values (unknown segmenter/dataset/v3 key) → HTTP 400 (runner misconfiguration).

Port map of the local services: MedGemma q4 `:8091` / bf16 `:8094`, pointers GUI-Actor `:8092` / Molmo2 `:8093`
(the only `v3.pointer` values), Study-A-only MolmoPoint `:8095` / Qwen3-VL `:8096`, SAM-Med3D segmenter `:8095`.
**`pointd_molmopoint` and `sammed3d` both default to `:8095`** (developed on separate branches): never run them at
the same time with defaults — start one with `--port` (and pass `--segmenter-url` / the study's `--molmopoint-url`).

**Combinations.** Every flag is independent except:
- `v3` (pointer or MedGemma set) + `dataset="beamster"` → **refused** (HTTP 400 / `ValueError` in `CaseState`; the
  runner refuses it too): the multi-lesion prompts and the seg_locate/second-reader paragraph were never combined
  and no experiment ran it. An all-off `v3` object (`{checkpoints: []}`) is v2 behaviour and composes with anything.
- `v3` + `segmenter="sammed3d"` and `dataset="beamster"` + `segmenter="sammed3d"` are mechanically supported and
  smoke-tested (`tests/test_flag_matrix.py`) but were never run as experiments.

**Equivalence tests** (`tests/test_identity.py`; driver `tests/_identity_driver.py` runs one scripted op sequence in
a subprocess per segd tree and compares observation text, PNG sha256, counters and the normalized trajectory):

| Flags | Reference |
|---|---|
| defaults | pinned fixture `tests/fixtures/segd_baseline` = `git archive harness-v2.0.1 harness/server/src/segd` (sha256 manifest; a second test checks it still equals the tag) |
| `segmenter=sammed3d` (fake service) | branch `phase6-sammed3d` @`d8e82cf`, `git archive` at test time |
| `v3` pointer+MedGemma (fake services) | branch `phase5-v3` @`7058be6`, `git archive` at test time |
| `dataset=beamster` (two-lesion phantom) | branch `phase5-v3` @`7058be6` |

Reproducing a logged experiment is `--preset <name>` (`experiments/presets/`, `research/REPRODUCE.md`); a preset only
sets runner flags, so the flags above are the complete experiment-defining surface of the harness.

## Deviations
- **2026-10-18 (fix round, reviewer findings 1–14).** Budget semantics (§4): EVERY `seg_*`
  call counts as one tool call, including failed calls, views and sheets; when the tool-call
  budget is exhausted every op except `finish` is rejected with "call seg_finish now"; the
  model-prompt budget counts only prompts actually sent and blocks only prompt ops
  ("model prompt budget used; edit with seg_paint/seg_edit or call seg_finish"). Enforced in
  `ops.run_op` (not in `CaseState`); rejected calls are logged to the trajectory. Counters
  appear in every observation and error text as `tool calls X/N, prompts Y/M` (supersedes the
  `prompts used …, tool calls …` order in the §4 example).
- **Error transport (§5).** Unexpected exceptions on agent ops also return HTTP 200
  `ok:false` with the generic text "internal harness error; the mask is unchanged; try again"
  + counters; 5xx is reserved for programmer errors on non-op endpoints. The trajectory
  record then carries the full traceback with `error_kind: "internal"` so scoring can tag
  `harness`; agent-facing OpErrors are logged with `error_kind: "op"`, budget rejections
  with `error_kind: "budget"`.
- **Model-prompt failures.** Any exception in a model prompt leaves the mask unchanged,
  marks the nni state dirty and raises `OpError("segmentation model error; the mask is
  unchanged, try again")`; `SessionExpiredError`/connection errors reconnect once (re-upload
  the image, new target buffer, current mask re-sent by the sync rule) and retry.
- **Interaction boxes (§4).** Lasso/scribble `interaction_bbox` has size 1 on the fixed axis;
  the in-plane margin is kept only where it stays inside the uploaded crop, and the box is
  clipped to the crop BEFORE the mask is cut (mask shape always equals the box).
- **Undo/checkout text (§4).** Reads `version vN (restored; was vM)`; the change is diffed
  against the previously current version (not the version-graph parent).
- **Lasso/paint vertex order (§4).** The agent's mark order is used as-is (no angle sorting;
  concave outlines matter) — the tool text must say to list IDs in order around the outline.
- **final_mask.nii.gz geometry (§4).** Written in the ORIGINAL source-file geometry: when the
  source NIfTI needed canonicalisation at load (`segd.data.load_case_geometry`), the mask is
  mapped back through the inverse nibabel orientation and saved with the source affine.
  RAS sources are unchanged. `case_meta.json` gains `source_orientation`.
- **`point` bounds.** A point resolved outside the uploaded crop errors with "that mark lies
  outside the scanned body region on this slice" (the server would 500).
- **2026-10-18 (marks/render lane B).** `MARKS_VERSION` bumped 6→7: v6 caches stored mark
  kinds with a fixed `'<U5'` dtype, truncating `'region'` to `'regio'`; v7 encodes kinds with a
  data-fitting dtype, `MarkSet.load` maps the legacy `'regio'` spelling back to `'region'`, and
  existing mark caches are recomputed. Also fixed: region (supervoxel) mark positions were
  clamped against the *downsampled* grid shape instead of the volume, collapsing most region
  marks into the volume's low-index corner (positions, not interface).
- **`MarkSet.visible(level='fine')` ⊇ `visible(level='coarse')` is now guaranteed** on the same
  slice/box: the coarse pass (10 mm thinning + cap) is computed identically for both levels and
  kept as the core of the fine set; fine candidates (6 mm thinning, 3 mm clearance from the kept
  coarse marks) fill the remaining label budget. §2 says fine = coarse + fine; this makes it exact.
- **`render_view` additive optional kwargs `title_suffix` (default `"| marks"`) and
  `single_panel` (default `False`)** (§3): defaults keep the previous behaviour byte-identical;
  `title_suffix` changes the marked panel's chip text, `single_panel=True` returns one panel
  (image + overlay + marks, marks recorded with panel index 0) instead of the clean+marked pair —
  for example figures and callers that do not need the comparison panel.
- **2026-10 (fix round 2, independent review).**
  - *Mark cache* (§2): `CaseState` accepts a cache only if `marks.cache_matches` holds
    (`MARKS_VERSION` + full params fingerprint, now including `REGION_STRENGTH_MM`,
    `REGION_SCALE_CLAMP_MM`, `BODY_GRID_MAX_DIM`, + volume shape), else recomputes; `MarkSet.save` is
    atomic (tmp + `os.replace`). Mark computation/parameters unchanged (ids/ijk/kinds/levels/scales of
    all 30 design caches are identical to before).
  - *Edge marks* (§2/§3): computed in the BODY on the slice (`render_view(..., body=)`; without it a
    per-slice Otsu keeping ALL components), never restricted to the agent's mask. Candidates within 15 mm
    of the current mask boundary (both sides) rank first (`edge_marks_2d(prefer_mask2d=)`); max 40.
  - *Labels*: all glyph footprints (edge and global) are reserved before any label placement; left-hand
    label placements are right-aligned.
  - *`seg_view` crop='marks'* (§6): requires `ids` (global or `E<n>`) shown in the CURRENT active view; the
    new view crops to those marks' in-plane bbox + 25 mm (min 60x60 mm) on the requested axis/slice and
    `marks='coarse'` is upgraded to `'fine'` there. Without ids: agent-facing error. 'full'/'mask' unchanged.
    A full view with `marks='fine'` that adds nothing says "fine marks appear in crops".
  - *Sheets* crop every cell to the whole-volume body bbox on that axis (`render_sheet(..., body=)`).
  - *Mutation feedback image* (§3/§4): every mask-changing op returns `render_change_3plane` — axial,
    coronal, sagittal through the new mask's centre of mass (empty mask: changed region, else volume
    centre), cropped to the new mask's 3D bbox + max(30 mm, 50%) per side, one shared scale, each panel
    <=768 px, total <=1568 px; new mask green outline, removed region red outline + tint. The observation
    appends `image: axial k, coronal j, sagittal i through mask centre`. `render_outline_change` is kept.
  - *Raw pointing*: rulers/title/scale bar are in margins outside the image area (`RAW_MARGIN_*_PX`);
    `ViewGeom` still describes the image area only (ruler numbers = image-area pixels, 0 = its top-left),
    so `panel_to_voxel`/`voxel_to_panel` are unchanged. The PNG is larger than `panel_w x panel_h`.
  - *Robustness*: a prompt with `nni is None` (earlier failed reconnect) reconnects first; mutating ops are
    all-or-nothing (`CaseState.rollback_on_error`) so a render failure never leaves a committed version;
    unknown ops / non-object args count as a tool call and are logged; budget display is capped at max.
- **2026-10-19 (candidate corrections, E14 integration).** New ops `candidates` (`seg_candidates`, no args)
  and `accept` (`seg_accept(ids)`) (§4/§6). `candidates` runs the GT-free 'intensity' generator
  (`segd.candidates.propose`, method/max_n from `config.CANDIDATE_METHOD`/`CANDIDATE_MAX_N`; the case's body
  mask is reused via the `propose` `cache`) on the CURRENT mask and caches the result per version id in
  `CaseState.candidate_cache`; the text lists C1..Cn (volume, mask contact, direction) and returns ONE
  `render.render_candidates(image, mask, candidates, spacing)` image (§3 addition): rows = candidates (max 5),
  columns axial/coronal/sagittal through the candidate's centre of mass, every cell cropped to
  bbox(mask ∪ candidate) + 20 mm; current mask = green outline, candidate = magenta outline + ≤20% tint with
  its id chip; orientation letters and slice chips; image ≤1568 px per side and ≤~1.2 MP. Proposing changes
  nothing and counts as a tool call (no candidates -> text only). `accept(ids)` unions the chosen
  add-candidates into ONE new version — a non-model edit, so `nni_dirty=True` per the sync rule — and returns
  the usual observation + 3-plane change image. Candidates belong to the version they were computed for: any
  mask change invalidates them (`CaseState.candidates_shown`, also cleared on rollback); accepting a stale id
  errors "candidates are out of date; call seg_candidates again", an unknown id "unknown candidate id(s)".
  Required review (§4 `finish`): the first review of a version ALSO computes candidates and returns the
  candidate sheet as a second image after the 3×5 review grid, plus the list and the judge text in the
  observation; the review_code mechanism is unchanged (a confirmation still needs the code printed in the
  latest review). Cap: after `config.CANDIDATE_REVIEW_MAX_ACCEPT_ROUNDS` (3) successful `seg_accept` calls in
  a case, reviews no longer show candidates and say so; `seg_candidates` keeps working. Empty mask -> no
  candidate round. Logging (§4): trajectory records gain the op-specific fields `candidates` (version,
  method, items with id/kind/volume_ml/how) and `accepted` (ids, from/to version, volume_ml, how),
  `timings.candidates` (proposal seconds); the sheet is saved to `renders/` like every agent image.
  Extension (§6): `seg_candidates`/`seg_accept` registered and added to `SEG_TOOL_NAMES`; `seg_finish`'s
  description mentions the candidate round.
- **2026-10-20 (candidate fix round).** Authorization no longer survives version round trips: every change
  of the current version (`CaseState.commit_mask`/`checkout`/`undo` and `rollback_on_error`, one
  `_clear_authorization` helper) revokes `candidates_shown` and the review authorization
  (`reviewed_version`/`review_code`); the per-version computation cache (`candidate_cache`) is kept, so
  re-calling `seg_candidates` on a cached version is cheap. Candidate visibility is set only after the
  candidate sheet rendered (a failed `render_candidates` authorizes nothing, in `candidates` and in the
  `finish` review alike). Mutating ops (`ops.TRANSACTIONAL_OPS`) are transactional through render saving
  and trajectory logging: a failure after the op body (save_render, record, observation) restores
  versions/current version, counters (`accept_rounds`) and authorization, logs an `error_kind:"internal"`
  record and lets app.py send the generic error. Review sheets show five slices per plane (three inside
  the mask extent + one just past each end); candidate id chips are placed collision-aware against the
  frame's occupied rectangles (no overlap with orientation letters, bottom-row fallback on narrow
  panels). `candidates._components` materializes only the largest `limit` qualifying components
  (2 x `CANDIDATE_MAX_N` at the generator call sites) instead of one crop-sized array per component.

## Deviations/v3 — harness v3 components (2026-10-20)

Optional local-model components (research/harness_v3_methodology.md §3 Study C), all OFF by
default: a case without the `v3` flags runs byte-identical v2 behaviour (same tools,
prompts, observations, trajectory fields). Enabled per case by the runner through the
case-open body; recorded in `case_meta.json` (both writes) and runner meta/batch summary
as the `v3` object. Implementation: `segd/v3.py`; constants in `config.py` (`V3_*`).

- **Case flags (§5 `POST /cases`).** Optional body key `v3`:
  `{pointer: null|"gui_actor"|"molmo2", pointer_url?, medgemma: null|"q4"|"bf16",
  medgemma_url?, checkpoints: ["localize","review","candidates"]}`. Unknown keys/values are
  a programmer error -> HTTP 400 (programmer-endpoint rule). URLs default to the standard
  loopback service ports (8092 gui_actor, 8093 molmo2, 8091 q4, 8094 bf16). `checkpoints`
  defaults to all three when the key is absent; an explicit subset/dedup is canonicalized
  to that order.
- **Tool `seg_locate` → op `locate` (§6).** Declared by the extension only when
  `v3.pointer` is set; the op itself also rejects (`seg_locate is disabled for this case`)
  when no pointer is configured, so the server honours the flag independently of the tool
  declaration. Args: `{query?: string}` (default: the case target text). Uses the CURRENT
  VIEW image exactly as rendered for the agent — `ActiveView` now stores the rendered PIL
  image (additive field `image`), and `seg_locate` crops the clean panel via
  `RenderInfo.panel_rects[0]` (below). POST `<pointer_url>/point {image_png_b64, query, k=5,
  verify: false}` → the ranked list of nearest DRAWN marks (global integer ids and per-
  render `E<n>` edges as one target set; dedup keeps the best-ranked point per mark; ties
  -> earlier point), each with the pointer score and snap distance in px; the image shows
  the proposed points numbered P1..P5 on both panels. Counts as a tool call, never a model
  prompt (not in `MODEL_PROMPT_OPS`); rejected by the tool-call budget like every call.
  Service down/timeout -> OpError (counts as a call; logged with error_kind "op" plus the
  v3 diagnostic record). The view shows no marks -> agent-facing error. No raw coordinates
  in the text (§0): only mark ids, scores, px snap distances.
- **RenderInfo.panel_rects (§3, additive).** Optional last dataclass field
  `panel_rects: list[(x0,y0,x1,y1)] | None = None`: the paste rectangle of every recorded
  panel in the combined image (per panel for overview/sheet/view; per candidate ROW for
  render_candidates). Default None keeps all previous call sites byte-identical; only
  v3 consumers read it.
- **Medical second reader (MedGemma 1.5 4B; §4 checkpoints).** Enabled by `v3.medgemma`;
  injected into tool observations at fixed checkpoints, once per checkpoint per version
  (fired-set `CaseState.v3_fired`; run_op snapshots and restores it for EVERY op — not
  only the transactional ones, since the localize hook also fires after
  open/sheet/view — so a failed op's retry still delivers the note):
  (a) `localize` — the FIRST seg_sheet observation (or, when the agent never calls a
  sheet, the first successful post-open op, using the seg_open overview rendered then):
  lesion box `[y0,x0,y1,x1]` in [0,1] on the shown image + confidence (temperature 0);
  the box centre is mapped to the nearest global marks (mm distance, top 3 echoed);
  (b) `review` — every first-stage `seg_finish` review (which is once per version by
  construction): strict JSON `{complete: yes|no|uncertain, problems: [{panel, issue:
  missed_lesion|leak|ok}], confidence}`; non-ok problems echoed;
  (c) `candidates` — one call per candidate of the finish review, with the candidate's
  sheet ROW (cropped via panel_rects): `{candidate_id, verdict: lesion|normal|uncertain,
  confidence}`; per-candidate verdicts appended. Every reply is bound to the REQUESTED
  candidate id (the row image it saw): a reply naming another candidate, without one of
  the contract's verdicts, or without a usable confidence is UNUSABLE — that row shows
  as unavailable and the reason lands in the trajectory record with `parsed: null`.
  Prompt templates: `harness/prompts/medgemma/{localize,review,candidates}.md` (initial
  versions, tuned on the design split only). Client: the Phase 5
  `harness/local_models/medgemma/medgemma_client.py` (loaded by path, single source);
  deterministic temperature 0. The checkpoint calls of one op share a TOTAL per-op
  deadline `config.V3_MEDGEMMA_GROUP_TIMEOUT_S` (150 s, kept below the extension HTTP
  deadline SEGD_OP_TIMEOUT_MS = 180 s: a finish op runs the review plus up to five
  candidate calls sequentially), each request clamped to min(V3_MEDGEMMA_TIMEOUT_S,
  remaining); when the group budget is spent a checkpoint is skipped WITHOUT sending and
  gets its unavailable note. ANY failure (transport, timeout, unparseable or unusable
  reply) appends the short note "Medical second reader
  (MedGemma): unavailable this checkpoint (service or parse error); continuing without
  it." (or the matching "cannot locate / no usable box" text for off-contract replies)
  and NEVER fails the op; each finish checkpoint is guarded separately, so a failed
  checkpoint cannot blank another's advice. The appended note is plain text; review
  codes, candidate authorization and the sync rule are untouched.
- **Trajectory (§4).** Every v3 request/response is appended to the current op's
  trajectory record under the extra field `v3` (list of records): locate calls
  (`{call: "locate", service, url, query, k, image_sha256, raw, latency_ms, ok, error,
  parsed, snapped: [{point, mark, score, dist_px}]}`) and second-reader checkpoints
  (`{call: "checkpoint", checkpoint, service, url, prompt_sha256, image_sha256,
  candidate_id (requested), returned_candidate_id, raw, latency_ms, ok, error, parsed,
  model}`). Locate records also carry the structured pointer response (returned points,
  model/service version) in `parsed`. These diagnostics are logged even when the op
  itself failed (error records carry only `v3`, never the op's success fields).
- **Scorer (offline).** `eval/v2/score.py` gains the columns `n_locate_calls` and
  `medgemma_checkpoints_seen` (empty on v2 runs); `eval/v2/v3_adherence.py` computes the
  RQ3 follow/override analysis (pointer proposals vs the first positive click — follow
  undefined without preceding advice, correctness evaluated on the followed points;
  candidate accepts matched version- and order-aware, unexposed distinct from overrides;
  the accept-operation-level Dice delta once per operation, labelled as such) from
  trajectories, versions.npz and offline GT.

## Deviations/Phase 4 — BEAMSTER multi-lesion changes (2026-10-21)

Phase 4 narrowed to BEAMSTER only (brain metastases, CE-T1 MRI, 140 cases: 30 design /
110 test, multiple small lesions per case, 1 mm iso conversions under
`data/phase4/beamster/nifti_iso/`). All changes are gated on the case's `dataset`;
`dataset="mstt"` (the default everywhere) keeps byte-identical v2/v3 behaviour.

1. **Dataset selection (decision 1).** Runner `--dataset mstt|beamster` (default `mstt` =
   unchanged behaviour); the case-open body (`POST /cases`) carries `dataset` (omitted on
   mstt: identical v2 request), validated like other body fields (unknown manifest → 400,
   unknown case → 404). `segd.data` gains the manifest reader `read_cases_phase4(dataset,
   split)` for `eval/v4/<dataset>_cases.csv` + `get_case(case, dataset="mstt")`;
   `CaseRow` gains `modality` and the GT-derived `n_lesions` (OFFLINE use only — scorer,
   diagnostics; CaseState never receives it, so no GT lesion count can reach any
   agent-visible path) and carries the manifest `target_text`. Path portability: env
   `SEGD_DATA_ROOT` (default `<project>/data`) — manifest absolute paths recorded under
   `/home/ubuntu/agentic-segmentation/data` are remapped onto `$SEGD_DATA_ROOT`
   (`remap_data_path`); MSTT `DATA_DIR` resolves under the same root (import time). The
   committed beamster manifest had `image_path`/`mask_paths` swapped with the `*_src`
   provenance columns (native paths in the harness columns, 1 mm iso paths in `*_src` —
   the reverse of the column semantics pinned by eval/v4/phase4_common.py and of
   `repoint_manifests_to_iso.py`); fixed in place (backup kept), all 140 rows verified to
   point at `nifti_iso/` with `n_lesions` matching the iso-mask components (140/140).
2. **Task = ALL lesions, ONE multi-component nnInteractive object (option a; decision
   2).** No expected-lesion-count disclosure anywhere (that would be a scoring leak).
   Prompts: `harness/prompts/system_multi.md` / `task_multi.md` are copies of
   `system.md`/`task.md` with exactly the minimal edits: 'one lesion' → 'all lesions of
   the target (there may be one or many, some very small)'; the MSK-specific 'What counts
   as the lesion' rule (rim/fascia/nerve) replaced by the neutral 'Include every lesion of
   the target type anywhere in the volume; check all slices, small lesions are easy to
   miss'; everything else identical (`task_multi.md` is currently a faithful copy — the
   'one lesion' phrase does not occur in `task.md`). The runner selects them as the
   default task/system files for non-mstt datasets (explicit `--system-file`/`--task-file`
   wins; no combined v3+multi variant exists — that combination errors and requires an
   explicit `--system-file`).
3. **Multi-component observation text (decision 3).** On non-mstt datasets every
   components line reports the component count and the volumes of up to the 10 largest
   components derived from the CURRENT mask only
   (`components N (mL: v1, v2, …; M smaller)`) — `mask_stats.component_ml` +
   `observe.components_text_multi`; MSTT keeps the legacy 'largest X%' line.
   `seg_edit keep_component`/`remove_component` were VERIFIED mark-based (they keep/remove
   the component containing the mark's nearest mask voxel, `ops.op_edit`) — they never
   reduce the mask to the largest component, so no fix was needed. Finish review: the
   union-extent grid's fraction columns can fall between separated lesions, so on
   non-mstt datasets a mask with ≥ 2 components additionally returns the COMPONENT SHEET
   (`render.render_review_components`): one row per component (largest first, up to 5),
   axial/coronal/sagittal through the component's centre of mass, each cropped to that
   component's bbox + max(20 mm, 30% of its extent) — chosen (simplest robust option) so
   every lesion's own boundary is reviewable while the existing union-extent grid keeps
   its past-the-union-extent meaning. The candidate sheet's image position/wording shifts
   accordingly ('third image'); review codes/authorisation are unchanged.
4. **Budgets (decision 4).** Runner per-dataset defaults (`DATASET_DEFAULTS`): beamster =
   50 tool calls, 20 model prompts, 12 min wall — applied only for flags the user did not
   pass explicitly; static per dataset, never derived from GT. segd's
   `config.DEFAULT_BUDGETS` (mstt 40/15) unchanged.
5. **Marks density option (decision 5).** `segd.marks.DENSITY_PROFILES` + `config.
   MARKS_PROFILE = {"beamster": "dense"}`: the dense profile (fine 3D thinning 5→3 mm,
   fine/coarse in-plane thinning 6/10→3/8 mm, per-panel cap 50→200) is fingerprinted into
   `MarkSet.params` (`cache_matches(…, profile=)`) so caches are profile-aware; MSTT uses
   the "default" profile (byte-identical marks). Measured on the 30 BEAMSTER design cases
   (72 GT lesions; hit = a DRAWN mark inside the lesion on its largest axial slice;
   eval/v4/mark_coverage_beamster.py, before/after files in eval/v4/marks_beamster/):
   <0.5 mL full-view fine hit 0.10→0.24, 0.5–2 mL 0.41→0.71, >2 mL 0.85→0.96; crop
   (zoom-view) fine hit <0.5 mL 0.55→0.62, ≥0.5 mL 1.00. Residual: 10/72 lesions have no
   mark even in crops (blob-candidate generation, unchanged by design here) and full-view
   panel-space strength ordering still biases against tiny lesions — discovery on full
   views stays imperfect by construction; sheets + `crop='marks'` zooms + `seg_candidates`
   are the compensating mechanisms.
6. **Multi-lesion scorer (decision 6).** `eval/v4/score_multi.py` (reuses eval/v2/score.py
   primitives): global Dice/HD95/NSD on the union; lesion-wise detection — GT lesion =
   connected component (6-conn, same convention as the manifest builder and
   `observe.mask_stats`), DETECTED when some predicted component overlaps ≥ 10% of its
   voxels, FP = predicted component with zero GT overlap; lesion sensitivity, FP count,
   lesion precision/F1, mean per-lesion Dice over DETECTED lesions (matched prediction =
   union of the ≥10%-overlapping predicted components, so a component bridging two
   lesions counts against both); breakdown by lesion size (<0.5 / 0.5–2 / >2 mL); the tag
   'none_detected' (no GT lesion detected) replaces the MSTT 'localization' tag (same
   precedence otherwise); per-case `n_lesions` (GT components, computed — cross-checked
   against the manifest column) in `scores_multi.csv` + per-lesion `lesions_multi.csv`.
7. **Tests (decision 7).** MSTT byte-identity: `test_identity.py` (formerly `test_v3_off_identity.py`)
   still passes — all Phase 4 paths are gated on `dataset != "mstt"` and every new
   CaseState/render parameter defaults to the previous behaviour. New:
   `harness/server/tests/test_phase4_beamster.py` (manifest reader + remap, MSTT defaults,
   multi observation text, dense-profile fingerprinting, component sheet, and a FakeNNI
   smoke path on the REAL design case Dataset_A_001 read-only from
   `/home/ubuntu/agentic-segmentation/data/phase4/beamster/nifti_iso` via
   `SEGD_DATA_ROOT`) and `eval/v4/test_score_multi.py` (detection rule, 10% boundary,
   FP/bucket/F1 math, tags, and a synthetic-manifest integration case with dataset
   inference from `case_meta.json`).

## Deviations/Phase 6 — segmenter swap (SAM-Med3D-turbo instead of nnInteractive)

Goal: harness v2 with SAM-Med3D-turbo as the interactive segmenter, apples-to-apples — the agent
sees identical tools, prompts, renders, marks, budgets and observation formats; ONLY the
segmenter behind `CaseState.nni` changes. Full write-up: `research/phase6_sammed3d.md`.

- **Selection.** `POST /cases` accepts `segmenter` (`"nninteractive"` default | `"sammed3d"`) and
  `segmenter_url` (sammed3d default `config.SAMED3D_URL` = `http://127.0.0.1:8095`; for nninteractive
  the existing `nni_url` is used, unchanged). Anything else → HTTP 400. `CaseState(segmenter=,
  segmenter_url=)`. Both appear in `case_meta.json` (`segmenter`, `segmenter_url`; `nni` is
  `"sammed3d"` for the swap, `"remote"`/`"fake"` otherwise) and in `GET /cases/{h}/status`
  (`segmenter`). The runner exposes `--segmenter` / `--segmenter-url`, recorded in each case's
  `meta.json` and in `batch_summary.json`. With the default segmenter nothing changes: observations
  and PNG bytes are identical to the pre-Phase-6 tree (`tests/test_identity.py`).
- **Client.** `segd.sammed3d` bridges to the local-model project `harness/local_models/sammed3d`
  (own uv env with torch; segd needs no torch). `sammed3d.client.SamMed3DSession` is a
  call-signature drop-in for `nnInteractiveRemoteInferenceSession` (identical method names,
  parameters and defaults for everything CaseState calls: `set_image`, `set_target_buffer`,
  `set_do_autozoom`, `reset_interactions`, `add_{point,bbox,lasso,scribble,initial_seg}_interaction`,
  `_predict`, `close`, ...; asserted by `tests/test_client_parity.py`). The sync rule (§4), the
  upload crop, the model-prompt budget and the reconnect-once rule apply unchanged (a sammed3d
  `SessionExpiredError` is a transport failure like nnInteractive's). `ops.py` is untouched.
- **Service** (`sammed3d-service`, FastAPI, 127.0.0.1:8095; `--fake` = numpy stand-in for offline
  tests). Every prediction returns the FULL session-space mask plus the prompt-mapping log. Per
  call it recomputes the 3D mask from ALL accumulated points and the previous prediction's low-res
  mask (the official iterative-clicking semantics); prompts sent with `run_prediction=False` only
  accumulate. `reset_interactions` clears points and the previous mask; `add_initial_seg` makes the
  given mask the previous-mask prompt AND, like nnInteractive's `add_initial_seg_interaction` ("WARNING THIS
  WILL RESET INTERACTIONS"), drops all earlier points and the first-positive anchor; a predicting initial seg
  (`seg_refine`) adds one EDT-max interior positive point as the window seed (the state-only sync call adds none).
- **Preprocessing / patches** (official `utils/infer_utils.py`, minus ground truth): resample to 1.5 mm
  (linear), per-patch `ZNormalization` over x>0, 128³ patches, logits upsampled trilinear, sigmoid>0.5,
  nearest-neighbour map back to the session grid. GT is NEVER used for patch placement: the first patch
  is centred on the first positive prompt; later patches cover bbox(positive points ∪ current mask);
  a box exceeding 128 model voxels (192 mm) on an axis is tiled (~25 % overlap, ±16-voxel margin, cap 27
  patches, the cap/coarsening is logged), each patch gets the prompt points inside it and its slice of
  the previous mask as the 32³ low-res mask prompt (a tile is skipped only when it has no point AND no
  high-res previous-mask foreground; the pooled prompt's sign is irrelevant); logits are averaged in overlaps.
  Mapping back to the session grid follows ITK/TorchIO nearest-neighbour rules (half-voxel support, round half
  up; exact-half ties are floating-point dependent in ITK: ≤1 model voxel).
- **Mapping** (logged per call in the response and stored in the trajectory record as
  `segmenter_interactions`: one entry per prompt call since the previous prediction with
  `mapped_points` = `[[i, j, k, label], ...]` in session voxels, then a summary entry with
  `patches_used`, `windows_model_vox`, `n_points_total`, `patch_note`): point → native point
  (label 1/0); bbox → centre + 4 quarter points of the box slice (label per include flag; the singleton fixed axis of a 2D box is the slice voxel index itself; a 3D box →
  centre + 8); lasso → ≤5 deterministic farthest-point samples inside the filled contour (first = EDT
  max); scribble → ≤5 deterministic evenly spaced samples along the stroke; initial_seg/refine →
  the mask as low-res mask prompt (+ EDT-max interior point for the predicting call). Deterministic, no
  test-time tuning.
- **Documented deviations from the official pipeline**: patch placement from prompts instead of the GT
  bbox centre (required: no GT); windows near the volume edge are zero-padded, not shifted (as the
  official CropOrPad); an all-zero / constant patch is normalised to 0 instead of raising; non-identical
  windows derive their mask prompt from the previous probability map (block-mean to 32³, logit) while an
  identical window reuses the raw low-res logits (the official carry-over).
- Not swapped: the candidate generator's optional `probe` method (unused; `config.CANDIDATE_METHOD`
  is `intensity`) still talks to nnInteractive; the system prompt text still says "nnInteractive"
  (kept verbatim for apples-to-apples).
- **Service sessions (Phase 6 review fixes).** A session idle for `--session-ttl` seconds (default 3600) is
  reaped — never while a request is in flight — whenever a session is created or `/healthz` is read, so
  capacity recovers after a segd crash without restarting the service; an expired session answers 404 →
  `SessionExpiredError` → segd's reconnect-once rule. `/healthz` reports `sessions: {count, max, busy,
  oldest_age_s, max_idle_s, ttl_s, reaped_total}`.
- **Identity test baseline.** `tests/test_identity.py` (§7): default flags vs the pinned `harness-v2.0.1` fixture
  (or `SEGD_BASELINE_SRC`), `segmenter=sammed3d` vs `phase6-sammed3d`; skips explicitly (never self-compares) when
  the reference is unavailable.
