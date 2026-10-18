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
- `POST /cases` body `{case_id, run_id, out_dir, nni_url|null, budgets?}` → opens state (loads image via
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
