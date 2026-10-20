"""The segd op set (CONTRACTS §4/§6): open/sheet/view/point/lasso/scribble/box/paint/
refine/edit/undo/checkout/candidates/accept/finish. Each op returns OpOutput(text, images); run_op()
wraps budget checks, trajectory logging, render saving and timing.

Mark resolution always goes through the active view (CaseState.resolve_ids); resolved
voxel coordinates are written to trajectory.jsonl only, never to agent-facing text.

nnInteractive conventions (verified against the nnInteractive 2.6.0 client/server source,
inference_session._add_image_interaction):
- add_point_interaction(coordinates, include_interaction, run_prediction): coordinates is
  (i, j, k) in the coordinate space of the image passed to set_image (crop space when we
  upload a crop).
- add_lasso_interaction(mask, include_interaction, run_prediction, interaction_bbox) and
  add_scribble_interaction(...): mask shape must equal (ub-lb) per axis of
  interaction_bbox; interaction_bbox is HALF-OPEN [[x0,x1],[y0,y1],[z0,z1]] (x0 inclusive,
  x1 exclusive — the session checks image.shape == (ub-lb) and offsets by the lower
  bounds). The bbox lives in the session's image space.
- add_bbox_interaction(bbox_coords, include_interaction, run_prediction): same half-open
  bbox format, no mask image; a 2D box has size 1 on the fixed axis.
- add_initial_seg_interaction(initial_seg, run_prediction): full-volume mask in session
  space; it resets the server interaction state itself.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from segd import config
from segd.case_state import (
    CaseState, OpError, parse_version, project_to_slice, vid_label, voxel_to_inplane,
)
from segd.observe import (
    _fmt_extent, budget_line, components_text_for, mask_stats, is_multi_lesion,
    observation_after_mutation, observation_after_restore,
    observation_for_view,
)

POLARITY_TRUE = {"add", "positive", "include", "pos", "true"}
POLARITY_FALSE = {"erase", "remove", "negative", "exclude", "neg", "false"}


def _candidates_module():
    """Lazy contract import: segd.candidates (candidate corrections; E14 benchmark)."""
    try:
        from segd import candidates
    except ImportError as e:
        raise OpError("segd.candidates is not available (candidates module missing)") from e
    return candidates


def _with_budget(cs: CaseState, msg: str) -> str:
    """Agent-facing error text with the budget counters appended: the agent must always
    see the remaining budget, also in error responses (decision on finding #3)."""
    return f"{msg} | {budget_line(cs)}"


@dataclass
class OpOutput:
    text: str
    images: list = field(default_factory=list)   # PIL images
    done: bool = False


# ------------------------------------------------------------ arg validation
def _polarity(p) -> bool:
    if isinstance(p, str):
        s = p.strip().lower()
        if s in POLARITY_TRUE:
            return True
        if s in POLARITY_FALSE:
            return False
    raise OpError(f"polarity must be add|erase (got {p!r})")


def _axis(a) -> str:
    if isinstance(a, str) and a.strip().lower() in config.AXES:
        return a.strip().lower()
    raise OpError(f"axis must be axial|coronal|sagittal (got {a!r})")


def _check_ids(ids) -> list:
    if not isinstance(ids, (list, tuple)) or len(ids) == 0:
        raise OpError("ids must be a non-empty list of mark IDs")
    return list(ids)


def _slice_range(val, cs: CaseState, axis: str, default_slice: int) -> list[int]:
    """Inclusive slice range along the active axis; None -> the active slice only."""
    if val is None:
        return [default_slice]
    try:
        start, stop = val
        start, stop = int(start), int(stop)
    except (TypeError, ValueError):
        raise OpError(f"slice_range must be [start, stop] (got {val!r})")
    n = cs.shape[config.FIXED_AXIS[axis]]
    if not (0 <= start <= stop < n):
        raise OpError(f"slice_range [{start}, {stop}] out of bounds for {axis} (n={n})")
    return list(range(start, stop + 1))


def _render():
    """Lazy contract import: segd.render may not exist during bring-up."""
    try:
        from segd import render
    except ImportError as e:
        raise OpError("segd.render is not available (render module missing)") from e
    return render


def _centroid_slice(mask: np.ndarray, axis: str, shape) -> int:
    a = config.FIXED_AXIS[axis]
    idx = np.nonzero((mask > 0).any(axis=tuple(i for i in range(3) if i != a)))[0]
    if idx.size:
        return int(round(float(idx.mean())))
    return shape[a] // 2


def _resolve_slice(cs: CaseState, slice_arg, axis: str) -> int:
    if slice_arg is None or (isinstance(slice_arg, str) and slice_arg.strip() == "mask"):
        return _centroid_slice(cs.current_mask(), axis, cs.shape)
    try:
        s = int(slice_arg)
    except (TypeError, ValueError):
        raise OpError(f"slice must be an integer or 'mask' (got {slice_arg!r})")
    n = cs.shape[config.FIXED_AXIS[axis]]
    if not (0 <= s < n):
        raise OpError(f"slice {s} out of bounds for {axis} (n={n})")
    return s


CROP_MARKS_MARGIN_MM = 25.0
CROP_MARKS_MIN_MM = 60.0


def _marks_crop_box(cs: CaseState, resolved, axis: str, slice_idx: int) -> tuple:
    """In-plane box (r0, c0, r1, c1), half-open, display-oriented, around the given marks
    on the requested view: bbox of their in-plane positions + 25 mm, at least 60x60 mm,
    clamped to the slice. Global marks use their 3D position (so the box is right even when
    the requested axis differs from the view the ids came from); edge marks use their
    resolved voxel."""
    rcs = []
    for r in resolved:
        ijk = cs.markset.marks[r.id].ijk if isinstance(r.id, int) else r.ijk
        rcs.append(voxel_to_inplane(axis, project_to_slice(ijk, axis, slice_idx), cs.shape))
    ra, ca = config.ROW_AXIS[axis], config.COL_AXIS[axis]
    n_rows, n_cols = cs.shape[ra], cs.shape[ca]
    lo_hi = []
    for vals, sp, n in (([p[0] for p in rcs], cs.spacing[ra], n_rows),
                        ([p[1] for p in rcs], cs.spacing[ca], n_cols)):
        lo, hi = min(vals), max(vals) + 1
        margin = CROP_MARKS_MARGIN_MM / sp
        lo, hi = lo - margin, hi + margin
        need = CROP_MARKS_MIN_MM / sp
        if hi - lo < need:
            mid = (lo + hi) / 2.0
            lo, hi = mid - need / 2.0, mid + need / 2.0
        lo_i, hi_i = int(np.floor(lo)), int(np.ceil(hi))
        if lo_i < 0:                       # shift (not shrink) at the slice border
            hi_i, lo_i = min(n, hi_i - lo_i), 0
        if hi_i > n:
            lo_i, hi_i = max(0, lo_i - (hi_i - n)), n
        lo_hi.append((lo_i, hi_i))
    (r0, r1), (c0, c1) = lo_hi
    return (r0, c0, r1, c1)


def _clip_box_to_crop(cs: CaseState, bbox) -> tuple:
    """Clip a full-space half-open box to the uploaded crop window; OpError when the box
    does not intersect the crop at all (finding #6 wording)."""
    out = []
    for (lo, hi), off, n in zip(bbox, cs.crop_offset, cs.crop_shape):
        lo2, hi2 = max(int(lo), int(off)), min(int(hi), int(off) + int(n))
        if lo2 >= hi2:
            raise OpError("that mark lies outside the scanned body region on this slice")
        out.append((int(lo2), int(hi2)))
    return tuple(out)


def _interaction_box_for(cs: CaseState, region: np.ndarray, axis: str, slice_idx: int) -> tuple:
    """Half-open full-space box around a drawn 2D interaction (reviewer blocker #1):

    - size 1 on the fixed axis (a 2D prompt; the real server requires mask shape to match
      the box per axis),
    - a small in-plane margin kept ONLY where it stays inside the uploaded crop,
    - the box clipped to the crop BEFORE the mask is cut from it, so the mask shape
      always equals the box (the old code cut the mask from the unclipped box and clipped
      only when translating, which crashed on cropped volumes near the body edge).
    """
    idx = np.nonzero(region)
    if idx[0].size == 0:
        raise OpError("the drawn interaction covers no voxel; choose marks closer together")
    fixed = config.FIXED_AXIS[axis]
    m = config.INTERACTION_BBOX_MARGIN
    box = []
    for a, (coords, n) in enumerate(zip(idx, region.shape)):
        if a == fixed:
            box.append((int(slice_idx), int(slice_idx) + 1))
        else:
            box.append((max(int(coords.min()) - m, 0), min(int(coords.max()) + 1 + m, n)))
    return _clip_box_to_crop(cs, box)


def _crop_mask_to_bbox(mask: np.ndarray, bbox) -> np.ndarray:
    return np.ascontiguousarray(mask[tuple(slice(lo, hi) for lo, hi in bbox)])


def _session_bbox(cs: CaseState, bbox) -> list[list[int]]:
    """Full-space half-open bbox (already clipped to the uploaded crop by
    _clip_box_to_crop) -> session (crop) space."""
    return [[int(lo) - int(off), int(hi) - int(off)] for (lo, hi), off in zip(bbox, cs.crop_offset)]


# ------------------------------------------------------- 2D raster helpers
def _polygon_grid(cs: CaseState, resolved, axis: str, polyline: bool = False) -> np.ndarray:
    """Filled polygon (or 1-voxel polyline) as a bool grid in DISPLAYED in-plane (row, col)
    order of the active slice — the same convention edge marks use (CONTRACTS §1/§2)."""
    from skimage import draw as skdraw
    a1, a2 = config.ROW_AXIS[axis], config.COL_AXIS[axis]
    pts = [voxel_to_inplane(axis, r.ijk, cs.shape) for r in resolved]
    grid = np.zeros((cs.shape[a1], cs.shape[a2]), dtype=bool)
    if polyline:
        for (r0, c0), (r1, c1) in zip(pts[:-1], pts[1:]):
            rr, cc = skdraw.line(int(round(r0)), int(round(c0)), int(round(r1)), int(round(c1)))
            grid[rr, cc] = True
    else:
        rr, cc = skdraw.polygon([p[0] for p in pts], [p[1] for p in pts], shape=grid.shape)
        grid[rr, cc] = True
    return grid


def _place_grid_on_slice(mask: np.ndarray, grid: np.ndarray, axis: str, slice_idx: int,
                         mode: str) -> None:
    """Write a displayed-order grid onto one slice of a full-volume mask (in place).

    plane = moveaxis(mask, [row_axis, col_axis], [0, 1]); plane[::-1, ::-1, s][r, c] is the
    voxel at (row_axis = n-1-r, col_axis = n-1-c), i.e. exactly the displayed pixel (r, c).
    """
    a1, a2 = config.ROW_AXIS[axis], config.COL_AXIS[axis]
    plane = np.moveaxis(mask, [a1, a2], [0, 1])   # a view; writes propagate to mask
    target = plane[::-1, ::-1, int(slice_idx)]
    if mode == "add":
        target[grid] = 1
    else:
        target[grid] = 0


def _mask_with_grid(cs: CaseState, grid: np.ndarray, axis: str, slice_idx: int) -> np.ndarray:
    mask = np.zeros(cs.shape, dtype=np.uint8)
    _place_grid_on_slice(mask, grid, axis, slice_idx, "add")
    return mask


# --------------------------------------------------------------------- ops
def op_open(cs: CaseState) -> OpOutput:
    render = _render()
    card = [
        f"case {cs.case_id} | sequence {cs.sequence} | shape (i,j,k)={cs.shape} | "
        f"spacing {cs.spacing[0]:g}x{cs.spacing[1]:g}x{cs.spacing[2]:g} mm",
        f"target: {cs.target_text}",
        "display: radiological. axial: patient right LEFT, anterior TOP. coronal: patient right LEFT, "
        "superior TOP. sagittal: anterior LEFT, superior TOP. R/L cannot be verified for single limbs.",
        "pointing: set-of-marks only. Global marks are integer IDs (3D); edge marks are 'E<n>' valid "
        "only in the current view. Spatial arguments resolve against the last seg_view; call seg_view "
        "before pointing. Slice indices are navigation.",
        "lasso/scribble/paint: list mark IDs IN ORDER around the outline (clockwise or "
        "counter-clockwise); an out-of-order list self-intersects and fills with holes.",
        f"budgets: {cs.budgets['max_tool_calls']} tool calls, {cs.budgets['max_model_prompts']} model "
        "prompts (exhausted -> call seg_finish).",
        observation_for_view(cs, "axial", cs.shape[2] // 2, "full", "none", "none"),
    ]
    img, _ri = render.render_overview(cs.image, cs.spacing, mask=cs.current_mask(), marks=None)  # overview is for orientation only; marks are usable only after seg_view
    return OpOutput("\n".join(card), [img])


def op_locate(cs: CaseState, query: str | None = None) -> OpOutput:
    """seg_locate (harness v3, only when a v3 pointer is configured): propose candidate
    marks for the target on the CURRENT VIEW image exactly as the agent saw it. A tool
    call, not a model prompt; failures (service down/timeout) are OpErrors like other
    failed calls. The body lives in segd.v3 (op_locate returns text + image)."""
    from segd import v3 as v3_mod
    text, images = v3_mod.op_locate(cs, query=query)
    return OpOutput(text, images)


def op_sheet(cs: CaseState, axis: str, start: int, stop: int, n: int = 5) -> OpOutput:
    axis = _axis(axis)
    n_slices = cs.shape[config.FIXED_AXIS[axis]]
    try:
        start, stop, n = int(start), int(stop), int(n)
    except (TypeError, ValueError):
        raise OpError(f"sheet start/stop/n must be integers (got {start!r}, {stop!r}, {n!r})")
    if not (0 <= start < n_slices and 0 <= stop < n_slices and start <= stop):
        raise OpError(f"sheet range [{start},{stop}] out of bounds for {axis} (n={n_slices})")
    n = max(1, min(n, 20))
    render = _render()
    img, ri = render.render_sheet(cs.image, axis, start, stop, n, mask=cs.current_mask(),
                                  spacing=cs.spacing, body=cs.body)
    cs._last_render_info = ri
    text = f"sheet {axis} {start}\u2013{stop}/{n_slices} ({n} slices) | {vid_label(cs.current)} | {budget_line(cs)}"
    return OpOutput(text, [img])


def op_view(cs: CaseState, axis: str, slice=None, crop: str = "full", overlay: str = "outline",
            marks: str = "coarse", ids=None) -> OpOutput:
    axis = _axis(axis)
    if crop not in ("full", "mask", "marks"):
        raise OpError(f"crop must be full|mask|marks (got {crop!r})")
    if overlay not in ("none", "outline", "fill"):
        raise OpError(f"overlay must be none|outline|fill (got {overlay!r})")
    if marks not in ("none", "coarse", "fine", "edges"):
        raise OpError(f"marks must be none|coarse|fine|edges (got {marks!r})")
    s = _resolve_slice(cs, slice, axis)
    render = _render()
    box2d = None
    if crop == "marks":
        # crop='marks' zooms on the GIVEN ids, which must be visible in the CURRENT active
        # view (resolved before the new view replaces it); fine marks are revealed there
        if not ids:
            raise OpError("crop 'marks' needs ids: pass mark ids (global or E<n>) shown in the "
                          "current view; the new view zooms on them")
        resolved = cs.resolve_ids(_check_ids(ids))
        box2d = _marks_crop_box(cs, resolved, axis, s)
        if marks == "coarse":
            marks = "fine"
    crop_spec = box2d if box2d is not None else crop
    img, ri = render.render_view(cs.image, axis, s, crop_spec, overlay, marks, cs.markset,
                                 mask=cs.current_mask(), spacing=cs.spacing, body=cs.body)
    from segd.case_state import ActiveView, primary_view_geom
    cs.active_view = ActiveView(axis, s, primary_view_geom(ri), ri, marks, box2d, img)
    cs._last_render_info = ri
    text = observation_for_view(cs, axis, s, crop, overlay, marks)
    if crop == "full" and marks == "fine" and cs.markset is not None:
        shown = [mid for mid, *_ in ri.marks]
        if all(cs.markset.marks[mid].level == "coarse" for mid in shown):
            text += " | fine marks appear in crops"
    return OpOutput(text, [img])


def _change_image(cs: CaseState, prev_mask: np.ndarray, new_mask: np.ndarray):
    """Three-plane feedback render for mask-changing ops (axial/coronal/sagittal through
    the new mask's centre, cropped around it; new mask green, removed red). Returns
    (image, render_seconds, text suffix naming the slices) so the trajectory's render
    timing is real (finding #10) and the agent can name the slices."""
    t0 = time.time()
    img, ri = _render().render_change_3plane(cs.image, prev_mask, new_mask,
                                             spacing=cs.spacing)
    sl = {v.axis: v.slice_idx for v in ri.views}
    suffix = (f"image: axial {sl['axial']}, coronal {sl['coronal']}, sagittal {sl['sagittal']} "
              f"through {ri.note} (green = new mask, red = removed)")
    return img, round(time.time() - t0, 3), suffix


def _prompt_version_flow(cs: CaseState, resolved, prompt_fn, note: str) -> OpOutput:
    """Shared tail for model-prompt ops: run the prompt, commit the version, observe."""
    parent_vid = cs.current
    with cs.rollback_on_error(parent_vid):   # a render/observation failure must not leave a commit behind
        cs.model_prompt(prompt_fn)
        nni_s = cs._nni_seconds
        img, render_s, suffix = _change_image(cs, cs.version_mask(parent_vid),
                                              cs.version_mask(cs.current))
        text = observation_after_mutation(cs, parent_vid, cs.current) + f" | {note} | {suffix}"
    cs._timings = {"render": render_s, "nni": round(nni_s, 3)}
    cs._last_resolved = [{"id": r.id, "ijk": [int(c) for c in r.ijk]} for r in resolved]
    return OpOutput(text, [img])


def op_point(cs: CaseState, ids, polarity: str = "add") -> OpOutput:
    ids = _check_ids(ids)
    include = _polarity(polarity)
    resolved = cs.resolve_ids(ids)
    session_coords = [cs.full_to_session_ijk(r.ijk) for r in resolved]

    def prompt_fn(session):
        for k, coord in enumerate(session_coords):
            # one prediction per op: run it on the last point only
            session.add_point_interaction(coord, include_interaction=include,
                                          run_prediction=(k == len(session_coords) - 1))

    return _prompt_version_flow(cs, resolved, prompt_fn, f"point {'add' if include else 'erase'} "
                                                          f"({len(resolved)} mark(s))")


def op_lasso(cs: CaseState, ids, polarity: str = "add") -> OpOutput:
    ids = _check_ids(ids)
    if len(ids) < 3:
        raise OpError("lasso needs at least 3 mark IDs")
    include = _polarity(polarity)
    resolved = cs.resolve_ids(ids)
    av = cs.require_active_view()
    drawn = _mask_with_grid(cs, _polygon_grid(cs, resolved, av.axis), av.axis, av.slice_idx)
    bbox = _interaction_box_for(cs, drawn > 0, av.axis, av.slice_idx)
    session_mask = _crop_mask_to_bbox(drawn, bbox)
    session_bbox = _session_bbox(cs, bbox)

    def prompt_fn(session):
        session.add_lasso_interaction(session_mask, include_interaction=include,
                                      run_prediction=True, interaction_bbox=session_bbox)

    return _prompt_version_flow(cs, resolved, prompt_fn, f"lasso {'add' if include else 'erase'}")


def op_scribble(cs: CaseState, ids, polarity: str = "add") -> OpOutput:
    ids = _check_ids(ids)
    if len(ids) < 2:
        raise OpError("scribble needs at least 2 mark IDs")
    include = _polarity(polarity)
    resolved = cs.resolve_ids(ids)
    av = cs.require_active_view()
    drawn = _mask_with_grid(cs, _polygon_grid(cs, resolved, av.axis, polyline=True),
                            av.axis, av.slice_idx)
    bbox = _interaction_box_for(cs, drawn > 0, av.axis, av.slice_idx)
    session_mask = _crop_mask_to_bbox(drawn, bbox)
    session_bbox = _session_bbox(cs, bbox)

    def prompt_fn(session):
        session.add_scribble_interaction(session_mask, include_interaction=include,
                                         run_prediction=True, interaction_bbox=session_bbox)

    return _prompt_version_flow(cs, resolved, prompt_fn, f"scribble {'add' if include else 'erase'}")


def op_box(cs: CaseState, id_a, id_b, polarity: str = "add") -> OpOutput:
    include = _polarity(polarity)
    av = cs.require_active_view()
    resolved = cs.resolve_ids([id_a, id_b])
    p1 = voxel_to_inplane(av.axis, resolved[0].ijk, cs.shape)
    p2 = voxel_to_inplane(av.axis, resolved[1].ijk, cs.shape)
    (r_lo, r_hi), (c_lo, c_hi) = sorted((p1[0], p2[0])), sorted((p1[1], p2[1]))
    a1, a2, fixed = config.ROW_AXIS[av.axis], config.COL_AXIS[av.axis], config.FIXED_AXIS[av.axis]
    # displayed (row, col) -> full-space half-open voxel bbox
    bbox = [None, None, None]
    bbox[a1] = (cs.shape[a1] - 1 - r_hi, cs.shape[a1] - r_lo)
    bbox[a2] = (cs.shape[a2] - 1 - c_hi, cs.shape[a2] - c_lo)
    bbox[fixed] = (av.slice_idx, av.slice_idx + 1)
    bbox = tuple(bbox)
    if any(hi - lo < 1 for lo, hi in bbox):
        raise OpError("box collapsed to a point; choose two different marks")
    bbox = _clip_box_to_crop(cs, bbox)   # a box mark outside the uploaded crop errors cleanly
    session_bbox = _session_bbox(cs, bbox)

    def prompt_fn(session):
        session.add_bbox_interaction(session_bbox, include_interaction=include, run_prediction=True)

    return _prompt_version_flow(cs, resolved, prompt_fn, f"box {'add' if include else 'erase'}")


def op_paint(cs: CaseState, ids, mode: str = "add", slice_range=None) -> OpOutput:
    ids = _check_ids(ids)
    if len(ids) < 3:
        raise OpError("paint needs at least 3 mark IDs forming a polygon")
    if mode not in ("add", "erase"):
        raise OpError(f"mode must be add|erase (got {mode!r})")
    resolved = cs.resolve_ids(ids)
    av = cs.require_active_view()
    slices = _slice_range(slice_range, cs, av.axis, av.slice_idx)
    grid = _polygon_grid(cs, resolved, av.axis)
    mask = cs.current_mask()
    for s in slices:
        _place_grid_on_slice(mask, grid, av.axis, s, "add" if mode == "add" else "erase")
    parent_vid = cs.current
    # render BEFORE committing: a render failure leaves the mask unchanged
    img, render_s, suffix = _change_image(cs, cs.current_mask(), mask)
    with cs.rollback_on_error(parent_vid):
        vid = cs.commit_mask(mask)
        cs._nni_dirty = True
        text = (observation_after_mutation(cs, parent_vid, vid)
                + f" | paint {mode} on {len(slices)} slice(s) | {suffix}")
    cs._last_resolved = [{"id": r.id, "ijk": [int(c) for c in r.ijk]} for r in resolved]
    cs._timings = {"render": render_s, "nni": 0.0}
    return OpOutput(text, [img])


def op_refine(cs: CaseState) -> OpOutput:
    mask_session = cs.current_mask_in_session_space()
    if not mask_session.any():
        raise OpError("refine: current mask is empty")

    def prompt_fn(session):
        session.add_initial_seg_interaction(mask_session, run_prediction=True)

    out = _prompt_version_flow(cs, [], prompt_fn, "refine from current mask")
    return out


def ball_morphology(m: np.ndarray, op: str, r_vox) -> np.ndarray:
    """Binary dilate/erode/open/close with the ellipsoid ball sum((d_i/r_i)^2) <= 1 (voxel offsets d),
    identical to scipy.ndimage binary_* with that structure and border_value=0, but computed with exact
    Euclidean distance transforms on a padded crop around the mask (scipy's cost grows with the ball volume:
    a 14 mm opening of a 500 mL mask took 11 min and blocked segd; Phase 3, V25)."""
    r = np.asarray(r_vox, dtype=float)
    pad = int(np.ceil(r.max())) + 2
    if not m.any():
        return np.zeros_like(m, dtype=bool)
    idx = np.nonzero(m)
    lo = [max(0, int(i.min()) - 2 * pad) for i in idx]
    hi = [min(n, int(i.max()) + 1 + 2 * pad) for i, n in zip(idx, m.shape)]
    crop = m[tuple(slice(a, b) for a, b in zip(lo, hi))]
    # zero padding beyond the real array emulates border_value=0; `inside` re-imposes the array bounds
    # after each step (scipy never sets voxels outside the array)
    pw = [(pad, pad)] * 3
    x = np.pad(crop, pw)
    inside = np.pad(np.ones(crop.shape, bool), pw)
    for a, (l, h, n) in enumerate(zip(lo, hi, m.shape)):
        sl = [slice(None)] * 3
        if l > 0:
            sl[a] = slice(0, pad); inside[tuple(sl)] = True
        if h < n:
            sl[a] = slice(x.shape[a] - pad, None); inside[tuple(sl)] = True
    samp = tuple(1.0 / r)
    tol = float(np.sqrt(1.0 + 1e-9))   # same tolerance as the scipy structure in op_edit

    def dil(b):
        if not b.any():
            return np.zeros_like(b)
        return (ndimage.distance_transform_edt(~b, sampling=samp) <= tol) & inside

    def ero(b):
        # out-of-array cells are 0 in b (zero padding), so they act as background (border_value=0)
        if not b.any():
            return np.zeros_like(b)
        return (ndimage.distance_transform_edt(b, sampling=samp) > tol) & inside

    seq = {"dilate": [dil], "erode": [ero], "open": [ero, dil], "close": [dil, ero]}[op]
    for f in seq:
        x = f(x)
    out = np.zeros(m.shape, bool)
    out[tuple(slice(a, b) for a, b in zip(lo, hi))] = x[pad:-pad, pad:-pad, pad:-pad]
    return out


def op_edit(cs: CaseState, op: str, mark=None, radius_mm=None) -> OpOutput:
    if op not in ("keep_component", "remove_component", "fill_holes", "close", "open", "dilate", "erode"):
        raise OpError("edit op must be keep_component|remove_component|fill_holes|close|open|dilate|erode "
                      f"(got {op!r})")
    mask = cs.current_mask()
    if not (mask > 0).any():
        raise OpError("edit: current mask is empty")
    parent_vid = cs.current
    resolved: list = []
    struct = None
    if op in ("close", "open", "dilate", "erode"):
        if radius_mm is None:
            # the seg_edit tool documents "radius_mm sets the morphology radius (default 2)"
            radius_mm = config.EDIT_DEFAULT_RADIUS_MM
        try:
            radius = float(radius_mm)
        except (TypeError, ValueError):
            raise OpError(f"radius_mm must be a number (got {radius_mm!r})")
        r_vox = [max(1, int(round(radius / s))) for s in cs.spacing]
        axes = [np.arange(-r, r + 1) / max(r, 1) for r in r_vox]
        grids = np.meshgrid(*axes, indexing="ij")
        struct = sum(g ** 2 for g in grids) <= 1.0 + 1e-9   # ellipsoid ball in voxel units
    if op in ("keep_component", "remove_component"):
        if mark is None:
            raise OpError(f"edit {op} needs a mark ID")
        resolved = cs.resolve_ids([mark])
        seed_vox = _nearest_mask_voxel(mask, resolved[0].ijk, 5)
        if seed_vox is None:
            raise OpError(f"mark {resolved[0].id} is not inside the current mask (no mask voxel within 5 voxels)")
        labels, n = ndimage.label(mask > 0, structure=ndimage.generate_binary_structure(3, 1))
        comp_id = labels[tuple(seed_vox)]
        comp = labels == comp_id
        mask = ((mask > 0) & (comp if op == "keep_component" else ~comp)).astype(np.uint8)
    elif op == "fill_holes":
        mask = ndimage.binary_fill_holes(mask > 0).astype(np.uint8)
    elif op in ("close", "open", "dilate", "erode"):
        mask = ball_morphology(mask > 0, op, r_vox).astype(np.uint8)
    img, render_s, suffix = _change_image(cs, cs.current_mask(), mask)   # render before commit
    with cs.rollback_on_error(parent_vid):
        vid = cs.commit_mask(mask)
        cs._nni_dirty = True
        text = observation_after_mutation(cs, parent_vid, vid) + f" | edit {op} | {suffix}"
    cs._last_resolved = [{"id": r.id, "ijk": [int(c) for c in r.ijk]} for r in resolved]
    cs._timings = {"render": render_s, "nni": 0.0}
    return OpOutput(text, [img])


def _nearest_mask_voxel(mask: np.ndarray, ijk, max_dist: int):
    """Nearest mask voxel within max_dist of ijk, or None (component ops are tolerant of
    a mark placed a few voxels off the component boundary)."""
    i, j, k = (int(c) for c in ijk)
    lo = [max(c - max_dist, 0) for c in (i, j, k)]
    hi = [min(c + max_dist + 1, n) for c, n in zip((i, j, k), mask.shape)]
    sub = mask[tuple(slice(a, b) for a, b in zip(lo, hi))]
    if not sub.any():
        return None
    idx = np.array(np.nonzero(sub)).T
    target = np.array([i, j, k]) - np.array(lo)
    best = idx[int(np.argmin(((idx - target) ** 2).sum(axis=1)))]
    return tuple(int(c) + o for c, o in zip(best, lo))


def op_undo(cs: CaseState) -> OpOutput:
    was_vid = cs.current
    with cs.rollback_on_error(was_vid):
        new_vid = cs.undo()
        cs._last_resolved = []
        # (restored; was vN), diffed against the previously current version (finding #8)
        img, render_s, suffix = _change_image(cs, cs.version_mask(was_vid), cs.version_mask(new_vid))
        text = observation_after_restore(cs, new_vid, was_vid) + f" | undo | {suffix}"
    cs._timings = {"render": render_s, "nni": 0.0}
    return OpOutput(text, [img])


def op_checkout(cs: CaseState, version) -> OpOutput:
    was_vid = cs.current
    with cs.rollback_on_error(was_vid):
        new_vid = cs.checkout(version)
        cs._last_resolved = []
        img, render_s, suffix = _change_image(cs, cs.version_mask(was_vid), cs.version_mask(new_vid))
        text = observation_after_restore(cs, new_vid, was_vid) + f" | checkout | {suffix}"
    cs._timings = {"render": render_s, "nni": 0.0}
    return OpOutput(text, [img])


# ------------------------------------------------------------- candidates
CANDIDATE_JUDGE_TEXT = (
    "For each candidate decide whether the tissue inside its outline is part of the same lesion "
    "(accept) or normal tissue (reject). Accept with seg_accept and call seg_finish again for a "
    "new review; if none belong, confirm with the review code.")
CANDIDATE_IMAGE_TEXT = ("candidate image: one row per candidate, columns axial / coronal / sagittal through "
                        "the candidate (green outline = current mask, magenta outline + light tint = candidate; "
                        "judge the tissue inside the magenta outline)")


def _candidates_for_current(cs: CaseState) -> list:
    """Add-candidates for the CURRENT version (GT-free 'intensity' generator, cached per
    version id in CaseState; the case's body mask is reused). Fills the trajectory's extra
    fields. Visibility (``candidates_shown``) is NOT set here: the calling op authorizes
    only after the candidate sheet rendered (and run_op's transaction makes the
    authorization fall back if saving/logging fails) — a failed render authorizes nothing."""
    cand_mod = _candidates_module()
    vid = cs.current
    if vid not in cs.candidate_cache:
        t0 = time.time()
        mask = cs.current_mask()
        cache = {"body": cs.body} if cs.body is not None and cs.body.any() else None
        found = cand_mod.propose(cs.image, mask, cs.spacing, method=config.CANDIDATE_METHOD,
                                 max_n=config.CANDIDATE_MAX_N, cache=cache)
        cs.candidate_cache[vid] = [c for c in found if c.kind == "add"]
        cs._timings_candidates = round(time.time() - t0, 3)
    cands = cs.candidate_cache[vid]
    cs._log_extra["candidates"] = {
        "version": vid_label(vid), "method": config.CANDIDATE_METHOD,
        "items": [{"id": c.id, "kind": c.kind, "volume_ml": round(c.volume_ml, 3), "how": c.how}
                  for c in cands],
    }
    return cands


def _candidate_lines(cs: CaseState, cands: list) -> str:
    return "\n".join(_candidates_module().describe_candidate(c, cs.current_mask(), cs.spacing)
                     for c in cands)


def _candidate_image(cs: CaseState, cands: list):
    """The candidate sheet + its RenderInfo (v3 uses the row paste rects for the
    per-candidate second-reader checkpoint)."""
    t0 = time.time()
    img, ri = _render().render_candidates(cs.image, cs.current_mask(), cands, cs.spacing)
    cs._timings["render"] = round((cs._timings.get("render") or 0.0) + (time.time() - t0), 3)
    return img, ri


def op_candidates(cs: CaseState) -> OpOutput:
    """seg_candidates: propose (GT-free) corrections for the current mask; text list + one
    image. Nothing changes until seg_accept."""
    if not (cs.current_mask() > 0).any():
        raise OpError("candidates: current mask is empty")
    cands = _candidates_for_current(cs)
    label = vid_label(cs.current)
    if not cands:
        return OpOutput(f"{label} | no candidates: nothing next to the mask looks like it could "
                        f"belong to it | {budget_line(cs)}", [])
    text = (f"{label} | {len(cands)} candidate(s): automatic proposals of possibly missed lesion tissue "
            "next to the mask, computed from image intensity only (often wrong; judge each by the tissue "
            "inside its outline).\n"
            f"{_candidate_lines(cs, cands)}\n{CANDIDATE_IMAGE_TEXT}. "
            f"Accept the ones that belong with seg_accept | {budget_line(cs)}")
    img, _ri = _candidate_image(cs, cands)   # render BEFORE anything is authorized
    cs.candidates_shown = cs.current      # visibility only after the sheet rendered; if saving
    return OpOutput(text, [img])          # or logging fails, run_op's transaction revokes it again


def op_accept(cs: CaseState, ids) -> OpOutput:
    """seg_accept: union of the chosen add-candidates as ONE new version (a direct edit:
    nni is marked dirty per the sync rule, no model prompt)."""
    ids = _check_ids(ids)
    wanted: list[str] = []
    for raw in ids:
        cid = str(raw).strip().upper()
        if cid not in wanted:
            wanted.append(cid)
    if cs.candidates_shown != cs.current or cs.current not in cs.candidate_cache:
        # candidates belong to the version they were computed for; any mask change invalidates them
        raise OpError("candidates are out of date; call seg_candidates again")
    shown = {c.id: c for c in cs.candidate_cache[cs.current]}
    unknown = [c for c in wanted if c not in shown]
    if unknown:
        raise OpError(f"unknown candidate id(s) {unknown}; current candidates: {sorted(shown) or 'none'}")
    parent_vid = cs.current
    mask = cs.current_mask()
    for cid in wanted:
        c = shown[cid]
        mask[c.bbox] |= c.crop.astype(mask.dtype)
    img, render_s, suffix = _change_image(cs, cs.current_mask(), mask)   # render before commit
    with cs.rollback_on_error(parent_vid):
        vid = cs.commit_mask(mask)
        cs._nni_dirty = True
        text = (observation_after_mutation(cs, parent_vid, vid)
                + f" | accept {', '.join(wanted)} (candidate edit, no model prompt) | {suffix}")
    cs.accept_rounds += 1                # (authorization was revoked by commit_mask; the
                                         # run_op transaction restores both on later failure)
    cs._log_extra["accepted"] = {"ids": wanted, "from_version": vid_label(parent_vid),
                                 "to_version": vid_label(vid),
                                 "volume_ml": round(sum(shown[c].volume_ml for c in wanted), 3),
                                 "how": [shown[c].how for c in wanted]}
    cs._last_resolved = []
    cs._timings = {"render": render_s, "nni": 0.0}
    return OpOutput(text, [img])


def _review_code(cs: CaseState) -> str:
    """Deterministic 4-digit code for one review (case, version, call count)."""
    import hashlib
    h = hashlib.sha256(f"{cs.case_id}|{cs.current}|{cs.tool_calls}".encode()).hexdigest()
    return str(1000 + int(h[:8], 16) % 9000)


def op_finish(cs: CaseState, confidence: str = "medium", reason: str = "",
              skip_review: bool = False, review_code=None) -> OpOutput:
    """Two-stage finish: the first call for a given current version returns a final
    review (3x5 grid) and does not finish; a second call with the same current version
    finishes. Exhausted tool budget (run_op already counted this call) and the internal
    skip_review flag (runner forced finish) finish immediately."""
    if confidence not in ("high", "medium", "low"):
        raise OpError("confidence must be high|medium|low")
    if not isinstance(skip_review, bool):
        raise OpError("skip_review must be a boolean")
    exhausted = cs.tool_calls > cs.budgets["max_tool_calls"]
    reviewed = cs.reviewed_version == cs.current
    code = getattr(cs, "review_code", None)
    if reviewed and not (skip_review or exhausted) and str(review_code or "").strip() != code:
        # The confirmation must quote the code shown in the latest review, so a review
        # cannot be confirmed without reading it (e.g. two seg_finish calls in one turn).
        return OpOutput("Not saved. To save, call seg_finish with review_code set to the code "
                        "printed in the latest final review (read the review first). "
                        f"{budget_line(cs)}", [])
    if not (skip_review or exhausted or reviewed):
        mask = cs.current_mask()
        st = mask_stats(mask, cs.spacing)
        label = vid_label(cs.current)
        t0 = time.time()
        img, _ri = _render().render_review_grid(cs.image, mask, cs.spacing)
        cs._timings = {"render": round(time.time() - t0, 3), "nni": 0.0}
        images = [img]
        cand_text = ""
        cands: list = []
        cand_row_rects: list = []
        cand_img = None
        comp_text = ""
        # Phase 4 (non-MSTT datasets): with several separated lesions the union-extent
        # grid's fraction columns can fall between them — add the per-component sheet
        # (rows through each component's centre) so every lesion's outline is reviewable.
        if is_multi_lesion(cs) and st["n_components"] >= 2:
            img_comp, _rc = _render().render_review_components(cs.image, mask, cs.spacing)
            images.append(img_comp)
            comp_text = ("The next image is the component sheet: one row per detected component "
                         "(largest first, up to 5), axial/coronal/sagittal through its centre. "
                         "Check each lesion's outline separately. ")
        if st["n_components"] > 0:
            # required candidate round: every first review for a version also proposes
            # candidates (capped after CANDIDATE_REVIEW_MAX_ACCEPT_ROUNDS accepted rounds)
            if cs.accept_rounds >= config.CANDIDATE_REVIEW_MAX_ACCEPT_ROUNDS:
                cand_text = (f"Candidates: none offered ({cs.accept_rounds} candidate rounds accepted; the "
                             f"review stops offering them after {config.CANDIDATE_REVIEW_MAX_ACCEPT_ROUNDS}). ")
            else:
                cands = _candidates_for_current(cs)
                if cands:
                    img_c, ri_c = _candidate_image(cs, cands)   # render BEFORE anything is authorized
                    images.append(img_c)
                    cand_img = img_c
                    cs.candidates_shown = cs.current      # after the sheet rendered; a later
                    # save_render/logging failure rolls the authorization back (run_op transaction)
                    nth = "third" if comp_text else "second"
                    cand_text = (f"Candidates ({nth} image): automatic proposals of possibly missed lesion "
                                 "tissue next to the mask, computed from image intensity only (often wrong).\n"
                                 f"{_candidate_lines(cs, cands)}\n{CANDIDATE_IMAGE_TEXT}. "
                                 f"{CANDIDATE_JUDGE_TEXT} ")
                    cand_row_rects = list(getattr(ri_c, "panel_rects", None) or [])
                else:
                    cand_text = "Candidates: none found. "
        cs.reviewed_version = cs.current
        cs.review_code = _review_code(cs)
        confirm = (f"If everything is right, call seg_finish again with review_code "
                   f"\"{cs.review_code}\" to save it.")
        if st["n_components"] == 0:
            text = (f"Final review of {label}: the mask is empty; finishing will save an empty mask. "
                    f"Image: three planes through the volume centre. {budget_line(cs)}. "
                    "If that is intended, " + confirm[len("If everything is right, "):] +
                    " Otherwise keep editing.")
        else:
            text = (f"Final review of {label}: volume {st['volume_ml']:.1f} mL | "
                    f"{components_text_for(cs, st)} | {_fmt_extent(st['extent'])} | {budget_line(cs)}. "
                    "The image shows five slices per plane (three inside the mask extent + one "
                    "just past each end; past-the-mask panels are orange). "
                    + comp_text
                    + "Check: (1) Is lesion tissue visible in the past-the-mask panels (orange)? "
                    "Then the mask stops too early: seg_view that slice and add positive points "
                    "on the lesion there. (2) Does the outline cut through lesion or include "
                    "normal tissue on any panel? (3) Stray components? If any answer shows a "
                    "problem, fix it before finishing. " + cand_text + confirm)
        # harness v3 second-reader checkpoints (review grid + per candidate), once per
        # version; appended to the observation, never able to fail the review
        v3_note = ""
        if cs.v3 is not None:
            from segd import v3 as v3_mod
            try:
                v3_note = v3_mod.finish_review_notes(cs, img, cand_img,
                                                     cand_row_rects, cands, cs.current)
            except Exception as e:                          # noqa: BLE001 (by design)
                # last resort so a hook crash cannot fail the review either — but it must
                # stay honest: record the failure and append the standard unavailable note
                # (the checkpoints above already convert their own failures to notes; the
                # old empty-string fallback silently dropped mandatory checkpoints)
                try:
                    v3_mod.record_call(cs, call="checkpoint", checkpoint="finish", ok=False,
                                       error=f"hook: {type(e).__name__}: {e}"[:300])
                    v3_note = v3_mod.MEDGEMMA_UNAVAILABLE
                except Exception:                           # noqa: BLE001
                    v3_note = ""
        if v3_note:
            text += "\n" + v3_note
        return OpOutput(text, images)
    st = mask_stats(cs.current_mask(), cs.spacing)
    text = cs.finish(confidence, reason)
    text += (f" | final volume {st['volume_ml']:.1f} mL, {components_text_for(cs, st)} "
             f"| {budget_line(cs)}")
    return OpOutput(text, [], done=True)


OPS = {
    "open": op_open,
    "sheet": op_sheet,
    "view": op_view,
    "locate": op_locate,
    "point": op_point,
    "lasso": op_lasso,
    "scribble": op_scribble,
    "box": op_box,
    "paint": op_paint,
    "refine": op_refine,
    "edit": op_edit,
    "undo": op_undo,
    "checkout": op_checkout,
    "candidates": op_candidates,
    "accept": op_accept,
    "finish": op_finish,
}

ARG_SPECS = {
    "open": set(),
    "sheet": {"axis", "start", "stop", "n"},
    "view": {"axis", "slice", "crop", "overlay", "marks", "ids"},
    "locate": {"query"},
    "point": {"ids", "polarity"},
    "lasso": {"ids", "polarity"},
    "scribble": {"ids", "polarity"},
    "box": {"id_a", "id_b", "polarity"},
    "paint": {"ids", "mode", "slice_range"},
    "refine": set(),
    "edit": {"op", "mark", "radius_mm"},
    "undo": set(),
    "checkout": {"version"},
    "candidates": set(),
    "accept": {"ids"},
    "finish": {"confidence", "reason", "skip_review", "review_code"},
}


# Ops that change durable state (versions, current version, candidate/review authorization,
# accept counter, done flag): when rendering or trajectory logging fails after the op body
# ran, the snapshot taken in run_op restores all of it, so the error reply is true ("nothing
# happened"). Pure-read ops (open/sheet/view) need no transaction.
TRANSACTIONAL_OPS = frozenset({
    "point", "lasso", "scribble", "box", "paint", "refine", "edit",
    "undo", "checkout", "candidates", "accept", "finish",
})


def _transaction_snapshot(cs: CaseState) -> dict:
    """Durable-state snapshot of ``cs`` before a TRANSACTIONAL_OPS op runs."""
    return {
        "versions": dict(cs.versions),
        "current": cs.current,
        "accept_rounds": cs.accept_rounds,
        "candidates_shown": cs.candidates_shown,
        "reviewed_version": cs.reviewed_version,
        "review_code": cs.review_code,
        "done": cs.done,
        "flushed_through": cs._versions_flushed_through,
        "v3_fired": set(cs.v3_fired),
    }


def _transaction_rollback(cs: CaseState, snap: dict | None) -> None:
    """Restore the durable state a failed op had changed (see TRANSACTIONAL_OPS)."""
    if snap is None:
        return
    cs.versions = snap["versions"]
    cs.current = snap["current"]
    cs.accept_rounds = snap["accept_rounds"]
    cs.candidates_shown = snap["candidates_shown"]
    cs.reviewed_version = snap["reviewed_version"]
    cs.review_code = snap["review_code"]
    cs.done = snap["done"]
    cs._versions_flushed_through = snap["flushed_through"]
    cs.v3_fired = snap["v3_fired"]
    for vid in [v for v in cs.candidate_cache if v not in snap["versions"]]:
        cs.candidate_cache.pop(vid, None)


def run_op(cs: CaseState, op_name: str, args: dict | None = None) -> OpOutput:
    """Validate + execute one op (app.py holds the per-case lock), with budget
    enforcement, trajectory logging and render saving. Raises OpError for agent-facing
    errors; unexpected exceptions propagate (app.py converts them to an HTTP 200
    ok:false generic response; run_op already logged the traceback, finding #2).

    Budget semantics (decision on finding #3): EVERY seg_* call counts as one tool call,
    including calls that fail and views/sheets; the model-prompt budget counts only
    prompts actually sent. Rejected calls are logged to the trajectory. When the tool
    call budget is exhausted, every op except `finish` is rejected; when the model-prompt
    budget is exhausted, model prompts are rejected but other ops still work."""
    if not isinstance(op_name, str):
        op_name = repr(op_name)
    bad_args = args is not None and not isinstance(args, dict)
    args = dict(args) if isinstance(args, dict) else {}

    def reject(msg: str, kind: str) -> OpError:
        cs.log.record(op=op_name, args=args, resolved=[], error=msg, error_kind=kind,
                      timings={"total": 0.0})
        return OpError(_with_budget(cs, msg))

    if cs.done:
        # after finish the case is over: reject without touching the final counters
        raise reject("case is finished; no further ops are allowed", "op")
    if op_name != "finish" and cs.tool_calls >= cs.budgets["max_tool_calls"]:
        raise reject(f"tool call budget exhausted ({cs.tool_calls}/{cs.budgets['max_tool_calls']}); "
                     "call seg_finish now", "budget")

    t0 = time.time()
    cs._timings = {}
    cs._timings_candidates = None
    cs._log_extra = {}
    cs._last_resolved = []
    cs._last_render_info = None
    # v3 MedGemma ABSOLUTE per-op deadline (finding #1): starts at op ENTRY (not at the
    # first call), so pre-hook work (renders, candidate generation) counts against the
    # same budget the extension's HTTP deadline sees; None when MedGemma is off.
    from segd.v3 import medgemma_on
    cs._v3_group_deadline = (time.perf_counter() + config.V3_MEDGEMMA_GROUP_TIMEOUT_S
                             if medgemma_on(cs) else None)
    cs.tool_calls += 1   # every call counts, including the ones that fail below
    if op_name in config.MODEL_PROMPT_OPS and cs.model_prompts >= cs.budgets["max_model_prompts"]:
        raise reject("model prompt budget used; edit with seg_paint/seg_edit or call seg_finish",
                     "budget")
    parent_vid = cs.current
    snap = _transaction_snapshot(cs) if op_name in TRANSACTIONAL_OPS else None
    # checkpoint-delivery state is snapshotted for EVERY op, independently of the durable-
    # mask transaction: the v3 localize hook also fires after non-transactional ops
    # (open/sheet/view), so a failed render save / trajectory record there must restore it
    # just the same — otherwise a failure permanently consumes the checkpoint and the
    # retry never delivers the advice
    v3_fired_snap = set(cs.v3_fired)
    try:
        # an unknown op / malformed args still count as a tool call and are logged below
        if bad_args:
            raise OpError(f"args for op {op_name!r} must be a JSON object")
        if op_name not in OPS:
            raise OpError(f"unknown op {op_name!r}")
        unknown = set(args) - ARG_SPECS[op_name]
        if unknown:
            raise OpError(f"unknown argument(s) for op {op_name}: {sorted(unknown)}")
        out = OPS[op_name](cs, **args)
        # harness v3 localize checkpoint injection (first sheet / open overview fallback);
        # no-op when the case has no v3 flags. Never raises (v3.after_op_hook catches).
        if cs.v3 is not None:
            from segd import v3 as v3_mod
            out = v3_mod.after_op_hook(cs, op_name, out)
    except OpError as e:
        # v3 request/response records (seg_locate, second-reader calls) are diagnostics
        # and are logged even when the op failed; the op's own success fields (e.g.
        # candidates/accepted) never appear on an error record (candidate-flow contract).
        v3_extra = {"v3": cs._log_extra["v3"]} if "v3" in cs._log_extra else None
        cs.log.record(op=op_name, args=args, resolved=cs._last_resolved, error=str(e),
                      error_kind="op", timings={"total": round(time.time() - t0, 3)},
                      parent=vid_label(parent_vid), extra=v3_extra)
        raise OpError(_with_budget(cs, str(e))) from e
    except Exception:
        # unexpected error: undo the op's state changes, then log the full traceback with
        # error_kind 'internal' so scoring can tag `harness` (finding #2)
        _transaction_rollback(cs, snap)
        cs.v3_fired = v3_fired_snap   # checkpoint-delivery state back to pre-op (all ops)
        v3_extra = {"v3": cs._log_extra["v3"]} if "v3" in cs._log_extra else None
        cs.log.record(op=op_name, args=args, resolved=cs._last_resolved,
                      error=traceback.format_exc(), error_kind="internal",
                      timings={"total": round(time.time() - t0, 3)},
                      parent=vid_label(parent_vid), extra=v3_extra)
        raise
    try:
        # render saving + trajectory logging stay INSIDE the transaction: the op is not
        # durable until both succeeded, so a failure here (save_render, record, ...) restores
        # versions/current version, counters and authorization and logs error_kind 'internal'
        # before app.py sends the generic error reply (fix round #3).
        render_files = [cs.log.save_render(img, op_name) for img in out.images]
        timings = {"render": cs._timings.get("render"), "nni": cs._timings.get("nni"),
                   "total": round(time.time() - t0, 3)}
        if cs._timings_candidates is not None:
            timings["candidates"] = cs._timings_candidates
        child = cs.current if cs.current != parent_vid else None
        cs.log.record(op=op_name, args=args, resolved=cs._last_resolved, timings=timings,
                      parent=vid_label(parent_vid),
                      child=vid_label(child) if child is not None else None,
                      observation=out.text, renders=render_files, extra=cs._log_extra)
    except Exception:
        _transaction_rollback(cs, snap)
        cs.v3_fired = v3_fired_snap   # checkpoint-delivery state back to pre-op (all ops):
        # the note was never delivered (the agent gets the error reply), so the retry must
        # still fire the checkpoint instead of seeing it as already consumed
        try:   # best effort: persistence itself may be what failed
            v3_extra = {"v3": cs._log_extra["v3"]} if "v3" in cs._log_extra else None
            cs.log.record(op=op_name, args=args, resolved=cs._last_resolved,
                          error=traceback.format_exc(), error_kind="internal",
                          timings={"total": round(time.time() - t0, 3)},
                          parent=vid_label(parent_vid), extra=v3_extra)
        except Exception:
            pass
        raise
    out.done = cs.done
    return out
