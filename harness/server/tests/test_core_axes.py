"""Placement on every display axis (finding #4) and the crop-path edge-mark regression
(reviewer blocker #1).

After fix #1 the lasso/scribble interaction box has size 1 on the fixed axis and is
clipped to the uploaded crop BEFORE the mask is cut, so FakeNNI can locate the drawing
plane from the size-1 dimension on any axis (it must not assume axial).
"""
from __future__ import annotations

import numpy as np
import pytest

import segd.ops
from segd import config
from segd.case_state import drawn_global_ids
from segd.fake_nni import FakeNNI
from _helpers import add_ring_marks, dice, install_stubs, lesion_center, make_case_state

AXES = ["axial", "coronal", "sagittal"]


def _ring_radii(lesion: np.ndarray, axis: str) -> tuple[int, int]:
    """(drow, dcol) ring radii for corner marks just INSIDE the lesion cross-section on
    ``axis``, so the corner polygon approximates the lesion (good polygon/lesion Dice)."""
    radii = []
    for a in (config.ROW_AXIS[axis], config.COL_AXIS[axis]):
        idx = np.nonzero(lesion.any(axis=tuple(i for i in range(3) if i != a)))[0]
        radii.append(max(int((idx.max() - idx.min()) // 2) - 1, 2))
    return radii[0], radii[1]


@pytest.mark.parametrize("axis", AXES)
def test_lasso_places_on_active_slice_per_axis(tmp_path, monkeypatch, axis):
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    c = lesion_center(lesion)
    s = c[config.FIXED_AXIS[axis]]
    drow, dcol = _ring_radii(lesion, axis)
    ids, _centre = add_ring_marks(cs, c, dx=dcol, dy=drow, axis=axis)
    segd.ops.run_op(cs, "view", {"axis": axis, "slice": s, "marks": "fine"})
    out = segd.ops.run_op(cs, "lasso", {"ids": ids})
    assert "lasso add" in out.text
    m = cs.current_mask()
    fixed = config.FIXED_AXIS[axis]
    # the mask covers the lesion cross-section on the ACTIVE slice of the right axis
    assert dice(np.take(m, s, axis=fixed), np.take(lesion, s, axis=fixed)) > 0.8
    # the fake (like the real model) only spreads a few slices around the drawn slice
    nz = np.nonzero(m.any(axis=tuple(i for i in range(3) if i != fixed)))[0]
    assert s in nz.tolist() and len(nz) <= 7


@pytest.mark.parametrize("axis", AXES)
def test_scribble_places_on_active_slice_per_axis(tmp_path, monkeypatch, axis):
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    c = lesion_center(lesion)
    s = c[config.FIXED_AXIS[axis]]
    drow, dcol = _ring_radii(lesion, axis)
    ids, _centre = add_ring_marks(cs, c, dx=dcol, dy=drow, axis=axis)
    segd.ops.run_op(cs, "view", {"axis": axis, "slice": s, "marks": "fine"})
    out = segd.ops.run_op(cs, "scribble", {"ids": [ids[0], ids[2]]})
    assert "scribble add" in out.text
    m = cs.current_mask()
    fixed = config.FIXED_AXIS[axis]
    assert dice(np.take(m, s, axis=fixed), np.take(lesion, s, axis=fixed)) > 0.5


@pytest.mark.parametrize("axis", AXES)
def test_box_places_on_active_slice_per_axis(tmp_path, monkeypatch, axis):
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    c = lesion_center(lesion)
    s = c[config.FIXED_AXIS[axis]]
    drow, dcol = _ring_radii(lesion, axis)
    ids, _centre = add_ring_marks(cs, c, dx=dcol, dy=drow, axis=axis)
    segd.ops.run_op(cs, "view", {"axis": axis, "slice": s, "marks": "fine"})
    segd.ops.run_op(cs, "box", {"id_a": ids[0], "id_b": ids[2]})
    m = cs.current_mask()
    fixed = config.FIXED_AXIS[axis]
    nz = np.nonzero(m.any(axis=tuple(i for i in range(3) if i != fixed)))[0]
    assert nz.tolist() == [s]                       # a 2D box acts on the active slice only
    assert dice(np.take(m, s, axis=fixed), np.take(lesion, s, axis=fixed)) > 0.7


@pytest.mark.parametrize("axis", AXES)
def test_paint_places_on_slices_per_axis(tmp_path, monkeypatch, axis):
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    c = lesion_center(lesion)
    s = c[config.FIXED_AXIS[axis]]
    drow, dcol = _ring_radii(lesion, axis)
    ids, _centre = add_ring_marks(cs, c, dx=dcol, dy=drow, axis=axis)
    segd.ops.run_op(cs, "view", {"axis": axis, "slice": s, "marks": "fine"})
    out = segd.ops.run_op(cs, "paint", {"ids": ids, "mode": "add", "slice_range": [s - 1, s + 1]})
    assert f"paint add on 3 slice(s)" in out.text
    m = cs.current_mask()
    fixed = config.FIXED_AXIS[axis]
    nz = np.nonzero(m.any(axis=tuple(i for i in range(3) if i != fixed)))[0]
    assert nz.tolist() == [s - 1, s, s + 1]         # exactly the requested range, no model spread
    assert m[tuple(c)] == 1                          # the lesion centre is inside the polygon
    segd.ops.run_op(cs, "paint", {"ids": ids, "mode": "erase", "slice_range": [s - 1, s + 1]})
    assert (cs.current_mask() > 0).sum() == 0


def _d2(p, q) -> float:
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def test_crop_edge_edge_mark_lasso_scribble_real_render(tmp_path, monkeypatch):
    """Reviewer blocker #1 regression, through the REAL segd.render/segd.marks modules on
    a cropped upload (M33-like): edge marks sit on the body surface only ~5 voxels from
    the crop edge, inside the old 8-voxel interaction margin, so the lasso/scribble box
    used to exceed the crop and crash. The box must be clipped to the crop BEFORE the
    mask is cut, with size 1 on the fixed axis."""
    pytest.importorskip("segd.render")
    install_stubs(monkeypatch, force=False)          # real segd.marks + segd.render
    monkeypatch.setattr(config, "UPLOAD_MAX_PACKED_BYTES", 1)   # force the upload crop
    cs, _lesion = make_case_state(tmp_path, monkeypatch, shape=(96, 96, 64))
    assert cs.crop_offset != (0, 0, 0)

    calls = []
    orig_lasso = FakeNNI.add_lasso_interaction
    orig_scrib = FakeNNI.add_scribble_interaction

    def _rec(name, orig):
        def wrapper(self, drawn, include_interaction, run_prediction=True,
                    override_capability_checks=False, interaction_bbox=None):
            calls.append((name, np.array(drawn), [list(b) for b in (interaction_bbox or [])]))
            return orig(self, drawn, include_interaction, run_prediction,
                        override_capability_checks, interaction_bbox)
        return wrapper

    monkeypatch.setattr(FakeNNI, "add_lasso_interaction", _rec("lasso", orig_lasso))
    monkeypatch.setattr(FakeNNI, "add_scribble_interaction", _rec("scribble", orig_scrib))

    # an axial slice where the body actually reaches the crop's tight j (anterior) edge
    jmax_per_k = [int(np.nonzero(cs.body[:, :, k].any(axis=0))[0].max())
                  for k in range(cs.shape[2]) if cs.body[:, :, k].any()]
    k = int(np.argmax(jmax_per_k))
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": k, "marks": "edges"})
    edges = cs.active_view.render_info.edge_marks
    assert len(edges) >= 3
    pts = {e[0]: (int(e[-2]), int(e[-1])) for e in edges}    # real RenderInfo 4-tuples
    # vertices: the topmost edge mark (min row = max j = the tight crop edge), then the
    # two farthest from it and each other -> a non-degenerate triangle at the crop edge
    a = min(pts, key=lambda e: pts[e][0])
    b = max(pts, key=lambda e: _d2(pts[e], pts[a]))
    c = max(pts, key=lambda e: _d2(pts[e], pts[a]) + _d2(pts[e], pts[b]))

    out = segd.ops.run_op(cs, "lasso", {"ids": [a, b, c]})
    assert out.text.startswith("version v1")
    resolved_j = [r["ijk"][1] for r in cs._last_resolved]
    # the drawn marks really reach within the crop margin of a crop edge (this tight
    # edge is what used to crash: margin 8 > the 5-voxel crop margin)
    assert min(resolved_j) - cs.crop_offset[1] <= config.CROP_MARGIN_VOX + 1
    name, drawn, bb = calls[-1]
    assert name == "lasso"
    assert np.shape(drawn) == tuple(hi - lo for lo, hi in bb)   # mask shape == box (the crash)
    assert bb[2][1] - bb[2][0] == 1                             # size 1 on the fixed axis
    assert all(0 <= lo < hi <= n for (lo, hi), n in zip(bb, cs.crop_shape))
    placed = np.zeros(cs.shape, bool)
    placed[tuple(slice(lo + o, hi + o) for (lo, hi), o in zip(bb, cs.crop_offset))] = drawn > 0
    assert np.unique(np.nonzero(placed)[2]).tolist() == [k]     # lands on the active slice
    assert (cs.current_mask() > 0).any()

    segd.ops.run_op(cs, "checkout", {"version": "v0"})
    segd.ops.run_op(cs, "scribble", {"ids": [a, b]})
    name, drawn, bb = calls[-1]
    assert name == "scribble"
    assert np.shape(drawn) == tuple(hi - lo for lo, hi in bb)
    assert bb[2][1] - bb[2][0] == 1
    assert all(0 <= lo < hi <= n for (lo, hi), n in zip(bb, cs.crop_shape))


def test_real_marks_and_render_modules(tmp_path, monkeypatch):
    """Finding #4: one offline test through the REAL segd.marks + segd.render modules
    (small phantom): real marks are computed, drawn with the real RenderInfo 4-tuple
    format, resolved in the active view and turned into a committed version."""
    pytest.importorskip("segd.render")
    install_stubs(monkeypatch, force=False)
    cs, lesion = make_case_state(tmp_path, monkeypatch, shape=(72, 72, 48))
    assert cs.markset is not None and len(cs.markset.marks) > 0
    c = lesion_center(lesion)
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": c[2], "marks": "fine"})
    drawn = sorted(drawn_global_ids(cs.active_view.render_info))
    assert drawn, "real render drew no global marks"
    mid = min(drawn, key=lambda m: sum((a - b) ** 2 for a, b in zip(cs.markset.marks[m].ijk, c)))
    out = segd.ops.run_op(cs, "point", {"ids": [mid]})
    assert "version v1" in out.text and "tool calls" in out.text
    assert (cs.current_mask() > 0).sum() > 0
