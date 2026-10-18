"""Ops: budgets, active-view mark resolution (rejections), the nni sync rule,
paint/edit raster ops, and coordinate round trips through real prompts."""
from __future__ import annotations

import json

import numpy as np
import pytest

import segd.ops
from segd.case_state import voxel_to_inplane
from segd import config
from segd.case_state import OpError, drawn_global_ids, inplane_to_voxel
from segd.fake_nni import FakeNNI
from _helpers import (
    StubMark, StubMarkSet, add_ring_marks, dice, install_stubs, lesion_center,
    make_case_state,
)


def _lesion_mark_id(cs, lesion):
    """The DRAWN mark closest to the lesion centroid — the agent can only point at marks
    the active view actually showed (test-side GT knowledge for the target)."""
    drawn = sorted(drawn_global_ids(cs.active_view.render_info))
    assert drawn, "no marks drawn in the active view"
    c = lesion_center(lesion)
    return min(drawn, key=lambda mid: sum((a - b) ** 2 for a, b in zip(cs.markset.marks[mid].ijk, c)))


def _hidden_fine_id(cs, lesion):
    """A fine mark that exists near the lesion slice but is not drawn in the active view."""
    drawn = drawn_global_ids(cs.active_view.render_info)
    c = lesion_center(lesion)
    cands = [mid for mid, m in cs.markset.marks.items()
             if m.level == "fine" and mid not in drawn and abs(m.ijk[2] - c[2]) <= 3]
    assert cands, "no hidden fine mark near the lesion slice"
    return cands[0]


def test_no_active_view_rejected(tmp_path, monkeypatch):
    install_stubs(monkeypatch, force=True)
    cs, _lesion = make_case_state(tmp_path, monkeypatch)
    with pytest.raises(OpError, match="no active view"):
        segd.ops.run_op(cs, "point", {"ids": [1]})


def test_marks_not_in_view_rejected(tmp_path, monkeypatch):
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    centre_id = min(cs.markset.marks,
                    key=lambda mid: sum((a - b) ** 2 for a, b in zip(cs.markset.marks[mid].ijk,
                                                                    lesion_center(lesion))))
    # the lesion-centre mark is fine-level: a coarse-only view must not accept it
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": lesion_center(lesion)[2], "marks": "coarse"})
    hidden = _hidden_fine_id(cs, lesion)
    with pytest.raises(OpError, match="not shown in the current view"):
        segd.ops.run_op(cs, "point", {"ids": [hidden]})
    # ... but a DRAWN mark resolves fine (positive control)
    drawn_mark = _lesion_mark_id(cs, lesion)
    segd.ops.run_op(cs, "point", {"ids": [drawn_mark]})
    assert cs.current == 1
    # and a view far away from the mark's slice rejects it
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": 5, "marks": "fine"})
    with pytest.raises(OpError, match="not shown in the current view"):
        segd.ops.run_op(cs, "point", {"ids": [centre_id]})
    # unknown IDs
    with pytest.raises(OpError, match="does not exist"):
        segd.ops.run_op(cs, "point", {"ids": [999999]})
    with pytest.raises(OpError, match="not shown in the current view"):
        segd.ops.run_op(cs, "point", {"ids": ["E99"]})

def test_mark_resolves_in_view_and_prompt_dice(tmp_path, monkeypatch):
    """Point on the lesion via a resolved mark: Dice > 0.9 proves the active-view
    projection AND the target-buffer copy."""
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    add_ring_marks(cs, lesion_center(lesion))          # guaranteed marks at/in the lesion
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": lesion_center(lesion)[2], "marks": "fine"})
    mid = _lesion_mark_id(cs, lesion)
    out = segd.ops.run_op(cs, "point", {"ids": [mid]})
    assert out.text.startswith("version v1 (parent v0)")
    assert "volume" in out.text and "tool calls 2/40, prompts 1/15" in out.text
    assert dice(cs.current_mask(), lesion) > 0.9
    # resolved voxel coordinates went into trajectory.jsonl, not into the text
    assert str(mid) not in out.text
    recs = [json.loads(l)["resolved"] for l in open(cs.out_dir / "trajectory.jsonl")]
    assert any(r and r[0]["id"] == mid for r in recs)

def test_edge_mark_resolution(tmp_path, monkeypatch):
    install_stubs(monkeypatch, force=True)
    cs, _lesion = make_case_state(tmp_path, monkeypatch)
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": 40, "marks": "edges"})
    edges = cs.active_view.render_info.edge_marks
    assert len(edges) >= 1
    eid, (row, col) = edges[0]
    out = segd.ops.run_op(cs, "point", {"ids": [eid]})
    assert "version v1" in out.text
    # resolved voxel matches the displayed in-plane convention
    expected = inplane_to_voxel("axial", row, col, 40, cs.shape)
    assert cs._last_resolved[0]["ijk"] == list(expected)

def _recording_fakeni(monkeypatch):
    class RecordingNNI(FakeNNI):
        calls = []

        def add_point_interaction(self, *a, **k):
            RecordingNNI.calls.append(("add_point_interaction", k.get("run_prediction", True)))
            return super().add_point_interaction(*a, **k)

        def add_lasso_interaction(self, *a, **k):
            RecordingNNI.calls.append(("add_lasso_interaction", k.get("run_prediction", True)))
            return super().add_lasso_interaction(*a, **k)

        def add_scribble_interaction(self, *a, **k):
            RecordingNNI.calls.append(("add_scribble_interaction", k.get("run_prediction", True)))
            return super().add_scribble_interaction(*a, **k)

        def add_bbox_interaction(self, *a, **k):
            RecordingNNI.calls.append(("add_bbox_interaction", k.get("run_prediction", True)))
            return super().add_bbox_interaction(*a, **k)

        def add_initial_seg_interaction(self, *a, **k):
            RecordingNNI.calls.append(("add_initial_seg_interaction", k.get("run_prediction", False)))
            return super().add_initial_seg_interaction(*a, **k)

        def reset_interactions(self):
            RecordingNNI.calls.append(("reset_interactions",))
            return super().reset_interactions()

    monkeypatch.setattr("segd.fake_nni.FakeNNI", RecordingNNI)
    return RecordingNNI

def test_nni_sync_rule(tmp_path, monkeypatch):
    """After any non-model change (paint/checkout/undo) the next model prompt must reset
    interactions and re-feed the current mask as the initial seg; back-to-back model
    prompts must not."""
    RecordingNNI = _recording_fakeni(monkeypatch)
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    ids, _centre = add_ring_marks(cs, lesion_center(lesion))

    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": lesion_center(lesion)[2], "marks": "fine"})
    mid = _lesion_mark_id(cs, lesion)
    segd.ops.run_op(cs, "point", {"ids": [mid]})
    assert RecordingNNI.calls == [("add_point_interaction", True)]

    segd.ops.run_op(cs, "paint", {"ids": ids[:4], "mode": "add"})
    assert cs._nni_dirty is True
    segd.ops.run_op(cs, "point", {"ids": [mid]})
    assert [c[0] for c in RecordingNNI.calls] == [
        "add_point_interaction", "reset_interactions", "add_initial_seg_interaction", "add_point_interaction"]
    assert RecordingNNI.calls[2][1] is False          # sync initial seg does not predict

    segd.ops.run_op(cs, "point", {"ids": [mid]})
    assert [c[0] for c in RecordingNNI.calls][-2:] == ["add_point_interaction", "add_point_interaction"]

    segd.ops.run_op(cs, "undo", {})
    assert cs._nni_dirty is True
    segd.ops.run_op(cs, "point", {"ids": [mid]})
    assert [c[0] for c in RecordingNNI.calls][-3:] == [
        "reset_interactions", "add_initial_seg_interaction", "add_point_interaction"]

def test_budgets(tmp_path, monkeypatch):
    """Decision on finding #3: every seg_* call counts (views/sheets included); when the
    tool-call budget is exhausted EVERY op except finish is rejected with a 'call
    seg_finish now' error; rejections are logged; counters show in the error text."""
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch, budgets={"max_tool_calls": 3, "max_model_prompts": 10})
    segd.ops.run_op(cs, "open", {})                 # 1
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": 40, "marks": "fine"})   # 2
    segd.ops.run_op(cs, "sheet", {"axis": "axial", "start": 10, "stop": 20})       # 3
    with pytest.raises(OpError, match=r"tool call budget exhausted \(3/3\).*call seg_finish now"):
        segd.ops.run_op(cs, "point", {"ids": [1]})
    # non-mutating ops are blocked too (only finish remains)
    with pytest.raises(OpError, match="call seg_finish now"):
        segd.ops.run_op(cs, "view", {"axis": "axial", "slice": 41, "marks": "fine"})
    assert cs.tool_calls == 3                        # rejections do not grow past the cap
    recs = [json.loads(l) for l in open(cs.out_dir / "trajectory.jsonl")]
    assert [r["error_kind"] for r in recs[-2:]] == ["budget", "budget"]
    assert "call seg_finish now" in recs[-1]["error"]
    segd.ops.run_op(cs, "finish", {"confidence": "medium", "reason": "budget test"})
    with pytest.raises(OpError, match="case is finished"):
        segd.ops.run_op(cs, "view", {"axis": "axial", "slice": 40, "marks": "fine"})
    recs = [json.loads(l) for l in open(cs.out_dir / "trajectory.jsonl")]
    assert recs[-1]["error_kind"] == "op" and "finished" in recs[-1]["error"]

def test_model_prompt_budget(tmp_path, monkeypatch):
    """The model-prompt budget blocks only prompts; other ops still work, the rejected
    call still counts as a tool call, and it is logged (decision on finding #3)."""
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch, budgets={"max_tool_calls": 10, "max_model_prompts": 2})
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": lesion_center(lesion)[2], "marks": "fine"})
    mid = _lesion_mark_id(cs, lesion)
    segd.ops.run_op(cs, "point", {"ids": [mid]})
    segd.ops.run_op(cs, "point", {"ids": [mid]})
    assert cs.model_prompts == 2
    with pytest.raises(OpError, match=r"model prompt budget used; edit with seg_paint/seg_edit or call seg_finish"):
        segd.ops.run_op(cs, "point", {"ids": [mid]})
    assert cs.model_prompts == 2
    assert cs.tool_calls == 4                        # view, 2 points, the rejected point
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": 41, "marks": "fine"})   # ... still works
    assert cs.tool_calls == 5
    recs = [json.loads(l) for l in open(cs.out_dir / "trajectory.jsonl")]
    assert recs[-2]["error_kind"] == "budget" and "model prompt budget used" in recs[-2]["error"]

def test_paint_add_erase(tmp_path, monkeypatch):
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch, z_halfwidth=0.25)
    ids, _centre = add_ring_marks(cs, lesion_center(lesion))
    k = lesion_center(lesion)[2]
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": k, "marks": "fine"})
    out = segd.ops.run_op(cs, "paint", {"ids": ids, "mode": "add", "slice_range": [k - 2, k + 2]})
    m = cs.current_mask()
    assert m[:, :, k - 2].any() and m[:, :, k + 2].any() and m[:, :, k - 3].sum() == 0
    assert "paint add on 5 slice(s)" in out.text
    before = (m > 0).sum()
    segd.ops.run_op(cs, "paint", {"ids": ids, "mode": "erase", "slice_range": [k - 2, k + 2]})
    assert cs.current_mask().sum() == 0
    assert before > 0
    assert cs._nni_dirty is True

def test_edit_ops(tmp_path, monkeypatch):
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    # a holed lesion + a distant separate blob
    holed = lesion.astype(np.uint8)
    c = lesion_center(lesion)
    holed[c[0] - 2:c[0] + 3, c[1] - 2:c[1] + 3, c[2] - 2:c[2] + 3] = 0
    blob = np.zeros(cs.shape, np.uint8)
    blob[10:14, 10:14, 10:12] = 1
    cs.commit_mask(holed | blob)
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": c[2], "marks": "fine"})
    mid = _lesion_mark_id(cs, lesion)
    before = float((cs.current_mask() > 0).sum())
    segd.ops.run_op(cs, "edit", {"op": "fill_holes"})
    assert (cs.current_mask() > 0).sum() > before
    # keep_component at the lesion mark: the distant blob disappears
    segd.ops.run_op(cs, "edit", {"op": "keep_component", "mark": mid})
    m = cs.current_mask()
    assert m[10:14, 10:14, 10:12].sum() == 0
    assert dice(m, lesion) > 0.95
    # remove_component at the lesion mark: everything except that component
    segd.ops.run_op(cs, "checkout", {"version": "v1"})   # holed | blob
    segd.ops.run_op(cs, "edit", {"op": "remove_component", "mark": mid})
    m = cs.current_mask()
    assert (m.astype(bool) & lesion).sum() < 10
    assert m[10:14, 10:14, 10:12].all()
    # morphology: dilate grows, erode shrinks back below
    segd.ops.run_op(cs, "checkout", {"version": "v1"})
    n0 = float((cs.version_mask(1) > 0).sum())
    segd.ops.run_op(cs, "edit", {"op": "dilate", "radius_mm": 2.0})
    assert (cs.current_mask() > 0).sum() > n0
    segd.ops.run_op(cs, "edit", {"op": "erode", "radius_mm": 3.0})
    assert (cs.current_mask() > 0).sum() < n0
    # radius_mm defaults to config.EDIT_DEFAULT_RADIUS_MM (the seg_edit tool documents it)
    out_close = segd.ops.run_op(cs, "edit", {"op": "close"})
    assert "edit close" in out_close.text
    with pytest.raises(OpError, match="radius_mm"):
        segd.ops.run_op(cs, "edit", {"op": "close", "radius_mm": "wide"})
    with pytest.raises(OpError, match="needs a mark"):
        segd.ops.run_op(cs, "edit", {"op": "keep_component"})

def test_crop_offset_round_trip_through_ops(tmp_path, monkeypatch):
    """Forced upload crop: point prompt translated into session space lands back on the
    lesion in full space (Dice > 0.9); FakeNNI received the cropped image."""
    install_stubs(monkeypatch, force=True)
    monkeypatch.setattr(config, "UPLOAD_MAX_PACKED_BYTES", 1)   # force the crop path
    cs, lesion = make_case_state(tmp_path, monkeypatch, crop_forced=False)
    assert cs.crop_offset != (0, 0, 0)
    assert cs.nni.image.shape == cs.crop_shape
    add_ring_marks(cs, lesion_center(lesion))
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": lesion_center(lesion)[2], "marks": "fine"})
    mid = _lesion_mark_id(cs, lesion)
    segd.ops.run_op(cs, "point", {"ids": [mid]})
    assert dice(cs.current_mask(), lesion) > 0.9

def test_full_upload_when_small(tmp_path, monkeypatch):
    install_stubs(monkeypatch, force=True)
    cs, _lesion = make_case_state(tmp_path, monkeypatch)
    assert cs.crop_offset == (0, 0, 0)
    assert cs.crop_shape == cs.shape
    assert cs.nni.image.shape == cs.shape

def test_lasso_scribble_box_offline(tmp_path, monkeypatch):
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch, z_halfwidth=0.08)   # thin lesion
    ids, _centre = add_ring_marks(cs, lesion_center(lesion))
    k = lesion_center(lesion)[2]
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": k, "marks": "fine"})

    out = segd.ops.run_op(cs, "lasso", {"ids": ids})
    assert dice(cs.current_mask(), lesion) > 0.8
    assert "lasso add" in out.text

    segd.ops.run_op(cs, "checkout", {"version": "v0"})
    out = segd.ops.run_op(cs, "scribble", {"ids": [ids[0], ids[2]]})
    assert dice(cs.current_mask(), lesion) > 0.5

    segd.ops.run_op(cs, "checkout", {"version": "v0"})
    segd.ops.run_op(cs, "box", {"id_a": ids[0], "id_b": ids[2]})
    m = cs.current_mask()
    assert m.sum() > 0
    # a 2D box acts on the active slice only (real 3D growth is the model's job)
    fixed = config.FIXED_AXIS["axial"]
    assert m.any(axis=tuple(i for i in range(3) if i != fixed)).sum() == 1
    slice_dice = dice(m[:, :, k], lesion[:, :, k])
    assert slice_dice > 0.7

    with pytest.raises(OpError, match="at least 3"):
        segd.ops.run_op(cs, "lasso", {"ids": ids[:2]})
    with pytest.raises(OpError, match="at least 2"):
        segd.ops.run_op(cs, "scribble", {"ids": [ids[0]]})

def test_refine_and_finish(tmp_path, monkeypatch):
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    add_ring_marks(cs, lesion_center(lesion))
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": lesion_center(lesion)[2], "marks": "fine"})
    mid = _lesion_mark_id(cs, lesion)
    with pytest.raises(OpError, match="empty"):
        segd.ops.run_op(cs, "refine", {})
    segd.ops.run_op(cs, "point", {"ids": [mid]})
    before = dice(cs.current_mask(), lesion)
    out = segd.ops.run_op(cs, "refine", {})
    assert "refine" in out.text
    assert dice(cs.current_mask(), lesion) > 0.9

    # invalid confidence is rejected BEFORE the case is finished
    with pytest.raises(OpError, match="confidence"):
        segd.ops.run_op(cs, "finish", {"confidence": "sure"})
    out = segd.ops.run_op(cs, "finish", {"confidence": "high", "reason": "looks complete"})
    assert out.done is False and "Final review of" in out.text      # first call = review
    import re as _re
    code = _re.search(r'review_code "(\d{4})"', out.text).group(1)
    out = segd.ops.run_op(cs, "finish", {"confidence": "high", "reason": "looks complete", "review_code": code})
    assert out.done is True
    assert (cs.out_dir / "final_mask.nii.gz").exists()
    assert (cs.out_dir / "versions.npz").exists()
    meta = (cs.out_dir / "case_meta.json").read_text()
    assert "confidence" in meta and "high" in meta
    # the final mask kept the original geometry
    import nibabel as nib
    im = nib.load(str(cs.out_dir / "final_mask.nii.gz"))
    assert im.shape == cs.shape
    assert np.allclose(im.affine, np.eye(4))
    with pytest.raises(OpError, match="case is finished"):
        segd.ops.run_op(cs, "point", {"ids": [mid]})

def test_trajectory_log_content(tmp_path, monkeypatch):
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    segd.ops.run_op(cs, "open", {})
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": 40, "marks": "fine"})
    lines = [json.loads(l) for l in open(cs.out_dir / "trajectory.jsonl")]
    assert [r["op"] for r in lines] == ["open", "view"]
    assert lines[0]["observation"] and lines[0]["timings"]["total"] >= 0
    assert lines[0]["renders"] == ["renders/001_open.png"]
    assert (cs.out_dir / "renders" / "001_open.png").exists()
    assert lines[1]["args"]["axis"] == "axial"


def test_model_prompt_failure_raises_operror_and_marks_dirty(tmp_path, monkeypatch):
    """Finding #2: any exception from a model prompt leaves the mask unchanged, marks nni
    dirty (so the next prompt re-syncs), does not consume the prompt budget, and surfaces
    as an agent-facing OpError with the generic model-error text."""
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    add_ring_marks(cs, lesion_center(lesion))
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": lesion_center(lesion)[2], "marks": "fine"})
    mid = _lesion_mark_id(cs, lesion)

    def boom(*a, **k):
        raise RuntimeError("model exploded")
    cs.nni.add_point_interaction = boom

    with pytest.raises(OpError, match=r"segmentation model error; the mask is unchanged, try again"):
        segd.ops.run_op(cs, "point", {"ids": [mid]})
    assert cs.current == 0                     # no version committed
    assert cs.model_prompts == 0               # failed prompt not counted (decision #3)
    assert cs.tool_calls == 2                  # view + the failing point both count
    assert cs._nni_dirty is True
    recs = [json.loads(l) for l in open(cs.out_dir / "trajectory.jsonl")]
    assert recs[-1]["error_kind"] == "op" and "model error" in recs[-1]["error"]


def test_model_prompt_reconnects_once_on_session_expiry(tmp_path, monkeypatch):
    """Finding #2: SessionExpiredError -> reconnect once (fresh session: image re-uploaded,
    new target buffer) and retry; the current mask is re-sent as the initial seg."""
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    add_ring_marks(cs, lesion_center(lesion))
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": lesion_center(lesion)[2], "marks": "fine"})
    mid = _lesion_mark_id(cs, lesion)
    segd.ops.run_op(cs, "point", {"ids": [mid]})                    # v1, healthy prompt

    from nnInteractive.inference.remote import SessionExpiredError
    from segd import fake_nni
    old_session = cs.nni
    initial_segs = []

    class FreshRecordingNNI(FakeNNI):
        def add_initial_seg_interaction(self, mask, run_prediction=False, **k):
            initial_segs.append(np.array(mask))
            return super().add_initial_seg_interaction(mask, run_prediction=run_prediction, **k)

    monkeypatch.setattr(fake_nni, "FakeNNI", FreshRecordingNNI)     # used by _reconnect

    def expired(coord, include_interaction, run_prediction=True, **k):
        raise SessionExpiredError("lease expired")
    cs.nni.add_point_interaction = expired                          # the OLD session only

    out = segd.ops.run_op(cs, "point", {"ids": [mid]})
    assert cs.nni is not old_session and isinstance(cs.nni, FreshRecordingNNI)
    assert cs.model_prompts == 2 and cs.current == 2                # retried prompt counted once
    assert len(initial_segs) == 1 and initial_segs[0].shape == cs.crop_shape
    assert dice(cs.current_mask(), lesion) > 0.9
    recs = [json.loads(l) for l in open(cs.out_dir / "trajectory.jsonl")]
    assert recs[-1]["error"] is None and recs[-1]["op"] == "point"


def test_point_outside_crop_rejected(tmp_path, monkeypatch):
    """Finding #6: a mark resolved onto a slice whose projected point falls outside the
    uploaded crop is a clean agent-facing error (the server would answer a 500)."""
    install_stubs(monkeypatch, force=True)
    monkeypatch.setattr(config, "UPLOAD_MAX_PACKED_BYTES", 1)   # force the crop path
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    assert cs.crop_offset != (0, 0, 0)
    c = lesion_center(lesion)
    # inject a mark at i=0 (outside the crop, which starts at crop_offset[0] >= 1 here)
    ms = cs.markset
    bad = StubMark(9001, "region", (0, c[1], c[2]), 5.0, "fine", strength=2.0)
    cs.markset = StubMarkSet(list(ms.marks.values()) + [bad], shape=ms.shape)
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": c[2], "marks": "fine"})
    with pytest.raises(OpError, match="that mark lies outside the scanned body region on this slice"):
        segd.ops.run_op(cs, "point", {"ids": [9001]})
    recs = [json.loads(l) for l in open(cs.out_dir / "trajectory.jsonl")]
    assert recs[-1]["error_kind"] == "op"


def test_clip_box_to_crop_rejects_fully_outside_box(tmp_path, monkeypatch):
    install_stubs(monkeypatch, force=True)
    cs, _lesion = make_case_state(tmp_path, monkeypatch, crop_forced=True)
    with pytest.raises(OpError, match="outside the scanned body region"):
        segd.ops._clip_box_to_crop(cs, ((0, 1), (0, 2), (0, 3)))


def test_change_image_is_3plane_with_spacing_and_timing(tmp_path, monkeypatch):
    """Mutating ops return the three-plane change render with the case spacing; the
    observation names the slices; the trajectory records a real render time."""
    install_stubs(monkeypatch, force=True)
    import segd.render
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    c = lesion_center(lesion)
    ids, _centre = add_ring_marks(cs, c, axis="coronal")     # ring visible on the coronal view
    segd.ops.run_op(cs, "view", {"axis": "coronal", "slice": c[1], "marks": "fine"})
    recorded = {}
    orig = segd.render.render_change_3plane

    def rec(image, prev, new, **k):
        recorded["spacing"] = k.get("spacing")
        return orig(image, prev, new, **k)

    monkeypatch.setattr(segd.render, "render_change_3plane", rec)
    out = segd.ops.run_op(cs, "paint", {"ids": ids, "mode": "add"})
    assert recorded["spacing"] == cs.spacing
    assert "image: axial " in out.text and "coronal " in out.text and "sagittal " in out.text
    assert "through mask centre" in out.text
    recs = [json.loads(l) for l in open(cs.out_dir / "trajectory.jsonl")]
    assert recs[-1]["timings"]["render"] is not None and recs[-1]["timings"]["render"] >= 0
    # undo/checkout use the same feedback image, independent of any active view
    cs2, _ = make_case_state(tmp_path, monkeypatch, case_id="phantom2")
    segd.ops.run_op(cs2, "open", {})
    m = np.zeros(cs2.shape, np.uint8); m[:, :, 40] = 1
    cs2.commit_mask(m)
    out2 = segd.ops.run_op(cs2, "undo", {})
    assert "image: axial " in out2.text


def test_undo_and_checkout_restored_text(tmp_path, monkeypatch):
    """Finding #8: 'version v0 (restored; was v1)'; the change is diffed against the
    previously current version (re-checkout of the current version: 'change none')."""
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    add_ring_marks(cs, lesion_center(lesion))
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": lesion_center(lesion)[2], "marks": "fine"})
    mid = _lesion_mark_id(cs, lesion)
    segd.ops.run_op(cs, "point", {"ids": [mid]})                       # v1 = lesion
    out = segd.ops.run_op(cs, "undo", {})
    assert "version v0 (restored; was v1)" in out.text
    assert "change +0.0 mL / -" in out.text                            # v1 removed, not "none"
    out = segd.ops.run_op(cs, "checkout", {"version": "v1"})
    assert "version v1 (restored; was v0)" in out.text
    assert "change +" in out.text
    out = segd.ops.run_op(cs, "checkout", {"version": "v1"})           # current -> current
    assert "version v1 (restored; was v1)" in out.text
    assert "change none" in out.text                                   # was the whole mask


def test_view_crop_marks_needs_ids_from_current_view(tmp_path, monkeypatch):
    """crop='marks' needs >=1 mark id visible in the CURRENT view; without ids, with an id
    that is not shown, or with no active view it is an agent-facing error."""
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    c = lesion_center(lesion)
    with pytest.raises(OpError, match="no active view"):
        segd.ops.run_op(cs, "view", {"axis": "axial", "slice": c[2], "crop": "marks", "ids": [1]})
    ids, _centre = add_ring_marks(cs, c, axis="axial")
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": c[2], "marks": "fine"})
    with pytest.raises(OpError, match="needs ids"):
        segd.ops.run_op(cs, "view", {"axis": "axial", "slice": c[2], "crop": "marks"})
    hidden = _hidden_fine_id(cs, lesion)
    with pytest.raises(OpError, match="not shown in the current view"):
        segd.ops.run_op(cs, "view", {"axis": "axial", "slice": c[2], "crop": "marks",
                                     "ids": [hidden]})
    # failed calls did not replace the active view
    assert cs.active_view.marks_level == "fine" and cs.active_view.box2d is None


def test_view_crop_marks_box_margin_and_minimum(tmp_path, monkeypatch):
    """The crop is the bbox of the given marks' in-plane positions + 25 mm, at least
    60x60 mm, on the requested slice; 'coarse' is upgraded to fine in that crop."""
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    c = lesion_center(lesion)
    ids, centre_id = add_ring_marks(cs, c, dx=10, dy=8, axis="axial")
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": c[2], "marks": "fine"})
    out = segd.ops.run_op(cs, "view", {"axis": "axial", "slice": c[2], "crop": "marks",
                                       "ids": ids, "marks": "coarse"})
    r0, c0, r1, c1 = cs.active_view.box2d
    rc = [voxel_to_inplane("axial", cs.markset.marks[i].ijk, cs.shape) for i in ids]
    exp = (min(r for r, _ in rc) - 25, min(q for _, q in rc) - 25,
           max(r for r, _ in rc) + 1 + 25, max(q for _, q in rc) + 1 + 25)
    assert (r0, c0, r1, c1) == exp
    assert "crop marks" in out.text and "marks fine" in out.text
    # a single mark -> the 60x60 mm minimum (1 mm voxels), containing the mark
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": c[2], "marks": "fine"})
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": c[2], "crop": "marks", "ids": [centre_id]})
    r0, c0, r1, c1 = cs.active_view.box2d
    assert r1 - r0 >= 60 and c1 - c0 >= 60
    mr, mc = voxel_to_inplane("axial", cs.markset.marks[centre_id].ijk, cs.shape)
    assert r0 <= mr < r1 and c0 <= mc < c1
    # an edge id is accepted too
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": c[2], "marks": "edges"})
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": c[2], "crop": "marks", "ids": ["E1"]})
    assert cs.active_view.box2d is not None


def test_view_full_fine_says_fine_appears_in_crops_when_nothing_added(tmp_path, monkeypatch):
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    # slice 5 shows no fine mark in the stub set -> fine adds nothing there
    out = segd.ops.run_op(cs, "view", {"axis": "axial", "slice": 12, "marks": "fine"})
    drawn = [m for m in cs.markset.marks.values() if m.id in
             {mid for mid, *_ in [(x[0],) for x in cs.active_view.render_info.marks]}]
    adds_nothing = all(m.level == "coarse" for m in drawn)
    assert ("fine marks appear in crops" in out.text) == adds_nothing


def test_final_mask_written_in_original_orientation(tmp_path, monkeypatch):
    """Finding #14: a source NIfTI in a non-RAS (LPS) orientation — final_mask.nii.gz
    must be written back in the ORIGINAL geometry (source affine + voxel order) via the
    inverse nibabel orientation transform, so a scorer canonicalising it again recovers
    the canonical mask."""
    install_stubs(monkeypatch, force=False)
    import nibabel as nib
    from nibabel.orientations import (
        apply_orientation, axcodes2ornt, io_orientation, ornt_transform,
    )
    from segd import data as data_mod
    from segd.case_state import CaseState
    from _helpers import make_phantom

    img_canon, lesion, _body = make_phantom(shape=(48, 48, 32))
    lps = np.diag([-1.0, -1.0, 1.0, 1.0]).astype(float)
    lps[:3, 3] = [90.0, 70.0, 0.0]
    ornt = io_orientation(lps)
    inv = ornt_transform(axcodes2ornt(("R", "A", "S")), ornt)
    src = apply_orientation(img_canon, inv)                 # what the LPS file stores
    src_path = tmp_path / "lps.nii.gz"
    nib.save(nib.Nifti1Image(src, lps), str(src_path))
    row = data_mod.CaseRow("LPSCASE", "M", "T2", False, "design", src_path, [])
    monkeypatch.setattr(data_mod, "get_case",
                        lambda case: row if case == "LPSCASE" else (_ for _ in ()).throw(KeyError(case)))

    g = data_mod.load_case_geometry(row)
    assert np.allclose(g.image, img_canon, atol=1e-5)       # canonicalised correctly at load
    assert g.orig_affine is not None and np.allclose(g.orig_affine, lps)
    cs = CaseState("LPSCASE", g.image, g.affine, g.spacing, row.target_text, row.sequence,
                   tmp_path / "out" / "LPSCASE", nni_url=None,
                   orig_geometry=(g.orig_affine, g.inv_ornt))
    cs.initialize()
    cs.commit_mask(lesion.astype(np.uint8))
    cs.finish("high", "orientation test")
    out = nib.load(str(cs.out_dir / "final_mask.nii.gz"))
    assert np.allclose(out.affine, lps)                     # source affine, not canonical
    expected = apply_orientation(lesion.astype(np.uint8), g.inv_ornt)
    assert ((np.asarray(out.dataobj) > 0) == (expected > 0)).all()   # original voxel order
    canon_back = nib.as_closest_canonical(out)               # a scorer canonicalising again...
    assert ((np.asarray(canon_back.dataobj) > 0) == (lesion > 0)).all()   # ...recovers the mask
    assert np.allclose(canon_back.affine, g.affine)


def test_final_mask_canonical_when_source_is_ras(tmp_path, monkeypatch):
    """RAS sources (all MSTT files) keep the previous behaviour: canonical affine,
    canonical voxel order, no orig geometry."""
    import nibabel as nib
    from segd import data as data_mod
    from segd.case_state import CaseState
    from _helpers import make_phantom
    img_canon, lesion, _body = make_phantom(shape=(48, 48, 32))
    src_path = tmp_path / "ras.nii.gz"
    nib.save(nib.Nifti1Image(img_canon, np.eye(4)), str(src_path))
    row = data_mod.CaseRow("RASCASE", "M", "T2", False, "design", src_path, [])
    g = data_mod.load_case_geometry(row)
    assert g.orig_affine is None and g.inv_ornt is None
    cs = CaseState("RASCASE", g.image, g.affine, g.spacing, row.target_text, row.sequence,
                   tmp_path / "out" / "RASCASE")
    cs.initialize()
    cs.commit_mask(lesion.astype(np.uint8))
    cs.finish("high", "ras test")
    out = nib.load(str(cs.out_dir / "final_mask.nii.gz"))
    assert np.allclose(out.affine, np.eye(4))
    assert ((np.asarray(out.dataobj) > 0) == (lesion > 0)).all()
