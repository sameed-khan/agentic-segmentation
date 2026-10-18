"""Two-stage finish (final review): review -> confirm, re-review after edits, budget and
skip_review bypasses, review image size limits, components text."""
from __future__ import annotations
import re

import numpy as np
import pytest

import segd.ops
from segd import render as R
from segd.observe import components_text
from segd.case_state import OpError

from _helpers import add_ring_marks, install_stubs, lesion_center, make_case_state


def _make_version(cs, lesion):
    from test_core_ops import _lesion_mark_id
    mid = _lesion_mark_id(cs, lesion)
    segd.ops.run_op(cs, "point", {"ids": [mid]})


def test_review_then_confirm_finishes(tmp_path, monkeypatch):
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    add_ring_marks(cs, lesion_center(lesion))
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": lesion_center(lesion)[2], "marks": "fine"})
    _make_version(cs, lesion)
    out = segd.ops.run_op(cs, "finish", {"confidence": "low", "reason": "first"})
    assert not out.done and not cs.done and len(out.images) == 1
    assert out.text.startswith(f"Final review of v{cs.current}: volume ")
    assert "components 1 (largest 100.0%)" in out.text and "tool calls" in out.text
    assert "call seg_finish again with review_code" in out.text
    code = re.search(r'review_code "(\d{4})"', out.text).group(1)
    calls = cs.tool_calls
    # a confirmation without (or with a wrong) code does not save
    out2 = segd.ops.run_op(cs, "finish", {"confidence": "high", "reason": "parallel"})
    assert not out2.done and "Not saved" in out2.text and code not in out2.text
    out2 = segd.ops.run_op(cs, "finish", {"confidence": "high", "reason": "x", "review_code": "0000"})
    assert not out2.done
    out = segd.ops.run_op(cs, "finish", {"confidence": "high", "reason": "second", "review_code": code})
    assert out.done and cs.done and cs.tool_calls == calls + 3
    assert '"confidence": "high"' in (cs.out_dir / "case_meta.json").read_text().replace("'", '"')
    assert "second" in (cs.out_dir / "case_meta.json").read_text()


def test_edit_after_review_requires_new_review(tmp_path, monkeypatch):
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    add_ring_marks(cs, lesion_center(lesion))
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": lesion_center(lesion)[2], "marks": "fine"})
    _make_version(cs, lesion)
    assert not segd.ops.run_op(cs, "finish", {"confidence": "high"}).done
    segd.ops.run_op(cs, "undo", {})                       # version changes
    out = segd.ops.run_op(cs, "finish", {"confidence": "high"})
    assert not out.done and out.text.startswith("Final review of v0") and "empty" in out.text
    code = re.search(r'review_code "(\d{4})"', out.text).group(1)
    assert segd.ops.run_op(cs, "finish", {"confidence": "high", "review_code": code}).done


def test_budget_exhausted_finish_is_immediate(tmp_path, monkeypatch):
    install_stubs(monkeypatch, force=True)
    cs, _ = make_case_state(tmp_path, monkeypatch, budgets={"max_tool_calls": 2, "max_model_prompts": 5})
    segd.ops.run_op(cs, "open", {})
    segd.ops.run_op(cs, "open", {})
    out = segd.ops.run_op(cs, "finish", {"confidence": "low", "reason": "budget"})
    assert out.done and cs.done


def test_skip_review_finishes_immediately(tmp_path, monkeypatch):
    install_stubs(monkeypatch, force=True)
    cs, _ = make_case_state(tmp_path, monkeypatch)
    with pytest.raises(OpError, match="boolean"):
        segd.ops.run_op(cs, "finish", {"confidence": "low", "skip_review": "yes"})
    out = segd.ops.run_op(cs, "finish", {"confidence": "low", "reason": "forced", "skip_review": True})
    assert out.done and (cs.out_dir / "final_mask.nii.gz").exists()


@pytest.mark.parametrize("spacing", [(1.0, 1.0, 1.0), (0.5, 0.5, 5.0), (3.0, 1.0, 1.0)])
def test_review_grid_size_limits(spacing):
    rng = np.random.default_rng(0)
    shape = (140, 120, 90)
    img = rng.normal(300, 20, shape).astype(np.float32)
    mask = np.zeros(shape, bool)
    mask[40:100, 30:70, 20:60] = True
    out, info = R.render_review_grid(img, mask, spacing)
    assert out.width <= 1568 and out.height <= 1568
    assert out.width * out.height <= 1.25e6
    assert len(info.views) == 15
    assert [v.axis for v in info.views] == ["axial"] * 5 + ["coronal"] * 5 + ["sagittal"] * 5
    # 8 mm before the first slice, 25/50/75% of the extent, 8 mm after the last slice
    k = int(np.ceil(8.0 / spacing[2]))
    assert [v.slice_idx for v in info.views[:5]] == [20 - k, 30, 40, 49, 59 + k]
    # tiny mask: scale is capped, still within limits
    mask[:] = False
    mask[60:62, 60:62, 40:42] = True
    out, _ = R.render_review_grid(img, mask, spacing)
    assert out.width <= 1568 and out.height <= 1568
    # empty mask -> overview-style three planes
    out, info = R.render_review_grid(img, np.zeros(shape, bool), spacing)
    assert len(info.views) == 3 and out.width <= 1568


def test_review_grid_past_mask_labels_and_clamping():
    shape = (60, 60, 60)
    img = np.full(shape, 300, np.float32)
    mask = np.zeros(shape, bool)
    mask[20:40, 20:40, 30:60] = True          # axial extent reaches the top of the volume
    out, info = R.render_review_grid(img, mask, (1.0, 1.0, 1.0))
    assert out.width <= 1568 and out.height <= 1568
    ax, cor, sag = info.views[:5], info.views[5:10], info.views[10:]
    assert [v.slice_idx for v in ax] == [22, 37, 44, 52, 59]     # last clamped onto mask end
    assert [v.slice_idx for v in cor] == [12, 25, 30, 34, 47]
    assert [v.slice_idx for v in sag] == [12, 25, 30, 34, 47]
    assert R._review_slices("axial", 30, 60, 60, 1.0)[0][1] == "8 mm below mask"
    assert R._review_slices("axial", 30, 60, 60, 1.0)[-1][1] == "volume edge, mask reaches it"
    assert R._review_slices("axial", 30, 58, 60, 1.0)[-1] == (59, "2 mm above mask (volume edge)")
    assert R._review_slices("coronal", 20, 40, 60, 1.0)[0] == (12, "8 mm posterior to mask")
    assert R._review_slices("coronal", 20, 40, 60, 1.0)[-1] == (47, "8 mm anterior to mask")
    assert R._review_slices("sagittal", 20, 40, 60, 1.0)[0][1] == "8 mm left of mask"
    assert R._review_slices("sagittal", 20, 40, 60, 1.0)[-1][1] == "8 mm right of mask"
    assert R._review_slices("axial", 0, 10, 60, 2.0)[0] == (0, "volume edge, mask reaches it")
    # past cells have an orange frame pixel, in-mask cells do not
    px = np.asarray(out)
    orange = np.all(px == R.REVIEW_PAST_COLOR, axis=-1)
    assert orange.any()


def test_components_text():
    st = {"n_components": 12, "largest_frac": 0.9996, "volume_ml": 100.0}
    assert components_text(st) == "components 12 (largest 99.9%; 11 others total 0.0 mL)"
    st = {"n_components": 12, "largest_frac": 0.976, "volume_ml": 100.0}
    assert components_text(st) == "components 12 (largest 97.6%; 11 others total 2.4 mL)"
    assert components_text({"n_components": 1, "largest_frac": 1.0, "volume_ml": 5.0}) == \
        "components 1 (largest 100.0%)"
    assert components_text({"n_components": 0, "largest_frac": 0.0, "volume_ml": 0.0}) == "components 0"
