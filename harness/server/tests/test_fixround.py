"""Fix-round regressions: mark-cache validation + atomic write, reconnect after a failed
reconnect, all-or-nothing mutating ops, unknown-op accounting, budget display, tiny-body
volumes, and the mark_coverage --no-save-cache flag."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import segd.ops
from segd import marks as M
from segd.case_state import OpError, drawn_global_ids
from segd.fake_nni import FakeNNI
from _helpers import (
    add_ring_marks, install_stubs, lesion_center, make_case_state,
)

SMALL = (48, 48, 32)


def _count_compute(monkeypatch):
    calls = []
    orig = M.compute_marks

    def counting(image, spacing, timings=None):
        calls.append(image.shape)
        return orig(image, spacing, timings=timings)

    monkeypatch.setattr(M, "compute_marks", counting)
    return calls


def test_cache_validated_against_version_params_and_shape(tmp_path, monkeypatch):
    install_stubs(monkeypatch, force=False)
    calls = _count_compute(monkeypatch)
    make_case_state(tmp_path, monkeypatch, shape=SMALL)
    assert len(calls) == 1
    cache_dir = tmp_path / "results" / "cache" / "marks"
    assert (cache_dir / "phantom.npz").exists()
    assert not list(cache_dir.glob("*.tmp*")), "atomic write leaves no temp file"
    make_case_state(tmp_path, monkeypatch, shape=SMALL)
    assert len(calls) == 1, "matching cache is reused"
    # every fingerprint ingredient invalidates the cache and the recompute is written back
    for name, value in (("REGION_STRENGTH_MM", 11.0), ("REGION_SCALE_CLAMP_MM", (9.0, 40.0)),
                        ("BODY_GRID_MAX_DIM", 200), ("MARKS_VERSION", 99)):
        n = len(calls)
        monkeypatch.setattr(M, name, value)
        make_case_state(tmp_path, monkeypatch, shape=SMALL)
        assert len(calls) == n + 1, f"{name} change must force a recompute"
        make_case_state(tmp_path, monkeypatch, shape=SMALL)
        assert len(calls) == n + 1, f"{name}: recomputed cache is valid afterwards"
    # a different volume shape under the same case id is not trusted
    n = len(calls)
    make_case_state(tmp_path, monkeypatch, shape=(40, 44, 30))
    assert len(calls) == n + 1


def test_params_fingerprint_covers_region_and_body_grid_constants():
    snap = M._params_snapshot()
    for key in ("region_strength_mm", "region_scale_clamp_mm", "body_grid_max_dim", "version"):
        assert key in snap


def test_marks_save_is_atomic_on_failure(tmp_path, monkeypatch):
    img = np.zeros((20, 20, 20), np.float32)
    ms = M.MarkSet({1: M.Mark(1, "region", (1, 2, 3), 8.0, "coarse")}, (1, 1, 1), (0.0, 1.0),
                   {1: 0.5}, params=M._params_snapshot(), shape=img.shape)
    path = tmp_path / "c.npz"
    ms.save(path)
    before = path.read_bytes()

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez_compressed", boom)
    with pytest.raises(OSError):
        ms.save(path)
    assert path.read_bytes() == before and not list(tmp_path.glob("*.tmp*"))


def test_compute_marks_tiny_body_does_not_crash():
    """Review #5: blobs found but no supervoxels -> all-fine fallback, no AttributeError."""
    rng = np.random.default_rng(0)
    for edge in (8, 10, 14):
        img = rng.normal(0, 3, (40, 40, 40)).astype(np.float32)
        img[15:15 + edge, 15:15 + edge, 15:15 + edge] += 400.0
        ms = M.compute_marks(img, (1.0, 1.0, 1.0))
        assert isinstance(ms, M.MarkSet)


def test_reconnect_failure_is_not_permanent(tmp_path, monkeypatch):
    """Review #6: a failed reconnect leaves nni None; the next prompt rebuilds the session."""
    install_stubs(monkeypatch, force=True)
    from nnInteractive.inference.remote import SessionExpiredError
    from segd import fake_nni
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    c = lesion_center(lesion)
    add_ring_marks(cs, c)
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": c[2], "marks": "fine"})
    mid = sorted(drawn_global_ids(cs.active_view.render_info))[0]

    def expired(*a, **k):
        raise SessionExpiredError("expired")

    cs.nni.add_point_interaction = expired

    class Boom(FakeNNI):
        def __init__(self, *a, **k):
            raise RuntimeError("server at capacity")

    monkeypatch.setattr(fake_nni, "FakeNNI", Boom)
    with pytest.raises(OpError):
        segd.ops.run_op(cs, "point", {"ids": [mid]})
    assert cs.current == 0 and cs.nni is None
    monkeypatch.setattr(fake_nni, "FakeNNI", FakeNNI)         # server healthy again
    segd.ops.run_op(cs, "point", {"ids": [mid]})
    assert cs.current == 1 and cs.nni is not None


def _lesion_view(cs, lesion):
    c = lesion_center(lesion)
    ids, _ = add_ring_marks(cs, c)
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": c[2], "marks": "fine"})
    return ids


@pytest.mark.parametrize("op,args", [("paint", None), ("edit", {"op": "dilate"}),
                                     ("point", None)])
def test_render_failure_leaves_no_committed_version(tmp_path, monkeypatch, op, args):
    """Review #7: a failing feedback render must not leave the version committed (the
    error text says the mask is unchanged)."""
    install_stubs(monkeypatch, force=True)
    import segd.render
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    ids = _lesion_view(cs, lesion)
    if op == "edit":
        m = np.zeros(cs.shape, np.uint8)
        m[50:60, 50:60, 30:40] = 1
        cs.commit_mask(m)
    n_versions, cur = len(cs.versions), cs.current

    def broken(*a, **k):
        raise RuntimeError("render exploded")

    monkeypatch.setattr(segd.render, "render_change_3plane", broken)
    if op == "paint":
        a = {"ids": ids, "mode": "add"}
    elif op == "point":
        a = {"ids": [ids[0]]}
    else:
        a = args
    with pytest.raises(RuntimeError):
        segd.ops.run_op(cs, op, a)
    assert len(cs.versions) == n_versions and cs.current == cur
    # undo/checkout: current is restored too
    cs.commit_mask(np.ones(cs.shape, np.uint8))
    cur = cs.current
    with pytest.raises(RuntimeError):
        segd.ops.run_op(cs, "undo", {})
    assert cs.current == cur


def test_unknown_op_and_bad_args_count_and_are_logged(tmp_path, monkeypatch):
    install_stubs(monkeypatch, force=True)
    cs, _ = make_case_state(tmp_path, monkeypatch)
    with pytest.raises(OpError, match="unknown op"):
        segd.ops.run_op(cs, "frobnicate", {})
    with pytest.raises(OpError, match="JSON object"):
        segd.ops.run_op(cs, "view", ["not", "a", "dict"])
    assert cs.tool_calls == 2
    recs = [json.loads(l) for l in open(cs.out_dir / "trajectory.jsonl")]
    assert [r["op"] for r in recs[-2:]] == ["frobnicate", "view"]
    assert all(r.get("error") for r in recs[-2:])


def test_finish_after_budget_never_displays_over_max(tmp_path, monkeypatch):
    install_stubs(monkeypatch, force=True)
    cs, _ = make_case_state(tmp_path, monkeypatch, budgets={"max_tool_calls": 2})
    for _ in range(2):
        segd.ops.run_op(cs, "view", {"axis": "axial", "slice": 10})
    out = segd.ops.run_op(cs, "finish", {"confidence": "low", "reason": "budget"})
    assert "tool calls 2/2" in out.text and "3/2" not in out.text


def test_mark_coverage_no_save_cache_writes_nothing(tmp_path):
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("mark_coverage", root / "eval" / "v2" / "mark_coverage.py")
    mc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mc)
    rng = np.random.default_rng(1)
    img = rng.normal(0, 3, (32, 32, 32)).astype(np.float32)
    img[8:24, 8:24, 8:24] += 300
    cache = tmp_path / "cache"
    mc.load_or_compute_marks("X", img, (1.0, 1.0, 1.0), cache, use_cache=False, save_cache=False)
    assert not cache.exists() or not list(cache.glob("*"))
    mc.load_or_compute_marks("X", img, (1.0, 1.0, 1.0), cache, use_cache=False, save_cache=True)
    assert (cache / "X.npz").exists()
