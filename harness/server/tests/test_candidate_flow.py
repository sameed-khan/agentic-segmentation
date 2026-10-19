"""Candidate corrections in the ops layer (CONTRACTS deviation 'candidate corrections'):

seg_candidates / seg_accept flows on the FakeNNI phantom, the required candidate round in
the first seg_finish review, staleness/unknown-id errors, budget semantics, the 3-accept
review cap, trajectory logging, and render_candidates size limits (real renderer).
"""
from __future__ import annotations
import json
import re

import numpy as np
import pytest

import segd.ops
from segd import config
from segd import render as R
from segd.candidates import Candidate
from segd.case_state import OpError

from _helpers import dice, install_stubs, lesion_center, make_case_state


def _partial_lesion_mask(lesion) -> np.ndarray:
    """Superior half of the lesion as v1: the missing inferior half is adjacent, same
    intensity band -> the intensity generator proposes it (test-side GT knowledge)."""
    kk = np.indices(lesion.shape)[2]
    return (lesion & (kk >= lesion_center(lesion)[2])).astype(np.uint8)


def _inject_candidate(cs, region) -> None:
    """Test-side candidate for the current version (accept never checks how one was made)."""
    idx = np.nonzero(region)
    bbox = tuple(slice(int(a.min()), int(a.max()) + 1) for a in idx)
    crop = np.ascontiguousarray(region[bbox])
    cand = Candidate("C1", "add", bbox, crop, cs.shape, float(crop.sum()) / 1000.0,
                     {"method": "test"})
    cs.candidate_cache[cs.current] = [cand]
    cs.candidates_shown = cs.current


def _break_render_candidates(monkeypatch):
    """Inject a render_candidates exception (real render module, stubs already installed)."""
    def boom(*a, **k):
        raise RuntimeError("candidate render exploded")
    monkeypatch.setattr(R, "render_candidates", boom)


def test_review_candidates_accept_confirm_flow(tmp_path, monkeypatch):
    """finish -> review WITH candidates -> seg_accept C1 -> new version -> new review -> confirm."""
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    partial = _partial_lesion_mask(lesion)
    cs.commit_mask(partial)                                  # v1: half the lesion
    out = segd.ops.run_op(cs, "finish", {"confidence": "medium", "reason": "check"})
    assert not out.done and not cs.done
    assert len(out.images) == 2                              # review grid + candidate sheet
    assert re.search(r"C1 add [0-9.]+ mL", out.text)
    assert ("For each candidate decide whether the tissue inside its outline is part of the "
            "same lesion (accept) or normal tissue (reject). Accept with seg_accept and call "
            "seg_finish again for a new review; if none belong, confirm with the review code.") in out.text
    code = re.search(r'review_code "(\d{4})"', out.text).group(1)
    assert "tool calls" in out.text
    calls, prompts = cs.tool_calls, cs.model_prompts

    out = segd.ops.run_op(cs, "accept", {"ids": ["C1"]})
    assert cs.current == 2                                   # ONE new version
    assert cs.tool_calls == calls + 1 and cs.model_prompts == prompts   # tool call, not a prompt
    assert cs._nni_dirty                                     # non-model edit -> nni sync rule
    grown = cs.version_mask(2)
    assert dice(grown, lesion) > dice(partial, lesion)       # the accepted candidate added missed tissue
    assert "accept C1" in out.text and "image: axial" in out.text   # 3-plane change image like other edits

    out = segd.ops.run_op(cs, "finish", {"confidence": "medium", "reason": "check again"})
    assert not out.done and f"Final review of v{cs.current}" in out.text
    code2 = re.search(r'review_code "(\d{4})"', out.text).group(1)
    out = segd.ops.run_op(cs, "finish", {"confidence": "high", "reason": "ok", "review_code": code2})
    assert out.done and cs.done


def test_seg_candidates_op(tmp_path, monkeypatch):
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    with pytest.raises(OpError, match="empty"):
        segd.ops.run_op(cs, "candidates", {})
    cs.commit_mask(_partial_lesion_mask(lesion))
    out = segd.ops.run_op(cs, "candidates", {})
    assert re.search(r"C1 add [0-9.]+ mL", out.text) and len(out.images) == 1
    assert "seg_accept" in out.text and f"v{cs.current}" in out.text
    assert cs.candidates_shown == cs.current and cs.candidate_cache[cs.current]
    assert cs.current == 1 and cs.model_prompts == 0         # nothing changed by proposing


def test_seg_candidates_none_found_is_text_only(tmp_path, monkeypatch):
    install_stubs(monkeypatch, force=True)
    cs, _lesion = make_case_state(tmp_path, monkeypatch)
    cs.commit_mask(cs.body.astype(np.uint8))                 # the whole body: nothing adjacent in-band
    out = segd.ops.run_op(cs, "candidates", {})
    assert out.images == [] and "no candidates" in out.text


def test_stale_candidate_ids_error(tmp_path, monkeypatch):
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    cs.commit_mask(_partial_lesion_mask(lesion))
    segd.ops.run_op(cs, "candidates", {})
    cs.commit_mask(cs.body.astype(np.uint8))                 # any mask change invalidates them
    with pytest.raises(OpError, match="candidates are out of date; call seg_candidates again"):
        segd.ops.run_op(cs, "accept", {"ids": ["C1"]})
    with pytest.raises(OpError, match="out of date"):        # never shown at all
        segd.ops.run_op(cs, "accept", {"ids": ["C1"]})


def test_unknown_candidate_id_error(tmp_path, monkeypatch):
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    cs.commit_mask(_partial_lesion_mask(lesion))
    segd.ops.run_op(cs, "candidates", {})
    with pytest.raises(OpError, match="unknown candidate"):
        segd.ops.run_op(cs, "accept", {"ids": ["C9"]})
    with pytest.raises(OpError, match="unknown candidate"):
        segd.ops.run_op(cs, "accept", {"ids": ["C1", "C9"]})   # nothing is accepted


def test_candidates_and_accept_count_as_tool_calls_not_prompts(tmp_path, monkeypatch):
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    cs.commit_mask(_partial_lesion_mask(lesion))
    segd.ops.run_op(cs, "candidates", {})
    calls, prompts = cs.tool_calls, cs.model_prompts
    segd.ops.run_op(cs, "accept", {"ids": ["C1"]})
    assert cs.tool_calls == calls + 1 and cs.model_prompts == prompts


def test_review_cap_after_three_accepts(tmp_path, monkeypatch):
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    cs.commit_mask(_partial_lesion_mask(lesion))
    kk = np.indices(lesion.shape)[2]
    missing = lesion & (kk < lesion_center(lesion)[2])
    for _ in range(config.CANDIDATE_REVIEW_MAX_ACCEPT_ROUNDS):
        _inject_candidate(cs, missing)
        segd.ops.run_op(cs, "accept", {"ids": ["C1"]})
    assert cs.accept_rounds == config.CANDIDATE_REVIEW_MAX_ACCEPT_ROUNDS
    out = segd.ops.run_op(cs, "finish", {"confidence": "medium", "reason": "cap"})
    assert not out.done
    assert len(out.images) == 1                              # review grid only: no candidate image
    assert "none offered" in out.text and "review_code" in out.text
    out2 = segd.ops.run_op(cs, "candidates", {})             # seg_candidates itself still works
    assert "candidate" in out2.text


def test_trajectory_logs_candidates_and_accept(tmp_path, monkeypatch):
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    cs.commit_mask(_partial_lesion_mask(lesion))
    segd.ops.run_op(cs, "candidates", {})
    segd.ops.run_op(cs, "accept", {"ids": ["C1"]})
    lines = [json.loads(line) for line in
             (cs.out_dir / "trajectory.jsonl").read_text().splitlines()]
    cand = [r for r in lines if r["op"] == "candidates"][-1]
    acc = [r for r in lines if r["op"] == "accept"][-1]
    item = cand["candidates"]["items"][0]
    assert item["id"] == "C1" and item["kind"] == "add" and item["volume_ml"] > 0
    assert item["how"]["method"] == "intensity" and "seed_ijk" in item["how"]
    assert cand["candidates"]["version"] == "v1" and cand["timings"]["candidates"] >= 0
    assert cand["renders"]                                   # candidate sheet saved like other images
    assert acc["accepted"]["ids"] == ["C1"] and acc["accepted"]["from_version"] == "v1"
    assert acc["accepted"]["to_version"] == "v2" and acc["accepted"]["how"][0]["method"] == "intensity"
    assert acc["renders"]


@pytest.mark.parametrize("n", [1, 2, 5])
def test_render_candidates_size_limits(n):
    shape = (200, 200, 120)
    img = np.full(shape, 300.0, np.float32)
    mask = np.zeros(shape, bool)
    mask[80:100, 80:100, 50:60] = True
    cands = []
    for i in range(n):
        region = np.zeros(shape, bool)
        region[80:100, 100:110, 50 + i * 8:56 + i * 8] = True
        idx = np.nonzero(region)
        bbox = tuple(slice(int(a.min()), int(a.max()) + 1) for a in idx)
        cands.append(Candidate(f"C{i + 1}", "add", bbox, np.ascontiguousarray(region[bbox]),
                               shape, 1.0, {"method": "test"}))
    out, info = R.render_candidates(img, mask, cands, (1.0, 1.0, 1.0))
    assert out.width <= 1568 and out.height <= 1568
    assert out.width * out.height <= 1_200_000               # ~1.2 MP budget
    assert len(info.views) == 3 * n                          # axial/coronal/sagittal per candidate row
    px = np.asarray(out)
    assert np.all(px == R.CANDIDATE_COLOR, axis=-1).any()    # magenta outline drawn
    assert np.all(px == R.OUTLINE_COLOR, axis=-1).any()      # green mask outline drawn
    assert R.CANDIDATE_COLOR not in (R.OUTLINE_COLOR, R.REMOVED_COLOR)


def test_render_candidates_center_of_mass_slices_contain_candidate():
    """The columns cut through the candidate even when its centre of mass falls outside it:
    the candidates are non-convex (a C-shape whose COM lands in its mouth; a dumbbell whose
    COM slice holds no candidate voxels at all), yet every rendered slice intersects it."""
    shape = (60, 60, 60)
    img = np.full(shape, 300.0, np.float32)
    mask = np.zeros(shape, bool)
    mask[20:30, 20:30, 20:30] = True
    c_shape = np.zeros(shape, bool)
    c_shape[22:26, 22:26, 30:40] = True
    c_shape[23:25, 22:26, 33:37] = False            # the C's mouth: the COM falls inside it
    dumbbell = np.zeros(shape, bool)
    dumbbell[22:26, 22:26, 30:34] = True
    dumbbell[22:26, 22:26, 40:44] = True            # COM slice k~=36: no candidate voxels there
    regions = [("C1", c_shape), ("C2", dumbbell)]
    cands = []
    for cid, region in regions:
        idx = np.nonzero(region)
        com = tuple(int(round(float(a.mean()))) for a in idx)
        assert not region[com], f"{cid}: genuinely non-convex (the COM misses it)"
        bbox = tuple(slice(int(a.min()), int(a.max()) + 1) for a in idx)
        cands.append(Candidate(cid, "add", bbox, np.ascontiguousarray(region[bbox]), shape, 1.0,
                               {"method": "test"}))
    out, info = R.render_candidates(img, mask, cands, (1.0, 1.0, 1.0))
    assert len(info.views) == 6
    for (_cid, region), views in zip(regions, (info.views[:3], info.views[3:])):
        for v in views:
            sl = np.take(region, v.slice_idx, axis=config.FIXED_AXIS[v.axis])
            assert sl.any(), f"rendered {v.axis} slice {v.slice_idx} must intersect the candidate"
    ax, cor, sag = info.views[0], info.views[1], info.views[2]
    assert 30 <= ax.slice_idx < 40
    assert 22 <= cor.slice_idx < 26 and 22 <= sag.slice_idx < 26


def test_candidate_chips_avoid_orientation_letters(monkeypatch):
    """Fix round #4: candidate id chips must not cover the orientation letters.  Narrow
    panels (coarse spacing -> tiny cells) are where the old fixed placement (right of the
    slice chip) collided with the top letter; placement is now collision-aware against the
    frame's occupied rectangles."""
    shape = (200, 200, 120)
    img = np.full(shape, 300.0, np.float32)
    mask = np.zeros(shape, bool)
    mask[95:105, 95:105, 55:65] = True               # small mask: 20 mm margins stay coarse
    region = np.zeros(shape, bool)
    region[95:105, 105:113, 55:63] = True
    idx = np.nonzero(region)
    bbox = tuple(slice(int(a.min()), int(a.max()) + 1) for a in idx)
    cands = [Candidate("C1", "add", bbox, np.ascontiguousarray(region[bbox]), shape, 1.0,
                       {"method": "test"})]
    placements = []
    orig = R._draw_candidate_label

    def spy(panel, label, occupied):
        plate = orig(panel, label, occupied)
        placements.append((panel.width, plate, list(occupied)))
        return plate

    monkeypatch.setattr(R, "_draw_candidate_label", spy)
    out, info = R.render_candidates(img, mask, cands, (5.0, 5.0, 5.0))
    assert len(placements) == 3                      # axial/coronal/sagittal
    from PIL import Image, ImageDraw
    d = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    f = R._font(R.CANDIDATE_LABEL_PX)
    l0, t0, r0, b0 = d.textbbox((0, 0), "C1", font=f)
    collided = 0
    for w, plate, occupied in placements:
        assert len(occupied) > 1                     # slice chip + orientation letters drawn
        # precondition: the OLD fixed placement (right of the slice chip) really collided
        old = (occupied[0][2] + 8 + l0 - 4, t0 - 3, occupied[0][2] + 8 + r0 + 4, b0 + 3)
        collided += any(R._boxes_overlap(old, ob) for ob in occupied[1:])
        for ob in occupied:                          # the fix: the drawn chip clears everything
            assert not R._boxes_overlap(plate, ob), f"chip {plate} overlaps frame box {ob}"
        assert 0 <= plate[0] and plate[2] <= w       # and stays inside the panel
    assert collided >= 1, "precondition: at least the narrow panels collide under the old placement"


def test_authorization_does_not_survive_mask_round_trip(tmp_path, monkeypatch):
    """Fix round #1: candidates on v1 -> edit -> undo (or checkout) -> accept C1 must FAIL:
    authorization never survives a mask change / version round trip, while the per-version
    computation cache is kept."""
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    cs.commit_mask(_partial_lesion_mask(lesion))                 # v1
    segd.ops.run_op(cs, "candidates", {})
    assert cs.candidates_shown == cs.current
    segd.ops.run_op(cs, "edit", {"op": "dilate"})                # v2
    segd.ops.run_op(cs, "undo")                                  # back to v1
    assert 1 in cs.candidate_cache                               # computation cache kept
    with pytest.raises(OpError, match="candidates are out of date; call seg_candidates again"):
        segd.ops.run_op(cs, "accept", {"ids": ["C1"]})
    segd.ops.run_op(cs, "checkout", {"version": "v2"})
    segd.ops.run_op(cs, "checkout", {"version": "v1"})           # checkout round trip: same rule
    with pytest.raises(OpError, match="out of date"):
        segd.ops.run_op(cs, "accept", {"ids": ["C1"]})
    segd.ops.run_op(cs, "candidates", {})                        # cache hit: re-authorized instantly
    out = segd.ops.run_op(cs, "accept", {"ids": ["C1"]})
    assert cs.current == 3 and "accept C1" in out.text           # v2 existed (the edit), accept adds v3


def test_review_code_does_not_survive_mask_round_trip(tmp_path, monkeypatch):
    """Fix round #1: review v1 -> accept -> undo (or checkout) -> finish with the OLD code
    must NOT save: the case gets a new review and a new code first."""
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    cs.commit_mask(_partial_lesion_mask(lesion))                 # v1
    out = segd.ops.run_op(cs, "finish", {"confidence": "medium", "reason": "check"})
    code = re.search(r'review_code "(\d{4})"', out.text).group(1)
    assert cs.reviewed_version == cs.current
    segd.ops.run_op(cs, "accept", {"ids": ["C1"]})               # v2
    segd.ops.run_op(cs, "undo")                                  # back to v1
    out = segd.ops.run_op(cs, "finish", {"confidence": "high", "reason": "stale",
                                         "review_code": code})
    assert not out.done and not cs.done                          # NOT saved
    assert f"Final review of v{cs.current}" in out.text          # a new review was produced
    code2 = re.search(r'review_code "(\d{4})"', out.text).group(1)
    assert code2 != code
    segd.ops.run_op(cs, "edit", {"op": "dilate"})                # v3
    segd.ops.run_op(cs, "checkout", {"version": "v1"})           # checkout round trip: same rule
    out = segd.ops.run_op(cs, "finish", {"confidence": "high", "reason": "stale",
                                         "review_code": code2})
    assert not out.done and not cs.done
    code3 = re.search(r'review_code "(\d{4})"', out.text).group(1)
    out = segd.ops.run_op(cs, "finish", {"confidence": "high", "reason": "ok", "review_code": code3})
    assert out.done and cs.done                                  # the fresh code confirms


def test_failed_candidate_render_authorizes_nothing(tmp_path, monkeypatch):
    """Fix round #2 (seg_candidates path): a render_candidates exception must leave NOTHING
    authorized — no visibility, and seg_accept rejects even a cached candidate id."""
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    cs.commit_mask(_partial_lesion_mask(lesion))
    _break_render_candidates(monkeypatch)
    with pytest.raises(RuntimeError):
        segd.ops.run_op(cs, "candidates", {})
    assert cs.candidates_shown is None
    assert cs.current == 1 and cs.candidate_cache[1]             # the computation itself is cached
    with pytest.raises(OpError, match="out of date"):
        segd.ops.run_op(cs, "accept", {"ids": ["C1"]})
    recs = [json.loads(line) for line in
            (cs.out_dir / "trajectory.jsonl").read_text().splitlines()]
    cand_recs = [r for r in recs if r["op"] == "candidates"]
    assert cand_recs[-1]["error_kind"] == "internal"
    assert "candidates" not in cand_recs[-1]                     # no success fields on the error record


def test_failed_candidate_render_in_review_authorizes_nothing(tmp_path, monkeypatch):
    """Fix round #2 (finish path): the same injected failure in the review's candidate sheet
    leaves neither candidate visibility nor review authorization behind."""
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    cs.commit_mask(_partial_lesion_mask(lesion))
    _break_render_candidates(monkeypatch)
    with pytest.raises(RuntimeError):
        segd.ops.run_op(cs, "finish", {"confidence": "medium", "reason": "check"})
    assert cs.candidates_shown is None and cs.reviewed_version is None
    with pytest.raises(OpError, match="out of date"):
        segd.ops.run_op(cs, "accept", {"ids": ["C1"]})
    from _helpers import stub_render_candidates
    monkeypatch.setattr(R, "render_candidates", stub_render_candidates)   # render fixed
    out = segd.ops.run_op(cs, "finish", {"confidence": "medium", "reason": "check"})
    assert len(out.images) == 2 and cs.candidates_shown == cs.current
    assert re.search(r'review_code "(\d{4})"', out.text)


def test_failed_review_save_leaves_no_review_authorization(tmp_path, monkeypatch):
    """Fix round #3 (finish): if saving the review images fails after the op body ran, the
    review authorization is rolled back too — nothing is authorized unseen."""
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    cs.commit_mask(_partial_lesion_mask(lesion))

    def boom(img, name):
        raise RuntimeError("disk full")

    monkeypatch.setattr(cs.log, "save_render", boom)
    with pytest.raises(RuntimeError):
        segd.ops.run_op(cs, "finish", {"confidence": "medium", "reason": "check"})
    assert cs.reviewed_version is None and cs.review_code is None
    assert not cs.done


def test_accept_is_transactional_through_render_saving(tmp_path, monkeypatch):
    """Fix round #3 (accept): a save_render failure AFTER the commit must restore the version
    list, current version, the accept counter and the authorization, log an 'internal' error
    record, and leave the same accept retryable."""
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    cs.commit_mask(_partial_lesion_mask(lesion))                 # v1
    segd.ops.run_op(cs, "candidates", {})
    n_versions, cur, rounds = len(cs.versions), cs.current, cs.accept_rounds
    save_render = cs.log.save_render

    def boom(img, name):
        raise RuntimeError("disk full")

    monkeypatch.setattr(cs.log, "save_render", boom)
    with pytest.raises(RuntimeError):
        segd.ops.run_op(cs, "accept", {"ids": ["C1"]})
    monkeypatch.setattr(cs.log, "save_render", save_render)      # persistence healthy again
    assert len(cs.versions) == n_versions and cs.current == cur  # no version left behind
    assert cs.accept_rounds == rounds                            # counter restored
    assert cs.candidates_shown == cur and cur in cs.candidate_cache   # authorization restored
    recs = [json.loads(line) for line in
            (cs.out_dir / "trajectory.jsonl").read_text().splitlines()]
    err = [r for r in recs if r["op"] == "accept"][-1]
    assert err["error_kind"] == "internal"
    assert "accepted" not in err                                 # no success fields on the error record
    out = segd.ops.run_op(cs, "accept", {"ids": ["C1"]})         # the retry simply succeeds
    assert cs.current == cur + 1 and cs.accept_rounds == rounds + 1
    assert dice(cs.version_mask(cs.current), lesion) > dice(cs.version_mask(cur), lesion)


def test_paint_is_transactional_through_render_saving(tmp_path, monkeypatch):
    """Fix round #3 (second mutating op): the same save_render failure in seg_paint restores
    the committed version and logs the internal error."""
    install_stubs(monkeypatch, force=True)
    from _helpers import add_ring_marks
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    ids, _c = add_ring_marks(cs, lesion_center(lesion))
    k = lesion_center(lesion)[2]
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": k, "marks": "fine"})
    n_versions, cur = len(cs.versions), cs.current

    def boom(img, name):
        raise RuntimeError("disk full")

    monkeypatch.setattr(cs.log, "save_render", boom)
    with pytest.raises(RuntimeError):
        segd.ops.run_op(cs, "paint", {"ids": ids, "mode": "add", "slice_range": [k, k]})
    assert len(cs.versions) == n_versions and cs.current == cur
    recs = [json.loads(line) for line in
            (cs.out_dir / "trajectory.jsonl").read_text().splitlines()]
    assert recs[-1]["op"] == "paint" and recs[-1]["error_kind"] == "internal"


def test_candidates_flow_with_real_render(tmp_path, monkeypatch):
    """Fix round #6: the candidate flow through the REAL segd.marks/segd.render/segd.candidates
    modules (no stubs): the review's candidate round, seg_candidates, accept, re-review, confirm."""
    pytest.importorskip("segd.render")
    install_stubs(monkeypatch, force=False)          # force=False: the real modules are used
    cs, lesion = make_case_state(tmp_path, monkeypatch, shape=(96, 96, 64))
    cs.commit_mask(_partial_lesion_mask(lesion))
    out = segd.ops.run_op(cs, "finish", {"confidence": "medium", "reason": "check"})
    assert not out.done and len(out.images) == 2     # review grid + candidate sheet, real renders
    assert re.search(r"C1 add [0-9.]+ mL", out.text)
    for img in out.images:
        assert img.width <= 1568 and img.height <= 1568
        assert img.width * img.height <= 1_200_000
    assert np.all(np.asarray(out.images[1]) == R.CANDIDATE_COLOR, axis=-1).any()  # magenta outline
    code = re.search(r'review_code "(\d{4})"', out.text).group(1)

    out = segd.ops.run_op(cs, "candidates", {})      # the dedicated op, real sheet
    assert re.search(r"C1 add [0-9.]+ mL", out.text)
    sheet = out.images[0]
    assert sheet.width <= 1568 and sheet.height <= 1568
    assert sheet.width * sheet.height <= 1_200_000

    out = segd.ops.run_op(cs, "accept", {"ids": ["C1"]})
    partial = cs.version_mask(1)
    assert cs.current == 2 and dice(cs.version_mask(2), lesion) > dice(partial, lesion)
    out = segd.ops.run_op(cs, "finish", {"confidence": "medium", "reason": "check again"})
    assert not out.done                              # the accept invalidated the old review
    code2 = re.search(r'review_code "(\d{4})"', out.text).group(1)
    assert code2 != code
    out = segd.ops.run_op(cs, "finish", {"confidence": "high", "reason": "ok", "review_code": code2})
    assert out.done and cs.done
