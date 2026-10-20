"""Service API contract tests via the --fake backend (no torch, no GPU): the client
against an in-process fake service — set_image/set_target_buffer, every interaction
endpoint, full-mask responses, mapping log, determinism, accumulation, reset, close
and session-expiry behaviour."""
from __future__ import annotations

import threading

import numpy as np
import pytest

from sammed3d import client, service
from conftest import lesion_centre, make_fake_session, make_phantom


@pytest.fixture
def sess_factory(monkeypatch):
    def make(img, spacing=(1.0, 1.0, 1.0)):
        sess, _app = make_fake_session(monkeypatch, img, spacing)
        return sess
    return make


def test_capabilities_and_session_lifecycle(monkeypatch):
    sess, _app = make_fake_session(monkeypatch, make_phantom()[0])
    assert sess.supported_interactions["points"] is True
    assert sess.supported_interactions["lasso"] is True
    assert sess.supports_undo is False
    assert sess.ping() is True
    # a service restart/reap (session gone, client still alive) -> SessionExpiredError
    raw = client.SamMed3DSession(str(sess._http.base_url))
    raw._http.delete("/sessions/" + sess._session_id)
    raw.close()
    with pytest.raises(client.SessionExpiredError):
        sess.add_point_interaction((1, 1, 1), True)
    # close() then use -> httpx closed client (same as the nnInteractive client)
    sess.close()
    with pytest.raises(RuntimeError):
        sess.ping()


def test_set_image_rejects_missing_spacing(monkeypatch):
    sess, _app = make_fake_session(monkeypatch, make_phantom()[0])
    import httpx
    with pytest.raises(httpx.HTTPStatusError):
        sess._post_npy("/set_image", {}, np.zeros((1, 2, 2, 2), dtype=np.float16))


def test_point_interaction_full_mask_response(sess_factory, phantom):
    img, lesion, _body = phantom
    centre = lesion_centre(lesion)
    sess = sess_factory(img)
    bbox = sess.add_point_interaction(centre, include_interaction=True)
    assert bbox is not None and all(hi > lo for lo, hi in bbox)
    mask = sess.target_buffer
    assert mask.dtype == np.uint8 and mask.shape == img.shape
    assert (mask > 0).sum() > 0
    # the grown region lands on the lesion (the fake's flood stands in for the model)
    d = 2.0 * float(((mask > 0) & lesion).sum()) / float(((mask > 0).sum() + lesion.sum()))
    assert d > 0.6, f"fake point growth should track the lesion, dice={d:.3f}"
    # the response carried the prompt-mapping log (Phase 6 trajectory requirement)
    assert sess.last_interactions is not None
    assert sess.last_interactions[0]["op"] == "point"
    assert sess.last_timings is not None
    sess.close()


def test_queued_prompts_accumulate_and_recompute_with_all_points(sess_factory):
    """Accumulation: run_prediction=False queues; the prediction recomputes with ALL
    points so far (official iterative clicking) — two distant positives both appear."""
    img, _lesion, body = make_phantom((160, 160, 120), seed=3)
    sess = sess_factory(img)
    pts = np.stack(np.nonzero(body), axis=1)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    p1 = lo + 0.2 * (hi - lo) + 0.5
    p2 = lo + 0.8 * (hi - lo) + 0.5
    sess.add_point_interaction(tuple(p1), True, run_prediction=False)   # queued
    sess.add_point_interaction(tuple(p2), True, run_prediction=True)    # predicts
    mask = sess.target_buffer > 0
    assert mask[tuple(int(round(c)) for c in p1)]
    assert mask[tuple(int(round(c)) for c in p2)]
    # the mapping log shows the one prediction consumed the accumulated prompts
    assert sess.last_interactions[0]["op"] == "point"


def test_negative_point_removes(sess_factory, phantom):
    img, lesion, _body = phantom
    centre = lesion_centre(lesion)
    sess = sess_factory(img)
    sess.add_point_interaction(centre, include_interaction=True)
    before = int((sess.target_buffer > 0).sum())
    # an erase point inside the mask shrinks it
    neg = (centre[0] + 2, centre[1], centre[2])
    sess.add_point_interaction(neg, include_interaction=False)
    after = int((sess.target_buffer > 0).sum())
    assert after < before
    assert sess.last_interactions[0]["label"] == 0


def test_bbox_interaction_maps_to_points(sess_factory, phantom):
    img, lesion, _body = phantom
    idx = np.nonzero(lesion)
    lo = [int(a.min()) for a in idx]
    hi = [int(a.max()) + 1 for a in idx]
    box = [(lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2] + 1)]   # 3D bbox incl. slab
    sess = sess_factory(img)
    bbox = sess.add_bbox_interaction(box, include_interaction=True)
    assert bbox is not None
    assert (sess.target_buffer > 0).sum() > 0
    assert sess.last_interactions[0]["n_points"] == 9   # centre + 8 quarter points


def test_lasso_and_scribble_interactions(sess_factory, phantom):
    img, lesion, _body = phantom
    centre = lesion_centre(lesion)
    sess = sess_factory(img)
    # draw a filled disc around the centre on the axial plane (voxel-order crop grid)
    r = 8
    yy, xx = np.mgrid[centre[0] - r:centre[0] + r, centre[1] - r:centre[1] + r]
    disc = ((yy - centre[0]) ** 2 + (xx - centre[1]) ** 2) < (r * 0.8) ** 2
    drawn = np.zeros((2 * r, 2 * r, 1), dtype=np.uint8)
    drawn[:, :, 0] = disc
    bbox_crop = [[centre[0] - r, centre[0] + r], [centre[1] - r, centre[1] + r],
                 [centre[2], centre[2] + 1]]
    out = sess.add_lasso_interaction(drawn, True, run_prediction=True,
                                     interaction_bbox=bbox_crop)
    assert out is not None
    assert (sess.target_buffer > 0).sum() > 0
    assert sess.last_interactions[0]["op"] == "lasso"
    assert 0 < sess.last_interactions[0]["n_points"] <= 5
    # scribble: a straight stroke across the lesion
    stroke = np.zeros((2 * r, 2 * r, 1), dtype=np.uint8)
    stroke[r, :, 0] = 1
    out2 = sess.add_scribble_interaction(stroke, True, run_prediction=True,
                                         interaction_bbox=bbox_crop)
    assert out2 is not None
    assert sess.last_interactions[0]["op"] == "scribble"


def test_initial_seg_and_refine_keep_mask(sess_factory, phantom):
    img, lesion, _body = phantom
    seg = lesion.astype(np.uint8)                     # test-side stand-in for a paint mask
    # refine on a fresh session (no points yet): the interior-most point (EDT max) is
    # added AND the given mask acts as the low-res mask prompt
    sess = sess_factory(img)
    sess.add_initial_seg_interaction(seg, run_prediction=True)
    mask = sess.target_buffer
    assert (mask > 0).sum() > 0
    d = 2.0 * float(((mask > 0) & (seg > 0)).sum()) / float(((mask > 0).sum() + (seg > 0).sum()))
    # the fake keeps the mask at the low-res (4 model voxels = 6 mm) block resolution and
    # this lesion is only ~3 blocks wide, so block alignment (which depends on the patch
    # window) moves the Dice by +-0.1; the real model refines the prompt instead
    assert d > 0.6, f"refine should keep the given mask, dice={d:.3f}"
    assert mask[lesion_centre(lesion)] > 0
    assert any(m.get("interior_point") for m in sess.last_interactions if isinstance(m, dict))
    # the sync-rule call (no prediction): state only; the client's buffer is
    # overwritten with the given mask (same as the nnInteractive client)
    sess2 = sess_factory(img)
    sess2.add_initial_seg_interaction(seg, run_prediction=False)
    assert sess2.last_interactions is None
    assert int((sess2.target_buffer > 0).sum()) == int((seg > 0).sum())


def test_reset_interactions_clears_state(sess_factory, phantom):
    img, lesion, _body = phantom
    centre = lesion_centre(lesion)
    sess = sess_factory(img)
    sess.add_point_interaction(centre, include_interaction=True)
    assert (sess.target_buffer > 0).sum() > 0
    sess.reset_interactions()
    assert (sess.target_buffer > 0).sum() == 0        # zeroed client-side
    assert sess._last_paste_bbox is None
    # a point on the same spot re-grows the same mask (deterministic)
    sess.add_point_interaction(centre, include_interaction=True)
    assert sess.target_buffer[tuple(centre)] > 0


def test_determinism_same_inputs_same_mask(sess_factory, phantom):
    img, lesion, _body = phantom
    centre = lesion_centre(lesion)
    masks = []
    for _ in range(2):
        sess = sess_factory(img)
        sess.add_point_interaction(centre, True)
        sess.add_bbox_interaction([(0, 40), (0, 40), (0, 1)], True)
        masks.append(sess.target_buffer.copy())
    assert (masks[0] == masks[1]).all()


def test_predict_endpoint_and_grow_accumulates(sess_factory, phantom):
    img, lesion, _body = phantom
    centre = lesion_centre(lesion)
    sess = sess_factory(img)
    sess.add_point_interaction(centre, True, run_prediction=False)    # queued only
    assert (sess.target_buffer > 0).sum() == 0        # queued: no prediction ran
    bbox = sess._predict()
    assert bbox is not None
    assert (sess.target_buffer > 0).sum() > 0


def test_prediction_bbox_uses_thresholded_previous_mask_and_pastes_at_union_offset(
        sess_factory):
    """Regression: the previous-prediction bbox must come from prob > 0.5 (not > 0) and
    the stitched union crop must be pasted at its model-space offset before mapping
    back; a second call on a lesion away from the volume origin must stay on it."""
    shape = (260, 200, 120)
    img, lesion, _body = make_phantom(shape, seed=5)
    sess = sess_factory(img)
    centre = lesion_centre(lesion)
    sess.add_point_interaction(centre, True)
    first = sess.target_buffer.copy() > 0
    assert first.sum() > 0 and not first[0, 0, 0]
    sess.add_point_interaction((centre[0] + 1, centre[1], centre[2]), True)
    second = sess.target_buffer > 0
    assert (second & first).sum() > 0.8 * first.sum()          # stays where it was
    idx = np.nonzero(second)
    assert idx[0].min() > centre[0] - 40 and idx[0].max() < centre[0] + 40
    patches = sess.last_interactions[1]
    assert patches["patches_used"] == 1 and patches["windows_model_vox"]


def test_mapping_log_lists_mapped_points(sess_factory, phantom):
    img, lesion, _body = phantom
    centre = lesion_centre(lesion)
    sess = sess_factory(img)
    sess.add_point_interaction(centre, True)
    log = sess.last_interactions[0]
    assert log["mapped_points"] == [[float(c) for c in centre] + [1]]
    idx = np.nonzero(lesion)
    box = [(int(a.min()), int(a.max()) + 1) for a in idx]
    sess.add_bbox_interaction(box, False)
    assert len(sess.last_interactions[0]["mapped_points"]) == 9
    assert all(m[3] == 0 for m in sess.last_interactions[0]["mapped_points"])


def test_mapping_log_covers_queued_calls_of_a_multi_point_op(sess_factory, phantom):
    """seg_point with several marks = several queued calls + one predicting call; the
    trajectory log must carry every mapped point, not just the last call's."""
    img, lesion, _body = phantom
    c = lesion_centre(lesion)
    sess = sess_factory(img)
    sess.add_point_interaction(c, True, run_prediction=False)
    sess.add_point_interaction((c[0] + 2, c[1], c[2]), True, run_prediction=False)
    sess.add_point_interaction((c[0], c[1] + 2, c[2]), False, run_prediction=True)
    calls = [e for e in sess.last_interactions if "mapped_points" in e]
    assert [e["mapped_points"][0][3] for e in calls] == [1, 1, 0]
    assert sess.last_interactions[-1]["n_points_total"] == 3
    # the queue is consumed by the prediction
    sess.add_point_interaction(c, True)
    assert len([e for e in sess.last_interactions if "mapped_points" in e]) == 1


def test_first_patch_is_centred_on_the_first_positive_in_model_space(sess_factory):
    """Regression: the window seed must be converted session -> model space (1.5 mm
    grid); with 1 mm voxels a session-space seed would be 1.5x too far from the origin."""
    img, lesion, _body = make_phantom((300, 200, 120), seed=2)
    sess = sess_factory(img)
    p = (150, 100, 60)
    sess.add_point_interaction(p, True)
    (lo, hi), = sess.last_interactions[-1]["windows_model_vox"]
    seed_model = [c / 1.5 - 0.0 for c in p]                         # ~ (100, 66.7, 40)
    for a in range(3):
        assert abs((lo[a] + hi[a]) / 2.0 - seed_model[a]) < 1.5


def test_box_on_a_slice_registers_points_on_that_slice(sess_factory, phantom):
    img, lesion, _body = phantom
    c = lesion_centre(lesion)
    sess = sess_factory(img)
    sess.add_bbox_interaction([(c[0] - 6, c[0] + 7), (c[1] - 6, c[1] + 7), (c[2], c[2] + 1)], True)
    mapped = sess.last_interactions[0]["mapped_points"]
    assert len(mapped) == 5 and all(m[2] == float(c[2]) for m in mapped)


class _SpyPredictor(service.FakePatchPredictor):
    """FakePatchPredictor that records every patch call (points, labels, mask prompt)."""

    def __init__(self, logit: float = None):
        self.calls = []
        self.logit = logit

    def predict_patch(self, patch_img, points_local, labels, mask_prompt):
        self.calls.append({"points": np.asarray(points_local).copy(),
                           "labels": np.asarray(labels).copy(),
                           "prompt": np.asarray(mask_prompt).copy()})
        if self.logit is not None:
            lo = np.full((32, 32, 32), self.logit, dtype=np.float32)
            return np.repeat(np.repeat(np.repeat(lo, 4, 0), 4, 1), 4, 2), lo
        return super().predict_patch(patch_img, points_local, labels, mask_prompt)


def test_initial_seg_resets_clicks_like_nninteractive(monkeypatch, phantom):
    """seg_refine = add_initial_seg(mask, run_prediction=True): earlier clicks and the
    first-positive anchor are dropped (nnInteractive 'WARNING THIS WILL RESET
    INTERACTIONS'); a session with a click history and a fresh session given the same
    mask must refine from identical inputs."""
    img, lesion, _body = phantom
    c = lesion_centre(lesion)
    seg = lesion.astype(np.uint8)
    spy_a, spy_b = _SpyPredictor(), _SpyPredictor()
    a, _ = make_fake_session(monkeypatch, img, predictor=spy_a)
    a.add_point_interaction((c[0] + 20, c[1] + 15, c[2]), True)       # clicks the refine must forget
    a.add_point_interaction((c[0] - 20, c[1], c[2]), True)
    a.add_point_interaction((c[0], c[1] - 15, c[2]), False)
    a.add_initial_seg_interaction(seg, run_prediction=True)
    b, _ = make_fake_session(monkeypatch, img, predictor=spy_b)
    b.add_initial_seg_interaction(seg, run_prediction=True)
    sa, sb = a.last_interactions[-1], b.last_interactions[-1]
    assert sa["n_points_total"] == sb["n_points_total"] == 1          # only the interior seed
    assert sa["windows_model_vox"] == sb["windows_model_vox"]
    ca, cb = spy_a.calls[-1], spy_b.calls[-1]
    assert np.array_equal(ca["points"], cb["points"]) and np.array_equal(ca["labels"], cb["labels"])
    assert np.array_equal(ca["prompt"], cb["prompt"])
    assert (a.target_buffer == b.target_buffer).all()
    log = [e for e in a.last_interactions if e.get("op") == "initial_seg"][0]
    assert log["points_reset"] == 3
    # the state-only call (sync rule) also resets, adds no seed, and a following click
    # anchors the window exactly like on a fresh session
    spy_c, spy_d = _SpyPredictor(), _SpyPredictor()
    cc, _ = make_fake_session(monkeypatch, img, predictor=spy_c)
    cc.add_point_interaction((c[0] + 20, c[1] + 15, c[2]), True)
    cc.add_initial_seg_interaction(seg, run_prediction=False)
    cc.add_point_interaction(c, True)
    dd, _ = make_fake_session(monkeypatch, img, predictor=spy_d)
    dd.add_initial_seg_interaction(seg, run_prediction=False)
    dd.add_point_interaction(c, True)
    assert cc.last_interactions[-1]["windows_model_vox"] == dd.last_interactions[-1]["windows_model_vox"]
    assert (cc.target_buffer == dd.target_buffer).all()


def test_thin_mask_only_tiles_are_not_skipped():
    """A long 1-voxel-thick previous mask pools to all-negative logits in every 4^3 block;
    every tile that contains foreground must still run (the skip test uses the high-res
    mask), otherwise the stitched full-mask response silently drops those regions."""
    spy = _SpyPredictor(logit=-1.0)                    # model says 'background' everywhere
    sess = service.SamSession("t", spy, threading.Lock())
    img = np.full((460, 40, 40), 300.0, dtype=np.float32)
    sess.set_image(img[None].astype(np.float16), {"spacing": [1.0, 1.0, 1.0]})
    sess.set_target_buffer(img.shape, "uint8")
    tube = np.zeros(img.shape, dtype=np.uint8)
    tube[30:430, 20, 20] = 1                           # 400 mm long, one voxel thick
    out = sess.add_initial_seg_interaction(tube, True, False)
    meta, _bytes = out
    n_windows = len(meta["patches"]["tiles"] and meta["interactions"][-1]["windows_model_vox"])
    assert n_windows >= 3
    assert len(spy.calls) == n_windows                 # no tile skipped
    assert meta["interactions"][-1]["patches_skipped"] == 0
    # pooled prompts really are all-negative in the tiles away from the seed
    assert any(not (c["prompt"] > 0).any() for c in spy.calls)
