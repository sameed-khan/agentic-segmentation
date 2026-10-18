"""FakeNNI in isolation: growth semantics, polarity, pending prompts, bbox convention."""
from __future__ import annotations

import numpy as np
import pytest

from segd.fake_nni import FakeNNI
from _helpers import dice, lesion_center, make_phantom


def _session_with_phantom(z_halfwidth=0.25, shape=(120, 120, 80)):
    img, lesion, _body = make_phantom(shape, z_halfwidth=z_halfwidth)
    s = FakeNNI()
    s.set_image(img.astype(np.float16)[None])
    s.set_target_buffer(np.zeros(shape, dtype=np.uint8))
    return s, img, lesion


def test_set_image_shape_check():
    s = FakeNNI()
    with pytest.raises(AssertionError):
        s.set_image(np.zeros((10, 10, 10), np.float16))


def test_point_grows_lesion_region():
    s, _img, lesion = _session_with_phantom()
    c = lesion_center(lesion)
    s.add_point_interaction(list(c), include_interaction=True)
    assert dice(s.target_buffer, lesion) > 0.9
    bbox = s._last_paste_bbox
    assert bbox is not None and all(hi > lo for lo, hi in bbox)


def test_negative_point_removes_similar_connected_region():
    s, img, lesion = _session_with_phantom()
    c = lesion_center(lesion)
    s.add_point_interaction(list(c), include_interaction=True)
    assert (s.target_buffer > 0).sum() > 0
    # a negative point inside the lesion removes the lesion region
    s.add_point_interaction(list(c), include_interaction=False)
    assert (s.target_buffer > 0).sum() == 0


def test_pending_prompts_applied_on_last_prediction():
    """run_prediction=False queues; the run_prediction=True prompt applies all pending."""
    s, _img, lesion = _session_with_phantom()
    c = lesion_center(lesion)
    s.add_point_interaction(list(c), include_interaction=True, run_prediction=False)
    assert s.target_buffer.sum() == 0                    # nothing predicted yet
    s.add_point_interaction(list((c[0] + 2, c[1], c[2])), include_interaction=True, run_prediction=True)
    assert dice(s.target_buffer, lesion) > 0.9
    # _predict() flushes pending too
    s2, _img2, lesion2 = _session_with_phantom()
    s2.add_point_interaction(list(c), include_interaction=True, run_prediction=False)
    s2._predict()
    assert dice(s2.target_buffer, lesion2) > 0.9


def test_lasso_half_open_bbox_convention():
    """interaction_bbox is [[x0,x1],[y0,y1],[z0,z1]] half-open; mask shape == (ub-lb)."""
    s, img, _lesion = _session_with_phantom(shape=(40, 40, 20))
    s.image[:, :, 5] = 300.0                              # content only on slice 5: propagation stops there
    poly = np.zeros((4, 4, 1), np.uint8)
    poly[1:3, 1:3] = 1
    s.add_lasso_interaction(poly, include_interaction=True, run_prediction=True,
                            interaction_bbox=[[2, 6], [2, 6], [5, 6]])
    # mask pixel (r, c) lands at full coords (r+2, c+2, slice 5)
    assert s.target_buffer[3:5, 3:5, 5].sum() == 4
    assert s.target_buffer[2, 2, 5] == 0                  # lower bound is inclusive, upper exclusive
    assert s.target_buffer[5, 5, 5] == 0
    assert s._last_paste_bbox == [[3, 5], [3, 5], [5, 6]]


def test_lasso_fills_and_propagates():
    s, img, lesion = _session_with_phantom(z_halfwidth=0.08)
    c = lesion_center(lesion)
    k = c[2]
    poly = lesion[:, :, k].astype(np.uint8)               # exact cross-section as the lasso
    nz = np.nonzero(poly)
    ibox = [[int(a.min()) - 1, int(a.max()) + 2] for a in nz]   # i, j half-open
    ibox.append([k, k + 1])                                    # drawn on a single slice
    small = np.zeros((ibox[0][1] - ibox[0][0], ibox[1][1] - ibox[1][0], 1), np.uint8)
    small[:, :, 0] = poly[tuple(slice(lo, hi) for lo, hi in ibox[:2])]
    s.add_lasso_interaction(small, include_interaction=True, run_prediction=True,
                            interaction_bbox=ibox)
    assert dice(s.target_buffer, lesion) > 0.8


def test_scribble_grows_into_similar_region():
    s, img, lesion = _session_with_phantom(z_halfwidth=0.08)
    c = lesion_center(lesion)
    k = c[2]
    scrib = np.zeros(img.shape, np.uint8)
    scrib[c[0] - 5:c[0] + 6, c[1], k] = 1                 # thin line across the lesion centre
    s.add_scribble_interaction(scrib, include_interaction=True, run_prediction=True)
    assert dice(s.target_buffer, lesion) > 0.5


def test_initial_seg_overwrites_and_cleans():
    s, _img, lesion = _session_with_phantom()
    holed = lesion.astype(np.uint8)
    c = lesion_center(lesion)
    holed[c[0] - 1:c[0] + 2, c[1] - 1:c[1] + 2, c[2] - 1:c[2] + 2] = 0
    s.add_initial_seg_interaction(holed, run_prediction=True)
    assert s.target_buffer.sum() >= lesion.sum() - 5      # hole filled by the stand-in refine
    assert dice(s.target_buffer, lesion) > 0.99
    assert s.supports_initial_label is True


def test_reset_zeroes_buffer():
    s, _img, lesion = _session_with_phantom()
    c = lesion_center(lesion)
    s.add_point_interaction(list(c), include_interaction=True)
    assert s.target_buffer.sum() > 0
    s.reset_interactions()
    assert s.target_buffer.sum() == 0
    assert s._last_paste_bbox is None


def test_drawn_plane_taken_from_size1_box_dimension():
    """Finding #4: the fake infers the drawing plane from the size-1 bbox dimension —
    after the ops fix (#1) that is always the true fixed axis, e.g. a coronal lasso
    (size 1 on j) must land on the coronal slice, not be misread as axial (the drawn
    cross-section matches the lesion on that slice -> dice check discriminates the
    wrong-plane behaviour)."""
    s, img, lesion = _session_with_phantom(shape=(40, 40, 30), z_halfwidth=0.08)
    c = lesion_center(lesion)
    j = c[1]
    # interaction crop: the exact lesion cross-section on the coronal slice j
    sub = lesion[:, j, :]                                   # (i, k) cross-section
    ii = np.nonzero(sub.any(axis=1))[0]
    kk = np.nonzero(sub.any(axis=0))[0]
    ib = [[int(ii.min()), int(ii.max()) + 1], [j, j + 1], [int(kk.min()), int(kk.max()) + 1]]
    poly = np.zeros((ib[0][1] - ib[0][0], 1, ib[2][1] - ib[2][0]), np.uint8)
    poly[:, 0, :] = sub[ib[0][0]:ib[0][1], ib[2][0]:ib[2][1]]
    assert poly.sum() > 0
    s.add_lasso_interaction(poly, include_interaction=True, run_prediction=True,
                            interaction_bbox=[[b[0], b[1]] for b in ib])
    buf = s.target_buffer
    assert buf[:, j, :].any()                               # mask placed on the coronal slice
    coronal = np.take(buf, j, axis=1) > 0
    assert dice(coronal, np.take(lesion, j, axis=1)) > 0.7
    # the mask only spreads a few slices around the drawn coronal slice
    nz = np.nonzero(buf.any(axis=(0, 2)))[0]
    assert j in nz.tolist() and len(nz) <= 7
