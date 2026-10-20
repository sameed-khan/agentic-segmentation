"""Geometry unit tests (Phase 6): resample equivalence vs torchio, patch placement /
tiling (incl. the 400 mm synthetic case), prompt mapping, stitching — no torch, no GT
in the segmenter code (GT here is test-side knowledge only)."""
from __future__ import annotations

import numpy as np
import pytest

from sammed3d import geometry


# -------------------------------------------------------------- resample
def test_resample_matches_torchio_exactly():
    torchio = pytest.importorskip("torchio")
    import torch
    rng = np.random.default_rng(7)
    for shape, sp in [((46, 60, 37), (1.0, 1.0, 1.0)), ((64, 64, 64), (0.75, 0.75, 0.75)),
                      ((10, 300, 200), (1.0, 1.0, 2.0)), ((121, 121, 81), (1.0, 1.0, 1.0))]:
        img = rng.random(shape).astype(np.float32)
        s = torchio.ScalarImage(tensor=torch.from_numpy(img)[None],
                                affine=np.diag(list(sp) + [1.0]))
        out_t = torchio.Resample(target=1.5)(s).data[0].numpy()
        out_n = geometry.resample_np(img, sp)
        assert out_t.shape == out_n.shape == geometry.resampled_shape(shape, sp)
        assert np.allclose(out_t, out_n, rtol=0, atol=1e-5), \
            f"torchio vs scipy mismatch for {shape} (maxdiff {np.abs(out_t - out_n).max()})"


def _tio_label_resample(mask, src_affine, target_img_shape, target_affine):
    torchio = pytest.importorskip("torchio")
    import torch
    src = torchio.LabelMap(tensor=torch.from_numpy(np.asarray(mask, dtype=np.float32))[None],
                           affine=src_affine)
    ref = torchio.ScalarImage(tensor=torch.zeros((1,) + tuple(target_img_shape)),
                              affine=target_affine)
    return torchio.Resample(target=ref, image_interpolation="nearest")(src).data[0].numpy()


def _model_affine(session_shape, spacing):
    origin, _size = geometry.transform_params(session_shape, spacing)
    aff = np.diag([geometry.TARGET_SPACING_MM] * 3 + [1.0])
    aff[:3, 3] = origin                       # session voxel 0 sits at 0 mm
    return aff


def _tie_voxels(sess, sp) -> np.ndarray:
    """Session voxels whose model-grid continuous index is exactly x.5 on some axis: ITK's
    rounding there depends on floating-point noise of its physical-point arithmetic (it
    differs per axis), so equality with TorchIO is only asserted off these voxels (a
    documented <= 1 model voxel tie-break difference; ours rounds half up)."""
    origin, _ = geometry.transform_params(sess, sp)
    ties = []
    for a in range(3):
        c = (np.arange(sess[a]) * sp[a] - origin[a]) / geometry.TARGET_SPACING_MM
        ties.append(np.abs((c + 0.5) - np.round(c + 0.5)) < 1e-9)
    return ties[0][:, None, None] | ties[1][None, :, None] | ties[2][None, None, :]


def _model_tie_voxels(sess, sp, model_shape) -> np.ndarray:
    origin, _ = geometry.transform_params(sess, sp)
    ties = []
    for a in range(3):
        c = (np.arange(model_shape[a]) * geometry.TARGET_SPACING_MM + origin[a]) / sp[a]
        ties.append(np.abs((c + 0.5) - np.round(c + 0.5)) < 1e-9)
    return ties[0][:, None, None] | ties[1][None, :, None] | ties[2][None, None, :]


def test_resample_matches_torchio_for_coarse_spacing_and_edge_voxels():
    """Spacing > 1.5 mm puts the first/last model voxel centres in the OUTER half voxel of
    the image: ITK treats them as inside (clamped edge value), not as zeros."""
    torchio = pytest.importorskip("torchio")
    import torch
    rng = np.random.default_rng(3)
    for shape, sp in [((10, 12, 9), (3.0, 3.0, 3.0)), ((10, 12, 9), (2.0, 2.5, 4.0)),
                      ((7, 8, 6), (1.6, 1.6, 1.6))]:
        img = rng.random(shape).astype(np.float32) + 1.0
        s = torchio.ScalarImage(tensor=torch.from_numpy(img)[None],
                                affine=np.diag(list(sp) + [1.0]))
        ref = torchio.Resample(target=1.5)(s).data[0].numpy()
        ours = geometry.resample_np(img, sp)
        assert ours.shape == ref.shape
        assert np.allclose(ours, ref, rtol=0, atol=1e-5), (shape, sp, np.abs(ours - ref).max())
        assert (ours > 0).all()               # nothing at the border was zeroed


def test_back_mapping_matches_torchio_nearest_including_boundary_voxels():
    rng = np.random.default_rng(5)
    # the reviewer's case: 6^3 session at 1 mm -> 4^3 model grid, origin 0.25 mm
    for sess, sp in [((6, 6, 6), (1.0, 1.0, 1.0)), ((31, 40, 27), (1.0, 1.0, 1.0)),
                     ((20, 17, 30), (1.0, 1.0, 2.0)), ((9, 11, 8), (3.0, 3.0, 3.0))]:
        model_shape = geometry.resampled_shape(sess, sp)
        for model in (np.ones(model_shape, dtype=bool),
                      rng.random(model_shape) > 0.5):
            ref = _tio_label_resample(model, _model_affine(sess, sp), sess, np.diag(list(sp) + [1.0]))
            ours = geometry.map_binary_model_to_session(model, sp, sess)
            assert ours.shape == tuple(sess)
            tie = _tie_voxels(sess, sp)
            assert (ours[~tie] == (ref > 0)[~tie]).all(), (sess, sp, int((ours != (ref > 0))[~tie].sum()))
    ones = geometry.map_binary_model_to_session(
        np.ones(geometry.resampled_shape((6, 6, 6), (1.0,) * 3), dtype=bool), (1.0,) * 3, (6, 6, 6))
    assert ones.all()                         # an all-ones model mask loses no outer face


def test_session_mask_to_model_matches_torchio_nearest():
    rng = np.random.default_rng(6)
    for sess, sp in [((31, 40, 27), (1.0, 1.0, 1.0)), ((9, 11, 8), (3.0, 3.0, 3.0))]:
        mask = rng.random(sess) > 0.5
        model_shape = geometry.resampled_shape(sess, sp)
        ref = _tio_label_resample(mask, np.diag(list(sp) + [1.0]), model_shape,
                                  _model_affine(sess, sp))
        ours = geometry.map_mask_session_to_model(mask, sp, model_shape)
        tie = _model_tie_voxels(sess, sp, model_shape)
        assert ((ours > 0) == (ref > 0))[~tie].all()


def test_znorm_matches_torchio_semantics():
    # official: tio.ZNormalization(masking_method=lambda x: x > 0) — mean/sample-std
    # over x>0, applied to ALL voxels
    x = np.zeros((4, 4, 4), dtype=np.float32)
    x[1:3, 1:3, 1:3] = np.array([100.0, 300.0, 500.0, 700.0, 900.0, 1100.0,
                                 1300.0, 1500.0], dtype=np.float32).reshape(2, 2, 2)
    z = geometry.znorm_masked(x)
    vals = x[x > 0]
    assert np.isclose(z[x > 0], (vals - vals.mean()) / vals.std(ddof=1), atol=1e-6).all()
    assert np.isclose(z[0, 0, 0], (0.0 - vals.mean()) / vals.std(ddof=1), atol=1e-6)
    torchio = pytest.importorskip("torchio")
    import torch
    ref = torchio.ZNormalization(masking_method=lambda t: t > 0)(
        torch.from_numpy(x)[None]).numpy()[0]
    assert np.allclose(z, ref, atol=1e-5)
    # degenerate patches stay all-zero (torchio would raise; documented deviation)
    assert (geometry.znorm_masked(np.zeros((8, 8, 8), np.float32)) == 0).all()
    const = np.full((8, 8, 8), 5.0, dtype=np.float32)
    assert (geometry.znorm_masked(const) == 0).all()


# ---------------------------------------------------------- patch placement
def test_single_patch_centred_on_first_positive_prompt():
    # two positive prompts 150 mm apart along i (1 mm iso -> 100 model vox apart):
    # box fits in one 128-vox window; window must COVER the box, centred (unshifted)
    # on the FIRST prompt
    spacing = (1.0, 1.0, 1.0)
    seed = np.asarray([100.0, 50.0, 50.0])                   # model-space voxels
    other = np.asarray([70.0, 50.0, 50.0])                   # 30 vox (45 mm) away
    box = geometry.bbox_union(geometry.bbox_of_points([seed]), geometry.bbox_of_points([other]))
    windows, info = geometry.choose_patches(box[0], box[1], seed, (300, 300, 300))
    assert len(windows) == 1 and not info["capped"]
    lo, hi = windows[0]
    local = geometry.points_inside_window([seed, other], lo, hi)
    assert local.shape[0] == 2                       # both prompts inside
    # feasible centring: seed exactly at the window centre (no shift needed)
    assert abs(float((lo[0] + hi[0]) / 2 - seed[0])) <= 0.5
    assert hi[0] - lo[0] == geometry.PATCH_VOX


def test_edge_prompt_window_stays_centred_and_is_padded():
    # a first prompt near the volume edge: the window stays centred on it (official
    # CropOrPad pads) instead of being shifted off the prompt
    seed = np.asarray([10.0, 60.0, 60.0])
    box = geometry.bbox_of_points([seed])
    (lo, hi), = geometry.choose_patches(box[0], box[1], seed, (200, 200, 200))[0]
    assert lo[0] < 0 and abs((lo[0] + hi[0]) / 2 - seed[0]) <= 0.5


def test_400mm_synthetic_mask_needs_multiple_patches_and_no_gt():
    """A 400 mm-long synthetic mask (one axis): 400 mm > 192 mm -> tiled with >= 25 %
    overlap; placement uses ONLY prompts and the previous mask, never GT."""
    spacing = (1.0, 1.0, 1.0)
    session_shape = (500, 300, 300)
    # synthetic 400 mm tube, session space (the segmenter never sees it: only the
    # previous prediction's mask and the prompt points reach patch placement)
    mask = np.zeros(session_shape, dtype=bool)
    mask[50:450, 150:151, 150:151] = True
    model_shape = geometry.resampled_shape(session_shape, spacing)
    prev_model = geometry.map_mask_session_to_model(mask, spacing, model_shape) > 0
    point_model = geometry.session_to_model([60.0, 150.0, 150.0], session_shape, spacing)
    box = geometry.bbox_union(geometry.bbox_of_points([point_model]),
                              geometry.bbox_of_points(np.stack(np.nonzero(prev_model), axis=1)))
    assert (box[1] - box[0]).max() * geometry.TARGET_SPACING_MM >= 390
    windows, info = geometry.choose_patches(box[0], box[1], point_model, model_shape)
    assert info["n_windows"] >= 2 and info["n_windows"] == len(windows)
    assert not info["capped"]                         # fits under the 27 cap
    # >= 25 % overlap: adjacent windows on the long axis start <= 96 vox apart
    starts = sorted({int(w[0][0]) for w in windows})
    strides = np.diff(starts)
    assert len(starts) >= 3 and (strides > 0).all()
    assert (strides <= geometry.PATCH_VOX - geometry.PATCH_OVERLAP_VOX).all()
    # the tiles cover the whole box on the long axis, no gap
    assert starts[0] <= box[0][0] and max(int(w[1][0]) for w in windows) >= box[1][0]
    # every window holds part of the previous mask (placement came from it)
    for lo, hi in windows:
        sl = tuple(slice(max(int(l), 0), int(h)) for l, h in zip(lo, hi))
        assert prev_model[sl].any()


def test_tiling_caps_at_27_patches_and_logs():
    # a huge box would need 125 tiles at 25 % overlap -> coarsened stride, capped
    box = (np.array([0, 0, 0]), np.array([600, 600, 600]))
    windows, info = geometry.choose_patches(box[0], box[1], None, (700, 700, 700))
    assert info["n_windows"] <= geometry.MAX_PATCHES
    assert info["capped"] and "capped" in info["note"]
    # kept windows are the ones nearest the box centre, unique, deterministic
    assert len({tuple(int(v) for v in w[0]) for w in windows}) == len(windows)
    again, again_info = geometry.choose_patches(box[0], box[1], None, (700, 700, 700))
    assert [tuple(int(v) for v in w[0]) for w in again] == \
        [tuple(int(v) for v in w[0]) for w in windows]


def test_windows_clipped_into_volume_and_padded_content():
    vol = np.ones((10, 130, 130), dtype=np.float32)
    lo, hi = np.array([-10, 1, 1]), np.array([118, 129, 129])
    patch = geometry.crop_patch(vol, lo, hi)
    assert patch.shape == (geometry.PATCH_VOX,) * 3
    assert (patch[:10] == 0).all()                    # out-of-volume head is padded 0
    assert (patch[10:20] == 1).all()                  # the in-volume part
    assert (patch[20:] == 0).all()


# -------------------------------------------------------- prompt mapping
def test_bbox_maps_to_centre_plus_quarter_points():
    pts = geometry.bbox_quarter_points((np.array([10, 10, 20]), np.array([110, 110, 20])),
                                       include=True)
    assert len(pts) == 5                              # 2D box: centre + 4 quarter points
    coords = [p for p, lab in pts]
    assert all(lab == 1 for _p, lab in pts)
    centre = (60.0, 60.0, 20.0)
    assert any(np.allclose(c, centre) for c in coords)
    quarters = sorted(tuple(c[:2]) for c in coords if not np.allclose(c, centre))
    assert quarters == [(35.0, 35.0), (35.0, 85.0), (85.0, 35.0), (85.0, 85.0)]
    neg = geometry.bbox_quarter_points((np.array([10, 10, 20]), np.array([110, 110, 20])),
                                       include=False)
    assert all(lab == 0 for _p, lab in neg)
    # a 3D box gets 8 quarter points + centre
    pts3 = geometry.bbox_quarter_points((np.array([10, 10, 10]), np.array([110, 110, 110])),
                                        include=True)
    assert len(pts3) == 9


def test_lasso_fps_points_deterministic_and_inside():
    region = np.zeros((60, 60, 4), dtype=bool)
    yy, xx, _zz = np.mgrid[:60, :60, :4]
    region[(yy - 30) ** 2 + (xx - 20) ** 2 < 15 ** 2] = True
    a = geometry.fps_points(region)
    b = geometry.fps_points(region)
    assert a == b                                     # deterministic
    assert 1 <= len(a) <= geometry.MAX_DRAWN_POINTS
    for p in a:
        assert region[int(p[0]), int(p[1]), int(p[2])]  # inside the filled contour
    if len(a) >= 2:
        d = np.linalg.norm(np.asarray(a[0]) - np.asarray(a[1]))
        assert d > 5                                  # spread out (FPS)


def test_scribble_points_deterministic_along_stroke():
    stroke = np.zeros((60, 60, 2), dtype=bool)
    stroke[30, 5:55, :] = True                        # a straight stroke
    a = geometry.stroke_points(stroke)
    b = geometry.stroke_points(stroke)
    assert a == b
    assert 1 <= len(a) <= geometry.MAX_DRAWN_POINTS
    if len(a) == geometry.MAX_DRAWN_POINTS:
        rows = [p[0] for p in a]
        cols = [p[1] for p in a]
        assert max(cols) - min(cols) > 20             # spread along the stroke
        assert all(abs(r - 30.0) < 1e-9 for r in rows)   # ON the stroke


def test_edt_max_voxel_is_interior_most():
    m = np.zeros((40, 40, 40), dtype=bool)
    m[5:15, 5:25, 5:15] = True
    v = geometry.edt_max_voxel(m)
    assert m[v]
    dist = np.asarray(m, dtype=int)
    from scipy.ndimage import distance_transform_edt
    d = distance_transform_edt(m)
    assert d[v] == d.max()


# ---------------------------------------------------------------- stitching
def test_stitch_logits_averages_overlaps():
    # two overlapping patches with different constant logits: the overlap averages
    logits_a = np.full((128, 128, 128), 2.0, dtype=np.float32)
    logits_b = np.full((128, 128, 128), 6.0, dtype=np.float32)
    stitched, lo = geometry.stitch_logits(
        [((np.array([0, 0, 0]), np.array([128, 128, 128])), logits_a),
         ((np.array([96, 0, 0]), np.array([224, 128, 128])), logits_b)],
        (256, 128, 128))
    assert stitched.shape == (224, 128, 128)
    assert lo[0] == 0
    assert np.allclose(stitched[:96], 2.0)
    assert np.allclose(stitched[128:], 6.0)
    assert np.allclose(stitched[96:128], 4.0)         # the overlap averaged
    # threshold 0.5 probability == logits > 0
    binary = stitched > 0
    assert binary.all()


def test_stitch_ignores_out_of_volume_part():
    stitched, lo = geometry.stitch_logits(
        [((np.array([100, 0, 0]), np.array([228, 128, 128])),
          np.full((128, 128, 128), 3.0, dtype=np.float32))],
        (150, 128, 128))
    assert stitched.shape == (50, 128, 128)
    assert np.allclose(stitched, 3.0)


# -------------------------------------------------------------- mask prompts
def test_mask_prompt_from_binary_mask_is_plusminus_logits():
    prev = np.zeros((128, 128, 128), dtype=np.float32)
    prev[20:24, 20:24, 20:24] = 1.0                   # one full 4^3 block -> solid prompt
    state = (np.array([0, 0, 0]), prev.astype(np.float16), [])
    lo32 = geometry.mask_prompt_for_window(state, (np.array([0, 0, 0]),
                                                   np.array([128, 128, 128])))
    assert lo32.shape == (32, 32, 32)
    assert lo32[5, 5, 5] == pytest.approx(geometry.MASK_PROMPT_LOGIT, abs=1e-3)
    assert lo32.min() == pytest.approx(-geometry.MASK_PROMPT_LOGIT, abs=1e-3)
    # a sub-block mask downsample-averages: graded logit, magnitude < the solid +/-
    sparse = np.zeros((128, 128, 128), dtype=np.float32)
    sparse[64, 64, 64] = 1.0
    lo32b = geometry.mask_prompt_for_window((np.array([0, 0, 0]),
                                             sparse.astype(np.float16), []),
                                            (np.array([0, 0, 0]), np.array([128, 128, 128])))
    assert -geometry.MASK_PROMPT_LOGIT < lo32b.max() < geometry.MASK_PROMPT_LOGIT
    # first prediction: all-zero prompt (official initial mask prompt)
    none32 = geometry.mask_prompt_for_window(None, (np.array([0, 0, 0]),
                                                    np.array([128, 128, 128])))
    assert (none32 == 0).all()


def test_mask_prompt_reuses_raw_lowres_for_identical_window():
    raw = np.full((32, 32, 32), 2.5, dtype=np.float32)
    state = (np.array([0, 0, 0]), np.zeros((128, 128, 128), np.float16),
             [((np.array([0, 0, 0]), np.array([128, 128, 128])), raw)])
    out = geometry.mask_prompt_for_window(state, (np.array([0, 0, 0]),
                                                  np.array([128, 128, 128])))
    assert (out == 2.5).all()


def test_tiled_windows_centre_short_axes_on_the_seed_and_pad_long_axes():
    # long axis 0 (300 vox -> tiled), short axes: one tile centred on the seed (not at
    # the box edge); tiled axis covers the box plus the margin on both sides
    seed = np.asarray([40.0, 100.0, 60.0])
    box = (np.array([40, 100, 60]), np.array([340, 101, 61]))
    windows, info = geometry.choose_patches(box[0], box[1], seed, (500, 300, 300))
    assert info["tiles"][1] == info["tiles"][2] == 1 and info["tiles"][0] >= 3
    for lo, hi in windows:
        assert abs((lo[1] + hi[1]) / 2 - 100.5) <= 1.0 and abs((lo[2] + hi[2]) / 2 - 60.5) <= 1.0
    assert min(int(w[0][0]) for w in windows) == 40 - geometry.TILE_MARGIN_VOX
    assert max(int(w[1][0]) for w in windows) >= 340 + geometry.TILE_MARGIN_VOX


def test_bbox_points_of_a_size_one_slice_box_sit_on_the_slice_voxel():
    # the harness sends the fixed axis as the half-open [slice, slice+1) (ops.op_box)
    pts = geometry.bbox_quarter_points((np.array([10, 10, 7]), np.array([110, 110, 8])), True)
    assert len(pts) == 5
    assert all(p[2] == 7.0 for p, _lab in pts)                # not 7.5
    pts3 = geometry.bbox_quarter_points((np.array([4, 10, 20]), np.array([5, 110, 120])), False)
    assert len(pts3) == 5 and all(p[0] == 4.0 for p, _lab in pts3)
    assert all(lab == 0 for _p, lab in pts3)
