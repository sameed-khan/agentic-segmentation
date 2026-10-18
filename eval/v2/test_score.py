"""Synthetic unit checks for eval/v2/score.py (Dice/HD95/NSD, first-click rule, tags).

Run:  uv run --project harness/server python -m pytest eval/v2/test_score.py -q
  or  uv run --project harness/server python eval/v2/test_score.py   (self-runner, no pytest)

Sphere geometry: all radii/offsets are in mm on a 1 mm isotropic grid unless spacing says
otherwise; voxelization error is bounded by ~1 voxel, so exact analytic values are asserted
with tolerances derived in each test.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as S  # noqa: E402


def sphere(shape, center, radius, spacing=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Voxelized ball: voxels whose mm distance to `center` is <= `radius` (uint8)."""
    g = np.indices(shape, dtype=float)
    for a in range(3):
        g[a] = (g[a] - center[a]) * spacing[a]
    return (np.sqrt((g ** 2).sum(0)) <= radius).astype(np.uint8)


def lens_volume(R: float, d: float) -> float:
    """Volume (mm^3) of the intersection of two radius-R spheres whose centres are d apart."""
    return math.pi * (4 * R + d) * (2 * R - d) ** 2 / 12


def rec(op, resolved=None, error=None, child=None, polarity=None, **args):
    """A trajectory.jsonl-shaped record."""
    args = dict(args)
    if polarity is not None:
        args["polarity"] = polarity
    return {"op": op, "args": args, "resolved": resolved or [], "error": error, "child": child}


# ---------------------------------------------------------------------- metrics
def test_dice_concentric_spheres_exact_ratio():
    gt = sphere((40, 40, 40), (20, 20, 20), 14)
    pred = sphere((40, 40, 40), (20, 20, 20), 10)      # pred inside GT
    dice = S.dice_from_counts(int((pred & gt).sum()), int(pred.sum()), int(gt.sum()))
    r = 10 / 14
    expected = 2 * r ** 3 / (1 + r ** 3)                # pred ⊂ GT: 2|P| / (|P|+|G|)
    print(f"dice={dice:.4f} expected={expected:.4f}")
    assert abs(dice - expected) < 0.01


def test_hd95_concentric_bounds_and_nsd_zero():
    # surface radii: inner (9,10], outer (13,14] -> every pairwise distance in (3,5)
    gt = sphere((40, 40, 40), (20, 20, 20), 14)
    pred = sphere((40, 40, 40), (20, 20, 20), 10)
    hd95, nsd = S.surface_metrics(pred, gt, (1.0, 1.0, 1.0))
    print(f"hd95={hd95:.3f} nsd2={nsd:.4f}")
    assert 3.0 < hd95 < 5.0
    assert nsd == 0.0                                  # all gaps > 2 mm


def test_dice_offset_spheres_lens_formula():
    gt = sphere((40, 40, 40), (20, 20, 20), 10)
    pred = sphere((40, 40, 40), (23, 20, 20), 10)      # 3 mm offset along i
    dice = S.dice_from_counts(int((pred & gt).sum()), int(pred.sum()), int(gt.sum()))
    expected = lens_volume(10, 3) / (4 / 3 * math.pi * 10 ** 3)
    print(f"dice={dice:.4f} expected={expected:.4f}")
    assert abs(dice - expected) < 0.01
    hd95, nsd = S.surface_metrics(pred, gt, (1.0, 1.0, 1.0))
    print(f"hd95={hd95:.3f} nsd2={nsd:.4f}")
    assert 0.0 < hd95 <= 3.0 + 1.0                     # max offset 3 mm + voxelization
    assert 0.0 < nsd <= 1.0


def test_hd95_nsd_known_shift_two_spheres():
    # identical spheres r=10 offset 3 mm: HD (100th pct) must be ~3 mm, HD95 below it
    gt = sphere((40, 40, 40), (20, 20, 20), 10)
    pred = sphere((40, 40, 40), (17, 20, 20), 10)
    hd95, nsd = S.surface_metrics(pred, gt, (1.0, 1.0, 1.0))
    far = S.mask_surface(pred)
    far_pts = np.argwhere(far)
    sg = S.mask_surface(gt)
    from scipy import ndimage
    dt = ndimage.distance_transform_edt(~sg, sampling=(1.0, 1.0, 1.0))
    hd100 = float(dt[far_pts[:, 0], far_pts[:, 1], far_pts[:, 2]].max())
    print(f"hd95={hd95:.3f} hd100={hd100:.3f} nsd2={nsd:.4f}")
    assert abs(hd100 - 3.0) < 1.0                      # far pole gap ~3 mm
    assert hd95 <= hd100 + 1e-9


def test_anisotropic_spacing_is_honoured():
    # 5 mm spheres, centres 6 mm apart along k, spacing (1,1,2) mm
    gt = sphere((30, 30, 30), (15, 15, 15), 5, spacing=(1, 1, 2))
    pred = sphere((30, 30, 30), (15, 15, 18), 5, spacing=(1, 1, 2))
    dice = S.dice_from_counts(int((pred & gt).sum()), int(pred.sum()), int(gt.sum()))
    hd95, nsd = S.surface_metrics(pred, gt, (1.0, 1.0, 2.0))
    print(f"dice={dice:.4f} hd95={hd95:.3f} nsd2={nsd:.4f}")
    expected = lens_volume(5, 6) / (4 / 3 * math.pi * 5 ** 3)
    assert abs(dice - expected) < 0.05                 # anisotropic voxelization error
    assert hd95 > 2.5                                  # would be ~half if spacing ignored
    assert hd95 <= 6.0 + 1.0


def test_empty_masks():
    gt = sphere((20, 20, 20), (10, 10, 10), 5)
    empty = np.zeros_like(gt)
    assert S.dice_from_counts(0, 0, int(gt.sum())) == 0.0
    assert S.dice_from_counts(0, 0, 0) == 0.0
    hd95, nsd = S.surface_metrics(empty, gt, (1.0, 1.0, 1.0))
    assert math.isnan(hd95) and math.isnan(nsd)


# ------------------------------------------------------------- versions.npz format
def test_unpack_version_matches_segd_packing():
    # build a versions.npz-like mapping exactly as segd.log.flush_versions writes it
    shape = (12, 12, 12)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[3:8, 2:9, 5:11] = (np.indices((5, 7, 6)).sum(0) % 2)   # checkerboard pattern
    idx = np.nonzero(mask)
    bbox = tuple((int(a.min()), int(a.max()) + 1) for a in idx)
    crop = mask[tuple(slice(lo, hi) for lo, hi in bbox)]
    z = {"shape": np.asarray(shape),
         "ids": np.asarray([0, 1]),
         "parents": np.asarray([-1, 0]),
         "v0_bbox": np.full((3, 2), -1), "v0_nvox": np.asarray(0), "v0_packed": np.frombuffer(b"", np.uint8),
         "v1_bbox": np.asarray(bbox), "v1_nvox": np.asarray(int(crop.size)),
         "v1_packed": np.frombuffer(np.packbits(crop.astype(bool), axis=None).tobytes(), np.uint8)}
    assert S.unpack_version(z, 0, shape) is None
    out = S.unpack_version(z, 1, shape)
    assert np.array_equal(out.astype(bool), mask.astype(bool))
    # dice path
    d = S.dice_from_counts(int((out.astype(bool) & (mask > 0)).sum()), int(out.sum()), int(mask.sum()))
    assert d == 1.0


# ---------------------------------------------------------------- first click rule
GT_CLICK = sphere((20, 20, 20), (10, 10, 10), 5)       # lesion at centre, r=5


def test_first_click_point_hit():
    fc = S.first_click([rec("view"), rec("point", [{"id": 1, "ijk": [10, 10, 10]}],
                                       child="v1", polarity="positive")], GT_CLICK, (1, 1, 1))
    assert fc["first_click_inside"] is True and fc["first_click_dist_mm"] == 0.0
    assert fc["first_click_op"] == "point" and fc["first_click_version"] == "v1"


def test_first_click_point_miss_distance():
    fc = S.first_click([rec("point", [{"id": 2, "ijk": [10, 4, 10]}], child="v1",
                             polarity="add")], GT_CLICK, (1, 1, 1))
    assert fc["first_click_inside"] is False
    assert fc["first_click_dist_mm"] == 1.0            # nearest GT voxel 1 mm away


def test_first_click_skips_errors_negative_and_refine():
    recs = [rec("point", [{"id": 9, "ijk": [10, 10, 10]}], error="mark 9 does not exist"),
            rec("refine", child="v1"),
            rec("point", [{"id": 1, "ijk": [10, 10, 10]}], child="v2", polarity="negative"),
            rec("point", [{"id": 1, "ijk": [10, 4, 10]}], child="v3", polarity="add")]
    fc = S.first_click(recs, GT_CLICK, (1, 1, 1))
    assert fc["first_click_inside"] is False and fc["first_click_dist_mm"] == 1.0
    assert fc["first_click_op"] == "point" and fc["first_click_version"] == "v3"


def test_first_click_lasso_centroid_rule():
    # vertices all outside; rounded centroid (10,10,8) inside -> hit (documented rule)
    verts = [{"id": 1, "ijk": [10, 3, 10]}, {"id": 2, "ijk": [10, 10, 3]},
             {"id": 3, "ijk": [10, 17, 10]}]
    fc = S.first_click([rec("lasso", verts, child="v1", polarity="positive")], GT_CLICK, (1, 1, 1))
    assert fc["first_click_inside"] is True and fc["first_click_dist_mm"] == 0.0


def test_first_click_box_midpoint_rule():
    corners = [{"id": 1, "ijk": [10, 3, 10]}, {"id": 2, "ijk": [10, 17, 10]}]
    fc = S.first_click([rec("box", corners, child="v1", polarity="add")], GT_CLICK, (1, 1, 1))
    assert fc["first_click_inside"] is True            # midpoint (10,10,10) inside


def test_first_click_box_can_miss():
    corners = [{"id": 1, "ijk": [10, 2, 10]}, {"id": 2, "ijk": [10, 8, 4]}]
    fc = S.first_click([rec("box", corners, child="v1", polarity="add")], GT_CLICK, (1, 1, 1))
    assert fc["first_click_inside"] is False
    assert 0.0 < fc["first_click_dist_mm"] < 3.0


def test_first_click_no_positive_prompt():
    fc = S.first_click([rec("view"), rec("refine", child="v1")], GT_CLICK, (1, 1, 1))
    assert fc["first_click_inside"] is False and math.isnan(fc["first_click_dist_mm"])
    assert fc["first_click_op"] == ""


def test_first_click_empty_gt():
    fc = S.first_click([rec("point", [{"id": 1, "ijk": [0, 0, 0]}], child="v1")],
                       np.zeros((5, 5, 5), np.uint8), (1, 1, 1))
    assert fc["first_click_inside"] is False and math.isnan(fc["first_click_dist_mm"])


# ------------------------------------------------------------- end-to-end score_case
def test_score_case_end_to_end(tmp_path=None):
    """Synthetic single-case run: v1 == GT -> dice 1, point inside -> tag none; meta errors -> harness."""
    import tempfile
    tmp_path = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
    row = S.get_case("V55")
    shape_ref, affine_ref, spacing = S._image_reference(row)
    gt = S.load_mask(row, affine_ref, shape_ref)
    assert gt.sum() > 0
    idx = np.nonzero(gt)
    bbox = tuple((int(a.min()), int(a.max()) + 1) for a in idx)
    crop = gt[tuple(slice(lo, hi) for lo, hi in bbox)]
    click = tuple(int(v) for v in np.argwhere(gt)[0])

    case_dir = tmp_path / "V55"
    case_dir.mkdir(parents=True)
    np.savez_compressed(
        case_dir / "versions.npz",
        shape=np.asarray(gt.shape), ids=np.asarray([0, 1]), parents=np.asarray([-1, 0]),
        v0_bbox=np.full((3, 2), -1), v0_nvox=np.asarray(0),
        v0_packed=np.frombuffer(b"", np.uint8),
        v1_bbox=np.asarray(bbox), v1_nvox=np.asarray(int(crop.size)),
        v1_packed=np.frombuffer(np.packbits(crop.astype(bool), axis=None).tobytes(), np.uint8))
    import nibabel as nib
    nib.save(nib.Nifti1Image(gt, affine_ref), str(case_dir / "final_mask.nii.gz"))
    (case_dir / "trajectory.jsonl").write_text(json.dumps(
        {"op": "point", "args": {"ids": [7], "polarity": "positive"},
         "resolved": [{"id": 7, "ijk": list(click)}], "child": "v1", "error": None}) + "\n")
    (case_dir / "case_meta.json").write_text(json.dumps(
        {"final_version": "v1", "spacing": list(spacing), "tool_calls": 2,
         "model_prompts": 1, "confidence": "high", "reason": "good"}))

    r = S.score_case(str(case_dir))
    print({k: r[k] for k in ("status", "dice", "hd95_mm", "nsd_2mm", "tag",
                             "first_click_inside", "first_click_dist_mm", "final_is_best")})
    assert r["status"] == "ok"
    assert r["dice"] == 1.0 and r["hd95_mm"] == 0.0 and r["nsd_2mm"] == 1.0
    assert r["best_version"] == "v1" and r["final_is_best"] is True
    assert r["first_click_inside"] is True and r["first_click_dist_mm"] == 0.0
    assert r["tag"] == "none"

    (case_dir / "meta.json").write_text(json.dumps(
        {"errors": ["prompt: boom"], "forced_finish": True, "nudges_used": 1,
         "token_usage": {"input": 10, "output": 5, "total": 15}, "cost_usd": 0.01,
         "harness_sha256": "deadbeef"}))
    r2 = S.score_case(str(case_dir))
    assert r2["tag"] == "harness" and r2["forced_finish"] is True and r2["nudges"] == 1
    assert r2["tokens_total"] == 15 and r2["cost_usd"] == 0.01
    assert r2["status"] == "ok"                       # meta errors do not break scoring


# ------------------------------------------------------------------------- tags
def base(**kw):
    d = dict(meta_errors=[], missing_final=False, ran=True, forced=False,
             forced_reason="", click_inside=True, dice=0.8)
    d.update(kw)
    return d


def test_tag_harness_variants():
    assert S.tag_case(**base(meta_errors=["prompt: boom"])) == "harness"
    assert S.tag_case(**base(missing_final=True)) == "harness"
    assert S.tag_case(**base(forced=True, forced_reason="forced: prompt error")) == "harness"


def test_tag_localization_and_delineation():
    assert S.tag_case(**base(click_inside=False, dice=0.1)) == "localization"
    assert S.tag_case(**base(click_inside=False, dice=0.8)) == "none"
    assert S.tag_case(**base(dice=0.3)) == "delineation"


def test_tag_budget_and_none():
    assert S.tag_case(**base(forced=True, forced_reason="forced: wall time exceeded")) == "budget"
    assert S.tag_case(**base(forced=True, forced_reason="forced: budget exhausted")) == "budget"
    assert S.tag_case(**base(forced=True, forced_reason="forced: max nudges reached")) == "budget"
    assert S.tag_case(**base()) == "none"


def test_tag_harness_precedes_others():
    assert S.tag_case(**base(meta_errors=["x"], click_inside=False)) == "harness"


# ----------------------------------------------------------------- self-runner
if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {name}: {e}")
        except Exception as e:                             # noqa: BLE001
            failed += 1
            print(f"ERROR {name}: {e!r}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
