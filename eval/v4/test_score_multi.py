"""Tests for the Phase 4 multi-lesion scorer (eval/v4/score_multi.py).

Unit tests on synthetic arrays (no data files needed) + one integration test that scores
a synthetic case dir through a tmp manifest (dataset inference from case_meta.json).
Run: uv run --project harness/server python -m pytest eval/v4/test_score_multi.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "harness" / "server" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import score_multi as SM                     # noqa: E402
import segd.data                             # noqa: E402
from score_multi import (                    # noqa: E402
    bucket_of, lesion_metrics, tag_case_multi,
)


def _ball(shape, centre, radius):
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return ((zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2) <= radius ** 2


def test_bucket_of():
    assert bucket_of(0.1) == "lt05"
    assert bucket_of(0.5) == "b05_2"
    assert bucket_of(1.9) == "b05_2"
    assert bucket_of(2.0) == "gt2"
    assert bucket_of(30.0) == "gt2"


def test_lesion_metrics_detection_fp_and_buckets():
    shape = (100, 100, 100)
    gt = np.zeros(shape, bool)
    gt |= _ball(shape, (30, 30, 30), 10)      # 4.2 mL  -> gt2
    gt |= _ball(shape, (60, 60, 60), 5)       # 0.52 mL -> b05_2
    gt |= _ball(shape, (80, 30, 60), 3)       # 0.11 mL -> lt05
    pred = np.zeros(shape, bool)
    pred |= _ball(shape, (30, 30, 30), 9)     # full detection, clean outline
    pred |= _ball(shape, (60, 60, 60), 4)     # partial overlap (~50% of lesion)
    pred |= _ball(shape, (80, 60, 60), 3)     # FP: no GT overlap
    # the small lesion is NOT covered at all

    m = lesion_metrics(gt, pred, (1.0, 1.0, 1.0))
    assert m["n_lesions"] == 3 and m["n_detected"] == 2 and m["n_fp"] == 1
    assert m["lesion_sensitivity"] == pytest.approx(2 / 3)
    assert m["lesion_precision"] == pytest.approx(2 / 3)
    assert m["lesion_f1"] == pytest.approx(2 / 3)
    # mean per-lesion dice over DETECTED lesions only; both detected lesions have dice < 1
    assert 0.0 < m["mean_lesion_dice"] < 1.0
    # buckets: <0.5 mL undetected; 0.5-2 detected; >2 detected
    assert m["n_lt05"] == 1 and m["det_lt05"] == 0
    assert m["n_b05_2"] == 1 and m["det_b05_2"] == 1
    assert m["n_gt2"] == 1 and m["det_gt2"] == 1
    assert np.isnan(m["dice_lt05"])
    assert 0.0 < m["dice_b05_2"] < 1.0
    # per-lesion rows: volumes, buckets, detection
    vols = sorted(r["volume_ml"] for r in m["per_lesion"])
    assert vols[0] == pytest.approx(0.113, abs=0.02)
    assert vols[1] == pytest.approx(0.524, abs=0.03)
    assert vols[2] == pytest.approx(4.19, abs=0.1)
    # per-lesion rows in ndi.label scan order; check detection per bucket
    by_bucket = {r["bucket"]: r["detected"] for r in m["per_lesion"]}
    assert by_bucket == {"gt2": True, "b05_2": True, "lt05": False}


def test_lesion_detection_threshold_is_10pct_overlap():
    shape = (60, 60, 60)
    gt = _ball(shape, (30, 30, 30), 10)            # ~4189 voxels
    inside_125 = _ball(shape, (30, 30, 30), 5)     # ~524 voxels = 12.5% of the lesion
    inside_064 = _ball(shape, (30, 30, 30), 4)     # ~268 voxels = 6.4%
    assert lesion_metrics(gt, inside_125, (1.0, 1.0, 1.0))["n_detected"] == 1
    assert lesion_metrics(gt, inside_064, (1.0, 1.0, 1.0))["n_detected"] == 0


def test_empty_cases():
    empty_gt = np.zeros((10, 10, 10), bool)
    empty_pred = np.zeros((10, 10, 10), bool)
    m = lesion_metrics(empty_gt, empty_pred, (1.0, 1.0, 1.0))
    assert m["n_lesions"] == 0 and m["n_detected"] == 0 and m["n_fp"] == 0
    m = lesion_metrics(empty_gt, _ball((10, 10, 10), (5, 5, 5), 2), (1.0, 1.0, 1.0))
    assert m["n_lesions"] == 0 and m["n_fp"] == 1   # pred with no GT anywhere is all FP


def test_tag_case_multi_none_detected_replaces_localization():
    common = dict(meta_errors=[], missing_final=False, ran=True, forced=False,
                  forced_reason="", internal_errors=0)
    # tp = 0 -> none_detected, independent of the MSTT first-click machinery
    assert tag_case_multi(**common, tp=0, dice=0.9) == "none_detected"
    assert tag_case_multi(**common, tp=0, dice=0.0) == "none_detected"
    assert tag_case_multi(**common, tp=2, dice=0.9) == "none"
    # precedence: harness > none_detected > delineation > budget
    assert tag_case_multi(**{**common, "meta_errors": ["boom"]}, tp=0, dice=0.9) == "harness"
    assert tag_case_multi(**{**common, "dice": 0.3}, tp=2) == "delineation"
    assert tag_case_multi(**{**common, "forced": True, "forced_reason": "wall time exceeded"},
                          tp=2, dice=0.9) == "budget"
    assert tag_case_multi(**{**common, "internal_errors": 1}, tp=2, dice=0.9) == "harness"


def test_score_case_synthetic_manifest(tmp_path, monkeypatch):
    """Integration: a synthetic case + manifest; dataset read from case_meta.json."""
    import nibabel as nib
    monkeypatch.setattr(segd.data, "PHASE4_CSV", str(tmp_path / "{dataset}_cases.csv"))
    shape = (64, 64, 64)
    gt = _ball(shape, (30, 30, 30), 6) | _ball(shape, (45, 40, 35), 4)
    img = np.where(gt, 500.0, 100.0).astype(np.float32) + np.random.default_rng(0).normal(0, 10, shape).astype(np.float32)
    d = tmp_path / "data" / "toy"
    d.mkdir(parents=True)
    nib.save(nib.Nifti1Image(img, np.eye(4)), str(d / "Toy_001.nii.gz"))
    nib.save(nib.Nifti1Image(gt.astype(np.uint8), np.eye(4)), str(d / "Toy_001_segm.nii.gz"))
    with open(tmp_path / "toy_cases.csv", "w") as f:
        f.write("case,target_text,modality,sequence,image_path,mask_paths,n_lesions,"
                "lesion_volume_ml,split,image_path_src,mask_paths_src\n")
        f.write(f"Toy_001,lesions (all),MRI,CE-T1,{d / 'Toy_001.nii.gz'},{d / 'Toy_001_segm.nii.gz'},"
                f"2,3.0,design,,,\n")

    run_dir = tmp_path / "run"
    case_dir = run_dir / "Toy_001"
    case_dir.mkdir(parents=True)
    pred = _ball(shape, (30, 30, 30), 6)          # detects lesion 1 only; lesion 2 missed
    nib.save(nib.Nifti1Image(pred.astype(np.uint8), np.eye(4)), str(case_dir / "final_mask.nii.gz"))
    (case_dir / "trajectory.jsonl").write_text(
        '{"ts": "2026-10-21T00:00:00Z", "op": "finish", "args": {"confidence": "high", "reason": "ok"}}\n')
    (case_dir / "case_meta.json").write_text(
        '{"case_id": "Toy_001", "dataset": "toy", "tool_calls": 9, "model_prompts": 3,\n'
        ' "created_at": "2026-10-21T00:00:00Z", "finished_at": "2026-10-21T00:05:00Z"}\n')

    rec = SM.score_case(str(case_dir))
    assert rec["dataset"] == "toy" and rec["split"] == "design"
    assert rec["status"] == "ok"
    assert rec["n_lesions"] == 2 and rec["n_lesions_manifest"] == 2
    assert rec["n_detected"] == 1 and rec["n_fp"] == 0
    assert rec["lesion_sensitivity"] == pytest.approx(0.5)
    assert rec["tag"] == "none"                   # one lesion detected, dice > 0.5
    assert rec["none_detected"] is False
    assert rec["n_tool_calls"] == 9 and rec["n_model_prompts"] == 3
    assert isinstance(rec["dice"], float) and rec["dice"] > 0.5
    # the per-lesion rows exported by main()
    rows = [dict(lr, case=rec["case"]) for lr in rec["per_lesion"]]
    assert len(rows) == 2 and sum(r["detected"] for r in rows) == 1
