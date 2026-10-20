"""Tests for the Phase 4 multi-lesion scorer (eval/v4/score_multi.py).

Unit tests on synthetic arrays (no data files needed) + one integration test that scores
a synthetic case dir through a tmp manifest (dataset inference from case_meta.json).
Run: uv run --project harness/server python -m pytest eval/v4/test_score_multi.py -q
"""
from __future__ import annotations

import math
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
    assert bucket_of(2.0) == "b05_2"      # 2.0 mL belongs to the middle bucket (review P2 #5)
    assert bucket_of(2.0001) == "gt2"
    assert bucket_of(30.0) == "gt2"


def test_bucket_boundary_2ml_aggregate_matches_bucket_of():
    """Regression (review P2 #5): a lesion of exactly 2.000 mL (2000 voxels at 1 mm spacing)
    used to land in NO aggregate bucket while bucket_of(2.0) said 'gt2'. Now 2.0 is in the
    middle bucket everywhere."""
    shape = (40, 40, 40)
    gt = np.zeros(shape, bool)
    gt[5:15, 5:15, 10:30] = True          # exactly 10*10*20 = 2000 voxels = 2.000 mL
    assert gt.sum() == 2000
    m = lesion_metrics(gt, gt.copy(), (1.0, 1.0, 1.0))
    assert m["per_lesion"][0]["bucket"] == "b05_2"
    assert m["n_lt05"] == 0 and m["n_b05_2"] == 1 and m["n_gt2"] == 0
    assert m["det_b05_2"] == 1


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


# --------------------------------------------------------------------- CLI level
def _write_case(tmp_path, monkeypatch, run_dir, case, gt, pred, n_lesions, dataset="toy",
                split="design"):
    """One synthetic case dir + manifest row, dataset read from case_meta.json."""
    import nibabel as nib
    monkeypatch.setattr(segd.data, "PHASE4_CSV", str(tmp_path / "{dataset}_cases.csv"))
    d = tmp_path / "data" / dataset
    d.mkdir(parents=True, exist_ok=True)
    img = np.full(gt.shape, 100.0, dtype=np.float32)
    nib.save(nib.Nifti1Image(img, np.eye(4)), str(d / f"{case}.nii.gz"))
    nib.save(nib.Nifti1Image(gt.astype(np.uint8), np.eye(4)), str(d / f"{case}_segm.nii.gz"))
    case_dir = run_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(pred.astype(np.uint8), np.eye(4)),
             str(case_dir / "final_mask.nii.gz"))
    (case_dir / "trajectory.jsonl").write_text(
        '{"ts": "2026-10-21T00:00:00Z", "op": "finish", "args": {"confidence": "high"}}\n')
    (case_dir / "case_meta.json").write_text(
        f'{{"case_id": "{case}", "dataset": "{dataset}", "tool_calls": 5, "model_prompts": 2,\n'
        ' "created_at": "2026-10-21T00:00:00Z", "finished_at": "2026-10-21T00:05:00Z"}\n')
    return (f"{case},{dataset}-target,MRI,CE-T1,{d / f'{case}.nii.gz'},{d / f'{case}_segm.nii.gz'},"
            f"{n_lesions},3.0,{split},,")


def _manifest_header():
    return ("case,target_text,modality,sequence,image_path,mask_paths,n_lesions,"
            "lesion_volume_ml,split,image_path_src,mask_paths_src\n")


def test_pooled_f1_equals_per_case_f1(tmp_path, monkeypatch):
    """Regression (review P1 #4): the pooled report used to compute 2*det/(2*det+FP),
    omitting FN — a single case with 2 GT lesions, 1 detected and 0 FP printed pooled
    F1 1.000 while its per-case F1 is 2/3. The pooled line must equal the per-case F1."""
    shape = (80, 80, 80)
    gt = _ball(shape, (25, 25, 25), 8) | _ball(shape, (60, 60, 60), 8)
    pred = _ball(shape, (25, 25, 25), 8)          # lesion 1 only, no FP
    rows = [_write_case(tmp_path, monkeypatch, tmp_path / "run", "Toy_001", gt, pred, 2)]
    (tmp_path / "toy_cases.csv").write_text(_manifest_header() + "\n".join(rows) + "\n")

    rc = SM.main([str(tmp_path / "run"), "--workers", "1"])
    assert rc == 0
    rec = SM.score_case(str(tmp_path / "run" / "Toy_001"))
    assert rec["n_lesions"] == 2 and rec["n_detected"] == 1 and rec["n_fp"] == 0
    assert rec["lesion_f1"] == pytest.approx(2 / 3)
    md = (tmp_path / "run" / "scores_multi.md").read_text()
    pooled = next(l for l in md.splitlines() if l.startswith("**Pooled**"))
    # sensitivity 0.5; F1 = 2*1/(2*1+0+1) = 2/3, identical to the per-case F1
    assert "sensitivity 0.500" in pooled and "lesion F1 0.667" in pooled, pooled


def test_cli_empty_gt_runs_and_reports_undefined(tmp_path, monkeypatch):
    """Regression (review P2 #6): with ALL scored GT masks empty the pooled summary used
    to crash on det/n_l (and the markdown on the same division when the empty-GT case
    had FP predictions). Both empty and non-empty predictions on empty GT must run and
    render sensitivity/F1 as explicitly undefined."""
    shape = (40, 40, 40)
    empty = np.zeros(shape, bool)
    rows = [
        _write_case(tmp_path, monkeypatch, tmp_path / "run", "Toy_empty_pred",
                    empty, empty, 0),
        _write_case(tmp_path, monkeypatch, tmp_path / "run", "Toy_fp_pred",
                    empty, _ball(shape, (20, 20, 20), 5), 0),
    ]
    (tmp_path / "toy_cases.csv").write_text(_manifest_header() + "\n".join(rows) + "\n")

    rc = SM.main([str(tmp_path / "run"), "--workers", "1"])
    assert rc == 0
    md = (tmp_path / "run" / "scores_multi.md").read_text()
    pooled = next(l for l in md.splitlines() if l.startswith("**Pooled**"))
    assert "0 GT lesions" in pooled and "1 FPs" in pooled, pooled
    # sensitivity: undefined (no GT lesions); F1: defined value (denominator 1 via the FP)
    assert "sensitivity undefined (no GT lesions)" in pooled and "lesion F1 0.000" in pooled
    # per-case rows render the undefined sensitivity explicitly, cases stay 'ok'
    assert md.count("| — |") >= 2 or "undefined" in md
    for case in ("Toy_empty_pred", "Toy_fp_pred"):
        rec = SM.score_case(str(tmp_path / "run" / case))
        assert rec["status"] == "ok" and "empty_gt" in rec["flags"]
        assert math.isnan(rec["lesion_sensitivity"])


def test_write_md_pooled_zero_denominator_is_undefined(tmp_path):
    """Regression (review P2 #6, direct): an ok case with ZERO GT lesions used to make the
    pooled markdown crash on det/n_l. It must render an explicit undefined value."""
    nan = float("nan")
    recs = [{"case": "Toy_000", "status": "ok", "split": "design", "dice": 0.0,
             "hd95_mm": nan, "nsd_2mm": nan, "n_lesions": 0, "n_detected": 0, "n_fp": 0,
             "lesion_sensitivity": nan, "lesion_f1": nan, "mean_lesion_dice": nan,
             "tag": "none_detected", "flags": "empty_gt", "error": ""}]
    SM.write_md(tmp_path, recs)
    md = (tmp_path / "scores_multi.md").read_text()
    pooled = next(l for l in md.splitlines() if l.startswith("**Pooled**"))
    assert "sensitivity undefined (no GT lesions)" in pooled, pooled
    assert "lesion F1 undefined" in pooled, pooled          # zero denominator: no crash
    assert "—" in next(l for l in md.splitlines() if l.startswith("| Toy_000"))
