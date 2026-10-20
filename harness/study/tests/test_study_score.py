"""Unit checks for harness/study/score.py with synthetic data (validation (b)).

Run:  uv run --project harness/server python -m pytest harness/study/tests -q
No real data needed: manifest rows and GT slices are synthetic; the R geometry uses
segd.render.ViewGeom + panel_to_voxel so the coordinate rule is tested as wired.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi

STUDY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_DIR))

import score as S  # noqa: E402
from segd.render import ViewGeom, voxel_to_panel  # noqa: E402

# synthetic volume [i, j, k] = (40, 30, 20); GT disc radius 5 at (i=20, j=15) on k*=10
SHAPE = (40, 30, 20)
K_STAR = 10
GT_I, GT_J, GT_R = 20, 15, 5.0

GEOM_R = ViewGeom("axial", K_STAR, 30, 40, (0, 0, 30, 40), 2.0, 80, 60)


@pytest.fixture(scope="module")
def gt_edt():
    ii, jj = np.ogrid[:SHAPE[0], :SHAPE[1]]
    gt2d = ((ii - GT_I) ** 2 + (jj - GT_J) ** 2) <= GT_R**2
    edt = ndi.distance_transform_edt(~gt2d, sampling=(1.0, 1.0))
    return gt2d, edt


@pytest.fixture(scope="module")
def row():
    return {
        "case": "T99",
        "k_star": K_STAR,
        "spacing": [1.0, 1.0, 1.0],
        "drawn_marks": [
            {"id": 12, "kind": "blob+", "level": "coarse", "panel_idx": 1,
             "panel_px": [10, 10], "inplane_row_col": [14, 19],
             "projected_ijk": [GT_I, GT_J, K_STAR], "inside_gt_on_k_star": True},
            {"id": 15, "kind": "region", "level": "coarse", "panel_idx": 1,
             "panel_px": [50, 50], "inplane_row_col": [1, 1],
             "projected_ijk": [38, 2, K_STAR], "inside_gt_on_k_star": False},
        ],
    }


def _resp(cond, text, rep=0, error=None, stop_reason="stop"):
    return {"case": "T99", "condition": cond, "rep": rep, "raw_text": text,
            "stop_reason": stop_reason, "error": error}


# ---------------------------------------------------------------- parsing
def test_parse_s_ok():
    assert S.parse_answer("I compared the panels.\nANSWER: 12", "S") == {"status": "ok", "value": 12}
    assert S.parse_answer("ANSWER: 7\n", "S")["value"] == 7  # trailing newline tolerated
    assert S.parse_answer("ANSWER:  42 ", "S")["value"] == 42


def test_parse_s_fail():
    assert S.parse_answer("ANSWER: 12 but also 15", "S")["status"] == "parse_fail"
    assert S.parse_answer("The answer is 12", "S")["status"] == "parse_fail"
    assert S.parse_answer("ANSWER: mark 12", "S")["status"] == "parse_fail"
    assert S.parse_answer(None, "S")["status"] == "parse_fail"
    assert S.parse_answer("", "S")["status"] == "parse_fail"
    assert S.parse_answer("ANSWER: 12\nextra junk after", "S")["status"] == "parse_fail"


def test_parse_r_ok():
    assert S.parse_answer("ANSWER: (123, 456)", "R") == {"status": "ok", "value": (123.0, 456.0)}
    assert S.parse_answer("ANSWER: ( 123.5 , 45 )", "R")["value"] == (123.5, 45.0)


def test_parse_r_fail():
    assert S.parse_answer("ANSWER: 123, 456", "R")["status"] == "parse_fail"
    assert S.parse_answer("ANSWER: (a, b)", "R")["status"] == "parse_fail"
    assert S.parse_answer("ANSWER: (-5, 10)", "R")["status"] == "parse_fail"  # negatives rejected
    assert S.parse_answer("x=123 y=456", "R")["status"] == "parse_fail"


# ---------------------------------------------------------------- scoring S
def test_s_correct_mark_inside(row, gt_edt):
    gt2d, edt = gt_edt
    sc = S.score_call(_resp("S", "Mark 12 is inside.\nANSWER: 12"), row, gt2d, edt, GEOM_R)
    assert sc["status"] == "ok" and sc["hit"] is True and sc["distance_mm"] == 0.0
    assert sc["voxel"] == [GT_I, GT_J, K_STAR]


def test_s_wrong_mark_outside(row, gt_edt):
    gt2d, edt = gt_edt
    sc = S.score_call(_resp("S", "ANSWER: 15"), row, gt2d, edt, GEOM_R)
    assert sc["status"] == "ok" and sc["hit"] is False
    assert sc["distance_mm"] > 0.0  # EDT to the disc, in mm


def test_s_invalid_mark(row, gt_edt):
    gt2d, edt = gt_edt
    sc = S.score_call(_resp("S", "ANSWER: 999"), row, gt2d, edt, GEOM_R)
    assert sc["status"] == "invalid_mark" and sc["hit"] is False and sc["distance_mm"] is None


def test_s_parse_fail_is_miss(row, gt_edt):
    gt2d, edt = gt_edt
    sc = S.score_call(_resp("S", "I choose mark 12."), row, gt2d, edt, GEOM_R)
    assert sc["status"] == "parse_fail" and sc["hit"] is False and sc["distance_mm"] is None


def test_response_error_is_miss(row, gt_edt):
    gt2d, edt = gt_edt
    sc = S.score_call(_resp("S", None, error="transport failed"), row, gt2d, edt, GEOM_R)
    assert sc["status"] == "response_error" and sc["hit"] is False


# ---------------------------------------------------------------- scoring R
def test_r_point_inside(row, gt_edt):
    gt2d, edt = gt_edt
    # centre of the GT voxel's footprint on the panel (voxel_to_panel + half voxel)
    x, y = voxel_to_panel("axial", K_STAR, (GT_I, GT_J, K_STAR), GEOM_R)
    sc = S.score_call(_resp("R", f"ANSWER: ({x + 1.0}, {y + 1.0})"), row, gt2d, edt, GEOM_R)
    assert sc["status"] == "ok" and sc["hit"] is True and sc["distance_mm"] == 0.0
    assert sc["voxel"] == [GT_I, GT_J, K_STAR]


def test_r_point_near_but_outside_gt(row, gt_edt):
    gt2d, edt = gt_edt
    # voxel (38, 2) is far from the disc; its panel centre must map back to it
    x, y = voxel_to_panel("axial", K_STAR, (38, 2, K_STAR), GEOM_R)
    sc = S.score_call(_resp("R", f"ANSWER: ({x + 1.0}, {y + 1.0})"), row, gt2d, edt, GEOM_R)
    assert sc["status"] == "ok" and sc["hit"] is False and sc["distance_mm"] > 0.0


def test_r_outside_panel_is_miss(row, gt_edt):
    gt2d, edt = gt_edt
    sc = S.score_call(_resp("R", "ANSWER: (5000, 10)"), row, gt2d, edt, GEOM_R)
    assert sc["status"] == "outside_panel" and sc["hit"] is False and sc["distance_mm"] is None


def test_r_fractional_inside(row, gt_edt):
    gt2d, edt = gt_edt
    x, y = voxel_to_panel("axial", K_STAR, (GT_I, GT_J, K_STAR), GEOM_R)
    sc = S.score_call(_resp("R", f"ANSWER: ({x + 0.9:.1f}, {y + 0.9:.1f})"), row, gt2d, edt, GEOM_R)
    assert sc["hit"] is True


# ---------------------------------------------------------------- majority
def test_majority_strict(row):
    calls = [_resp("S", "ANSWER: 12", rep=0), _resp("S", "ANSWER: 12", rep=1),
             _resp("S", "ANSWER: 15", rep=2)]
    for c in calls:
        c.update(S.score_call(c, row, *gt_edt_fixture(), GEOM_R))
    kind, best, hit = S.majority_vote(calls)
    assert kind == "strict" and best == 12 and hit is True


def gt_edt_fixture():
    ii, jj = np.ogrid[:SHAPE[0], :SHAPE[1]]
    gt2d = ((ii - GT_I) ** 2 + (jj - GT_J) ** 2) <= GT_R**2
    return gt2d, ndi.distance_transform_edt(~gt2d, sampling=(1.0, 1.0))


def test_majority_tie_breaks_lowest(row):
    calls = [_resp("S", "ANSWER: 15", rep=0), _resp("S", "ANSWER: 12", rep=1),
             _resp("S", "ANSWER: 18", rep=2)]
    for c in calls:
        c.update(S.score_call(c, row, *gt_edt_fixture(), GEOM_R))
    kind, best, _hit = S.majority_vote(calls)
    assert kind == "tie" and best == 12  # 1 vote each -> lowest id


def test_majority_ignores_unparseable(row):
    calls = [_resp("S", "ANSWER: 12", rep=0), _resp("S", "no answer", rep=1),
             _resp("S", "ANSWER: 15", rep=2)]
    for c in calls:
        c.update(S.score_call(c, row, *gt_edt_fixture(), GEOM_R))
    kind, best, _hit = S.majority_vote(calls)
    assert kind == "tie" and best == 12  # only ok calls vote; 12 < 15


# ---------------------------------------------------------------- aggregation
def test_aggregate_and_gate(tmp_path, row):
    gt2d, edt = gt_edt_fixture()
    calls = [
        S.score_call(_resp("S", "ANSWER: 12", rep=0), row, gt2d, edt, GEOM_R),
        S.score_call(_resp("S", "ANSWER: 12", rep=1), row, gt2d, edt, GEOM_R),
        S.score_call(_resp("S", "ANSWER: 15", rep=2), row, gt2d, edt, GEOM_R),
        S.score_call(_resp("S", "oops", rep=3), row, gt2d, edt, GEOM_R),
        S.score_call(_resp("R", f"ANSWER: (39, 29)", rep=0), row, gt2d, edt, GEOM_R),
        S.score_call(_resp("R", "ANSWER: (5000, 10)", rep=1), row, gt2d, edt, GEOM_R),
    ]
    agg = S.aggregate(calls)
    assert agg["S"]["n_calls"] == 4
    assert agg["S"]["n_parse_fail"] == 1
    assert agg["S"]["hit_rate_mean_over_reps"] == 0.5   # 2 hits / 4 calls
    assert agg["S"]["majority_hit_rate_per_case"] == 1.0  # majority 12 -> hit
    assert agg["R"]["n_outside_panel"] == 1
    assert agg["R"]["hit_rate_mean_over_reps"] == 0.5
    lines = S.gate_evaluation(agg, coverage_rate=1.0)
    assert any("FAIL" in l for l in lines)  # S=0.5 < 0.80, coverage 1.0 -> run S-desc/S-ex
