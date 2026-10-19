"""Study A unit tests (fake services, synthetic GT): snap-to-mark, scorer math
(reused score.py GT-hit paths), aggregation, McNemar/bootstrap, resumability,
and the runner's response-line contract (service calls mocked).

Run:  uv run --project harness/server python -m pytest harness/study/tests -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi

STUDY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_DIR))

import local_point_study as runner  # noqa: E402
import score as phase2_score  # noqa: E402
import score_local_point as S  # noqa: E402
from segd.render import ViewGeom, voxel_to_panel  # noqa: E402

# synthetic volume [i, j, k] = (40, 30, 20); GT disc radius 5 at (i=20, j=15) on k*=10.
# Axial display orientation (segd.marks._INPLANE): r = rows-1-j, c = cols-1-i, so the
# GT voxel (i=20, j=15) sits at display (r=14, c=19) -> panel px (38, 28) at scale 2.
SHAPE = (40, 30, 20)
K_STAR = 10
GT_I, GT_J, GT_R = 20, 15, 5.0
GT_R_DISPLAY, GT_C_DISPLAY = 29 - GT_J, 39 - GT_I  # (14, 19)
PANEL_W, PANEL_H = 80, 60
GEOM_R = ViewGeom("axial", K_STAR, 30, 40, (0, 0, 30, 40), 2.0, PANEL_W, PANEL_H)
# combined S image: marked panel + gap + clean panel
PANEL_GAP = 16


@pytest.fixture(scope="module")
def gt_edt():
    ii, jj = np.ogrid[:SHAPE[0], :SHAPE[1]]
    gt2d = ((ii - GT_I) ** 2 + (jj - GT_J) ** 2) <= GT_R**2
    edt = ndi.distance_transform_edt(~gt2d, sampling=(1.0, 1.0))
    return gt2d, edt


@pytest.fixture(scope="module")
def row():
    """Manifest row with two drawn marks: 12 inside GT (voxel 20,15), 15 outside."""
    def mark(mid, ijk, inside, combined_px):
        return {"id": mid, "kind": "blob+", "level": "coarse", "panel_idx": 1,
                "panel_px": [combined_px[0] - (PANEL_W + PANEL_GAP), combined_px[1]],
                "combined_px": combined_px, "inplane_row_col": [ijk[0], ijk[1]],
                "projected_ijk": list(ijk), "inside_gt_on_k_star": inside}

    return {
        "case": "T99", "histotype": "F", "sequence": "T2", "k_star": K_STAR,
        "spacing": [1.0, 1.0, 1.0],
        "coverage_on_k_star": True,
        "drawn_marks": [
            mark(12, [GT_I, GT_J, K_STAR], True,
                 [PANEL_W + PANEL_GAP + 2.0 * GT_C_DISPLAY, 2.0 * GT_R_DISPLAY]),
            mark(15, [2, 2, K_STAR], False,
                 [PANEL_W + PANEL_GAP + 2.0 * (39 - 2), 2.0 * (29 - 2)]),
        ],
        "view_geom_R": {"axis": "axial", "slice_idx": K_STAR, "rows": 30, "cols": 40,
                        "crop": [0, 0, 30, 40], "scale": 2.0, "panel_w": PANEL_W, "panel_h": PANEL_H},
        "view_geom_S": {"axis": "axial", "slice_idx": K_STAR, "rows": 30, "cols": 40,
                        "crop": [0, 0, 30, 40], "scale": 2.0, "panel_w": PANEL_W, "panel_h": PANEL_H},
    }


def _resp(model, condition, points=None, parsed=None, parse_error=None, rep=0, error=None):
    return {"run": "r", "case": "T99", "histotype": "F", "sequence": "T2", "model": model,
            "condition": condition, "rep": rep, "raw": "raw output", "points": points or [],
            "parsed": parsed, "parse_error": parse_error, "latency_ms": 12.0, "error": error}


# ------------------------------------------------------------------ snapping
def test_snap_to_mark_nearest(row):
    marks = row["drawn_marks"]
    mx, my = marks[0]["combined_px"]
    mark, dist = S.snap_to_mark(mx + 1.0, my + 1.0, marks)
    assert mark["id"] == 12
    assert dist == pytest.approx(np.hypot(1.0, 1.0), abs=1e-6)


def test_snap_to_mark_tie_breaks_lowest_id(row):
    marks = row["drawn_marks"]
    # equidistant point between the two marks -> lowest id wins (12)
    mid = [(a + b) / 2 for a, b in zip(marks[0]["combined_px"], marks[1]["combined_px"])]
    mark, _dist = S.snap_to_mark(mid[0], mid[1], marks)
    assert mark["id"] == 12


def test_snap_to_mark_empty(row):
    assert S.snap_to_mark(0, 0, [])[0] is None


# ------------------------------------------------------------------ raw condition
# Raw-pointing frame: the sent PNG is NOT the panel — the image area is pasted at
# (40, 22) with a 26 px bottom margin (segd.render), so PNG pixel = panel px + (40, 22)
# and PNG dims = (panel_w + 40, panel_h + 48). The old convention (panel px sent
# directly) shifted every raw-condition answer.
PNG_ML, PNG_MT, PNG_MB = S.RAW_MARGIN_LEFT_PX, S.RAW_MARGIN_TOP_PX, S.RAW_MARGIN_BOTTOM_PX


def test_raw_point_inside_gt(row, gt_edt):
    gt2d, edt = gt_edt
    # panel px (39, 29) -> display voxel (14, 19) -> raw (i=20, j=15) inside GT;
    # sent as a whole-PNG pixel: +(40, 22) margin offset
    out = S.score_response(_resp("gui_actor", "raw",
                                 points=[{"x": 39.0 + PNG_ML, "y": 29.0 + PNG_MT}]),
                           row, gt2d, edt)
    assert len(out) == 1
    assert out[0]["rank"] == 0 and out[0]["status"] == "ok"
    assert out[0]["hit"] is True and out[0]["distance_mm"] == 0.0
    # the scored point is the converted PANEL pixel; the sent PNG pixel is kept
    assert out[0]["point"] == [39.0, 29.0]
    assert out[0]["png_point"] == [39.0 + PNG_ML, 29.0 + PNG_MT]


def test_raw_point_outside_gt_distance_edt(row, gt_edt):
    gt2d, edt = gt_edt
    out = S.score_response(_resp("gui_actor", "raw",
                                 points=[{"x": 75.0 + PNG_ML, "y": 55.0 + PNG_MT}]),
                           row, gt2d, edt)
    assert out[0]["hit"] is False
    assert out[0]["distance_mm"] == pytest.approx(float(edt[2, 2]), abs=0.01)


def test_raw_point_outside_panel(row, gt_edt):
    """score.py's outside_panel status is preserved through the reused path."""
    gt2d, edt = gt_edt
    out = S.score_response(_resp("gui_actor", "raw", points=[{"x": 500.0 + PNG_ML, "y": 5.0 + PNG_MT}]),
                           row, gt2d, edt)
    assert out[0]["status"] == "outside_panel" and out[0]["hit"] is False


def test_raw_point_in_margin_is_outside_panel(row, gt_edt):
    """Ruler/title-margin pixels (left of / above / below the image area) are misses."""
    gt2d, edt = gt_edt
    pw, ph = PANEL_W, PANEL_H
    for x, y in [(10.0, 30.0),               # left ruler margin
                 (10.0 + pw, 5.0),           # top title/ruler margin
                 (10.0 + pw, 22.0 + ph + 10.0)]:  # bottom margin (scale bar/title)
        out = S.score_response(_resp("gui_actor", "raw", points=[{"x": x, "y": y}]),
                               row, gt2d, edt)
        assert out[0]["status"] == "outside_panel" and out[0]["hit"] is False, (x, y)


def test_raw_no_points_is_abstain(row, gt_edt):
    gt2d, edt = gt_edt
    out = S.score_response(_resp("molmo2", "raw", points=[]), row, gt2d, edt)
    assert out[0]["status"] == "abstain" and out[0]["hit"] is False


def test_raw_hit3_uses_ranked_points(row, gt_edt):
    gt2d, edt = gt_edt
    # rank 0 misses, rank 2 hits -> hit@1 False, hit@3 True for this case (PNG coords)
    points = [{"x": 4.0 + PNG_ML, "y": 4.0 + PNG_MT},
              {"x": 6.0 + PNG_ML, "y": 6.0 + PNG_MT},
              {"x": 39.0 + PNG_ML, "y": 29.0 + PNG_MT}]
    out = S.score_response(_resp("gui_actor", "raw", points=points), row, gt2d, edt)
    assert [o["hit"] for o in out] == [False, False, True]
    assert S.case_flag_at_k(out, 1) == 0 and S.case_flag_at_k(out, 3) == 1


def test_response_error(row, gt_edt):
    gt2d, edt = gt_edt
    out = S.score_response(_resp("gui_actor", "raw", error="boom"), row, gt2d, edt)
    assert out[0]["status"] == "response_error" and out[0]["hit"] is False


# ------------------------------------------------------------------ marks condition
def test_marks_point_snaps_to_inside_mark(row, gt_edt):
    gt2d, edt = gt_edt
    mx, my = row["drawn_marks"][0]["combined_px"]
    out = S.score_response(_resp("gui_actor", "marks", points=[{"x": mx + 1.0, "y": my + 1.0}]),
                           row, gt2d, edt)
    assert out[0]["snapped_mark_id"] == 12
    assert out[0]["status"] == "ok" and out[0]["hit"] is True and out[0]["distance_mm"] == 0.0
    # raw (unsnapped) point is scored separately as a secondary diagnostic
    assert out[0]["raw_point_hit"] is True


def test_marks_point_snaps_to_outside_mark(row, gt_edt):
    gt2d, edt = gt_edt
    mx, my = row["drawn_marks"][1]["combined_px"]
    out = S.score_response(_resp("gui_actor", "marks", points=[{"x": mx + 1.0, "y": my + 1.0}]),
                           row, gt2d, edt)
    assert out[0]["snapped_mark_id"] == 15
    assert out[0]["hit"] is False and out[0]["distance_mm"] > 0.0


def test_marks_raw_point_on_clean_panel_is_none(row, gt_edt):
    """A returned point on the LEFT (clean) panel has no marked-panel voxel."""
    gt2d, edt = gt_edt
    out = S.score_response(_resp("gui_actor", "marks", points=[{"x": 10.0, "y": 10.0}]),
                           row, gt2d, edt)
    assert out[0]["raw_point_hit"] is None and out[0]["snapped_mark_id"] is not None


# ------------------------------------------------------------------ MedGemma conditions
def test_raw_box_centre_inside_gt(row, gt_edt):
    gt2d, edt = gt_edt
    # fractions of the FULL raw PNG (panel + 40 left + 22 top + 26 bottom margins)
    png_w, png_h = PANEL_W + PNG_ML, PANEL_H + PNG_MT + PNG_MB
    cx, cy = (2.0 * GT_C_DISPLAY + PNG_ML) / png_w, (2.0 * GT_R_DISPLAY + PNG_MT) / png_h
    box = [cy - 0.05, cx - 0.05, cy + 0.05, cx + 0.05]  # [y0, x0, y1, x1]
    out = S.score_response(_resp("medgemma_q4", "raw_box", parsed={"box": box, "confidence": 0.8}),
                           row, gt2d, edt)
    assert out[0]["status"] == "ok" and out[0]["hit"] is True
    # centre converted back to panel px, PNG px recorded
    assert out[0]["point"] == [round(2.0 * GT_C_DISPLAY, 2), round(2.0 * GT_R_DISPLAY, 2)]
    assert out[0]["png_point"] == [round(cx * png_w, 2), round(cy * png_h, 2)]


def test_raw_box_margin_centre_is_outside_panel(row, gt_edt):
    """A box centred in the ruler margin must not be silently projected into the panel."""
    gt2d, edt = gt_edt
    png_w, png_h = PANEL_W + PNG_ML, PANEL_H + PNG_MT + PNG_MB
    box = [0.5, 20.0 / png_w, 0.6, 30.0 / png_w]  # centre x ~ 25 px < 40 (left margin)
    out = S.score_response(_resp("medgemma_q4", "raw_box", parsed={"box": box}),
                           row, gt2d, edt)
    assert out[0]["status"] == "outside_panel" and out[0]["hit"] is False


def test_raw_box_bad_value_is_parse_fail(row, gt_edt):
    gt2d, edt = gt_edt
    out = S.score_response(_resp("medgemma_q4", "raw_box", parsed={"box": [0.1, 0.2, 1.5, 0.4]}),
                           row, gt2d, edt)
    assert out[0]["status"] == "parse_fail" and out[0]["hit"] is False
    out = S.score_response(_resp("medgemma_q4", "raw_box", parse_error="no JSON"), row, gt2d, edt)
    assert out[0]["status"] == "parse_fail"


def test_marks_box_snaps_centre(row, gt_edt):
    gt2d, edt = gt_edt
    mx, my = row["drawn_marks"][0]["combined_px"]
    cx, cy = mx / (2 * PANEL_W + PANEL_GAP), my / PANEL_H
    box = [cy - 0.05, cx - 0.05, cy + 0.05, cx + 0.05]
    out = S.score_response(_resp("medgemma_bf16", "marks_box", parsed={"box": box, "confidence": 0.7}),
                           row, gt2d, edt)
    assert out[0]["snapped_mark_id"] == 12
    assert out[0]["hit"] is True and out[0]["snap_distance_px"] == pytest.approx(0.0, abs=0.51)


def test_marks_id_direct_and_invalid(row, gt_edt):
    gt2d, edt = gt_edt
    ok = S.score_response(_resp("medgemma_bf16", "marks_id", parsed={"mark_id": 12, "confidence": 0.9}),
                          row, gt2d, edt)
    assert ok[0]["status"] == "ok" and ok[0]["hit"] is True
    bad = S.score_response(_resp("medgemma_bf16", "marks_id", parsed={"mark_id": 999}),
                           row, gt2d, edt)
    assert bad[0]["status"] == "invalid_mark" and bad[0]["hit"] is False
    miss = S.score_response(_resp("medgemma_bf16", "marks_id", parsed={}), row, gt2d, edt)
    assert miss[0]["status"] == "parse_fail"


def test_uncertain_is_abstain(row, gt_edt):
    gt2d, edt = gt_edt
    out = S.score_response(_resp("medgemma_q4", "raw_box", parsed={"answer": "uncertain"}),
                           row, gt2d, edt)
    assert out[0]["status"] == "abstain" and out[0]["hit"] is False


# ------------------------------------------------------------------ raw PNG frame (P1-1)
def _raw_render_row(base_row):
    """A REAL segd.render.render_raw_pointing output (margins included) + its manifest row."""
    from segd import render as segd_render

    ii, jj = np.ogrid[:SHAPE[0], :SHAPE[1]]
    disc = ((ii - GT_I) ** 2 + (jj - GT_J) ** 2 <= GT_R**2)
    img = np.broadcast_to(disc[:, :, None], SHAPE).astype(np.float32) * 800.0 + 100.0
    canvas, ri = segd_render.render_raw_pointing(img, "axial", K_STAR, crop="full",
                                                 spacing=(1.0, 1.0, 1.0))
    g = ri.views[0]
    assert canvas.size == (g.panel_w + 40, g.panel_h + 22 + 26)  # margins ARE in the PNG
    raw_row = dict(base_row, view_geom_R={"axis": g.axis, "slice_idx": g.slice_idx,
                                          "rows": g.rows, "cols": g.cols, "crop": list(g.crop),
                                          "scale": g.scale, "panel_w": g.panel_w,
                                          "panel_h": g.panel_h})
    return canvas, ri, raw_row


def test_raw_pointer_point_scores_in_png_frame(row, gt_edt):
    """GUI-Actor/Molmo2 answer in whole-PNG pixels: the scorer subtracts the raw margins
    before the Phase 2 panel_to_voxel path (a real render_raw_pointing output, not a
    margin-free synthetic panel)."""
    from segd.render import voxel_to_panel

    gt2d, edt = gt_edt
    canvas, ri, raw_row = _raw_render_row(row)
    g = ri.views[0]
    # the GT voxel (20, 15) mapped to panel px by the REAL geom, nudged inside its footprint
    px_v, py_v = voxel_to_panel("axial", K_STAR, (GT_I, GT_J, K_STAR), g)
    panel_x, panel_y = px_v + 1.0, py_v + 1.0
    png_x, png_y = panel_x + 40, panel_y + 22
    out = S.score_response(_resp("gui_actor", "raw", points=[{"x": png_x, "y": png_y}]),
                           raw_row, gt2d, edt)
    assert out[0]["status"] == "ok" and out[0]["hit"] is True and out[0]["distance_mm"] == 0.0
    assert out[0]["point"] == [round(panel_x, 2), round(panel_y, 2)]
    assert out[0]["png_point"] == [round(png_x, 2), round(png_y, 2)]
    # a margin pixel never maps into the image area
    margin = S.score_response(_resp("gui_actor", "raw",
                                    points=[{"x": 10.0, "y": png_y}]), raw_row, gt2d, edt)
    assert margin[0]["status"] == "outside_panel" and margin[0]["hit"] is False


def test_raw_box_scores_in_png_frame(row, gt_edt):
    """raw_box fractions are of the FULL PNG: scaled by PNG dims, then margin-offset."""
    from segd.render import voxel_to_panel

    gt2d, edt = gt_edt
    canvas, ri, raw_row = _raw_render_row(row)
    g = ri.views[0]
    png_w, png_h = g.panel_w + 40, g.panel_h + 22 + 26
    px_v, py_v = voxel_to_panel("axial", K_STAR, (GT_I, GT_J, K_STAR), g)
    panel_x, panel_y = px_v + 1.0, py_v + 1.0
    cx, cy = (panel_x + 40) / png_w, (panel_y + 22) / png_h
    box = [cy - 0.05, cx - 0.05, cy + 0.05, cx + 0.05]
    out = S.score_response(_resp("medgemma_q4", "raw_box", parsed={"box": box, "confidence": 0.9}),
                           raw_row, gt2d, edt)
    assert out[0]["status"] == "ok" and out[0]["hit"] is True
    assert out[0]["point"] == [round(panel_x, 2), round(panel_y, 2)]


# ------------------------------------------------------------------ marks_id strictness (P1-10)
def test_marks_id_strict_validation(row, gt_edt):
    """Non-integer mark ids are parse failures — never crashes, never coerced."""
    gt2d, edt = gt_edt
    ok = S.score_response(_resp("medgemma_bf16", "marks_id", parsed={"mark_id": 12}), row, gt2d, edt)
    assert ok[0]["status"] == "ok" and ok[0]["hit"] is True and ok[0]["snapped_mark_id"] == 12
    integral = S.score_response(_resp("medgemma_bf16", "marks_id", parsed={"mark_id": 12.0}),
                                row, gt2d, edt)
    assert integral[0]["status"] == "ok" and integral[0]["snapped_mark_id"] == 12
    for bad in ("unknown", 12.9, True, [12], None):
        out = S.score_response(_resp("medgemma_bf16", "marks_id", parsed={"mark_id": bad}),
                               row, gt2d, edt)
        assert out[0]["status"] == "parse_fail" and out[0]["hit"] is False, bad
        assert out[0]["snapped_mark_id"] is None


# ------------------------------------------------------------------ aggregation
def test_aggregate_metrics(row, gt_edt):
    gt2d, edt = gt_edt
    calls = []
    for rep, (px, py) in enumerate([(39.0, 29.0), (75.0, 55.0)]):  # rep0 hit, rep1 miss
        calls.extend(S.score_response(_resp("gui_actor", "raw",
                                            points=[{"x": px + PNG_ML, "y": py + PNG_MT}], rep=rep),
                                      row, gt2d, edt))
    a = S.aggregate(calls)
    assert a["n_calls"] == 2 and a["n_cases"] == 1
    # per-case majority over the two reps: 1 hit / 1 miss -> tie -> False (phase2 convention)
    assert a["hit1"] == 0.0 and a["hit3"] == 0.0
    assert a["hit1_mean_over_calls"] == 0.5
    assert a["n_abstain"] == 0 and a["n_parse_fail"] == 0
    assert a["latency_mean_ms"] == 12.0


def test_aggregate_single_rep(row, gt_edt):
    """Deterministic decoding (reps=1, the default): per-case hit@k = the call's flag."""
    gt2d, edt = gt_edt
    points = [{"x": 4.0 + PNG_ML, "y": 4.0 + PNG_MT},
              {"x": 6.0 + PNG_ML, "y": 6.0 + PNG_MT},
              {"x": 39.0 + PNG_ML, "y": 29.0 + PNG_MT}]
    calls = S.score_response(_resp("gui_actor", "raw", points=points), row, gt2d, edt)
    a = S.aggregate(calls)
    assert a["hit1"] == 0.0 and a["hit3"] == 1.0 and a["hit1_mean_over_calls"] == 0.0


def test_mcnemar_exact_known_values():
    assert S.mcnemar_exact(0, 0) == 1.0
    assert S.mcnemar_exact(1, 0) == 1.0
    assert S.mcnemar_exact(2, 0) == 0.5
    assert S.mcnemar_exact(9, 0) == pytest.approx(2 * 0.5**9)
    assert S.mcnemar_exact(3, 3) == 1.0


def test_paired_vs_baseline_math():
    local = {"A": True, "B": True, "C": False, "D": False}
    base = {"A": False, "B": True, "C": True, "D": False}
    comp = S.paired_vs_baseline(local, base, n_boot=50, seed=1)
    assert comp["n_paired_cases"] == 4
    assert comp["n_baseline_hit_local_miss"] == 1  # C
    assert comp["n_local_hit_baseline_miss"] == 1  # A
    assert comp["mcnemar_p"] == 1.0
    assert comp["diff_hit_rate"] == 0.0


# ------------------------------------------------- per-case McNemar summary (P1-7)
def test_per_case_hit1_matches_case_flag(row, gt_edt):
    """The McNemar outcome is the reported hit@1 summary (case_flag_at_k majority over
    reps), never 'first successful rep': a rep-0 hit must not carry a case whose other
    reps missed."""
    gt2d, edt = gt_edt
    hit = _resp("gui_actor", "raw", points=[{"x": 79.0, "y": 51.0}], rep=0)
    miss = _resp("gui_actor", "raw", points=[{"x": 115.0, "y": 77.0}], rep=1)
    calls = S.score_response(hit, row, gt2d, edt) + S.score_response(miss, row, gt2d, edt)
    assert S.per_case_hit1(calls)["T99"] is False  # 1 hit / 1 miss -> tie -> miss
    assert S.case_flag_at_k(calls, 1) == 0
    # 2 hits / 1 miss -> majority hit; row order must not matter
    hit2 = S.score_response(_resp("gui_actor", "raw", points=[{"x": 79.0, "y": 51.0}], rep=2),
                            row, gt2d, edt)
    calls3 = S.score_response(hit, row, gt2d, edt) + S.score_response(miss, row, gt2d, edt) + hit2
    assert S.per_case_hit1(calls3)["T99"] is True
    shuffled = list(reversed(calls3))
    assert S.per_case_hit1(shuffled) == S.per_case_hit1(calls3)


def test_per_case_hit1_counts_error_and_parse_reps_as_misses(row, gt_edt):
    gt2d, edt = gt_edt
    hit = S.score_response(_resp("gui_actor", "raw", points=[{"x": 79.0, "y": 51.0}], rep=0),
                           row, gt2d, edt)
    err = S.score_response(_resp("gui_actor", "raw", rep=1, error="timeout"), row, gt2d, edt)
    parse = S.score_response(_resp("medgemma_bf16", "marks_id", parsed={"mark_id": "unknown"}, rep=2),
                             row, gt2d, edt)
    assert S.per_case_hit1(hit + err)["T99"] is False
    assert S.per_case_hit1(hit + err + parse)["T99"] is False  # 1 hit / 2 misses
    hit2 = S.score_response(_resp("gui_actor", "raw", points=[{"x": 79.0, "y": 51.0}], rep=2),
                            row, gt2d, edt)
    assert S.per_case_hit1(hit + err + parse + hit2)["T99"] is True  # 2 hits / 2 misses -> majority


def test_rep_hit_majority_frontier_convention():
    """The frontier side aggregates per-rep hit flags the same way (ties -> miss)."""
    assert S.rep_hit_majority([{"rep": 0, "hit": True}, {"rep": 1, "hit": False}]) is False
    assert S.rep_hit_majority([{"rep": 0, "hit": True}, {"rep": 1, "hit": True}]) is True
    assert S.rep_hit_majority([{"rep": 0, "hit": False}, {"rep": 1, "hit": False},
                               {"rep": 2, "hit": True}]) is False
    assert S.rep_hit_majority([{"rep": 0, "hit": True}, {"rep": 1, "hit": True},
                               {"rep": 2, "hit": False}]) is True


def test_bootstrap_ci_brackets_point_estimate():
    flags = [1] * 8 + [0] * 2  # 0.8
    lo, hi = S.bootstrap_ci(flags, n_boot=400, seed=7)
    assert lo <= 0.8 <= hi and 0.0 <= lo and hi <= 1.0


# ------------------------------------------------------------------ resumability
def test_runner_resume_keeps_ok_drops_errors(tmp_path):
    resp = tmp_path / "responses.jsonl"
    ok_line = json.dumps({"run": "r", "case": "A", "model": "gui_actor", "condition": "raw",
                          "rep": 0, "ok": True})
    err_line = json.dumps({"run": "r", "case": "B", "model": "gui_actor", "condition": "raw",
                           "rep": 0, "ok": False, "error": "boom"})
    resp.write_text(ok_line + "\n" + err_line + "\n{malformed}\n")
    done = runner.load_done_and_prune(resp)
    assert done == {("A", "gui_actor", "raw", 0)}
    kept = [json.loads(l) for l in resp.read_text().splitlines()]
    assert len(kept) == 1 and kept[0]["case"] == "A"  # error + malformed rows dropped


# ------------------------------------------------------------------ runner contract
def test_build_call_pointd_contract(row, monkeypatch, tmp_path):
    """The runner's response line carries the documented fields (service mocked)."""
    import io as _io

    from PIL import Image

    image_path = tmp_path / "img.png"
    Image.new("RGB", (8, 8)).save(image_path)
    row = dict(row, image_R=str(image_path), image_S=str(image_path))

    captured = {}

    def fake_call_pointd(url, image, query, k, timeout_s):
        captured.update({"url": url, "query": query, "k": k})
        return {"points": [{"x": 1.0, "y": 2.0, "score": 0.5, "rank": 0}], "raw": "out",
                "latency_ms": 5.0, "model": "microsoft/GUI-Actor-2B-Qwen2-VL", "version": "1.0.0",
                "image_w": 8, "image_h": 8, "verifier": None}

    monkeypatch.setattr(runner, "call_pointd", fake_call_pointd)
    opts = runner.parse_args(["--run", "x", "--prepare-dir", str(tmp_path), "--models", "gui_actor",
                              "--conditions", "raw"])
    (tmp_path / "manifest.jsonl").write_text(json.dumps(row) + "\n")
    opts.prepare_dir = str(tmp_path)

    line = runner.build_call(row, "gui_actor", "raw", 0, opts)
    assert captured["k"] == 3 and captured["query"].strip() == "Point to the fibrous soft-tissue tumor."
    assert line["ok"] is True and line["error"] is None
    assert line["points"][0]["x"] == 1.0
    assert line["model_id"] == "microsoft/GUI-Actor-2B-Qwen2-VL"
    assert line["prompt_template"] == "harness/study/prompts/local_point_raw.md"
    assert len(line["prompt_sha256"]) == 64
    assert line["histotype"] == "F" and line["condition"] == "raw" and line["rep"] == 0


def test_build_call_error_recorded(row, monkeypatch, tmp_path):
    from PIL import Image

    image_path = tmp_path / "img.png"
    Image.new("RGB", (8, 8)).save(image_path)
    row = dict(row, image_R=str(image_path), image_S=str(image_path))

    def failing(url, image, query, k, timeout_s):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(runner, "call_pointd", failing)
    opts = runner.parse_args(["--run", "x", "--prepare-dir", str(tmp_path), "--models", "gui_actor",
                              "--conditions", "raw"])
    opts.prepare_dir = str(tmp_path)
    line = runner.build_call(row, "gui_actor", "raw", 0, opts)
    assert line["ok"] is False and "connection refused" in line["error"]
