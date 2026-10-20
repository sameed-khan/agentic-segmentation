"""Study A unit tests (fake services, synthetic GT): snap-to-mark, scorer math
(reused score.py GT-hit paths), aggregation, McNemar/bootstrap, resumability,
and the runner's response-line contract (service calls mocked).

Run:  uv run --project harness/server python -m pytest harness/study/tests -q
"""
from __future__ import annotations

import hashlib
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
    # a marked-panel point is never folded; pointd rows carry no box scale
    assert out[0]["folded"] is False and out[0]["box_scale"] is None


def test_marks_point_snaps_to_outside_mark(row, gt_edt):
    gt2d, edt = gt_edt
    mx, my = row["drawn_marks"][1]["combined_px"]
    out = S.score_response(_resp("gui_actor", "marks", points=[{"x": mx + 1.0, "y": my + 1.0}]),
                           row, gt2d, edt)
    assert out[0]["snapped_mark_id"] == 15
    assert out[0]["hit"] is False and out[0]["distance_mm"] > 0.0


def test_marks_raw_point_on_clean_panel_none_when_fold_off(row, gt_edt):
    """Fold OFF (legacy): a returned point on the LEFT (clean) panel has no marked-panel
    voxel, so the raw diagnostic is None. With the fold on the diagnostic is scored on
    the point's own panel coordinates (test_marks_fold_raw_diagnostic...)."""
    gt2d, edt = gt_edt
    out = S.score_response(_resp("gui_actor", "marks", points=[{"x": 10.0, "y": 10.0}]),
                           row, gt2d, edt, clean_panel_fold=False)
    assert out[0]["raw_point_hit"] is None and out[0]["snapped_mark_id"] is not None


# ------------------------------------------------------------------ clean-panel fold
def test_marks_fold_moves_clean_panel_point_to_marked_panel(row, gt_edt):
    """Fold on (default): a point whose x lies on the clean panel is translated to the
    same position on the marked panel (x + panel_w + gap, y kept) before snapping."""
    gt2d, edt = gt_edt
    # clean-panel mirror of drawn mark 15 (outside GT): 170 - (80 + 16) = 74
    out = S.score_response(_resp("molmo2", "marks", points=[{"x": 74.0, "y": 54.0}]),
                           row, gt2d, edt)
    assert out[0]["folded"] is True and out[0]["snapped_mark_id"] == 15
    assert out[0]["hit"] is False
    # fold off: the same point stays on the clean panel and snaps to mark 12 (hit)
    off = S.score_response(_resp("molmo2", "marks", points=[{"x": 74.0, "y": 54.0}]),
                           row, gt2d, edt, clean_panel_fold=False)
    assert off[0]["folded"] is False and off[0]["snapped_mark_id"] == 12
    assert off[0]["hit"] is True


def test_marks_fold_gap_point_not_folded(row, gt_edt):
    """Points in the 16 px gap are not folded (they snap as is)."""
    gt2d, edt = gt_edt
    out = S.score_response(_resp("molmo2", "marks", points=[{"x": 88.0, "y": 28.0}]),
                           row, gt2d, edt)
    assert out[0]["folded"] is False and out[0]["snapped_mark_id"] == 12


def test_marks_fold_raw_diagnostic_uses_point_panel_coords(row, gt_edt):
    """The raw (unsnapped) diagnostic scores the point on ITS OWN panel coordinates: the
    clean and marked panels share the same view geometry."""
    gt2d, edt = gt_edt
    # clean-panel mirror of mark 12 (inside GT): 134 - (80 + 16) = 38 -> the GT position
    out = S.score_response(_resp("molmo2", "marks", points=[{"x": 38.0, "y": 28.0}]),
                           row, gt2d, edt)
    assert out[0]["folded"] is True and out[0]["snapped_mark_id"] == 12
    assert out[0]["hit"] is True
    assert out[0]["raw_point_hit"] is True and out[0]["raw_point_distance_mm"] == 0.0
    # the raw `point` field keeps the sent (clean-panel) coordinates
    assert out[0]["point"] == [38.0, 28.0]


def test_marks_dedup_by_snapped_mark_fold_on(row, gt_edt):
    """Fold on: later points snapping to an already-used mark are dropped and do not
    consume a rank, so hit@3 means 3 distinct marks."""
    gt2d, edt = gt_edt
    points = [{"x": 38.0, "y": 28.0},   # folds onto mark 12
              {"x": 135.0, "y": 29.0},  # marked panel: snaps to 12 again -> dropped
              {"x": 170.0, "y": 54.0}]  # mark 15
    out = S.score_response(_resp("molmo2", "marks", points=points), row, gt2d, edt)
    assert [o["rank"] for o in out] == [0, 1]
    assert [o["snapped_mark_id"] for o in out] == [12, 15]
    assert [o["folded"] for o in out] == [True, False]
    # fold off keeps every rank (pre-fold behaviour)
    off = S.score_response(_resp("molmo2", "marks", points=points), row, gt2d, edt,
                           clean_panel_fold=False)
    assert [o["rank"] for o in off] == [0, 1, 2]
    assert [o["snapped_mark_id"] for o in off] == [12, 12, 15]


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
    # centre converted back to panel px, PNG px recorded; [0,1] values = unit scale
    assert out[0]["point"] == [round(2.0 * GT_C_DISPLAY, 2), round(2.0 * GT_R_DISPLAY, 2)]
    assert out[0]["png_point"] == [round(cx * png_w, 2), round(cy * png_h, 2)]
    assert out[0]["box_scale"] == "unit"


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
    # a value outside [0, 1000] fits neither declared frame -> parse failure
    out = S.score_response(_resp("medgemma_q4", "raw_box", parsed={"box": [0.1, 0.2, 1500.0, 0.4]}),
                           row, gt2d, edt)
    assert out[0]["status"] == "parse_fail" and out[0]["hit"] is False
    assert out[0]["box_scale"] is None
    out = S.score_response(_resp("medgemma_q4", "raw_box", parse_error="no JSON"), row, gt2d, edt)
    assert out[0]["status"] == "parse_fail"


def test_raw_box_per1000_scale_accepted(row, gt_edt):
    """Gemma convention: boxes on the 0-1000 integer scale, e.g. [350, 230, 580, 530]."""
    gt2d, edt = gt_edt
    out = S.score_response(_resp("medgemma_q4", "raw_box",
                                 parsed={"box": [350, 230, 580, 530], "confidence": 0.98}),
                           row, gt2d, edt)
    assert out[0]["box_scale"] == "per1000" and out[0]["status"] == "ok"
    png_w, png_h = PANEL_W + PNG_ML, PANEL_H + PNG_MT + PNG_MB
    cx, cy = (230 + 530) / 2 / 1000, (350 + 580) / 2 / 1000
    assert out[0]["point"] == [round(cx * png_w - PNG_ML, 2), round(cy * png_h - PNG_MT, 2)]


def test_raw_box_per1000_hits_gt(row, gt_edt):
    gt2d, edt = gt_edt
    png_w, png_h = PANEL_W + PNG_ML, PANEL_H + PNG_MT + PNG_MB
    cx, cy = (2.0 * GT_C_DISPLAY + PNG_ML) / png_w, (2.0 * GT_R_DISPLAY + PNG_MT) / png_h
    box = [round((cy - 0.05) * 1000), round((cx - 0.05) * 1000),
           round((cy + 0.05) * 1000), round((cx + 0.05) * 1000)]
    out = S.score_response(_resp("medgemma_bf16", "raw_box", parsed={"box": box}), row, gt2d, edt)
    assert out[0]["box_scale"] == "per1000"
    assert out[0]["hit"] is True and out[0]["distance_mm"] == 0.0


def test_raw_box_slight_overshoot_is_parse_fail_not_per1000(row, gt_edt):
    """A unit-scale box that slightly overshoots (e.g. [0.1, 0.2, 1.02, 0.4]) is a parse
    failure — NOT reinterpreted as per1000 (that would shrink it to a tiny box near
    (0,0), an accidental outside_panel miss or, for marks_box, a snap from the corner)."""
    gt2d, edt = gt_edt
    out = S.score_response(_resp("medgemma_q4", "raw_box", parsed={"box": [0.1, 0.2, 1.5, 0.4]}),
                           row, gt2d, edt)
    assert out[0]["status"] == "parse_fail" and out[0]["box_scale"] is None


def test_box_degenerate_is_parse_fail(row, gt_edt):
    """y1 < y0 or x1 < x0 is a parse failure on either scale."""
    gt2d, edt = gt_edt
    out = S.score_response(_resp("medgemma_q4", "raw_box", parsed={"box": [0.1, 0.2, 0.05, 0.4]}),
                           row, gt2d, edt)
    assert out[0]["status"] == "parse_fail" and out[0]["box_scale"] is None
    out = S.score_response(_resp("medgemma_q4", "raw_box", parsed={"box": [580, 530, 350, 230]}),
                           row, gt2d, edt)
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
    assert out[0]["box_scale"] == "unit" and out[0]["folded"] is False


def test_marks_box_folds_clean_panel_centre(row, gt_edt):
    """marks_box: a box centre on the clean panel is folded before snapping; the raw box
    centre is scored on its own (clean-panel) coordinates."""
    gt2d, edt = gt_edt
    cx, cy = 38.0 / (2 * PANEL_W + PANEL_GAP), 28.0 / PANEL_H
    box = [cy - 0.05, cx - 0.05, cy + 0.05, cx + 0.05]
    out = S.score_response(_resp("medgemma_q4", "marks_box", parsed={"box": box}), row, gt2d, edt)
    assert out[0]["folded"] is True and out[0]["snapped_mark_id"] == 12
    assert out[0]["hit"] is True
    assert out[0]["raw_point_hit"] is True          # scored on the clean-panel coords
    assert out[0]["point"] == [38.0, 28.0]
    off = S.score_response(_resp("medgemma_q4", "marks_box", parsed={"box": box}),
                           row, gt2d, edt, clean_panel_fold=False)
    assert off[0]["folded"] is False and off[0]["snapped_mark_id"] == 12


def test_marks_box_per1000_snaps(row, gt_edt):
    gt2d, edt = gt_edt
    mx, my = row["drawn_marks"][0]["combined_px"]
    cx, cy = mx / (2 * PANEL_W + PANEL_GAP), my / PANEL_H
    box = [round((cy - 0.05) * 1000), round((cx - 0.05) * 1000),
           round((cy + 0.05) * 1000), round((cx + 0.05) * 1000)]
    out = S.score_response(_resp("medgemma_q4", "marks_box", parsed={"box": box}), row, gt2d, edt)
    assert out[0]["box_scale"] == "per1000" and out[0]["folded"] is False
    assert out[0]["snapped_mark_id"] == 12 and out[0]["hit"] is True


def test_box_scale_absent_for_non_box_rows(row, gt_edt):
    gt2d, edt = gt_edt
    out = S.score_response(_resp("medgemma_bf16", "marks_id", parsed={"mark_id": 12}),
                           row, gt2d, edt)
    assert out[0]["box_scale"] is None and out[0]["folded"] is False


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
    # distinct rep for the added hit (it must NOT share the parse rep) so the fixture is
    # 2 hits / 2 misses over 4 DISTINCT reps -> tie -> miss by the majority convention
    hit2 = S.score_response(_resp("gui_actor", "raw", points=[{"x": 79.0, "y": 51.0}], rep=3),
                            row, gt2d, edt)
    assert S.per_case_hit1(hit + err + parse + hit2)["T99"] is False  # 2 hits / 2 misses -> tie
    # a third hit rep gives 3 hits / 2 misses -> majority
    hit3 = S.score_response(_resp("gui_actor", "raw", points=[{"x": 79.0, "y": 51.0}], rep=4),
                            row, gt2d, edt)
    assert S.per_case_hit1(hit + err + parse + hit2 + hit3)["T99"] is True


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
    done, versions = runner.load_done_and_prune(resp)
    assert done == {("A", "gui_actor", "raw", 0)}
    assert versions == {"gui_actor": {None}}  # service_version for the resume check (M4)
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


# ------------------------------------------------------------------ prompt overrides

def test_parse_args_prompt_override_flag(tmp_path):
    """--prompt is repeatable; COND=PATH pairs land in opts.prompt_overrides."""
    variant = tmp_path / "v.md"
    variant.write_text("V")
    opts = runner.parse_args(["--run", "x", "--prepare-dir", str(tmp_path),
                              "--prompt", f"raw={variant}", "--prompt", f"marks={variant}"])
    assert opts.prompt_overrides == {"raw": variant, "marks": variant}


def test_prompt_override_resolution_and_errors(tmp_path):
    """PATH is absolute or repo-root relative; unknown condition / missing file /
    malformed spec are SystemExit with a clear message."""
    variant = tmp_path / "molmo_marks_v2.md"
    variant.write_text("VARIANT")
    assert runner.resolve_prompt_overrides([f"marks={variant}"], runner.PROJECT_ROOT) == \
        {"marks": variant}
    (tmp_path / "rel.md").write_text("REL")
    assert runner.resolve_prompt_overrides(["raw=rel.md"], tmp_path) == \
        {"raw": tmp_path / "rel.md"}
    with pytest.raises(SystemExit, match="unknown --prompt condition"):
        runner.resolve_prompt_overrides(["nope=x.md"], tmp_path)
    with pytest.raises(SystemExit, match="not found at"):
        runner.resolve_prompt_overrides([f"raw={tmp_path / 'missing.md'}"], tmp_path)
    with pytest.raises(SystemExit, match="COND=PATH"):
        runner.resolve_prompt_overrides(["justapath.md"], tmp_path)


def test_build_call_prompt_override_contract(row, monkeypatch, tmp_path):
    """An override template is used verbatim: prompt_template/prompt_sha256 record the
    override path + content hash; without an override the default label is kept."""
    from PIL import Image

    image_path = tmp_path / "img.png"
    Image.new("RGB", (8, 8)).save(image_path)
    r = dict(row, image_R=str(image_path), image_S=str(image_path))
    variant = tmp_path / "marks_v2.md"
    variant.write_text("MARKS-VARIANT: point at the {{target}}.")

    def fake_call_pointd(url, image, query, k, timeout_s):
        return {"points": [{"x": 1.0, "y": 2.0}], "raw": "out", "latency_ms": 5.0,
                "model": "m", "version": "1", "verifier": None}

    monkeypatch.setattr(runner, "call_pointd", fake_call_pointd)
    opts = runner.parse_args(["--run", "x", "--prepare-dir", str(tmp_path), "--models", "gui_actor",
                              "--conditions", "marks", "--prompt", f"marks={variant}"])
    opts.prepare_dir = str(tmp_path)
    line = runner.build_call(r, "gui_actor", "marks", 0, opts)
    assert line["prompt_template"] == str(variant)
    assert line["prompt_sha256"] == hashlib.sha256(variant.read_bytes()).hexdigest()
    assert line["prompt_text"].strip() == "MARKS-VARIANT: point at the fibrous soft-tissue tumor."
    assert line["ok"] is True
    opts2 = runner.parse_args(["--run", "x", "--prepare-dir", str(tmp_path), "--models", "gui_actor",
                               "--conditions", "marks"])
    opts2.prepare_dir = str(tmp_path)
    line2 = runner.build_call(r, "gui_actor", "marks", 0, opts2)
    assert line2["prompt_template"] == "harness/study/prompts/local_point_marks.md"
    assert line2["prompt_sha256"] == hashlib.sha256(
        (runner.PROMPTS_DIR / "local_point_marks.md").read_bytes()).hexdigest()


def test_prompt_provenance_records_effective_templates(tmp_path):
    """meta.json `prompts` records the effective template path + sha256 per condition; an
    override replaces the default entry."""
    variant = tmp_path / "raw_v9.md"
    variant.write_text("RAW-VARIANT")
    prov = runner.prompt_provenance({"raw": variant})
    assert set(prov) == set(runner.ALL_CONDITIONS)
    assert prov["raw"] == {"path": str(variant),
                           "sha256": hashlib.sha256(variant.read_bytes()).hexdigest()}
    assert prov["marks"] == {"path": "harness/study/prompts/local_point_marks.md",
                             "sha256": hashlib.sha256(
                                 (runner.PROMPTS_DIR / "local_point_marks.md").read_bytes()).hexdigest()}


def test_runner_refuses_prompt_override_drift_on_resume(tmp_path):
    """The effective per-condition templates are part of the immutable-on-resume config
    (IMMUTABLE_META_KEYS `prompts`): resuming with different templates must fail."""
    run_dir = tmp_path / "studyA" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "responses.jsonl").write_text(
        json.dumps({"case": "A", "model": "gui_actor", "condition": "raw", "rep": 0,
                    "ok": True}) + "\n")
    prior = {"prompts": {"raw": {"path": "harness/study/prompts/local_point_raw.md",
                                 "sha256": "aaa"}}}
    (run_dir / "meta.json").write_text(json.dumps(prior))
    meta = {"prompts": {"raw": {"path": "variants/raw_v9.md", "sha256": "bbb"}}}
    with pytest.raises(SystemExit, match="drifted"):
        runner.check_immutable_config(run_dir, meta, {("A", "gui_actor", "raw", 0)})
    runner.check_immutable_config(run_dir, dict(prior), {("A", "gui_actor", "raw", 0)})


# ------------------------------------------------------------------ missing tests (review m6)
def test_marks_fold_gap_point_raw_diagnostic_is_none(row, gt_edt):
    """Fold on: a gap point is not folded and has no marked-panel voxel -> the raw
    diagnostic is None (it snaps from the gap, but the unsnapped point is unscoreable)."""
    gt2d, edt = gt_edt
    out = S.score_response(_resp("molmo2", "marks", points=[{"x": 88.0, "y": 28.0}]),
                           row, gt2d, edt)
    assert out[0]["folded"] is False
    assert out[0]["snapped_mark_id"] == 12  # still snaps (nearest mark from the gap)
    assert out[0]["raw_point_hit"] is None and out[0]["raw_point_distance_mm"] is None


def test_marks_dedup_with_no_drawn_marks(row, gt_edt):
    """Dedup when the snap returns None (no drawn marks in the manifest row): every rank
    is a parse_fail, no dedup happens, ranks still consume 0..k-1."""
    gt2d, edt = gt_edt
    empty = dict(row, drawn_marks=[])
    points = [{"x": 38.0, "y": 28.0}, {"x": 38.0, "y": 28.0}, {"x": 38.0, "y": 28.0}]
    out = S.score_response(_resp("molmo2", "marks", points=points), empty, gt2d, edt)
    assert [o["rank"] for o in out] == [0, 1, 2]
    assert all(o["status"] == "parse_fail" and o["snapped_mark_id"] is None for o in out)


def test_clean_panel_fold_half_open_at_panel_w(row, gt_edt):
    """The clean panel is the half-open range [0, panel_w): x = panel_w - 1e-9 folds,
    x = panel_w (the first gap pixel) does not."""
    assert S.clean_panel_fold_px(PANEL_W - 1e-9, 5.0, GEOM_R, True)[2] is True
    assert S.clean_panel_fold_px(float(PANEL_W), 5.0, GEOM_R, True)[2] is False
    gt2d, edt = gt_edt
    eps = S.score_response(_resp("molmo2", "marks", points=[{"x": PANEL_W - 1e-9, "y": 28.0}]),
                           row, gt2d, edt)
    assert eps[0]["folded"] is True
    gap = S.score_response(_resp("molmo2", "marks", points=[{"x": float(PANEL_W), "y": 28.0}]),
                           row, gt2d, edt)
    assert gap[0]["folded"] is False


def test_marks_box_straddling_centre_in_gap_snaps_unfolded(row, gt_edt):
    """DECLARED: a marks_box that straddles both panels has its centre in the 16 px gap;
    a gap centre is NOT folded and snaps from its gap position (nearest mark, Euclidean).
    The raw diagnostic is None (no panel voxel in the gap)."""
    gt2d, edt = gt_edt
    total_w = 2 * PANEL_W + PANEL_GAP
    cy = 28.0 / PANEL_H
    box = [cy - 0.02, 70.0 / total_w, cy + 0.02, 100.0 / total_w]  # x spans 70..100 px
    out = S.score_response(_resp("medgemma_bf16", "marks_box", parsed={"box": box}),
                           row, gt2d, edt)
    assert out[0]["folded"] is False
    assert out[0]["snapped_mark_id"] == 12  # mark 12 (x=134) is nearer than mark 15 (x=170)
    assert out[0]["hit"] is True
    assert out[0]["raw_point_hit"] is None  # centre in the gap: no panel voxel


# ------------------------------------------------------------------ fold aggregation (M2)
def test_aggregate_fold_rate_and_nofold_sensitivity(row, gt_edt):
    """aggregate() gains n_folded/fold_rate over rank-0 rows; fold_off_sensitivity()
    applies the same case-level convention to the fold-off rows."""
    gt2d, edt = gt_edt
    # point on the clean panel (folds) + a second rep on the marked panel (no fold)
    calls = (S.score_response(_resp("molmo2", "marks", points=[{"x": 38.0, "y": 28.0}], rep=0),
                              row, gt2d, edt, True)
             + S.score_response(_resp("molmo2", "marks", points=[{"x": 135.0, "y": 29.0}], rep=0),
                                row, gt2d, edt, True))
    a = S.aggregate(calls)
    assert a["n_calls"] == 2 and a["n_cases"] == 1
    assert a["n_folded"] == 1 and a["fold_rate"] == 0.5
    # fold off: the clean-panel point snaps to a different mark -> sensitivity differs
    off = S.score_response(_resp("molmo2", "marks", points=[{"x": 74.0, "y": 54.0}], rep=0),
                           row, gt2d, edt, False)
    sens = S.fold_off_sensitivity(off)
    assert sens["hit1_nofold"] == 1.0 and sens["hit3_nofold"] == 1.0
    assert S.score_response(_resp("molmo2", "marks", points=[{"x": 74.0, "y": 54.0}], rep=0),
                            row, gt2d, edt, True)[0]["hit"] is False  # fold changes this case


def test_raw_condition_nofold_sensitivity_equals_primary(row, gt_edt):
    """The fold only applies to the marks conditions: for `raw`, nofold == primary."""
    gt2d, edt = gt_edt
    calls = S.score_response(_resp("gui_actor", "raw",
                                   points=[{"x": 39.0 + PNG_ML, "y": 29.0 + PNG_MT}]),
                             row, gt2d, edt)
    assert S.fold_off_sensitivity(list(S.score_response(
        _resp("gui_actor", "raw", points=[{"x": 39.0 + PNG_ML, "y": 29.0 + PNG_MT}]),
        row, gt2d, edt, False)).copy()) == {"hit1_nofold": 1.0, "hit3_nofold": 1.0}


# ------------------------------------------------------------------ latency semantics (M3)
def test_scored_row_latency_uses_service_latency(row, gt_edt):
    """The scorer reports service_latency_ms; the legacy latency_ms stays as recorded."""
    gt2d, edt = gt_edt
    resp = _resp("molmo2", "raw", points=[{"x": 79.0, "y": 51.0}])
    resp["latency_ms"] = 90.0           # legacy runner field
    resp["service_latency_ms"] = 30.0   # server-reported
    out = S.score_response(resp, row, gt2d, edt)
    assert out[0]["service_latency_ms"] == 30.0 and out[0]["latency_ms"] == 90.0
    assert S.aggregate(out)["latency_mean_ms"] == 30.0
    # rows recorded before service_latency_ms: fall back to legacy latency_ms
    old = S.score_response(_resp("gui_actor", "raw", points=[{"x": 79.0, "y": 51.0}]),
                           row, gt2d, edt)
    assert old[0]["service_latency_ms"] == 12.0
    assert S.aggregate(old)["latency_mean_ms"] == 12.0
    # an error row with neither field reports None, not a fabricated number
    err_resp = _resp("gui_actor", "raw", rep=1, error="boom")
    err_resp["latency_ms"] = None
    err = S.score_response(err_resp, row, gt2d, edt)
    assert err[0]["service_latency_ms"] is None


# ------------------------------------------------------------------ box rule + unwrap (m1)
def test_per1000_requires_integer_values(row, gt_edt):
    gt2d, edt = gt_edt
    out = S.score_response(_resp("medgemma_q4", "raw_box", parsed={"box": [350.5, 230, 580, 530]}),
                           row, gt2d, edt)
    assert out[0]["status"] == "parse_fail" and out[0]["box_scale"] is None


def test_box_bool_values_are_parse_fail(row, gt_edt):
    """true/false are not numbers for box purposes (bool passes isinstance(int))."""
    gt2d, edt = gt_edt
    out = S.score_response(_resp("medgemma_q4", "raw_box",
                                 parsed={"box": [True, 0.2, 0.9, 0.4]}), row, gt2d, edt)
    assert out[0]["status"] == "parse_fail" and out[0]["box_scale"] is None
    out = S.score_response(_resp("medgemma_q4", "raw_box",
                                 parsed={"box": [500, 200, False, 400]}), row, gt2d, edt)
    assert out[0]["status"] == "parse_fail"


def test_nested_box_list_is_unwrapped(row, gt_edt):
    """MedGemma emits a single nested list [[y0, x0, y1, x1]]: unwrapped to the inner
    four numbers (declared normalisation, box_unwrapped=true); exactly one level."""
    gt2d, edt = gt_edt
    png_w, png_h = PANEL_W + PNG_ML, PANEL_H + PNG_MT + PNG_MB
    cx, cy = (2.0 * GT_C_DISPLAY + PNG_ML) / png_w, (2.0 * GT_R_DISPLAY + PNG_MT) / png_h
    box = [cy - 0.05, cx - 0.05, cy + 0.05, cx + 0.05]
    out = S.score_response(_resp("medgemma_bf16", "raw_box", parsed={"box": [box]}),
                           row, gt2d, edt)
    assert out[0]["box_unwrapped"] is True and out[0]["box_scale"] == "unit"
    assert out[0]["hit"] is True
    mx, my = row["drawn_marks"][0]["combined_px"]
    box2 = [my / PANEL_H - 0.05, mx / (2 * PANEL_W + PANEL_GAP) - 0.05,
            my / PANEL_H + 0.05, mx / (2 * PANEL_W + PANEL_GAP) + 0.05]
    out2 = S.score_response(_resp("medgemma_bf16", "marks_box", parsed={"box": [box2]}),
                            row, gt2d, edt)
    assert out2[0]["box_unwrapped"] is True and out2[0]["snapped_mark_id"] == 12
    # a plain box is NOT marked as unwrapped
    out3 = S.score_response(_resp("medgemma_bf16", "raw_box", parsed={"box": box}), row, gt2d, edt)
    assert out3[0]["box_unwrapped"] is False
    # deeper nesting does not unwrap twice -> parse failure
    out4 = S.score_response(_resp("medgemma_bf16", "raw_box", parsed={"box": [[box]]}),
                            row, gt2d, edt)
    assert out4[0]["status"] == "parse_fail" and out4[0]["box_unwrapped"] is False


# ------------------------------------------------------------------ service versions (M4)
def test_load_done_collects_service_versions(tmp_path):
    resp = tmp_path / "responses.jsonl"
    resp.write_text("\n".join([
        json.dumps({"case": "A", "model": "gui_actor", "condition": "raw", "rep": 0,
                    "ok": True, "service_version": "1.0.0"}),
        json.dumps({"case": "B", "model": "gui_actor", "condition": "raw", "rep": 0,
                    "ok": True, "service_version": "1.0.0"}),
        json.dumps({"case": "C", "model": "medgemma_q4", "condition": "raw_box", "rep": 0,
                    "ok": True, "service_version": None}),
    ]) + "\n")
    done, versions = runner.load_done_and_prune(resp)
    assert len(done) == 3
    assert versions == {"gui_actor": {"1.0.0"}, "medgemma_q4": {None}}


def test_service_version_resume_refuses_mismatch():
    """Resuming rows from a different service build is refused (M4: the 1.0.1 servers
    added the inference lock; pre-lock responses must not be mixed in)."""
    with pytest.raises(SystemExit, match="service_version"):
        runner.check_service_version_resume({"gui_actor": {"1.0.0"}}, {"gui_actor": "1.0.1"})
    with pytest.raises(SystemExit, match="service_version"):
        runner.check_service_version_resume({"medgemma_bf16": {None}}, {"medgemma_bf16": "1.0.1"})
    runner.check_service_version_resume({"gui_actor": {"1.0.1"}}, {"gui_actor": "1.0.1"})
    # llama-server exposes no /healthz version: the check is skipped for that model
    runner.check_service_version_resume({"medgemma_q4": {None}}, {"medgemma_q4": None})
    runner.check_service_version_resume({}, {"gui_actor": "1.0.1"})


def test_fetch_service_versions_healthz(monkeypatch):
    """/healthz once per model; a service that is down is a hard error (the calls would
    fail anyway and the version check must not silently degrade)."""
    class FakeClient:
        def __init__(self, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url):
            assert url.endswith("/healthz")

            class R:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"status": "ok", "version": "1.0.1"}

            return R()

    monkeypatch.setattr("httpx.Client", FakeClient)
    assert runner.fetch_service_versions({"gui_actor": "http://x:8092"}, ["gui_actor"]) == \
        {"gui_actor": "1.0.1"}

    class Down:
        def __init__(self, timeout=None):
            raise RuntimeError("connection refused")

    monkeypatch.setattr("httpx.Client", Down)
    with pytest.raises(SystemExit, match="/healthz failed"):
        runner.fetch_service_versions({"gui_actor": "http://x:8092"}, ["gui_actor"])


# ------------------------------------------------------------------ legacy keys (m4)
def test_immutable_config_normalises_legacy_prompt_keys(tmp_path):
    """meta.json `prompts` keys were renamed local_point_<cond> -> <cond>: an old run
    resumes when the templates themselves are identical."""
    run_dir = tmp_path / "studyA" / "old"
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(json.dumps(
        {"prompts": {"local_point_raw": {"path": "harness/study/prompts/local_point_raw.md",
                                         "sha256": "samesha"},
                     "local_point_marks": {"path": "harness/study/prompts/local_point_marks.md",
                                           "sha256": "samesha2"}}}))
    meta = {"prompts": {"raw": {"path": "harness/study/prompts/local_point_raw.md",
                                "sha256": "samesha"},
                       "marks": {"path": "harness/study/prompts/local_point_marks.md",
                                 "sha256": "samesha2"}}}
    runner.check_immutable_config(run_dir, meta, {("A", "gui_actor", "raw", 0)})  # no exit
    meta_bad = {"prompts": {"raw": {"path": "variants/raw_v9.md", "sha256": "other"}}}
    with pytest.raises(SystemExit, match="drifted"):
        runner.check_immutable_config(run_dir, meta_bad, {("A", "gui_actor", "raw", 0)})


# ------------------------------------------------------------------ byte hashes (m5)
def test_prompt_sha256_hashes_raw_template_bytes(row, monkeypatch, tmp_path):
    """Row + meta prompt hashes read the template BYTES: a CRLF template hashes its
    actual bytes, not the universal-newline text read_text() would return."""
    from PIL import Image

    image_path = tmp_path / "img.png"
    Image.new("RGB", (8, 8)).save(image_path)
    r = dict(row, image_R=str(image_path), image_S=str(image_path))
    variant = tmp_path / "crlf.md"
    variant.write_bytes(b"Point at the {{target}}.\r\n")
    assert hashlib.sha256(variant.read_bytes()).hexdigest() != \
        hashlib.sha256(variant.read_text().encode()).hexdigest()

    def fake_call_pointd(url, image, query, k, timeout_s):
        return {"points": [{"x": 1.0, "y": 2.0}], "raw": "out", "latency_ms": 5.0,
                "model": "m", "version": "1.0.1", "verifier": None}

    monkeypatch.setattr(runner, "call_pointd", fake_call_pointd)
    opts = runner.parse_args(["--run", "x", "--prepare-dir", str(tmp_path),
                              "--models", "gui_actor", "--conditions", "raw",
                              "--prompt", f"raw={variant}"])
    opts.prepare_dir = str(tmp_path)
    line = runner.build_call(r, "gui_actor", "raw", 0, opts)
    assert line["prompt_sha256"] == hashlib.sha256(variant.read_bytes()).hexdigest()
    prov = runner.prompt_provenance({"raw": variant})
    assert prov["raw"]["sha256"] == hashlib.sha256(variant.read_bytes()).hexdigest()


# ------------------------------------------------------------------ runner latency rows (M3)
def test_build_call_row_latency_semantics(row, monkeypatch, tmp_path):
    """Rows carry service_latency_ms + wall_ms next to the legacy latency_ms field:
    pointd latency_ms/service_latency_ms = the server's number; chat latency_ms = the
    client wall time and service_latency_ms = the server payload's."""
    from PIL import Image

    image_path = tmp_path / "img.png"
    Image.new("RGB", (8, 8)).save(image_path)
    r = dict(row, image_R=str(image_path), image_S=str(image_path))

    def fake_call_pointd(url, image, query, k, timeout_s):
        return {"points": [{"x": 1.0, "y": 2.0}], "raw": "out", "latency_ms": 5.0,
                "model": "m", "version": "1.0.1", "verifier": None}

    monkeypatch.setattr(runner, "call_pointd", fake_call_pointd)
    opts = runner.parse_args(["--run", "x", "--prepare-dir", str(tmp_path),
                              "--models", "gui_actor", "--conditions", "raw"])
    opts.prepare_dir = str(tmp_path)
    line = runner.build_call(r, "gui_actor", "raw", 0, opts)
    assert line["service_latency_ms"] == 5.0 and line["latency_ms"] == 5.0
    assert line["wall_ms"] >= 0.0  # whole-call wall clock (fake returns instantly)
    assert line["service_version"] == "1.0.1"

    def fake_call_chat(client, prompt, image, max_tokens):
        return {"raw": "{}", "parsed": {"answer": "uncertain"}, "parse_error": None,
                "uncertain": True, "wall_ms": 7.5, "service_latency_ms": None,
                "model": "m", "usage": None, "raw_body": "{}"}

    monkeypatch.setattr(runner, "call_chat", fake_call_chat)
    opts = runner.parse_args(["--run", "x", "--prepare-dir", str(tmp_path),
                              "--models", "medgemma_bf16", "--conditions", "marks_id"])
    opts.prepare_dir = str(tmp_path)
    opts.service_versions = {"medgemma_bf16": "1.0.1"}
    line = runner.build_call(r, "medgemma_bf16", "marks_id", 0, opts)
    assert line["latency_ms"] == 7.5 and line["service_latency_ms"] is None
    assert line["wall_ms"] >= 0.0
    # chat rows are stamped with the /healthz version fetched at run start (llama-server
    # exposes none per call): this is what the service-version resume check compares
    assert line["service_version"] == "1.0.1"


def test_runner_main_refuses_pre_lock_resume(row, tmp_path, monkeypatch):
    """Wiring (M4): main() fetches /healthz once per model and refuses to resume a run
    whose existing ok rows were served by a different service build."""
    run_dir = tmp_path / "studyA" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "responses.jsonl").write_text("\n".join(
        json.dumps({"run": "r1", "case": "T99", "model": "gui_actor", "condition": cond,
                    "rep": 0, "ok": True, "service_version": "1.0.0"})
        for cond in ("raw", "marks")) + "\n")
    (tmp_path / "manifest.jsonl").write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(runner, "fetch_service_versions",
                        lambda urls, models: {"gui_actor": "1.0.1"})
    monkeypatch.setattr(sys, "argv",
                        ["local_point_study.py", "--run", "r1", "--models", "gui_actor",
                         "--prepare-dir", str(tmp_path), "--results-root", str(tmp_path / "studyA")])
    with pytest.raises(SystemExit, match="service_version"):
        runner.main()
    # matching versions resume cleanly (every call done -> no service call needed)
    (run_dir / "responses.jsonl").write_text("\n".join(
        json.dumps({"run": "r1", "case": "T99", "model": "gui_actor", "condition": cond,
                    "rep": 0, "ok": True, "service_version": "1.0.1"})
        for cond in ("raw", "marks")) + "\n")
    runner.main()
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["service_versions"] == {"gui_actor": "1.0.1"}
    assert meta["planned_calls"] == 2 and meta["resumed_from_lines"] == 2
