#!/usr/bin/env python3
"""Phase 5 Study A scorer: hit@k, distance, parse/abstain, latency, per histotype,
paired McNemar vs a Phase 2 frontier run, case-bootstrap CIs.

Reuses the Phase 2 geometry/GT-hit code (harness/study/score.py) instead of
reimplementing: every local-model answer is expressed in the study's own answer
format and scored through `score.score_call` — the exact code path that produced
the frontier baselines (panel_to_voxel inside-GT test, mark projected_ijk test,
EDT distance, invalid/parse statuses).

Coordinate frames (score.score_call's R path expects PANEL pixels of the image
area; segd.render.render_raw_pointing pastes that area at (40, 22) on a canvas
with a 26 px bottom margin):
- raw    : pointer services answer in WHOLE-PNG pixels -> the scorer subtracts the
           margins before panel_to_voxel; points in the ruler/title margins are
           `outside_panel` misses.
- raw_box: the prompt asks for fractions of the FULL PNG -> the scorer scales by
           the PNG dimensions (panel + margins), then subtracts the margins.
- marks  : the combined S image has no margins; points snap to drawn marks.

Snap-to-mark is the Study-A-specific piece (predeclared rule: nearest
drawn mark by Euclidean distance over the manifest's combined_px, ties -> lowest id).

Declared clean-panel fold rule (--clean-panel-fold on, default): the marks image is
clean panel 0 (x in [0, panel_w)) + 16 px gap + marked panel 1; pointer models point at
the tumour on BOTH panels, so for `marks` (pointd) and `marks_box` (box centre) a point
whose x lies on the clean panel is translated to the same position on the marked panel
(x + panel_w + gap, y unchanged) BEFORE snapping (the panels share the same view
geometry); gap points are not folded. Pointd `marks` ranks are then deduplicated by
snapped mark id (first occurrence in rank order; duplicates are dropped and do not
consume a rank) so hit@3 means 3 distinct marks. Per-rank `folded` records whether the
point was folded; --clean-panel-fold off keeps the pre-fold behaviour exactly.

MedGemma box scale: Gemma answers boxes as [y0, x0, y1, x1] on a 0-1000 integer scale
(e.g. {"box": [350, 230, 580, 530]}), the Phase 5 prompts ask for [0,1] fractions; the
scorer accepts both (all four values in [0,1] -> `unit`; else all in [0,1000] ->
`per1000`, divided by 1000; else parse failure; degenerate boxes y1<y0 or x1<x0 are
parse failures) and records `box_scale` per scored row.

Inputs: a run dir from local_point_study.py (responses.jsonl) + the prepare manifest
+ GT masks (offline only). Baseline: a Phase 2 run dir (e.g. the main tree's
results/phase2_test/p2-test-high, read-only) — its scored.jsonl rows are per-call;
per-case hits use the SAME majority-over-reps hit-flag rule as the headline metrics
(case_flag_at_k convention: ties -> miss; error/parse reps count as misses).

Outputs (in the run dir): scored.jsonl (one row per response×rank), score.json,
report.md.

Run:  uv run --project harness/server python harness/study/score_local_point.py \
          --run studyA-test-gui [--prepare-dir ...] \
          [--baseline-dir /home/ubuntu/agentic-segmentation/results/phase2_test/p2-test-high]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDY_DIR = PROJECT_ROOT / "harness" / "study"
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

import score as phase2_score  # noqa: E402  (harness/study/score.py — geometry + GT-hit code)
from segd.render import (  # noqa: E402  (segd on sys.path via the harness/server env)
    RAW_MARGIN_BOTTOM_PX as RAW_MARGIN_BOTTOM_PX,
    RAW_MARGIN_LEFT_PX as RAW_MARGIN_LEFT_PX,
    RAW_MARGIN_TOP_PX as RAW_MARGIN_TOP_PX,
)

HISTOTYPES = ("F", "L", "M", "N", "V")
POINTD_CONDITIONS = ("raw", "marks")
CHAT_CONDITIONS = ("raw_box", "marks_box", "marks_id")
DEFAULT_RESULTS_ROOT = "results/phase5/studyA"
DEFAULT_BASELINE = Path("/home/ubuntu/agentic-segmentation/results/phase2_test/p2-test-high")
PANEL_GAP_PX = 16  # segd.render.PANEL_GAP_PX (kept in sync; only used for S-image panels)


# ------------------------------------------------------------------ raw PNG frame
def raw_png_size(geom) -> tuple[int, int]:
    """Dimensions of the sent raw-pointing PNG: image area + ruler/title margins."""
    return geom.panel_w + RAW_MARGIN_LEFT_PX, geom.panel_h + RAW_MARGIN_TOP_PX + RAW_MARGIN_BOTTOM_PX


def png_to_panel_px(x: float, y: float) -> tuple[float, float]:
    """Whole-PNG pixel -> panel (image-area) pixel: subtract the margin offsets.

    render_raw_pointing pastes the panel at (RAW_MARGIN_LEFT_PX, RAW_MARGIN_TOP_PX);
    pointer services answer in whole-PNG pixels, score.py's R path wants panel px.
    """
    return x - RAW_MARGIN_LEFT_PX, y - RAW_MARGIN_TOP_PX


def _strict_int(value) -> int | None:
    """A genuine integer mark id: int (not bool) or integral float; nothing coerced.

    {"mark_id": "unknown"} / 12.9 / true are parse failures, never crashes or silent
    conversions (review P1-10).
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _box_scale(box) -> str | None:
    """Declared MedGemma box-scale rule: Gemma answers boxes as [y0, x0, y1, x1] on a
    0-1000 integer scale (e.g. {"box": [350, 230, 580, 530]}), the Phase 5 prompts ask
    for [0,1] fractions. All four values in [0,1] -> "unit"; else all in [0,1000] ->
    "per1000" (the scorer divides by 1000); otherwise None (parse failure: wrong
    shape/type, a value outside [0,1000], or mixed-scale values that fit neither
    declared frame)."""
    if not (isinstance(box, list) and len(box) == 4
            and all(isinstance(v, (int, float)) for v in box)):
        return None
    if all(0 <= v <= 1 for v in box):
        return "unit"
    if all(0 <= v <= 1000 for v in box):
        return "per1000"
    return None


# ------------------------------------------------------------------ snapping
def snap_to_mark(px: float, py: float, drawn_marks: list[dict]) -> tuple[dict | None, float]:
    """Nearest drawn mark (combined-image pixel coords); ties -> lowest id.

    Predeclared snap rule (research note §169-case protocol). Returns (mark, dist_px).
    """
    best = None
    best_d = float("inf")
    for mark in sorted(drawn_marks, key=lambda m: int(m["id"])):
        x, y = mark["combined_px"]
        d = math.hypot(px - x, py - y)
        if d < best_d:
            best, best_d = mark, d
    return best, best_d


def clean_panel_fold_px(x: float, y: float, geom, enabled: bool) -> tuple[float, float, bool]:
    """Declared clean-panel fold rule (--clean-panel-fold on): the S image is clean
    panel 0 (x in [0, panel_w)) + PANEL_GAP_PX gap + marked panel 1, and the two panels
    share the same view geometry, so a point whose x lies on the clean panel is
    translated to the same position on the marked panel (x + panel_w + gap, y kept)
    before snapping. Points on the marked panel, in the gap, or beyond it are returned
    unchanged. Returns (x, y, folded)."""
    if enabled and 0 <= x < geom.panel_w:
        return x + geom.panel_w + PANEL_GAP_PX, y, True
    return x, y, False


def answer_snippet(mark_id: int) -> str:
    """The study answer in Phase 2 S-format, scored through score.score_call."""
    return f"ANSWER: {mark_id}"


def answer_point(x: float, y: float) -> str:
    """The study answer in Phase 2 R-format, scored through score.score_call."""
    return f"ANSWER: ({x:.2f}, {y:.2f})"


# ------------------------------------------------------------------ per response
def score_response(resp: dict, row: dict, gt2d: np.ndarray, edt: np.ndarray,
                   clean_panel_fold: bool = True) -> list[dict]:
    """Score one responses.jsonl line -> per-rank scored rows (rank 0 = top point)."""
    base = {
        "run": resp.get("run"), "case": resp["case"], "histotype": row["histotype"],
        "sequence": row["sequence"], "model": resp["model"], "condition": resp["condition"],
        "rep": resp.get("rep"), "latency_ms": resp.get("latency_ms"),
        "response_error": bool(resp.get("error")),
        "n_points": len(resp.get("points") or []),
        "folded": False, "box_scale": None,
    }
    if base["response_error"]:
        base.update({"rank": 0, "status": "response_error", "hit": False, "distance_mm": None,
                     "point": None, "png_point": None, "snapped_mark_id": None,
                     "snap_distance_px": None, "raw_point_hit": None, "raw_point_distance_mm": None})
        return [base]

    condition = resp["condition"]
    if condition in POINTD_CONDITIONS:
        return _score_pointd(resp, row, gt2d, edt, base, clean_panel_fold)
    return _score_chat(resp, row, gt2d, edt, base, clean_panel_fold)


def _score_pointd(resp: dict, row: dict, gt2d, edt, base, clean_panel_fold: bool) -> list[dict]:
    """Pointer services: k pixel points; ranks scored independently (hit@1/hit@3).

    With --clean-panel-fold on, `marks` points on the clean panel are folded to the
    marked panel before snapping, and ranks are deduplicated by snapped mark id (first
    occurrence in rank order; a later point snapping to an already-used mark is dropped
    and does not consume a rank) so hit@3 means 3 distinct marks."""
    points = resp.get("points") or []
    if not points:
        base.update({"rank": 0, "status": "abstain", "hit": False, "distance_mm": None,
                     "point": None, "png_point": None, "snapped_mark_id": None,
                     "snap_distance_px": None, "raw_point_hit": None, "raw_point_distance_mm": None})
        return [base]
    out = []
    seen_marks: set[int] = set()
    rank = 0
    for point in points:
        row_rank = dict(base, point=[point["x"], point["y"]], png_point=None)
        if resp["condition"] == "marks":
            fx, fy, folded = clean_panel_fold_px(float(point["x"]), float(point["y"]),
                                                 _geom(row, "S"), clean_panel_fold)
            mark, snap_d = snap_to_mark(fx, fy, row["drawn_marks"])
            if clean_panel_fold and mark is not None and int(mark["id"]) in seen_marks:
                continue  # duplicate snapped mark: dropped, does not consume a rank
            row_rank.update({"rank": rank, "folded": folded})
            rank += 1
            if mark is None:  # no drawn marks in the manifest row: cannot snap
                row_rank.update({"snapped_mark_id": None, "snap_distance_px": None,
                                 "status": "parse_fail", "hit": False, "distance_mm": None,
                                 "raw_point_hit": None, "raw_point_distance_mm": None})
                out.append(row_rank)
                continue
            row_rank["snapped_mark_id"] = mark["id"]
            seen_marks.add(int(mark["id"]))
            row_rank["snap_distance_px"] = round(snap_d, 2)
            call = _phase2_call(resp, answer_snippet(mark["id"]), "S")
            scored = phase2_score.score_call(call, row, gt2d, edt, _geom(row, "S"))
            # secondary: the RAW (unsnapped) point on the panel it landed on — never
            # confused with the snapped answer (research note: report both, never mix).
            _attach_raw_point(row_rank, resp, row, gt2d, edt, point, clean_panel_fold)
        else:
            row_rank.update({"rank": rank})
            rank += 1
            # The sent PNG is NOT the panel: render_raw_pointing pastes the image area at
            # (RAW_MARGIN_LEFT_PX, RAW_MARGIN_TOP_PX) with a bottom margin. Convert whole-PNG
            # pixels to panel pixels first; a point in a margin is an outside_panel miss.
            row_rank["png_point"] = [round(float(point["x"]), 2), round(float(point["y"]), 2)]
            row_rank.update({"snapped_mark_id": None, "snap_distance_px": None,
                             "raw_point_hit": None, "raw_point_distance_mm": None})
            sx, sy = png_to_panel_px(float(point["x"]), float(point["y"]))
            geom_r = _geom(row, "R")
            if not (0 <= sx < geom_r.panel_w and 0 <= sy < geom_r.panel_h):
                row_rank.update({"point": None, "status": "outside_panel", "hit": False,
                                 "distance_mm": None})
                out.append(row_rank)
                continue
            row_rank["point"] = [round(sx, 2), round(sy, 2)]
            call = _phase2_call(resp, answer_point(sx, sy), "R")
            scored = phase2_score.score_call(call, row, gt2d, edt, geom_r)
        row_rank.update({"status": scored["status"], "hit": scored["hit"],
                         "distance_mm": scored["distance_mm"]})
        out.append(row_rank)
    return out


def _score_chat(resp: dict, row: dict, gt2d, edt, base, clean_panel_fold: bool) -> list[dict]:
    """MedGemma: strict-JSON answers (box -> centre; mark id; explicit uncertain)."""
    parsed = resp.get("parsed")
    status = "parse_fail" if (resp.get("parse_error") or parsed is None) else None
    answer = (parsed or {}).get("answer")
    uncertain = isinstance(answer, str) and answer.strip().lower() == "uncertain"
    row0 = dict(base, rank=0, point=None, png_point=None, snapped_mark_id=None,
                snap_distance_px=None, raw_point_hit=None, raw_point_distance_mm=None)
    if status:
        row0["status"] = status
        row0.update({"hit": False, "distance_mm": None})
        return [row0]
    if uncertain:
        row0["status"] = "abstain"  # explicit `uncertain` option: counted, never retried
        row0.update({"hit": False, "distance_mm": None})
        return [row0]

    if resp["condition"] == "marks_id":
        mark_id = _strict_int((parsed or {}).get("mark_id"))
        if mark_id is None:  # non-integer id: parse failure, never coerced (review P1-10)
            row0.update({"status": "parse_fail", "hit": False, "distance_mm": None})
            return [row0]
        call = _phase2_call(resp, answer_snippet(mark_id), "S")
        scored = phase2_score.score_call(call, row, gt2d, edt, _geom(row, "S"))
        row0["snapped_mark_id"] = mark_id
        row0.update({"status": scored["status"], "hit": scored["hit"],
                     "distance_mm": scored["distance_mm"]})
        return [row0]

    box = (parsed or {}).get("box")
    box_scale = _box_scale(box)
    if box_scale is None:
        row0.update({"status": "parse_fail", "hit": False, "distance_mm": None})
        return [row0]
    div = 1000.0 if box_scale == "per1000" else 1.0
    y0, x0, y1, x1 = (float(v) / div for v in box)  # MedGemma/Gemma convention: [y0, x0, y1, x1]
    if y1 < y0 or x1 < x0:  # degenerate box: parse failure, never silently re-centred
        row0.update({"status": "parse_fail", "hit": False, "distance_mm": None})
        return [row0]
    row0["box_scale"] = box_scale
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    if resp["condition"] == "raw_box":
        geom = _geom(row, "R")
        # The raw_box prompt asks for fractions of the FULL raw PNG (image area + ruler
        # and title margins), not of the panel: scale by the PNG dimensions, then
        # subtract the margin offsets to get panel pixels.
        png_w, png_h = raw_png_size(geom)
        cx_png, cy_png = cx * png_w, cy * png_h
        px, py = png_to_panel_px(cx_png, cy_png)
        row0["png_point"] = [round(cx_png, 2), round(cy_png, 2)]
        if not (0 <= px < geom.panel_w and 0 <= py < geom.panel_h):
            row0.update({"status": "outside_panel", "hit": False, "distance_mm": None})
            return [row0]
        call = _phase2_call(resp, answer_point(px, py), "R")
        scored = phase2_score.score_call(call, row, gt2d, edt, geom)
        row0["point"] = [round(px, 2), round(py, 2)]
        row0.update({"status": scored["status"], "hit": scored["hit"],
                     "distance_mm": scored["distance_mm"]})
        return [row0]
    # marks_box: centre on the combined S image (clean panel + gap + marked panel),
    # clean-panel fold applied first, then snapped to the nearest drawn mark
    geom = _geom(row, "S")
    px, py = cx * geom_total_width(geom), cy * geom.panel_h
    row0["point"] = [round(px, 2), round(py, 2)]
    fx, fy, folded = clean_panel_fold_px(px, py, geom, clean_panel_fold)
    row0["folded"] = folded
    mark, snap_d = snap_to_mark(fx, fy, row["drawn_marks"])
    row0["snapped_mark_id"] = mark["id"] if mark else None
    row0["snap_distance_px"] = round(snap_d, 2) if mark else None
    if mark is None:
        row0.update({"status": "parse_fail", "hit": False, "distance_mm": None})
        return [row0]
    call = _phase2_call(resp, answer_snippet(mark["id"]), "S")
    scored = phase2_score.score_call(call, row, gt2d, edt, _geom(row, "S"))
    _attach_raw_point(row0, resp, row, gt2d, edt, {"x": px, "y": py}, clean_panel_fold)
    row0.update({"status": scored["status"], "hit": scored["hit"],
                 "distance_mm": scored["distance_mm"]})
    return [row0]


def geom_total_width(geom) -> int:
    """Combined S-image width: marked panel + gap + clean panel (both panel_w wide)."""
    return geom.panel_w * 2 + PANEL_GAP_PX


def _phase2_call(resp: dict, raw_text: str, condition: str) -> dict:
    """A Phase 2-shaped response row for score.score_call (constructed answers)."""
    return {"case": resp["case"], "condition": condition, "rep": resp.get("rep"),
            "raw_text": raw_text, "error": None, "stop_reason": "stop"}


def _geom(row: dict, which: str):
    from segd.render import ViewGeom

    key = "view_geom_R" if which == "R" else "view_geom_S"
    g = row[key]
    return ViewGeom(g["axis"], int(g["slice_idx"]), int(g["rows"]), int(g["cols"]),
                    tuple(g["crop"]), float(g["scale"]), int(g["panel_w"]), int(g["panel_h"]))


def _attach_raw_point(row_rank: dict, resp: dict, row: dict, gt2d, edt, point: dict,
                      clean_panel_fold: bool) -> None:
    """Secondary diagnostic: the raw (unsnapped) point scored on the panel it landed on.

    The clean and marked panels share the same view geometry, so a point is expressed in
    marked-panel px for the R path: a marked-panel point subtracts panel 0 width + gap;
    a clean-panel point keeps its own (x, y). With --clean-panel-fold off the legacy
    behaviour is kept (a clean-panel point has no marked-panel voxel -> None); a gap or
    out-of-panel point has no panel voxel either way. Never mixed with the snapped
    answer.
    """
    geom = _geom(row, "S")
    x, y = float(point["x"]), float(point["y"])
    if clean_panel_fold and 0 <= x < geom.panel_w:
        px, py = x, y  # clean panel 0: same view geometry as the marked panel
    else:
        px = x - (geom.panel_w + PANEL_GAP_PX)
        py = y
    if px < 0:  # landed in the gap (or, fold off, on the clean panel): no panel voxel
        row_rank["raw_point_hit"], row_rank["raw_point_distance_mm"] = None, None
        return
    call = _phase2_call(resp, answer_point(px, py), "R")
    scored = phase2_score.score_call(call, row, gt2d, edt, geom)
    row_rank["raw_point_hit"] = scored["hit"]
    row_rank["raw_point_distance_mm"] = scored["distance_mm"]


# ------------------------------------------------------------------ aggregation
def case_flag_at_k(case_calls: list[dict], k: int) -> int:
    """hit@k flag for one case: any of ranks 0..k-1 hit, majority over reps (deterministic
    decoding -> reps agree; ties break to False, phase2 majority_vote convention)."""
    reps = sorted({c["rep"] for c in case_calls})
    flags = [int(any(c["hit"] for c in case_calls if c["rep"] == rep and c["rank"] < k))
             for rep in reps]
    return int(sum(flags) * 2 > len(reps))


def aggregate(calls: list[dict]) -> dict:
    """Metrics over scored rows: hit@1 (rank 0), hit@3 (any of ranks 0-2), distance,
    parse/abstain rates, latency. Per-case hit@k = majority over reps (case-level,
    matching the McNemar pairing); hit1_mean_over_calls is the phase2
    hit_rate_mean_over_reps analog."""
    rank0 = [c for c in calls if c["rank"] == 0]
    by_case: dict[str, list[dict]] = {}
    for c in calls:
        by_case.setdefault(c["case"], []).append(c)
    cases = sorted(by_case)

    def hit_rate(k: int) -> float:
        return round(sum(case_flag_at_k(by_case[case], k) for case in cases) / max(len(cases), 1), 4)

    dists = [c["distance_mm"] for c in rank0 if c["distance_mm"] is not None]
    lat = [c["latency_ms"] for c in calls if c["latency_ms"] is not None]
    n = max(len(rank0), 1)
    return {
        "n_calls": len(calls),
        "n_cases": len(cases),
        "hit1": hit_rate(1),
        "hit3": hit_rate(3),
        "hit1_mean_over_calls": round(sum(c["hit"] for c in rank0) / n, 4),
        "distance_mean_mm": round(float(np.mean(dists)), 3) if dists else None,
        "distance_median_mm": round(float(np.median(dists)), 3) if dists else None,
        "n_parse_fail": sum(c["status"] == "parse_fail" for c in rank0),
        "n_abstain": sum(c["status"] == "abstain" for c in rank0),
        "n_response_error": sum(c["status"] == "response_error" for c in rank0),
        "parse_fail_rate": round(sum(c["status"] == "parse_fail" for c in rank0) / n, 4),
        "abstain_rate": round(sum(c["status"] == "abstain" for c in rank0) / n, 4),
        "latency_mean_ms": round(float(np.mean(lat)), 1) if lat else None,
        "latency_median_ms": round(float(np.median(lat)), 1) if lat else None,
    }


def bootstrap_ci(flags: list[int], n_boot: int, seed: int) -> tuple[float, float]:
    """Case-level bootstrap percentile CI for a hit rate (list of 0/1 per case)."""
    if not flags:
        return (None, None)
    rng = np.random.default_rng(seed)
    arr = np.asarray(flags, dtype=float)
    stats = [arr[rng.integers(0, len(arr), size=len(arr))].mean() for _ in range(n_boot)]
    return (round(float(np.percentile(stats, 2.5)), 4), round(float(np.percentile(stats, 97.5)), 4))


def per_case_hit1(calls: list[dict]) -> dict[str, bool]:
    """Per-case hit@1 flags: the SAME case_flag_at_k(..., 1) majority-over-reps summary
    the headline metrics report (review P1-7: never 'first successful rep'). Ties ->
    False; error/parse reps carry hit=False and count as misses."""
    by_case: dict[str, list[dict]] = {}
    for c in calls:
        by_case.setdefault(c["case"], []).append(c)
    return {case: bool(case_flag_at_k(cs, 1)) for case, cs in sorted(by_case.items())}


def rep_hit_majority(calls: list[dict]) -> bool:
    """Majority over reps of the per-rep hit flag (ties -> False) for the Phase 2
    frontier rows (one answer per rep, no ranks): the same convention as
    case_flag_at_k so both McNemar arms are aggregated identically. Error/parse reps
    carry hit=False and count as misses."""
    reps = sorted({c.get("rep") for c in calls})
    flags = [int(any(bool(c["hit"]) for c in calls if c.get("rep") == rep)) for rep in reps]
    return bool(sum(flags) * 2 > len(reps))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial n=b+c, p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2.0 * tail * 0.5 ** n)


def paired_vs_baseline(local_by_case: dict[str, bool], baseline_by_case: dict[str, bool],
                       n_boot: int, seed: int) -> dict:
    """McNemar (case-paired) + bootstrap CI of the hit-rate difference."""
    cases = sorted(set(local_by_case) & set(baseline_by_case))
    n_baseline_hit_local_miss = sum(1 for c in cases
                                    if baseline_by_case[c] and not local_by_case[c])
    n_local_hit_baseline_miss = sum(1 for c in cases
                                    if not baseline_by_case[c] and local_by_case[c])
    diff_flags = [int(local_by_case[c]) - int(baseline_by_case[c]) for c in cases]
    rng = np.random.default_rng(seed)
    arr = np.asarray(diff_flags, dtype=float)
    lo = hi = None
    if len(arr):
        stats = [arr[rng.integers(0, len(arr), size=len(arr))].mean() for _ in range(n_boot)]
        lo, hi = (round(float(np.percentile(stats, 2.5)), 4), round(float(np.percentile(stats, 97.5)), 4))
    return {
        "n_paired_cases": len(cases),
        "n_baseline_hit_local_miss": n_baseline_hit_local_miss,
        "n_local_hit_baseline_miss": n_local_hit_baseline_miss,
        "mcnemar_p": round(mcnemar_exact(n_baseline_hit_local_miss, n_local_hit_baseline_miss), 5),
        "diff_hit_rate": round(sum(diff_flags) / max(len(cases), 1), 4) if cases else None,
        "diff_ci95": [lo, hi],
    }


# ------------------------------------------------------------------ report
def write_report(run_dir: Path, cond_keys: list[tuple], agg: dict, per_hist: dict,
                 baseline: dict, meta: dict) -> None:
    lines = [f"# Phase 5 Study A — standalone localization, run `{run_dir.name}`", ""]
    lines.append(f"Scored {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} against "
                 f"prepare manifest `{meta.get('prepare_manifest')}`. "
                 "hit@1 = top point inside GT on k*; hit@3 = any of ranks 0-2; distance via the "
                 "Phase 2 EDT; snap rule = nearest drawn mark (ties -> lowest id).")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append("| model | condition | n cases | hit@1 | CI95 | hit@3 | dist mean mm | parse fail | abstain | "
                 "latency mean ms |")
    lines.append("|---|" + "---|" * 9)
    for key in cond_keys:
        a = agg[key]
        ci = a.get("hit1_ci95")
        lines.append(f"| {key[0]} | {key[1]} | {a['n_cases']} | {a['hit1']:.3f} | "
                     f"[{ci[0]}, {ci[1]}] | {a['hit3']:.3f} | {a['distance_mean_mm']} | "
                     f"{a['n_parse_fail']} | {a['n_abstain']} | {a['latency_mean_ms']} |")
    lines.append("")
    if baseline:
        lines.append("## Paired vs Phase 2 frontier baseline (case-level hit@1: majority "
                     "over reps of per-rep hit flags, ties -> miss; errors/parse failures "
                     "count as misses)")
        lines.append("")
        lines.append("| model | condition | vs baseline | n paired | b (base hit, local miss) | "
                     "c (local hit, base miss) | McNemar p | Δ hit rate | CI95 |")
        lines.append("|---|" + "---|" * 8)
        for (model, condition), comp in sorted(baseline.items()):
            lines.append(f"| {model} | {condition} | {comp['baseline']} | {comp['n_paired_cases']} | "
                         f"{comp['n_baseline_hit_local_miss']} | {comp['n_local_hit_baseline_miss']} | "
                         f"{comp['mcnemar_p']} | {comp['diff_hit_rate']} | {comp['diff_ci95']} |")
        lines.append("")
    lines.append("## Per histotype (hit@1)")
    lines.append("")
    lines.append("| model | condition | " + " | ".join(HISTOTYPES) + " |")
    lines.append("|---" * 2 + "---|" * len(HISTOTYPES))
    for key in sorted(per_hist):
        cells = [f"{per_hist[key][h]['hit1']:.3f}" if h in per_hist[key] else "–"
                 for h in HISTOTYPES]
        lines.append(f"| {key[0]} | {key[1]} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Scoring rules")
    lines.append("- raw / raw_box: the point (or box centre, scaled by the FULL PNG "
                 "dimensions) minus the raw margins (panel pasted at (40, 22), 26 px "
                 "bottom margin) -> panel_to_voxel -> inside GT on k* (score.py R path). "
                 "Points in the ruler/title margins are outside_panel misses.")
    lines.append("- marks / marks_box: point or box centre snapped to the nearest drawn mark "
                 "(combined_px, ties -> lowest id) -> that mark's projected voxel inside GT "
                 "(score.py S path). The unsnapped point is reported separately "
                 "(raw_point_hit), never mixed with the snapped answer.")
    if meta.get("clean_panel_fold", "on") == "on":
        lines.append("- Clean-panel fold (declared rule, --clean-panel-fold on): the marks image is "
                     "clean panel 0 (x < panel_w) + 16 px gap + marked panel 1; a point/box centre "
                     "whose x lies on the clean panel is translated to the same position on the "
                     "marked panel (x + panel_w + gap, y unchanged) BEFORE snapping (the panels "
                     "share the same view geometry); points in the gap are not folded. Per-rank "
                     "`folded` records whether the point was folded. Pointer `marks` ranks are "
                     "deduplicated by snapped mark id (first occurrence in rank order; later "
                     "points snapping to an already-used mark are dropped and do not consume a "
                     "rank) so hit@3 means 3 distinct marks.")
    else:
        lines.append("- Clean-panel fold is OFF (--clean-panel-fold off): every point/box centre "
                     "snaps as-is and ranks are not deduplicated.")
    lines.append("- raw_box / marks_box boxes: [y0, x0, y1, x1]; all four values in [0,1] are "
                 "fractions (box_scale `unit`), else all in [0,1000] are the Gemma 0-1000 "
                 "integer scale (box_scale `per1000`, divided by 1000); anything else — wrong "
                 "shape/type, a value outside [0,1000], or a degenerate box (y1 < y0 or "
                 "x1 < x0) — is a parse failure.")
    lines.append("- marks_id: mark id answered directly (S path; a non-integer id is a "
                 "parse failure, an unknown integer id = invalid_mark).")
    lines.append("- Parse failures and the explicit `uncertain` answer count as misses and are "
                 "reported separately (parse_fail / abstain); never retried.")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="run name under --results-root")
    ap.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--prepare-dir", default=None,
                    help="prepare manifest dir (default: the run's meta.json prepare_manifest)")
    ap.add_argument("--baseline-dir", default=None,
                    help="Phase 2 frontier run dir (scored.jsonl) for the paired McNemar; "
                         "pass '' to skip. Default: the main tree's p2-test-high when present")
    ap.add_argument("--baseline-condition", default="S",
                    help="frontier condition to compare against (S = marks baseline, R = raw)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--clean-panel-fold", choices=("on", "off"), default="on",
                    help="declared clean-panel fold rule for the marks conditions: a point/box "
                         "centre on the clean panel is translated to the same position on the "
                         "marked panel before snapping, and pointd `marks` ranks are "
                         "deduplicated by snapped mark id (default on; off = pre-fold behaviour)")
    args = ap.parse_args()

    run_dir = PROJECT_ROOT / args.results_root / args.run
    resp_path = run_dir / "responses.jsonl"
    if not resp_path.exists():
        raise SystemExit(f"no responses.jsonl in {run_dir}")

    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    prepare_hint = args.prepare_dir or meta.get("prepare_manifest")
    prepare_dir = Path(prepare_hint).parent if prepare_hint and Path(prepare_hint).name == "manifest.jsonl" \
        else (Path(prepare_hint) if prepare_hint else None)
    if prepare_dir is None or not (prepare_dir / "manifest.jsonl").exists():
        raise SystemExit("cannot locate prepare manifest (--prepare-dir or run meta.json)")
    rows = {r["case"]: r for r in (json.loads(line) for line in
                                    (prepare_dir / "manifest.jsonl").read_text().splitlines()
                                    if line.strip())}

    resp_lines = [json.loads(l) for l in resp_path.read_text().splitlines() if l.strip()]
    unknown = sorted({r["case"] for r in resp_lines} - set(rows))
    if unknown:
        raise SystemExit(f"responses for unknown cases: {unknown}")

    from segd.data import get_case, load_mask

    fold = args.clean_panel_fold == "on"
    scored: list[dict] = []
    gt_cache: dict[str, tuple] = {}
    for resp in resp_lines:
        case = resp["case"]
        if case not in gt_cache:
            mask = load_mask(get_case(case))
            gt_cache[case] = phase2_score.gt_slices_and_edt(rows[case], mask)
        gt2d, edt = gt_cache[case]
        scored.extend(score_response(resp, rows[case], gt2d, edt, fold))
    (run_dir / "scored.jsonl").write_text("".join(json.dumps(c) + "\n" for c in scored))

    # ---- aggregation: (model, condition) -> metrics --------------------------
    cond_keys = sorted({(c["model"], c["condition"]) for c in scored})
    agg: dict[tuple, dict] = {}
    per_case_hits: dict[tuple, dict[str, bool]] = {}
    for key in cond_keys:
        model, condition = key
        cc = [c for c in scored if c["model"] == model and c["condition"] == condition]
        a = aggregate(cc)
        by_case: dict[str, list[dict]] = {}
        for c in cc:
            by_case.setdefault(c["case"], []).append(c)
        a["hit1_ci95"] = bootstrap_ci(
            [case_flag_at_k(by_case[case], 1) for case in sorted(by_case)], args.n_boot, args.seed)
        a["hit3_ci95"] = bootstrap_ci(
            [case_flag_at_k(by_case[case], 3) for case in sorted(by_case)], args.n_boot, args.seed)
        agg[key] = a
        # per-case majority hits for McNemar pairing: the SAME per-case summary as the
        # headline hit@1 (case_flag_at_k majority over reps), review P1-7.
        per_case_hits[key] = per_case_hit1(cc)

    per_hist: dict[tuple, dict] = {}
    for key in cond_keys:
        model, condition = key
        per_hist[key] = {}
        for h in HISTOTYPES:
            hcc = [c for c in scored if c["model"] == model and c["condition"] == condition
                   and c["histotype"] == h]
            if hcc:
                per_hist[key][h] = aggregate(hcc)

    # ---- paired McNemar vs the Phase 2 frontier run --------------------------
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else (
        DEFAULT_BASELINE if DEFAULT_BASELINE.exists() else None)
    baseline: dict[tuple, dict] = {}
    if baseline_dir:
        base_scored_path = baseline_dir / "scored.jsonl"
        if not base_scored_path.exists():
            print(f"[score] baseline scored.jsonl missing at {base_scored_path}; skipping McNemar")
        else:
            base_rows = [json.loads(l) for l in base_scored_path.read_text().splitlines() if l.strip()]
            base_by_case: dict[str, list[dict]] = {}
            for r in base_rows:
                if r.get("condition") == args.baseline_condition:
                    base_by_case.setdefault(r["case"], []).append(r)
            baseline_hits = {}
            for case, calls_ in base_by_case.items():
                # majority over reps of per-rep hit flags (same convention as the local
                # side's case_flag_at_k); error/parse reps count as misses
                baseline_hits[case] = rep_hit_majority(calls_)
            for key in cond_keys:
                model, condition = key
                comp = paired_vs_baseline(per_case_hits[key], baseline_hits, args.n_boot, args.seed)
                comp["baseline"] = f"{baseline_dir.name}:{args.baseline_condition}"
                baseline[key] = comp

    (run_dir / "score.json").write_text(json.dumps({
        "aggregate": {f"{m}/{c}": a for (m, c), a in agg.items()},
        "per_histotype": {f"{m}/{c}": a for (m, c), a in per_hist.items()},
        "baseline_comparison": {f"{m}/{c}": comp for (m, c), comp in baseline.items()},
        "baseline": {"dir": str(baseline_dir), "condition": args.baseline_condition} if baseline_dir else None,
        "n_boot": args.n_boot, "seed": args.seed,
        "clean_panel_fold": args.clean_panel_fold,
    }, indent=2) + "\n")
    write_report(run_dir, cond_keys, agg, per_hist, baseline,
                 {**meta, "prepare_manifest": str(prepare_dir / "manifest.jsonl"),
                  "clean_panel_fold": args.clean_panel_fold})

    print(f"[score] {len(scored)} scored rows -> {run_dir / 'report.md'}")
    for key in cond_keys:
        a = agg[key]
        print(f"[score] {key[0]}/{key[1]}: hit@1 {a['hit1']:.3f} hit@3 {a['hit3']:.3f} "
              f"dist={a['distance_mean_mm']} parse_fail={a['n_parse_fail']} abstain={a['n_abstain']}")
        if key in baseline:
            comp = baseline[key]
            print(f"[score]   vs {comp['baseline']}: McNemar p={comp['mcnemar_p']} "
                  f"(b={comp['n_baseline_hit_local_miss']}, c={comp['n_local_hit_baseline_miss']})")


if __name__ == "__main__":
    main()
