#!/usr/bin/env python3
"""Phase 5 Study A scorer: hit@k, distance, parse/abstain, latency, per histotype,
paired McNemar vs a Phase 2 frontier run, case-bootstrap CIs.

Reuses the Phase 2 geometry/GT-hit code (harness/study/score.py) instead of
reimplementing: every local-model answer is expressed in the study's own answer
format (a point on the raw image, or a mark id) and scored through
`score.score_call` — the exact code path that produced the frontier baselines
(panel_to_voxel inside-GT test, mark projected_ijk test, EDT distance, invalid/parse
statuses). Snap-to-mark is the Study-A-specific piece (predeclared rule: nearest
drawn mark by Euclidean distance over the manifest's combined_px, ties -> lowest id).

Inputs: a run dir from local_point_study.py (responses.jsonl) + the prepare manifest
+ GT masks (offline only). Baseline: a Phase 2 run dir (e.g. the main tree's
results/phase2_test/p2-test-high, read-only) — its scored.jsonl rows are per-call;
per-case hits use score.majority_vote, the same convention as its report.

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

HISTOTYPES = ("F", "L", "M", "N", "V")
POINTD_CONDITIONS = ("raw", "marks")
CHAT_CONDITIONS = ("raw_box", "marks_box", "marks_id")
DEFAULT_RESULTS_ROOT = "results/phase5/studyA"
DEFAULT_BASELINE = Path("/home/ubuntu/agentic-segmentation/results/phase2_test/p2-test-high")
PANEL_GAP_PX = 16  # segd.render.PANEL_GAP_PX (kept in sync; only used for S-image panels)


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


def answer_snippet(mark_id: int) -> str:
    """The study answer in Phase 2 S-format, scored through score.score_call."""
    return f"ANSWER: {mark_id}"


def answer_point(x: float, y: float) -> str:
    """The study answer in Phase 2 R-format, scored through score.score_call."""
    return f"ANSWER: ({x:.2f}, {y:.2f})"


# ------------------------------------------------------------------ per response
def score_response(resp: dict, row: dict, gt2d: np.ndarray, edt: np.ndarray) -> list[dict]:
    """Score one responses.jsonl line -> per-rank scored rows (rank 0 = top point)."""
    base = {
        "run": resp.get("run"), "case": resp["case"], "histotype": row["histotype"],
        "sequence": row["sequence"], "model": resp["model"], "condition": resp["condition"],
        "rep": resp.get("rep"), "latency_ms": resp.get("latency_ms"),
        "response_error": bool(resp.get("error")),
        "n_points": len(resp.get("points") or []),
    }
    if base["response_error"]:
        base.update({"rank": 0, "status": "response_error", "hit": False, "distance_mm": None,
                     "point": None, "snapped_mark_id": None, "snap_distance_px": None,
                     "raw_point_hit": None, "raw_point_distance_mm": None})
        return [base]

    condition = resp["condition"]
    if condition in POINTD_CONDITIONS:
        return _score_pointd(resp, row, gt2d, edt, base)
    return _score_chat(resp, row, gt2d, edt, base)


def _score_pointd(resp: dict, row: dict, gt2d, edt, base) -> list[dict]:
    """Pointer services: k pixel points; ranks scored independently (hit@1/hit@3)."""
    points = resp.get("points") or []
    if not points:
        base.update({"rank": 0, "status": "abstain", "hit": False, "distance_mm": None,
                     "point": None, "snapped_mark_id": None, "snap_distance_px": None,
                     "raw_point_hit": None, "raw_point_distance_mm": None})
        return [base]
    out = []
    for rank, point in enumerate(points):
        row_rank = dict(base, rank=rank, point=[point["x"], point["y"]])
        if resp["condition"] == "marks":
            mark, snap_d = snap_to_mark(point["x"], point["y"], row["drawn_marks"])
            if mark is None:  # no drawn marks in the manifest row: cannot snap
                row_rank.update({"snapped_mark_id": None, "snap_distance_px": None,
                                 "status": "parse_fail", "hit": False, "distance_mm": None,
                                 "raw_point_hit": None, "raw_point_distance_mm": None})
                out.append(row_rank)
                continue
            row_rank["snapped_mark_id"] = mark["id"]
            row_rank["snap_distance_px"] = round(snap_d, 2)
            call = _phase2_call(resp, answer_snippet(mark["id"]), "S")
            scored = phase2_score.score_call(call, row, gt2d, edt, _geom(row, "S"))
            # secondary: the RAW (unsnapped) point on the marked panel — never confused
            # with the snapped answer (research note: report both, never mix).
            _attach_raw_point(row_rank, resp, row, gt2d, edt, point)
        else:
            row_rank.update({"snapped_mark_id": None, "snap_distance_px": None,
                             "raw_point_hit": None, "raw_point_distance_mm": None})
            call = _phase2_call(resp, answer_point(point["x"], point["y"]), "R")
            scored = phase2_score.score_call(call, row, gt2d, edt, _geom(row, "R"))
        row_rank.update({"status": scored["status"], "hit": scored["hit"],
                         "distance_mm": scored["distance_mm"]})
        out.append(row_rank)
    return out


def _score_chat(resp: dict, row: dict, gt2d, edt, base) -> list[dict]:
    """MedGemma: strict-JSON answers (box -> centre; mark id; explicit uncertain)."""
    parsed = resp.get("parsed")
    status = "parse_fail" if (resp.get("parse_error") or parsed is None) else None
    answer = (parsed or {}).get("answer")
    uncertain = isinstance(answer, str) and answer.strip().lower() == "uncertain"
    row0 = dict(base, rank=0, point=None, snapped_mark_id=None, snap_distance_px=None,
                raw_point_hit=None, raw_point_distance_mm=None)
    if status:
        row0["status"] = status
        row0.update({"hit": False, "distance_mm": None})
        return [row0]
    if uncertain:
        row0["status"] = "abstain"  # explicit `uncertain` option: counted, never retried
        row0.update({"hit": False, "distance_mm": None})
        return [row0]

    if resp["condition"] == "marks_id":
        mark_id = (parsed or {}).get("mark_id")
        if mark_id is None:
            row0.update({"status": "parse_fail", "hit": False, "distance_mm": None})
            return [row0]
        call = _phase2_call(resp, answer_snippet(int(mark_id)), "S")
        scored = phase2_score.score_call(call, row, gt2d, edt, _geom(row, "S"))
        row0["snapped_mark_id"] = int(mark_id)
        row0.update({"status": scored["status"], "hit": scored["hit"],
                     "distance_mm": scored["distance_mm"]})
        return [row0]

    box = (parsed or {}).get("box")
    if not (isinstance(box, list) and len(box) == 4 and all(
            isinstance(v, (int, float)) and 0 <= v <= 1 for v in box)):
        row0.update({"status": "parse_fail", "hit": False, "distance_mm": None})
        return [row0]
    y0, x0, y1, x1 = (float(v) for v in box)  # MedGemma/Gemma convention: [y0, x0, y1, x1]
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    if resp["condition"] == "raw_box":
        geom = _geom(row, "R")
        px, py = cx * geom.panel_w, cy * geom.panel_h
        call = _phase2_call(resp, answer_point(px, py), "R")
        scored = phase2_score.score_call(call, row, gt2d, edt, geom)
        row0["point"] = [round(px, 2), round(py, 2)]
        row0.update({"status": scored["status"], "hit": scored["hit"],
                     "distance_mm": scored["distance_mm"]})
        return [row0]
    # marks_box: centre on the combined S image (two panel_w panels + gap), snapped
    # to the nearest drawn mark
    geom = _geom(row, "S")
    px, py = cx * geom_total_width(geom), cy * geom.panel_h
    row0["point"] = [round(px, 2), round(py, 2)]
    mark, snap_d = snap_to_mark(px, py, row["drawn_marks"])
    row0["snapped_mark_id"] = mark["id"] if mark else None
    row0["snap_distance_px"] = round(snap_d, 2) if mark else None
    if mark is None:
        row0.update({"status": "parse_fail", "hit": False, "distance_mm": None})
        return [row0]
    call = _phase2_call(resp, answer_snippet(mark["id"]), "S")
    scored = phase2_score.score_call(call, row, gt2d, edt, _geom(row, "S"))
    _attach_raw_point(row0, resp, row, gt2d, edt, {"x": px, "y": py})
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


def _attach_raw_point(row_rank: dict, resp: dict, row: dict, gt2d, edt, point: dict) -> None:
    """Secondary diagnostic: the raw (unsnapped) point on the marked panel.

    combined-image px -> marked-panel px (subtract panel 0 width + gap); the raw point
    is scored with the R path on view_geom_S. Never mixed with the snapped answer.
    """
    geom = _geom(row, "S")
    px = point["x"] - (geom.panel_w + PANEL_GAP_PX)
    py = point["y"]
    if px < 0:  # landed on the clean panel: no marked-panel voxel
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
        lines.append("## Paired vs Phase 2 frontier baseline (case-level majority hits)")
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
    lines.append("- raw / raw_box: point (or box centre) on the raw image -> panel_to_voxel -> "
                 "inside GT on k* (score.py R path).")
    lines.append("- marks / marks_box: point or box centre snapped to the nearest drawn mark "
                 "(combined_px, ties -> lowest id) -> that mark's projected voxel inside GT "
                 "(score.py S path). The unsnapped point is reported separately "
                 "(raw_point_hit), never mixed with the snapped answer.")
    lines.append("- marks_id: mark id answered directly (S path; unknown id = invalid_mark).")
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

    scored: list[dict] = []
    gt_cache: dict[str, tuple] = {}
    for resp in resp_lines:
        case = resp["case"]
        if case not in gt_cache:
            mask = load_mask(get_case(case))
            gt_cache[case] = phase2_score.gt_slices_and_edt(rows[case], mask)
        gt2d, edt = gt_cache[case]
        scored.extend(score_response(resp, rows[case], gt2d, edt))
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
        # per-case majority hits (rank 0, majority over reps) for McNemar pairing
        per_case_hits[key] = {}
        for case in sorted(by_case):
            calls = [c for c in by_case[case] if c["rank"] == 0]
            kind, _val, hit = phase2_score.majority_vote(
                [{"status": c["status"], "value": c["rank"], "hit": c["hit"]} for c in calls])
            per_case_hits[key][case] = bool(hit) if hit is not None else False

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
                kind, _val, hit = phase2_score.majority_vote(calls_)
                baseline_hits[case] = bool(hit) if hit is not None else False
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
    }, indent=2) + "\n")
    write_report(run_dir, cond_keys, agg, per_hist, baseline,
                 {**meta, "prepare_manifest": str(prepare_dir / "manifest.jsonl")})

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
