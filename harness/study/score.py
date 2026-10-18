#!/usr/bin/env python3
"""Phase 2 localization study, stage 3: score responses and write the gate report.

Scoring (design §3 Phase 2; strict answer parsing):
- S family (S, S-gen, S-desc, S-ex): last line of the reply must be `ANSWER: <n>`.
  n must be among the marks drawn for that case (manifest `drawn_marks`); a hit is
  n's projected in-plane voxel inside GT on k* (the CaseState active-view resolution
  rule). A mark number that is not drawn = `invalid_mark` (miss). Anything else =
  `parse_fail` (miss). Parse failures and invalid marks are reported separately.
- R: last line must be `ANSWER: (x, y)`; (x, y) are panel pixels of the raw-pointing
  image (render_raw_pointing rulers). Outside the image panel = miss (`outside_panel`).
  Otherwise panel_to_voxel maps to a voxel on k*; hit = that voxel inside GT on k*.
- distance to GT in mm on k*: 0 if inside, else the 2D Euclidean distance transform
  to the nearest GT voxel (raw [i, j] grid), evaluated at the chosen voxel. Calls
  with parse failures / invalid marks / response errors carry no distance.
- coverage: cases with >=1 drawn coarse mark inside GT on k* (from prepare manifest).

Gate (harness_design_v2.md §3, thresholds fixed before data):
- S hit rate >= 0.80 and S - R >= 0.10  -> proceed with S.
- S < 0.80: coverage < 0.90 -> fix marks (on design cases only) and rerun;
  coverage >= 0.90 -> run S-desc / S-ex and adopt whichever passes.

Run:  uv run --project harness/server python harness/study/score.py --run <name>
Reads results/phase2_dev/<run>/responses.jsonl (+ the prepare manifest), writes
scored.jsonl, score.json and report.md into the same run directory.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREPARE_DIR = PROJECT_ROOT / "results" / "phase2_dev" / "prepare"

S_FAMILY = ("S", "S-gen", "S-desc", "S-ex")
HISTOTYPES = ("F", "L", "M", "N", "V")

RE_S = re.compile(r"^ANSWER:\s*(\d+)\s*$")
RE_R = re.compile(r"^ANSWER:\s*\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)\s*$")

GATE_S_MIN = 0.80
GATE_DIFF_MIN = 0.10
GATE_COVERAGE_MIN = 0.90


# ------------------------------------------------------------------ parsing
def parse_answer(raw_text: str | None, condition: str) -> dict:
    """Strict parse of the final ANSWER line. Returns {status, value|None}.

    Trailing blank lines are tolerated (generation artefact); any other content on
    or after the final line fails. `condition == "R"` parses `(x, y)`, else `<n>`.
    """
    if raw_text is None or str(raw_text).strip() == "":
        return {"status": "parse_fail", "value": None}
    last = str(raw_text).rstrip().split("\n")[-1].strip()
    if condition == "R":
        m = RE_R.match(last)
        if m:
            return {"status": "ok", "value": (float(m.group(1)), float(m.group(2)))}
    else:
        m = RE_S.match(last)
        if m:
            return {"status": "ok", "value": int(m.group(1))}
    return {"status": "parse_fail", "value": None}


# ------------------------------------------------------------------ scoring
def gt_slices_and_edt(row: dict, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """GT on k* (raw [i, j] bool) and its EDT-to-GT in mm (0 on GT voxels)."""
    k = int(row["k_star"])
    gt2d = np.asarray(mask[:, :, k] > 0)
    spacing = tuple(float(s) for s in row["spacing"])
    edt = ndi.distance_transform_edt(~gt2d, sampling=(spacing[0], spacing[1]))
    return gt2d, edt


def score_call(resp: dict, row: dict, gt2d: np.ndarray, edt: np.ndarray, view_geom) -> dict:
    """Score one response line against one manifest row. Pure; no GT file access."""
    out = {
        "case": resp["case"], "condition": resp["condition"], "rep": resp.get("rep"),
        "status": None, "value": None, "hit": False, "distance_mm": None,
        "voxel": None, "inplane_row_col": None,
        "response_error": bool(resp.get("error")) or resp.get("stop_reason") == "error",
    }
    if out["response_error"]:
        out["status"] = "response_error"
        return out
    parsed = parse_answer(resp.get("raw_text"), resp["condition"])
    out["status"] = parsed["status"]
    out["value"] = parsed["value"]
    if parsed["status"] != "ok":
        return out

    if resp["condition"] == "R":
        import segd.render as R
        x, y = parsed["value"]
        if not (0 <= x < view_geom.panel_w and 0 <= y < view_geom.panel_h):
            out["status"] = "outside_panel"
            return out
        ijk = R.panel_to_voxel(view_geom.axis, view_geom.slice_idx, (x, y), view_geom)
        i, j = int(ijk[0]), int(ijk[1])          # raw [i, j] in-plane voxel
        out["voxel"] = [int(v) for v in ijk]
        out["inplane_row_col"] = [i, j]
        inside = bool(gt2d[i, j])
    else:
        drawn = {int(m["id"]): m for m in row["drawn_marks"]}
        mark = drawn.get(int(parsed["value"]))
        if mark is None:
            out["status"] = "invalid_mark"
            return out
        i, j = int(mark["projected_ijk"][0]), int(mark["projected_ijk"][1])
        out["voxel"] = list(mark["projected_ijk"])
        out["inplane_row_col"] = [i, j]
        inside = bool(mark["inside_gt_on_k_star"]) and bool(gt2d[i, j])
    out["hit"] = inside
    out["distance_mm"] = 0.0 if inside else float(edt[i, j])
    return out


# ------------------------------------------------------------------ aggregation
def majority_vote(calls: list[dict]) -> tuple[str, object | None, bool | None]:
    """Majority answer over a case's rep calls (only `status == "ok"` calls vote).

    Tie-break: lowest mark id (S) / lexicographically smallest (x, y) (R).
    Returns (kind, answer, hit): kind ∈ none | strict | tie."""
    ok = [c for c in calls if c["status"] == "ok"]
    if not ok:
        return ("none", None, None)
    counts = Counter(c["value"] for c in ok)
    top = max(counts.values())
    winners = sorted(a for a, n in counts.items() if n == top)
    best = winners[0]
    hit = next(c["hit"] for c in ok if c["value"] == best)
    kind = "strict" if top * 2 > len(ok) else "tie"
    return (kind, best, bool(hit))


def aggregate(calls: list[dict]) -> dict:
    agg: dict[str, dict] = {}
    for cond in sorted({c["condition"] for c in calls}):
        cc = [c for c in calls if c["condition"] == cond]
        dists = [c["distance_mm"] for c in cc if c["distance_mm"] is not None]
        cases = sorted({c["case"] for c in cc})
        maj_hit = 0
        maj_strict_hit = 0
        n_strict = 0
        for case in cases:
            kind, _answer, hit = majority_vote([c for c in cc if c["case"] == case])
            if hit is None:
                continue
            maj_hit += int(hit)
            if kind == "strict":
                n_strict += 1
                maj_strict_hit += int(hit)
        agg[cond] = {
            "n_calls": len(cc),
            "n_parse_fail": sum(c["status"] == "parse_fail" for c in cc),
            "n_invalid_mark": sum(c["status"] == "invalid_mark" for c in cc),
            "n_outside_panel": sum(c["status"] == "outside_panel" for c in cc),
            "n_response_error": sum(c["status"] == "response_error" for c in cc),
            "hit_rate_mean_over_reps": round(sum(c["hit"] for c in cc) / max(len(cc), 1), 4),
            "distance_mean_mm": round(float(np.mean(dists)), 3) if dists else None,
            "distance_median_mm": round(float(np.median(dists)), 3) if dists else None,
            "n_cases": len(cases),
            "n_cases_with_majority_answer": sum(
                majority_vote([c for c in cc if c["case"] == case])[0] != "none"
                for case in cases),
            "majority_hit_rate_per_case": round(maj_hit / max(len(cases), 1), 4),
            "strict_majority_hit_rate": round(maj_strict_hit / max(n_strict, 1), 4),
        }
    return agg


def gate_evaluation(agg: dict, coverage_rate: float) -> list[str]:
    lines = []
    s = agg.get("S", {}).get("hit_rate_mean_over_reps", 0.0)
    r = agg.get("R", {}).get("hit_rate_mean_over_reps", 0.0)
    diff = s - r
    lines.append(f"- S hit rate (mean over reps): **{s:.3f}**; R: **{r:.3f}**; S − R: **{diff:+.3f}**")
    if s >= GATE_S_MIN and diff >= GATE_DIFF_MIN:
        lines.append(f"- Gate (S ≥ {GATE_S_MIN} and S − R ≥ {GATE_DIFF_MIN}): **PASS** → proceed with S.")
    elif s < GATE_S_MIN:
        if coverage_rate < GATE_COVERAGE_MIN:
            lines.append(f"- Gate: **FAIL** (S < {GATE_S_MIN}) and drawn-mark coverage "
                         f"{coverage_rate:.3f} < {GATE_COVERAGE_MIN} → **fix marks** (design "
                         "cases only), then rerun.")
        else:
            lines.append(f"- Gate: **FAIL** (S < {GATE_S_MIN}); coverage {coverage_rate:.3f} ≥ "
                         f"{GATE_COVERAGE_MIN} → **run S-desc / S-ex** and adopt whichever passes.")
    else:
        lines.append(f"- Gate: **FAIL** (S − R = {diff:+.3f} < {GATE_DIFF_MIN} although S ≥ "
                     f"{GATE_S_MIN}) → stop and discuss.")
    return lines


# ------------------------------------------------------------------ report
def write_report(run_dir: Path, rows: dict[str, dict], calls: list[dict],
                 coverage_rate: float, meta: dict) -> None:
    hist_of = {case: row["histotype"] for case, row in rows.items()}
    agg = aggregate(calls)
    cond_names = sorted(agg)
    per_hist: dict[str, dict[str, dict]] = {}
    for h in HISTOTYPES:
        hc = [c for c in calls if hist_of.get(c["case"]) == h]
        if hc:
            per_hist[h] = aggregate(hc)

    uncovered = sorted(c for c, r in rows.items() if not r["coverage_on_k_star"])
    lines: list[str] = []
    lines.append(f"# Phase 2 localization study — run `{meta.get('run', run_dir.name)}`")
    lines.append("")
    lines.append(f"Model `{meta.get('model')}`, thinking `{meta.get('thinking')}`, "
                 f"{meta.get('reps')} reps × {len(rows)} design cases; scored "
                 f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}. "
                 f"pi SDK {meta.get('pi_sdk_version')} (auth shim {meta.get('auth_shim_version')}).")
    lines.append("")
    lines.append("## Caveats")
    lines.append("- **Dev cases**: the same 30 design cases the marks were tuned on → coverage is optimistic.")
    lines.append("- **n = 6 per histotype** (one case ≈ 17 points) → the gate is a coarse go/no-go signal; report per-case results.")
    lines.append("- Coverage counts drawn coarse marks inside GT on k* only (single-slice study; the full test-split study runs later).")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append("| condition | n | hit rate (mean over reps) | majority-vote hit rate (per case) | "
                 "coverage | parse fail | invalid mark | outside panel | resp error | dist mean mm | dist median mm |")
    lines.append("|---|" + "---|" * 10)
    for cond in cond_names:
        a = agg[cond]
        cov = f"{coverage_rate:.3f}" if cond in S_FAMILY else "–"
        lines.append(
            f"| {cond} | {a['n_calls']} | {a['hit_rate_mean_over_reps']:.3f} | "
            f"{a['majority_hit_rate_per_case']:.3f} | {cov} | {a['n_parse_fail']} | "
            f"{a['n_invalid_mark']} | {a['n_outside_panel']} | {a['n_response_error']} | "
            f"{a['distance_mean_mm'] if a['distance_mean_mm'] is not None else '–'} | "
            f"{a['distance_median_mm'] if a['distance_median_mm'] is not None else '–'} |")
    lines.append("")
    lines.append(f"Drawn-mark coverage on k*: {len(rows) - len(uncovered)}/{len(rows)} = "
                 f"{coverage_rate:.3f}"
                 + (f"; cases without a drawn mark inside GT on k*: {', '.join(uncovered)}."
                    if uncovered else "."))
    lines.append("")
    lines.append("## Gate (harness_design_v2.md §3, thresholds fixed before data)")
    lines.extend(gate_evaluation(agg, coverage_rate))
    lines.append("")
    s = agg.get("S", {}).get("hit_rate_mean_over_reps")
    gen = agg.get("S-gen", {}).get("hit_rate_mean_over_reps")
    if s is not None and gen is not None:
        verdict = ("S ≈ S-gen → the model is not using histotype knowledge"
                   if abs(s - gen) < 0.10 else
                   "S > S-gen → the model uses histotype knowledge"
                   if s > gen else "S < S-gen — unexpected; inspect per-case results")
        lines.append("## S vs S-gen (knowledge question)")
        lines.append(f"S {s:.3f} vs S-gen {gen:.3f} (Δ {s - gen:+.3f}) → {verdict}.")
        lines.append("")
    lines.append("## Per histotype (hit rate, mean over reps)")
    lines.append("")
    lines.append("| histotype | " + " | ".join(cond_names) + " |")
    lines.append("|---" * (len(cond_names) + 1) + "|")
    for h in HISTOTYPES:
        cells = []
        for cond in cond_names:
            a = per_hist.get(h, {}).get(cond)
            cells.append(f"{a['hit_rate_mean_over_reps']:.3f}" if a else "–")
        lines.append(f"| {h} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Per-case table")
    lines.append("")
    lines.append("| case | h | seq | k* | " + " | ".join(f"{c}" for c in cond_names) + " | covered |")
    lines.append("|---" * (len(cond_names) + 6) + "|")
    for case in sorted(rows):
        row = rows[case]
        cells = [case, row["histotype"], row["sequence"], str(row["k_star"])]
        for cond in cond_names:
            cc = sorted((c for c in calls if c["case"] == case and c["condition"] == cond),
                        key=lambda c: c["rep"])
            if not cc:
                cells.append("–")
                continue
            parts = []
            for c in cc:
                if c["status"] == "ok":
                    v = c["value"]
                    vs = f"({v[0]:g},{v[1]:g})" if isinstance(v, tuple) else str(v)
                    tail = f" d={c['distance_mm']:.0f}" if c["distance_mm"] is not None else ""
                    parts.append(f"{vs}{'✓' if c['hit'] else '✗'}{tail}")
                else:
                    parts.append(c["status"])
            cells.append(", ".join(parts))
        cells.append("yes" if row["coverage_on_k_star"] else "**no**")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Scoring rules")
    lines.append("- S family: chosen mark's projected voxel inside GT on k*; `parse_fail` / "
                 "`invalid_mark` / response errors count as misses and are reported separately.")
    lines.append("- Majority vote per case over reps; ties broken by lowest mark id (S) / "
                 "smallest (x, y) (R). `strict_majority_hit_rate` in score.json is restricted "
                 "to cases where ≥2 reps agree.")
    lines.append("- R: `panel_to_voxel(x, y)` inside GT on k*; a point outside the image panel = miss.")
    lines.append("- distance: 0 if inside; else 2D EDT to the nearest GT voxel on k* (mm).")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="run name under results/phase2_dev/")
    ap.add_argument("--prepare-dir", default=str(PREPARE_DIR))
    args = ap.parse_args()
    run_dir = PROJECT_ROOT / "results" / "phase2_dev" / args.run
    resp_path = run_dir / "responses.jsonl"
    if not resp_path.exists():
        raise SystemExit(f"no responses.jsonl in {run_dir}")

    rows = {r["case"]: r for r in (json.loads(l) for l in open(Path(args.prepare_dir) / "manifest.jsonl"))}
    resp_lines = [json.loads(l) for l in open(resp_path)]
    unknown = sorted({r["case"] for r in resp_lines} - set(rows))
    if unknown:
        raise SystemExit(f"responses for unknown cases: {unknown}")

    from segd.data import load_mask
    from segd.render import ViewGeom

    gt_cache: dict[str, tuple] = {}
    calls = []
    for resp in resp_lines:
        case = resp["case"]
        if case not in gt_cache:
            gt_cache[case] = gt_slices_and_edt(rows[case], load_mask(segd_data_row(case)))
        gt2d, edt = gt_cache[case]
        g = rows[case]["view_geom_R"]
        geom = ViewGeom(g["axis"], int(g["slice_idx"]), int(g["rows"]), int(g["cols"]),
                        tuple(g["crop"]), float(g["scale"]), int(g["panel_w"]), int(g["panel_h"]))
        calls.append(score_call(resp, rows[case], gt2d, edt, geom))

    (run_dir / "scored.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    coverage_rate = sum(r["coverage_on_k_star"] for r in rows.values()) / max(len(rows), 1)
    agg = aggregate(calls)
    (run_dir / "score.json").write_text(json.dumps(
        {"aggregate": agg, "coverage_rate": round(coverage_rate, 4)}, indent=2) + "\n")

    meta = {}
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    write_report(run_dir, rows, calls, coverage_rate, meta)
    print(f"[score] {len(calls)} calls scored -> {run_dir / 'report.md'}")
    for cond, a in agg.items():
        print(f"[score] {cond}: hit {a['hit_rate_mean_over_reps']:.3f} "
              f"(majority/case {a['majority_hit_rate_per_case']:.3f}) "
              f"parse_fail={a['n_parse_fail']} invalid={a['n_invalid_mark']} "
              f"outside={a['n_outside_panel']} dist_mean={a['distance_mean_mm']}")


def segd_data_row(case: str):
    from segd import data as segd_data
    return segd_data.get_case(case)


if __name__ == "__main__":
    main()
