#!/usr/bin/env python3
"""E16 candidate-decision replay study, stage 2: score responses and write the report.

Strict answer parsing (done here, never in the TS harness): the last line of the
reply must be `ACCEPT: none` or `ACCEPT: C1, C3, ...` (case-insensitive `none`;
candidate ids must be uppercase `C<number>`). An id that is not in the review is
`invalid_ids` (counted; the call carries no confusion counts and no Dice, same
convention as `invalid_mark` in harness/study/score.py). Anything else is a
`parse_fail` (counted, no counts, no Dice).

Per replayed review (manifest row results/candidate_replay/manifest.jsonl):
- TP/FP/FN/TN over candidates, positive class = label 'lesion' (GT-derived).
- Dice after accepting the chosen ids: candidates are disjoint add-regions
  outside the mask (asserted by prepare_candidate_replay.py), so
  dice_after = 2*(mask_gt_vox + sum(inter_vox)) / (mask_vox + sum(region_vox) + gt_vox)
  and dice_change = dice_after - dice_reviewed. Accepting nothing changes nothing.
- Oracle: accept exactly the lesion-labelled candidates (same formula).

Aggregates per condition: pooled sensitivity/specificity, mean accepted per
review, mean Dice change vs dice_reviewed (per review = mean over reps, then
mean over reviews), oracle Dice change, per-histotype breakdown, parse failures,
invalid ids, response errors, cost/latency, and a per-review table. The original
agent decisions (manifest `agent_accepted`) are case-level across the whole
trajectory, not per review version, so they are shown for reference only.

Run:  python harness/study/score_candidate_replay.py --run <name>
Reads results/candidate_replay/<run>/responses.jsonl (+ the manifest), writes
scored.jsonl, score.json and report.md into the same run directory.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "results" / "candidate_replay" / "manifest.jsonl"
HISTOTYPES = ("F", "L", "M", "N", "V")

# Strict: uppercase C ids; `none` case-insensitive (spec).
RE_ACCEPT = re.compile(r"^ACCEPT:\s*(?:(?i:none)|C\d+(?:\s*,\s*C\d+)*)\s*$")


# ------------------------------------------------------------------ parsing
def parse_accept(raw_text: str | None) -> dict:
    """Strict parse of the final ACCEPT line. Returns {status, value}.

    Trailing blank lines are tolerated (generation artefact). status is one of
    ok | parse_fail. Value is the list of accepted candidate ids ([] for none).
    """
    if raw_text is None or str(raw_text).strip() == "":
        return {"status": "parse_fail", "value": []}
    last = str(raw_text).rstrip().split("\n")[-1].strip()
    if not RE_ACCEPT.match(last):
        return {"status": "parse_fail", "value": []}
    payload = last.split(":", 1)[1].strip()
    if payload.lower() == "none":
        return {"status": "ok", "value": []}
    ids = [part.strip() for part in payload.split(",")]
    return {"status": "ok", "value": ids}


# ------------------------------------------------------------------ dice
def dice_after(row: dict, accepted: list[str]) -> tuple[float, float]:
    """Dice after accepting `accepted` ids and the change vs dice_reviewed."""
    by_id = {c["id"]: c for c in row["candidates"]}
    inter = sum(int(by_id[i]["inter_vox"]) for i in accepted)
    region = sum(int(by_id[i]["region_vox"]) for i in accepted)
    denom = int(row["mask_vox"]) + region + int(row["gt_vox"])
    num = 2 * (int(row["mask_gt_vox"]) + inter)
    dice = num / denom if denom else 0.0
    return dice, dice - float(row["dice_reviewed"])


def oracle_dice(row: dict) -> tuple[float, float]:
    """Dice and change for accepting exactly the lesion-labelled candidates."""
    lesion = [c["id"] for c in row["candidates"] if c["label"] == "lesion"]
    return dice_after(row, lesion)


# ------------------------------------------------------------------ per call
def score_call(resp: dict, row: dict) -> dict:
    """Score one response line against one manifest row. Pure; no GT file access."""
    out = {
        "key": resp["key"], "case": resp["case"], "condition": resp["condition"],
        "rep": resp.get("rep"), "status": None, "accepted": [],
        "invalid_ids": [], "n_candidates": len(row["candidates"]),
        "n_lesion": sum(c["label"] == "lesion" for c in row["candidates"]),
        "tp": None, "fp": None, "fn": None, "tn": None,
        "dice_after": None, "dice_change": None,
        "response_error": bool(resp.get("error")) or resp.get("stop_reason") == "error",
        "latency_ms": resp.get("latency_ms"), "cost_usd": (resp.get("usage") or {}).get("cost_usd"),
    }
    if out["response_error"]:
        out["status"] = "response_error"
        return out
    parsed = parse_accept(resp.get("raw_text"))
    if parsed["status"] != "ok":
        out["status"] = parsed["status"]
        return out
    known = {c["id"] for c in row["candidates"]}
    invalid = [i for i in parsed["value"] if i not in known]
    if invalid:
        out["status"] = "invalid_ids"
        out["invalid_ids"] = sorted(invalid)
        return out
    accepted = parsed["value"]
    if len(set(accepted)) != len(accepted):
        out["status"] = "duplicate_ids"
        return out
    lesion = {c["id"] for c in row["candidates"] if c["label"] == "lesion"}
    out["status"] = "ok"
    out["accepted"] = accepted
    out["tp"] = len([i for i in accepted if i in lesion])
    out["fp"] = len([i for i in accepted if i not in lesion])
    out["fn"] = len(lesion) - out["tp"]
    out["tn"] = out["n_candidates"] - out["n_lesion"] - out["fp"]
    out["dice_after"], out["dice_change"] = dice_after(row, accepted)
    return out


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def aggregate(calls: list[dict], oracle_by_key: dict[str, float]) -> dict:
    """Aggregate one subset of scored calls (all calls of one condition, or one
    histotype). Sensitivity/specificity are pooled over ok calls; mean Dice
    change is per review (mean over reps) then over reviews."""
    ok = [c for c in calls if c["status"] == "ok"]
    tp = sum(c["tp"] for c in ok)
    fp = sum(c["fp"] for c in ok)
    fn = sum(c["fn"] for c in ok)
    tn = sum(c["tn"] for c in ok)
    dice_by_key: dict[str, list[float]] = {}
    for c in ok:
        dice_by_key.setdefault(c["key"], []).append(c["dice_change"])
    review_means = [sum(v) / len(v) for v in dice_by_key.values()]
    keys_here = sorted({c["key"] for c in calls})
    return {
        "n_calls": len(calls),
        "n_reviews": len(keys_here),
        "n_ok": len(ok),
        "n_parse_fail": sum(c["status"] == "parse_fail" for c in calls),
        "n_invalid_ids": sum(c["status"] == "invalid_ids" for c in calls),
        "n_duplicate_ids": sum(c["status"] == "duplicate_ids" for c in calls),
        "n_response_error": sum(c["status"] == "response_error" for c in calls),
        "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) else None,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "mean_accepted_per_review": _mean([len(c["accepted"]) for c in ok]),
        "mean_dice_change_vs_reviewed": _mean(review_means),
        "oracle_dice_change": _mean([oracle_by_key[k] for k in keys_here if k in oracle_by_key]),
        "mean_latency_ms": _mean([c["latency_ms"] for c in calls if c["latency_ms"] is not None]),
        "cost_usd": round(sum(c["cost_usd"] for c in calls if c["cost_usd"] is not None), 4),
    }


# ------------------------------------------------------------------ report
def fmt(x, scale=1, nd=3) -> str:
    return "–" if x is None else f"{x * scale:.{nd}f}"


def write_report(run_dir: Path, rows: dict[str, dict], calls: list[dict], meta: dict) -> None:
    oracle_by_key = {key: oracle_dice(row)[1] for key, row in rows.items()}
    conds = sorted({c["condition"] for c in calls})
    agg = {cond: aggregate([c for c in calls if c["condition"] == cond], oracle_by_key) for cond in conds}

    lines: list[str] = []
    lines.append(f"# E16 candidate-decision replay — run `{run_dir.name}`")
    lines.append("")
    lines.append(
        f"Model `{meta.get('model')}`, thinking `{meta.get('thinking')}`, {meta.get('reps')} reps × "
        f"{len(rows)} replayed reviews × {len(conds)} conditions ({', '.join(conds)}); scored "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}. "
        f"pi SDK {meta.get('pi_sdk_version')} (auth shim {meta.get('auth_shim_version')}). "
        f"Positive class = candidate labelled 'lesion' by GT (precision > 0.5).")
    lines.append("")
    lines.append("## Caveats")
    lines.append("- Single tool-less decision turn over replayed review images; no interaction, "
                 "so the candidate sets are frozen (the agent could iterate with seg_accept/seg_view).")
    lines.append("- GT labels are per candidate (precision > 0.5 → 'lesion'); borderline candidates "
                 "(precision near 0.5) are noisy ground truth.")
    lines.append("- The original agent decisions (`agent_accepted`) are case-level over the whole "
                 "trajectory, not per review version — shown in the per-review table for reference only.")
    lines.append("")
    lines.append("## Headline (pooled over reviews, reps averaged for Dice)")
    lines.append("")
    lines.append("| condition | n calls | sens | spec | mean accepted/review | mean Dice Δ vs reviewed | "
                 "oracle Dice Δ | parse fail | invalid ids | resp error |")
    lines.append("|---|" + "---|" * 9)
    for cond in conds:
        a = agg[cond]
        lines.append(
            f"| {cond} | {a['n_calls']} | {fmt(a['sensitivity'])} | {fmt(a['specificity'])} | "
            f"{fmt(a['mean_accepted_per_review'], 1, 2)} | {fmt(a['mean_dice_change_vs_reviewed'])} | "
            f"{fmt(a['oracle_dice_change'])} | {a['n_parse_fail']} | {a['n_invalid_ids']} | "
            f"{a['n_response_error']} |")
    lines.append("")
    lines.append("## Per histotype (sens / spec / mean Dice Δ)")
    lines.append("")
    header_cells = ["histotype"]
    for cond in conds:
        header_cells += [f"{cond} sens", f"{cond} spec", f"{cond} Dice Δ"]
    header_cells.append("n reviews")
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|---" * len(header_cells) + "|")
    for h in HISTOTYPES:
        hcalls = {cond: [c for c in calls if c["condition"] == cond
                         and rows.get(c["key"], {}).get("histotype") == h] for cond in conds}
        if not any(hcalls.values()):
            continue
        cells = [h]
        for cond in conds:
            if hcalls[cond]:
                a = aggregate(hcalls[cond], oracle_by_key)
                cells += [fmt(a['sensitivity']), fmt(a['specificity']), fmt(a['mean_dice_change_vs_reviewed'])]
            else:
                cells += ["–", "–", "–"]
        n_rev = len({c["key"] for cond in conds for c in hcalls[cond]})
        cells.append(str(n_rev))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Per-review table")
    lines.append("")
    header = "| key | h | seq | cands (lesion) | dice reviewed | oracle Δ | " \
             + " | ".join(f"{cond} reps (accepted, Δ)" for cond in conds) \
             + " | agent accepted (case-level, reference) |"
    lines.append(header)
    lines.append("|---" * (6 + len(conds) + 1) + "|")
    for key in sorted(rows):
        row = rows[key]
        o_change = oracle_dice(row)[1]
        cells = [key, row["histotype"], row["sequence"],
                 f"{len(row['candidates'])} ({len([c for c in row['candidates'] if c['label'] == 'lesion'])})",
                 f"{row['dice_reviewed']:.3f}", f"{o_change:+.3f}"]
        for cond in conds:
            cc = sorted((c for c in calls if c["key"] == key and c["condition"] == cond), key=lambda c: c["rep"])
            if not cc:
                cells.append("–")
                continue
            parts = []
            for c in cc:
                if c["status"] == "ok":
                    ids = ",".join(c["accepted"]) if c["accepted"] else "none"
                    parts.append(f"{ids} ({c['dice_change']:+.3f})")
                else:
                    parts.append(c["status"] + (f" {','.join(c['invalid_ids'])}" if c["invalid_ids"] else ""))
            cells.append("; ".join(parts))
        ref = sorted(set(row.get("agent_accepted") or []) & {c["id"] for c in row["candidates"]})
        cells.append(",".join(ref) if ref else "–")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lat = [c["latency_ms"] for c in calls if c["latency_ms"] is not None]
    cost = sum(c["cost_usd"] for c in calls if c["cost_usd"] is not None)
    lines.append(f"## Cost / latency")
    lines.append("")
    lines.append(f"- {len(calls)} calls, total cost ≈ **${cost:.2f}**, mean latency "
                 f"{_mean([float(x) for x in lat]) / 1000:.1f}s"
                 + ("" if lat else " (no latency recorded)"))
    lines.append("")
    lines.append("## Scoring rules")
    lines.append("- Last line strict `ACCEPT: none` or `ACCEPT: C1, C2` (`none` case-insensitive, "
                 "ids uppercase `C<number>`); anything else = `parse_fail`.")
    lines.append("- Ids not in the review = `invalid_ids` (counted; no confusion counts, no Dice).")
    lines.append("- dice_after = 2·(mask_gt_vox + Σ inter_vox) / (mask_vox + Σ region_vox + gt_vox); "
                 "candidates are disjoint and outside the mask (asserted at prepare time).")
    lines.append("- Mean Dice Δ: per review mean over reps, then mean over reviews.")
    lines.append("- Oracle Δ: accept exactly the lesion-labelled candidates (condition-independent).")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")
    return


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="run name under results/candidate_replay/")
    ap.add_argument("--manifest", default=str(MANIFEST_PATH))
    args = ap.parse_args()
    run_dir = PROJECT_ROOT / "results" / "candidate_replay" / args.run
    resp_path = run_dir / "responses.jsonl"
    if not resp_path.exists():
        raise SystemExit(f"no responses.jsonl in {run_dir}")

    rows = {r["key"]: r for r in (json.loads(l) for l in open(args.manifest))}
    resp_lines = [json.loads(l) for l in open(resp_path)]
    unknown = sorted({r["key"] for r in resp_lines} - set(rows))
    if unknown:
        raise SystemExit(f"responses for unknown manifest keys: {unknown}")

    calls = [score_call(resp, rows[resp["key"]]) for resp in resp_lines]
    (run_dir / "scored.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))

    oracle_by_key = {key: oracle_dice(row)[1] for key, row in rows.items()}
    conds = sorted({c["condition"] for c in calls})
    keys = sorted({r["key"] for r in resp_lines})
    agg = {cond: aggregate([c for c in calls if c["condition"] == cond], oracle_by_key) for cond in conds}
    per_hist = {}
    for h in HISTOTYPES:
        hkeys = {k for k, r in rows.items() if r["histotype"] == h} & set(keys)
        sub = {cond: [c for c in calls if c["condition"] == cond and c["key"] in hkeys] for cond in conds}
        if any(sub.values()):
            per_hist[h] = {cond: aggregate(sub[cond], oracle_by_key) for cond in conds if sub[cond]}

    meta = {}
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    (run_dir / "score.json").write_text(json.dumps(
        {"aggregate": agg, "per_histotype": per_hist, "oracle_dice_change_by_key": oracle_by_key,
         "manifest_keys_scored": keys}, indent=2) + "\n")
    write_report(run_dir, rows, calls, meta)

    print(f"[score] {len(calls)} calls scored -> {run_dir / 'report.md'}")
    for cond in conds:
        a = agg[cond]
        print(f"[score] {cond}: sens={a['sensitivity']} spec={a['specificity']} "
              f"accepted/review={a['mean_accepted_per_review']} diceΔ={a['mean_dice_change_vs_reviewed']} "
              f"oracleΔ={a['oracle_dice_change']} parse_fail={a['n_parse_fail']} invalid={a['n_invalid_ids']}")


if __name__ == "__main__":
    main()
