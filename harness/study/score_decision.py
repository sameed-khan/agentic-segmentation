#!/usr/bin/env python3
"""Phase 5 Study B scorer: sensitivity/specificity/AUROC for the decision replays.

Works for any run under results/phase5/studyB/<set>/<run>/ (MedGemma via
replay_local.py, frontier via replay_decisions.ts — both record raw replies; JSON
parsing happens HERE so local and frontier calls are parsed identically).

Per set (labels are the GT-derived manifest labels; positive class named below):
- candidate_judgement: label lesion; sens = lesion candidates called lesion,
  spec = normal candidates called not_lesion. AUROC from `confidence` (lesion: p =
  confidence; not_lesion: p = 1 - confidence).
- done_check:         positive class = not_done (the risky miss is calling a
  fixable state done); AUROC from p(not_done).
- wrong_structure:    positive class = wrong_structure; sensitivity at specificity
  >= 0.95 via the score threshold (design-set threshold choice is frozen before
  any test-split use).
Answers: `uncertain` rows are excluded from sens/spec (counted as abstain; a
second sens/spec WITH uncertain-as-negative is reported for completeness).
Parse failures count as errors (misses), never retried.
Case-level bootstrap CIs; per-histotype breakdown; paired McNemar between two
runs of the same set (--compare-run) for local-vs-frontier contrasts.

Run:  uv run --project harness/server python harness/study/score_decision.py \
          --set done_check --run medgemma-q4 [--compare-run sonnet-high]
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
sys.path.insert(0, str(PROJECT_ROOT / "harness" / "local_models" / "medgemma"))

from medgemma_client import extract_json_object  # noqa: E402

HISTOTYPES = ("F", "L", "M", "N", "V")
DEFAULT_RESULTS_ROOT = "results/phase5/studyB"

# set -> positive class + allowed answers (strict JSON contract of the prompts)
SET_SPEC = {
    "candidate_judgement": {"positive": "lesion", "answers": ("lesion", "not_lesion", "uncertain"),
                            "answer_field": "answer"},
    "done_check": {"positive": "not_done", "answers": ("done", "not_done", "uncertain"),
                   "answer_field": "answer"},
    "wrong_structure": {"positive": "wrong_structure", "answers": ("wrong_structure", "structures_ok", "uncertain"),
                        "answer_field": "answer"},
}
SPECIFICITY_TARGET = 0.95


# ------------------------------------------------------------------ parsing
def parse_answer(raw_text: str | None, answers: tuple[str, ...]) -> dict:
    """Strict JSON parse of the reply. Returns {status, answer, confidence}.

    status: ok | uncertain | parse_fail. Unknown answer values are parse failures.
    """
    if raw_text is None or not str(raw_text).strip():
        return {"status": "parse_fail", "answer": None, "confidence": None}
    try:
        data = extract_json_object(str(raw_text))
    except (ValueError, json.JSONDecodeError):
        return {"status": "parse_fail", "answer": None, "confidence": None}
    answer = data.get("answer")
    if not isinstance(answer, str) or answer.strip().lower() not in answers:
        return {"status": "parse_fail", "answer": answer, "confidence": None}
    answer = answer.strip().lower()
    confidence = data.get("confidence")
    if not isinstance(confidence, (int, float)) or not (0 <= float(confidence) <= 1):
        confidence = None  # missing/malformed confidence: AUROC uses rank fallback below
    return {"status": ("uncertain" if answer == "uncertain" else "ok"),
            "answer": answer, "confidence": confidence}


def positive_probability(answer: str, confidence: float | None, positive: str) -> float:
    """P(positive) for AUROC: confidence in the stated answer, flipped for the negative.

    confidence missing (None never reaches here) -> 0.5 tie placeholder handled by the caller.
    """
    c = 0.5 if confidence is None else float(confidence)
    return c if answer == positive else 1.0 - c


# ------------------------------------------------------------------ scoring
def score_call(resp: dict, row: dict) -> dict:
    spec = SET_SPEC[row["set"]]
    positive = spec["positive"]
    out = {
        "run": resp.get("run"), "key": resp["key"], "case": resp["case"], "set": row["set"],
        "histotype": row["histotype"], "sequence": row["sequence"],
        "model": resp.get("model"), "rep": resp.get("rep"),
        "label": row["label"], "is_positive": int(row["label"] == positive),
        "latency_ms": resp.get("latency_ms"),
        "cost_usd": (resp.get("usage") or {}).get("cost_usd"),
        "response_error": bool(resp.get("error")),
        "status": None, "answer": None, "confidence": None, "p_positive": None,
        "tp": 0, "fp": 0, "tn": 0, "fn": 0, "counted": False,
    }
    if out["response_error"]:
        out["status"] = "response_error"
        return out
    parsed = parse_answer(resp.get("raw_text", resp.get("raw")), spec["answers"])
    out.update({"status": parsed["status"], "answer": parsed["answer"],
                "confidence": parsed["confidence"]})
    if parsed["status"] == "parse_fail":
        return out
    if parsed["status"] == "uncertain":
        return out  # abstain: excluded from sens/spec, counted in the abstain rate
    answer = parsed["answer"]
    out["p_positive"] = positive_probability(answer, parsed["confidence"], positive)
    out["counted"] = True
    predicted_positive = answer == positive
    if out["is_positive"]:
        out["tp" if predicted_positive else "fn"] = 1
    else:
        out["tn" if not predicted_positive else "fp"] = 1
    return out


# ------------------------------------------------------------------ metrics
def confusion(calls: list[dict]) -> dict:
    tp = sum(c["tp"] for c in calls)
    fp = sum(c["fp"] for c in calls)
    tn = sum(c["tn"] for c in calls)
    fn = sum(c["fn"] for c in calls)
    return {
        "n": len(calls),
        "n_counted": sum(c["counted"] for c in calls),
        "n_uncertain": sum(c["status"] == "uncertain" for c in calls),
        "n_parse_fail": sum(c["status"] == "parse_fail" for c in calls),
        "n_response_error": sum(c["status"] == "response_error" for c in calls),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) else None,
        "accuracy": round((tp + tn) / max(tp + tn + fp + fn, 1), 4),
    }


def auroc(scores_pos: list[float], scores_neg: list[float]) -> float | None:
    """AUROC = Mann-Whitney U / (n_pos*n_neg), tie-corrected. None when one side is empty."""
    if not scores_pos or not scores_neg:
        return None
    pos = np.asarray(scores_pos, dtype=float)
    neg = np.asarray(scores_neg, dtype=float)
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1, dtype=float)
    # average ranks within ties
    values = np.concatenate([pos, neg])
    for v in np.unique(values):
        mask = values == v
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    r_pos = ranks[: len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return round(float(u / (len(pos) * len(neg))), 4)


def threshold_at_specificity(scores_neg: list[float], target: float) -> float | None:
    """Smallest threshold (on p(positive)) achieving specificity >= target."""
    if not scores_neg:
        return None
    neg = np.sort(np.asarray(scores_neg, dtype=float))
    # specificity(t) = fraction of negatives with score < t; pick the t that first
    # reaches the target (scores are in [0,1]; +1e-9 so the threshold is achievable)
    idx = math.ceil(target * len(neg))
    idx = min(max(idx, 1), len(neg))
    return float(neg[idx - 1]) + 1e-9


def bootstrap_ci(flags: list[int], n_boot: int, seed: int) -> tuple[float, float]:
    if not flags:
        return (None, None)
    rng = np.random.default_rng(seed)
    arr = np.asarray(flags, dtype=float)
    stats = [arr[rng.integers(0, len(arr), size=len(arr))].mean() for _ in range(n_boot)]
    return (round(float(np.percentile(stats, 2.5)), 4), round(float(np.percentile(stats, 97.5)), 4))


def metrics_for(calls: list[dict], n_boot: int, seed: int) -> dict:
    conf = confusion(calls)
    counted = [c for c in calls if c["counted"]]
    pos_scores = [c["p_positive"] for c in counted if c["is_positive"]]
    neg_scores = [c["p_positive"] for c in counted if not c["is_positive"]]
    conf["auroc"] = auroc(pos_scores, neg_scores)
    conf["auroc_all_rows"] = auroc(  # uncertain rows enter as p=0.5 (rank ties)
        [c["p_positive"] if c["p_positive"] is not None else 0.5
         for c in calls if c["status"] in ("ok", "uncertain") and c["is_positive"]],
        [c["p_positive"] if c["p_positive"] is not None else 0.5
         for c in calls if c["status"] in ("ok", "uncertain") and not c["is_positive"]])
    # sensitivity CI over positive rows (tp as 0/1), specificity CI over negative rows
    conf["sens_ci95"] = bootstrap_ci([c["tp"] for c in counted if c["is_positive"]], n_boot, seed)
    conf["spec_ci95"] = bootstrap_ci([c["tn"] for c in counted if not c["is_positive"]], n_boot, seed)
    thr = threshold_at_specificity(neg_scores, SPECIFICITY_TARGET)
    conf["threshold_at_spec_0.95"] = round(thr, 4) if thr is not None else None
    if thr is not None and pos_scores:
        sens = sum(1 for s in pos_scores if s >= thr) / len(pos_scores)
        spec = sum(1 for s in neg_scores if s < thr) / len(neg_scores)
        conf["sens_at_spec_0.95"] = round(sens, 4)
        conf["spec_at_threshold"] = round(spec, 4)
    return conf


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2.0 * tail * 0.5 ** n)


def paired_compare(calls_a: list[dict], calls_b: list[dict], positive: str) -> dict | None:
    """McNemar between two runs of the same set, paired per key (majority over reps).

    Only keys where BOTH calls produced a counted decision enter (abstains excluded).
    """
    def by_key(calls: list[dict]) -> dict[str, int]:
        out: dict[str, list[int]] = {}
        for c in calls:
            if c["counted"]:
                out.setdefault(c["key"], []).append(c["tp"])
        return {k: int(np.mean(v) >= 0.5) for k, v in out.items()}

    a, b = by_key(calls_a), by_key(calls_b)
    keys = sorted(set(a) & set(b))
    n_b = sum(1 for k in keys if b[k] and not a[k])
    n_c = sum(1 for k in keys if a[k] and not b[k])
    if not keys:
        return None
    return {"n_paired_keys": len(keys), "n_b_hit_a_miss": n_b, "n_a_hit_b_miss": n_c,
            "mcnemar_p": round(mcnemar_exact(n_b, n_c), 5)}


# ------------------------------------------------------------------ report
def fmt(x, nd=3) -> str:
    return "–" if x is None else f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", required=True, dest="set_name")
    ap.add_argument("--run", required=True, help="run name under results/phase5/studyB/<set>/")
    ap.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--compare-run", default=None,
                    help="second run of the SAME set: paired McNemar (e.g. local vs frontier)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261019)
    args = ap.parse_args()

    set_dir = PROJECT_ROOT / args.results_root / args.set_name
    manifest_path = set_dir / "manifest.jsonl"
    resp_path = set_dir / args.run / "responses.jsonl"
    if not manifest_path.exists() or not resp_path.exists():
        raise SystemExit(f"need {manifest_path} and {resp_path}")
    rows = {r["key"]: r for r in (json.loads(line) for line in manifest_path.read_text().splitlines()
                                   if line.strip())}
    resp_lines = [json.loads(l) for l in resp_path.read_text().splitlines() if l.strip()]
    unknown = sorted({r["key"] for r in resp_lines} - set(rows))
    if unknown:
        raise SystemExit(f"responses for unknown manifest keys: {unknown}")
    spec = SET_SPEC[args.set_name]

    calls = [score_call(resp, rows[resp["key"]]) for resp in resp_lines]
    (set_dir / args.run / "scored.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in calls))

    metrics = metrics_for(calls, args.n_boot, args.seed)
    per_hist = {}
    for h in HISTOTYPES:
        hcc = [c for c in calls if c["histotype"] == h]
        if hcc and any(c["counted"] or c["status"] in ("parse_fail", "uncertain") for c in hcc):
            per_hist[h] = metrics_for(hcc, args.n_boot, args.seed)

    comparison = None
    compare_calls = None
    if args.compare_run:
        compare_path = set_dir / args.compare_run / "responses.jsonl"
        if not compare_path.exists():
            raise SystemExit(f"no responses for --compare-run: {compare_path}")
        compare_resp = [json.loads(l) for l in compare_path.read_text().splitlines() if l.strip()]
        compare_calls = [score_call(resp, rows[resp["key"]]) for resp in compare_resp]
        comparison = paired_compare(calls, compare_calls, spec["positive"])
        (set_dir / args.compare_run / "scored.jsonl").write_text(
            "".join(json.dumps(c) + "\n" for c in compare_calls))

    out = {"set": args.set_name, "run": args.run, "positive_class": spec["positive"],
           "metrics": metrics, "per_histotype": per_hist, "n_boot": args.n_boot, "seed": args.seed}
    if comparison:
        out["paired_vs"] = {"run": args.compare_run, **comparison}
    (set_dir / args.run / "score.json").write_text(json.dumps(out, indent=2) + "\n")

    # ---- report -------------------------------------------------------------
    lines = [f"# Phase 5 Study B — decision replay, set `{args.set_name}`, run `{args.run}`", ""]
    lines.append(f"Scored {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}; "
                 f"positive class = **{spec['positive']}**; `uncertain` rows excluded from "
                 f"sens/spec (n_uncertain reported); parse failures are errors.")
    lines.append("")
    lines.append("| n | counted | sens | spec | acc | AUROC | sens @ spec 0.95 | threshold | "
                 "uncertain | parse fail | resp error |")
    lines.append("|---|" + "---|" * 10)
    m = metrics
    lines.append(f"| {m['n']} | {m['n_counted']} | {fmt(m['sensitivity'])} | {fmt(m['specificity'])} | "
                 f"{fmt(m['accuracy'])} | {fmt(m['auroc'])} | {fmt(m.get('sens_at_spec_0.95'))} | "
                 f"{fmt(m.get('threshold_at_spec_0.95'))} | {m['n_uncertain']} | "
                 f"{m['n_parse_fail']} | {m['n_response_error']} |")
    lines.append("")
    lines.append(f"Sensitivity CI95 {m['sens_ci95']}, specificity CI95 {m['spec_ci95']} "
                 f"(case-level bootstrap, {args.n_boot} resamples, seed {args.seed}).")
    if comparison:
        lines.append("")
        lines.append(f"Paired vs `{args.compare_run}`: {comparison['n_paired_keys']} paired keys, "
                     f"McNemar p = {comparison['mcnemar_p']} "
                     f"(b = {comparison['n_b_hit_a_miss']}, c = {comparison['n_a_hit_b_miss']}).")
    lines.append("")
    lines.append("## Per histotype")
    lines.append("")
    lines.append("| h | n | sens | spec | AUROC |")
    lines.append("|---|" + "---|" * 4)
    for h in HISTOTYPES:
        if h not in per_hist:
            continue
        hm = per_hist[h]
        lines.append(f"| {h} | {hm['n']} | {fmt(hm['sensitivity'])} | {fmt(hm['specificity'])} | "
                     f"{fmt(hm['auroc'])} |")
    lines.append("")
    lines.append("## Scoring rules")
    lines.append("- Strict JSON: {\"answer\": ...} among " + ", ".join(spec["answers"]) +
                 "; anything else = parse_fail (counted, never retried).")
    lines.append("- AUROC from `confidence` in the stated answer (p(positive) = confidence for "
                 "positive answers, 1 - confidence otherwise; missing confidence -> 0.5).")
    if args.set_name == "wrong_structure":
        lines.append(f"- `sens @ spec 0.95` uses the smallest threshold with specificity >= "
                     f"{SPECIFICITY_TARGET} ON THIS SET — a design-set quantity to freeze "
                     "before any test-split use.")
    (set_dir / args.run / "report.md").write_text("\n".join(lines) + "\n")

    print(f"[score] {args.set_name}/{args.run}: sens={metrics['sensitivity']} "
          f"spec={metrics['specificity']} auroc={metrics['auroc']} "
          f"uncertain={metrics['n_uncertain']} parse_fail={metrics['n_parse_fail']} "
          f"-> {set_dir / args.run / 'report.md'}")


if __name__ == "__main__":
    main()
