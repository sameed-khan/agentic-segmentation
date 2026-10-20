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
Parse failures and response errors are terminal wrong answers: they enter the
sensitivity/specificity/accuracy denominators (fn on a positive row, fp on a
negative) with explicit counts, never retried.
Case-level CLUSTER bootstrap CIs (resamples cases, carrying every decision of a
case together); per-histotype breakdown; paired McNemar between two runs of the
same set (--compare-run) on per-decision correctness (tp + tn, both classes).

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
        "tp": 0, "fp": 0, "tn": 0, "fn": 0, "counted": False, "missing": False,
    }

    def count_as_error() -> None:
        """Parse/transport failures are terminal wrong answers: fn on a positive row,
        fp on a negative row (review P1-4: never dropped from the denominators)."""
        if out["is_positive"]:
            out["fn"] = 1
        else:
            out["fp"] = 1
        out["counted"] = True

    if out["response_error"]:
        out["status"] = "response_error"
        count_as_error()
        return out
    parsed = parse_answer(resp.get("raw_text", resp.get("raw")), spec["answers"])
    out.update({"status": parsed["status"], "answer": parsed["answer"],
                "confidence": parsed["confidence"]})
    if parsed["status"] == "parse_fail":
        count_as_error()
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
        "n_missing": sum(1 for c in calls if c.get("missing")),
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


def cluster_bootstrap_ci(rows: list[dict], value_fn, n_boot: int, seed: int) -> tuple[float, float]:
    """Case-level (cluster) bootstrap percentile CI for a mean over decision rows.

    Resamples CASE ids with replacement and carries EVERY decision row of each selected
    case together (a case's candidates/reps are correlated), recomputing the mean over
    all rows of the resampled cases each draw (review P1-5: never resample decisions).
    """
    by_case: dict[str, list[float]] = {}
    for c in rows:
        by_case.setdefault(c["case"], []).append(float(value_fn(c)))
    cases = sorted(by_case)
    if not cases:
        return (None, None)
    rng = np.random.default_rng(seed)
    values = [np.asarray(by_case[c], dtype=float) for c in cases]
    stats = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(cases), size=len(cases))
        stats.append(float(np.concatenate([values[i] for i in picked]).mean()))
    return (round(float(np.percentile(stats, 2.5)), 4),
            round(float(np.percentile(stats, 97.5)), 4))


def metrics_for(calls: list[dict], n_boot: int, seed: int) -> dict:
    conf = confusion(calls)
    ok_rows = [c for c in calls if c["status"] == "ok"]
    pos_scores = [c["p_positive"] for c in ok_rows if c["is_positive"]]
    neg_scores = [c["p_positive"] for c in ok_rows if not c["is_positive"]]
    conf["auroc"] = auroc(pos_scores, neg_scores)
    conf["auroc_all_rows"] = auroc(  # uncertain rows enter as p=0.5 (rank ties)
        [c["p_positive"] if c["p_positive"] is not None else 0.5
         for c in calls if c["status"] in ("ok", "uncertain") and c["is_positive"]],
        [c["p_positive"] if c["p_positive"] is not None else 0.5
         for c in calls if c["status"] in ("ok", "uncertain") and not c["is_positive"]])
    # case-level (cluster) bootstrap CIs: sensitivity over counted positive rows
    # (parse/transport failures included with tp=0), specificity over counted negatives
    conf["sens_ci95"] = cluster_bootstrap_ci(
        [c for c in calls if c["counted"] and c["is_positive"]], lambda c: c["tp"], n_boot, seed)
    conf["spec_ci95"] = cluster_bootstrap_ci(
        [c for c in calls if c["counted"] and not c["is_positive"]], lambda c: c["tn"], n_boot, seed)
    # secondary completeness metrics: uncertain counted as a negative prediction. The
    # secondary policy only ADDITIONALLY maps the abstentions — it keeps every primary
    # counted row with its assignment, so parse/transport/missing failures stay wrong
    # (FN/FP) and, with zero uncertain answers, the secondary metrics equal the primary
    # ones exactly (review fix round: the old filter dropped failures from the
    # denominators again and could report secondary sensitivity 1.0 with primary 0.1).
    u_pos = [c for c in calls if c["is_positive"]]
    u_neg = [c for c in calls if not c["is_positive"]]

    def predicted_pos(c: dict) -> bool:
        """Negative prediction iff the primary assignment says so: an uncertain abstain
        row has tp/fp/tn/fn all zero, a stated positive answer sets tp or fp, a stated
        negative answer or a failure sets tn or fn."""
        return c["tp"] == 1 or c["fp"] == 1

    conf["sensitivity_uncertain_as_negative"] = (
        round(sum(1 for c in u_pos if predicted_pos(c)) / len(u_pos), 4)
        if u_pos else None)
    conf["specificity_uncertain_as_negative"] = (
        round(sum(1 for c in u_neg if not predicted_pos(c)) / len(u_neg), 4)
        if u_neg else None)
    thr = threshold_at_specificity(neg_scores, SPECIFICITY_TARGET)
    # EXACT operational threshold (review P2-3: a rounded value cannot reproduce the
    # metric; the convention is positive iff p_positive >= threshold)
    conf["threshold_at_spec_0.95"] = thr
    conf["threshold_convention"] = "positive iff p_positive >= threshold (negatives score < threshold)"
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
    """McNemar between two runs of the same set on per-decision CORRECTNESS
    (tp + tn: both classes; parse/transport failures are wrong; uncertain abstains
    excluded), paired per key and majority-aggregated over reps (ties -> not correct).

    Review P1-6: aggregating tp alone hides every negative-class discordance — on a
    negative row both a TN and an FP have tp=0, so an arm that falsely accepts all
    negatives looked identical to one that rejects them. The choice (both-class
    correctness, not a positive-only sensitivity test) is documented in the report.
    """
    def correct_by_key(calls: list[dict]) -> dict[str, list[int]]:
        per_rep: dict[tuple[str, object], int] = {}
        for c in calls:
            if c["counted"]:
                per_rep[(c["key"], c.get("rep"))] = int(c["tp"] + c["tn"])
        per_key: dict[str, list[int]] = {}
        for (key, _rep), correct in per_rep.items():
            per_key.setdefault(key, []).append(correct)
        return per_key

    def majority(flags: list[int]) -> int:
        return int(sum(flags) * 2 > len(flags))

    a = {k: majority(v) for k, v in correct_by_key(calls_a).items()}
    b = {k: majority(v) for k, v in correct_by_key(calls_b).items()}
    keys = sorted(set(a) & set(b))
    n_b = sum(1 for k in keys if b[k] and not a[k])
    n_c = sum(1 for k in keys if a[k] and not b[k])
    if not keys:
        return None
    return {"n_paired_keys": len(keys), "n_b_hit_a_miss": n_b, "n_a_hit_b_miss": n_c,
            "mcnemar_p": round(mcnemar_exact(n_b, n_c), 5),
            "outcome": "per-decision correctness (tp+tn, both classes), majority over reps"}


def synthesize_missing_calls(resp_lines: list[dict], meta_path, run_name: str,
                             rows: dict) -> tuple[list[dict], list[tuple[str, int]], int]:
    """Coverage normalization (review P1-4): every planned key x rep without a recorded
    response becomes an explicit missing-call error row — fn on a positive row, fp on a
    negative row — instead of silently vanishing from the denominators. The SAME policy
    is applied to the main run and to any comparison run BEFORE scoring/pairing, so
    paired counts and p-values cannot depend on the order the runs are passed in (and a
    partially recorded key still pairs through its remaining reps).

    Returns (new_rows, missing, planned) — the synthesized error rows (missing=True),
    the (key, rep) list never recorded, and the planned call count."""
    meta = json.loads(meta_path.read_text()) if Path(meta_path).exists() else {}
    reps_planned = meta.get("reps")
    if reps_planned is None:
        reps_planned = max((r.get("rep") for r in resp_lines), default=-1) + 1
    planned_keys = [k for k in (meta.get("selected_keys") or []) if k in rows]
    if not planned_keys:
        planned_keys = sorted(rows)
    recorded = {(r["key"], r.get("rep")) for r in resp_lines}
    missing = [(key, rep) for key in planned_keys for rep in range(reps_planned)
               if (key, rep) not in recorded]
    new_rows = []
    for m_key, m_rep in missing:
        m_row = rows[m_key]
        m_call = score_call({"run": run_name, "key": m_key, "case": m_row["case"],
                             "rep": m_rep, "model": None, "raw_text": "",
                             "error": "missing response: planned call never recorded",
                             "latency_ms": None, "usage": None}, m_row)
        m_call["missing"] = True
        new_rows.append(m_call)
    return new_rows, missing, len(planned_keys) * reps_planned


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

    # ---- coverage normalization (review P1-4): every planned manifest key × rep must
    # have a response row; missing calls become explicit error rows instead of vanishing.
    missing_calls, missing, planned_calls = synthesize_missing_calls(
        resp_lines, set_dir / args.run / "meta.json", args.run, rows)
    calls.extend(missing_calls)
    recorded = {(r["key"], r.get("rep")) for r in resp_lines}

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
    compare_missing = []
    if args.compare_run:
        compare_path = set_dir / args.compare_run / "responses.jsonl"
        if not compare_path.exists():
            raise SystemExit(f"no responses for --compare-run: {compare_path}")
        compare_resp = [json.loads(l) for l in compare_path.read_text().splitlines() if l.strip()]
        compare_calls = [score_call(resp, rows[resp["key"]]) for resp in compare_resp]
        # the SAME coverage normalization for the comparison run: its missing calls
        # become explicit error rows BEFORE scoring/pairing, so paired counts and
        # p-values cannot depend on the --run/--compare-run argument order (review
        # fix round). Partially recorded keys still pair through their remaining reps.
        compare_missing_calls, compare_missing, _ = synthesize_missing_calls(
            compare_resp, set_dir / args.compare_run / "meta.json", args.compare_run, rows)
        compare_calls.extend(compare_missing_calls)
        comparison = paired_compare(calls, compare_calls, spec["positive"])
        (set_dir / args.compare_run / "scored.jsonl").write_text(
            "".join(json.dumps(c) + "\n" for c in compare_calls))

    out = {"set": args.set_name, "run": args.run, "positive_class": spec["positive"],
           "metrics": metrics, "per_histotype": per_hist, "n_boot": args.n_boot, "seed": args.seed,
           "coverage": {"planned": planned_calls, "recorded": len(recorded),
                        "missing": len(missing),
                        "missing_keys": [f"{k}/r{rep}" for k, rep in missing]}}
    if comparison:
        out["paired_vs"] = {"run": args.compare_run, **comparison,
                            "compare_missing": [f"{k}/r{rep}" for k, rep in compare_missing]}
    (set_dir / args.run / "score.json").write_text(json.dumps(out, indent=2) + "\n")

    # ---- report -------------------------------------------------------------
    lines = [f"# Phase 5 Study B — decision replay, set `{args.set_name}`, run `{args.run}`", ""]
    lines.append(f"Scored {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}; "
                 f"positive class = **{spec['positive']}**; `uncertain` rows excluded from "
                 f"sens/spec (n_uncertain reported; a secondary uncertain-as-negative "
                 f"sens/spec is reported); parse failures and response errors are terminal "
                 f"wrong answers in the denominators (never retried).")
    if missing:
        lines.append(f"")
        lines.append(f"**Coverage: {len(missing)} planned call(s) never recorded "
                     f"({', '.join(f'{k}/r{rep}' for k, rep in missing[:10])}"
                     f"{'…' if len(missing) > 10 else ''}) — counted as errors.**")
    lines.append("")
    lines.append("| n | counted | sens | spec | acc | AUROC | sens @ spec 0.95 | threshold | "
                 "uncertain | parse fail | resp error | missing |")
    lines.append("|---|" + "---|" * 11)
    m = metrics
    lines.append(f"| {m['n']} | {m['n_counted']} | {fmt(m['sensitivity'])} | {fmt(m['specificity'])} | "
                 f"{fmt(m['accuracy'])} | {fmt(m['auroc'])} | {fmt(m.get('sens_at_spec_0.95'))} | "
                 f"{fmt(m.get('threshold_at_spec_0.95'))} | {m['n_uncertain']} | "
                 f"{m['n_parse_fail']} | {m['n_response_error']} | {m['n_missing']} |")
    lines.append("")
    lines.append(f"Secondary (uncertain counted as negative): sens "
                 f"{fmt(m.get('sensitivity_uncertain_as_negative'))}, spec "
                 f"{fmt(m.get('specificity_uncertain_as_negative'))}.")
    lines.append("")
    lines.append(f"Sensitivity CI95 {m['sens_ci95']}, specificity CI95 {m['spec_ci95']} "
                 f"(case-level CLUSTER bootstrap: resamples cases with replacement, carrying "
                 f"every decision of each selected case; {args.n_boot} resamples, seed {args.seed}).")
    lines.append("")
    lines.append("`sens @ spec 0.95` uses the exact operational threshold stored in score.json "
                 "(`threshold_at_spec_0.95`, applied as p_positive >= threshold).")
    if comparison:
        lines.append("")
        lines.append(f"Paired vs `{args.compare_run}`: {comparison['n_paired_keys']} paired keys, "
                     f"McNemar p = {comparison['mcnemar_p']} "
                     f"(b = {comparison['n_b_hit_a_miss']}, c = {comparison['n_a_hit_b_miss']}) "
                     f"on per-decision correctness (tp + tn, BOTH classes; uncertain abstains "
                     "excluded; keys aggregated over reps by majority). Chosen over a "
                     "positive-only test so negative-class discordances count.")
        if compare_missing:
            lines.append(f"  note: {len(compare_missing)} response(s) of `{args.compare_run}` "
                         f"were never recorded ({', '.join(f'{k}/r{rep}' for k, rep in compare_missing[:5])}"
                         f"{'…' if len(compare_missing) > 5 else ''}) — those calls are scored "
                         "as errors under the same missing-as-error policy as the main run; "
                         "keys with remaining recorded reps still pair.")
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
                 "; anything else = parse_fail, counted as a wrong answer in the "
                 "denominators (never retried; missing responses count as errors too).")
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
