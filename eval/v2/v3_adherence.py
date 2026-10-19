"""v3 adherence analysis (RQ3, research/harness_v3_methodology.md §3 Study C): did the
agent FOLLOW the local-model advice, and was following it correct?

Offline helper next to eval/v2/score.py (never reads GT in segd; GT access is the
offline-only `segd.data.load_mask` like score.py). Reads per case: trajectory.jsonl
(the `v3` call records + `resolved` clicks + `accepted` ids), versions.npz (per-version
Dice) and case_meta.json; GT offline via segd.data when available.

Adherence questions
-------------------
pointer (seg_locate proposals -> first positive click):
  locate_proposed_marks   marks proposed by the LAST successful seg_locate before the
                          first positive pointing op (trajectory order, best first);
                          edge-mark proposals ("E1") are view-local, so they are dropped
                          when a successful seg_view happened in between (E1 of a later
                          view is not the E1 that was proposed).
  first_positive_ids      mark ids of that first positive point/lasso/scribble/box op
                          (trajectory `resolved`).
  pointer_followed        the first positive click used a proposed mark (any overlap);
                          undefined ("") when there was no preceding locate advice
                          (unexposed, not an override).
  first_click_inside      that first positive click landed inside GT (score.py rule,
                          all submitted points; the v2 metric, unchanged).
  pointer_followed_inside following was right: the resolved points of the FOLLOWED marks
                          (advice actually used) are inside GT — not the other submitted
                          points, which the agent chose independently.
medgemma (second reader -> decisions):
  medgemma_checkpoints_seen, review verdicts (last review record),
  candidate verdicts per (version, candidate id) — ids restart at C1 for every proposal
  set, so each accept is matched only to advice already delivered for ITS from-version
  and candidate id, BEFORE the accept (advice produced later belongs to a different
  proposal set or could not have caused the acceptance):
  accept_adherence        per accept operation: followed (a matched 'lesion' verdict was
                          accepted) / override (matched advice accepted against) /
                          unexposed (no matching advice), 'v1->v2:...;...'.
  accept_dice_delta       Dice(to_version) - Dice(from_version) from versions.npz, ONE
                          per accept operation (operation-level GT-free correctness
                          proxy for the whole accepted set, labelled as such; NOT
                          per-candidate correctness — one accept op applies its ids
                          jointly). Individual candidate correctness and rejection
                          adherence need separate candidate-level outcomes.

Usage:
  uv run --project harness/server python eval/v2/v3_adherence.py results/<run_id>
Writes ``v3_adherence.csv`` in the run dir and prints a markdown summary.
Rows: one per case (empty columns on v2 cases / missing GT: they degrade to "").

Pure functions (testable without files): collect_v3_events, pointer_adherence,
candidate_adherence; adherence_case glues them to a case directory.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np


def _load_score_module():
    """eval/v2/score.py loaded by explicit path under a unique module name.

    A bare ``import score`` binds whichever module named 'score' is already in
    sys.modules — in a combined pytest run harness/study/score.py owns that name, which
    once broke every adherence test here. The name is unique to this helper."""
    path = Path(__file__).resolve().parent / "score.py"
    spec = importlib.util.spec_from_file_location("eval_v2_score", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_v2_score"] = mod
    spec.loader.exec_module(mod)
    return mod


S = _load_score_module()   # same directory; reuses first_click + version unpacking

FIELDS = ["case", "histotype", "status", "final_version", "n_locate_calls",
          "locate_proposed_marks", "first_positive_ids", "pointer_followed",
          "first_click_inside", "pointer_followed_inside",
          "medgemma_checkpoints_seen", "review_complete", "review_confidence",
          "candidate_verdicts", "accepted_ids", "n_accepts",
          "n_accepts_medgemma_lesion", "n_accepts_overridden", "n_accepts_unexposed",
          "accept_adherence", "accept_dice_delta", "error"]

V3_RECORDS = "v3"          # trajectory key carrying the harness v3 call records


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _positive_polarity(rec: dict) -> bool:
    pol = str((rec.get("args") or {}).get("polarity", "add")).strip().lower()
    return pol in S.POSITIVE_POLARITY


def collect_v3_events(traj: list[dict]) -> dict:
    """Extract the v3-relevant events from one trajectory (record order preserved via
    monotonically increasing ``order`` indices): locate proposals (mark ids in rank
    order) and successful view ops between a locate and a click (edge proposals are
    view-local), first positive click (op + resolved entries), medgemma checkpoint call
    records tagged with the enclosing record's order and version context (candidate ids
    restart at C1 per proposal set), accepted candidate ids."""
    events: dict = {"locates": [], "views": [], "first_positive": None, "medgemma": [],
                    "accepts": []}
    for order, r in enumerate(traj):
        if r.get("error"):
            continue                                   # failed calls are not advice
        op = r.get("op")
        if op == "locate":
            snaps = None
            for e in (r.get(V3_RECORDS) or []):
                if isinstance(e, dict) and e.get("call") == "locate":
                    snaps = e.get("snapped") or []
            marks = [s["mark"] for s in snaps or [] if s.get("mark")]
            events["locates"].append({"order": order, "marks": marks})
        elif op == "view":
            events["views"].append(order)              # active-view changes: edge marks
        elif (op in S.POINTING_OPS and _positive_polarity(r)              # are view-local
              and events["first_positive"] is None):
            ids = [str(item.get("id")) for item in (r.get("resolved") or [])
                   if isinstance(item, dict) and item.get("id") is not None]
            if ids:
                events["first_positive"] = {"order": order, "op": op, "ids": ids,
                                            "resolved": [item for item in (r.get("resolved") or [])
                                                         if isinstance(item, dict)],
                                            "child": r.get("child")}
        for e in (r.get(V3_RECORDS) or []):
            if isinstance(e, dict) and e.get("call") == "checkpoint":
                ev = dict(e)                           # enclosing record's order/version:
                ev["order"] = order                    # the proposal set a candidate
                ev["version"] = r.get("parent")        # verdict belongs to
                events["medgemma"].append(ev)
        accepted = (r.get("accepted") or {}).get("ids")
        if accepted:
            events["accepts"].append({"ids": [str(i) for i in accepted],
                                      "from": r.get("parent"), "to": r.get("child"),
                                      "order": order})
    return events


def _latest_locate_before(events: dict) -> list[str]:
    """Marks proposed by the last locate BEFORE the first positive click (or the last
    locate overall when there is no click; [] when there is no locate at all).

    Edge-mark proposals ("E1") keep their view identity: a successful seg_view between
    the locate and the click may show a DIFFERENT view's edge labels, so those
    proposals are dropped there (global mark ids stay comparable across views)."""
    fp = events["first_positive"]
    earlier = [l for l in events["locates"]
               if fp is None or l["order"] < fp["order"]]
    if not earlier:
        return []
    marks = list(earlier[-1]["marks"])
    if fp is not None:
        locate_order = earlier[-1]["order"]
        if any(locate_order < v < fp["order"] for v in events["views"]):
            marks = [m for m in marks if not str(m).upper().startswith("E")]
    return marks


def followed_click_inside(events: dict, gt: np.ndarray | None,
                          spacing) -> bool | None:
    """Was the FOLLOWED advice right: first_click_inside restricted to the resolved
    points of the marks the click actually followed (None when nothing was followed, no
    usable resolved points, or no GT)."""
    fp = events["first_positive"]
    followed = set(fp["ids"]) & set(_latest_locate_before(events)) if fp else set()
    if not followed or gt is None or not gt.sum():
        return None
    items = [item for item in (fp.get("resolved") or [])
             if str(item.get("id")) in followed and item.get("ijk") is not None]
    cands = S.click_candidates(fp["op"], items)
    if not cands:
        return None
    return any(S.in_gt(c, gt) for c in cands)


def pointer_adherence(events: dict, click_inside: bool | None,
                      followed_click_inside: bool | None = None) -> dict:
    """Did the first positive click use a proposed mark, and was following it right?

    pointer_followed compares the submitted mark ids with the proposals; without
    preceding locate advice it stays undefined ("" — unexposed, not an override, so it
    stays out of the follow/override denominators). Correctness (pointer_followed_inside)
    is evaluated on the resolved points of the FOLLOWED marks (followed_click_inside):
    an independently chosen point that lands inside GT must not make following a bad
    proposal look right."""
    out = {"locate_proposed_marks": "", "first_positive_ids": "", "pointer_followed": "",
           "first_click_inside": "", "pointer_followed_inside": ""}
    proposed = _latest_locate_before(events)
    fp = events["first_positive"]
    if proposed:
        out["locate_proposed_marks"] = ";".join(proposed)
    if fp is not None:
        out["first_positive_ids"] = ";".join(fp["ids"])
        if not proposed:
            out["pointer_followed"] = ""               # no advice exposure: undefined
        else:
            out["pointer_followed"] = bool(set(fp["ids"]) & set(proposed))
        if click_inside is not None:
            out["first_click_inside"] = bool(click_inside)
            if out["pointer_followed"] is False:
                out["pointer_followed_inside"] = False
            elif out["pointer_followed"] is True and followed_click_inside is not None:
                out["pointer_followed_inside"] = bool(followed_click_inside)
    return out


def candidate_adherence(events: dict, gt: np.ndarray | None, spacing,
                        versions_dice: dict[str, float] | None) -> dict:
    """Accepted candidates vs the MedGemma verdicts, and the accept-operation Dice delta.

    Candidate ids restart at C1 for every proposal set, so advice is keyed by the
    (version, candidate id) it was delivered for and each accept is matched only to
    advice already delivered for ITS from-version and candidate id, BEFORE the accept
    (later advice belongs to a different proposal set, or was produced after the
    acceptance and cannot have caused it). Accepts without such advice are UNEXPOSED and
    reported separately from overrides.

    accept_dice_delta is the existing whole-operation Dice delta, reported ONCE per
    accept operation with its adherence status (operation-level GT-free correctness
    proxy: one accept op applies its ids jointly, so it is not per-candidate
    correctness)."""
    out = {"review_complete": "", "review_confidence": "", "candidate_verdicts": "",
           "accepted_ids": "", "n_accepts": 0, "n_accepts_medgemma_lesion": 0,
           "n_accepts_overridden": 0, "n_accepts_unexposed": 0,
           "accept_adherence": "", "accept_dice_delta": ""}
    reviews = [e for e in events["medgemma"] if e.get("checkpoint") == "review"]
    if reviews:
        parsed = reviews[-1].get("parsed") or {}
        out["review_complete"] = str(parsed.get("complete", ""))
        out["review_confidence"] = parsed.get("confidence", "")
    # advice deliveries in record order: what was said about which candidate, for which
    # proposal set (version), and when (the server binds every checkpoint record to the
    # requested candidate id; unusable replies arrive with parsed=None and no verdict)
    advice: list[dict] = []
    for e in events["medgemma"]:
        if e.get("checkpoint") != "candidates":
            continue
        parsed = e.get("parsed") or {}
        cid = str(e.get("candidate_id") or parsed.get("candidate_id") or "")
        verdict = str(parsed.get("verdict", "")).strip().lower()
        if cid and verdict:
            advice.append({"cid": cid, "verdict": verdict,
                           "order": e.get("order"), "version": e.get("version")})
    if advice:
        out["candidate_verdicts"] = ";".join(
            f"{a['version'] or '?'}:{a['cid']}:{a['verdict']}" for a in advice)
    accepted: list[str] = []
    op_labels: list[str] = []
    deltas: list[str] = []
    n_lesion = n_overridden = n_unexposed = 0
    for a in events["accepts"]:
        statuses: list[str] = []
        for cid in a["ids"]:
            accepted.append(cid)
            # advice for this proposal set (from-version) + candidate, before the accept
            matches = [x for x in advice
                       if x["cid"] == cid and x["version"] == a["from"]
                       and (x["order"] is None or a["order"] is None
                            or x["order"] < a["order"])]
            if not matches:
                statuses.append("unexposed")           # distinct from overrides
                n_unexposed += 1
            elif matches[-1]["verdict"] == "lesion":   # last delivery before the accept
                statuses.append("followed")
                n_lesion += 1
            else:
                statuses.append("override")            # accepted against its advice
                n_overridden += 1
        if statuses:
            op_labels.append(f"{a['from']}->{a['to']}:{'-'.join(statuses)}")
            if versions_dice and a["from"] and a["to"]:
                d0 = versions_dice.get(str(a["from"]))
                d1 = versions_dice.get(str(a["to"]))
                if d0 is not None and d1 is not None and not (math.isnan(d0) or math.isnan(d1)):
                    deltas.append(f"{a['from']}->{a['to']}:{round(d1 - d0, 4):+.4f}")
    if accepted:
        out["accepted_ids"] = ";".join(accepted)
        out["n_accepts"] = len(accepted)
        out["n_accepts_medgemma_lesion"] = n_lesion
        out["n_accepts_overridden"] = n_overridden
        out["n_accepts_unexposed"] = n_unexposed
        out["accept_adherence"] = ";".join(op_labels)
        out["accept_dice_delta"] = ";".join(deltas)   # one entry per accept operation
    return out


def versions_dice_map(z, gt: np.ndarray | None) -> dict[str, float]:
    """Per-version Dice of a versions.npz (same rule as score.py; [] when no GT)."""
    out: dict[str, float] = {}
    if z is None or gt is None:
        return out
    for vid, _parent in zip((int(v) for v in z["ids"]), (int(p) for p in z["parents"])):
        crop = S.version_crop(z, vid)
        if crop is None:
            out[f"v{vid}"] = 0.0
            continue
        bbox, c = crop
        inter = int((c & (gt[tuple(slice(lo, hi) for lo, hi in bbox)] > 0)).sum())
        out[f"v{vid}"] = S.dice_from_counts(inter, int(c.sum()), int(gt.sum()))
    return out


def adherence_case(case_dir: Path) -> dict:
    """One case dir -> adherence row (never raises; errors land in status/error)."""
    case = case_dir.name
    rec = {k: "" for k in FIELDS}
    rec.update(case=case, status="ok")
    traj = read_jsonl(case_dir / "trajectory.jsonl")
    case_meta = read_json(case_dir / "case_meta.json") or {}
    rec["final_version"] = case_meta.get("final_version", "")
    gt = None
    spacing = (1.0, 1.0, 1.0)
    try:
        from segd.data import get_case, load_mask
        row = get_case(case)
        ref = S._image_reference(row)
        spacing = ref[2]
        gt = load_mask(row, ref[1], ref[0])
        rec["histotype"] = row.histotype
    except Exception as e:                                  # noqa: BLE001 (GT optional here)
        rec["status"] = "missing_gt"
        rec["error"] = f"gt: {e!r}"[:200]
    events = collect_v3_events(traj)
    rec["n_locate_calls"] = len([r for r in traj if r.get("op") == "locate"
                                 and not r.get("error")])
    seen = []
    for e in events["medgemma"]:
        if e.get("checkpoint") not in seen:
            seen.append(e.get("checkpoint"))
    rec["medgemma_checkpoints_seen"] = ";".join(seen)
    click_inside = None
    if gt is not None and gt.sum():
        click = S.first_click(traj, gt, spacing)
        click_inside = bool(click["first_click_inside"])
    rec.update(pointer_adherence(events, click_inside,
                                 followed_click_inside(events, gt, spacing)))
    z = None
    if (case_dir / "versions.npz").exists():
        try:
            z = np.load(case_dir / "versions.npz")
        except Exception:                                   # noqa: BLE001
            z = None
    rec.update(candidate_adherence(events, gt, spacing, versions_dice_map(z, gt)))
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="v3 adherence analysis (RQ3).")
    ap.add_argument("run_dir", type=Path, help="results/<run_id>/ or a single case dir")
    ap.add_argument("--cases", default="", help="comma-separated case ids")
    args = ap.parse_args(argv)
    if not args.run_dir.exists():
        print(f"ERROR: {args.run_dir} does not exist", file=sys.stderr)
        return 1
    only = {c.strip() for c in args.cases.split(",") if c.strip()} or None
    out_dir, case_dirs = S.discover_case_dirs(args.run_dir, only)
    if not case_dirs:
        print(f"ERROR: no case dirs under {args.run_dir}", file=sys.stderr)
        return 1
    recs = [adherence_case(p) for p in sorted(case_dirs)]
    with open(out_dir / "v3_adherence.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow({k: S._fmt(r[k]) for k in FIELDS})

    n_follow = [r for r in recs if r["pointer_followed"] != ""]
    followed = sum(1 for r in n_follow if r["pointer_followed"])
    inside = [r for r in n_follow if r["first_click_inside"] != ""]
    followed_inside = sum(1 for r in inside if r["pointer_followed_inside"])
    n_lesion = sum(int(r["n_accepts_medgemma_lesion"] or 0) for r in recs)
    n_over = sum(int(r["n_accepts_overridden"] or 0) for r in recs)
    n_unexp = sum(int(r["n_accepts_unexposed"] or 0) for r in recs)
    print(f"# v3 adherence — {out_dir.name}: {len(recs)} case(s)")
    print(f"- seg_locate cases: {len([r for r in recs if r['n_locate_calls'] != ''])} "
          f"| first positive click followed a proposed mark: "
          f"{followed}/{len(n_follow) or 0}"
          + (f" (following was right: {followed_inside}/{followed})" if followed else ""))
    print(f"- MedGemma checkpoints seen: {sorted({r['medgemma_checkpoints_seen'] for r in recs if r['medgemma_checkpoints_seen']})}")
    print(f"- accepted candidate ids: {n_lesion} followed a 'lesion' verdict, "
          f"{n_over} overrode advice, {n_unexp} unexposed (accept-operation Dice deltas "
          f"in accept_dice_delta)")
    print(f"wrote {out_dir / 'v3_adherence.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
