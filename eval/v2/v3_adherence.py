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
                          first positive pointing op (trajectory order, best first).
  first_positive_ids      mark ids of that first positive point/lasso/scribble/box op
                          (trajectory `resolved`).
  pointer_followed        the first positive click used a proposed mark (any overlap).
  first_click_inside      that first positive click landed inside GT (score.py rule).
  pointer_followed_inside followed AND inside (RQ3: following was right).
medgemma (second reader -> decisions):
  medgemma_checkpoints_seen, review verdicts (last review record),
  candidate verdicts per accepted id + whether accepting was RIGHT:
  accept_dice_delta       Dice(to_version) - Dice(from_version) from versions.npz
                          (positive = the acceptance added GT-covered tissue; the GT-free
                          proxy for "the MedGemma verdict was correct"). Verdicts are
                          matched by candidate id.

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
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as S  # noqa: E402  (same directory; reuses first_click + version unpacking)

FIELDS = ["case", "histotype", "status", "final_version", "n_locate_calls",
          "locate_proposed_marks", "first_positive_ids", "pointer_followed",
          "first_click_inside", "pointer_followed_inside",
          "medgemma_checkpoints_seen", "review_complete", "review_confidence",
          "candidate_verdicts", "accepted_ids", "n_accepts",
          "n_accepts_medgemma_lesion", "accept_dice_delta", "error"]

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
    order), first positive click (op + resolved mark ids), medgemma checkpoint call
    records (parsed verdicts), accepted candidate ids."""
    events: dict = {"locates": [], "first_positive": None, "medgemma": [],
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
        elif (op in S.POINTING_OPS and _positive_polarity(r)
              and events["first_positive"] is None):
            ids = [str(item.get("id")) for item in (r.get("resolved") or [])
                   if isinstance(item, dict) and item.get("id") is not None]
            if ids:
                events["first_positive"] = {"order": order, "op": op, "ids": ids,
                                            "child": r.get("child")}
        for e in (r.get(V3_RECORDS) or []):
            if isinstance(e, dict) and e.get("call") == "checkpoint":
                events["medgemma"].append(e)
        accepted = (r.get("accepted") or {}).get("ids")
        if accepted:
            events["accepts"].append({"ids": [str(i) for i in accepted],
                                      "from": r.get("parent"), "to": r.get("child")})
    return events


def _latest_locate_before(events: dict) -> list[str]:
    """Marks proposed by the last locate BEFORE the first positive click (or the last
    locate overall when there is no click; '' list when there is no locate at all)."""
    fp = events["first_positive"]
    earlier = [l for l in events["locates"]
               if fp is None or l["order"] < fp["order"]]
    return earlier[-1]["marks"] if earlier else []


def pointer_adherence(events: dict, click_inside: bool | None) -> dict:
    """Did the first positive click use a proposed mark, and was that click inside GT?"""
    out = {"locate_proposed_marks": "", "first_positive_ids": "", "pointer_followed": "",
           "first_click_inside": "", "pointer_followed_inside": ""}
    proposed = _latest_locate_before(events)
    fp = events["first_positive"]
    if proposed:
        out["locate_proposed_marks"] = ";".join(proposed)
    if fp is not None:
        out["first_positive_ids"] = ";".join(fp["ids"])
        out["pointer_followed"] = bool(set(fp["ids"]) & set(proposed))
        if click_inside is not None:
            out["first_click_inside"] = bool(click_inside)
            out["pointer_followed_inside"] = bool(out["pointer_followed"]
                                                  and click_inside)
    return out


def candidate_adherence(events: dict, gt: np.ndarray | None, spacing,
                        versions_dice: dict[str, float] | None) -> dict:
    """Per-accepted-candidate: MedGemma verdict (when the checkpoint ran) and whether
    accepting changed Dice for the better (GT-free correctness proxy, from versions.npz)."""
    out = {"review_complete": "", "review_confidence": "", "candidate_verdicts": "",
           "accepted_ids": "", "n_accepts": 0, "n_accepts_medgemma_lesion": "",
           "accept_dice_delta": ""}
    verdicts: dict[str, tuple] = {}
    reviews = [e for e in events["medgemma"] if e.get("checkpoint") == "review"]
    if reviews:
        parsed = reviews[-1].get("parsed") or {}
        out["review_complete"] = str(parsed.get("complete", ""))
        out["review_confidence"] = parsed.get("confidence", "")
    for e in events["medgemma"]:
        if e.get("checkpoint") != "candidates":
            continue
        parsed = e.get("parsed") or {}
        cid = str(parsed.get("candidate_id", ""))
        if cid:
            verdicts[cid] = (str(parsed.get("verdict", "")), parsed.get("confidence", ""))
    if verdicts:
        out["candidate_verdicts"] = ";".join(
            f"{cid}:{v}" for cid, (v, _c) in sorted(verdicts.items()))
    accepted: list[str] = []
    deltas: list[float] = []
    n_lesion_followed = 0
    for a in events["accepts"]:
        for cid in a["ids"]:
            accepted.append(cid)
            v = verdicts.get(cid, ("", ""))[0]
            if v == "lesion":
                n_lesion_followed += 1
            if versions_dice and a["from"] and a["to"]:
                d0 = versions_dice.get(str(a["from"]))
                d1 = versions_dice.get(str(a["to"]))
                if d0 is not None and d1 is not None and not (math.isnan(d0) or math.isnan(d1)):
                    deltas.append(round(d1 - d0, 4))
    if accepted:
        out["accepted_ids"] = ";".join(accepted)
        out["n_accepts"] = len(accepted)
        out["n_accepts_medgemma_lesion"] = n_lesion_followed
        if deltas:
            out["accept_dice_delta"] = round(sum(deltas) / len(deltas), 4)
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
    rec.update(pointer_adherence(events, click_inside))
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
    print(f"# v3 adherence — {out_dir.name}: {len(recs)} case(s)")
    print(f"- seg_locate cases: {len([r for r in recs if r['n_locate_calls'] != ''])} "
          f"| first positive click followed a proposed mark: "
          f"{followed}/{len(n_follow) or 0}"
          + (f" (of those inside GT: {followed_inside}/{followed})" if followed else ""))
    print(f"- MedGemma checkpoints seen: {sorted({r['medgemma_checkpoints_seen'] for r in recs if r['medgemma_checkpoints_seen']})}")
    print(f"- accepts with a MedGemma 'lesion' verdict: "
          f"{sum(int(r['n_accepts_medgemma_lesion'] or 0) for r in recs)}/"
          f"{sum(int(r['n_accepts'] or 0) for r in recs)}")
    print(f"wrote {out_dir / 'v3_adherence.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
