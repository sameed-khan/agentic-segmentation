#!/usr/bin/env python3
"""Phase 5 Study B: build the three decision sets from the main tree's DESIGN runs.

Inputs (READ-ONLY; the experiment keeps running from that tree):
- candidate judgement : the E16 manifest <runs-root>/candidate_replay/manifest.jsonl,
  expanded one row per candidate (per-candidate decisions, like the planned v3
  second-reader checkpoint; labels from the same GT precision > 0.5 rule).
- 'are we done?'      : for every design run (phase1-06-high, phase1-07-xhigh,
  phase1-08-max, phase1-09-xhigh-ex, phase1-10-high-cand) and case, the FIRST
  finish review grid. Label 'not_done' iff a later/best version in the run
  (versions.npz, unpacked with eval/v2/score.py unpack_version) improves Dice by
  >= 0.05, or oracle-accepting the GT-positive candidates at that review does
  (E13/E14 machinery; candidates recomputed deterministically and matched, as in
  prepare_candidate_replay.py).
- wrong-structure     : the state right after the first model prompt (first
  successful version-creating op, eval/v2 convention). Label 'wrong_structure'
  iff Dice of that version < 0.1 (harness_v3_methodology.md Study B).

Outputs under results/phase5/studyB/<set>/: manifest.jsonl (one row per decision;
GT-derived fields carry the labels for OFFLINE scoring and are never sent to the
model — prompts get only image paths and GT-free text) + set_meta.json.

Run:  uv run --project harness/server python harness/study/build_decision_sets.py \
          [--runs-root /home/ubuntu/agentic-segmentation/results] [--out results/phase5/studyB]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "harness" / "server" / "src"))

DEFAULT_RUNS_ROOT = Path("/home/ubuntu/agentic-segmentation/results")
DESIGN_RUNS = ("phase1-06-high", "phase1-07-xhigh", "phase1-08-max",
               "phase1-09-xhigh-ex", "phase1-10-high-cand")
DONE_IMPROVEMENT_MIN = 0.05      # not-done threshold (methodology Study B)
WRONG_STRUCTURE_DICE_MAX = 0.10  # wrong-structure label (methodology Study B)
SETS = ("candidate_judgement", "done_check", "wrong_structure")


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_eval_score() -> dict:
    """eval/v2/score.py under a distinct module name (harness/study/score.py is
    imported as `score` elsewhere); only unpack_version / dice_from_counts used."""
    path = PROJECT_ROOT / "eval" / "v2" / "score.py"
    spec = importlib.util.spec_from_file_location("phase5_eval_score", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ per-case cache
class CaseData:
    """Image + GT + spacing for one case, loaded once and reused across runs."""

    def __init__(self, case: str, img, gt: np.ndarray, spacing: tuple[float, float, float], body):
        self.case = case
        self.img = img
        self.gt = gt
        self.spacing = spacing
        self.body = body


class CaseCache:
    """Bounded cache (<= MAX_CACHED volumes, ~600 MB each) over the 30 design cases."""

    MAX_CACHED = 4

    def __init__(self, runs_root: Path):
        self.runs_root = runs_root
        self._cache: OrderedDict[str, CaseData] = OrderedDict()

    def get(self, case: str) -> CaseData:
        if case in self._cache:
            self._cache.move_to_end(case)
            return self._cache[case]
        from segd.data import get_case, load_image, load_mask
        from segd.marks import body_mask

        row = get_case(case)
        img, aff, sp = load_image(row)
        gt = load_mask(row) > 0
        data = CaseData(case, img, gt, sp, body_mask(img))
        self._cache[case] = data
        while len(self._cache) > self.MAX_CACHED:
            self._cache.popitem(last=False)
        return data

    def gt_only(self, case: str) -> tuple[np.ndarray, np.ndarray, CaseData | None]:
        """GT + volume shape; full CaseData only when available under the cache cap."""
        data = self.get(case)
        return data.gt, tuple(int(v) for v in np.asarray(data.img.shape)), data


# ------------------------------------------------------------------ dice helpers
def dice_of(eval_score, z, vid: int, shape: tuple, gt: np.ndarray) -> float | None:
    """Dice of one version (None when the version is empty)."""
    mask = eval_score.unpack_version(z, int(vid), shape)
    if mask is None:
        return None
    mask = np.asarray(mask) > 0
    inter = int((mask & gt).sum())
    return eval_score.dice_from_counts(inter, int(mask.sum()), int(gt.sum()))


def best_version_dice(eval_score, z, shape: tuple, gt: np.ndarray) -> tuple[str, float]:
    """(version id with the highest Dice, that Dice) over all versions in the npz;
    empty versions count as Dice 0.0; ties -> lowest version id."""
    best_vid, best_dice = None, 0.0
    for vid in (int(v) for v in z["ids"]):
        d = dice_of(eval_score, z, vid, shape, gt)
        d = 0.0 if d is None else d
        if best_vid is None or d > best_dice:
            best_vid, best_dice = vid, d
    return f"v{best_vid}", float(best_dice)


# ------------------------------------------------------------------ candidate oracle
def candidate_oracle_gain(eval_score, data: CaseData, review_mask: np.ndarray,
                          candidates_items: list[dict]) -> tuple[float, list[dict]]:
    """Oracle-accept Dice gain at a review with candidates (E16 machinery).

    Recomputes the deterministic intensity candidates on the reviewed mask version,
    asserts id/volume match with the logged items, labels 'lesion' iff candidate
    precision (volume inside GT) > 0.5, and returns the Dice gain of accepting
    exactly the lesion-labelled ones (candidates are disjoint add-regions outside
    the mask, so the accepted-subset Dice is additive — see prepare_candidate_replay).
    """
    from segd import candidates as C

    cands = C.propose(data.img, review_mask, data.spacing, method="intensity", max_n=5,
                      cache={"body": data.body})
    assert [c.id for c in cands] == [i["id"] for i in candidates_items], \
        f"{data.case}: candidate ids drifted from the logged review"
    inter = sum(int((c.region & data.gt).sum()) for c in cands)
    region = sum(int(c.region.sum()) for c in cands)
    gt_n = int(data.gt.sum())
    mask_n = int(review_mask.sum())
    mask_gt = int((review_mask & data.gt).sum())
    dice_state = eval_score.dice_from_counts(mask_gt, mask_n, gt_n)
    # accepted candidates grow the mask (disjoint add-regions outside it):
    # dice_after = 2*(mask_gt + inter) / (mask_n + region + gt_n)
    dice_oracle = eval_score.dice_from_counts(mask_gt + inter, mask_n + region, gt_n)
    labels = [{"id": c.id, "label": "lesion" if float((c.region & data.gt).sum()) /
               max(int(c.region.sum()), 1) > 0.5 else "normal"} for c in cands]
    return float(dice_oracle - dice_state), labels


# ------------------------------------------------------------------ states
VERSION_CREATING_OPS = frozenset({"point", "lasso", "scribble", "box", "paint", "refine", "edit"})


def first_version_creating_state(traj: list[dict]) -> dict | None:
    """First successful version-creating op (eval/v2 VERSION_CREATING_OPS semantics):
    the state right after the first model prompt."""
    for op in traj:
        if op["op"] in VERSION_CREATING_OPS and op.get("child") and not op.get("error"):
            return op
    return None


def first_finish_review(traj: list[dict]) -> dict | None:
    """First finish op that rendered a review grid (no error)."""
    for op in traj:
        if op["op"] == "finish" and (op.get("renders") or []) and not op.get("error"):
            return op
    return None


# ------------------------------------------------------------------ builders
def image_rows(case_dir: Path, renders: list[str], source_root: Path) -> tuple[list[str], list[str]]:
    rel = [str((case_dir / p).relative_to(source_root)) for p in renders]
    return rel, [str(case_dir / p) for p in renders]


def build_done_and_wrong(runs_root: Path, source_root: Path, cache: CaseCache,
                         eval_score, runs: tuple[str, ...] = DESIGN_RUNS,
                         meta_of=None) -> tuple[list[dict], list[dict]]:
    """Walk design runs -> 'are we done?' rows (first finish review grid) and
    wrong-structure rows (state after the first model prompt).

    `meta_of(case)` provides (histotype, sequence); default segd.data lookup
    (tests inject a synthetic mapping).
    """
    if meta_of is None:
        def meta_of(case: str) -> tuple[str, str]:  # noqa: F811
            from segd.data import get_case

            row = get_case(case)
            return row.histotype, row.sequence
    done_rows: list[dict] = []
    wrong_rows: list[dict] = []
    for run in runs:
        run_dir = runs_root / run
        if not run_dir.is_dir():
            raise SystemExit(f"design run not found: {run_dir}")
        for case_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and len(p.name) == 3):
            case = case_dir.name
            histotype, sequence = meta_of(case)
            traj = [json.loads(l) for l in open(case_dir / "trajectory.jsonl")]
            data = None
            z = None

            # ---- 'are we done?' state: first finish review grid --------------
            review = first_finish_review(traj)
            if review is not None:
                vid = int(str(review["parent"]).lstrip("v"))
                data = cache.get(case)
                z = np.load(case_dir / "versions.npz")
                shape = tuple(int(v) for v in z["shape"])
                dice_state = dice_of(eval_score, z, vid, shape, data.gt)
                best_vid, dice_best = best_version_dice(eval_score, z, shape, data.gt)
                version_gain = dice_best - (0.0 if dice_state is None else dice_state)
                oracle_gain = None
                cands = review.get("candidates")
                items = (cands.get("items") if isinstance(cands, dict) else cands) or []
                if items:
                    review_mask = eval_score.unpack_version(z, vid, shape)
                    if review_mask is not None:
                        oracle_gain, _labels = candidate_oracle_gain(
                            eval_score, data, np.asarray(review_mask) > 0, items)
                improvement = max(version_gain, oracle_gain if oracle_gain is not None else -1.0)
                img_rel, img_abs = image_rows(case_dir, review["renders"], source_root)
                done_rows.append({
                    "key": f"{run}:{case}:v{vid}:done", "set": "done_check", "case": case,
                    "histotype": histotype, "sequence": sequence,
                    "source_run": str((run_dir).relative_to(source_root)), "version": f"v{vid}",
                    "images": img_rel, "images_abs": img_abs,
                    "text": review.get("observation", ""),
                    # GT-derived (offline only, never prompted):
                    "label": "not_done" if improvement >= DONE_IMPROVEMENT_MIN else "done",
                    "dice_state": round(dice_state, 4) if dice_state is not None else 0.0,
                    "best_version": best_vid, "dice_best": round(dice_best, 4),
                    "version_dice_gain": round(version_gain, 4),
                    "oracle_dice_gain": round(oracle_gain, 4) if oracle_gain is not None else None,
                    "n_candidates": len(items),
                })
                data = None  # z is reused by the wrong-structure block below

            # ---- wrong-structure state: after the first model prompt ---------
            state_op = first_version_creating_state(traj)
            if state_op is not None:
                vid = int(str(state_op["child"]).lstrip("v"))
                if z is None:
                    z = np.load(case_dir / "versions.npz")
                shape = tuple(int(v) for v in z["shape"])
                gt, _shape2, _data = cache.gt_only(case)
                dice_state = dice_of(eval_score, z, vid, shape, gt)
                img_rel, img_abs = image_rows(case_dir, state_op.get("renders") or [], source_root)
                wrong_rows.append({
                    "key": f"{run}:{case}:v{vid}:wrong", "set": "wrong_structure", "case": case,
                    "histotype": histotype, "sequence": sequence,
                    "source_run": str((run_dir).relative_to(source_root)), "version": f"v{vid}",
                    "images": img_rel, "images_abs": img_abs,
                    "text": state_op.get("observation", ""),
                    # GT-derived (offline only, never prompted):
                    "label": "wrong_structure" if (dice_state is not None and
                                                   dice_state < WRONG_STRUCTURE_DICE_MAX) else "structures_ok",
                    "dice_state": round(dice_state, 4) if dice_state is not None else 0.0,
                })
    return done_rows, wrong_rows


def build_candidate_judgement(runs_root: Path, source_root: Path) -> list[dict]:
    """Expand the E16 candidate replay manifest into per-candidate decisions."""
    e16_path = runs_root / "candidate_replay" / "manifest.jsonl"
    if not e16_path.exists():
        raise SystemExit(f"E16 manifest not found: {e16_path} (prepare_candidate_replay.py output)")
    rows = []
    for review in (json.loads(l) for l in open(e16_path) if l.strip()):
        for cand in review["candidates"]:
            rows.append({
                "key": f"{review['key']}:{cand['id']}", "set": "candidate_judgement",
                "case": review["case"], "histotype": review["histotype"],
                "sequence": review["sequence"], "version": review["version"],
                "source_run": review["source_run"], "review_key": review["key"],
                "images": list(review["images"]),
                "images_abs": [str(source_root / p) for p in review["images"]],
                "text": review["review_text"],
                "candidate": {"id": cand["id"], "volume_ml": cand["volume_ml"],
                              "overlap_mask_vox": cand["overlap_mask_vox"]},
                # GT-derived (offline only, never prompted):
                "label": cand["label"], "precision": cand["precision"],
                "region_vox": cand["region_vox"], "inter_vox": cand["inter_vox"],
                "dice_reviewed": review["dice_reviewed"],
            })
    return rows


def _case_meta_of():
    """Default (histotype, sequence) lookup for build_done_and_wrong (segd CSV)."""
    def meta_of(case: str) -> tuple[str, str]:
        from segd.data import get_case

        row = get_case(case)
        return row.histotype, row.sequence
    return meta_of


# ------------------------------------------------------------------ main
def write_set(out_root: Path, name: str, rows: list[dict], extra_meta: dict) -> Path:
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "manifest.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    labels: dict[str, int] = {}
    for row in rows:
        labels[row["label"]] = labels.get(row["label"], 0) + 1
    meta = {"set": name, "built_at": utcnow(), "n_rows": len(rows), "labels": labels,
            **extra_meta}
    (out_dir / "set_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[build] {name}: {len(rows)} rows ({labels}) -> {out_dir / 'manifest.jsonl'}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT),
                    help="main tree results root (READ-ONLY; its experiment keeps running)")
    ap.add_argument("--out", default="results/phase5/studyB")
    ap.add_argument("--sets", default=",".join(SETS), help=f"subset of {','.join(SETS)}")
    args = ap.parse_args()

    runs_root = Path(args.runs_root).resolve()
    source_root = runs_root.parent
    out_root = PROJECT_ROOT / args.out
    wanted = [s.strip() for s in args.sets.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in SETS]
    if unknown:
        raise SystemExit(f"unknown sets: {unknown}")

    from segd.data import read_cases

    design_cases = {r.case for r in read_cases("design")}

    if "candidate_judgement" in wanted:
        rows = build_candidate_judgement(runs_root, source_root)
        bad = {r["case"] for r in rows} - design_cases
        if bad:
            raise SystemExit(f"non-design cases in the E16 manifest: {sorted(bad)}")
        write_set(out_root, "candidate_judgement", rows, {
            "source": "results/candidate_replay/manifest.jsonl (E16, run results/phase1-10-high-cand)",
            "label_rule": "candidate precision (volume inside GT) > 0.5 -> lesion",
            "unit": "one decision per candidate",
            "source_root": str(source_root),
        })

    if {"done_check", "wrong_structure"} & set(wanted):
        eval_score = load_eval_score()
        cache = CaseCache(runs_root)
        done_rows, wrong_rows = build_done_and_wrong(
            runs_root, source_root, cache, eval_score, meta_of=_case_meta_of())
        non_design = {r["case"] for r in done_rows + wrong_rows} - design_cases
        if non_design:
            raise SystemExit(f"non-design cases in the design runs: {sorted(non_design)}")
        if "done_check" in wanted:
            write_set(out_root, "done_check", done_rows, {
                "runs": list(DESIGN_RUNS),
                "state": "first finish review grid per case per design run",
                "label_rule": f"not_done iff best-version Dice gain >= {DONE_IMPROVEMENT_MIN} "
                              f"or oracle-accept of GT-positive candidates >= {DONE_IMPROVEMENT_MIN}",
                "source_root": str(source_root),
            })
        if "wrong_structure" in wanted:
            write_set(out_root, "wrong_structure", wrong_rows, {
                "runs": list(DESIGN_RUNS),
                "state": "first successful version-creating op per case per design run "
                         "(state after the first model prompt)",
                "label_rule": f"wrong_structure iff state Dice < {WRONG_STRUCTURE_DICE_MAX}",
                "source_root": str(source_root),
            })


if __name__ == "__main__":
    main()
