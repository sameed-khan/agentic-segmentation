"""Offline multi-lesion scorer for Phase 4 agent runs (BEAMSTER; reuses eval/v2/score.py).

Usage:
  uv run --project harness/server python eval/v4/score_multi.py results/<run_id> [--dataset beamster] [--cases ...] [--workers 8]
  uv run --project harness/server python eval/v4/score_multi.py results/<run_id>/<case>     # single case

Same run/case directory layout as eval/v2/score.py (trajectory.jsonl, versions.npz,
final_mask.nii.gz, case_meta.json + optional runner meta.json). ``--dataset`` defaults to
the case_meta.json ``dataset`` field (written by segd since Phase 4); the manifest is
eval/v4/<dataset>_cases.csv.

Metrics per case (GT loaded OFFLINE via segd.data.get_case/load_mask, CONTRACTS §0):
  global    Dice / HD95 / NSD@2mm of final_mask.nii.gz vs the full GT mask (union over all
            lesions) — same definitions as eval/v2/score.py (dice_from_counts, surface_metrics;
            non-iso spacing honoured, though Phase 4 inputs are 1 mm iso).
  lesions   GT lesions = connected components of GT (>0, 6-connectivity, the SAME convention
            as eval/v4/phase4_common.lesion_stats and segd.observe.mask_stats); predicted
            components likewise. Detection rule (decision 6): a GT lesion is DETECTED when
            some predicted component overlaps >= 10% of the GT lesion's voxels; an FP is a
            predicted component with ZERO overlap with GT.
  outputs   n_lesions (GT components), n_lesions_manifest (manifest cross-check),
            n_detected, lesion_sensitivity, n_fp, lesion_precision, lesion_f1
            (2PR/(P+R), 0 when no detections and no FPs... defined 0 when TP=0),
            mean_lesion_dice (mean over DETECTED lesions of Dice(matched prediction, lesion),
            where the matched prediction is the union of the >=10%-overlapping predicted
            components — a predicted component bridging two lesions counts against both),
            per-size-bucket breakdown (<0.5 mL / 0.5-2 mL / >2 mL: n, sensitivity, mean dice),
            n_tool_calls, n_model_prompts (case_meta.json).
  tag       same precedence as eval/v2/score.py's tag_case, with the MSTT 'localization' tag
            REPLACED by 'none_detected' (no GT lesion detected): harness > none_detected >
            delineation (global Dice < 0.5) > budget > none.

Outputs (next to the case dirs): scores_multi.csv, lesions_multi.csv (one row per GT lesion),
scores_multi.md, and a printed summary block.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "harness" / "server" / "src"))

import segd.data  # noqa: E402,F401  (offline-only GT access, CONTRACTS §0)
from segd.data import get_case, load_mask  # offline-only GT access (CONTRACTS §0)  # noqa: E402

# Reused single definitions from eval/v2/score.py, loaded by FILE PATH under a unique
# module name (a bare ``from score import ...`` is ambiguous under pytest —
# harness/study/score.py, a DIFFERENT scorer, can own 'score'; and eval/v2/v3_adherence.py
# owns 'eval_v2_score' with its own by-path loader + regression assertion).
import importlib.util                        # noqa: E402

_v2_spec = importlib.util.spec_from_file_location("eval_v2_score_phase4",
                                                  PROJECT_ROOT / "eval" / "v2" / "score.py")
_v2 = importlib.util.module_from_spec(_v2_spec)
sys.modules["eval_v2_score_phase4"] = _v2
_v2_spec.loader.exec_module(_v2)

DICE_LOCALIZE_OK = _v2.DICE_LOCALIZE_OK
discover_case_dirs = _v2.discover_case_dirs
dice_from_counts = _v2.dice_from_counts
read_json = _v2.read_json
read_jsonl = _v2.read_jsonl
surface_metrics = _v2.surface_metrics

DETECT_FRAC = 0.10                      # GT lesion detected when >= this fraction overlapped
BUCKET_NAMES = {"lt05": "<0.5 mL", "b05_2": "0.5-2 mL", "gt2": ">2 mL"}

FIELDS = ["case", "dataset", "split", "status", "dice", "hd95_mm", "nsd_2mm",
          "n_lesions", "n_lesions_manifest", "n_detected", "lesion_sensitivity",
          "n_fp", "lesion_precision", "lesion_f1", "mean_lesion_dice",
          "sens_lt05", "det_lt05", "dice_lt05", "n_lt05", "sens_b05_2", "det_b05_2",
          "dice_b05_2", "n_b05_2", "sens_gt2", "det_gt2", "dice_gt2", "n_gt2",
          "none_detected", "n_tool_calls",
          "n_model_prompts", "n_errors_traj", "n_errors_internal", "forced_finish",
          "wall_time_s", "tag", "error"]


# --------------------------------------------------------------------- lesion metrics
def _components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """6-connected components (>0), the same convention as phase4_common.lesion_stats and
    segd.observe.mask_stats (scipy default cross structure)."""
    return ndimage.label(np.asarray(mask) > 0)


def _overlap_matrix(lab_g: np.ndarray, n_g: int, lab_p: np.ndarray, n_p: int) -> np.ndarray:
    """inter[g, p] = voxels of GT component g inside predicted component p."""
    both = (lab_g > 0) & (lab_p > 0)
    if not both.any():
        return np.zeros((n_g + 1, n_p + 1), dtype=np.int64)
    pair = lab_g[both].astype(np.int64) * (n_p + 1) + lab_p[both]
    counts = np.bincount(pair.ravel(), minlength=(n_g + 1) * (n_p + 1))
    return counts.reshape(n_g + 1, n_p + 1)


def lesion_metrics(gt: np.ndarray, pred: np.ndarray, spacing: tuple) -> dict:
    """Lesion-wise detection metrics (decision 6). ``spacing`` in mm; lesion volumes mL."""
    lab_g, n_g = _components(gt)
    lab_p, n_p = _components(pred)
    inter = _overlap_matrix(lab_g, n_g, lab_p, n_p)
    gt_sizes = np.bincount(lab_g.ravel(), minlength=n_g + 1)[1:].astype(np.float64)
    pred_sizes = np.bincount(lab_p.ravel(), minlength=n_p + 1)[1:].astype(np.float64)
    vox_ml = float(np.prod(spacing)) / 1000.0
    gt_ml = gt_sizes * vox_ml
    pred_ml = pred_sizes * vox_ml

    detected = np.zeros(n_g, dtype=bool)
    les_dice = np.zeros(n_g, dtype=np.float64)
    per_lesion: list[dict] = []
    for g in range(n_g):
        matched = np.nonzero(inter[g + 1, 1:] >= DETECT_FRAC * gt_sizes[g])[0] + 1
        detected[g] = matched.size > 0
        if detected[g]:
            # matched prediction = union of the >=10%-overlapping predicted components;
            # its full Dice against the GT lesion (spill into normal tissue counts against it)
            sel = np.isin(lab_p, matched)
            les_dice[g] = dice_from_counts(int((sel & (lab_g == g + 1)).sum()),
                                           int(sel.sum()), int(gt_sizes[g]))
        per_lesion.append({
            "case": "", "lesion_id": g + 1, "volume_ml": round(float(gt_ml[g]), 4),
            "bucket": bucket_of(float(gt_ml[g])), "detected": bool(detected[g]),
            "lesion_dice": round(float(les_dice[g]), 4) if detected[g] else "",
        })
    tp, fp = int(detected.sum()), int((inter[1:, 1:].sum(axis=0) == 0).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / n_g if n_g else float("nan")   # empty GT: sensitivity is undefined
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    out = {
        "n_lesions": n_g, "n_detected": tp, "n_fp": fp,
        "lesion_sensitivity": recall, "lesion_precision": precision, "lesion_f1": f1,
        "mean_lesion_dice": float(les_dice[detected].mean()) if tp else float("nan"),
        "per_lesion": per_lesion,
    }
    # breakdown by lesion size (decision 6 buckets: <0.5 / 0.5-2 (2.0 inclusive) / >2 mL;
    # the same boundary convention as bucket_of — a 2.000 mL lesion is in b05_2 everywhere)
    for key in ("lt05", "b05_2", "gt2"):
        if key == "lt05":
            in_b = gt_ml < 0.5
        elif key == "b05_2":
            in_b = (gt_ml >= 0.5) & (gt_ml <= 2.0)
        else:
            in_b = gt_ml > 2.0
        n_b, det_b = int(in_b.sum()), int((in_b & detected).sum())
        out[f"n_{key}"] = n_b
        out[f"det_{key}"] = det_b
        out[f"sens_{key}"] = det_b / n_b if n_b else float("nan")
        out[f"dice_{key}"] = float(les_dice[in_b & detected].mean()) if det_b else float("nan")
    return out


def bucket_of(volume_ml: float) -> str:
    """Size-bucket key of one lesion volume (decision 6: <0.5 / 0.5-2 (2.0 inclusive) / >2 mL)."""
    if volume_ml < 0.5:
        return "lt05"
    if volume_ml <= 2.0:
        return "b05_2"
    return "gt2"


def tag_case_multi(*, meta_errors: list, missing_final: bool, ran: bool, forced: bool,
                   forced_reason: str, tp: int, dice: float, internal_errors: int = 0) -> str:
    """Failure tags; precedence harness > none_detected > delineation > budget > none.
    Replaces the MSTT 'localization' tag (single-lesion first-click rule) with
    'none_detected': NO GT lesion was detected in the final mask (decision 6)."""
    if ran and (meta_errors or missing_final or internal_errors
                or (forced and "error" in forced_reason)):
        return "harness"
    if tp == 0:
        return "none_detected"
    if not math.isnan(dice) and dice < DICE_LOCALIZE_OK:
        return "delineation"
    if forced:
        rl = forced_reason.lower()
        if any(k in rl for k in ("wall", "budget", "nudge")):
            return "budget"
    return "none"


# ------------------------------------------------------------------------ per case
def _image_reference(row) -> tuple[tuple, np.ndarray, tuple]:
    """Canonical (shape, affine, spacing) without materialising the data (header only)."""
    im = nib.as_closest_canonical(nib.load(str(row.image_path)))
    return (tuple(int(v) for v in im.shape), np.asarray(im.affine),
            tuple(float(s) for s in im.header.get_zooms()[:3]))


def score_case(case_dir_str: str, dataset_arg: str | None = None) -> dict:
    """Score one case directory; never raises (errors land in status/error)."""
    case_dir = Path(case_dir_str)
    case = case_dir.name
    rec = {k: "" for k in FIELDS}
    rec.update(case=case, status="error")
    flags: list[str] = []
    error = ""

    traj = read_jsonl(case_dir / "trajectory.jsonl")
    case_meta = read_json(case_dir / "case_meta.json") or {}
    meta = read_json(case_dir / "meta.json") or {}
    final_path = case_dir / "final_mask.nii.gz"

    ran = bool(traj) or bool(case_meta)
    missing_final = ran and not final_path.exists()
    if missing_final:
        flags.append("missing_final_mask")
    dataset = dataset_arg or str(case_meta.get("dataset") or "") or None

    gt: np.ndarray | None = None
    spacing = (1.0, 1.0, 1.0)
    try:
        row = get_case(case, dataset)
        rec["dataset"] = dataset or ""
        rec["split"] = row.split
        shape_ref, affine_ref, spacing = _image_reference(row)
        gt = load_mask(row, affine_ref, shape_ref)
    except Exception as e:                                  # noqa: BLE001
        error = f"gt: {e!r}"
        flags.append("missing_gt")

    rec["n_lesions_manifest"] = row.n_lesions if gt is not None else ""
    rec["n_tool_calls"] = case_meta.get("tool_calls", "")
    rec["n_model_prompts"] = case_meta.get("model_prompts", "")
    rec["n_errors_traj"] = len([r for r in traj if r.get("error")])
    rec["n_errors_internal"] = len([r for r in traj if r.get("error_kind") == "internal"])
    forced = bool(meta.get("forced_finish"))
    reason = str(case_meta.get("reason") or meta.get("forced_finish_reason") or
                 (meta.get("finish_reason") or ""))
    rec["forced_finish"] = bool(forced or str(reason).startswith("forced:"))
    wall = meta.get("wall_time_s")
    rec["wall_time_s"] = "" if wall is None else round(float(wall), 2)

    if final_path.exists() and gt is not None:
        try:
            final = np.asanyarray(nib.load(str(final_path)).dataobj) > 0
            inter = int((final & (gt > 0)).sum())
            rec["dice"] = dice_from_counts(inter, int(final.sum()), int((gt > 0).sum()))
            rec["hd95_mm"], rec["nsd_2mm"] = surface_metrics(final, gt, spacing)
            if not final.any():
                flags.append("empty_pred")
            lm = lesion_metrics(gt > 0, final, spacing)
            rec["per_lesion"] = lm.pop("per_lesion")
            rec.update(lm)
            rec["none_detected"] = lm["n_detected"] == 0
            if int((gt > 0).sum()) == 0:
                flags.append("empty_gt")
        except Exception as e:                              # noqa: BLE001
            error = f"{error} score: {e!r}".strip(" ;")
            flags.append("score_error")
    elif gt is not None:
        rec["none_detected"] = True

    rec["flags"] = ";".join(flags)
    if gt is None:
        rec["tag"] = ""                                     # cannot judge without GT
        rec["status"] = "missing_gt"
    else:
        rec["tag"] = tag_case_multi(
            meta_errors=list(meta.get("errors") or []), missing_final=missing_final, ran=ran,
            forced=bool(rec["forced_finish"]), forced_reason=str(reason),
            tp=int(rec.get("n_detected") or 0),
            dice=float(rec["dice"]) if isinstance(rec["dice"], float) else float("nan"),
            internal_errors=rec["n_errors_internal"])
        rec["status"] = "missing_final" if missing_final else ("error" if "score_error" in flags else "ok")
    rec["error"] = error
    return rec


# ------------------------------------------------------------------------ output
def _fmt(x) -> str:
    if x is None or x == "":
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, float):
        if abs(x) >= 1e4:
            return f"{x:.1f}"
        return f"{x:.4f}".rstrip("0").rstrip(".")
    return str(x)


def _md(x) -> str:
    return _fmt(x) or "—"


def write_md(out_dir: Path, recs: list[dict]) -> None:
    ok = [r for r in recs if r["status"] == "ok"]
    lines = [f"# Multi-lesion scores — {out_dir.name}", "",
             f"Cases scored: {len(recs)} (ok {len(ok)}); detection: GT lesion detected when a predicted "
             f"component overlaps >= {DETECT_FRAC:.0%} of its voxels; FP = predicted component with zero GT overlap.",
             "Tags replace the MSTT 'localization' tag with 'none_detected'.", "",
             "| case | split | dice | HD95 mm | NSD@2mm | lesions | det | sens | FP | F1 | mean les dice | tag |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["case"], _md(r["split"]), _md(r["dice"]), _md(r["hd95_mm"]), _md(r["nsd_2mm"]),
            _md(r["n_lesions"]), _md(r["n_detected"]), _md(r["lesion_sensitivity"]),
            _md(r["n_fp"]), _md(r["lesion_f1"]), _md(r["mean_lesion_dice"]), r["tag"] or "—"))
    if ok:
        n_l = sum(int(r["n_lesions"]) for r in ok)
        det = sum(int(r["n_detected"]) for r in ok)
        fp = sum(int(r["n_fp"]) for r in ok)
        dice_vals = [r["dice"] for r in ok if isinstance(r["dice"], float)]
        f1_vals = [r["lesion_f1"] for r in ok
                   if isinstance(r["lesion_f1"], float) and not math.isnan(r["lesion_f1"])]
        # pooled F1 counts false negatives (review P1 #4): 2*det / (2*det + FP + FN),
        # FN = n_l - det — the same definition as the per-case 2PR/(P+R). Zero
        # denominator (no GT lesions at all) renders undefined, never divides by zero.
        denom = 2 * det + fp + (n_l - det)
        sens_s = f"{det / n_l:.3f}" if n_l else "undefined (no GT lesions)"
        f1_s = f"{2 * det / denom:.3f}" if denom else "undefined"
        lines += ["", f"**Pooled**: {n_l} GT lesions, {det} detected (sensitivity {sens_s}), "
                  f"{fp} FPs, lesion F1 {f1_s}",
                  f"mean global Dice {np.mean(dice_vals):.3f}, mean lesion F1 {np.mean(f1_vals):.3f}",
                  "", "## By lesion size", "",
                  "| bucket | lesions | detected | sensitivity | mean lesion dice |", "|---|---|---|---|---|"]
        for key, name in BUCKET_NAMES.items():
            n_b = sum(int(r.get(f"n_{key}") or 0) for r in ok)
            d_b = sum(int(r.get(f"det_{key}") or 0) for r in ok)
            dice_b = [r[f"dice_{key}"] for r in ok
                      if isinstance(r.get(f"dice_{key}"), float) and not math.isnan(r[f"dice_{key}"])]
            sens = f"{d_b / n_b:.3f}" if n_b else "—"
            mean_d = f"{np.mean(dice_b):.3f}" if dice_b else "—"
            lines.append(f"| {name} | {n_b} | {d_b} | {sens} | {mean_d} |")
    lines += ["", "## Notes", ""]
    notes = [f"- {r['case']}: status={r['status']}{', ' + r['error'] if r['error'] else ''}"
             f"{', flags ' + r['flags'] if r['flags'] else ''}" for r in recs if r["status"] != "ok"]
    lines += notes if notes else ["- none."]
    (out_dir / "scores_multi.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Offline multi-lesion scorer (Phase 4).")
    ap.add_argument("run_dir", type=Path,
                    help="results/<run_id>/ (one <case>/ subdir each) or a single case dir")
    ap.add_argument("--dataset", default=None,
                    help="manifest dataset (default: case_meta.json 'dataset')")
    ap.add_argument("--cases", default="", help="comma-separated case ids to score")
    ap.add_argument("--workers", type=int, default=8, help="process pool size (default 8)")
    args = ap.parse_args(argv)

    if not args.run_dir.exists():
        print(f"ERROR: {args.run_dir} does not exist", file=sys.stderr)
        return 1
    only = {c.strip() for c in args.cases.split(",") if c.strip()} or None
    out_dir, case_dirs = discover_case_dirs(args.run_dir, only)
    if not case_dirs:
        print(f"ERROR: no case dirs under {args.run_dir}", file=sys.stderr)
        return 1
    print(f"{len(case_dirs)} case(s) in {out_dir}; dataset={args.dataset or '(from case_meta)'}",
          flush=True)

    t0 = time.time()
    if args.workers <= 1 or len(case_dirs) == 1:
        recs = [score_case(str(p), args.dataset) for p in case_dirs]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(case_dirs))) as ex:
            recs = list(ex.map(score_case, [str(p) for p in case_dirs], [args.dataset] * len(case_dirs)))
    print(f"scored {len(recs)} case(s) in {time.time() - t0:.1f}s", flush=True)

    ordered = sorted(recs, key=lambda r: r["case"])
    with open(out_dir / "scores_multi.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in ordered:
            w.writerow({k: _fmt(r.get(k, "")) for k in FIELDS})
    lesion_rows = [dict(lr, case=r["case"]) for r in ordered for lr in r.get("per_lesion", [])]
    if lesion_rows:
        with open(out_dir / "lesions_multi.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["case", "lesion_id", "volume_ml", "bucket",
                                              "detected", "lesion_dice"])
            w.writeheader()
            w.writerows(lesion_rows)
    write_md(out_dir, ordered)
    print(f"wrote {out_dir / 'scores_multi.csv'}"
          + (f", {out_dir / 'lesions_multi.csv'}" if lesion_rows else "")
          + f", {out_dir / 'scores_multi.md'}", flush=True)
    ok = [r for r in ordered if r["status"] == "ok"]
    if ok:
        det = sum(int(r["n_detected"]) for r in ok)
        n_l = sum(int(r["n_lesions"]) for r in ok)
        sens = f"({det / n_l:.3f})" if n_l else "(undefined: no GT lesions)"
        print(f"summary: {len(ok)} cases, {n_l} GT lesions, {det} detected {sens}, "
              f"FPs {sum(int(r['n_fp']) for r in ok)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
