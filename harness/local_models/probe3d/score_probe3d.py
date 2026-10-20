#!/usr/bin/env python3
"""probe3d scorer: offline GT scoring for the 3D lesion-localizer probes.

Scores the binary masks written by the probe runners (results/phase5/probe3d/
<model>-<prompt>/preds/<case>.nii.gz, case's canonical 1 mm grid) against the
MSTT-199 GT masks resolved through segd.data.load_mask (OFFLINE ONLY - the runner
never sees GT). k_star per case comes from the Phase 2 prepare manifest
(results/phase2_dev/prepare/manifest.jsonl; spacing 1 mm isotropic, arrays are
[i, j, k], k = axial).

Metrics per case (declared):
  - 3D hit: the deepest point of the LARGEST predicted connected component
    (argmax over the component of the GT EDT-depth map D = edt(gt, 1 mm); ties ->
    first index in row-major order) lies inside GT. D > 0 iff the point is in GT.
  - k*-slice hit: the same rule restricted to axial slice k* using the FULL
    predicted mask on that slice (empty on k* = miss).
  - Dice (3D) of the full prediction vs GT, plus largest-component Dice.
  - empty-prediction flag (no predicted voxels at all).
  - distance from the 3D point to GT in mm (0 if inside; else EDT of ~gt at the
    point; NaN for empty predictions - excluded from the mean).

Per-arm outputs (written into the arm dir): cases.csv (per-case metrics +
runtime/VRAM/existence carried over from runs.jsonl), score.json, report.md.

Run (harness/server env; SEGD_DATA_ROOT points at the read-only data root):
  SEGD_DATA_ROOT=/home/ubuntu/agentic-segmentation/data \
    uv run --project harness/server python harness/local_models/probe3d/score_probe3d.py \
    [--arm-dir results/phase5/probe3d/biomedparse-tumor ... | --all]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

PROBE3D_ROOT = Path(__file__).resolve().parent
WORKTREE = PROBE3D_ROOT.parents[2]   # .../pi-worktree-*/ (probe3d -> local_models -> harness -> worktree)
RESULTS_ROOT = WORKTREE / "results" / "phase5" / "probe3d"

sys.path.insert(0, str(WORKTREE / "harness" / "server" / "src"))
import segd.data as segd_data  # noqa: E402  (mask resolution = harness single source of truth)

DEFAULT_MANIFEST = Path("/home/ubuntu/agentic-segmentation/results/phase2_dev/prepare/manifest.jsonl")


def load_manifest(manifest_path: Path) -> dict[str, dict]:
    rows = {}
    with open(manifest_path) as f:
        for line in f:
            r = json.loads(line)
            rows[r["case"]] = r
    return rows


def score_case(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float],
               k_star: int, case_key: str = "", _cache: dict | None = None) -> dict:
    """3D hit / k*-slice hit / Dice / empty / distance, per the module docstring.
    _cache: optional {case_key: (depth, dist_to_gt)} shared across arms (the GT EDT
    maps are identical for every arm)."""
    out = {
        "n_pred_voxels": int(pred.sum()),
        "n_gt_voxels": int(gt.sum()),
    }
    empty = out["n_pred_voxels"] == 0
    out["empty"] = empty

    key = case_key
    if _cache is not None and key in _cache:
        depth, dist_to_gt = _cache[key]
    else:
        depth = ndi.distance_transform_edt(gt, sampling=spacing)          # 0 outside GT
        dist_to_gt = ndi.distance_transform_edt(gt == 0, sampling=spacing)
        if _cache is not None:
            _cache[key] = (depth, dist_to_gt)

    if empty:
        out.update({
            "hit_3d": False, "point_3d": None, "depth_at_point_mm": None,
            "hit_kstar": False, "empty_on_kstar": True, "point_kstar": None,
            "dice": 0.0, "dice_largest_cc": 0.0, "distance_mm": None,
            "distance_min_mm": None,
            "n_components": 0, "largest_cc_voxels": 0,
        })
        return out

    labels, n_comp = ndi.label(pred)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = int(sizes.argmax())
    lcc = labels == largest
    out["n_components"] = int(n_comp)
    out["largest_cc_voxels"] = int(lcc.sum())

    # (a) 3D hit: deepest point of the largest component
    depth_masked = np.where(lcc, depth, -1.0)
    v_star = np.unravel_index(int(depth_masked.argmax()), depth_masked.shape)
    out["point_3d"] = [int(v) for v in v_star]
    out["depth_at_point_mm"] = float(depth[v_star])
    out["hit_3d"] = bool(depth[v_star] > 0)
    out["distance_mm"] = float(dist_to_gt[v_star])
    # secondary: closest predicted voxel to GT (stable miss distance; the argmax
    # tie-break among zero-depth voxels is row-major-arbitrary)
    out["distance_min_mm"] = float(dist_to_gt[pred > 0].min())

    # (b) k*-slice hit: deepest point of the (full) predicted mask on slice k*
    gt2d = gt[:, :, k_star] > 0
    pred2d = pred[:, :, k_star] > 0
    out["empty_on_kstar"] = bool(not pred2d.any())
    if pred2d.any():
        depth2d = ndi.distance_transform_edt(gt2d, sampling=(spacing[0], spacing[1]))
        masked2d = np.where(pred2d, depth2d, -1.0)
        v2 = np.unravel_index(int(masked2d.argmax()), masked2d.shape)
        out["point_kstar"] = [int(v) for v in v2]
        out["hit_kstar"] = bool(depth2d[v2] > 0)
    else:
        out["hit_kstar"] = False
        out["point_kstar"] = None

    # (c) Dice
    inter = int(((pred > 0) & (gt > 0)).sum())
    denom = out["n_pred_voxels"] + out["n_gt_voxels"]
    out["dice"] = 2.0 * inter / denom if denom else 0.0
    inter_lcc = int((lcc & (gt > 0)).sum())
    out["dice_largest_cc"] = 2.0 * inter_lcc / (out["largest_cc_voxels"] + out["n_gt_voxels"])
    return out


def score_arm(arm: Path, manifest: dict[str, dict], row_of: dict[str, dict],
              _edt_cache: dict | None = None) -> dict:
    runs = {}
    runs_path = arm / "runs.jsonl"
    if runs_path.exists():
        with open(runs_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    runs[r["case"]] = r   # last write wins

    pred_files = sorted((arm / "preds").glob("*.nii.gz"))
    cases = sorted({p.stem.replace(".nii", "") for p in pred_files} |
                   {c for c in runs if c in manifest})
    per_case = []
    for case in cases:
        man = manifest[case]
        gt = segd_data.load_mask(row_of[case]).astype(bool)
        pred_path = arm / "preds" / f"{case}.nii.gz"
        if not pred_path.exists():
            continue
        pred = np.asarray(nib.load(str(pred_path)).dataobj).astype(np.uint8)
        m = score_case(pred, gt, tuple(float(s) for s in man["spacing"]), int(man["k_star"]),
                       case_key=case, _cache=_edt_cache)
        run = runs.get(case, {})
        m.update({
            "case": case,
            "histotype": run.get("histotype", row_of[case].histotype),
            "sequence": run.get("sequence", ""),
            "prompt": run.get("prompt", arm.name),
            "existence_prob_max": run.get("existence_prob_max"),
            "existence_prob_mean": run.get("existence_prob_mean"),
            "runtime_s": run.get("case_wall_s"),
            "peak_vram_gib": run.get("peak_vram_gib"),
        })
        per_case.append(m)

    n = len(per_case)
    nonempty = [m for m in per_case if not m["empty"]]
    dists = [m["distance_mm"] for m in nonempty if m["distance_mm"] is not None]
    summary = {
        "arm": arm.name,
        "n_cases": n,
        "hit_3d": sum(m["hit_3d"] for m in per_case) / max(n, 1),
        "hit_kstar": sum(m["hit_kstar"] for m in per_case) / max(n, 1),
        "mean_dice": sum(m["dice"] for m in per_case) / max(n, 1),
        "mean_dice_nonempty": (sum(m["dice"] for m in nonempty) / len(nonempty)) if nonempty else 0.0,
        "empty_rate": sum(m["empty"] for m in per_case) / max(n, 1),
        "mean_distance_mm": float(np.mean(dists)) if dists else None,
        "median_distance_mm": float(np.median(dists)) if dists else None,
        "mean_distance_min_mm": (lambda v: float(np.mean(v)) if v else None)(
            [m["distance_min_mm"] for m in nonempty if m["distance_min_mm"] is not None]),
        "mean_runtime_s": (lambda v: float(np.mean(v)) if v else None)([m["runtime_s"] for m in per_case if m["runtime_s"] is not None]),
        "peak_vram_gib": max((m["peak_vram_gib"] for m in per_case if m["peak_vram_gib"] is not None), default=None),
        "mean_existence_prob_max": (lambda v: float(np.mean(v)) if v else None)(
            [m["existence_prob_max"] for m in per_case if m["existence_prob_max"] is not None]),
    }
    return {"summary": summary, "per_case": per_case}


def write_outputs(arm: Path, result: dict) -> None:
    per_case = result["per_case"]
    summary = result["summary"]
    cols = ["case", "histotype", "sequence", "prompt", "empty", "hit_3d", "hit_kstar",
            "dice", "dice_largest_cc", "distance_mm", "distance_min_mm", "depth_at_point_mm",
            "point_3d", "point_kstar", "empty_on_kstar", "n_components",
            "largest_cc_voxels", "n_pred_voxels", "n_gt_voxels",
            "existence_prob_max", "existence_prob_mean", "runtime_s", "peak_vram_gib"]
    with open(arm / "cases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for m in per_case:
            w.writerow({c: (json.dumps(m[c]) if isinstance(m[c], list) else m[c]) for c in cols})
    with open(arm / "score.json", "w") as f:
        json.dump({"summary": summary,
                   "scored_at": datetime.now(timezone.utc).isoformat(),
                   "scorer": "harness/local_models/probe3d/score_probe3d.py",
                   "gt_loader": "segd.data.load_mask",
                   "per_case": per_case}, f, indent=2)

    s = summary
    lines = [
        f"# probe3d — {s['arm']}",
        "",
        f"Scored {s['n_cases']} design cases. GT via `segd.data.load_mask`; k* from the "
        "phase2_dev prepare manifest. Hit rules: 3D = deepest point (GT-EDT argmax) of the "
        "largest predicted component inside GT; k*-slice = same rule on slice k* over the "
        "full predicted mask (empty on k* = miss).",
        "",
        "| metric | value |",
        "|---|---|",
        f"| 3D hit rate | {s['hit_3d']:.3f} |",
        f"| k*-slice hit rate | {s['hit_kstar']:.3f} |",
        f"| mean Dice | {s['mean_dice']:.3f} |",
        f"| mean Dice (nonempty preds) | {s['mean_dice_nonempty']:.3f} |",
        f"| empty-prediction rate | {s['empty_rate']:.3f} |",
        (f"| mean distance to GT (mm) | {s['mean_distance_mm']:.1f} |" if s["mean_distance_mm"] is not None else "| mean distance to GT (mm) | n/a (all empty) |"),
        (f"| mean runtime (s/case) | {s['mean_runtime_s']:.1f} |" if s["mean_runtime_s"] is not None else ""),
        (f"| peak VRAM (GiB, torch allocated) | {s['peak_vram_gib']:.2f} |" if s["peak_vram_gib"] is not None else ""),
    ]
    lines += ["", "## Per-case", "",
              "| case | histotype | seq | k*-hit | 3D-hit | Dice | dist mm | empty | runtime s |",
              "|---|---|---|---|---|---|---|---|---|"]
    for m in per_case:
        dist_s = "-" if m["distance_mm"] is None else f"{m['distance_mm']:.1f}"
        rt_s = "-" if m["runtime_s"] is None else f"{m['runtime_s']:.1f}"
        lines.append(
            f"| {m['case']} | {m['histotype']} | {m['sequence']} | "
            f"{'Y' if m['hit_kstar'] else 'n'} | {'Y' if m['hit_3d'] else 'n'} | "
            f"{m['dice']:.3f} | {dist_s} | {'Y' if m['empty'] else 'n'} | {rt_s} |")
    (arm / "report.md").write_text("\n".join(l for l in lines if l != "") + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-dir", action="append", default=[])
    ap.add_argument("--all", action="store_true", help="score every arm under results/phase5/probe3d/")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = ap.parse_args()

    arms = [Path(a) for a in args.arm_dir]
    if args.all:
        arms += sorted(p for p in RESULTS_ROOT.iterdir() if p.is_dir() and (p / "preds").exists())

    manifest = load_manifest(Path(args.manifest))
    # segd row objects for load_mask (reads only mask paths + data root)
    row_of = {r.case: r for r in segd_data.read_cases()}

    edt_cache: dict = {}
    for arm in arms:
        res = score_arm(arm, manifest, row_of, _edt_cache=edt_cache)
        write_outputs(arm, res)
        s = res["summary"]
        print(f"{s['arm']}: n={s['n_cases']} hit3d={s['hit_3d']:.3f} hitK*={s['hit_kstar']:.3f} "
              f"dice={s['mean_dice']:.3f} empty={s['empty_rate']:.3f} "
              f"dist={s['mean_distance_mm'] if s['mean_distance_mm'] is None else round(s['mean_distance_mm'], 1)}")


if __name__ == "__main__":
    main()
