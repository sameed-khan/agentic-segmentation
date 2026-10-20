#!/usr/bin/env python
"""Offline diagnostic: set-of-marks coverage of EVERY GT lesion (Phase 4 / BEAMSTER).

Adapts eval/v2/mark_coverage.py to the Phase 4 manifests (decision 5): BEAMSTER cases
have MULTIPLE small lesions per case, so the unit of analysis is the single GT lesion
(connected component of the mask, 6-connectivity — the same convention as
eval/v4/phase4_common.lesion_stats and segd.observe.mask_stats), not the whole mask.

READS GT MASKS via ``segd.data.load_mask`` — offline diagnostic ONLY. Never import this
module from agent-reachable code.

For every design case and every GT lesion it reports
  (a) 3D coverage: any coarse-or-fine mark voxel inside the lesion;
  (b) in-slice coverage on the LESION's largest axial cross-section for the EXACT marks
      drawn by ``render_view(axis='axial', crop='full', marks_level='coarse'|'fine')`` —
      the visible/thinned/capped set, projected onto that slice the way the CONTRACTS
      active-view rule resolves an agent click; hit = lesion voxel at the clicked position.
Aggregates: fraction of GT lesions with >= 1 drawn mark, by lesion size
(< 0.5 / 0.5-2 / > 2 mL) and marks level.

Marks are cached in the CaseState location (``<cache-dir>/<case>.npz``; default
``results/cache/marks``); a cache entry is reused only when its params fingerprint and
the volume shape match (segd.marks.cache_matches).

Usage:
  uv run --project harness/server python eval/v4/mark_coverage_beamster.py [--split design] [--jobs 8]
Outputs: eval/v4/marks_beamster/coverage.csv + coverage.md (and prints the summary).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "harness" / "server" / "src"))

from segd import marks as M          # noqa: E402
from segd import render as R         # noqa: E402
from segd.config import MARKS_PROFILE  # noqa: E402
from segd.data import load_image, load_mask, read_cases_phase4   # noqa: E402

OUT_DIR = PROJECT_ROOT / "eval" / "v4" / "marks_beamster"
DEFAULT_CACHE = PROJECT_ROOT / "results" / "cache" / "marks"   # = CaseState's location
BUCKETS = {"lt05": "<0.5", "b05_2": "0.5-2", "gt2": ">2"}


def bucket_of(volume_ml: float) -> str:
    """Size buckets (decision 6): <0.5 / 0.5-2 (2.0 inclusive) / >2 mL — the same
    boundary convention as eval/v4/score_multi.bucket_of."""
    if volume_ml < 0.5:
        return "lt05"
    if volume_ml <= 2.0:
        return "b05_2"
    return "gt2"


def load_or_compute_marks(case_id: str, image: np.ndarray, spacing, cache_dir: Path,
                          profile: str, use_cache: bool = True,
                          save_cache: bool = True) -> tuple[M.MarkSet, str, float]:
    """CaseState-compatible mark cache (<cache>/marks/<case>.npz), profile-fingerprinted."""
    path = cache_dir / f"{case_id}.npz"
    if use_cache and path.exists():
        try:
            ms = M.MarkSet.load(str(path))
            if M.cache_matches(ms, image.shape, profile=profile):
                return ms, "cache", 0.0
        except Exception:
            pass  # stale/corrupt cache: recompute below
    t0 = time.monotonic()
    ms = M.compute_marks(image, spacing, profile=profile)
    wall = time.monotonic() - t0
    if save_cache:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            ms.save(str(path))
        except Exception:
            pass  # cache write is an optimisation, never a failure
    return ms, "compute", wall


def drawn_ids(img: np.ndarray, ms: M.MarkSet, k: int, level: str) -> list[int]:
    """The exact global mark ids drawn on the full axial slice ``k`` at ``level``."""
    _, ri = R.render_view(img, "axial", k, "full", "none", level, ms)
    return [mid for mid, _p, _x, _y in ri.marks]


def hits_lesion(ids: list[int], ms: M.MarkSet, comp2d: np.ndarray) -> bool:
    """Any drawn mark's projected click inside the lesion on its largest axial slice
    (CONTRACTS active-view projection: in-plane position of the mark, slice = active)."""
    for mid in ids:
        m = ms.marks[int(mid)]
        i, j = int(m.ijk[0]), int(m.ijk[1])
        if comp2d[i, j]:
            return True
    return False


def lesion_crop_box(comp: np.ndarray, k: int, margin: int = 15) -> tuple[int, int, int, int]:
    """Display-oriented in-plane box around the lesion on axial slice ``k`` (+margin),
    the same crop 'mask' uses (bbox + 15 mm, CONTRACTS §3)."""
    disp = M.extract_slice(comp.astype(np.uint8), "axial", k).astype(bool)
    rr, cc = np.nonzero(disp)
    rows, cols = disp.shape
    r0, r1 = int(rr.min()), int(rr.max()) + 1
    c0, c1 = int(cc.min()), int(cc.max()) + 1
    return (max(0, r0 - margin), max(0, c0 - margin), min(rows, r1 + margin), min(cols, c1 + margin))


def evaluate_case(row, cache_dir: Path, use_cache: bool, save_cache: bool, profile: str) -> list[dict]:
    case_id = row.case
    img, aff, sp = load_image(row)
    gt = load_mask(row, aff, img.shape).astype(bool)      # OFFLINE ONLY
    ms, source, wall = load_or_compute_marks(case_id, img, sp, cache_dir, profile, use_cache, save_cache)

    lab, n = ndi.label(gt)                    # 6-conn, same as the manifest builder
    sizes = np.bincount(lab.ravel())[1:] if n else np.zeros(0)
    vox_ml = float(np.prod(sp)) / 1000.0
    out = []
    for g in range(n):
        comp = lab == (g + 1)
        k = int(np.argmax(comp.sum(axis=(0, 1))))       # the LESION's largest axial slice
        comp2d = comp[:, :, k]
        ml = float(sizes[g]) * vox_ml
        cov3d = any(comp[m.ijk] for m in ms.marks.values())
        ids_c = drawn_ids(img, ms, k, "coarse")
        ids_f = drawn_ids(img, ms, k, "fine")
        # zoom-crop coverage: fine marks DRAWN in the lesion crop (what the agent sees in
        # seg_view crop='mask'/'marks' zooms) — where the per-panel cap never binds
        box = lesion_crop_box(comp, k)
        _, ri_crop = R.render_view(img, "axial", k, box, "none", "fine", ms)
        ids_crop = [mid for mid, _p, _x, _y in ri_crop.marks]
        out.append({
            "case": case_id, "lesion_id": g + 1, "split": row.split,
            "lesion_ml": round(ml, 4), "bucket": bucket_of(ml), "k_max": k,
            "cov3d": int(cov3d), "hit_coarse": int(hits_lesion(ids_c, ms, comp2d)),
            "hit_fine": int(hits_lesion(ids_f, ms, comp2d)),
            "hit_crop_fine": int(hits_lesion(ids_crop, ms, comp2d)),
            "n_coarse_drawn": len(ids_c), "n_fine_drawn": len(ids_f),
            "n_crop_fine_drawn": len(ids_crop),
            "n_marks": len(ms), "source": source, "wall_s": round(wall, 1),
        })
    return out


def rate(rows: list[dict], key: str) -> str:
    return f"{sum(int(r[key]) for r in rows)}/{len(rows)}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="beamster")
    ap.add_argument("--split", default="design")
    ap.add_argument("--cases", default=None, help="comma-separated case ids (default: whole split)")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--no-cache", action="store_true", help="recompute marks, do not read cache")
    ap.add_argument("--no-save-cache", action="store_true")
    ap.add_argument("--profile", default=None,
                    help="marks density profile (default: config.MARKS_PROFILE for the "
                         "dataset, i.e. what CaseState computes; pass 'default' for the MSTT profile)")
    args = ap.parse_args()
    profile = args.profile or MARKS_PROFILE.get(args.dataset, "default")

    rows_meta = read_cases_phase4(args.dataset, args.split)
    if args.cases:
        want = {c.strip() for c in args.cases.split(",")}
        rows_meta = [r for r in rows_meta if r.case in want]
    print(f"{len(rows_meta)} case(s); dataset={args.dataset} split={args.split}; "
          f"cache={args.cache_dir}", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    per_lesion: list[dict] = []
    if args.jobs <= 1:
        for r in rows_meta:
            per_lesion.extend(evaluate_case(r, Path(args.cache_dir),
                                            not args.no_cache, not args.no_save_cache, profile))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for rows in ex.map(evaluate_case, rows_meta, [Path(args.cache_dir)] * len(rows_meta),
                               [not args.no_cache] * len(rows_meta),
                               [not args.no_save_cache] * len(rows_meta),
                               [profile] * len(rows_meta)):
                per_lesion.extend(rows)
    per_lesion.sort(key=lambda r: (r["case"], r["lesion_id"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["case", "lesion_id", "split", "lesion_ml", "bucket", "k_max", "cov3d",
            "hit_coarse", "hit_fine", "hit_crop_fine", "n_coarse_drawn", "n_fine_drawn",
            "n_crop_fine_drawn", "n_marks", "source", "wall_s"]
    with open(OUT_DIR / "coverage.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(per_lesion)

    n = len(per_lesion)
    lines = [
        f"# Mark coverage of GT lesions — {args.dataset} {args.split} split (per lesion, profile={profile})",
        "",
        f"Params: " + ", ".join(f"{k}={v}" for k, v in M._params_snapshot(profile).items()),
        f"Units: {n} GT lesions on {len(rows_meta)} cases; hit = a DRAWN mark (CONTRACTS "
        "projection) lands inside the lesion on its largest axial slice; full = full axial "
        "view, crop = fine marks drawn in the lesion crop (seg_view zoom).",
        "",
        "| lesion size | n | 3D any mark | full coarse | full fine | crop fine |",
        "|---|---|---|---|---|---|",
    ]
    for key, name in BUCKETS.items():
        rows = [r for r in per_lesion if r["bucket"] == key]
        if not rows:
            lines.append(f"| {name} mL | 0 | — | — | — | — |")
            continue
        lines.append(f"| {name} mL | {len(rows)} | {rate(rows, 'cov3d')} "
                     f"= {sum(int(r['cov3d']) for r in rows) / len(rows):.2f} | "
                     f"{rate(rows, 'hit_coarse')} = {sum(int(r['hit_coarse']) for r in rows) / len(rows):.2f} | "
                     f"{rate(rows, 'hit_fine')} = {sum(int(r['hit_fine']) for r in rows) / len(rows):.2f} | "
                     f"{rate(rows, 'hit_crop_fine')} = {sum(int(r['hit_crop_fine']) for r in rows) / len(rows):.2f} |")
    lines.append(f"| **all** | {n} | {rate(per_lesion, 'cov3d')} | "
                 f"{rate(per_lesion, 'hit_coarse')} | {rate(per_lesion, 'hit_fine')} | "
                 f"{rate(per_lesion, 'hit_crop_fine')} |")

    fails = [r for r in per_lesion if not r["hit_fine"] and not r["hit_crop_fine"]]
    lines += ["", "Lesions with NO drawn mark even in the fine crop view:", ""]
    lines += [f"- {r['case']} L{r['lesion_id']} ({r['lesion_ml']} mL, slice {r['k_max']})"
              for r in fails] or ["- none"]
    (OUT_DIR / "coverage.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
