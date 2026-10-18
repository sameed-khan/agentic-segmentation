#!/usr/bin/env python
"""Offline diagnostic: set-of-marks coverage of the GT lesion on the design cases.

Implements harness_design_v2.md §5.4 item 8 and the Phase 1 gate: for each design
case, check whether the marks the agent would actually see cover the GT lesion.

READS GT MASKS via ``segd.data.load_mask`` — offline diagnostic ONLY. Never import
this module from agent-reachable code (marks.py / render.py / case_state.py / ops.py).

Per case it reports
  (a) 3D coverage: any coarse-or-fine blob/region mark voxel inside the GT;
  (b) in-slice coverage on the GT's largest axial cross-section for the EXACT marks
      drawn by ``render_view(axis='axial', crop='full', marks_level='coarse')`` —
      the visible/thinned/capped set, projected onto that slice the way the
      CONTRACTS active-view rule resolves an agent click (in-plane coords of the
      mark, slice = active slice), hit = GT voxel at the clicked position;
  (c) the same at marks_level='fine' (reference);
  (d) number of labels drawn per level.
Small lesions (V55, F15, V57) additionally get a fine-level check on a crop around
the lesion region.

Marks are cached in the same location CaseState uses
(``<results root>/cache/marks/<case>.npz`` = ``results/cache/marks/<case>.npz``,
matching ``CaseState._load_or_compute_marks`` with ``out_dir=results/<run>/<case>``).
A cache entry is reused only if its recorded params match the current module
constants and the volume shape matches.

Usage: uv run python eval/v2/mark_coverage.py [--jobs 8] [--cases A,B] [--no-cache]
Outputs: eval/v2/marks_dev/coverage.csv + coverage.md (and prints the summary).
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
from segd.data import (CASES_CSV, get_case, load_image, load_mask,   # noqa: E402
                       read_cases)

OUT_DIR = PROJECT_ROOT / "eval" / "v2" / "marks_dev"
DEFAULT_CACHE = PROJECT_ROOT / "results" / "cache" / "marks"   # = CaseState's location
SMALL_LESIONS = ("V55", "F15", "V57")
RENDER_CASES = ("F14", "L19", "M33", "N01", "V37", "V55")  # one-two per histotype


# --------------------------------------------------------------------- helpers
def load_or_compute_marks(case_id: str, image: np.ndarray, spacing,
                          cache_dir: Path, use_cache: bool = True,
                          save_cache: bool = True) -> tuple[M.MarkSet, str, dict]:
    """CaseState-compatible mark cache (results/cache/marks/<case>.npz)."""
    path = cache_dir / f"{case_id}.npz"
    if use_cache and path.exists():
        try:
            ms = M.MarkSet.load(str(path))
            if M.cache_matches(ms, image.shape):
                return ms, "cache", {}
        except Exception:
            pass  # stale/corrupt cache: recompute below
    timings: dict = {}
    t0 = time.monotonic()
    ms = M.compute_marks(image, spacing, timings=timings)
    wall = time.monotonic() - t0
    if save_cache:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            ms.save(str(path))
        except Exception:
            pass  # cache write is an optimisation, never a failure
    return ms, "compute", {"wall_s": wall, **{f"s_{k}": round(v, 2) for k, v in timings.items()}}


def largest_axial(gt: np.ndarray) -> tuple[int, int]:
    """(slice_idx, area) of the GT's largest axial cross-section."""
    areas = gt.sum(axis=(0, 1))
    k = int(np.argmax(areas))
    return k, int(areas[k])


def click_hits_and_dist(drawn_ids, ms: M.MarkSet, gt: np.ndarray, k: int
                        ) -> tuple[bool, float]:
    """Any drawn mark's projected click inside GT on slice ``k``; min distance in mm.

    Projection follows the CONTRACTS active-view rule: click voxel =
    (mark i, mark j, k) — the mark's in-plane position on the active slice.
    Distance is to the nearest GT voxel of that axial slice (0 when inside).
    """
    gt2d = gt[:, :, k] > 0
    best = np.inf
    hit = False
    for mid in drawn_ids:
        m = ms.marks[int(mid)]
        i, j = int(m.ijk[0]), int(m.ijk[1])
        if gt2d[i, j]:
            hit = True
            best = 0.0
        else:
            best = min(best, float(_dist_to_mask(gt2d, i, j)))
    return hit, (best if np.isfinite(best) else -1.0)


def _dist_to_mask(mask2d: np.ndarray, r: int, c: int) -> float:
    """Euclidean distance (voxels) from (r, c) to the nearest True voxel."""
    if not mask2d.any():
        return float(np.hypot(*mask2d.shape))
    dist = ndi.distance_transform_edt(~mask2d)
    return float(dist[r, c])


def lesion_crop_box(gt: np.ndarray, k: int, margin: int = 15) -> tuple[int, int, int, int]:
    """Display-oriented in-plane box around the GT on axial slice ``k`` (+margin)."""
    disp = M.extract_slice(gt.astype(np.uint8), "axial", k).astype(bool)
    rr, cc = np.nonzero(disp)
    rows, cols = disp.shape
    r0, r1 = int(rr.min()), int(rr.max()) + 1
    c0, c1 = int(cc.min()), int(cc.max()) + 1
    return (max(0, r0 - margin), max(0, c0 - margin), min(rows, r1 + margin), min(cols, c1 + margin))


# ------------------------------------------------------------------- renders
def write_renders(case_id: str, cache_dir: Path, out_dir: Path) -> list[str]:
    """One PNG per render kind for a case (offline; masks only for the change demo)."""
    row = get_case(case_id)
    img, aff, sp = load_image(row)
    ms, _, _ = load_or_compute_marks(case_id, img, sp, cache_dir)
    gt = load_mask(row, aff, img.shape).astype(bool)      # OFFLINE ONLY (change demo)
    k = int(np.argmax(gt.sum(axis=(0, 1))))
    box = lesion_crop_box(gt, k)
    prev = ndi.binary_erosion(gt, iterations=3).astype(np.uint8)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    kinds = (
        ("overview", lambda: R.render_overview(img, sp, marks=ms)),
        ("view_full_coarse", lambda: R.render_view(img, "axial", k, "full", "outline", "coarse", ms)),
        ("view_crop_fine", lambda: R.render_view(img, "axial", k, box, "outline", "fine", ms,
                                                 mark_boxes=[box])),
        ("edges", lambda: R.render_view(img, "axial", k, "full", "none", "edges", ms)),
        ("sheet", lambda: R.render_sheet(img, "axial", max(0, k - 60), k + 60, 9, markset=ms)),
        ("outline_change", lambda: R.render_outline_change(img, prev, gt.astype(np.uint8), "axial",
                                                           markset=ms)),
        ("change_3plane", lambda: R.render_change_3plane(img, prev, gt.astype(np.uint8),
                                                         markset=ms)),
        ("raw_pointing", lambda: R.render_raw_pointing(img, "axial", k, "full")),
    )
    for name, fn in kinds:
        im, _ = fn()
        p = out_dir / f"{case_id}_{name}.png"
        p.write_bytes(_png_bytes(im))
        made.append(p.name)
    return made


def _png_bytes(im) -> bytes:
    import io
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


# ------------------------------------------------------------------ per case
def evaluate_case(case_id: str, cache_dir: Path, use_cache: bool = True,
                  save_cache: bool = True) -> dict:
    row = get_case(case_id)
    img, aff, sp = load_image(row)
    gt = load_mask(row, aff, img.shape).astype(bool)      # OFFLINE ONLY
    ms, source, perf = load_or_compute_marks(case_id, img, sp, cache_dir, use_cache, save_cache)

    k_max, area = largest_axial(gt)
    cov3d = any(gt[m.ijk] for m in ms.marks.values()) if gt.any() else False

    # EXACT drawn sets for the two agent-facing levels on the full axial slice
    _, ri_c = R.render_view(img, "axial", k_max, "full", "none", "coarse", ms)
    _, ri_f = R.render_view(img, "axial", k_max, "full", "none", "fine", ms)
    ids_c = [mid for mid, _p, _x, _y in ri_c.marks]
    ids_f = [mid for mid, _p, _x, _y in ri_f.marks]
    hit_c, dist_c = click_hits_and_dist(ids_c, ms, gt, k_max)
    hit_f, dist_f = click_hits_and_dist(ids_f, ms, gt, k_max)

    out = {
        "case": case_id, "histotype": row.histotype, "sequence": row.sequence,
        "gt_ml": round(float(gt.sum()) / 1000.0, 2),
        "k_max": k_max, "area_mm2": area,
        "cov3d": int(cov3d),
        "cov_slice_coarse": int(hit_c), "dist_coarse_mm": round(dist_c, 1),
        "cov_slice_fine": int(hit_f), "dist_fine_mm": round(dist_f, 1),
        "n_coarse_drawn": len(ids_c), "n_fine_drawn": len(ids_f),
        "n_marks": len(ms), "n_coarse": sum(1 for m in ms.marks.values() if m.level == "coarse"),
        "n_fine": sum(1 for m in ms.marks.values() if m.level == "fine"),
        "source": source, **perf,
    }

    # small-lesion check: fine marks on a crop around the lesion region
    if case_id in SMALL_LESIONS or out["gt_ml"] <= 4.0:
        box = lesion_crop_box(gt, k_max)
        _, ri_crop = R.render_view(img, "axial", k_max, box, "none", "fine", ms)
        ids_crop = [mid for mid, _p, _x, _y in ri_crop.marks]
        hit_crop, dist_crop = click_hits_and_dist(ids_crop, ms, gt, k_max)
        out["small"] = int(case_id in SMALL_LESIONS)
        out["cov_crop_fine"] = int(hit_crop)
        out["dist_crop_fine_mm"] = round(dist_crop, 1)
        out["n_crop_fine_drawn"] = len(ids_crop)
    return out


# --------------------------------------------------------------------- driver
def run(cases, cache_dir: Path, jobs: int, use_cache: bool,
        save_cache: bool = True) -> list[dict]:
    from concurrent.futures import ProcessPoolExecutor
    rows = []
    if jobs <= 1:
        rows = [evaluate_case(c, cache_dir, use_cache, save_cache) for c in cases]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            rows = list(ex.map(evaluate_case, cases, [cache_dir] * len(cases),
                               [use_cache] * len(cases), [save_cache] * len(cases)))
    rows.sort(key=lambda r: r["case"])
    return rows


def histotype_rates(rows, key):
    out = {}
    for r in rows:
        out.setdefault(r["histotype"], []).append(r[key])
    return {h: f"{sum(v)}/{len(v)}" for h, v in sorted(out.items())}


def rate(rows, key):
    return f"{sum(int(r[key]) for r in rows)}/{len(rows)}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", default=None, help="comma-separated case ids (default: all design)")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--no-cache", action="store_true", help="recompute marks, do not read cache")
    ap.add_argument("--no-save-cache", action="store_true")
    ap.add_argument("--renders", action="store_true", help="write example PNGs for RENDER_CASES")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    if args.renders:
        for cid in RENDER_CASES:
            made = write_renders(cid, cache_dir, OUT_DIR / "renders")
            print(cid, "->", len(made), "renders:", ", ".join(made))
        return 0
    design = [r.case for r in read_cases("design")]
    cases = args.cases.split(",") if args.cases else design
    print(f"{len(cases)} case(s); cache={cache_dir}")

    rows = run(cases, cache_dir, args.jobs, use_cache=not args.no_cache,
               save_cache=not args.no_save_cache)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["case", "histotype", "sequence", "gt_ml", "k_max", "area_mm2", "cov3d",
            "cov_slice_coarse", "dist_coarse_mm", "cov_slice_fine", "dist_fine_mm",
            "n_coarse_drawn", "n_fine_drawn", "n_marks", "n_coarse", "n_fine",
            "source", "wall_s", "s_body_mask", "s_log_blobs", "s_supervoxels", "s_thin",
            "small", "cov_crop_fine", "dist_crop_fine_mm", "n_crop_fine_drawn"]
    with open(OUT_DIR / "coverage.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    lines = [
        "# Mark coverage of GT lesions — design cases",
        "",
        f"Params: " + ", ".join(f"{k}={v}" for k, v in M._params_snapshot().items()),
        f"Cases: {n}; thinning/label cap: visible() cap {M.VISIBLE_CAP}, "
        f"in-plane thin coarse {M.VISIBLE_THIN_MM['coarse']} mm / fine {M.VISIBLE_THIN_MM['fine']} mm",
        "",
        "| metric | result |",
        "|---|---|",
        f"| (a) 3D coverage (any mark voxel inside GT) | {rate(rows, 'cov3d')} = "
        f"{sum(int(r['cov3d']) for r in rows) / n:.2f} |",
        f"| (b) in-slice, drawn coarse set, full axial slice at GT max | {rate(rows, 'cov_slice_coarse')} = "
        f"{sum(int(r['cov_slice_coarse']) for r in rows) / n:.2f} |",
        f"| (c) in-slice, drawn fine set, full axial slice at GT max | {rate(rows, 'cov_slice_fine')} = "
        f"{sum(int(r['cov_slice_fine']) for r in rows) / n:.2f} |",
        f"| (d) labels drawn (coarse): max {max(r['n_coarse_drawn'] for r in rows)}, "
        f"median {int(np.median([r['n_coarse_drawn'] for r in rows]))} |",
        f"| (d) labels drawn (fine): max {max(r['n_fine_drawn'] for r in rows)}, "
        f"median {int(np.median([r['n_fine_drawn'] for r in rows]))} |",
        "",
        "Per histotype (passed/total):",
        "",
    ] + [f"- (a) {h}: {histotype_rates(rows, 'cov3d')[h]}; "
         f"(b) coarse: {histotype_rates(rows, 'cov_slice_coarse')[h]}; "
         f"(c) fine: {histotype_rates(rows, 'cov_slice_fine')[h]}"
         for h in sorted(histotype_rates(rows, "cov3d"))]

    fails = [r for r in rows if not int(r["cov_slice_coarse"])]
    lines += ["", "Failing (b) — no drawn coarse mark inside GT on the largest axial slice:", ""]
    lines += [f"- {r['case']} ({r['histotype']}, {r['gt_ml']} mL): nearest drawn coarse mark "
              f"{r['dist_coarse_mm']} mm from GT in-slice; fine drawn-hit dist {r['dist_fine_mm']} mm"
              for r in fails] or ["- none"]

    small = [r for r in rows if r.get("small")]
    lines += ["", "Small lesions (fine level on a crop around the lesion region):", ""]
    lines += [f"- {r['case']} ({r['gt_ml']} mL): crop fine hit {bool(r.get('cov_crop_fine'))} "
              f"(dist {r.get('dist_crop_fine_mm')} mm, {r.get('n_crop_fine_drawn')} labels); "
              f"coarse full-slice hit {bool(r['cov_slice_coarse'])}"
              for r in small]

    times = [r["wall_s"] for r in rows if r.get("wall_s")]
    if times:
        lines += ["", f"compute_marks wall: median {np.median(times):.1f}s, "
                  f"max {max(times):.1f}s (source: {[r['source'] for r in rows].count('compute')} computed)"]
    (OUT_DIR / "coverage.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
