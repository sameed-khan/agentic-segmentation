#!/usr/bin/env python3
"""Phase 2 localization study, stage 1: prepare the 30 design cases (design §3 Phase 2,
revision 2026-10-18: Phase 2 runs on the 30 DESIGN cases).

For every design case (segd.data.read_cases('design')):
- k*        : the axial slice with the largest GT cross-section (ties -> lower k).
- marks     : the same MarkSet the agent sees, from the CaseState cache
              (<project>/results/cache/marks/<case>.npz; see CaseState._load_or_compute_marks
              in harness/server/src/segd/case_state.py), recomputed when missing or stale.
- image_S   : render_view(image, 'axial', k*, crop='full', overlay='none', marks_level='coarse',
              markset) - clean + marked side by side, exactly the agent's first full axial view.
- image_R   : render_raw_pointing(image, 'axial', k*, crop='full') with pixel rulers
              (markset passed only so R and S share the identical intensity window).
- image_ex  : per case, the S-ex example - three orthogonal panels (donor axial k*,
              coronal/sagittal through the donor GT centroid) with the GT outlined,
              no marks. Donor = leave-one-out design case of the same histotype and
              sequence, chosen deterministically as the candidate whose GT area on its
              own k* is closest to that candidate group's median (ties -> lower case id).
              L33 (the only T2 lipomatous design case) draws its donor from the T1
              lipomatous design cases; that is logged in `example_note`.

Outputs under results/phase2_dev/prepare/:
- images/<case>_S.png, <case>_R.png, <case>_ex.png
- manifest.jsonl  (one line per case; carries ground-truth-derived fields for OFFLINE
  scoring only - this file never reaches the model)

Run:  uv run --project harness/server python harness/study/prepare.py [--cases F14,M07]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "harness" / "server" / "src"))

from segd import data as segd_data  # noqa: E402
from segd import marks as segd_marks  # noqa: E402
from segd import render as segd_render  # noqa: E402

# CaseState caches marks at <out_dir>.parent.parent / "cache" / "marks" / <case>.npz with
# out_dir = results/<run_id>/<case>, i.e. exactly this path (case_state.py:174-177).
MARKS_CACHE_DIR = PROJECT_ROOT / "results" / "cache" / "marks"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "phase2_dev" / "prepare"

EXAMPLE_PANEL_LONG_PX = 512  # three panels + 2 gaps fit the ~1568 px image budget


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel_path(path: str | Path) -> str:
    return str(Path(path).relative_to(PROJECT_ROOT))


def to_jsonable(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------- pass 1: k* per case
def best_axial_slice(mask: np.ndarray, spacing: tuple[float, float, float]) -> tuple[int, float]:
    """(k*, GT area in mm2): axial slice with the largest GT area, ties -> lower k."""
    counts = np.asarray(mask[:, :, :].sum(axis=(0, 1)), dtype=np.float64)
    inplane_mm = float(spacing[0]) * float(spacing[1])
    areas_mm2 = counts * inplane_mm
    k_star = int(np.argmax(areas_mm2))  # argmax returns the FIRST maximum -> lower k
    return k_star, float(areas_mm2[k_star])


# ---------------------------------------------------------------- marks cache
def load_or_compute_marks(case_id: str, image: np.ndarray,
                          spacing: tuple[float, float, float]) -> tuple[object, str]:
    """Same cache/function as CaseState._load_or_compute_marks (CONTRACTS §4), plus a
    params check: a cache written by an older marks.py is recomputed and overwritten so
    this study and the Phase 1 agent always see the current frozen parameters."""
    import segd.marks as M

    cache_path = MARKS_CACHE_DIR / f"{case_id}.npz"
    current_params = M._params_snapshot()
    if cache_path.exists():
        try:
            ms = M.MarkSet.load(str(cache_path))
            if ms.shape == tuple(image.shape) and ms.params == current_params:
                return ms, "cache"
            stale = "stale params" if ms.params != current_params else "stale shape"
        except Exception as exc:  # unreadable/incompatible cache: recompute
            stale = f"unreadable cache ({exc})"
    else:
        stale = "no cache"
    ms = M.compute_marks(image, spacing)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ms.save(str(cache_path))
    return ms, f"computed ({stale})"


# ---------------------------------------------------------------- example donor
def choose_donor(case: str, rows_by_case: dict, areas: dict[str, float]) -> tuple[str | None, str]:
    """Deterministic leave-one-out donor: same histotype (+same sequence when possible),
    GT area on its own k* closest to the candidate group's median, ties -> lower case id."""
    row = rows_by_case[case]
    same_histo = [r.case for r in rows_by_case.values()
                  if r.histotype == row.histotype and r.case != case]
    same_seq = [c for c in same_histo if rows_by_case[c].sequence == row.sequence]
    note = ""
    if same_seq:
        candidates = sorted(same_seq)
    else:
        # e.g. L33: only T2 lipomatous design case -> fall back to the other sequence
        candidates = sorted(same_histo)
        note = (f"{case} is the only {row.sequence} {row.histotype} design case; "
                f"example donor drawn from the {row.histotype} design cases of the other sequence")
    if not candidates:
        return None, "no candidate donor exists"
    vals = [areas[c] for c in candidates]
    med = float(np.median(vals))
    donor = min(candidates, key=lambda c: (abs(areas[c] - med), c))
    return donor, note


def gt_centroid_slices(mask: np.ndarray) -> tuple[int, int]:
    """(coronal slice j, sagittal slice i) through the GT centroid (floor(+0.5), as
    segd.render.render_overview rounds the mask centroid)."""
    idx = np.argwhere(np.asarray(mask) > 0)
    if idx.size == 0:
        return mask.shape[1] // 2, mask.shape[0] // 2
    c = np.clip(np.floor(idx.mean(axis=0) + 0.5), [0, 0, 0], np.asarray(mask.shape) - 1)
    return int(c[1]), int(c[0])


# ---------------------------------------------------------------- example image
def outlined_panel(image, mask, axis: str, slice_idx: int, spacing):
    """Single GT-outlined clean-of-marks panel: render_view's second panel
    (overlay='outline', mask, marks_level='none'), retitled to drop the misleading
    "| marks" suffix (this panel carries an outline, not marks), then resized."""
    img, ri = segd_render.render_view(image, axis, int(slice_idx), crop="full", overlay="outline",
                                      marks_level="none", mask=mask, spacing=spacing)
    geom = ri.views[1]
    gap = segd_render.PANEL_GAP_PX
    panel = img.crop((geom.panel_w + gap, 0, geom.panel_w + gap + geom.panel_w, geom.panel_h))
    panel = _retitle(panel, f"{axis} {int(slice_idx)}")
    scale = EXAMPLE_PANEL_LONG_PX / max(panel.width, panel.height)
    panel = panel.resize((max(1, round(panel.width * scale)), max(1, round(panel.height * scale))),
                         Image.Resampling.LANCZOS)
    return panel, geom


_FONT_PATH = Path(segd_render.__file__).parent / "fonts" / "DejaVuSans-Bold.ttf"
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}


def _chip_font(size: int):
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = ImageFont.truetype(str(_FONT_PATH), size)
    return _FONT_CACHE[size]


def _retitle(panel: Image.Image, title: str) -> Image.Image:
    """Replace the top-left title chip (drawn by segd.render._chip) with `title`.
    Same style as the render frame: SMALL_FONT_PX white text on a black chip."""
    fs = segd_render.SMALL_FONT_PX
    draw = ImageDraw.Draw(panel)
    old = draw.textbbox((4, 4), f"{title} | marks", font=_chip_font(fs))
    draw.rectangle((0, 0, old[2] + 5, old[3] + 5), fill=(0, 0, 0))
    l, t, r, b = draw.textbbox((4, 4), title, font=_chip_font(fs))
    pad = 3
    draw.rectangle((l - pad, t - pad, r + pad, b + pad), fill=(0, 0, 0))
    draw.text((4, 4), title, font=_chip_font(fs), fill=(255, 255, 255))
    return panel


def combine_panels(panels):
    """Same panel layout as segd.render._combine (gap + dark background)."""
    gap = segd_render.PANEL_GAP_PX
    w = sum(p.width for p in panels) + gap * (len(panels) - 1)
    h = max(p.height for p in panels)
    out = Image.new("RGB", (w, h), (24, 24, 24))
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += p.width + gap
    return out


# ---------------------------------------------------------------- main
def prepare_case(row, out_dir: Path, area_of: dict[str, float], rows_by_case: dict,
                 loader) -> dict:
    from segd.data import load_image, load_mask

    case = row.case
    image, affine, spacing = load_image(row)
    mask = load_mask(row, affine, image.shape)
    k_star, gt_area_mm2 = best_axial_slice(mask, spacing)

    markset, markset_source = load_or_compute_marks(case, image, spacing)
    if markset is None:
        raise RuntimeError(f"{case}: markset unavailable (segd.marks broken?)")

    img_s, ri_s = segd_render.render_view(image, "axial", k_star, crop="full", overlay="none",
                                          marks_level="coarse", markset=markset, spacing=spacing)
    img_r, ri_r = segd_render.render_raw_pointing(image, "axial", k_star, crop="full",
                                                  markset=markset, spacing=spacing)

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    path_s = images_dir / f"{case}_S.png"
    path_r = images_dir / f"{case}_R.png"
    img_s.save(path_s)
    img_r.save(path_r)

    gt2d = np.asarray(mask[:, :, k_star] > 0)  # GT on k* (axial fixed axis = 2)
    rows_n, cols_n = segd_marks.slice_shape(image.shape, "axial")

    drawn = []
    for mid, panel_idx, x, y in ri_s.marks:
        mark = markset.marks[int(mid)]
        ijk = list(mark.ijk)
        ijk[2] = k_star  # project_to_slice: fixed-axis coord becomes the active slice
        r, c = segd_marks.inplane_of(tuple(ijk), "axial", rows_n, cols_n)
        # GT lookup uses the raw projected voxel [i, j, k*]; (r, c) is display-oriented
        # (rows/cols are flipped relative to [i, j]) and recorded for reference only.
        inside = bool(gt2d[ijk[0], ijk[1]])
        drawn.append({
            "id": int(mid),
            "kind": mark.kind,
            "level": mark.level,
            "panel_idx": int(panel_idx),
            "panel_px": [round(float(x), 2), round(float(y), 2)],
            "combined_px": [round(float(x) + ri_s.views[1].panel_w + segd_render.PANEL_GAP_PX, 2),
                            round(float(y), 2)],
            "inplane_row_col": [int(r), int(c)],
            "projected_ijk": [int(v) for v in ijk],
            "inside_gt_on_k_star": inside,
        })

    # S-ex example (leave-one-out donor)
    donor, note = choose_donor(case, rows_by_case, area_of)
    record = {
        "case": case,
        "histotype": row.histotype,
        "sequence": row.sequence,
        "split": row.split,
        "spacing": [float(s) for s in spacing],
        "k_star": k_star,
        "gt_area_mm2": round(gt_area_mm2, 3),
        "image_S": rel_path(path_s),
        "image_R": rel_path(path_r),
        "image_example": None,
        "example_donor": donor,
        "example_note": note,
        "example_slices": None,
        "example_donor_gt_area_mm2": None,
        "example_geom": None,
        "view_geom_S": to_jsonable(ri_s.views[0]),
        "view_geom_R": to_jsonable(ri_r.views[0]),
        "drawn_marks": drawn,
        "n_visible_marks": len(markset.visible("axial", k_star, None, "coarse")),
        "n_drawn_marks": len(drawn),
        "coverage_on_k_star": any(d["inside_gt_on_k_star"] for d in drawn),
        "markset_cache": rel_path(MARKS_CACHE_DIR / f"{case}.npz"),
        "markset_source": markset_source,
        "marks_params": markset.params,
        "prepared_at": utcnow(),
    }

    if donor is not None:
        d_img, d_aff, d_sp = loader(donor)
        d_mask = load_mask(rows_by_case[donor], d_aff, d_img.shape)
        d_k = int(np.argmax(np.asarray(d_mask.sum(axis=(0, 1)), dtype=np.float64)))
        cj, ci = gt_centroid_slices(d_mask)
        p_ax, g_ax = outlined_panel(d_img, d_mask, "axial", d_k, d_sp)
        p_co, g_co = outlined_panel(d_img, d_mask, "coronal", cj, d_sp)
        p_sa, g_sa = outlined_panel(d_img, d_mask, "sagittal", ci, d_sp)
        path_ex = images_dir / f"{case}_ex.png"
        combine_panels([p_ax, p_co, p_sa]).save(path_ex)
        record["image_example"] = rel_path(path_ex)
        record["example_slices"] = {"axial": d_k, "coronal": cj, "sagittal": ci}
        record["example_donor_gt_area_mm2"] = round(area_of[donor], 3)
        record["example_geom"] = {"axial": to_jsonable(g_ax), "coronal": to_jsonable(g_co),
                                  "sagittal": to_jsonable(g_sa)}
        if not note:
            note2 = f"donor {donor} ({rows_by_case[donor].sequence}): closest to group-median GT area"
            record["example_note"] = note2

    del image, mask
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", default="", help="comma-separated case ids (default: all design)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = [c.strip() for c in args.cases.split(",") if c.strip()]
    all_rows = segd_data.read_cases("design")
    rows = all_rows if not wanted else [r for r in all_rows if r.case in wanted]
    missing = set(wanted) - {r.case for r in all_rows}
    if missing:
        raise SystemExit(f"unknown design cases: {sorted(missing)}")
    rows_by_case = {r.case: r for r in rows}
    all_design = {r.case: r for r in all_rows}
    if len(all_design) != len(all_rows) or len(rows_by_case) != len(rows):
        raise SystemExit("duplicate case ids in design split")

    # Pass 1 (headers + masks only): k* GT areas, needed before donor selection.
    import nibabel as nib
    from segd.data import load_mask
    area_of: dict[str, float] = {}
    spacing_of: dict[str, tuple] = {}
    t0 = time.time()
    for r in all_rows:
        hdr = nib.as_closest_canonical(nib.load(str(r.image_path))).header
        spacing_of[r.case] = tuple(float(s) for s in hdr.get_zooms()[:3])
        mask = load_mask(r)
        _, a = best_axial_slice(mask, spacing_of[r.case])
        area_of[r.case] = a
        del mask
    print(f"[prepare] pass 1 done: {len(all_rows)} masks in {time.time() - t0:.1f}s", flush=True)

    loader_cache: dict[str, tuple] = {}

    def loader(case_id: str):
        if case_id not in loader_cache:
            from segd.data import load_image
            d_row = all_design[case_id]
            loader_cache[case_id] = load_image(d_row)
            if len(loader_cache) > 6:  # ~6 volumes x <=0.4 GB keeps memory bounded
                for k in list(loader_cache)[:-3]:
                    if k != case_id:
                        loader_cache.pop(k, None)
        return loader_cache[case_id]

    manifest_path = out_dir / "manifest.jsonl"
    t0 = time.time()
    covered = 0
    with open(manifest_path, "w") as fh:
        for r in rows:
            rec = prepare_case(r, out_dir, area_of, all_design, loader)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            covered += int(rec["coverage_on_k_star"])
            print(f"[prepare] {r.case} {r.histotype}/{r.sequence} k*={rec['k_star']} "
                  f"area={rec['gt_area_mm2']:.0f}mm2 marks={rec['n_drawn_marks']} "
                  f"(visible {rec['n_visible_marks']}) inside_gt={rec['coverage_on_k_star']} "
                  f"donor={rec['example_donor']} [{time.time() - t0:.0f}s]", flush=True)

    summary = {
        "n_cases": len(rows),
        "n_coverage": covered,
        "coverage_rate": round(covered / max(len(rows), 1), 4),
        "conditions_prepared": ["R", "S", "S-gen", "S-desc", "S-ex"],
        "marks_cache_dir": rel_path(MARKS_CACHE_DIR),
        "manifest": rel_path(manifest_path),
        "notes": [
            "coverage = cases with >=1 drawn coarse mark whose projection lies inside GT on k*",
            "dev cases: marks are tuned on these same 30 cases (optimistic coverage)",
        ],
        "finished_at": utcnow(),
    }
    (out_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[prepare] coverage {covered}/{len(rows)} -> {summary['coverage_rate']:.3f}")
    print(f"[prepare] wrote {manifest_path}")


if __name__ == "__main__":
    main()
