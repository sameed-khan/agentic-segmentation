"""Offline scorer for agent-harness runs (Phase 1/3): Dice/HD95/NSD, localization, failure tags.

Usage:
  uv run --project harness/server python eval/v2/score.py results/<run_id> [--cases F14,M07] [--workers 8]
  uv run --project harness/server python eval/v2/score.py results/<run_id>/<case>          # single case

run_dir holds one ``<case>/`` subdir per case (harness/runner/README.md "Output"); a single case
dir is accepted and its outputs are written to the parent (treated as the run_dir). Reads per
case: ``trajectory.jsonl``, ``versions.npz``, ``final_mask.nii.gz``, ``case_meta.json`` (segd;
CONTRACTS.md §4) and, when present, ``meta.json`` + ``batch_summary.json`` (runner) — segd-only
runs without runner meta degrade gracefully. Writes ``scores.csv``, ``scores.md``,
``versions_dice.csv`` and prints a design_log-ready markdown block (harness_design_v2.md §3).

GT is loaded OFFLINE from data/mstt199 via segd.data.get_case/load_mask (CONTRACTS §0: only
offline code may read masks). The image itself is never materialised: its canonical header
(shape/affine/zooms) is read lazily for the load_mask reference, mirroring segd.data.load_image
without the data cost. The agent-side upload crop (case_meta crop_offset/crop_shape) never
matters here: versions and final_mask are stored in full-volume coordinates.

Metrics (final_mask.nii.gz vs the full multi-component GT; Dice also for every version):
  dice  = 2|P∩G| / (|P|+|G|). Empty pred or empty GT -> Dice 0.0, flagged empty_pred/empty_gt
          (HD95/NSD are NaN then). Versions are bit-packed bbox crops: versions.npz stores
          ``shape`` (full volume), ``ids``, ``parents`` (-1 for v0) and per version ``vN_bbox``
          (3x2 half-open bbox in FULL-volume coords, all -1 when empty), ``vN_nvox`` (= bbox
          crop size = packed bit count) and ``vN_packed`` (np.packbits of the bbox-cropped
          mask); unpacking mirrors segd.case_state.unpack_region.
  hd95  = 95th percentile of surface-to-surface distances, both directions concatenated (MedPy
          convention): surface = mask XOR binary_erosion(mask, 6-connectivity, border_value=1);
          distances in mm via distance_transform_edt(sampling=spacing) (1 mm isotropic for
          MSTT, but non-iso spacing is honoured). NaN when either mask or surface is empty.
  nsd2  = normalized surface Dice at 2 mm (Nikolov et al.): fraction of surface voxels of either
          side within 2 mm of the other side's surface. NaN when either surface is empty.
  HD95/NSD are computed on the union bbox of both masks; it contains every surface voxel of
  both sides, so the cropped distance transforms are exact.
  best version = highest per-version Dice (ties -> smallest version id); empty versions rank
  below any non-empty one (all-empty run: best_dice 0.0, best_version ""). The final version id
  comes from case_meta.json ``final_version``; if that file is missing it is re-identified by
  exact mask match against final_mask.nii.gz. final_is_best compares the final Dice to the best
  version Dice.

First positive click inside the lesion (rule; GT = full multi-component mask):
  The first trajectory.jsonl record with op in {point, lasso, scribble, box}, no error field and
  positive polarity (args.polarity in {add, positive, include, pos, true}; absent -> positive;
  mirrors segd.ops.POLARITY_TRUE). Candidates from the RESOLVED full-volume voxels recorded
  there: point -> the resolved voxel(s); lasso/scribble -> the resolved vertex voxels plus their
  rounded centroid; box -> the two corner voxels plus their rounded midpoint. inside = any
  candidate voxel lies in GT. distance_mm = distance from the closest candidate to the nearest
  GT voxel (0.0 when inside; NaN when no positive pointing prompt exists). The rasterized lasso
  interior / scribble polyline is NOT reconstructed (the trajectory stores vertices only); a box
  that straddles the lesion without a corner or its midpoint in GT counts as a miss — cheap,
  documented, overridable later by hand.

Failure tag (heuristic precedence harness > localization > delineation > budget > none):
  harness       infrastructure problem: runner meta.json ``errors`` non-empty; or
                final_mask.nii.gz missing although the case ran (trajectory/versions present);
                or forced finish due to an error (forced reason contains "error", e.g. "prompt
                error"); or any trajectory record with ``error_kind == "internal"`` (segd
                exception, not caused by agent input). Other trajectory ``error`` fields are agent-input errors (OpError: bad mark
                id, unknown arg, budget exhaustion, ...; CONTRACTS §5) and do NOT trigger this.
  localization  no positive point/lasso/scribble/box prompt ever landed inside GT and final
                Dice < 0.5 (a click on the lesion edge that still yields a good mask is not a
                localization failure).
  delineation   landed, but final Dice < 0.5.
  budget        forced finish with reason "wall time exceeded" / "budget exhausted", or nudge
                exhaustion ("max nudges reached" / "not done after max nudges").
  none          none of the above.
  wrong_structure is a separate boolean flag (not a tag): the final mask is non-empty but
  < 10% of its volume lies inside GT.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from segd.data import get_case, load_mask  # offline-only GT access (CONTRACTS §0)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ONECLICK_RESULTS_CSV = PROJECT_ROOT / "results" / "oneclick_design" / "results.csv"

# MSTT-199 published per-histotype U-Net Dice means (arXiv 2409.03110), reference only.
PUBLISHED_UNET_MEANS = {"F": 0.507, "L": 0.744, "M": 0.823, "N": 0.748, "V": 0.584}
PUBLISHED_UNET_AVG = 0.681

POSITIVE_POLARITY = {"add", "positive", "include", "pos", "true"}   # mirrors segd.ops.POLARITY_TRUE
POINTING_OPS = frozenset({"point", "lasso", "scribble", "box"})     # ops that carry a click
MODEL_PROMPT_OPS = frozenset({"point", "lasso", "scribble", "box", "refine"})  # = segd.config
VERSION_CREATING_OPS = frozenset({"point", "lasso", "scribble", "box", "paint",
                                  "refine", "edit"})               # exclude undo/checkout
DICE_LOCALIZE_OK = 0.5            # delineation threshold (docstring)
WRONG_STRUCTURE_FRAC = 0.10       # <10% of the final volume inside GT -> wrong_structure
NSD_TOL_MM = 2.0

FIELDS = ["case", "histotype", "sequence", "status", "dice", "hd95_mm", "nsd_2mm", "flags",
          "best_version", "best_dice", "final_version", "final_is_best", "dice_landed_frac",
          "wrong_structure", "first_click_inside", "first_click_dist_mm", "first_click_op",
          "first_click_version", "n_tool_calls", "n_model_prompts", "n_errors_traj",
          "n_errors_internal", "n_errors_meta", "wall_time_s", "confidence", "finish_reason", "forced_finish",
          "nudges", "tokens_in", "tokens_out", "tokens_total", "cost_usd", "oneclick_dice",
          "delta_dice", "tag", "error"]


# --------------------------------------------------------------------- io helpers
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


def parse_ts(ts: str) -> float:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


# ------------------------------------------------------------------- version npz
def version_crop(z, vid: int) -> tuple[tuple, np.ndarray] | None:
    """(bbox, bool crop) of one version from versions.npz, or None when empty.

    Mirrors segd.case_state.pack_region/unpack_region: ``vN_bbox`` is a half-open bbox in
    full-volume coordinates (all -1 when empty), ``vN_packed`` is np.packbits of the
    bbox-cropped mask flattened in C order, ``vN_nvox`` is the crop size (bit count).
    """
    nvox = int(z[f"v{vid}_nvox"])
    bbox = np.asarray(z[f"v{vid}_bbox"])
    if nvox == 0 or bbox[0, 0] < 0:
        return None
    bits = np.unpackbits(np.frombuffer(z[f"v{vid}_packed"].tobytes(), dtype=np.uint8))[:nvox]
    crop = bits.reshape(tuple(int(hi) - int(lo) for lo, hi in bbox)).astype(bool)
    return tuple((int(lo), int(hi)) for lo, hi in bbox), crop


def unpack_version(z, vid: int, shape: tuple) -> np.ndarray | None:
    """Full-volume uint8 mask of one version (None when empty) — used for exact-match checks."""
    crop = version_crop(z, vid)
    if crop is None:
        return None
    bbox, c = crop
    out = np.zeros(shape, dtype=np.uint8)
    out[tuple(slice(lo, hi) for lo, hi in bbox)] = c
    return out


def dice_from_counts(inter: int, pn: int, gn: int) -> float:
    if pn == 0 or gn == 0:          # documented: empty side -> Dice 0.0
        return 0.0
    return 2.0 * float(inter) / float(pn + gn)


# ---------------------------------------------------------------------- metrics
def mask_surface(mask: np.ndarray) -> np.ndarray:
    """Border voxels, MedPy convention: mask XOR binary_erosion(mask, 6-conn, border_value=1)."""
    m = np.asarray(mask).astype(bool)
    return m ^ ndimage.binary_erosion(m, structure=ndimage.generate_binary_structure(3, 1),
                                      border_value=1)


def surface_metrics(pred: np.ndarray, gt: np.ndarray, spacing: tuple,
                    nsd_tol_mm: float = NSD_TOL_MM) -> tuple[float, float]:
    """(hd95_mm, nsd) between two masks; NaN when either mask (or surface) is empty.

    Both masks are cropped to their union bbox padded by 1 voxel: every surface voxel of both
    sides lies strictly inside the crop (surface = mask XOR one-step 6-conn erosion), so the
    cropped surfaces and distance transforms are exact. At the volume border itself the
    MedPy ``border_value=1`` erosion treats out-of-array voxels as mask, i.e. border-touching
    voxels are not surface — identically in the crop and in full space.
    """
    p = np.asarray(pred).astype(bool)
    g = np.asarray(gt).astype(bool)
    if p.sum() == 0 or g.sum() == 0:
        return float("nan"), float("nan")
    idx = np.nonzero(p | g)
    sl = tuple(slice(max(int(a.min()) - 1, 0), min(int(a.max()) + 2, n))
               for a, n in zip(idx, p.shape))
    sp = mask_surface(p[sl])
    sg = mask_surface(g[sl])
    if not sp.any() or not sg.any():
        return float("nan"), float("nan")
    dt_to_gt = ndimage.distance_transform_edt(~sg, sampling=spacing)
    d_pg = dt_to_gt[sp]                      # pred surface -> nearest GT surface
    del dt_to_gt
    dt_to_pred = ndimage.distance_transform_edt(~sp, sampling=spacing)
    d_gp = dt_to_pred[sg]                    # GT surface -> nearest pred surface
    del dt_to_pred
    hd95 = float(np.percentile(np.concatenate([d_pg, d_gp]), 95))
    nsd = float((int((d_pg <= nsd_tol_mm).sum()) + int((d_gp <= nsd_tol_mm).sum()))
                / (int(d_pg.size) + int(d_gp.size)))
    return hd95, nsd


def distance_to_gt(points: list[tuple], gt: np.ndarray, spacing: tuple) -> float:
    """Distance in mm from the closest of `points` to the nearest GT voxel (0.0 when inside).

    Computed on a crop covering the GT bbox and all candidate points (exact: the nearest GT
    voxel to any candidate lies inside the crop). Requires a non-empty gt.
    """
    idx = np.nonzero(gt)
    lo = [int(a.min()) for a in idx]
    hi = [int(a.max()) + 1 for a in idx]
    for c in points:
        for a in range(3):
            lo[a] = min(lo[a], int(c[a]))
            hi[a] = max(hi[a], int(c[a]) + 1)
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    dt = ndimage.distance_transform_edt(gt[sl] == 0, sampling=spacing)
    return min(float(dt[tuple(int(c[a]) - lo[a] for a in range(3))]) for c in points)


# ----------------------------------------------------------------- first click
def click_candidates(op: str, resolved: list[dict]) -> list[tuple]:
    """Candidate click voxels (full volume) from a trajectory record's `resolved` entries.

    point -> the resolved voxel(s); lasso/scribble/box -> resolved vertices plus the rounded
    centroid of the vertices (see module docstring for the rationale and limitations).
    """
    pts = [tuple(int(c) for c in r["ijk"]) for r in resolved if r and r.get("ijk") is not None]
    if not pts:
        return []
    if op == "point":
        return pts
    centroid = tuple(int(v) for v in np.round(np.mean(np.asarray(pts, dtype=float), axis=0)))
    return pts + [centroid]


def in_gt(cand: tuple, gt: np.ndarray) -> bool:
    shape = gt.shape
    return all(0 <= c < shape[a] for a, c in enumerate(cand)) and bool(gt[cand])


def first_click(recs: list[dict], gt: np.ndarray, spacing: tuple) -> dict:
    """First positive point/lasso/scribble/box prompt: did it land inside GT, and how far away
    was the closest candidate click (mm)? Rule documented in the module docstring."""
    out = {"first_click_inside": False, "first_click_dist_mm": float("nan"),
           "first_click_op": "", "first_click_version": ""}
    if gt.sum() == 0:
        return out
    for r in recs:
        if r.get("error"):
            continue
        op = r.get("op")
        if op not in POINTING_OPS:
            continue
        pol = str((r.get("args") or {}).get("polarity", "add")).strip().lower()
        if pol not in POSITIVE_POLARITY:
            continue
        cand = click_candidates(op, r.get("resolved") or [])
        if not cand:
            continue
        out["first_click_inside"] = any(in_gt(c, gt) for c in cand)
        out["first_click_dist_mm"] = round(distance_to_gt(cand, gt, spacing), 3)
        out["first_click_op"] = op
        out["first_click_version"] = r.get("child") or ""
        return out
    return out


# ------------------------------------------------------------------ failure tag
def tag_case(*, meta_errors: list, missing_final: bool, ran: bool, forced: bool,
             forced_reason: str, click_inside: bool, dice: float, internal_errors: int = 0) -> str:
    """Failure tag; precedence harness > localization > delineation > budget > none (docstring)."""
    if ran and (meta_errors or missing_final or internal_errors
                or (forced and "error" in forced_reason)):
        return "harness"
    if not click_inside and (math.isnan(dice) or dice < DICE_LOCALIZE_OK):
        return "localization"
    if not math.isnan(dice) and dice < DICE_LOCALIZE_OK:
        return "delineation"
    if forced:
        rl = forced_reason.lower()
        if any(k in rl for k in ("wall", "budget", "nudge")):
            return "budget"
    return "none"


# --------------------------------------------------------------------- per case
def _image_reference(row) -> tuple[tuple, np.ndarray, tuple]:
    """Canonical (shape, affine, spacing) of the case image WITHOUT materialising the data —
    the header-only part of segd.data.load_image, used only as the load_mask reference."""
    im = nib.as_closest_canonical(nib.load(str(row.image_path)))
    return (tuple(int(v) for v in im.shape), np.asarray(im.affine),
            tuple(float(s) for s in im.header.get_zooms()[:3]))


def _finish_record(recs: list[dict]) -> dict | None:
    for r in recs:
        if r.get("op") == "finish" and not r.get("error"):
            return r
    return None


def _identify_final_version(z, shape: tuple, final_mask: np.ndarray) -> int | None:
    """Version whose unpacked mask equals final_mask.nii.gz (used when case_meta.json is absent)."""
    fsum = int(final_mask.sum())
    for vid in (int(v) for v in z["ids"]):
        crop = version_crop(z, vid)
        if crop is None or int(crop[1].sum()) != fsum:
            continue
        g = unpack_version(z, vid, shape)
        if g is not None and np.array_equal(g.astype(bool), final_mask.astype(bool)):
            return vid
    return None


def score_case(case_dir_str: str) -> dict:
    """Score one case directory; never raises (errors land in status/error)."""
    case_dir = Path(case_dir_str)
    case = case_dir.name
    rec = {k: "" for k in FIELDS}
    rec.update(case=case, status="error", _best_dice_raw="")
    flags: list[str] = []

    traj = read_jsonl(case_dir / "trajectory.jsonl")
    case_meta = read_json(case_dir / "case_meta.json") or {}
    meta = read_json(case_dir / "meta.json") or {}          # runner meta; optional
    versions_path = case_dir / "versions.npz"
    final_path = case_dir / "final_mask.nii.gz"

    ran = bool(traj) or versions_path.exists() or bool(case_meta)
    missing_final = ran and not final_path.exists()
    if missing_final:
        flags.append("missing_final_mask")

    # ---- GT (offline; CONTRACTS §0) ---------------------------------------------
    gt: np.ndarray | None = None
    spacing = (1.0, 1.0, 1.0)
    gt_error = ""
    try:
        row = get_case(case)
        rec["histotype"], rec["sequence"] = row.histotype, row.sequence
        shape_ref, affine_ref, spacing = _image_reference(row)
        gt = load_mask(row, affine_ref, shape_ref)
    except Exception as e:                                  # noqa: BLE001
        gt_error = f"gt: {e!r}"
        flags.append("missing_gt")

    # ---- final mask metrics -------------------------------------------------------
    dice = float("nan")
    landed_frac = float("nan")
    final_mask: np.ndarray | None = None
    if final_path.exists() and gt is not None:
        try:
            final_mask = np.asanyarray(nib.load(str(final_path)).dataobj)
            inter = int(((final_mask > 0) & (gt > 0)).sum())
            pn, gn = int((final_mask > 0).sum()), int(gt.sum())
            dice = dice_from_counts(inter, pn, gn)
            rec["dice"] = dice
            rec["hd95_mm"], rec["nsd_2mm"] = surface_metrics(final_mask, gt, spacing)
            if pn == 0:
                flags.append("empty_pred")
            if gn == 0:
                flags.append("empty_gt")
            if pn > 0:
                landed_frac = inter / pn
                if landed_frac < WRONG_STRUCTURE_FRAC:
                    flags.append("wrong_structure")
        except Exception as e:                              # noqa: BLE001
            gt_error = f"{gt_error} final: {e!r}".strip() if gt_error else f"final: {e!r}"
            flags.append("final_metric_error")

    # ---- versions ------------------------------------------------------------------
    version_rows: list[tuple] = []      # (case, version, parent, op, dice)
    best_vid, best_dice = None, float("nan")
    z = None
    if versions_path.exists():
        try:
            z = np.load(versions_path)
            shape = tuple(int(v) for v in z["shape"])
            op_by_version: dict[int, str] = {}
            for r in traj:
                child = r.get("child")
                if not r.get("error") and child and r.get("op") in VERSION_CREATING_OPS:
                    try:
                        op_by_version.setdefault(int(str(child)[1:]), r["op"])
                    except ValueError:
                        pass
            nonempty: list[tuple[int, float]] = []
            for vid, parent in zip((int(v) for v in z["ids"]), (int(p) for p in z["parents"])):
                crop = version_crop(z, vid)
                if gt is None or crop is None:
                    d = 0.0 if (gt is not None and crop is None) else float("nan")
                else:
                    bbox, c = crop
                    inter = int((c & (gt[tuple(slice(lo, hi) for lo, hi in bbox)] > 0)).sum())
                    d = dice_from_counts(inter, int(c.sum()), int(gt.sum()))
                version_rows.append((case, f"v{vid}", "" if parent < 0 else f"v{parent}",
                                     op_by_version.get(vid, ""),
                                     "" if math.isnan(d) else round(d, 4)))
                if crop is not None and gt is not None:
                    nonempty.append((vid, d))
            if nonempty:
                best_vid, best_dice = min(nonempty, key=lambda t: (-t[1], t[0]))
            elif gt is not None:
                best_dice = 0.0
            rec["_best_dice_raw"] = "" if math.isnan(best_dice) else best_dice
            rec["best_dice"] = "" if math.isnan(best_dice) else round(best_dice, 4)
            rec["best_version"] = "" if best_vid is None else f"v{best_vid}"
        except Exception as e:                              # noqa: BLE001
            gt_error = f"{gt_error} versions: {e!r}".strip() if gt_error else f"versions: {e!r}"
            flags.append("versions_error")
            z = None

    # final version id + final == best
    final_vid = case_meta.get("final_version", "")
    if not final_vid and final_mask is not None and z is not None:
        vid = _identify_final_version(z, tuple(int(v) for v in z["shape"]), final_mask)
        final_vid = "" if vid is None else f"v{vid}"
    rec["final_version"] = final_vid
    rec["final_is_best"] = bool(
        rec["best_dice"] != "" and isinstance(dice, float) and not math.isnan(dice)
        and rec["_best_dice_raw"] != "" and abs(dice - float(rec["_best_dice_raw"])) < 1e-9)

    # ---- trajectory / meta bookkeeping ----------------------------------------------
    ok_records = [r for r in traj if not r.get("error")]
    rec["n_tool_calls"] = case_meta.get(
        "tool_calls", len([r for r in ok_records if r.get("op") != "finish"])
        + (1 if _finish_record(traj) else 0))
    rec["n_model_prompts"] = case_meta.get(
        "model_prompts", len([r for r in ok_records if r.get("op") in MODEL_PROMPT_OPS
                              and r.get("child")]))
    rec["n_errors_traj"] = len([r for r in traj if r.get("error")])
    rec["n_errors_internal"] = len([r for r in traj if r.get("error_kind") == "internal"])
    meta_errors = list(meta.get("errors") or [])
    rec["n_errors_meta"] = len(meta_errors)
    fin = _finish_record(traj)
    rec["confidence"] = (fin or {}).get("args", {}).get("confidence", case_meta.get("confidence", ""))
    reason = (fin or {}).get("args", {}).get("reason", case_meta.get("reason", ""))
    rec["finish_reason"] = reason
    forced = bool(meta.get("forced_finish")) or str(reason).startswith("forced:")
    rec["forced_finish"] = bool(forced)
    rec["nudges"] = meta.get("nudges_used", "")
    tok = meta.get("token_usage") or {}
    rec["tokens_in"], rec["tokens_out"], rec["tokens_total"] = (
        tok.get("input", ""), tok.get("output", ""), tok.get("total", ""))
    rec["cost_usd"] = meta.get("cost_usd", "")
    if meta_errors:
        flags.append("meta_errors")
        gt_error = f"{gt_error} meta: {meta_errors[0]}".strip() if gt_error \
            else f"meta: {meta_errors[0]}"
    rec["error"] = gt_error

    # wall time: runner meta -> case_meta created/finished -> trajectory ts span
    wall = float("nan")
    if meta.get("wall_time_s") is not None:
        wall = float(meta["wall_time_s"])
    elif case_meta.get("created_at") and case_meta.get("finished_at"):
        try:
            wall = parse_ts(case_meta["finished_at"]) - parse_ts(case_meta["created_at"])
        except ValueError:
            pass
    elif traj:
        try:
            wall = parse_ts(traj[-1]["ts"]) - parse_ts(traj[0]["ts"])
        except (ValueError, KeyError):
            pass
    rec["wall_time_s"] = "" if math.isnan(wall) else round(wall, 2)

    # ---- localization -----------------------------------------------------------------
    if gt is not None:
        rec.update(first_click(traj, gt, spacing))
    rec["dice_landed_frac"] = "" if math.isnan(landed_frac) else landed_frac

    # ---- tag --------------------------------------------------------------------------
    rec["flags"] = ";".join(flags)
    rec["wrong_structure"] = "wrong_structure" in flags
    scoring_error = "final_metric_error" in flags or "versions_error" in flags
    if gt is None:
        rec["tag"] = ""                                     # cannot judge without GT
        rec["status"] = "missing_gt"
    else:
        rec["tag"] = tag_case(meta_errors=meta_errors, missing_final=missing_final, ran=ran,
                              forced=bool(forced), forced_reason=str(reason),
                              click_inside=bool(rec["first_click_inside"]), dice=dice,
                              internal_errors=rec.get("n_errors_internal", 0))
        rec["status"] = "missing_final" if missing_final else ("error" if scoring_error else "ok")
    rec["_version_rows"] = version_rows
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


def discover_case_dirs(run_dir: Path, only: set[str] | None) -> tuple[Path, list[Path]]:
    """Resolve run_dir (run root or a single case dir) -> (output_dir, [case dirs])."""
    if (run_dir / "trajectory.jsonl").exists() or (run_dir / "versions.npz").exists() \
            or (run_dir / "case_meta.json").exists():
        return run_dir.parent, [run_dir]
    case_dirs = sorted(p for p in run_dir.iterdir() if p.is_dir()
                       and ((p / "trajectory.jsonl").exists() or (p / "versions.npz").exists()
                            or (p / "case_meta.json").exists()))
    if only:
        case_dirs = [p for p in case_dirs if p.name in only]
    return run_dir, case_dirs


def load_oneclick_dice() -> dict[str, float]:
    if not ONECLICK_RESULTS_CSV.exists():
        return {}
    out = {}
    with open(ONECLICK_RESULTS_CSV) as f:
        for r in csv.DictReader(f):
            try:
                out[r["case"]] = float(r["dice_all"])
            except (KeyError, ValueError):
                continue
    return out


def fmt_agg(vals: list[float]) -> tuple[str, str]:
    if not vals:
        return "", ""
    a = np.asarray(vals, dtype=float)
    return f"{a.mean():.3f}", f"{np.median(a):.3f}"


def write_scores_md(out_dir: Path, recs: list[dict], oneclick: dict[str, float],
                    harness_sha: str, model: str) -> None:
    ok = [r for r in recs if r["status"] == "ok"]
    lines = [f"# Scores — {out_dir.name} ({datetime.now(timezone.utc).date().isoformat()})", ""]
    lines.append(f"Cases scored: {len(recs)} (ok {len(ok)}) | harness_sha256: "
                 f"`{harness_sha or 'n/a'}` | model: {model or 'n/a'}")
    lines.append("")
    lines.append("Dice/HD95/NSD of final_mask.nii.gz vs full GT; rules in eval/v2/score.py docstring.")
    lines.append("")
    lines.append("| case | histotype | dice | HD95 mm | NSD@2mm | best | best dice | final==best | "
                 "first click in GT | dist mm | tag | flags | oneclick | Δ |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in recs:
        oc = oneclick.get(r["case"])
        delta = (r["dice"] - oc) if (oc is not None and isinstance(r["dice"], float)
                                     and not math.isnan(r["dice"])) else None
        fci = ("yes" if r["first_click_inside"] else "no") if r["first_click_inside"] != "" else ""
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["case"], r["histotype"] or "—", _md(r["dice"]), _md(r["hd95_mm"]), _md(r["nsd_2mm"]),
            r["best_version"] or "—", _md(r["best_dice"]),
            ("✓" if r["final_is_best"] else "✗") if r["status"] == "ok" else "—",
            _md(fci), _md(r["first_click_dist_mm"]), r["tag"] or "—", r["flags"] or "—",
            f"{oc:.3f}" if oc is not None else "—",
            f"{delta:+.3f}" if delta is not None else "—"))
    lines.append("")
    lines.append("## Per-histotype Dice (final mask, vs full GT)")
    lines.append("")
    lines.append("| histotype | n | mean | median | published U-Net mean | Δ vs U-Net | "
                 "oneclick mean | Δ vs oneclick |")
    lines.append("|---|---|---|---|---|---|---|---|")
    hts = sorted({r["histotype"] for r in ok}, key=lambda h: list(PUBLISHED_UNET_MEANS).index(h)
                 if h in PUBLISHED_UNET_MEANS else 99)
    all_dice: list[float] = []
    for ht in hts:
        rs = [r for r in ok if r["histotype"] == ht]
        ds = [r["dice"] for r in rs]
        all_dice.extend(ds)
        mean, med = fmt_agg(ds)
        oc_mean, _ = fmt_agg([oneclick[r["case"]] for r in rs if r["case"] in oneclick])
        pub = PUBLISHED_UNET_MEANS.get(ht)
        d_pub = (f"{float(mean) - pub:+.3f}" if (pub is not None and mean) else "—")
        d_oc = f"{float(mean) - float(oc_mean):+.3f}" if (mean and oc_mean) else "—"
        lines.append(f"| {ht} | {len(rs)} | {_md(mean)} | {_md(med)} | "
                     f"{f'{pub:.3f}' if pub is not None else '—'} | {d_pub} | {_md(oc_mean)} | {d_oc} |")
    if all_dice:
        mean, med = fmt_agg(all_dice)
        oc_mean, _ = fmt_agg([oneclick[r["case"]] for r in ok if r["case"] in oneclick])
        d_pub = f"{float(mean) - PUBLISHED_UNET_AVG:+.3f}" if mean else "—"
        d_oc = f"{float(mean) - float(oc_mean):+.3f}" if (mean and oc_mean) else "—"
        lines.append(f"| **overall** | {len(all_dice)} | {_md(mean)} | {_md(med)} | "
                     f"{PUBLISHED_UNET_AVG:.3f} | {d_pub} | {_md(oc_mean)} | {d_oc} |")
    lines += ["", "## Notes", ""]
    notes = []
    for r in recs:
        if r["status"] != "ok":
            notes.append(f"- {r['case']}: status={r['status']} — {r['error']}")
        elif r["flags"]:
            notes.append(f"- {r['case']} ({r['histotype']}): flags {r['flags']} "
                         f"(dice={_fmt(r['dice'])}, tag={r['tag']}).")
    lines += notes if notes else ["- none."]
    lines.append("")
    (out_dir / "scores.md").write_text("\n".join(lines) + "\n")


def design_log_block(recs: list[dict], harness_sha: str, title: str) -> str:
    """design_log-ready markdown (§3: date, harness sha, per-case Dice, first click, tag)."""
    today = datetime.now(timezone.utc).date().isoformat()
    lines = [f"## {today} — {title} (offline scorer)",
             "",
             f"harness_sha256: `{harness_sha or 'n/a (no runner meta.json)'}`",
             "",
             "| case | dice | first_positive_click_inside | dist_mm | tag | notes |",
             "|---|---|---|---|---|---|"]
    for r in recs:
        fci = ("yes" if r["first_click_inside"] else "no") if r["first_click_inside"] != "" else ""
        notes = [x for x in (r["flags"], r["error"]) if x]
        if r["n_errors_traj"]:
            notes.append(f"{r['n_errors_traj']} agent-input errors")
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            r["case"], _md(r["dice"]), _md(fci), _md(r["first_click_dist_mm"]),
            r["tag"] or "—", "; ".join(notes)))
    ok = [r for r in recs if r["status"] == "ok"]
    ds = [r["dice"] for r in ok]
    if ds:
        mean, med = fmt_agg(ds)
        lines += ["", f"overall: mean {mean}, median {med} over {len(ds)} scored cases."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Offline scorer for agent-harness runs.")
    ap.add_argument("run_dir", type=Path,
                    help="results/<run_id>/ (one <case>/ subdir each) or a single case dir")
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
    print(f"{len(case_dirs)} case(s) in {out_dir}; {args.workers} workers", flush=True)

    t0 = time.time()
    if args.workers <= 1 or len(case_dirs) == 1:
        recs = [score_case(str(p)) for p in case_dirs]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(case_dirs))) as ex:
            recs = list(ex.map(score_case, [str(p) for p in case_dirs]))
    print(f"scored {len(recs)} case(s) in {time.time() - t0:.1f}s", flush=True)

    # runner-level context (optional): batch_summary.json, else the first meta.json
    batch = read_json(out_dir / "batch_summary.json") or {}
    metas = [m for m in (read_json(p / "meta.json") for p in case_dirs) if m]
    harness_sha = batch.get("harness_sha256") or next(
        (m.get("harness_sha256") for m in metas if m.get("harness_sha256")), "")
    model = batch.get("model") or next((m.get("model") for m in metas if m.get("model")), "")

    oneclick = load_oneclick_dice()
    for r in recs:
        oc = oneclick.get(r["case"])
        r["oneclick_dice"] = oc if oc is not None else ""
        r["delta_dice"] = (round(r["dice"] - oc, 4)
                           if oc is not None and isinstance(r["dice"], float)
                           and not math.isnan(r["dice"]) else "")

    ordered = sorted(recs, key=lambda r: r["case"])
    with open(out_dir / "scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in ordered:
            w.writerow({k: _fmt(r[k]) for k in FIELDS})

    version_rows = [vr for r in ordered for vr in r["_version_rows"]]
    if version_rows:
        with open(out_dir / "versions_dice.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case", "version", "parent", "op", "dice"])
            w.writerows(version_rows)

    write_scores_md(out_dir, ordered, oneclick, harness_sha, model)

    print("\n----- design_log entry (paste into eval/v2/design_log.md) -----", flush=True)
    print(design_log_block(ordered, harness_sha, args.run_dir.name), flush=True)
    print("----- end design_log entry -----", flush=True)
    print(f"wrote {out_dir / 'scores.csv'}, {out_dir / 'scores.md'}"
          + (f", {out_dir / 'versions_dice.csv'}" if version_rows else ""), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
