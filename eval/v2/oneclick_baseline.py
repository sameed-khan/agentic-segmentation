"""One-click nnInteractive baseline on the MSTT design cases (no LLM).

For each case: seed the GT largest connected component at its centroid (snapping to the
nearest voxel inside the component when the rounded centroid lands outside it), upload the
image as float16 ``(1, I, J, K)``, set a uint8 target buffer, fire a single positive
``add_point_interaction`` and score the result.

Scores: Dice vs the full GT mask (all components) and vs the GT largest component, predicted
volume (mL), GT volume (mL), number of predicted connected components; timings for the image
upload and the point prompt.

Writes ``results/oneclick_design/results.csv``, ``results/oneclick_design/masks/<case>.nii.gz``
(uint8, original affine) and ``results/oneclick_design/summary.md``.

Run:  uv run --project harness/server eval/v2/oneclick_baseline.py [split]
"""
from __future__ import annotations

import csv
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from nnInteractive.inference.remote import nnInteractiveRemoteInferenceSession
from nnInteractive.inference.remote.remote_session import (
    ServerAtCapacityError,
    SessionExpiredError,
)

from segd.data import PROJECT_ROOT, load_image, load_mask, read_cases

URL = "http://127.0.0.1:1527"
MAX_WORKERS = 4
RETRYABLE_ERRORS = (httpx.TransportError, ConnectionError, OSError,
                    ServerAtCapacityError, SessionExpiredError)

# MSTT-199 published per-histotype U-Net Dice (means; arXiv 2409.03110), for reference only.
PUBLISHED_UNET_MEANS = {"F": 0.507, "L": 0.744, "M": 0.823, "N": 0.748, "V": 0.584}
PUBLISHED_UNET_AVG = 0.681

FIELDS = ["case", "histotype", "sequence", "seed_i", "seed_j", "seed_k", "snap_rule",
          "snap_dist", "dice_all", "dice_lcc", "pred_ml", "gt_ml", "pred_components",
          "gt_components", "gt_largest_frac", "t_set_image_s", "t_prompt_s", "status", "error"]

split = sys.argv[1] if len(sys.argv) > 1 else "design"
OUT = PROJECT_ROOT / "results" / "oneclick_design"
MASKS = OUT / "masks"

stop = threading.Event()  # set once the server looks unreachable; no new cases start


def healthz(timeout: float = 5.0) -> bool:
    try:
        return httpx.get(f"{URL}/healthz", timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


def pick_seed(comp: np.ndarray) -> tuple[tuple[int, int, int], str, float]:
    """Rounded centroid of `comp`; nearest voxel in `comp` if that centroid is outside it."""
    centroid = np.round(ndi.center_of_mass(comp)).astype(int)
    if comp[tuple(centroid)]:
        return tuple(int(v) for v in centroid), "centroid_inside", 0.0
    dist, idx = ndi.distance_transform_edt(~comp, return_indices=True)
    snap = tuple(int(idx[a][tuple(centroid)]) for a in range(3))
    return snap, "nearest_voxel", float(dist[tuple(centroid)])


def attempt(row) -> dict:
    """One full case run through a fresh session. Raises on failure; caller retries/closes."""
    rec: dict = {k: "" for k in FIELDS}
    rec.update(case=row.case, histotype=row.histotype, sequence=row.sequence, status="ok")

    img, aff, sp = load_image(row)
    gt = load_mask(row, aff, img.shape)
    voxel_ml = float(np.prod(sp)) / 1000.0
    lab, n_gt = ndi.label(gt)
    if n_gt == 0:
        rec.update(status="empty_gt", error="GT mask is empty")
        return rec
    sizes = ndi.sum(gt, lab, range(1, n_gt + 1))
    lcc = lab == (1 + int(np.argmax(sizes)))
    seed, rule, snap_dist = pick_seed(lcc)
    rec.update(seed_i=seed[0], seed_j=seed[1], seed_k=seed[2], snap_rule=rule,
               snap_dist=round(snap_dist, 3), gt_ml=round(float(gt.sum()) * voxel_ml, 3),
               gt_components=int(n_gt), gt_largest_frac=round(float(sizes.max() / gt.sum()), 4))

    img16 = img.astype(np.float16)[None]          # (1, I, J, K)
    tb = np.zeros(img.shape, dtype=np.uint8)
    session = None
    try:
        session = nnInteractiveRemoteInferenceSession(URL, read_timeout=300, write_timeout=600)
        t0 = time.time()
        session.set_image(img16)
        t1 = time.time()
        session.set_target_buffer(tb)
        t2 = time.time()
        session.add_point_interaction(list(seed), include_interaction=True)
        t3 = time.time()
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass

    pred = tb > 0
    pred_n = int(pred.sum())
    gt_n = int(gt.sum())
    lcc_n = int(lcc.sum())
    inter_all = int(np.logical_and(pred, gt).sum())
    inter_lcc = int(np.logical_and(pred, lcc).sum())
    rec.update(
        dice_all=round(2 * inter_all / (pred_n + gt_n), 4),
        dice_lcc=round(2 * inter_lcc / (pred_n + lcc_n), 4),
        pred_ml=round(pred_n * voxel_ml, 3),
        pred_components=int(ndi.label(pred)[1]),
        t_set_image_s=round(t1 - t0, 2),
        t_prompt_s=round(t3 - t2, 2),
    )
    nib.save(nib.Nifti1Image(pred.astype(np.uint8), aff), str(MASKS / f"{row.case}.nii.gz"))
    return rec


def worker(row) -> dict:
    if stop.is_set():
        rec = {k: "" for k in FIELDS}
        rec.update(case=row.case, histotype=row.histotype, sequence=row.sequence,
                   status="skipped", error="server unreachable")
        return rec
    error: Exception | None = None
    for i in range(2):  # one retry on connection errors
        try:
            return attempt(row)
        except RETRYABLE_ERRORS as e:
            error = e
            time.sleep(2.0 * (i + 1))
        except Exception as e:
            error = e
            break
    if not healthz():
        stop.set()
    rec = {k: "" for k in FIELDS}
    rec.update(case=row.case, histotype=row.histotype, sequence=row.sequence,
               status="failed", error=repr(error))
    return rec


def write_summary(recs: list[dict]) -> None:
    ok = [r for r in recs if r["status"] == "ok"]
    lines = ["# One-click nnInteractive baseline — MSTT design cases", "",
             f"Cases: {len(recs)}  |  succeeded: {len(ok)}  |  "
             f"failed/skipped: {len(recs) - len(ok)}", "",
             "One positive point at the GT largest-component centroid (snapped inside the "
             "component when the rounded centroid lands outside), no LLM. Dice is computed "
             "against the full GT mask.", "",
             "| histotype | n | Dice (all) mean | Dice (all) median | Dice (LCC) mean | "
             "Dice (LCC) median | published U-Net mean (ref) |",
             "|---|---|---|---|---|---|---|"]
    all_dice, all_dice_lcc = [], []
    for ht in sorted({r["histotype"] for r in ok}, key=lambda h: list(PUBLISHED_UNET_MEANS).index(h)):
        rs = [r for r in ok if r["histotype"] == ht]
        da = np.array([r["dice_all"] for r in rs], dtype=float)
        dl = np.array([r["dice_lcc"] for r in rs], dtype=float)
        all_dice.extend(da.tolist())
        all_dice_lcc.extend(dl.tolist())
        lines.append(f"| {ht} | {len(rs)} | {da.mean():.3f} | {np.median(da):.3f} | "
                     f"{dl.mean():.3f} | {np.median(dl):.3f} | {PUBLISHED_UNET_MEANS[ht]:.3f} |")
    if all_dice:
        da, dl = np.array(all_dice), np.array(all_dice_lcc)
        lines.append(f"| **overall** | {len(ok)} | {da.mean():.3f} | {np.median(da):.3f} | "
                     f"{dl.mean():.3f} | {np.median(dl):.3f} | {PUBLISHED_UNET_AVG:.3f} |")
    if ok:
        lines += ["", f"Mean timings over {len(ok)} cases: set_image "
                      f"{np.mean([r['t_set_image_s'] for r in ok]):.2f}s, prompt "
                      f"{np.mean([r['t_prompt_s'] for r in ok]):.2f}s."]
    lines += ["", "## Notes", "",
              "- The published U-Net means are for the full MSTT-199 dataset (n=199) and are "
              "**not directly comparable**: different case set, and here the prompt uses an "
              "oracle GT seed. Treat them as rough context only.", ""]

    flags = []
    for r in ok:
        if r["pred_components"] == 0:
            flags.append(f"- {r['case']} ({r['histotype']}): predicted mask is **empty** "
                         f"(dice_all={r['dice_all']}).")
        elif r["gt_ml"] and r["pred_ml"] > 2 * r["gt_ml"]:
            flags.append(f"- {r['case']} ({r['histotype']}): predicted volume "
                         f"{r['pred_ml']} mL is >2x GT {r['gt_ml']} mL "
                         f"(dice_all={r['dice_all']}, pred_components={r['pred_components']}).")
        elif r["gt_ml"] and r["pred_ml"] < 0.2 * r["gt_ml"]:
            flags.append(f"- {r['case']} ({r['histotype']}): predicted volume "
                         f"{r['pred_ml']} mL is <0.2x GT {r['gt_ml']} mL "
                         f"(dice_all={r['dice_all']}).")
    failed = [r for r in recs if r["status"] != "ok"]
    if failed:
        lines += ["## Failed/skipped cases", ""]
        lines += [f"- {r['case']}: {r['status']} — {r['error']}" for r in failed]
        lines.append("")
    lines += ["## Notable cases", ""]
    lines += flags if flags else ["- none"]
    lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    if not healthz():
        print(f"ERROR: server not reachable at {URL}/healthz — stopping.", flush=True)
        return 1
    rows = read_cases(split)
    MASKS.mkdir(parents=True, exist_ok=True)
    print(f"{len(rows)} {split} cases; {MAX_WORKERS} workers; server {URL}", flush=True)

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(worker, row): row for row in rows}
        for fut in futures:
            row = futures[fut]
            rec = fut.result()
            results[row.case] = rec
            print(f"  [{len(results):2d}/{len(rows)}] {row.case} {row.histotype} "
                  f"{rec['status']} dice_all={rec['dice_all']} "
                  f"pred_ml={rec['pred_ml']} err={rec['error']}", flush=True)

    recs = [results[row.case] for row in rows]
    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(recs)
    write_summary(recs)
    print(f"wrote {OUT / 'results.csv'} and {OUT / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
