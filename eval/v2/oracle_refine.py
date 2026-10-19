"""Oracle iterative-refinement ceiling for nnInteractive on the design cases (no LLM).

    uv run --project harness/server python eval/v2/oracle_refine.py [--clicks 10] [--out results/oracle_refine]

Protocol (standard interactive-segmentation simulation, cf. SAM/nnInteractive click simulation):
click 1 = positive point at the GT largest-component centroid (same seed as oneclick_baseline.py);
then for k = 2..N: take the largest connected error region (false negative or false positive, by voxel count),
click at its interior point (max of the Euclidean distance transform inside that region), positive for FN,
negative for FP. Dice is recorded after every click. GT is used only to place clicks (oracle user).

Purpose: upper bound for "make the agent iterate with point prompts". If Dice saturates far below 1, the gap is
nnInteractive/annotation convention rather than agent behaviour.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from nnInteractive.inference.remote import nnInteractiveRemoteInferenceSession

from segd.data import PROJECT_ROOT, load_image, load_mask, read_cases

URL = "http://127.0.0.1:1527"


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = int(a.sum()) + int(b.sum())
    return 2.0 * int(np.logical_and(a, b).sum()) / s if s else 1.0


def interior_point(region: np.ndarray) -> tuple[int, int, int]:
    sl = ndi.find_objects(region.astype(np.uint8))[0]
    sub = np.pad(region[sl], 1)
    dt = ndi.distance_transform_edt(sub)
    p = np.unravel_index(int(np.argmax(dt)), dt.shape)
    return tuple(int(p[a] - 1 + sl[a].start) for a in range(3))


def run_case(row, n_clicks: int) -> dict:
    img, aff, sp = load_image(row)
    gt = load_mask(row, aff, img.shape) > 0
    lab, n = ndi.label(gt)
    sizes = ndi.sum(gt, lab, range(1, n + 1))
    lcc = lab == (1 + int(np.argmax(sizes)))
    c = np.round(ndi.center_of_mass(lcc)).astype(int)
    seed = tuple(int(v) for v in c) if lcc[tuple(c)] else interior_point(lcc)
    tb = np.zeros(img.shape, dtype=np.uint8)
    s = nnInteractiveRemoteInferenceSession(URL, read_timeout=300, write_timeout=600)
    trace = []
    try:
        s.set_image(img.astype(np.float16)[None])
        s.set_target_buffer(tb)
        s.add_point_interaction(list(seed), include_interaction=True)
        trace.append({"k": 1, "kind": "pos", "ijk": seed, "dice": dice(tb > 0, gt)})
        for k in range(2, n_clicks + 1):
            pred = tb > 0
            fn, fp = gt & ~pred, pred & ~gt
            best = None
            for kind, err in (("pos", fn), ("neg", fp)):
                if not err.any():
                    continue
                el, en = ndi.label(err)
                es = np.bincount(el.ravel())[1:]
                i = int(np.argmax(es))
                if best is None or es[i] > best[2]:
                    best = (kind, el == i + 1, int(es[i]))
            if best is None:
                break
            kind, region, _ = best
            p = interior_point(region)
            s.add_point_interaction(list(p), include_interaction=(kind == "pos"))
            trace.append({"k": k, "kind": kind, "ijk": p, "dice": dice(tb > 0, gt)})
    finally:
        try:
            s.close()
        except Exception:
            pass
    return {"case": row.case, "histotype": row.histotype, "gt_ml": round(float(gt.sum()) / 1000, 2), "trace": trace}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clicks", type=int, default=10)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "oracle_refine"))
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_cases("design")
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda r: run_case(r, a.clicks), rows))
    with open(out / "traces.jsonl", "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    K = a.clicks
    with open(out / "dice_by_click.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "histotype", "gt_ml"] + [f"dice_{k}" for k in range(1, K + 1)])
        for r in res:
            d = [t["dice"] for t in r["trace"]]
            d += [d[-1]] * (K - len(d))
            w.writerow([r["case"], r["histotype"], r["gt_ml"]] + [f"{v:.4f}" for v in d])
    means = []
    for k in range(K):
        v = [(r["trace"][min(k, len(r["trace"]) - 1)]["dice"]) for r in res]
        means.append((k + 1, float(np.mean(v)), float(np.median(v))))
    lines = ["# Oracle iterative refinement (no LLM), design cases", "",
             "Click 1 = GT centroid; then largest error region interior, +/- by FN/FP. "
             f"{len(res)} cases, {time.time() - t0:.0f} s.", "", "| clicks | mean Dice | median Dice |", "|---|---|---|"]
    lines += [f"| {k} | {m:.3f} | {md:.3f} |" for k, m, md in means]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
