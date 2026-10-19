"""Offline benchmark of candidate-correction generators (segd.candidates) against GT.

    cd harness/server && uv run python ../../eval/v2/candidate_bench.py \
        [--runs phase1-05,phase1-07-xhigh] [--cases M33,N34] \
        [--methods 'intensity;supervoxel:hops=3;ends;combo']  (';'-separated specs) [--probe-runs phase1-05,phase1-07-xhigh] \
        [--out ../../results/candidate_bench] [--workers 8] [--probe-workers 3] [--render]

Starting state = the agent's final mask results/<run>/<case>/final_mask.nii.gz (compared with GT exactly like
eval/v2/score.py, i.e. the raw array). States whose final mask does not overlap GT are skipped (counted).
GT is used ONLY here, to score candidates the generators produced without it.

Method specs: ``name`` or ``name:key=val,key=val`` (e.g. ``intensity:lo=2,hi=98,D_mm=20``); the spec string is
the method label in the outputs. ``probe`` needs the nnInteractive server (NNI_URL, default 127.0.0.1:1527)
and runs on --probe-runs only; applying probe candidates is union/difference of their regions, an
approximation of sequential clicking (clicks interact; the real result after accepting several can differ).

Metrics per (state, method): FN-recall@k (k=1,3,8: share of missed GT volume covered by add-candidates among the
top k), candidate precision (add: share inside GT; remove: share outside GT), oracle-accept Dice (apply exactly
the candidates a perfect chooser accepts: add precision > 0.5, remove precision > 0.5), accept-all Dice,
accept-top-1 Dice, candidate count and compute time (body mask excluded: computed once per case).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

from segd import candidates as C
from segd import marks as M
from segd.data import PROJECT_ROOT, get_case, load_image, load_mask, read_cases

RUNS = ["phase1-01", "phase1-02", "phase1-03", "phase1-04", "phase1-05", "phase1-06-high",
        "phase1-07-xhigh", "phase1-09-xhigh-ex"]
CHEAP = ["intensity", "supervoxel", "ends", "combo"]
RENDER_STATES = [("phase1-07-xhigh", c) for c in ("M33", "N34", "F32", "V65")]
NNI_URL = os.environ.get("NNI_URL", "http://127.0.0.1:1527")
KS = (1, 3, 8)
FIELDS = ["run", "case", "histotype", "method", "start_dice", "start_recall", "start_precision", "gt_ml",
          "mask_ml", "undersegmented", "n_cand", "n_add", "n_remove", "fn_recall1", "fn_recall3", "fn_recall8",
          "add_prec_mean", "add_prec_gt50_frac", "rem_prec_mean", "rem_prec_gt50_frac", "oracle_dice",
          "all_dice", "top1_dice", "n_oracle_accept", "time_s", "probe_latency_mean_s"]


# ------------------------------------------------------------------------------------------ helpers
def parse_spec(spec: str) -> tuple[str, dict]:
    name, _, rest = spec.partition(":")
    params = {}
    for kv in filter(None, rest.split(",")):
        k, v = kv.split("=")
        try:
            params[k] = int(v)
        except ValueError:
            try:
                params[k] = float(v)
            except ValueError:
                params[k] = {"true": True, "false": False}.get(v.lower(), v)
    return name, params


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = int(a.sum()) + int(b.sum())
    return 2.0 * int((a & b).sum()) / s if s else 1.0


def load_final(run: str, case: str) -> np.ndarray | None:
    p = PROJECT_ROOT / "results" / run / case / "final_mask.nii.gz"
    if not p.exists():
        return None
    return np.asanyarray(nib.load(str(p)).dataobj) > 0


def evaluate(mask: np.ndarray, gt: np.ndarray, cands: list, vox_ml: float) -> dict:
    """All metrics for one candidate list, computed on the union bbox of mask, GT and candidates."""
    boxes = [C._box_of(mask | gt)] + [c.bbox for c in cands]
    lo = [min(b[a].start for b in boxes) for a in range(3)]
    hi = [max(b[a].stop for b in boxes) for a in range(3)]
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    m, g = mask[sl], gt[sl]
    regs = []
    for c in cands:
        r = np.zeros(m.shape, dtype=bool)
        r[tuple(slice(s.start - l, s.stop - l) for s, l in zip(c.bbox, lo))] = c.crop
        regs.append(r)
    miss = g & ~m
    nmiss = int(miss.sum())
    out = {"n_cand": len(cands), "n_add": sum(c.kind == "add" for c in cands),
           "n_remove": sum(c.kind == "remove" for c in cands)}
    adds = [(c, r) for c, r in zip(cands, regs) if c.kind == "add"]
    for k in KS:
        if nmiss == 0:
            out[f"fn_recall{k}"] = float("nan")
            continue
        cov = np.zeros(m.shape, dtype=bool)
        for _, r in adds[:k]:
            cov |= r
        out[f"fn_recall{k}"] = float((cov & miss).sum()) / nmiss
    prec = []
    for c, r in zip(cands, regs):
        n = int(r.sum())
        inside = int((r & g).sum()) / n
        prec.append(inside if c.kind == "add" else 1.0 - inside)
    ap = [p for c, p in zip(cands, prec) if c.kind == "add"]
    rp = [p for c, p in zip(cands, prec) if c.kind == "remove"]
    out["add_prec_mean"] = float(np.mean(ap)) if ap else float("nan")
    out["add_prec_gt50_frac"] = float(np.mean([p > 0.5 for p in ap])) if ap else float("nan")
    out["rem_prec_mean"] = float(np.mean(rp)) if rp else float("nan")
    out["rem_prec_gt50_frac"] = float(np.mean([p > 0.5 for p in rp])) if rp else float("nan")

    def apply(idx):
        o = m.copy()
        for i in idx:
            if cands[i].kind == "add":
                o |= regs[i]
        for i in idx:
            if cands[i].kind == "remove":
                o &= ~regs[i]
        return o

    good = [i for i, p in enumerate(prec) if p > 0.5]
    out["oracle_dice"] = dice(apply(good), g)
    out["n_oracle_accept"] = len(good)
    out["all_dice"] = dice(apply(range(len(cands))), g)
    out["top1_dice"] = dice(apply([0] if cands else []), g)
    return out


def case_states(case: str, runs: list[str]):
    row = get_case(case)
    img, aff, sp = load_image(row)
    gt = load_mask(row, aff, img.shape) > 0
    return row, img, sp, gt


# ------------------------------------------------------------------------------------------ workers
def _state_rows(run, row, img, sp, gt, mask, specs, body, nni, keep: dict | None):
    vox_ml = float(np.prod(sp)) / 1000.0
    inter = int((mask & gt).sum())
    base = {"run": run, "case": row.case, "histotype": row.histotype,
            "start_dice": dice(mask, gt), "start_recall": inter / max(int(gt.sum()), 1),
            "start_precision": inter / max(int(mask.sum()), 1),
            "gt_ml": gt.sum() * vox_ml, "mask_ml": mask.sum() * vox_ml}
    base["undersegmented"] = int(base["start_recall"] < 0.7)
    rows = []
    cache = {"body": body}                   # per state: body + supervoxels shared across method specs
    for spec in specs:
        name, params = parse_spec(spec)
        t0 = time.monotonic()
        cands = C.propose(img, mask, sp, method=name, nni=nni, cache=cache, **params)
        secs = time.monotonic() - t0
        r = dict(base, method=spec, time_s=secs)
        r.update(evaluate(mask, gt, cands, vox_ml))
        lat = [c.how["latency_s"] for c in cands if "latency_s" in c.how]
        r["probe_latency_mean_s"] = float(np.mean(lat)) if lat else float("nan")
        rows.append(r)
        if keep is not None:
            keep[spec] = cands
    return rows


def run_case_cheap(args):
    case, runs, specs, render_dir = args
    row, img, sp, gt = case_states(case, runs)
    body = M.body_mask(img)
    rows, skipped = [], []
    for run in runs:
        mask = load_final(run, case)
        if mask is None:
            continue
        if not (mask & gt).any():
            skipped.append({"run": run, "case": case, "reason": "no overlap with GT"})
            continue
        keep = {} if (render_dir and (run, case) in RENDER_STATES) else None
        rows += _state_rows(run, row, img, sp, gt, mask, specs, body, None, keep)
        if keep:
            render_state(render_dir, run, row, img, sp, gt, mask, keep)
    return rows, skipped


def run_case_probe(args):
    case, runs, specs, render_dir = args
    row, img, sp, gt = case_states(case, runs)
    body = M.body_mask(img)
    nni = C.NNIProbe(NNI_URL, img, sp, body=body)
    rows, skipped = [], []
    try:
        for run in runs:
            mask = load_final(run, case)
            if mask is None:
                continue
            if not (mask & gt).any():
                skipped.append({"run": run, "case": case, "reason": "no overlap with GT"})
                continue
            keep = {} if (render_dir and (run, case) in RENDER_STATES) else None
            rows += _state_rows(run, row, img, sp, gt, mask, specs, body, nni, keep)
            if keep:
                render_state(render_dir, run, row, img, sp, gt, mask, keep)
    finally:
        nni.close()
    return rows, skipped


# ------------------------------------------------------------------------------------------ rendering
PALETTE = [(255, 214, 0), (0, 229, 255), (255, 61, 143), (167, 111, 255), (255, 152, 0), (120, 200, 255),
           (255, 255, 255), (200, 120, 60)]
GT_COLOR = (255, 255, 255)


def render_state(out_dir, run, row, img, sp, gt, mask, keep: dict) -> None:
    """3-plane image per method: current mask green outline, candidates as filled colour + label (add: C1..,
    remove: red-dashed id with 'x'), GT as thin white outline (offline sanity check only)."""
    from PIL import Image, ImageDraw
    from segd import render as R
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec, cands in keep.items():
        if not cands:
            continue
        ci = [int(round(v)) for v in np.mean([[(s.start + s.stop - 1) / 2 for s in c.bbox] for c in cands[:1]], 0)]
        com = [int(round(v)) for v in np.array(np.nonzero(mask)).mean(1)]
        panels = []
        for axis in ("axial", "coronal", "sagittal"):
            ax_i = M.AXIS_INDEX[axis]
            idx = ci[ax_i] if len(cands) else com[ax_i]
            union = mask.copy()
            for c in cands:
                union[c.bbox] |= c.crop
            rows_, cols_ = M.slice_shape(img.shape, axis)
            u2 = M.extract_slice(union | gt, axis, idx)
            crop = R._resolve_crop("mask", rows_, cols_, u2) if u2.any() else (0, 0, rows_, cols_)
            panel, info = R.render_view(img, axis, idx, crop=crop, overlay="outline", mask=mask,
                                        spacing=sp, single_panel=True)
            g = info.geom(0)
            panel = R._overlay_mask(panel, M.extract_slice(gt, axis, idx), g.crop, g.scale, "outline", GT_COLOR)
            d = ImageDraw.Draw(panel)
            for n, c in enumerate(cands):
                sub = np.zeros(img.shape[:3], dtype=bool)
                sub[c.bbox] = c.crop
                m2 = M.extract_slice(sub, axis, idx)
                if not m2.any():
                    continue
                col = PALETTE[n % len(PALETTE)] if c.kind == "add" else R.REMOVED_COLOR
                panel = R._overlay_mask(panel, m2, g.crop, g.scale, "fill", col)
                d = ImageDraw.Draw(panel)
                rr, cc = np.nonzero(m2)
                x = (cc.mean() + 0.5 - g.crop[1]) * g.scale
                y = (rr.mean() + 0.5 - g.crop[0]) * g.scale
                tag = c.id + ("" if c.kind == "add" else " x")
                d.text((x - 10, y - 8), tag, fill=(0, 0, 0), font=R._font(20), stroke_width=2, stroke_fill=col)
            panels.append(panel)
        h = max(p.height for p in panels)
        w = sum(p.width for p in panels) + 16 * (len(panels) - 1)
        canvas = Image.new("RGB", (w, h + 34), (0, 0, 0))
        x = 0
        for p in panels:
            canvas.paste(p, (x, 34))
            x += p.width + 16
        legend = " | ".join(f"{c.id} {c.kind} {c.volume_ml:.0f}mL" for c in cands)
        ImageDraw.Draw(canvas).text((6, 6), f"{row.case} {run} [{spec}] green=mask white=GT | {legend}"[:230],
                                    fill=(255, 255, 255), font=R._font(16))
        safe = spec.replace(":", "_").replace(",", "_").replace("=", "")
        canvas.save(out_dir / f"{row.case}_{run}_{safe}.png")


# ------------------------------------------------------------------------------------------ aggregate
def _nanmean(v):
    v = [x for x in v if x == x]
    return float(np.mean(v)) if v else float("nan")


def summarize(rows: list[dict], skipped: list[dict], n_total_states: int) -> str:
    by = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)
    methods = list(by)

    def table(sel, title):
        lines = [f"### {title}", "",
                 "| method | states | cand/state | FN-rec@1 | @3 | @8 | add prec | add prec>.5 | rem prec | start Dice | "
                 "oracle-accept Dice | gain | accept-all Dice | accept-top1 Dice | time/state s |",
                 "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for m in methods:
            rs = [r for r in by[m] if sel(r)]
            if not rs:
                continue
            f = lambda k: _nanmean([r[k] for r in rs])
            lines.append(
                f"| {m} | {len(rs)} | {f('n_cand'):.1f} | {f('fn_recall1'):.3f} | {f('fn_recall3'):.3f} | "
                f"{f('fn_recall8'):.3f} | {f('add_prec_mean'):.3f} | {f('add_prec_gt50_frac'):.3f} | "
                f"{f('rem_prec_mean'):.3f} | {f('start_dice'):.3f} | {f('oracle_dice'):.3f} | "
                f"{f('oracle_dice') - f('start_dice'):+.3f} | {f('all_dice'):.3f} | {f('top1_dice'):.3f} | "
                f"{f('time_s'):.1f} |")
        return lines + [""]

    # common subset for fair comparison with probe (states evaluated for every method)
    keyset = [{(r["run"], r["case"]) for r in by[m]} for m in methods]
    common = set.intersection(*keyset) if keyset else set()
    lines = ["# Candidate-correction benchmark", "",
             f"States: {n_total_states} requested, {len(skipped)} skipped (final mask has no overlap with GT = "
             f"localization failure), {len({(r['run'], r['case']) for r in rows})} evaluated for at least one method.",
             "Oracle-accept = apply exactly the candidates a perfect chooser accepts (add: precision > .5, remove: "
             "> .5 outside GT). Probe candidates are applied as union/difference of the predicted regions "
             "(approximation of sequential clicking). Time excludes the body mask (once per case).", ""]
    lines += table(lambda r: True, "All evaluated states (each method on its own state set)")
    lines += table(lambda r: r["undersegmented"] == 1, "Under-segmented states (start recall < 0.7)")
    if common and len(common) < max(len(k) for k in keyset):
        lines += table(lambda r: (r["run"], r["case"]) in common,
                       f"Common subset evaluated by all methods ({len(common)} states)")
        lines += table(lambda r: (r["run"], r["case"]) in common and r["undersegmented"] == 1,
                       "Common subset, under-segmented")
    for h in "FLMNV":
        lines += table(lambda r, h=h: r["histotype"] == h and r["undersegmented"] == 1,
                       f"Histotype {h}, under-segmented")
    lines += ["### Per-method gain distribution (oracle-accept Dice - start Dice)", "",
              "| method | share of states gaining > .02 | share losing > .02 | median gain | max gain |", "|---|---|---|---|---|"]
    for m in methods:
        g = np.array([r["oracle_dice"] - r["start_dice"] for r in by[m]])
        lines.append(f"| {m} | {np.mean(g > .02):.2f} | {np.mean(g < -.02):.2f} | {np.median(g):+.3f} | {g.max():+.3f} |")
    lines += ["", "Skipped states (no GT overlap): " + (", ".join(f"{s['run']}/{s['case']}" for s in skipped) or "none")]
    return "\n".join(lines) + "\n"


def merge(files: list[str], out: Path) -> None:
    rows = []
    for fn in files:
        with open(fn) as f:
            for r in csv.DictReader(f):
                rows.append({k: (float(v) if k not in ("run", "case", "histotype", "method") else v)
                             for k, v in r.items()})
    rows.sort(key=lambda r: (r["method"], r["run"], r["case"]))
    with open(out / "per_state.csv", "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    all_states = {(r["run"], r["case"]) for r in rows}
    n_total = len(RUNS) * len(read_cases("design"))
    skipped = [{"run": run, "case": c.case} for run in RUNS for c in read_cases("design")
               if (run, c.case) not in all_states]
    (out / "summary.md").write_text(summarize(rows, skipped, n_total))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=",".join(RUNS))
    ap.add_argument("--cases", default="")
    ap.add_argument("--methods", default=";".join(CHEAP))
    ap.add_argument("--probe-runs", default="", help="runs to evaluate with the 'probe' method (needs nnInteractive)")
    ap.add_argument("--probe-specs", default="probe")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "candidate_bench"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--probe-workers", type=int, default=3)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--merge", default="", help="comma-separated per_state csv files: merge them into "
                    "<out>/per_state.csv and rewrite <out>/summary.md (no computation)")
    ap.add_argument("--tag", default="", help="suffix for output file names")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.merge:
        merge(a.merge.split(","), out)
        return
    runs = [r for r in a.runs.split(",") if r]
    cases = [c for c in a.cases.split(",") if c] or [r.case for r in read_cases("design")]
    specs = [m for m in a.methods.split(";") if m]
    render_dir = out / "renders" if a.render else None
    t0 = time.time()
    rows, skipped = [], []
    if specs:
        jobs = [(c, runs, specs, render_dir) for c in cases]
        with ProcessPoolExecutor(a.workers) as ex:
            for r, s in ex.map(run_case_cheap, jobs):
                rows += r
                skipped += s
        print(f"cheap methods done in {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    probe_runs = [r for r in a.probe_runs.split(",") if r]
    if probe_runs:
        t1 = time.time()
        pspecs = [m for m in a.probe_specs.split(";") if m]
        jobs = [(c, probe_runs, pspecs, render_dir) for c in cases]
        with ThreadPoolExecutor(a.probe_workers) as ex:
            for r, s in ex.map(run_case_probe, jobs):
                rows += r
                if not specs:
                    skipped += s
        print(f"probe done in {time.time() - t1:.0f}s", file=sys.stderr, flush=True)
    rows.sort(key=lambda r: (r["method"], r["run"], r["case"]))
    with open(out / f"per_state{a.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    n_total = len(runs) * len(cases)
    uniq_skipped = {(s["run"], s["case"]): s for s in skipped}
    md = summarize(rows, list(uniq_skipped.values()), n_total)
    (out / f"summary{a.tag}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
