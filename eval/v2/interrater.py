"""Pairwise inter-rater Dice for the multi-rater (J/K/N) test cases, and a run's Dice vs each rater and vs the
majority vote. Usage: uv run --project harness/server python eval/v2/interrater.py results/phase3-high"""
import csv, sys, itertools, statistics as st
from pathlib import Path
import numpy as np, nibabel as nib
ROOT = Path(__file__).resolve().parents[2]
run = Path(sys.argv[1])
def load(p): return np.asarray(nib.as_closest_canonical(nib.load(str(p))).dataobj) > 0.5
def dice(a, b): s = a.sum() + b.sum(); return 2 * (a & b).sum() / s if s else 1.0
rows = [r for r in csv.DictReader(open(ROOT / "eval/v2/mstt_cases.csv")) if r["split"] == "test" and len(r["masks_kaggle"].split(";")) > 1]
out = []
for r in rows:
    ms = [load(ROOT / "data/mstt199/mask" / Path(m).name) for m in r["masks_kaggle"].split(";")]
    mv = (np.sum(ms, 0) >= 2)
    pred = load(run / r["case"] / "final_mask.nii.gz")
    pw = [dice(a, b) for a, b in itertools.combinations(ms, 2)]
    out.append((r["case"], st.mean(pw), st.mean(dice(pred, m) for m in ms), dice(pred, mv)))
    print(f"{r['case']} interrater {out[-1][1]:.3f}  run-vs-raters {out[-1][2]:.3f}  run-vs-majority {out[-1][3]:.3f}", flush=True)
print(f"MEAN interrater {st.mean(o[1] for o in out):.3f} run-vs-raters {st.mean(o[2] for o in out):.3f} run-vs-majority {st.mean(o[3] for o in out):.3f} (n={len(out)})")
