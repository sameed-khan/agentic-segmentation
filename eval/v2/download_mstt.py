"""Download MSTT-199 isotropic NIfTI files for a split from Kaggle (CC0; anonymous API download).

Usage: uv run eval/v2/download_mstt.py [design|test|all] [dest_dir]
Layout:  <dest>/image/<case>_<seq>.nii   <dest>/mask/<case>_<seq>[_<rater>].nii
Files are float64 NIfTI with 1 mm isotropic spacing (as distributed). Idempotent.
"""
import csv, subprocess, sys, urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent
split = sys.argv[1] if len(sys.argv) > 1 else "design"
dest = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parents[1] / "data" / "mstt199"
BASE = "https://www.kaggle.com/api/v1/datasets/download/reasat/mstt-199/"

jobs = []
for r in csv.DictReader(open(HERE / "mstt_cases.csv")):
    if split != "all" and r["split"] != split:
        continue
    jobs.append((r["image_kaggle"], dest / "image" / Path(r["image_kaggle"]).name))
    for m in r["masks_kaggle"].split(";"):
        jobs.append((m, dest / "mask" / Path(m).name))

for i, (src, out) in enumerate(jobs, 1):
    if out.exists() and out.stat().st_size > 0:
        continue
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(".part")
    subprocess.run(["curl", "-sSfL", "--retry", "3", "-o", str(part), BASE + urllib.parse.quote(src, safe="")], check=True)
    part.rename(out)
    print(f"[{i}/{len(jobs)}] {out.name} {out.stat().st_size/1e6:.0f}MB", flush=True)
print("done", len(jobs), "files ->", dest)
