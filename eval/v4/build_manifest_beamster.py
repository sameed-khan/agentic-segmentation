"""Build eval/v4/beamster_cases.csv for BEAMSTER (140 CE-T1 MRI, 260 lesions).

Source: data/phase4/beamster/Dataset_BEAMSTER/ (download_beamster.py) - flat files
Dataset_{A,B}_YYY.nii.gz (image) + Dataset_{A,B}_YYY_segm.nii.gz (binary mask).
Split: 30 design / 110 test, seed 2026, stratified by release half (A=24, B=6 design
slots; B is enriched with small metastases) and by lesion count (single vs multiple).

Run: uv run --project harness/server python eval/v4/build_manifest_beamster.py [dest_dir]
"""
import random
import sys
from pathlib import Path

from phase4_common import dataset_dir, load_nii, lesion_stats, write_manifest

TARGET_TEXT = "brain metastases (all enhancing lesions on CE-T1 MRI)"

src = dataset_dir("beamster")
out_csv = Path(__file__).resolve().parent / "beamster_cases.csv"
root = src / "Dataset_BEAMSTER" if (src / "Dataset_BEAMSTER").exists() else src

rows = []
cases = sorted(p for p in root.glob("Dataset_*.nii.gz") if not p.name.endswith("_segm.nii.gz"))
assert cases, f"no BEAMSTER NIfTI files found in {root}"
for img_path in cases:
    seg_path = img_path.with_name(img_path.stem.replace(".nii", "") + "_segm.nii.gz")
    if not seg_path.exists():
        raise FileNotFoundError(seg_path)
    img, aff, sp = load_nii(img_path)          # small enough to just read spacing
    mask, _, _ = load_nii(seg_path)
    n_les, vol_ml = lesion_stats(mask, sp, use_labels=False)
    case = img_path.stem.replace(".nii", "")
    half = case.split("_")[1]
    rows.append(dict(case=case, target_text=TARGET_TEXT, modality="MRI", sequence="CE-T1",
                     image_path=str(img_path), mask_paths=str(seg_path),
                     n_lesions=n_les, lesion_volume_ml=round(vol_ml, 2), split="test"))

rng = random.Random(2026)
design = set()
for half, k in (("A", 24), ("B", 6)):  # B over-weighted: small-lesion stress cases
    for cls in (1, 2):  # 1 = single lesion, 2 = multiple lesions
        pool = sorted(r["case"] for r in rows
                      if r["case"].startswith(f"Dataset_{half}_")
                      and (r["n_lesions"] == 1) == (cls == 1))
        want = k // 2 if cls == 1 else k - k // 2
        design |= set(rng.sample(pool, min(want, len(pool))))
for r in rows:
    r["split"] = "design" if r["case"] in design else "test"
write_manifest(out_csv, rows)
from collections import Counter
print(Counter((r["split"], r["case"][8], r["n_lesions"] == 1) for r in rows))
print("cases:", len(rows), "->", out_csv)
