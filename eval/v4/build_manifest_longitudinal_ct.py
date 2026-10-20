"""Build eval/v4/longitudinal_ct_cases.csv (autoPET/CT-IV Longitudinal-CT, melanoma).

Source: data/phase4/longitudinal_ct/Longitudinal_CT_v3.zip (download_longitudinal_ct.py),
unzipped to inputsTr/ + targetsTr/. 300 patients, baseline (BL) + follow-up (FU) whole-body
CT, 3 mm slices. Mask files are INTEGER label volumes (one label per malignant lesion);
lesion metadata in per-patient CSVs (inputsTr/<pid>.csv) and Grand-Challenge JSON COGs.

We use the BASELINE (BL) scan of each patient; follow-up scans are a phase-5 option.
Split: BY PATIENT - 30 design patients, 170 test patients (fixed random subset, seed 2026),
100 patients unused (very large dataset -> fixed test subset per Phase 4 plan). The
use_for_challenge flag (autoPET/CT-IV challenge subset) is VERIFIED ON DATA to be set for
(at least one lesion of) all 300 patients, so it cannot be used for split hygiene; it is
documented as an earlier-exposure caveat in research/phase4_datasets.md instead.

Run: uv run --project harness/server python eval/v4/build_manifest_longitudinal_ct.py [dest_dir]
"""
import csv
import random
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

from phase4_common import dataset_dir, lesion_stats, write_manifest

TARGET_TEXT = "malignant melanoma lesions (all malignant lesions: primary and metastases)"

src = dataset_dir("longitudinal_ct")
out_csv = Path(__file__).resolve().parent / "longitudinal_ct_cases.csv"
root = src / "Longitudinal_CT_v3" if (src / "Longitudinal_CT_v3").exists() else src
if not root.exists() or not (root / "inputsTr").exists():
    z = src / "Longitudinal_CT_v3.zip"
    if not z.exists():
        raise SystemExit(f"missing: {z} (run download_longitudinal_ct.py first)")
    print("unzipping", z, "->", src)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(src)
    root = src / "Longitudinal_CT_v3" if (src / "Longitudinal_CT_v3").exists() else src

inputs = sorted((root / "inputsTr").glob("*_BL_img_*.nii.gz"))
if not inputs:
    raise SystemExit(f"no baseline images found in {root / 'inputsTr'}")

# per-patient challenge flag from the lesion CSVs (per-lesion flag; a patient counts as
# exposed if ANY of its lesions was in the autoPET/CT-IV challenge subset).
# VERIFIED ON DATA: all 300 patients have >=1 challenge lesion, so the flag CANNOT be
# used to hold design back; it is reported in phase4_datasets.md instead.
challenge = {}
csvs = list((root / "inputsTr").glob("*.csv"))
for cpath in csvs:
    pid = cpath.name.removesuffix(".csv")
    with open(cpath) as f:
        challenge[pid] = any(r.get("use_for_challenge", "").lower() == "true"
                             for r in csv.DictReader(f))

rows_by_patient = defaultdict(list)
for img_path in inputs:
    pid = img_path.name.split("_")[0]
    stem = img_path.name.split("_img_")[0]
    idx = img_path.name.split("_img_")[1]  # e.g. '00.nii.gz' (index within the body-region split of one scan)
    case = f"{stem}_{idx.split('.')[0]}"  # UNIQUE: a scan can be split into several body-region volumes (_00,_01,..)
    mask_path = root / "inputsTr" / f"{stem}_mask_{idx}"
    if not mask_path.exists():
        mask_path = root / "targetsTr" / f"{stem}_mask_{idx}"
    if not mask_path.exists():
        raise FileNotFoundError(f"{stem}_mask_{idx} in inputsTr/ or targetsTr/")
    import nibabel as nib
    import numpy as np
    m = np.asarray(nib.load(str(mask_path)).dataobj)
    sp = nib.load(str(mask_path)).header.get_zooms()[:3]
    n_les, vol_ml = lesion_stats(m, sp, use_labels=True)
    rows_by_patient[pid].append(dict(
        case=case,  # <pid>_<timepoint>_<region-idx>; body-region index keeps cases unique
        target_text=TARGET_TEXT, modality="CT", sequence="CT",
        image_path=str(img_path), mask_paths=str(mask_path),
        n_lesions=n_les, lesion_volume_ml=round(vol_ml, 2), split="test"))

rng = random.Random(2026)
pids = sorted(rows_by_patient)
design = set(rng.sample(pids, 30))
rest = [p for p in pids if p not in design]
test = set(rng.sample(rest, 170))
rows = []
for pid in pids:
    split = "design" if pid in design else ("test" if pid in test else "unused")
    for r in rows_by_patient[pid]:
        r["split"] = split
        rows.append(r)
write_manifest(out_csv, rows)
from collections import Counter
print("patients:", len(pids), "| challenge-exposed patients:", sum(challenge.get(p, False) for p in pids))
print(Counter(r["split"] for r in rows), "| scans:", len(rows), "->", out_csv)
