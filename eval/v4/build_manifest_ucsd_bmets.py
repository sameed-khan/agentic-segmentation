"""Build eval/v4/ucsd_bmets_cases.csv for UCSD-BMETS-Longitudinal (brain mets MRI).

Source: data/phase4/ucsd_bmets/ - the TCIA NIfTI zip unpacked (download_ucsd_bmets.py).
300 patients, 1,065 MRI timepoints; lesion segmentations in NIfTI. The archive layout
pairs each image volume with a neuroradiologist-approved lesion mask; we pair files by
name (mask filename contains "seg"), one case per image volume.

Split: BY PATIENT - 30 design patients, 170 test patients (fixed random subset, seed
2026), 100 patients unused (very large dataset -> fixed test subset per Phase 4 plan).
NOTE: the archive layout could not be verified at build time (dataset not downloaded -
Aspera-only distribution); if unpacked names differ, adjust IMG_GLOB/MASK_HINT below.

Run: uv run --project harness/server python eval/v4/build_manifest_ucsd_bmets.py [dest_dir]
"""
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

from phase4_common import dataset_dir, lesion_stats, write_manifest

TARGET_TEXT = "brain metastases (all metastatic lesions on MRI)"
IMG_GLOB = "*.nii.gz"
MASK_HINT = "seg"  # mask filenames must contain this, image filenames must not

src = dataset_dir("ucsd_bmets")
out_csv = Path(__file__).resolve().parent / "ucsd_bmets_cases.csv"
niftis = sorted(src.rglob(IMG_GLOB))
if not niftis:
    raise SystemExit(f"no NIfTI files under {src} (run download_ucsd_bmets.py first)")


def patient_of(p: Path) -> str:
    return p.name.split("_")[0]


rows_by_patient = defaultdict(list)
for p in niftis:
    if MASK_HINT in p.name.lower():
        continue
    # mask: same directory, same stem, containing the hint
    cands = [q for q in p.parent.glob(p.name.replace(".nii.gz", "") + "*.nii.gz")
             if MASK_HINT in q.name.lower()]
    if not cands:
        cands = [q for q in p.parent.glob("*.nii.gz")
                 if MASK_HINT in q.name.lower()
                 and re.sub(MASK_HINT, "", q.name.lower()) == p.name.lower().replace(".nii.gz", "")]
    if not cands:
        raise FileNotFoundError(f"no mask found for {p}")
    mask_path = cands[0]
    m = np.asarray(nib.load(str(mask_path)).dataobj)
    sp = nib.load(str(p)).header.get_zooms()[:3]
    n_les, vol_ml = lesion_stats(m, sp, use_labels=False)
    case = p.name.removesuffix(".nii.gz")
    rows_by_patient[patient_of(p)].append(dict(
        case=case, target_text=TARGET_TEXT, modality="MRI", sequence="CE-T1",
        image_path=str(p), mask_paths=str(mask_path),
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
print("patients:", len(pids))
print(Counter(r["split"] for r in rows), "| scans:", len(rows), "->", out_csv)
