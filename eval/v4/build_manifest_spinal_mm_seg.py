"""Build eval/v4/spinal_mm_seg_cases.csv for Spinal-Multiple-Myeloma-SEG (DECT).

Source: data/phase4/spinal_mm_seg/nifti/<case>/ from eval/v4/spinal_nifti_from_dicom.py
(conventional CT series -> ct.nii.gz int16 HU; lesion DICOM-SEG -> lesion_mask.nii.gz
uint8 + lesion_labels.nii.gz uint16 + seg_summary.json). One case per SCAN (patients
with two scans get case suffixes _sc2...; 67 patients / 72 scans). n_lesions = number
of annotated lesion segments present in the mask; lesion_volume_ml = total lesion
volume at native spacing.

Split: BY PATIENT - 20 design patients / 47 test patients, seed 2026; patients with 2
scans stay entirely in one split (no leakage).

Run: uv run --project harness/server python eval/v4/build_manifest_spinal_mm_seg.py
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from phase4_common import dataset_dir, write_manifest

TARGET_TEXT = "multiple myeloma focal lesions in the spine (all lesions)"

src = dataset_dir("spinal_mm_seg")
out_csv = Path(__file__).resolve().parent / "spinal_mm_seg_cases.csv"
cases = sorted((src / "nifti").glob("*/ct.nii.gz"))
if not cases:
    raise SystemExit(f"no converted cases under {src / 'nifti'}; run spinal_nifti_from_dicom.py first")

rows_by_patient = defaultdict(list)
for ct in cases:
    cdir = ct.parent
    summary = json.loads((cdir / "seg_summary.json").read_text())
    case = cdir.name
    pid = case.split("_sc")[0]
    n_les = summary["n_lesions"]
    vol_ml = round(sum(summary["lesion_volumes_ml"]), 2)
    rows_by_patient[pid].append(dict(
        case=case, target_text=TARGET_TEXT, modality="CT", sequence="CT (conventional, konv)",
        image_path=str(cdir / "ct.nii.gz"),
        mask_paths=str(cdir / "lesion_labels.nii.gz"),  # integer labels; binary = mask > 0
        n_lesions=n_les, lesion_volume_ml=vol_ml, split="test"))

rng = random.Random(2026)
pids = sorted(rows_by_patient)
assert len(pids) == 67, f"expected 67 patients, found {len(pids)}"
design = set(rng.sample(pids, 20))
rows = []
for pid in pids:
    for r in rows_by_patient[pid]:
        r["split"] = "design" if pid in design else "test"
        rows.append(r)
write_manifest(out_csv, rows)
from collections import Counter
print("patients:", len(pids), "| design patients:", len(design))
print(Counter(r["split"] for r in rows), "| scans:", len(rows), "->", out_csv)
print("lesions per scan (design):", sorted(r["n_lesions"] for r in rows if r["split"] == "design"))
