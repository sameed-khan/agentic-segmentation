"""Point the Phase 4 manifests at the 1 mm iso conversions (originals kept).

For every row of eval/v4/<dataset>_cases.csv: if
data/phase4/<dataset>/nifti_iso/image/<case>.nii.gz and mask/<case>.nii.gz exist,
set image_path/mask_paths to the iso files and move the native paths into the
image_path_src/mask_paths_src columns (provenance; originals are NOT deleted).
Rows without an iso pair keep their native paths (e.g. ucsd_bmets, blocked).

Run: uv run --project harness/server python eval/v4/repoint_manifests_to_iso.py [dataset ...]
"""
import sys
from pathlib import Path

from phase4_common import dataset_dir, read_manifest, write_manifest

here = Path(__file__).resolve().parent
datasets = sys.argv[1:] or ["beamster", "spinal_mm_seg", "ucsd_bmets", "longitudinal_ct"]
for ds in datasets:
    csv_path = here / f"{ds}_cases.csv"
    if not csv_path.exists():
        print(f"[{ds}] no manifest (skipped)")
        continue
    rows = read_manifest(csv_path)
    iso_root = dataset_dir(ds) / "nifti_iso"
    n = 0
    for r in rows:
        im = iso_root / "image" / f"{r['case']}.nii.gz"
        mk = iso_root / "mask" / f"{r['case']}.nii.gz"
        if im.exists() and mk.exists():
            if "image_path_src" not in r or not r["image_path_src"]:
                r["image_path_src"], r["mask_paths_src"] = r["image_path"], r["mask_paths"]
            r["image_path"], r["mask_paths"] = str(im), str(mk)
            n += 1
    write_manifest(csv_path, rows)
    print(f"[{ds}] {n}/{len(rows)} rows pointed at nifti_iso -> {csv_path}")
