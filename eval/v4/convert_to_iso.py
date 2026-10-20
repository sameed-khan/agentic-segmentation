"""Convert a Phase 4 dataset to the harness's 1 mm isotropic canonical-RAS NIfTI layout.

MSTT-199 is distributed as 1 mm isotropic; the harness assumes it (segd/marks.py
MarkSet.visible slice windowing is in 1 mm voxels, CONTRACTS §2; render scale bars in mm).
Phase 4 sources are NOT isotropic (BEAMSTER ~1.0x1.0x1.5 mm MRI; whole-body CT 3 mm
slices; spinal DECT conv CT ~0.5-1.0 mm anisotropic), so cases are resampled to 1 mm
isotropic in a separate conversion step:
  image  -> data/phase4/<dataset>/nifti_iso/image/<case>.nii.gz   (linear, float32)
  mask   -> data/phase4/<dataset>/nifti_iso/mask/<case>.nii.gz    (nearest neighbour)

Mask semantics (keep integer labels for per-lesion scoring, phase4_harness_changes.md §5):
  - longitudinal_ct masks are INTEGER per-lesion label volumes -> converted as uint16,
    preserving lesion IDs (nearest-neighbour resampling keeps labels intact).
  - binary masks (beamster, spinal_mm_seg) -> uint8 (mask > 0).
Resampling of the mask uses the IMAGE's target grid so image and mask stay co-registered.

Usage: uv run --project harness/server python eval/v4/convert_to_iso.py <dataset> [design|test|all] [--jobs N]
"""
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from phase4_common import dataset_dir, read_manifest

LABELLED = {"longitudinal_ct", "spinal_mm_seg"}  # masks with integer per-lesion labels


def convert_one(row: dict, dataset: str, out_root: Path) -> str:
    import nibabel as nib
    from phase4_common import load_nii, resample_to_iso, save_nii

    img_path = Path(row["image_path"])
    out_img = out_root / "image" / f"{row['case']}.nii.gz"
    out_mask = out_root / "mask" / f"{row['case']}.nii.gz"
    if out_img.exists() and out_mask.exists():
        return f"skip {row['case']} (exists)"
    img, aff, sp = load_nii(img_path)                      # canonical RAS float32
    if out_img.exists():
        # image already converted: reuse its saved grid for the mask (never recompute)
        iso_aff = nib.load(str(out_img)).affine
    else:
        iso_img, iso_aff = resample_to_iso(img, aff, order=1)
        save_nii(out_img, iso_img, iso_aff, np.float32)
    mcan = nib.as_closest_canonical(nib.load(row["mask_paths"]))
    mask = np.asarray(mcan.dataobj)
    if dataset in LABELLED:
        iso_mask, _ = resample_to_iso(mask.astype(np.uint16), mcan.affine, order=0)
        save_nii(out_mask, iso_mask, iso_aff, np.uint16)
    else:
        # binary mask: linear interp + 0.5 threshold (anti-aliased downsample; plain
        # nearest at 1 mm from ~0.5 mm voxels decimates small lesions)
        iso_mask, _ = resample_to_iso((mask > 0).astype(np.float32), mcan.affine, order=1)
        save_nii(out_mask, (iso_mask >= 0.5).astype(np.uint8), iso_aff, np.uint8)
    return (f"{row['case']}: {img.shape}@{tuple(round(s, 2) for s in sp)} -> "
            f"{out_img.stat().st_size // (1024*1024)} MB img @1.0 iso")


def main() -> None:
    argv = sys.argv[1:]
    jobs = int(argv[argv.index("--jobs") + 1]) if "--jobs" in argv else 1
    dataset = argv[0] if argv else "beamster"
    split_arg = argv[1] if len(argv) > 1 and not argv[1].startswith("--") else "design"
    here = Path(__file__).resolve().parent
    rows = read_manifest(here / f"{dataset}_cases.csv")
    rows = [r for r in rows if split_arg == "all" or r["split"] == split_arg]
    if not rows:
        raise SystemExit(f"no rows for {dataset} split={split_arg}; build the manifest first")
    out_root = dataset_dir(dataset) / "nifti_iso"
    n_ok = 0
    if jobs <= 1:
        for r in rows:
            print(convert_one(r, dataset, out_root), flush=True)
            n_ok += 1
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(convert_one, r, dataset, out_root): r for r in rows}
            for fut in as_completed(futs):
                print(fut.result(), flush=True)
                n_ok += 1
    print("converted", n_ok, "cases ->", out_root, flush=True)


if __name__ == "__main__":
    main()
