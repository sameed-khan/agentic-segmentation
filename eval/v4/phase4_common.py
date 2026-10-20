"""Shared helpers for Phase 4 dataset preparation (eval/v4).

Data lives OUTSIDE git at /home/ubuntu/agentic-segmentation/data/phase4/<dataset>/
(never commit it; the main tree must not be modified except this data dir, per task).
Manifests are CSVs with the harness-compatible columns pinned by Phase 4:
  case, target_text, modality, sequence, image_path, mask_paths (';'-separated),
  n_lesions, lesion_volume_ml, split (design/test/unused)

MSTT (eval/v2) is 1 mm isotropic by construction; Phase 4 datasets are NOT (BEAMSTER
~1x1x1.5 mm, whole-body CT 3 mm slices), and the harness hard-assumes 1 mm in places
(segd/marks.py visible() windowing, CONTRACTS §2). eval/v4/convert_to_iso.py resamples
to 1 mm isotropic canonical RAS as a separate conversion step -> nifti_iso/.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

PHASE4_ROOT = Path("/home/ubuntu/agentic-segmentation/data/phase4")
COLUMNS = ["case", "target_text", "modality", "sequence", "image_path",
           "mask_paths", "n_lesions", "lesion_volume_ml", "split",
           "image_path_src", "mask_paths_src"]  # *_src: native (pre-iso) paths, kept for provenance


def dataset_dir(name: str) -> Path:
    return PHASE4_ROOT / name


def write_manifest(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLUMNS})


def read_manifest(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def load_nii(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Canonicalised image array (float32), affine, spacing mm."""
    import nibabel as nib
    im = nib.as_closest_canonical(nib.load(str(path)))
    return (np.asarray(im.dataobj, dtype=np.float32), im.affine.copy(),
            tuple(float(s) for s in im.header.get_zooms()[:3]))


def lesion_stats(mask: np.ndarray, spacing, use_labels: bool) -> tuple[int, float]:
    """(number of lesions, total lesion volume mL).

    use_labels=True: mask is an integer per-lesion label volume (autoPET Longitudinal-CT)
    -> count distinct labels. use_labels=False: binary mask -> 26-connected components
    (scipy default connectivity for ndi.label is 1, matching eval/v2/data_check.py).
    """
    from scipy import ndimage as ndi
    if use_labels:
        labels = np.unique(mask)
        labels = labels[labels > 0]
        return len(labels), float((mask > 0).sum() * np.prod(spacing) / 1000)
    lab, n = ndi.label(mask > 0)
    return int(n), float((mask > 0).sum() * np.prod(spacing) / 1000)


def resample_to_iso(img: np.ndarray, affine: np.ndarray, order: int,
                    target_spacing: float = 1.0):
    """Resample a canonical-RAS volume to a target-spacing isotropic grid. order=1 image
    (linear), order=0 mask (nearest). Returns (array, new_affine).

    Grid derivation: the voxel extent along axis i is shape[i]*zoom[i] mm (zoom = diagonal
    of the canonical affine); at target spacing s the new voxel count is shape*zoom/s and
    the new affine columns are the OLD columns scaled to step s, i.e. col_i/zoom_i*s
    (BUG FIX vs the first draft, which multiplied the columns by zoom instead — that
    compressed the world box by zoom and mis-scaled every converted volume)."""
    import nibabel as nib
    from nibabel.processing import resample_from_to
    im = nib.Nifti1Image(img, affine)
    zoom = np.abs(np.diag(affine[:3, :3])).astype(float)
    shape = np.asarray(img.shape, dtype=float)
    target_shape = tuple(int(x) for x in np.round(shape * zoom / target_spacing))
    target_affine = affine.copy()
    target_affine[:3, :3] = affine[:3, :3] @ np.diag(target_spacing / zoom)
    out = resample_from_to(im, (target_shape, target_affine), order=order)
    return np.asarray(out.dataobj), out.affine


def save_nii(path: Path, arr: np.ndarray, affine: np.ndarray, dtype) -> None:
    import nibabel as nib
    path.parent.mkdir(parents=True, exist_ok=True)
    im = nib.Nifti1Image(arr.astype(dtype), affine)
    im.header.set_zooms((1.0, 1.0, 1.0))
    nib.save(im, str(path))


def preview(image: np.ndarray, mask: np.ndarray | None, out: Path, window=None) -> None:
    """3-panel orthogonal montage through the image (mask) centre, mask outlined red."""
    from PIL import Image
    from scipy import ndimage as ndi
    if window is None:
        sel = image[image > np.percentile(image, 5)] if (image > 0).any() else image
        window = tuple(np.percentile(sel, [0.5, 99.5]))
    lo, hi = window
    c = np.array(ndi.center_of_mass(mask if mask is not None and mask.any() else np.ones_like(image))).round().astype(int)
    panels = []
    for ax in range(3):
        sl = np.take(image, min(c[ax], image.shape[ax] - 1), axis=ax)
        g = (np.clip((sl - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)
        rgb = np.stack([g] * 3, -1)
        if mask is not None and mask.any():
            ms = np.take(mask, min(c[ax], mask.shape[ax] - 1), axis=ax) > 0
            edge = ms & ~ndi.binary_erosion(ms)
            rgb[edge] = [255, 0, 0]
        rgb = np.flipud(np.transpose(rgb, (1, 0, 2)))
        panels.append(Image.fromarray(rgb))
    h = max(p.height for p in panels)
    canvas = Image.new("RGB", (sum(p.width for p in panels) + 20, h), (40, 40, 40))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0)); x += p.width + 10
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
