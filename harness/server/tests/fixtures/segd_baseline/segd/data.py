"""Case loading for MSTT-199 (isotropic). Single source of truth for paths and array conventions.

Conventions (used by every segd module):
- Arrays are indexed [i, j, k] = voxel indices of the loaded NIfTI after nibabel `as_closest_canonical`.
- MSTT-199 isotropic files carry an identity affine (1 mm, nominal RAS). The true anatomical orientation of
  the axes is recorded in ORIENTATION below after visual verification (see eval/v2/data_check.md).
- Images are float32; masks are uint8 {0,1}. Multi-rater masks -> voxel-wise majority vote.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "mstt199"
CASES_CSV = PROJECT_ROOT / "eval" / "v2" / "mstt_cases.csv"

HISTOTYPE_TEXT = {
    "F": "fibrous soft-tissue tumor",
    "L": "lipomatous soft-tissue tumor",
    "M": "myxoid soft-tissue tumor",
    "N": "peripheral nerve sheath tumor",
    "V": "vascular soft-tissue tumor",
}


@dataclass
class CaseRow:
    case: str
    histotype: str
    sequence: str
    multi_rater: bool
    split: str
    image_path: Path
    mask_paths: list[Path] = field(default_factory=list)

    @property
    def target_text(self) -> str:
        return HISTOTYPE_TEXT[self.histotype]


@dataclass
class CaseData:
    row: CaseRow
    image: np.ndarray          # float32 [i,j,k]
    affine: np.ndarray         # 4x4, canonical
    spacing: tuple[float, float, float]
    mask: np.ndarray | None    # uint8 [i,j,k] or None if not loaded


@dataclass
class CaseGeometry:
    """Image + geometry needed for export: when the source NIfTI is not canonical/RAS,
    `orig_affine`/`inv_ornt` describe the canonical->source orientation transform so
    final_mask.nii.gz can be written in the ORIGINAL file geometry (finding #14).
    Both are None when the source was already RAS (no reorientation happened)."""
    image: np.ndarray                     # float32 [i,j,k], canonical
    affine: np.ndarray                    # 4x4, canonical
    spacing: tuple[float, float, float]
    orig_affine: np.ndarray | None
    inv_ornt: np.ndarray | None


def read_cases(split: str | None = None) -> list[CaseRow]:
    rows = []
    with open(CASES_CSV) as f:
        for r in csv.DictReader(f):
            if split and r["split"] != split:
                continue
            img = DATA_DIR / "image" / Path(r["image_kaggle"]).name
            masks = [DATA_DIR / "mask" / Path(m).name for m in r["masks_kaggle"].split(";") if m]
            rows.append(CaseRow(r["case"], r["histotype"], r["sequence"], r["multi_rater"] == "1",
                                r["split"], img, masks))
    return rows


def get_case(case: str) -> CaseRow:
    for r in read_cases():
        if r.case == case:
            return r
    raise KeyError(case)


def _load_canonical(path: Path) -> nib.Nifti1Image:
    return nib.as_closest_canonical(nib.load(str(path)))


def load_case_geometry(row: CaseRow) -> CaseGeometry:
    """Load the image canonicalised (nibabel as_closest_canonical, CONTRACTS §1/§5.6) and
    remember the source geometry: for a non-RAS source, final masks are mapped back to
    the source orientation at save time (finding #14)."""
    from nibabel.orientations import (
        axcodes2ornt, io_orientation, ornt_transform,
    )
    im = nib.load(str(row.image_path))
    ornt = io_orientation(im.affine)
    ras = axcodes2ornt(("R", "A", "S"))
    canon = nib.as_closest_canonical(im)
    arr = np.asarray(canon.dataobj, dtype=np.float32)
    spacing = tuple(float(s) for s in canon.header.get_zooms()[:3])
    if np.array_equal(ornt_transform(ornt, ras), ras):   # already RAS: no reorientation
        return CaseGeometry(arr, canon.affine.copy(), spacing, None, None)
    # canonical -> source transform: applying it to the canonical array restores the
    # original array and shape (verified by round-trip tests)
    inv = ornt_transform(ras, ornt)
    return CaseGeometry(arr, canon.affine.copy(), spacing, im.affine.copy(), inv)


def load_image(row: CaseRow) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    g = load_case_geometry(row)
    return g.image, g.affine, g.spacing


def load_mask(row: CaseRow, ref_affine: np.ndarray | None = None, ref_shape=None) -> np.ndarray:
    """GT mask (uint8). OFFLINE USE ONLY: never call from code paths reachable by the agent."""
    ms = []
    for p in row.mask_paths:
        im = _load_canonical(p)
        if ref_affine is not None and not np.allclose(im.affine, ref_affine, atol=1e-3):
            raise ValueError(f"{p.name}: affine mismatch with image")
        a = np.asarray(im.dataobj) > 0.5
        if ref_shape is not None and a.shape != tuple(ref_shape):
            raise ValueError(f"{p.name}: shape {a.shape} != image {ref_shape}")
        ms.append(a)
    if len(ms) == 1:
        return ms[0].astype(np.uint8)
    return (np.sum(ms, axis=0) * 2 > len(ms)).astype(np.uint8)


def load_case(case: str, with_mask: bool = False) -> CaseData:
    row = get_case(case)
    img, aff, sp = load_image(row)
    mask = load_mask(row, aff, img.shape) if with_mask else None
    return CaseData(row, img, aff, sp, mask)
