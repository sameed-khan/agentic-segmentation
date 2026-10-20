"""Case loading for MSTT-199 (isotropic) and the Phase 4 manifest datasets.
Single source of truth for paths and array conventions.

Conventions (used by every segd module):
- Arrays are indexed [i, j, k] = voxel indices of the loaded NIfTI after nibabel `as_closest_canonical`.
- MSTT-199 isotropic files carry an identity affine (1 mm, nominal RAS). The true anatomical orientation of
  the axes is recorded in ORIENTATION below after visual verification (see eval/v2/data_check.md).
- Images are float32; masks are uint8 {0,1}. Multi-rater masks -> voxel-wise majority vote.

Phase 4 (BEAMSTER, `--dataset beamster`): cases come from manifest CSVs
`eval/v4/<dataset>_cases.csv` (columns `case, target_text, modality, sequence, image_path,
mask_paths, n_lesions, lesion_volume_ml, split, ...`; absolute paths). `n_lesions` and
`lesion_volume_ml` are GT-derived manifest columns for OFFLINE use (scorer, diagnostics);
they must never reach agent-visible paths (CaseState never receives them).

Path portability (A100 mount): the manifest builder recorded absolute paths under the
build machine's data root; paths starting with MANIFEST_DATA_ROOT are remapped onto
``data_root()`` = $SEGD_DATA_ROOT (default <project>/data). MSTT's DATA_DIR honours the
same variable.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
# Absolute data root of the machine that built the Phase 4 manifests (kept as a constant
# so $SEGD_DATA_ROOT remapping has one definition; MSTT paths are always PROJECT_ROOT-relative).
MANIFEST_DATA_ROOT = Path("/home/ubuntu/agentic-segmentation/data")


def data_root() -> Path:
    """Root of the data tree: $SEGD_DATA_ROOT, default <project>/data.

    Phase 4 path portability: on machines that mount the data elsewhere, set
    SEGD_DATA_ROOT; manifest paths recorded under MANIFEST_DATA_ROOT are remapped here
    (call time), and MSTT's DATA_DIR resolves under this root too (import time: set the
    variable before importing segd.data)."""
    return Path(os.environ.get("SEGD_DATA_ROOT") or (PROJECT_ROOT / "data"))


DATA_DIR = data_root() / "mstt199"
CASES_CSV = PROJECT_ROOT / "eval" / "v2" / "mstt_cases.csv"
PHASE4_CSV = PROJECT_ROOT / "eval" / "v4" / "{dataset}_cases.csv"   # .format(dataset=...)

DATASETS = ("mstt", "beamster")   # runner-selectable datasets (Phase 4 = beamster only)

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
    # --- Phase 4 manifest rows (empty/None on MSTT rows) -------------------------
    modality: str = ""                     # manifest `modality` ("MRI" for beamster)
    n_lesions: int | None = None           # GT component count from the manifest; OFFLINE
    #                            use only (scorer/diagnostics) — never agent-visible
    manifest_target_text: str | None = None   # manifest `target_text` (full target phrase)

    @property
    def target_text(self) -> str:
        if self.manifest_target_text:
            return self.manifest_target_text
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


def remap_data_path(p: Path | str) -> Path:
    """Manifest absolute path -> current data root (Phase 4 portability, $SEGD_DATA_ROOT).

    Paths recorded under MANIFEST_DATA_ROOT by the manifest builder are re-rooted onto
    data_root(); anything else is returned unchanged (relative paths, other mounts)."""
    s = str(p)
    prefix = str(MANIFEST_DATA_ROOT)
    if s.startswith(prefix + "/") or s == prefix:
        return data_root() / s[len(prefix):].lstrip("/")
    return Path(s)


def phase4_cases_csv(dataset: str) -> Path:
    return Path(str(PHASE4_CSV).format(dataset=dataset))


def read_cases_phase4(dataset: str, split: str | None = None) -> list[CaseRow]:
    """Manifest-driven reader for eval/v4/<dataset>_cases.csv (Phase 4, decision 1).

    Columns (eval/v4/phase4_common.py): case, target_text, modality, sequence,
    image_path, mask_paths (';'-separated), n_lesions, lesion_volume_ml, split,
    image_path_src/mask_paths_src (native pre-iso provenance, ignored here).
    Absolute paths are remapped through remap_data_path. n_lesions / lesion_volume_ml
    are GT-derived: they land on CaseRow for OFFLINE code only — CaseState (the agent
    path) receives target_text/sequence/modality-free geometry, never these.
    """
    csv_path = phase4_cases_csv(dataset)
    if not csv_path.exists():
        raise FileNotFoundError(f"unknown dataset {dataset!r}: no manifest at {csv_path}")
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if split and r["split"] != split:
                continue
            masks = [remap_data_path(m) for m in r["mask_paths"].split(";") if m]
            rows.append(CaseRow(
                case=r["case"], histotype="", sequence=r["sequence"], multi_rater=False,
                split=r["split"], image_path=remap_data_path(r["image_path"]), mask_paths=masks,
                modality=r.get("modality", ""),
                n_lesions=int(r["n_lesions"]) if r.get("n_lesions") else None,
                manifest_target_text=r["target_text"],
            ))
    return rows


def get_case(case: str, dataset: str = "mstt") -> CaseRow:
    if dataset == "mstt":
        for r in read_cases():
            if r.case == case:
                return r
        raise KeyError(case)
    for r in read_cases_phase4(dataset):
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
