"""Convert Spinal-Multiple-Myeloma-SEG DICOM (NBIA download) to NIfTI per case.

Input:  data/phase4/spinal_mm_seg/dicom/<PatientID>/<SeriesNum>_<desc>_<uid8>/
        - exactly one conventional CT series (SeriesDescription contains "konv"),
        - exactly one lesion SEG (segments "Lesion_N", SNOMED property Neoplasm) and one
          vertebra SEG (segments "vertebrae_<X>" with SNOMED CT vertebra type codes).
The SEG objects reference the DECT-derived sibling series of the same scan (NOT the
conventional CT — verified: their SourceImageSequence SOP UIDs match none of the konv
series), but they share the CT's FrameOfReferenceUID and lie on the SAME grid
(identical ImageOrientationPatient / in-plane ImagePositionPatient; frame
ImagePositionPatient positions coincide with konv slice positions to < 1e-3 mm,
asserted per frame during conversion). SEG frames are therefore mapped directly onto
the conventional CT slices by world position — no resampling needed.

Geometry: slices are ordered by projection of ImagePositionPatient on the slice normal
(cross of IOP row/col vectors); the NIfTI affine is built from IOP/PixelSpacing/slice
step in RAS (LPS->RAS: negate x/y). Image, lesion mask and lesion label volume share
one affine, so canonicalisation (nibabel as_closest_canonical, used downstream by
eval/v4/convert_to_iso.py and segd.data.load_image) is consistent across all three.

Output: data/phase4/spinal_mm_seg/nifti/<case>/
        ct.nii.gz            (int16 HU, RescaleSlope/Intercept applied)
        lesion_mask.nii.gz   (uint8, union of all lesion segments, on the CT grid)
        lesion_labels.nii.gz (uint16, one integer label per lesion segment)
        seg_summary.json     (lesion segment list + per-lesion volumes, vertebra
                              labels/types, referenced-series note)

Case naming: <PatientID> when the patient has one scan; <PatientID>_sc<2..> for the
5 patients with two scans (72 scans / 67 patients).

Usage: uv run --project harness/server --with pydicom --with highdicom --with nibabel \
       python -u eval/v4/spinal_nifti_from_dicom.py [--jobs N]
"""
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pydicom

DICOM = Path("/home/ubuntu/agentic-segmentation/data/phase4/spinal_mm_seg/dicom")
OUT = Path("/home/ubuntu/agentic-segmentation/data/phase4/spinal_mm_seg/nifti")
POS_TOL = 1e-3  # mm tolerance for matching SEG frame position to a konv slice position
REF_NOTE = ("SEG SourceImageSequence SOP UIDs match none of the downloaded konv series: "
            "the SEGs annotate the DECT-derived sibling series of the same scan. They share "
            "the konv CT's FrameOfReferenceUID and identical grid (same IOP and in-plane "
            "position), so frames are mapped onto konv slices by ImagePositionPatient "
            "(per-frame deviation asserted < 1e-3 mm); no resampling required.")


def pair_scans(pdir: Path) -> list[tuple[Path, list[Path]]]:
    """Pair each konv CT series with its own SEG objects (patients with 2 scans have 2
    lesion-SEG + 2 vertebra-SEG objects). Pairing key: identical in-plane
    ImagePositionPatient origin and pixel spacing (verified on Myel_052, the only such
    case pattern; z-range containment is additionally asserted in convert_scan)."""
    ct_dirs = sorted(d for d in pdir.iterdir() if d.is_dir() and "konv" in d.name.lower())
    seg_dirs = sorted(d for d in pdir.iterdir() if d.is_dir() and d.name.startswith("0300_"))
    segs = []
    for sd in seg_dirs:
        ds = pydicom.dcmread(next(p for p in sd.iterdir() if p.name.endswith(".dcm")),
                             stop_before_pixels=True)
        pm = getattr(getattr(ds.SharedFunctionalGroupsSequence[0], "PixelMeasuresSequence", [None])[0]
                     if ds.SharedFunctionalGroupsSequence else None, "PixelSpacing", None)
        f0 = ds.PerFrameFunctionalGroupsSequence[0].PlanePositionSequence[0].ImagePositionPatient
        segs.append((sd, np.array([float(v) for v in f0][:2]),
                     np.array([float(x) for x in pm]) if pm is not None else None))
    pairs = []
    for ctdir in ct_dirs:
        heads = pydicom.dcmread(next(p for p in ctdir.iterdir() if p.name.endswith(".dcm")),
                                stop_before_pixels=True)
        key = np.asarray([float(v) for v in heads.ImagePositionPatient][:2])
        ps = np.asarray([float(x) for x in heads.PixelSpacing])
        mine = [sd for sd, xy, sps in segs
                if np.abs(xy - key).max() < 1e-2 and (sps is None or np.allclose(sps, ps, atol=1e-3))]
        if not mine:
            raise AssertionError(f"{ctdir}: no SEG object matches in-plane origin/spacing")
        pairs.append((ctdir, mine))
    if sum(len(m) for _, m in pairs) != len(seg_dirs):
        raise AssertionError(f"{pdir.name}: SEG-to-scan pairing left SEG objects unassigned")
    return pairs


def convert_scan(ctdir: Path, seg_dirs: list[Path], case: str, outdir: Path) -> dict:
    import nibabel as nib

    if (outdir / "ct.nii.gz").exists() and (outdir / "lesion_mask.nii.gz").exists():
        return {"case": case, "skipped": True}

    # ---- read konv CT series (headers + pixels), order slices along the normal ----
    names = sorted(p for p in ctdir.iterdir() if p.name.endswith(".dcm"))
    heads = [pydicom.dcmread(p, stop_before_pixels=True) for p in names]
    iop = np.asarray([float(v) for v in heads[0].ImageOrientationPatient])
    normal = np.cross(iop[:3], iop[3:6])
    proj = [float(np.dot([float(v) for v in h.ImagePositionPatient], normal)) for h in heads]
    order = np.argsort(proj)
    heads = [heads[i] for i in order]
    proj = [proj[i] for i in order]
    dz = float(np.median(np.diff(proj)))
    assert all(np.diff(proj) > 0), f"{case}: non-monotonic slice positions"
    ps = [float(x) for x in heads[0].PixelSpacing]          # [row spacing, col spacing]
    vol = np.empty((len(heads), int(heads[0].Rows), int(heads[0].Columns)), np.float32)
    for i, (h, p) in enumerate(zip(heads, [names[j] for j in order])):
        raw = pydicom.dcmread(p).pixel_array
        vol[i] = raw.astype(np.float32) * float(getattr(h, "RescaleSlope", 1)) + \
            float(getattr(h, "RescaleIntercept", 0))

    # affine (numpy array (z, y, x) = (slices, rows, cols)) in RAS mm
    xdir, ydir, zdir = iop[:3], iop[3:6], normal            # LPS column/row/normal dirs
    flip = np.diag([-1.0, -1.0, 1.0])                       # LPS -> RAS
    aff = np.eye(4)
    aff[:3, 0] = flip[:3, :3] @ (xdir * ps[1])              # numpy x = image columns
    aff[:3, 1] = flip[:3, :3] @ (ydir * ps[0])              # numpy y = image rows
    aff[:3, 2] = flip[:3, :3] @ (zdir * dz)
    aff[:3, 3] = flip[:3, :3] @ np.asarray([float(v) for v in heads[0].ImagePositionPatient])

    outdir.mkdir(parents=True, exist_ok=True)
    import nibabel as nib2
    im = nib2.Nifti1Image(np.transpose(vol, (2, 1, 0)).astype(np.int16), aff)  # (x,y,z)
    nib2.save(im, str(outdir / "ct.nii.gz"))

    # ---- SEG objects -> lesion masks on the CT grid ----
    summary = {"case": case, "n_lesions": 0, "lesions": [], "vertebrae": [],
               "spacing_mm": [round(abs(a), 4) for a in (ps[1], ps[0], dz)],
               "n_slices": len(heads), "reference_note": REF_NOTE}
    lesion_lbl = np.zeros(vol.shape, np.uint16)
    lesion_bin = np.zeros(vol.shape, np.uint8)
    for sd in seg_dirs:
        seg_files = [p for p in sd.iterdir() if p.name.endswith(".dcm")]
        assert len(seg_files) == 1, f"{sd}: unexpected SEG file count {len(seg_files)}"
        ds = pydicom.dcmread(seg_files[0])
        segs = {s.SegmentNumber: s for s in ds.SegmentSequence}
        px = ds.pixel_array                                  # (frames, rows, cols) 0/1
        lesion_nums = [n for n, s in segs.items() if s.SegmentLabel.lower().startswith("lesion")]
        for s in segs.values():
            if s.SegmentLabel.lower().startswith("lesion"):
                summary["lesions"].append(s.SegmentLabel)
            else:
                rec = {"label": s.SegmentLabel}
                if hasattr(s, "SegmentedPropertyTypeCodeSequence"):
                    c = s.SegmentedPropertyTypeCodeSequence[0]
                    rec["type"] = f"{c.CodeMeaning} ({c.CodingSchemeDesignator}:{c.CodeValue})"
                summary["vertebrae"].append(rec)
        seg_num_of_frame = np.array([
            int(fg.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
            for fg in ds.PerFrameFunctionalGroupsSequence])
        frame_proj = np.array([float(np.dot(
            [float(v) for v in fg.PlanePositionSequence[0].ImagePositionPatient], normal))
            for fg in ds.PerFrameFunctionalGroupsSequence])
        idx = np.clip(np.searchsorted(np.asarray(proj), frame_proj), 1, len(proj) - 1)
        d_lo, d_hi = np.abs(np.asarray(proj)[idx - 1] - frame_proj), np.abs(np.asarray(proj)[idx] - frame_proj)
        frame_slice = np.where(d_hi <= d_lo, idx, idx - 1)
        nearest = np.minimum(d_lo, d_hi)
        if (nearest > POS_TOL).any():
            raise AssertionError(
                f"{case}: {(nearest > POS_TOL).sum()} SEG frames off the konv slice grid "
                f"(max dev {nearest.max():.4f} mm)")
        # in-plane grid identity (position and spacing)
        f0 = np.asarray([float(v) for v in
                         ds.PerFrameFunctionalGroupsSequence[0].PlanePositionSequence[0].ImagePositionPatient])
        h0 = np.asarray([float(v) for v in heads[0].ImagePositionPatient])
        # SEG carries no top-level PixelSpacing: in-plane spacing lives in the shared
        # PixelMeasuresSequence; fall back to the konv spacing (positions are re-checked per frame).
        seg_pm = getattr(ds.SharedFunctionalGroupsSequence[0], "PixelMeasuresSequence", [None])[0]
        seg_ps = ([float(x) for x in seg_pm.PixelSpacing] if seg_pm is not None
                  and hasattr(seg_pm, "PixelSpacing") else ps)
        if (np.abs(f0 - h0)[:2] > POS_TOL).any() or not np.allclose(seg_ps, ps, atol=1e-4):
            raise AssertionError(f"{case}: SEG in-plane grid differs from konv CT")
        for n in lesion_nums:
            fidx = np.flatnonzero(seg_num_of_frame == n)
            if not len(fidx):
                continue
            li = int(lesion_lbl.max()) + 1
            for fi in fidx:
                si = frame_slice[fi]
                lesion_lbl[si] = np.where(px[fi] > 0, li, lesion_lbl[si])
            lesion_bin[lesion_lbl == li] = 1
    summary["n_lesions"] = int(lesion_lbl.max())
    vox_ml = float(abs(ps[1] * ps[0] * dz) / 1000)
    summary["lesion_volumes_ml"] = [
        round(int((lesion_lbl == li).sum()) * vox_ml, 3)
        for li in range(1, int(lesion_lbl.max()) + 1)]
    for name, arr, dt in [("lesion_mask.nii.gz", lesion_bin, np.uint8),
                          ("lesion_labels.nii.gz", lesion_lbl, np.uint16)]:
        nib2.save(nib2.Nifti1Image(np.transpose(arr, (2, 1, 0)), aff), str(outdir / name))
    (outdir / "seg_summary.json").write_text(json.dumps(summary, indent=1))
    return {"case": case, "shape": list(vol.shape), "spacing": summary["spacing_mm"],
            "hu": [int(vol.min()), int(vol.max())],
            "n_lesions": summary["n_lesions"],
            "n_vertebra_labels": len(summary["vertebrae"])}


def convert_patient(pdir: Path) -> str:
    out = []
    for k, (ctdir, seg_dirs) in enumerate(pair_scans(pdir)):
        patient = pdir.name
        case = patient if len(ct_dirs := sorted(d for d in pdir.iterdir()
                                                if d.is_dir() and "konv" in d.name.lower())) == 1 \
            else f"{patient}_sc{k + 1}"
        try:
            r = convert_scan(ctdir, seg_dirs, case, OUT / case)
        except Exception:  # noqa: BLE001
            out.append(f"FAILED {case}")
            traceback.print_exc()
            continue
        out.append((f"skip {case}" if r.get("skipped") else
                    f"{case}: ct{r['shape']}@{r['spacing']} HU{r['hu']} "
                    f"lesions={r['n_lesions']} vertebra_labels={r['n_vertebra_labels']}"))
    return "; ".join(out)


def main() -> None:
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 4
    pdirs = sorted(d for d in DICOM.iterdir() if d.is_dir())
    n_ok = 0
    if jobs <= 1:
        for p in pdirs:
            print(convert_patient(p), flush=True)
            n_ok += 1
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(convert_patient, p): p for p in pdirs}
            for fut in as_completed(futs):
                print(fut.result(), flush=True)
                n_ok += 1
    print("converted", n_ok, "/", len(pdirs), "patients ->", OUT, flush=True)


if __name__ == "__main__":
    main()
