"""Phase 4 data check: shapes, spacing, intensity ranges (CT HU!), mask labels,
lesion counts/volumes + PNG previews, for every case in the dataset manifests.

Writes per-dataset JSONL to eval/v4/data_check_phase4/<dataset>.jsonl (STREAMED: one
line per case, appended as it is checked so progress is visible in the log) and
previews to data/phase4/<dataset>/check/<case>.png (3-panel orthogonal montage,
lesion outline red). Cases with split=unused are skipped (not part of design/test).

Run: uv run --project harness/server python eval/v4/data_check_phase4.py [dataset|all] [n_previews] [--jobs N]
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from phase4_common import dataset_dir, load_nii, preview, read_manifest

here = Path(__file__).resolve().parent


def check_case(rec_path: Path, preview_path: Path | None, r: dict) -> dict:
    rec = {"case": r["case"], "split": r["split"], "n_lesions_manifest": r["n_lesions"]}
    try:
        img, aff, sp = load_nii(Path(r["image_path"]))
        mask = np.asarray(
            __import__("nibabel").as_closest_canonical(
                __import__("nibabel").load(r["mask_paths"])).dataobj)
        labels = sorted(int(v) for v in np.unique(mask) if v > 0)
        lab, n = ndi.label(mask > 0)
        sizes = ndi.sum(mask > 0, lab, range(1, n + 1)) if n else np.array([0.0])
        vox_ml = float(np.prod(sp) / 1000)
        les_vols = sorted((np.array(sizes) * vox_ml).tolist(), reverse=True)[:10]
        rec.update(
            shape=list(img.shape), spacing=[round(s, 3) for s in sp],
            dtype=str(img.dtype),
            img_min=float(img.min()), img_max=float(img.max()),
            img_p1=float(np.percentile(img, 1)), img_p99=float(np.percentile(img, 99)),
            mask_labels=labels[:20], n_lesions_computed=int(n),
            lesion_volume_ml=[round(v, 3) for v in les_vols],
            tiny_lesions_1ml=int(sum(v < 1.0 for v in sizes)),
            ok=True)
        if r["modality"] == "CT":  # HU sanity: air must be strongly negative
            rec["ct_hu_ok"] = bool(img.min() < -300 and img.max() > 100)
        if preview_path:
            preview(img, mask > 0, preview_path,
                    window=(0, 300) if r["modality"] == "CT" else None)
    except Exception as e:  # noqa: BLE001 - keep checking remaining cases
        rec.update(ok=False, error=repr(e))
    return rec


def main() -> None:
    argv = sys.argv[1:]
    jobs = int(argv[argv.index("--jobs") + 1]) if "--jobs" in argv else 1
    which = argv[0] if argv and not argv[0].startswith("--") else "all"
    n_previews = int(argv[1]) if len(argv) > 1 and not argv[1].startswith("--") else 6

    for ds in ["beamster", "spinal_mm_seg", "ucsd_bmets", "longitudinal_ct"]:
        if which != "all" and which != ds:
            continue
        csv_path = here / f"{ds}_cases.csv"
        if not csv_path.exists():
            print(f"[{ds}] no manifest (skipped)", flush=True)
            continue
        rows = read_manifest(csv_path)
        rows = [r for r in rows if r["split"] != "unused"]
        out = here / "data_check_phase4"
        out.mkdir(exist_ok=True)
        rec_path = out / f"{ds}.jsonl"
        prev_cases = {r["case"] for r in rows[:n_previews]}  # first n cases get PNG previews
        done = {r["case"] for r in rows}
        n_ok = 0
        with open(rec_path, "a") as f:
            def emit(rec):
                nonlocal n_ok
                f.write(json.dumps(rec) + "\n")
                f.flush()
                if rec.get("ok"):
                    n_ok += 1
            if jobs <= 1:
                for r in rows:
                    rec = check_case(rec_path, dataset_dir(ds) / "check" / f"{r['case']}.png"
                                     if r["case"] in prev_cases else None, r)
                    emit(rec)
                    print(f"[{ds}] {r['case']} ok={rec.get('ok')}", flush=True)
            else:
                with ProcessPoolExecutor(max_workers=jobs) as ex:
                    futs = {ex.submit(check_case, rec_path,
                                      dataset_dir(ds) / "check" / f"{r['case']}.png"
                                      if r["case"] in prev_cases else None, r): r for r in rows}
                    for fut in as_completed(futs):
                        emit(fut.result())
        print(f"[{ds}] {n_ok}/{len(rows)} cases ok -> {rec_path}", flush=True)


if __name__ == "__main__":
    main()
