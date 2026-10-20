"""Downloader for Spinal-Multiple-Myeloma-SEG (TCIA): dual-energy CT + spine lesion masks.

Dataset: Herrmann et al. (MISAG-BUT), "Spinal-Multiple-Myeloma-SEG: Segmentation of spinal
multiple myeloma lesions in dual-energy CT", Scientific Data 2026-08-11,
paper DOI 10.1038/s41597-026-08061-x; collection DOI 10.7937/k4qv-hh78 (TCIA), CC BY 4.0,
public/complete, 67 patients / 72 scans / 576 CT series / 564,464 slices, 304.43 GB.

Access (anonymous, no registration) — two NBIA REST channels:
  - CT images: `getSeries?Collection=...&Modality=CT` -> 576 series; the 72 *conventional*
    CT series are identified by SeriesDescription containing "konv"; the other 504 series
    are DECT-derived (VMI 40/80/120 keV, calcium-suppressed SI 25/50/75/100).
    Conventional-CT subset: 72 series / 70,558 slices / 37.2 GB.
  - Segmentations: `getSeries?Collection=...&Modality=SEG` -> 144 DICOM-SEG series
    (6.6 GB, one or more SEG objects per scan; QIICR/dcmqi), all 67 patients. These are
    the lesion + vertebra masks as DICOM SEG objects — the previous "NIfTI package needs
    Aspera" restriction does NOT apply to the SEG channel.
    (The separate Aspera NIfTI package id 1273 remains unused; not plain HTTP.)

Downloads ONLY the conventional CT series (default `--conventional-only`, disable with
`--all-ct`) and ALL SEG series, via the anonymous `getImage` endpoint, into
  <dest>/dicom/<PatientID>/<SeriesNumber>_<SeriesDescription>_<uid8>/
with a `.complete` marker per series (resumable: re-run to skip finished series),
at most PARALLEL parallel downloads.

Usage: uv run --project harness/server --with pydicom python eval/v4/download_spinal_mm_seg.py
       [--all-ct] [--manifest-only] [dest_dir]
"""
import concurrent.futures as futures
import io
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://nbia.cancerimagingarchive.net/nbia-api/services/v1"
COLLECTION = "Spinal-Multiple-Myeloma-SEG"
PARALLEL = 4

args = [a for a in sys.argv[1:] if not a.startswith("-")]
all_ct = "--all-ct" in sys.argv
manifest_only = "--manifest-only" in sys.argv
dest = Path(args[0] if args else "/home/ubuntu/agentic-segmentation/data/phase4/spinal_mm_seg")

_lock = threading.Lock()


def api(path: str, params: dict) -> bytes:
    from urllib.parse import urlencode
    req = urllib.request.Request(f"{BASE}/{path}?{urlencode(params)}")
    tok = os.environ.get("NBIA_BEARER")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def fetch_series(s: dict) -> tuple[str, bool, str]:
    """Download one series zip into its dir; returns (uid, ok, msg)."""
    uid = s["SeriesInstanceUID"]
    desc = re.sub(r"[^A-Za-z0-9]+", "_", s.get("SeriesDescription", "")).strip("_")
    outdir = dest / "dicom" / s["PatientID"] / f"{int(s.get('SeriesNumber', 0)):04d}_{desc}_{uid[-8:]}"
    marker = outdir / ".complete"
    if marker.exists():
        return uid, True, "already complete"
    outdir.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            blob = api("getImage", {"SeriesInstanceUID": uid})
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                z.extractall(outdir)
            marker.write_text(json.dumps({k: s.get(k) for k in
                                          ("SeriesInstanceUID", "StudyInstanceUID", "PatientID",
                                           "SeriesDescription", "Modality", "ImageCount")}, indent=1))
            with _lock:
                print(f"done {s['PatientID']} {s.get('SeriesDescription', '')[:40]} "
                      f"({s.get('ImageCount', '?')} obj) [{_log_count[0]}]", flush=True)
            return uid, True, "ok"
        except (urllib.error.URLError, TimeoutError, zipfile.BadZipFile, OSError) as e:
            if attempt == 3:
                return uid, False, repr(e)
            time.sleep(10 * (attempt + 1))
    return uid, False, "unreachable"


_log_count = [0]


def main() -> None:
    ct = json.loads(api("getSeries", {"Collection": COLLECTION, "Modality": "CT"}))
    seg = json.loads(api("getSeries", {"Collection": COLLECTION, "Modality": "SEG"}))
    if not all_ct:
        ct = [s for s in ct if "konv" in s.get("SeriesDescription", "").lower()]
    conv_slice_gb = 37.2 / 70558  # GB per CT slice, from TCIA series sizes
    ct_gb = sum(int(s.get("ImageCount", 0)) for s in ct) * conv_slice_gb
    seg_gb = sum(int(s.get("FileSize", 0)) for s in seg) / 1e9
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "series_manifest.json").write_text(json.dumps({"CT": ct, "SEG": seg}, indent=1))
    print(f"{len(ct)} CT series (~{ct_gb:.1f} GB) + {len(seg)} SEG series (~{seg_gb:.1f} GB)",
          flush=True)
    if manifest_only:
        return
    if ct_gb + seg_gb > 300:
        raise SystemExit(f"BLOCKED: selected subset ~{ct_gb + seg_gb:.0f} GB exceeds the 300 GB limit.")
    todo = ct + seg
    failed = []
    with futures.ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(fetch_series, s): s for s in todo}
        for fut in futures.as_completed(futs):
            uid, ok, msg = fut.result()
            if not ok:
                failed.append((uid, msg))
    if failed:
        print(f"FAILED ({len(failed)}):", flush=True)
        for uid, msg in failed:
            print(" ", uid, msg, flush=True)
        raise SystemExit(1)
    print(f"complete: {len(todo)} series -> {dest / 'dicom'}", flush=True)


if __name__ == "__main__":
    main()
