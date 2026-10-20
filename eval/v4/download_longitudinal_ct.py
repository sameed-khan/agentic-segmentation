"""Download Longitudinal-CT (autoPET/CT-IV melanoma whole-body CT lesion dataset) from FDAT.

Dataset: "Longitudinal-CT", University Hospital Tübingen, FDAT DOI 10.57754/FDAT.qe950-g4h94
(v3, record created 2026-09-18), CC BY-NC 4.0. 300 melanoma patients, baseline + follow-up
whole-body CT (NIfTI, 3 mm slices), manually annotated malignant-lesion masks with per-lesion
integer labels + lesion COG JSONs + a per-lesion CSV. ~51 GB as a single zip, anonymous
direct HTTP download (Invenio generates a signed URL; no registration).

Usage: uv run eval/v4/download_longitudinal_ct.py [dest_dir]
Layout:  <dest>/Longitudinal_CT_v3.zip
Idempotent and resumable: curl -C - resumes a partial .part file; the completed zip is
renamed in place. Size is capped at 300 GB (task rule) unless --force.
"""
import subprocess
import sys
from pathlib import Path

RECORD = "https://fdat.uni-tuebingen.de/api/records/qe950-g4h94"
FILE = "Longitudinal_CT_v3.zip"
DOWNLOAD_URL = f"https://fdat.uni-tuebingen.de/records/{RECORD.split('/')[-1]}/files/{FILE}?download=1"
MAX_GB = 300  # task rule: stop and report if a dataset is > 300 GB

dest_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/agentic-segmentation/data/phase4/longitudinal_ct")
force = "--force" in sys.argv


def remote_size() -> int:
    """Exact file size (and md5 via the record API) without a range request."""
    import json, urllib.request
    req = urllib.request.Request(f"{RECORD}/files/{FILE}", headers={"Accept": "application/json"})
    e = json.loads(urllib.request.urlopen(req, timeout=60).read())
    return int(e["size"])


def main() -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / FILE
    part = out.with_suffix(".zip.part")
    if out.exists():
        print(f"already downloaded: {out} ({out.stat().st_size / 1e9:.2f} GB)")
        return
    size = remote_size()
    gb = size / 1e9
    print(f"remote size: {gb:.2f} GB")
    if gb > MAX_GB and not force:
        raise SystemExit(f"BLOCKED: dataset is {gb:.1f} GB > {MAX_GB} GB limit; "
                         "report back instead of downloading (use --force to override).")
    subprocess.run(["curl", "-sSfL", "--retry", "5", "--retry-delay", "10",
                    "-C", "-", "-o", str(part), DOWNLOAD_URL], check=True)
    if part.stat().st_size != size:
        raise SystemExit(f"size mismatch after download: {part.stat().st_size} != {size}")
    part.rename(out)
    print(f"done: {out} ({gb:.2f} GB)")


if __name__ == "__main__":
    main()
