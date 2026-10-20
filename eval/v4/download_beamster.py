"""Download BEAMSTER (Brain mEtAstases segMentation for STEreotactic Radiotherapy) from figshare.

Dataset: Nohel et al., Scientific Data 2026 (paper DOI 10.1038/s41597-026-07777-0),
data DOI 10.6084/m9.figshare.29365844 (CC BY 4.0, published 2026-06-01).
140 patients, 260 annotated metastatic lesions, CE-T1 MRI registered to the planning-CT
space, NIfTI (.nii.gz), defaced. ~4.3 GB, 285 files, anonymous direct download.

Usage: uv run eval/v4/download_beamster.py [dest_dir]
Layout:  <dest>/Dataset_BEAMSTER/Dataset_{A,B}/Dataset_X_YYY[_segm].nii.gz  + spreadsheets/
Idempotent and resumable: files already present with the expected size are skipped;
partial files (.part) are resumed with curl -C -. No registration required.
"""
import json
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ARTICLE = 29365844
API = f"https://api.figshare.com/v2/articles/{ARTICLE}"
dest = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/agentic-segmentation/data/phase4/beamster")


def main() -> None:
    meta = json.loads(urllib.request.urlopen(API, timeout=60).read())
    files = meta["files"]
    dest.mkdir(parents=True, exist_ok=True)
    total = sum(f["size"] for f in files)
    print(f"{meta['title'][:60]}: {len(files)} files, {total / 1e9:.2f} GB -> {dest}")
    for i, f in enumerate(files, 1):
        out = dest / f["name"]
        if out.exists() and out.stat().st_size == f["size"]:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        part = out.with_suffix(out.suffix + ".part")
        if not (part.exists() and part.stat().st_size < f["size"]):
            part.write_bytes(b"")  # start fresh if missing or too large
        url = urllib.parse.quote(f["download_url"], safe="")  # ndownloader URL is already plain
        subprocess.run(["curl", "-sSfL", "--retry", "5", "--retry-delay", "5",
                        "-C", "-", "-o", str(part), f["download_url"]], check=True)
        part.rename(out)
        print(f"[{i}/{len(files)}] {out.name} {out.stat().st_size / 1e6:.1f} MB", flush=True)
    print("done:", len(files), "files ->", dest)


if __name__ == "__main__":
    main()
