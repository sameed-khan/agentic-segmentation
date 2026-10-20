"""Downloader for UCSD-BMETS-Longitudinal (TCIA): longitudinal brain-metastases MRI.

Dataset: "The University of California San Diego Longitudinal Brain Metastases Annotated
Multimodal MRI Dataset (UCSD-BMETS-Longitudinal)", collection DOI 10.7937/5ejb-9g94
(TCIA, DataCite record created 2026-09-04; collection updated 2026/10/05), CC BY 4.0,
public/complete. 300 patients, 1,065 (page header) / 1,066 (stats table) MRI timepoints,
neuroradiologist-approved lesion segmentations distributed as NIfTI. 163 GB.

Access: the ONLY distribution channel is a TCIA "IBM Aspera" faspex public package (the
collection has no NBIA/DICOM listing):
  https://faspex.cancerimagingarchive.net/aspera/faspex/public/package?context=<token on the collection page>
Anonymous (no registration), but scripted download requires a faspex transfer token that
IBM Aspera Connect obtains through an OAuth browser round-trip. This script does NOT bypass
that: pass a bearer token obtained from an authenticated Aspera Connect session via
env UCSD_BMETS_BEARER, or download the NIfTI zip manually:
  1. install IBM Aspera Connect (browser plugin + ascp CLI),
  2. open the faspex package URL below, click Download, unzip under dest.

Usage: uv run eval/v4/download_ucsd_bmets.py [dest_dir]
NOT RUN in Phase 4: no scripted faspex transfer token available in this environment
(env UCSD_BMETS_BEARER unset; ascp/faspex API rejects anonymous transfer_spec requests,
verified: POST /aspera/faspex/api/v5/packages/<id>/transfer_spec/download -> 401).
"""
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

FASPEX_URL = ("https://faspex.cancerimagingarchive.net/aspera/faspex/public/package?context="
              "eyJyZXNvdXJjZSI6InBhY2thZ2VzIiwidHlwZSI6ImV4dGVybmFsX2Rvd25sb2FkX3BhY2thZ2UiLCJpZCI6IjEzNDMiLCJwYXNzY29kZSI6ImI5MDI5NmEwNTgxNDhlMGIzZTM1MjY2MDQ4NjRjOWM1ZDQ3YWVkZmMiLCJwYWNrYWdlX2lkIjoiMTM0MyIsImVtYWlsIjoiaGVscEBjYW5jZXJpbWFnaW5nYXJjaGl2ZS5uZXQifQ==")
API = "https://faspex.cancerimagingarchive.net/aspera/faspex/api/v5"

dest = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/agentic-segmentation/data/phase4/ucsd_bmets")


def main() -> None:
    dest.mkdir(parents=True, exist_ok=True)
    bearer = os.environ.get("UCSD_BMETS_BEARER")
    if not bearer:
        print(__doc__)
        raise SystemExit("BLOCKED: UCSD_BMETS_BEARER not set and no anonymous faspex transfer path "
                         f"is available. Download manually with IBM Aspera Connect: {FASPEX_URL}")
    # With a token, fetch the download transfer spec and stream files over the HTTP gateway.
    package_id = json.loads(__import__("base64").urlsafe_b64decode(
        FASPEX_URL.split("context=")[1] + "==="))["package_id"]
    req = urllib.request.Request(
        f"{API}/packages/{package_id}/transfer_spec/download?transfer_type=http_gateway&type=external_download_package",
        data=b"{}", method="POST",
        headers={"Authorization": f"Bearer {bearer}", "Content-Type": "application/json",
                 "Accept": "application/json"})
    spec = json.loads(urllib.request.urlopen(req, timeout=60).read())
    print("got transfer spec keys:", sorted(spec)[:10])
    # The spec lists files and an http gateway; stream each file with curl.
    for f in spec.get("files", []):
        rel = f["path"].lstrip("/")
        out = dest / rel
        if out.exists() and out.stat().st_size == f.get("size", -1):
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sSfL", "--retry", "5", "-C", "-", "-o", str(out),
                        f"{spec['http_gateway_url']}/presign", "-X", "POST",
                        "-H", f"Authorization: Bearer {bearer}",
                        "-H", "Content-Type: application/json",
                        "-d", json.dumps({"transfer_spec": spec, "method": "GET",
                                          "paths": [f["path"]]})], check=True)
        print(rel, flush=True)


if __name__ == "__main__":
    main()
