#!/usr/bin/env python
"""Download the official SAM-Med3D-turbo checkpoint into the project cache and verify
its pinned sha256 (research/phase6_sammed3d.md, "checkpoint provenance").

Usage (from harness/local_models/sammed3d):
    uv run python scripts/fetch_checkpoint.py [--url URL] [--out PATH]

Default URL: the official HuggingFace checkpoint
https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

DEFAULT_URL = "https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth"
CACHE_DIR = Path(__file__).resolve().parents[1] / "checkpoints"
TARGET = CACHE_DIR / "sam_med3d_turbo.pth"
# pinned sha256 of the official sam_med3d_turbo.pth (verified at download and at load)
PINNED_SHA256 = "899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", default=str(TARGET))
    args = parser.parse_args()
    out = Path(args.out)
    if out.is_file():
        got = sha256_of(out)
        if got == PINNED_SHA256:
            print(f"checkpoint already present and verified: {out} ({got})")
            (CACHE_DIR / "sam_med3d_turbo.sha256").write_text(got + "\n")
            return 0
        print(f"existing file has sha256 {got} != pinned; re-downloading", file=sys.stderr)
        out.unlink()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    print(f"downloading {args.url} -> {tmp}")
    with urllib.request.urlopen(args.url, timeout=120) as resp, open(tmp, "wb") as f:
        total = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            if total % (64 << 20) < (1 << 20):
                print(f"  {total / 1e6:.0f} MB")
    got = sha256_of(tmp)
    if got != PINNED_SHA256:
        tmp.unlink(missing_ok=True)
        print(f"sha256 MISMATCH: got {got}, pinned {PINNED_SHA256}", file=sys.stderr)
        return 1
    tmp.replace(out)
    (CACHE_DIR / "sam_med3d_turbo.sha256").write_text(got + "\n")
    print(f"verified {got} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
