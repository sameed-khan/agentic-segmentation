#!/usr/bin/env python3
"""Phase 5 smoke test: hit every local model service once and sanity-check outputs.

Runs against REAL services (on the A100, after setup_a100.sh) or --fake services
(anywhere, no GPU): one MRI slice from the Phase 2 design prepare dir, plus a
synthetic UI screenshot sanity check for GUI-Actor (its home domain — verifies the
pointer-head path end-to-end before any MRI conclusion).

Checks per service: /healthz reachable; /point returns k in-bounds pixel points
(GUI-Actor/Molmo2); chat completions return strictly-parseable JSON with the
`uncertain` option (MedGemma arms). Latency and raw outputs are printed for the run
log. Exits non-zero on transport/contract failures (a low-confidence MRI answer is
NOT a failure — this is a plumbing check, not a study).

Run (no GPU, fake services):
    uv run --project harness/server python harness/local_models/smoke_test.py --fake
Run (A100, real services):
    uv run --project harness/server python harness/local_models/smoke_test.py \
        --prepare-dir /home/ubuntu/agentic-segmentation/results/phase2_dev/prepare
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "medgemma"))

from medgemma_client import MedGemmaClient  # noqa: E402

SERVICES = {
    "gui_actor": {"base_url": "http://127.0.0.1:8092"},
    "molmo2": {"base_url": "http://127.0.0.1:8093"},
    "medgemma_q4": {"base_url": "http://127.0.0.1:8091"},
    "medgemma_bf16": {"base_url": "http://127.0.0.1:8094"},
}
PROMPTS_DIR = HERE.parent / "study" / "prompts"


def b64_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def http_json(url: str, payload: dict | None = None, method: str = "POST") -> dict:
    import urllib.error
    import urllib.request

    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode())


def pick_slice(prepare_dir: Path, case: str | None) -> dict:
    rows = [json.loads(l) for l in (prepare_dir / "manifest.jsonl").read_text().splitlines() if l.strip()]
    if case:
        rows = [r for r in rows if r["case"] == case]
        if not rows:
            raise SystemExit(f"case {case} not in {prepare_dir}/manifest.jsonl")
    return rows[0]


def resolve_image(prepare_dir: Path, images_root: Path | None, rel: str | None) -> Path | None:
    if not rel:
        return None
    if images_root:  # explicit root that already contains the manifest's relative paths
        return images_root / rel
    return prepare_dir.parents[2] / rel  # manifest paths are relative to the source project root


def ui_screenshot_png(path: Path) -> Path:
    """Deterministic synthetic UI window (grey canvas, dark toolbar, blue 'OK' button)."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (640, 400), (234, 234, 234))
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, 620, 60], fill=(60, 60, 60))
    d.text((30, 30), "Settings", fill=(255, 255, 255))
    d.rectangle([20, 60, 620, 380], fill=(250, 250, 250), outline=(180, 180, 180))
    d.rectangle([420, 300, 580, 350], fill=(52, 120, 246), outline=(30, 80, 180))
    d.text((480, 318), "OK", fill=(255, 255, 255))
    img.save(path)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prepare-dir", default=None,
                    help="Phase 2 design prepare dir (default: the main tree's "
                         "results/phase2_dev/prepare if present, else results/phase2_dev/prepare)")
    ap.add_argument("--images-root", default=None,
                    help="root the manifest's image paths resolve against (default: the prepare "
                         "dir's project root)")
    ap.add_argument("--case", default=None, help="manifest case id (default: first row)")
    ap.add_argument("--fake", action="store_true", help="expect/use --fake services (no GPU)")
    ap.add_argument("--skip-medgemma", action="store_true")
    ap.add_argument("--skip-gui-screenshot", action="store_true")
    args = ap.parse_args()

    prepare_dir = Path(args.prepare_dir) if args.prepare_dir else None
    if prepare_dir is None:
        for cand in (Path("/home/ubuntu/agentic-segmentation/results/phase2_dev/prepare"),
                     Path("results/phase2_dev/prepare")):
            if (cand / "manifest.jsonl").exists():
                prepare_dir = cand
                break
        if prepare_dir is None:
            raise SystemExit("no prepare manifest found; pass --prepare-dir")
    images_root = Path(args.images_root) if args.images_root else None

    row = pick_slice(prepare_dir, args.case)
    img_r = resolve_image(prepare_dir, images_root, row["image_R"])
    print(f"[smoke] case {row['case']} ({row['histotype']}/{row['sequence']}), k*={row['k_star']}, "
          f"slice image: {img_r}")

    failures: list[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        mark = "ok " if ok else "FAIL"
        print(f"[smoke] {mark} {name}: {detail}")
        if not ok:
            failures.append(name)

    query = "the " + {"F": "fibrous", "L": "lipomatous", "M": "myxoid",
                        "N": "peripheral nerve sheath", "V": "vascular"}.get(
        row["histotype"], "") + " soft-tissue tumor"

    # ---- pointer services --------------------------------------------------
    for name in ("gui_actor", "molmo2"):
        base = SERVICES[name]["base_url"]
        try:
            health = http_json(base + "/healthz", method="GET")
            out = http_json(base + "/point",
                            {"image_png_b64": b64_image(img_r), "query": query, "k": 3})
            pts = out.get("points") or []
            inb = all(0 <= p["x"] <= out["image_w"] and 0 <= p["y"] <= out["image_h"] for p in pts)
            check(f"{name}/healthz", health.get("status") == "ok", json.dumps(health))
            check(f"{name}/point", len(pts) >= 1 and inb,
                  f"{len(pts)} pts, {out.get('latency_ms')}ms, raw: {str(out.get('raw'))[:120]!r}")
        except Exception as exc:
            check(f"{name}", False, f"unreachable or failed: {exc}")

    # ---- GUI-Actor UI-screenshot sanity (its home domain) -------------------
    if not args.skip_gui_screenshot:
        shot = ui_screenshot_png(Path("/tmp/phase5_ui_sanity.png"))
        try:
            out = http_json(SERVICES["gui_actor"]["base_url"] + "/point",
                            {"image_png_b64": b64_image(shot), "query": "the OK button", "k": 3})
            pts = out.get("points") or []
            inside = bool(pts) and 420 <= pts[0]["x"] <= 580 and 300 <= pts[0]["y"] <= 350
            check("gui_actor ui-screenshot", bool(pts),
                  f"top point {pts[0] if pts else None} (button at [420,300,580,350]; "
                  f"{'INSIDE' if inside else 'not judged — inspect on A100'})")
        except Exception as exc:
            check("gui_actor ui-screenshot", False, f"failed: {exc}")

    # ---- MedGemma arms ------------------------------------------------------
    if not args.skip_medgemma:
        prompt = ("This is one axial MRI slice. Reply with JSON only: "
                  '{"answer": "lesion_visible" | "no_lesion" | "uncertain", "confidence": 0-1}.')
        for name in ("medgemma_q4", "medgemma_bf16"):
            base = SERVICES[name]["base_url"]
            try:
                client = MedGemmaClient(base_url=base)
                result = client.chat_json(prompt, [img_r], max_tokens=64)
                ok = result.parse_error is None
                check(f"{name} chat_json", ok,
                      f"parsed={result.parsed} raw: {result.text[:120]!r} "
                      f"({result.latency_ms}ms)" if ok else result.parse_error or "parse failed")
            except Exception as exc:
                check(name, False, f"unreachable or failed: {exc}")

    print(f"\n[smoke] {'ALL OK' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
