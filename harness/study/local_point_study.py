#!/usr/bin/env python3
"""Phase 5 Study A (standalone localization): run local pointer/medical models on the
Phase 2 prepare manifests (design or test), collecting responses for offline scoring.

Models (all via their local HTTP services, decoding deterministic; see
harness/local_models/ and research/phase5_implementation.md):
- gui_actor    : pointd-gui-actor :8092 (GUI-Actor-2B, official pointer head) — conditions raw, marks
- molmo2       : pointd-molmo2   :8093 (Molmo2-8B, <points> output)          — conditions raw, marks
- medgemma_q4  : llama-server    :8091 (Q4_K_M GGUF, OpenAI-compatible)      — conditions raw_box, marks_box, marks_id
- medgemma_bf16: BF16 server     :8094 (Transformers, OpenAI-compatible)     — conditions raw_box, marks_box, marks_id

Conditions (identical inputs to the Phase 2 frontier study):
- raw       : clean image (manifest image_R) -> a point inside the tumor (pointer
              models' query; MedGemma: normalized box -> centre).
- marks     : marked image (image_S, clean left + numbered marks right) -> point
              snapped to the nearest drawn mark at SCORING time (raw point also kept);
              MedGemma additionally answers a mark id (marks_id) and a box snapped to
              the nearest mark (marks_box).
GT-derived manifest fields are never sent: only the images and the prompt text.

Output: results/phase5/studyA/<run>/responses.jsonl (one line per
case×model×condition×rep; resumable exactly like ask.ts: ok rows are skipped, error
rows dropped and retried) + meta.json. Scoring: score_local_point.py.

Run:  uv run --project harness/server python harness/study/local_point_study.py \
          --run studyA-design-gui --models gui_actor [--prepare-dir <main tree>/results/phase2_dev/prepare]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "harness" / "local_models" / "medgemma"))
sys.path.insert(0, str(PROJECT_ROOT / "harness" / "study"))

from medgemma_client import MedGemmaClient  # noqa: E402

# Study model key -> service kind, default port and conditions. Keep in sync with
# harness/local_models/ (ports) and score_local_point.py MODELS.
MODELS: dict[str, dict] = {
    "gui_actor": {"kind": "pointd", "port": 8092, "conditions": ("raw", "marks")},
    "molmo2": {"kind": "pointd", "port": 8093, "conditions": ("raw", "marks")},
    "medgemma_q4": {"kind": "chat", "port": 8091, "conditions": ("raw_box", "marks_box", "marks_id")},
    "medgemma_bf16": {"kind": "chat", "port": 8094, "conditions": ("raw_box", "marks_box", "marks_id")},
}
POINTD_CONDITIONS = ("raw", "marks")
CHAT_CONDITIONS = ("raw_box", "marks_box", "marks_id")
ALL_CONDITIONS = sorted(set(POINTD_CONDITIONS) | set(CHAT_CONDITIONS))

HISTOTYPE_TEXT = {
    "F": "fibrous soft-tissue tumor",
    "L": "lipomatous soft-tissue tumor",
    "M": "myxoid soft-tissue tumor",
    "N": "peripheral nerve sheath tumor",
    "V": "vascular soft-tissue tumor",
}
SEQUENCE_TEXT = {"T1": "T1-weighted", "T2": "T2-weighted fat-suppressed"}

# condition -> (prompt template, image of the manifest row)
CONDITION_SPEC = {
    "raw": ("local_point_raw.md", "image_R"),
    "marks": ("local_point_marks.md", "image_S"),
    "raw_box": ("local_point_raw_box.md", "image_R"),
    "marks_box": ("local_point_marks_box.md", "image_S"),
    "marks_id": ("local_point_marks_id.md", "image_S"),
}
PROMPTS_DIR = PROJECT_ROOT / "harness" / "study" / "prompts"
DEFAULT_RESULTS_ROOT = "results/phase5/studyA"


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_provenance() -> dict:
    """git state for the run meta (preregistration discipline; same fields as ask.ts)."""
    import subprocess

    def git(cmd: str) -> str | None:
        try:
            return subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT, capture_output=True,
                                  text=True, timeout=30).stdout.strip() or None
        except Exception:
            return None

    return {
        "git_commit": git("git rev-parse HEAD"),
        "git_describe": git("git describe --tags --always --dirty"),
        "git_dirty_files": [l for l in (git("git status --porcelain") or "").splitlines() if l.strip()],
    }


def render_template(text: str, vars_: dict[str, str]) -> str:
    """{{var}} substitution, same convention as ask.ts / run_cases.ts."""
    import re

    return re.sub(r"\{\{(\w+)\}\}", lambda m: vars_.get(m.group(1), m.group(0)), text)


def resolve_image(prepare_dir: Path, images_root: Path | None, rel: str | None) -> Path | None:
    """Manifest image paths are relative to the source project root (prepare dir's
    tree); --images-root overrides that root."""
    if not rel:
        return None
    if images_root:
        return images_root / rel
    return prepare_dir.parents[2] / rel


# ------------------------------------------------------------------ service calls
def call_pointd(url: str, image_path: Path, query: str, k: int, timeout_s: float) -> dict:
    import httpx

    png_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    with httpx.Client(timeout=timeout_s) as client:
        resp = client.post(url + "/point", json={"image_png_b64": png_b64, "query": query, "k": k})
        resp.raise_for_status()
        return resp.json()


def call_chat(client: MedGemmaClient, prompt: str, image_path: Path, max_tokens: int) -> dict:
    result = client.chat_json(prompt, [image_path], max_tokens=max_tokens)
    return {"raw": result.text, "parsed": result.parsed, "parse_error": result.parse_error,
            "uncertain": result.uncertain, "latency_ms": result.latency_ms,
            "model": result.model, "usage": result.usage, "raw_body": result.raw_body}


# ------------------------------------------------------------------ main loop
def build_call(row: dict, model: str, condition: str, rep: int, opts) -> dict:
    """Build and execute one service call; returns the responses.jsonl line."""
    spec = MODELS[model]
    template_name, image_field = CONDITION_SPEC[condition]
    template_path = PROMPTS_DIR / template_name
    template = template_path.read_text()
    vars_ = {"target": HISTOTYPE_TEXT[row["histotype"]], "sequence_text": SEQUENCE_TEXT[row["sequence"]]}
    prompt_text = render_template(template, vars_)
    image = resolve_image(Path(opts.prepare_dir), Path(opts.images_root) if opts.images_root else None,
                          row[image_field])
    if image is None or not image.exists():
        raise FileNotFoundError(f"{row['case']}: {image_field} not found at {image}")

    line = {
        "run": opts.run, "case": row["case"], "histotype": row["histotype"],
        "sequence": row["sequence"], "model": model, "condition": condition, "rep": rep,
        "model_id": None, "service_version": None,
        "prompt_template": f"harness/study/prompts/{template_name}",
        "prompt_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "prompt_text": prompt_text, "image": row[image_field], "k": opts.k,
        "raw": "", "points": [], "parsed": None, "parse_error": None, "uncertain": None,
        "usage": None, "latency_ms": None, "ok": False, "error": None,
        "created_at": utcnow(),
    }
    t0 = time.perf_counter()
    try:
        if spec["kind"] == "pointd":
            out = call_pointd(opts.urls[model], image, prompt_text.strip(), opts.k, opts.timeout)
            line.update({
                "raw": out.get("raw", ""), "points": out.get("points", []),
                "model_id": out.get("model"), "service_version": out.get("version"),
                "latency_ms": out.get("latency_ms"), "verifier": out.get("verifier"),
            })
        else:
            out = call_chat(MedGemmaClient(base_url=opts.urls[model], timeout_s=opts.timeout),
                            prompt_text.strip(), image, opts.max_tokens)
            line.update({
                "raw": out["raw"], "parsed": out["parsed"], "parse_error": out["parse_error"],
                "uncertain": out["uncertain"], "model_id": out["model"],
                "latency_ms": out["latency_ms"], "usage": out["usage"],
                "raw_body": out["raw_body"][:2000], "service_version": None,
            })
        line["ok"] = True
    except Exception as exc:  # transport/HTTP failure: record, retry on resume
        line["error"] = f"{type(exc).__name__}: {exc}"
    line["wall_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
    return line


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="run name under results/phase5/studyA/")
    ap.add_argument("--models", default="gui_actor,molmo2,medgemma_q4,medgemma_bf16",
                    help="comma list from: " + ",".join(MODELS))
    ap.add_argument("--conditions", default="",
                    help="optional comma filter (default: each model's full condition set); "
                         f"known: {','.join(ALL_CONDITIONS)}")
    ap.add_argument("--prepare-dir", required=True,
                    help="Phase 2 prepare manifest dir (design: <main tree>/results/phase2_dev/prepare, "
                         "test: .../results/phase2_test/prepare) — read-only")
    ap.add_argument("--images-root", default=None,
                    help="root the manifest's image paths resolve against (default: the prepare "
                         "dir's project root)")
    ap.add_argument("--cases", default="", help="comma-separated case ids (default: all)")
    ap.add_argument("--reps", type=int, default=1, help="repeats per case×model×condition (default 1: "
                                                         "decoding is deterministic at temperature 0)")
    ap.add_argument("--k", type=int, default=3, help="points requested from pointer services (default 3)")
    ap.add_argument("--max-tokens", type=int, default=256, help="max new tokens for MedGemma calls")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=900.0, help="per-call timeout seconds")
    ap.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--gui-actor-url", default=None)
    ap.add_argument("--molmo2-url", default=None)
    ap.add_argument("--medgemma-q4-url", default=None)
    ap.add_argument("--medgemma-bf16-url", default=None)
    args = ap.parse_args(argv)

    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in args.models if m not in MODELS]
    if unknown:
        raise SystemExit(f"unknown models: {unknown} (known: {', '.join(MODELS)})")
    args.urls = {
        "gui_actor": args.gui_actor_url or f"http://127.0.0.1:{MODELS['gui_actor']['port']}",
        "molmo2": args.molmo2_url or f"http://127.0.0.1:{MODELS['molmo2']['port']}",
        "medgemma_q4": args.medgemma_q4_url or f"http://127.0.0.1:{MODELS['medgemma_q4']['port']}",
        "medgemma_bf16": args.medgemma_bf16_url or f"http://127.0.0.1:{MODELS['medgemma_bf16']['port']}",
    }
    args.conditions_filter = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for cond in args.conditions_filter:
        if cond not in ALL_CONDITIONS:
            raise SystemExit(f"unknown condition: {cond} (known: {', '.join(ALL_CONDITIONS)})")
    return args


def load_done_and_prune(resp_path: Path) -> set[tuple]:
    """Resume bookkeeping (ask.ts convention): returns the completed ok keys;
    rewrites the file without error/malformed/duplicate rows so a rerun with the same
    --run retries exactly the failures."""
    done: set[tuple] = set()
    kept: list[str] = []
    n_seen = 0
    if not resp_path.exists():
        return done
    for line in resp_path.read_text().splitlines():
        if not line.strip():
            continue
        n_seen += 1
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("ok") is True:
            key = (rec["case"], rec["model"], rec["condition"], rec["rep"])
            if key not in done:
                done.add(key)
                kept.append(line)
    if len(kept) != n_seen or len(done) != len(kept):
        resp_path.write_text("".join(l + "\n" for l in kept))
        print(f"[studyA] resume: dropping {n_seen - len(kept)} row(s) for retry")
    return done


def main() -> None:
    opts = parse_args()
    run_dir = PROJECT_ROOT / opts.results_root / opts.run
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(opts.prepare_dir) / "manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"no prepare manifest at {manifest_path}")
    rows = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
    if opts.cases:
        wanted = {c.strip() for c in opts.cases.split(",") if c.strip()}
        missing = wanted - {r["case"] for r in rows}
        if missing:
            raise SystemExit(f"unknown cases: {sorted(missing)}")
        rows = [r for r in rows if r["case"] in wanted]

    calls = []
    for row in rows:
        for model in opts.models:
            conds = MODELS[model]["conditions"]
            if opts.conditions_filter:
                conds = tuple(c for c in conds if c in opts.conditions_filter)
            for condition in conds:
                for rep in range(opts.reps):
                    calls.append((row, model, condition, rep))

    # Resumability (ask.ts convention): keep ok rows, drop error/malformed rows so a
    # rerun with the same --run retries exactly the failures.
    resp_path = run_dir / "responses.jsonl"
    done = load_done_and_prune(resp_path)
    todo = [c for c in calls if (c[0]["case"], c[1], c[2], c[3]) not in done]

    meta = {
        "run": opts.run, "stage": "phase5_studyA_local_point", "created_at": utcnow(),
        **git_provenance(),
        "models": opts.models, "conditions": opts.conditions_filter or "per-model defaults",
        "reps": opts.reps, "k": opts.k, "max_tokens": opts.max_tokens,
        "prepare_manifest": str(manifest_path), "images_root": opts.images_root,
        "urls": opts.urls, "timeout_s": opts.timeout, "concurrency": opts.concurrency,
        "planned_calls": len(calls), "resumed_from_lines": len(done),
        "prompts": {name: {"path": f"harness/study/prompts/{name}.md",
                           "sha256": hashlib.sha256((PROMPTS_DIR / f"{name}.md").read_bytes()).hexdigest()}
                    for name in ("local_point_raw", "local_point_marks", "local_point_raw_box",
                                 "local_point_marks_box", "local_point_marks_id")},
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"[studyA] run={opts.run} models={','.join(opts.models)} calls={len(calls)} "
          f"(done {len(done)}, todo {len(todo)}) concurrency={opts.concurrency}")
    lock = threading.Lock()
    error_rows = 0

    def worker(call):
        row, model, condition, rep = call
        return build_call(row, model, condition, rep, opts)

    with ThreadPoolExecutor(max_workers=max(1, opts.concurrency)) as pool:
        futures = {pool.submit(worker, call): call for call in todo}
        for fut in as_completed(futures):
            line = fut.result()
            with lock:
                with open(resp_path, "a") as fh:
                    fh.write(json.dumps(line) + "\n")
                if not line["ok"]:
                    error_rows += 1
            print(f"[studyA] {line['case']}/{line['model']}/{line['condition']}/r{line['rep']} "
                  f"ok={line['ok']} {line['latency_ms']}ms"
                  + (f" error={line['error']}" if not line["ok"] else ""), flush=True)

    if error_rows:
        print(f"[studyA] {error_rows} error rows recorded (rerun the same --run to retry)")
        sys.exit(3)
    print(f"[studyA] wrote {resp_path}")


if __name__ == "__main__":
    main()
