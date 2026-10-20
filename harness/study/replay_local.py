#!/usr/bin/env python3
"""Phase 5 Study B replay: send each decision-set row to a local MedGemma service.

Decision sets (harness/study/build_decision_sets.py):
- candidate_judgement : per candidate -> {answer: lesion|not_lesion|uncertain, confidence}
- done_check          : per first finish review -> {answer: done|not_done|uncertain, confidence}
- wrong_structure     : per first-prompt state -> {answer: wrong_structure|structures_ok|uncertain, confidence}

Strict JSON with the explicit `uncertain` option (MedGemmaClient.chat_json); parse
failures are recorded, never retried. A parse failure is a FINAL outcome: the row is
completed with ok=True and parse_error set, stays in responses.jsonl on resume and in
the scorer's denominators (as a wrong answer). Only transport/HTTP failures are
ok=False and retried on resume. Prompts come from
harness/study/prompts/decision_<set>.md (sha256 of the raw template BYTES recorded);
only image paths and the GT-free `text` enter the prompt — labels/precision/dice stay
offline.

Latency rows (review M3): `service_latency_ms` = server-reported inference time (None
when the server does not report it), `wall_ms` = whole-call client wall clock;
`latency_ms` keeps its legacy meaning (client request wall time).

Output: results/phase5/studyB/<set>/<run>/responses.jsonl (+ meta.json), resumable:
ok rows are skipped on resume (INCLUDING parse failures — they are final outcomes),
only transport/HTTP error rows are dropped and retried (the ask.ts convention).
Frontier models run through harness/study/replay_decisions.ts (same manifests,
same templates, same scoring).

Run:  uv run --project harness/server python harness/study/replay_local.py \
          --set done_check --run medgemma-q4 [--model medgemma_q4|medgemma_bf16]
"""
from __future__ import annotations

import argparse
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

HISTOTYPE_TEXT = {
    "F": "fibrous soft-tissue tumor",
    "L": "lipomatous soft-tissue tumor",
    "M": "myxoid soft-tissue tumor",
    "N": "peripheral nerve sheath tumor",
    "V": "vascular soft-tissue tumor",
}
SEQUENCE_TEXT = {"T1": "T1-weighted", "T2": "T2-weighted fat-suppressed"}
DEFAULT_RESULTS_ROOT = "results/phase5/studyB"
# set -> user prompt template (shared with replay_decisions.ts / the frontier replay)
SET_TEMPLATE = {
    "candidate_judgement": "decision_candidate.md",
    "done_check": "decision_done.md",
    "wrong_structure": "decision_wrong_structure.md",
}


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
    import re

    return re.sub(r"\{\{(\w+)\}\}", lambda m: vars_.get(m.group(1), m.group(0)), text)


def resolve_image(row: dict, images_root: Path | None) -> list[Path]:
    """Manifest image paths: relative to the SOURCE tree root (row['images'] +
    row['source_root'] / set meta), or absolute at build time (images_abs)."""
    paths = []
    abs_list = row.get("images_abs") or []
    for i, rel in enumerate(row["images"]):
        if i < len(abs_list) and Path(abs_list[i]).exists():
            paths.append(Path(abs_list[i]))
        elif images_root:
            paths.append(images_root / rel)
        elif Path(rel).exists():  # already relative to the current project
            paths.append(Path(rel))
        else:
            raise FileNotFoundError(f"{row['key']}: image {rel} not found "
                                    f"(pass --images-root <source tree root>)")
    return paths


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", required=True, dest="set_name",
                    help=f"set under results/phase5/studyB (built by build_decision_sets.py); "
                         f"one of: candidate_judgement, done_check, wrong_structure")
    ap.add_argument("--run", required=True, help="run name under results/phase5/studyB/<set>/")
    ap.add_argument("--model", default="medgemma_q4", choices=("medgemma_q4", "medgemma_bf16"))
    ap.add_argument("--url", default=None, help="service URL (default: port 8091 for q4, 8094 for bf16)")
    ap.add_argument("--keys", default="", help="comma-separated manifest keys (default: all)")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--images-root", default=None)
    args = ap.parse_args(argv)
    if args.url is None:
        args.url = "http://127.0.0.1:8091" if args.model == "medgemma_q4" else "http://127.0.0.1:8094"
    return args


def build_call(row: dict, rep: int, opts, template_path: Path, template: str) -> dict:
    prompt_text = render_template(template, {
        "text": row.get("text", ""), "case": row["case"],
        "target": HISTOTYPE_TEXT.get(row["histotype"], row["histotype"]),
        "sequence_text": SEQUENCE_TEXT.get(row["sequence"], row["sequence"]),
        "candidate_id": (row.get("candidate") or {}).get("id", ""),
    })
    images = resolve_image(row, Path(opts.images_root) if opts.images_root else None)
    line = {
        "run": opts.run, "key": row["key"], "case": row["case"], "set": row["set"],
        "histotype": row["histotype"], "sequence": row["sequence"],
        "source_run": row.get("source_run"), "version": row.get("version"),
        "model": opts.model, "rep": rep,
        "model_id": None, "prompt_template": f"harness/study/prompts/{template_path.name}",
        "prompt_sha256": hashlib.sha256(template_path.read_bytes()).hexdigest(),
        "prompt_text": prompt_text, "images": row["images"],
        "raw": "", "parsed": None, "parse_error": None, "uncertain": None,
        "usage": None, "latency_ms": None, "service_latency_ms": None, "wall_ms": None,
        "ok": False, "error": None,
        "created_at": utcnow(),
    }
    client = MedGemmaClient(base_url=opts.url, timeout_s=opts.timeout)
    t0 = time.perf_counter()
    try:
        result = client.chat_json(prompt_text.strip(), images, max_tokens=opts.max_tokens)
        line.update({
            "raw": result.text, "parsed": result.parsed, "parse_error": result.parse_error,
            "uncertain": result.uncertain, "usage": result.usage,
            "latency_ms": result.wall_ms,                  # legacy field: client wall time
            "service_latency_ms": result.service_latency_ms,  # server-reported (None if absent)
            "model_id": result.model,
            "raw_body": result.raw_body[:2000],
            # Receiving a completion is the outcome: a JSON parse failure is itself
            # recorded (ok=True, parse_error set) — a final result, kept on resume and
            # counted by the scorer, NEVER retried (review P1-3). Only the except branch
            # (transport/HTTP) is ok=False and retried on resume.
            "ok": True,
        })
    except Exception as exc:  # transport/HTTP failure: record, retry on resume
        line["error"] = f"{type(exc).__name__}: {exc}"
    # wall_ms = whole-call client wall clock (review M3), service_latency_ms = server-reported
    line["wall_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
    return line


# Immutable run configuration (review P2-4): resuming must never mix variants.
IMMUTABLE_META_KEYS = ("set", "model", "reps", "max_tokens", "manifest", "manifest_sha256",
                       "selected_keys", "images_root", "prompt_template", "prompt_sha256")


def check_immutable_config(run_dir: Path, meta: dict, done: set) -> None:
    """Refuse to resume a run whose immutable configuration drifted from the prior
    meta.json (a changed prompt/decoding would mix variants in one aggregate)."""
    if not done:
        return
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return
    prior = json.loads(meta_path.read_text())
    drifted = {k: (prior.get(k), meta.get(k)) for k in IMMUTABLE_META_KEYS
               if prior.get(k) != meta.get(k)}
    if drifted:
        raise SystemExit(f"run config drifted from prior meta.json for {run_dir}; "
                         f"use a new --run name. Drifted keys: {drifted}")


def load_done_and_prune(resp_path: Path) -> set[tuple]:
    """Resume bookkeeping (ask.ts convention): returns the completed (key, model, rep)
    keys; rewrites the file without error/malformed/duplicate rows so a rerun with
    the same --run retries exactly the failures. Parse failures are NOT failures here:
    they are completed rows (ok=True, parse_error set) and are never retried (review
    P1-3)."""
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
            key = (rec["key"], rec["model"], rec["rep"])
            if key not in done:
                done.add(key)
                kept.append(line)
    if len(kept) != n_seen or len(done) != len(kept):
        resp_path.write_text("".join(l + "\n" for l in kept))
        print(f"[replay] resume: dropping {n_seen - len(kept)} row(s) for retry")
    return done


def main() -> None:
    opts = parse_args()
    set_dir = PROJECT_ROOT / DEFAULT_RESULTS_ROOT / opts.set_name
    manifest_path = set_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"no decision-set manifest at {manifest_path} — run build_decision_sets.py")
    rows = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
    if opts.keys:
        wanted = {k.strip() for k in opts.keys.split(",") if k.strip()}
        missing = wanted - {r["key"] for r in rows}
        if missing:
            raise SystemExit(f"unknown keys: {sorted(missing)}")
        rows = [r for r in rows if r["key"] in wanted]

    template_path = PROJECT_ROOT / "harness" / "study" / "prompts" / SET_TEMPLATE[opts.set_name]
    if not template_path.exists():
        raise SystemExit(f"no decision prompt template: {template_path}")
    template = template_path.read_text()

    run_dir = set_dir / opts.run
    run_dir.mkdir(parents=True, exist_ok=True)
    resp_path = run_dir / "responses.jsonl"

    # Resumability (ask.ts convention): keep ok rows (including parse failures — they
    # are final outcomes); drop error/malformed rows so a rerun with the same --run
    # retries exactly the transport/HTTP failures.
    done = load_done_and_prune(resp_path)
    todo = [(row, rep) for row in rows for rep in range(opts.reps)
            if (row["key"], opts.model, rep) not in done]

    meta = {
        "run": opts.run, "stage": "phase5_studyB_decision_replay", "created_at": utcnow(),
        **git_provenance(),
        "set": opts.set_name, "model": opts.model, "url": opts.url, "reps": opts.reps,
        "max_tokens": opts.max_tokens, "concurrency": opts.concurrency,
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "selected_keys": [r["key"] for r in rows],
        "images_root": opts.images_root,
        "prompt_template": f"harness/study/prompts/{template_path.name}",
        "prompt_sha256": hashlib.sha256(template_path.read_bytes()).hexdigest(),
        "planned_calls": len(rows) * opts.reps, "resumed_from_lines": len(done),
        "gt_fields_never_sent": ["label", "precision", "dice_state", "dice_best",
                                 "version_dice_gain", "oracle_dice_gain", "region_vox",
                                 "inter_vox", "dice_reviewed"],
    }
    check_immutable_config(run_dir, meta, done)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"[replay] set={opts.set_name} model={opts.model} rows={len(rows)} "
          f"todo={len(todo)} (done {len(done)}) url={opts.url}")

    lock = threading.Lock()
    error_rows = 0

    def worker(call):
        row, rep = call
        return build_call(row, rep, opts, template_path, template)

    with ThreadPoolExecutor(max_workers=max(1, opts.concurrency)) as pool:
        futures = {pool.submit(worker, call): call for call in todo}
        for fut in as_completed(futures):
            line = fut.result()
            with lock:
                with open(resp_path, "a") as fh:
                    fh.write(json.dumps(line) + "\n")
                if not line["ok"]:
                    error_rows += 1
            print(f"[replay] {line['key']}/r{line['rep']} ok={line['ok']} "
                  f"{line['latency_ms']}ms"
                  + (f" error={line['error']}" if not line["ok"] else "")
                  + (f" parse_error={line['parse_error']}" if line.get("parse_error") else ""),
                  flush=True)

    if error_rows:
        print(f"[replay] {error_rows} transport error rows recorded (rerun the same --run "
              f"to retry; parse failures are final and were kept)")
        sys.exit(3)
    print(f"[replay] wrote {resp_path}")


if __name__ == "__main__":
    main()
