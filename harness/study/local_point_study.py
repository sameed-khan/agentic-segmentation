#!/usr/bin/env python3
"""Phase 5 Study A (standalone localization): run local pointer/medical models on the
Phase 2 prepare manifests (design or test), collecting responses for offline scoring.

Models (all via their local HTTP services, decoding deterministic; see
harness/local_models/ and research/phase5_implementation.md):
- gui_actor    : pointd-gui-actor :8092 (GUI-Actor-2B, official pointer head) — conditions raw, marks
- molmo2       : pointd-molmo2   :8093 (Molmo2-8B, <points> output)          — conditions raw, marks
- molmopoint   : pointd-molmopoint :8095 (MolmoPoint-8B, grounding tokens)   — conditions raw, marks
- qwen3vl      : pointd-qwen3vl  :8096 (Qwen3-VL-8B-Instruct, point_2d JSON) — conditions raw, marks
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

--prompt COND=PATH (repeatable) replaces the default template for one condition
(harness/study/prompts/local_point_<COND>.md; PATH relative to the repo root or
absolute). With --models selecting a single model this gives per-model×condition
prompt variants; the effective template path + sha256 per condition are recorded in
meta.json `prompts` and are immutable on resume (legacy metas that used the old
`local_point_<COND>` keys still resume when the templates are identical).

Latency semantics (review M3): every response row records BOTH `service_latency_ms`
(server-reported inference time, queue wait behind the per-process inference lock
excluded: pointd `latency_ms` field; BF16 server `latency_ms` extension; llama-server
`timings.prompt_ms + timings.predicted_ms`; None when a server does not report it) and
`wall_ms` (client wall clock around the whole call, queue wait included). The scorer
reports service latency and says so. `latency_ms` keeps its legacy per-model meaning
(pointd: server latency; chat: client wall time) for old rows/old scorers.

Service-version resume check (review M4): /healthz is fetched once per selected model
at run start; resuming refuses when an existing ok row's `service_version` differs from
the live service's version (the 1.0.1 servers serialise inference; responses collected
before that lock under concurrency > 1 may be corrupted and must not be mixed into a
run). Chat rows carry no per-call version (llama-server exposes none in its payload),
so they are stamped with the /healthz version fetched at start; llama-server's /healthz
has no `version` field at all — its rows keep `service_version: null` and the check is
skipped for that model (documented limitation: a q4 run cannot be verified this way).

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
    "molmopoint": {"kind": "pointd", "port": 8095, "conditions": ("raw", "marks")},
    "qwen3vl": {"kind": "pointd", "port": 8096, "conditions": ("raw", "marks")},
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

# condition -> manifest image field of the row (the default template is
# harness/study/prompts/local_point_<cond>.md; --prompt overrides replace it per condition)
CONDITION_SPEC = {
    "raw": "image_R",
    "marks": "image_S",
    "raw_box": "image_R",
    "marks_box": "image_S",
    "marks_id": "image_S",
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


def resolve_prompt_overrides(specs: list[str], root: Path) -> dict[str, Path]:
    """Parse --prompt COND=PATH specs (repeatable): PATH (absolute, or relative to the
    repo root) replaces the default local_point_<COND>.md template for that condition.
    Unknown condition or missing file -> SystemExit with a clear message."""
    overrides: dict[str, Path] = {}
    for spec in specs:
        cond, sep, path = spec.partition("=")
        if not sep or not cond.strip() or not path:
            raise SystemExit(f"--prompt expects COND=PATH, got {spec!r} "
                             f"(known conditions: {', '.join(ALL_CONDITIONS)})")
        cond = cond.strip()
        if cond not in ALL_CONDITIONS:
            raise SystemExit(f"unknown --prompt condition: {cond} "
                             f"(known: {', '.join(ALL_CONDITIONS)})")
        template = Path(path)
        if not template.is_absolute():
            template = root / template
        if not template.is_file():
            raise SystemExit(f"--prompt {cond}: template not found at {template}")
        overrides[cond] = template
    return overrides


def template_for(condition: str, overrides: dict[str, Path] | None = None) -> tuple[Path, str]:
    """The effective prompt template for a condition: the --prompt override, else the
    default harness/study/prompts/local_point_<cond>.md. Returns (path, provenance
    label as recorded in responses.jsonl `prompt_template` / meta.json `prompts`)."""
    path = (overrides or {}).get(condition) or PROMPTS_DIR / f"local_point_{condition}.md"
    try:
        label = str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        label = str(path)
    return path, label


def prompt_provenance(overrides: dict[str, Path] | None = None) -> dict:
    """Effective template path + sha256 per condition (default or --prompt override);
    part of the immutable-on-resume config (IMMUTABLE_META_KEYS: `prompts`)."""
    out = {}
    for cond in ALL_CONDITIONS:
        path, label = template_for(cond, overrides)
        out[cond] = {"path": label, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return out


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
            "uncertain": result.uncertain, "wall_ms": result.wall_ms,
            "service_latency_ms": result.service_latency_ms,
            "model": result.model, "usage": result.usage, "raw_body": result.raw_body}


def fetch_service_versions(urls: dict[str, str], models: list[str]) -> dict[str, str | None]:
    """GET /healthz once per selected model at run start (service-version resume check,
    review M4). Returns each service's `version`; None when the endpoint exposes no
    version field (llama-server). A service that is not up exits: the runner cannot make
    its calls anyway, and a skipped check could silently mix pre/post-lock responses."""
    import httpx

    out: dict[str, str | None] = {}
    for model in models:
        try:
            with httpx.Client(timeout=10.0) as client:
                base = urls[model].rstrip("/")
                r = client.get(base + "/healthz")
                if r.status_code == 404:
                    # llama-server (Q4 arm) has no /healthz route, only /health; it exposes
                    # no version and runs --parallel 1 (never had the concurrency hazard),
                    # so the version check is skipped for it (version None). Review B1.
                    r = client.get(base + "/health")
                    r.raise_for_status()
                    out[model] = None
                    continue
                r.raise_for_status()
                payload = r.json()
            out[model] = payload.get("version") if isinstance(payload, dict) else None
        except Exception as exc:
            raise SystemExit(f"{model}: /healthz failed at {urls[model]} "
                             f"({type(exc).__name__}: {exc}) — start the service first; the "
                             "service_version resume check needs the live version")
    return out


def check_service_version_resume(prior: dict[str, set], live: dict[str, str | None]) -> None:
    """Refuse to resume when an existing ok row's `service_version` (per model) differs
    from the live service's /healthz version (review M4: the 1.0.1 servers added the
    inference lock; responses collected before it under concurrency > 1 may be corrupted
    and must not be mixed into one aggregate).

    A live version of None means the service exposes no version (llama-server): the
    check is skipped for that model. Existing rows with no version at all (chat rows
    recorded before this check) cannot be verified either -> also refused when a live
    version is known."""
    for model, versions in sorted(prior.items()):
        live_v = live.get(model)
        if live_v is None:
            continue  # llama-server: no version to check against (documented limitation)
        bad = sorted({"null" if v is None else str(v) for v in versions if v != live_v})
        if bad:
            raise SystemExit(
                f"{model}: existing ok rows record service_version {bad} but the live "
                f"service reports {live_v!r} (pre-lock responses must not be mixed into "
                "this run) — use a new --run name or re-run the affected model")


# ------------------------------------------------------------------ main loop
def build_call(row: dict, model: str, condition: str, rep: int, opts) -> dict:
    """Build and execute one service call; returns the responses.jsonl line."""
    spec = MODELS[model]
    image_field = CONDITION_SPEC[condition]
    template_path, template_label = template_for(condition, opts.prompt_overrides)
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
        "prompt_template": template_label,
        "prompt_sha256": hashlib.sha256(template_path.read_bytes()).hexdigest(),
        "prompt_text": prompt_text, "image": row[image_field], "k": opts.k,
        "raw": "", "points": [], "parsed": None, "parse_error": None, "uncertain": None,
        "usage": None, "latency_ms": None, "service_latency_ms": None, "wall_ms": None,
        "ok": False, "error": None,
        "created_at": utcnow(),
    }
    t0 = time.perf_counter()
    try:
        if spec["kind"] == "pointd":
            out = call_pointd(opts.urls[model], image, prompt_text.strip(), opts.k, opts.timeout)
            line.update({
                "raw": out.get("raw", ""), "points": out.get("points", []),
                "model_id": out.get("model"), "service_version": out.get("version"),
                "latency_ms": out.get("latency_ms"),          # legacy field: server latency
                "service_latency_ms": out.get("latency_ms"),  # server-reported inference time
                "verifier": out.get("verifier"),
            })
        else:
            out = call_chat(MedGemmaClient(base_url=opts.urls[model], timeout_s=opts.timeout),
                            prompt_text.strip(), image, opts.max_tokens)
            line.update({
                "raw": out["raw"], "parsed": out["parsed"], "parse_error": out["parse_error"],
                "uncertain": out["uncertain"], "model_id": out["model"],
                "latency_ms": out["wall_ms"],                 # legacy field: client wall time
                "service_latency_ms": out["service_latency_ms"],
                "usage": out["usage"],
                "raw_body": out["raw_body"][:2000],
                # No per-call version from llama-server: stamp the /healthz version fetched
                # once at run start (empty in tests that bypass the fetch).
                "service_version": (opts.service_versions or {}).get(model),
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
    ap.add_argument("--prompt", action="append", default=[], metavar="COND=PATH",
                    help="replace the default prompt template for one condition (repeatable; "
                         "PATH is relative to the repo root or absolute), e.g. "
                         "--prompt marks=harness/study/prompts/variants/molmo_marks_v2.md")
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
    ap.add_argument("--max-tokens", type=int, default=2048, help="max new tokens for MedGemma calls (2048: room for MedGemma 1.5 thinking mode)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=900.0, help="per-call timeout seconds")
    ap.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--gui-actor-url", default=None)
    ap.add_argument("--molmo2-url", default=None)
    ap.add_argument("--molmopoint-url", default=None)
    ap.add_argument("--qwen3vl-url", default=None)
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
        "molmopoint": args.molmopoint_url or f"http://127.0.0.1:{MODELS['molmopoint']['port']}",
        "qwen3vl": args.qwen3vl_url or f"http://127.0.0.1:{MODELS['qwen3vl']['port']}",
        "medgemma_q4": args.medgemma_q4_url or f"http://127.0.0.1:{MODELS['medgemma_q4']['port']}",
        "medgemma_bf16": args.medgemma_bf16_url or f"http://127.0.0.1:{MODELS['medgemma_bf16']['port']}",
    }
    args.service_versions: dict[str, str | None] = {}  # filled by main() from /healthz
    args.conditions_filter = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for cond in args.conditions_filter:
        if cond not in ALL_CONDITIONS:
            raise SystemExit(f"unknown condition: {cond} (known: {', '.join(ALL_CONDITIONS)})")
    args.prompt_overrides = resolve_prompt_overrides(args.prompt, PROJECT_ROOT)
    return args


def load_done_and_prune(resp_path: Path) -> tuple[set, dict[str, set]]:
    """Resume bookkeeping (ask.ts convention): returns the completed ok keys AND the
    service_version values recorded per model (service-version resume check, review
    M4); rewrites the file without error/malformed/duplicate rows so a rerun with the
    same --run retries exactly the failures."""
    done: set[tuple] = set()
    prior_versions: dict[str, set] = {}
    kept: list[str] = []
    n_seen = 0
    if not resp_path.exists():
        return done, prior_versions
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
                prior_versions.setdefault(rec["model"], set()).add(rec.get("service_version"))
                kept.append(line)
    if len(kept) != n_seen or len(done) != len(kept):
        resp_path.write_text("".join(l + "\n" for l in kept))
        print(f"[studyA] resume: dropping {n_seen - len(kept)} row(s) for retry")
    return done, prior_versions


def _normalise_legacy_prompt_keys(prompts) -> dict:
    """meta.json `prompts` keys were renamed `local_point_<cond>` -> `<cond>` (Study A
    fix round): old runs stay resumable when the templates themselves are identical."""
    out = {}
    for key, value in (prompts or {}).items():
        out[key[len("local_point_"):] if key.startswith("local_point_") else key] = value
    return out


# Immutable run configuration (review P2-4): resuming must never mix variants.
IMMUTABLE_META_KEYS = ("models", "conditions", "reps", "k", "max_tokens",
                       "prepare_manifest", "prepare_manifest_sha256", "selected_cases",
                       "images_root", "prompts")


def check_immutable_config(run_dir: Path, meta: dict, done: set) -> None:
    """Refuse to resume a run whose immutable configuration drifted from the prior
    meta.json (a changed prompt/image/decoding config would mix variants in one
    aggregate); require a new run name on mismatch. Legacy `prompts` key spellings are
    normalised before the comparison (see _normalise_legacy_prompt_keys)."""
    if not done:
        return
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return
    prior = json.loads(meta_path.read_text())
    drifted = {}
    for k in IMMUTABLE_META_KEYS:
        p, m = prior.get(k), meta.get(k)
        if k == "prompts":
            p, m = _normalise_legacy_prompt_keys(p), _normalise_legacy_prompt_keys(m)
        if p != m:
            drifted[k] = (prior.get(k), meta.get(k))
    if drifted:
        raise SystemExit(f"run config drifted from prior meta.json for {run_dir}; "
                         f"use a new --run name. Drifted keys: {drifted}")


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
    done, prior_service_versions = load_done_and_prune(resp_path)
    todo = [c for c in calls if (c[0]["case"], c[1], c[2], c[3]) not in done]

    # Service-version resume check (review M4): /healthz once per selected model, then
    # refuse to mix rows from a different service build (pre-lock responses may be
    # corrupted; llama-server exposes no version -> skipped, see the docstring).
    opts.service_versions = fetch_service_versions(opts.urls, opts.models)
    check_service_version_resume(prior_service_versions, opts.service_versions)

    meta = {
        "run": opts.run, "stage": "phase5_studyA_local_point", "created_at": utcnow(),
        **git_provenance(),
        "models": opts.models, "conditions": opts.conditions_filter or "per-model defaults",
        "reps": opts.reps, "k": opts.k, "max_tokens": opts.max_tokens,
        "prepare_manifest": str(manifest_path),
        "prepare_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "selected_cases": [r["case"] for r in rows],
        "images_root": opts.images_root,
        "urls": opts.urls, "timeout_s": opts.timeout, "concurrency": opts.concurrency,
        "service_versions": opts.service_versions,
        "planned_calls": len(calls), "resumed_from_lines": len(done),
        "prompts": prompt_provenance(opts.prompt_overrides),
    }
    check_immutable_config(run_dir, meta, done)
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
