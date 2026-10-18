"""Export the compact, text-only record of a run from results/ (DVC-tracked, heavy) into git.

    uv run --project harness/server python eval/v2/export_run.py results/phase1-05 [results/...]
    uv run --project harness/server python eval/v2/export_run.py --all        # every known run
    uv run --project harness/server python eval/v2/export_run.py --index-only # rebuild the registry

Why: results/ holds images, masks, versions and full pi sessions (hundreds of MB per pass) and is
tracked with DVC. Everything needed to mine the experiments later (scores, per-case metadata, every
segd operation, the agent's full transcript without image payloads, Phase 2 raw answers) is small
text; it is copied here into research/experiments/runs/<run_id>/ and committed to git, so the
experiment log can cite exact files. Image payloads in transcripts are replaced by
{"type": "image", "omitted": true, "sha256": ..., "bytes": ...}; the originals stay in results/ (DVC).

Registry: research/experiments/runs.csv (one row per exported run; rebuilt on every call).
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = ROOT / "research" / "experiments" / "runs"
INDEX = ROOT / "research" / "experiments" / "runs.csv"

PHASE1_RUN_FILES = ["batch_summary.json", "scores.csv", "scores.md", "versions_dice.csv"]
PHASE1_CASE_FILES = ["meta.json", "case_meta.json", "trajectory.jsonl"]
PHASE2_FILES = ["meta.json", "responses.jsonl", "scored.jsonl", "score.json", "report.md"]
PREPARE_FILES = ["manifest.jsonl", "prepare_summary.json"]
ONECLICK_FILES = ["results.csv", "summary.md"]


def _strip_images(obj):
    """Recursively replace base64 image payloads with a hash stub."""
    if isinstance(obj, dict):
        if obj.get("type") == "image" and isinstance(obj.get("data"), str):
            raw = obj["data"]
            try:
                b = base64.b64decode(raw, validate=False)
            except Exception:
                b = raw.encode()
            return {"type": "image", "omitted": True, "mimeType": obj.get("mimeType"),
                    "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}
        return {k: _strip_images(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_images(v) for v in obj]
    return obj


def _strip_session(src: Path, dst: Path) -> None:
    with open(src) as f, open(dst, "w") as g:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                g.write(json.dumps(_strip_images(json.loads(line)), ensure_ascii=False) + "\n")
            except json.JSONDecodeError:
                g.write(line + "\n")


def _copy(src: Path, dst: Path) -> bool:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def kind_of(run_dir: Path) -> str:
    if (run_dir / "batch_summary.json").exists():
        return "phase1_agent"
    if (run_dir / "responses.jsonl").exists():
        return "phase2_localization"
    if (run_dir / "manifest.jsonl").exists():
        return "phase2_prepare"
    if (run_dir / "results.csv").exists():
        return "oneclick_baseline"
    return "unknown"


def run_id_of(run_dir: Path) -> str:
    rel = run_dir.resolve().relative_to(RESULTS.resolve())
    return "__".join(rel.parts)       # phase2_dev/dev-01 -> phase2_dev__dev-01


def export(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    kind = kind_of(run_dir)
    dst = OUT / run_id_of(run_dir)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    if kind == "phase1_agent":
        for f in PHASE1_RUN_FILES:
            _copy(run_dir / f, dst / f)
        for case_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            for f in PHASE1_CASE_FILES:
                _copy(case_dir / f, dst / "cases" / case_dir.name / f)
            if (case_dir / "session.jsonl").exists():
                (dst / "cases" / case_dir.name).mkdir(parents=True, exist_ok=True)
                _strip_session(case_dir / "session.jsonl", dst / "cases" / case_dir.name / "transcript.jsonl")
    elif kind == "phase2_localization":
        for f in PHASE2_FILES:
            _copy(run_dir / f, dst / f)
    elif kind == "phase2_prepare":
        for f in PREPARE_FILES:
            _copy(run_dir / f, dst / f)
    elif kind == "oneclick_baseline":
        for f in ONECLICK_FILES:
            _copy(run_dir / f, dst / f)
    else:
        raise SystemExit(f"unknown run layout: {run_dir}")
    (dst / "EXPORT.json").write_text(json.dumps({
        "source": str(run_dir.relative_to(ROOT)), "kind": kind,
        "note": "text-only export; heavy artifacts (renders, masks, versions, full sessions) are in "
                "results/ (DVC). Transcript image payloads are replaced by sha256 stubs."}, indent=2) + "\n")
    print(f"[export] {run_dir.relative_to(ROOT)} -> {dst.relative_to(ROOT)} ({kind})")
    return dst


def _summ(vals: list[float]) -> tuple[str, str]:
    if not vals:
        return "", ""
    return f"{statistics.mean(vals):.3f}", f"{statistics.median(vals):.3f}"


def rebuild_index() -> None:
    rows = []
    for d in sorted(p for p in OUT.iterdir() if p.is_dir()):
        exp = json.loads((d / "EXPORT.json").read_text()) if (d / "EXPORT.json").exists() else {}
        kind = exp.get("kind", "")
        row = {"run_id": d.name, "kind": kind, "source": exp.get("source", ""), "started_at": "",
               "model": "", "thinking": "", "git": "", "harness_sha256": "", "n": "",
               "mean_dice": "", "median_dice": "", "extra": "", "note": ""}
        if kind == "phase1_agent":
            bs = json.loads((d / "batch_summary.json").read_text())
            row.update(started_at=bs.get("started_at", ""), model=bs.get("model", ""),
                       thinking=bs.get("thinking_level", ""), git=bs.get("git_describe") or "",
                       harness_sha256=bs.get("harness_sha256", ""), note=bs.get("note") or "")
            if (d / "scores.csv").exists():
                sc = list(csv.DictReader(open(d / "scores.csv")))
                dice = [float(r["dice"]) for r in sc if r.get("dice") not in ("", None, "nan")]
                row["n"] = str(len(sc))
                row["mean_dice"], row["median_dice"] = _summ(dice)
                tc = [int(r["n_tool_calls"]) for r in sc if r.get("n_tool_calls")]
                mp = [int(r["n_model_prompts"]) for r in sc if r.get("n_model_prompts")]
                loc = sum(1 for r in sc if r.get("tag") == "localization")
                row["extra"] = (f"tool_calls_mean={statistics.mean(tc):.2f}; prompts_mean={statistics.mean(mp):.2f}; "
                                f"localization_failures={loc}") if tc else ""
        elif kind == "phase2_localization":
            meta = json.loads((d / "meta.json").read_text())
            sc = json.loads((d / "score.json").read_text()) if (d / "score.json").exists() else {}
            row.update(started_at=meta.get("created_at", ""), model=meta.get("model", ""),
                       thinking=meta.get("thinking", ""), git=meta.get("git_describe") or "")
            conds = sc.get("aggregate") or {}
            parts = []
            for c, v in conds.items():
                if isinstance(v, dict) and "hit_rate_mean_over_reps" in v:
                    parts.append(f"{c}={v['hit_rate_mean_over_reps']:.3f}")
                    row["n"] = str(v.get("n_cases", ""))
            row["extra"] = "hit: " + ", ".join(parts) if parts else ""
        elif kind == "oneclick_baseline":
            sc = list(csv.DictReader(open(d / "results.csv")))
            dice = [float(r["dice_all"]) for r in sc if r.get("dice_all") not in ("", None)]
            row["n"] = str(len(dice))
            row["mean_dice"], row["median_dice"] = _summ(dice)
            row["model"] = "nnInteractive one click at GT centroid (no LLM)"
        rows.append(row)
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run_id"])
        w.writeheader()
        w.writerows(rows)
    print(f"[export] registry {INDEX.relative_to(ROOT)}: {len(rows)} runs")


def all_runs() -> list[Path]:
    out = []
    for p in sorted(RESULTS.iterdir()):
        if not p.is_dir() or p.name in ("cache",):
            continue
        if kind_of(p) != "unknown":
            out.append(p)
        else:
            for q in sorted(p.iterdir()):
                if q.is_dir() and kind_of(q) != "unknown":
                    out.append(q)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--index-only", action="store_true")
    a = ap.parse_args()
    if not a.index_only:
        runs = all_runs() if a.all else [Path(r) for r in a.runs]
        if not runs:
            sys.exit("pass run dirs, --all, or --index-only")
        for r in runs:
            export(r)
    rebuild_index()


if __name__ == "__main__":
    main()
