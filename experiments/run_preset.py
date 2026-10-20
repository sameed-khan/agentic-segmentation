#!/usr/bin/env python3
"""Run, show or list an experiment preset (experiments/presets/**/*.json).

    python experiments/run_preset.py list [--kind agent_run|study]
    python experiments/run_preset.py show <name>
    python experiments/run_preset.py run <name> [--dry-run] [-- <extra run_cases.ts flags>]

``agent_run`` presets are executed by the agent runner (`npx tsx run_cases.ts --preset <name>`
in harness/runner; extra flags after ``--`` override the preset). ``study`` presets (Phase 5
Study A/B scripts, frontier pointing, 3D probes) are a recorded list of commands, run in order
from their ``cwd`` (relative to the repo root); ``{repo}`` in an argument is the repo root.
Services (segd, nnInteractive, local-model servers) must already be running; this launcher
starts nothing and stdlib only (python >= 3.9).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PRESET_DIR = REPO / "experiments" / "presets"
KINDS = ("agent_run", "study")


def preset_files() -> dict[str, Path]:
    """name -> file for agent presets (presets/*.json) and study presets (presets/studies/*.json)."""
    files: dict[str, Path] = {}
    for f in sorted(PRESET_DIR.glob("*.json")) + sorted((PRESET_DIR / "studies").glob("*.json")):
        if f.stem in files:
            raise SystemExit(f"duplicate preset name {f.stem!r}: {files[f.stem]} and {f}")
        files[f.stem] = f
    return files


def load_preset(name: str) -> dict:
    files = preset_files()
    if name not in files:
        raise SystemExit(f"unknown preset {name!r} (python experiments/run_preset.py list)")
    preset = json.loads(files[name].read_text())
    if preset.get("name") != name or preset.get("schema") != 1 or preset.get("kind") not in KINDS:
        raise SystemExit(f"{files[name]}: bad schema/name/kind")
    return preset


def commands_for(preset: dict, extra: list[str] | None = None) -> list[dict]:
    """The commands a preset stands for: [{cwd (absolute), argv, env}]."""
    extra = extra or []
    if preset["kind"] == "agent_run":
        return [{"cwd": str(REPO / "harness" / "runner"),
                 "argv": ["npx", "tsx", "run_cases.ts", "--preset", preset["name"], *extra], "env": {}}]
    if extra:
        raise SystemExit("extra flags are only supported for agent_run presets")
    return [{"cwd": str((REPO / c["cwd"]).resolve()),
             "argv": [a.replace("{repo}", str(REPO)) for a in c["argv"]],
             "env": c.get("env", {})} for c in preset["study"]["commands"]]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_list = sub.add_parser("list")
    p_list.add_argument("--kind", choices=KINDS)
    sub.add_parser("show").add_argument("name")
    p_run = sub.add_parser("run")
    p_run.add_argument("name")
    p_run.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        for name in preset_files():
            p = load_preset(name)
            if args.kind in (None, p["kind"]):
                print(f"{name}\t{p['kind']}\t{p['provenance']['log_entry']}\t{p['provenance']['git_tag']}")
        return 0
    preset = load_preset(args.name)
    if args.cmd == "show":
        print(json.dumps(preset, indent=2))
        return 0
    for cmd in commands_for(preset, extra):
        line = " ".join(cmd["argv"])
        print(f"[preset {preset['name']}] (cd {cmd['cwd']}) {line}", flush=True)
        if args.dry_run:
            continue
        rc = subprocess.run(cmd["argv"], cwd=cmd["cwd"], env={**os.environ, **cmd["env"]}).returncode
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
