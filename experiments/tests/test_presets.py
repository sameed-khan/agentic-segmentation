"""Preset files: schema, provenance, consistency with the exported run records, launcher.

Run:  cd <repo> && uv run --project harness/server pytest -q experiments/tests
(stdlib + pytest only). The agent runner's own loader/expansion is tested in
harness/runner/test/presets.test.mts.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments"))
import run_preset  # noqa: E402

FILES = run_preset.preset_files()
AGENT = sorted(n for n in FILES if json.loads(FILES[n].read_text())["kind"] == "agent_run")
STUDY = sorted(n for n in FILES if json.loads(FILES[n].read_text())["kind"] == "study")
EXPORTS = REPO / "research" / "experiments" / "runs"
LOGS = "".join(p.read_text() for p in sorted((REPO / "research").glob("experiment_log*.md")))

RUNNER_KEYS = {"run_id", "model", "thinking", "dataset", "split", "cases", "concurrency", "wall_min",
               "max_nudges", "max_tool_calls", "max_model_prompts", "segmenter", "segmenter_url", "v3",
               "system_file", "task_file", "system_prompt_sha256", "note"}


def _preset(name: str) -> dict:
    return json.loads(FILES[name].read_text())


def _git_has(rev: str) -> bool:
    return subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{rev}^{{commit}}"],
                          capture_output=True).returncode == 0


def _test_cases() -> set[str]:
    with open(REPO / "eval" / "v2" / "mstt_cases.csv") as f:
        return {r["case"] for r in csv.DictReader(f) if r["split"].strip() == "test"}


def test_counts_and_expected_presets_present():
    assert len(AGENT) >= 18 and len(STUDY) >= 60
    for want in ["phase3-high", "phase3-xhigh", "p6-sammed3d-high", "p3-opus55-medium", "p3-opus55-high",
                 "p3-opus55-xhigh", "p3-haiku55-xhigh", "p3-haiku55-max", "p3-dsv41flash-xhigh",
                 "p3-dsv41flash-max", "p3-muse13-high", "p3-muse13-xhigh", "p3-gpt61sol-high",
                 "p3-gpt61sol-xhigh", "p3-gpt6luna-xhigh", "p3-gpt6luna-max", "p3-argon4-high"]:
        assert want in AGENT
    # every results/p3-*.dvc has a preset of the same name
    for dvc in (REPO / "results").glob("p3-*.dvc"):
        assert dvc.stem in AGENT, f"no preset for {dvc.name}"
    for run in ("p6-sammed3d-high", "phase3-high", "phase3-xhigh"):
        assert (REPO / "results" / f"{run}.dvc").is_file()


@pytest.mark.parametrize("name", sorted(FILES))
def test_common_schema_and_provenance(name):
    p = _preset(name)
    assert p["schema"] == 1 and p["name"] == name and p["description"]
    assert p["kind"] in run_preset.KINDS
    assert (FILES[name].parent.name == "studies") == (p["kind"] == "study")
    prov = p["provenance"]
    for k in ("git_tag", "git_commit", "results", "log_entry"):
        assert prov[k], f"{name}: provenance.{k}"
    assert re.fullmatch(r"[0-9a-f]{40}", prov["git_commit"])
    # results artifact: the .dvc file named in provenance.results exists
    dvc = REPO / prov["results"].split(" (")[0]
    assert dvc.is_file(), f"{name}: {dvc} missing"
    # every E-number the preset cites exists in the experiment logs
    for e in re.findall(r"E\d+", prov["log_entry"]):
        assert re.search(rf"^#+ {e}\b", LOGS, re.M), f"{name}: no heading for {e} in the experiment logs"
    if _git_has(prov["git_commit"]):
        if subprocess.run(["git", "-C", str(REPO), "rev-parse", "-q", "--verify",
                           f"refs/tags/{prov['git_tag']}"], capture_output=True).returncode == 0:
            # the tag is an ancestor-or-self of the recorded commit (the run code descends from it)
            # — except the Phase 6 tag, which IS the run commit
            r = subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor",
                                f"{prov['git_tag']}^{{commit}}", prov["git_commit"]], capture_output=True)
            assert r.returncode == 0, f"{name}: {prov['git_tag']} is not an ancestor of {prov['git_commit'][:8]}"


@pytest.mark.parametrize("name", AGENT)
def test_agent_run_preset_is_complete_and_consistent(name):
    p = _preset(name)
    r = p["runner"]
    assert set(r) <= RUNNER_KEYS and {"run_id", "model", "thinking", "dataset", "concurrency", "wall_min",
                                      "max_nudges", "max_tool_calls", "max_model_prompts", "segmenter",
                                      "v3", "system_file", "task_file", "note"} <= set(r)
    assert r["run_id"] == name
    assert ("split" in r) != ("cases" in r)
    assert r["dataset"] in ("mstt", "beamster") and r["segmenter"] in ("nninteractive", "sammed3d")
    if "cases" in r:
        assert set(r["cases"]) <= _test_cases() | {"F14", "L20", "M07", "N26", "V37"}
        assert len(r["cases"]) == len(set(r["cases"]))
    assert (REPO / r["system_file"]).is_file() and (REPO / r["task_file"]).is_file()
    # recorded rendered-system-prompt hash == the repo's system.md rendered with the preset budgets
    if "system_prompt_sha256" in r:
        text = (REPO / r["system_file"]).read_text()
        text = text.replace("{{max_tool_calls}}", str(r["max_tool_calls"])).replace(
            "{{max_model_prompts}}", str(r["max_model_prompts"]))
        assert hashlib.sha256(text.encode()).hexdigest() == r["system_prompt_sha256"]
    # v3 not part of any logged experiment so far; if one is added it must be a valid spec
    assert r["v3"] is None or set(r["v3"]) <= {"pointer", "pointer_url", "medgemma", "medgemma_url", "checkpoints"}
    # defaults of the harness-v2.0.1 experiments
    assert (r["wall_min"], r["max_nudges"], r["max_tool_calls"], r["max_model_prompts"]) == (10, 3, 40, 15)
    assert r["concurrency"] == (4 if name.startswith("phase3-") else 6 if name == "p6-sammed3d-high"
                                else 2 if name == "p6-sammed3d-pilot5b" else 8)


@pytest.mark.parametrize("name", AGENT)
def test_agent_preset_matches_exported_run_record(name):
    """Cross-check against the committed export of the original run (batch_summary.json):
    model, thinking, concurrency, segmenter, system prompt hash, case set, code revision."""
    export = EXPORTS / name / "batch_summary.json"
    if not export.is_file():
        pytest.skip(f"no exported batch_summary for {name}")
    rec = json.loads(export.read_text())
    r = _preset(name)["runner"]
    assert rec["model"] == r["model"] and rec["thinking_level"] == r["thinking"]
    assert rec["concurrency"] == r["concurrency"]
    assert rec.get("segmenter", "nninteractive") == r["segmenter"]
    assert rec.get("segmenter_url") == r.get("segmenter_url")
    assert rec["system_prompt_sha256"] == r["system_prompt_sha256"]
    cases = {c["case"] for c in rec["cases"]}
    if "cases" in r:
        assert cases == set(r["cases"])
    else:
        assert r["split"] == "test" and cases <= _test_cases()
    prov = _preset(name)["provenance"]
    if name == "phase3-high":    # last (resume) batch = the V25 rerun on harness-v2.0.1
        assert rec["git_commit"] == prov["git_commit"] == "0b22d7a3250a7b73b521de49baacb634d86fc57e" or \
            rec["git_commit"] in [b["git_commit"] for b in prov["original_batches"]]
    else:
        assert rec["git_commit"] == prov["git_commit"]
        assert rec["git_describe"].startswith(prov["git_describe"].split(" ")[0]) or \
            prov["git_describe"].startswith(rec["git_describe"])


@pytest.mark.parametrize("name", STUDY)
def test_study_preset_commands_reference_existing_scripts(name):
    p = _preset(name)
    cmds = p["study"]["commands"]
    assert cmds
    for c in cmds:
        assert (REPO / c["cwd"]).is_dir()
        script = next(a for a in c["argv"] if a.endswith((".py", ".ts")))
        assert ((REPO / c["cwd"] / script).is_file() or (REPO / script).is_file()), f"{name}: {script}"
        assert all(isinstance(a, str) for a in c["argv"])
    assert not [a for c in cmds for a in c["argv"] if "/home/ubuntu" in a], "absolute machine paths"


def test_study_a_preset_arguments_match_recorded_meta():
    """dA-molmo2-marks-p4: values recorded in results/phase5/studyA/<run>/meta.json."""
    cmds = run_preset.commands_for(run_preset.load_preset("dA-molmo2-marks-p4"))
    argv = cmds[0]["argv"]
    flag = lambda f: argv[argv.index(f) + 1]  # noqa: E731
    assert (flag("--models"), flag("--conditions"), flag("--max-tokens"), flag("--concurrency"),
            flag("--reps"), flag("--k")) == ("molmo2", "marks", "256", "2", "1", "3")
    assert flag("--prompt") == "marks=harness/study/prompts/variants/studyA/point_v4.md"
    assert flag("--prepare-dir") == "results/phase2_dev/prepare"
    assert "score_local_point.py" in " ".join(cmds[1]["argv"])
    assert cmds[1]["argv"][cmds[1]["argv"].index("--baseline-dir") + 1] == "results/phase2_dev/dev-02-high"


def test_launcher_expands_agent_and_study_presets():
    agent = run_preset.commands_for(run_preset.load_preset("p3-opus55-high"), ["--concurrency", "2"])
    assert agent == [{"cwd": str(REPO / "harness" / "runner"),
                      "argv": ["npx", "tsx", "run_cases.ts", "--preset", "p3-opus55-high", "--concurrency", "2"],
                      "env": {}}]
    study = run_preset.commands_for(run_preset.load_preset("studyB-done_check-v0-sonnet55-high"))
    assert study[0]["argv"][-2:] == ["--images-root", str(REPO)]
    with pytest.raises(SystemExit):
        run_preset.commands_for(run_preset.load_preset("studyB-done_check-v0-sonnet55-high"), ["--x"])
    with pytest.raises(SystemExit):
        run_preset.load_preset("nope")
    out = subprocess.run([sys.executable, str(REPO / "experiments" / "run_preset.py"), "run", "phase3-high",
                          "--dry-run"], capture_output=True, text=True, check=True).stdout
    assert "run_cases.ts --preset phase3-high" in out
