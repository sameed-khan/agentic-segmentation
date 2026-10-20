"""Cross-revision equivalence of the integrated segd (one main line, behaviour by flags).

The SAME scripted op sequence (tests/_identity_driver.py) runs in a subprocess on (a) THIS
worktree's segd and (b) a PINNED reference segd; observation text, the sha256 of every rendered
PNG, the counters and the normalized trajectory must be identical.

| flags (case open)                | reference                                              |
|----------------------------------|--------------------------------------------------------|
| defaults                         | harness-v2.0.1 (fixture tests/fixtures/segd_baseline)  |
| segmenter="sammed3d", fake svc   | phase6-sammed3d @d8e82cf (git archive, test time)      |
| v3 pointer+MedGemma, fake svcs   | phase5-v3 @7058be6 (git archive, test time)            |
| dataset="beamster"               | phase5-v3 @7058be6 (git archive, test time)            |

The default baseline is a committed fixture (``git archive harness-v2.0.1
harness/server/src/segd``, sha256 manifest verified before use; a second test checks that the
fixture still equals the tag when the tag is present). The branch references are generated at
test time from the pinned commit ids (both are ancestors of the integration merge, so they
exist in every full clone); a shallow clone without them SKIPS with an explicit reason. The
fake services (local_models) are the integrated tree's for BOTH sides: only segd differs.
FakeNNI (nni_url=None) is the nnInteractive stand-in, exactly like the v2 core tests.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
WORKTREE_SERVER = TESTS.parent                                  # harness/server
HARNESS = WORKTREE_SERVER.parent                                # harness
REPO = HARNESS.parent
FIXTURE = TESTS / "fixtures" / "segd_baseline"
DRIVER = TESTS / "_identity_driver.py"

BASELINE_TAG = "harness-v2.0.1"
PHASE5_V3 = "7058be6dfde4cfb7a8603387bdef96a2d039a796"        # branch phase5-v3 tip
PHASE6_SAMMED3D = "d8e82cf7f101d8a00dad601ebb7412bb45b40e80"  # branch phase6-sammed3d tip
# what each reference tree needs next to harness/server/src (segd locates the MedGemma client
# and prompt templates relative to its own position in the repo layout)
REFERENCE_PATHS = ["harness/server/src", "harness/local_models/medgemma/medgemma_client.py",
                   "harness/prompts/medgemma"]


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True)


def _have(rev: str) -> bool:
    return _git("cat-file", "-e", f"{rev}^{{commit}}").returncode == 0


def _materialize(rev: str, dest: Path) -> Path:
    """``git archive`` the pinned revision into dest; returns the segd source root."""
    if not _have(rev):
        pytest.skip(f"reference revision {rev[:8]} not in this clone (shallow?): cross-revision "
                    "identity NOT checked")
    dest.mkdir(parents=True, exist_ok=True)
    paths = [p for p in REFERENCE_PATHS if _git("cat-file", "-e", f"{rev}:{p}").returncode == 0]
    arch = subprocess.run(["git", "-C", str(REPO), "archive", rev, *paths],
                          capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(dest)], input=arch.stdout, check=True)
    return dest / "harness" / "server" / "src"


def _run_driver(src_root: Path, scenario: str, out_dir: Path, label: str) -> dict:
    out_json = out_dir / f"{label}.json"
    res = subprocess.run(
        [sys.executable, str(DRIVER), str(src_root), str(out_json),
         str(out_dir / label / "run" / "phantom"), scenario],
        capture_output=True, text=True, timeout=280, cwd=str(WORKTREE_SERVER),
        env={**os.environ, "NNI_URL": "", "SEGD_IDENT_SERVICES_ROOT": str(HARNESS),
             "SAMED3D_SRC": str(HARNESS / "local_models" / "sammed3d" / "src")})
    assert res.returncode == 0, f"driver failed on {src_root}:\n{res.stdout}\n{res.stderr}"
    return json.loads(out_json.read_text())


def _assert_identical(a: dict, b: dict, what: str) -> None:
    differing = sorted({k for k in a if a.get(k) != b.get(k)} | {k for k in b if k not in a})
    assert not differing, f"{what} diverged in: {', '.join(differing)}"


def _pinned_baseline() -> tuple[Path, str]:
    env = os.environ.get("SEGD_BASELINE_SRC")        # override: a dir containing the segd package
    if env:
        src = Path(env)
        if not (src / "segd" / "case_state.py").is_file():
            pytest.skip(f"SEGD_BASELINE_SRC={env} has no segd package")
        return src, os.environ.get("SEGD_BASELINE_COMMIT", "unrecorded (SEGD_BASELINE_SRC)")
    info = json.loads((FIXTURE / "BASELINE.json").read_text())
    for rel, want in info["files"].items():
        got = hashlib.sha256((FIXTURE / rel).read_bytes()).hexdigest()
        assert got == want, f"baseline fixture {rel} was modified (pinned to {info['commit']})"
    return FIXTURE, info["commit"]


# ------------------------------------------------------------------ defaults == v2.0.1
def test_default_flags_byte_identity_with_harness_v2_0_1(tmp_path):
    """v3 off, dataset mstt, segmenter nninteractive (no optional argument at all) must give
    the Phase 3 frozen harness's observations and PNG bytes."""
    base_src, commit = _pinned_baseline()
    worktree_src = WORKTREE_SERVER / "src"
    assert base_src.resolve() != worktree_src.resolve()
    # the baseline must really be the pre-v3 / pre-swap / pre-Phase-4 code
    base_cs = (base_src / "segd" / "case_state.py").read_text()
    assert "segmenter" not in base_cs and "v3" not in base_cs
    assert not (base_src / "segd" / "sammed3d.py").exists() and not (base_src / "segd" / "v3.py").exists()
    assert (worktree_src / "segd" / "sammed3d.py").is_file() and (worktree_src / "segd" / "v3.py").is_file()
    a = _run_driver(worktree_src, "default", tmp_path, "integrated")
    b = _run_driver(base_src, "default", tmp_path, "baseline")
    _assert_identical(a, b, f"default flags vs harness-v2.0.1 ({commit[:8]})")
    assert len(a["finish"]["png"]) == 2            # review WITH the candidate sheet
    assert a["_meta"]["done"] is True


def test_baseline_fixture_is_the_harness_v2_0_1_source(tmp_path):
    if _git("rev-parse", "--verify", "-q", f"refs/tags/{BASELINE_TAG}").returncode != 0:
        pytest.skip(f"tag {BASELINE_TAG} not present in this clone")
    src = _materialize(BASELINE_TAG, tmp_path)
    for f in sorted((src / "segd").rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts:
            rel = f.relative_to(src)
            assert (FIXTURE / rel).read_bytes() == f.read_bytes(), f"fixture differs from tag at {rel}"
    tracked = {p.relative_to(FIXTURE) for p in (FIXTURE / "segd").rglob("*")
               if p.is_file() and "__pycache__" not in p.parts}
    assert tracked == {f.relative_to(src) for f in (src / "segd").rglob("*")
                       if f.is_file() and "__pycache__" not in f.parts}


def test_driver_is_deterministic_worktree_self_check(tmp_path):
    """Determinism of the driver itself (NOT a cross-revision check)."""
    src = WORKTREE_SERVER / "src"
    _assert_identical(_run_driver(src, "default", tmp_path, "a"),
                      _run_driver(src, "default", tmp_path, "b"), "driver repeat")


# ------------------------------------------------------------------ flag paths == branches
def test_sammed3d_path_equals_phase6_branch(tmp_path):
    ref = _materialize(PHASE6_SAMMED3D, tmp_path / "ref")
    a = _run_driver(WORKTREE_SERVER / "src", "sammed3d", tmp_path, "integrated")
    b = _run_driver(ref, "sammed3d", tmp_path, "phase6")
    _assert_identical(a, b, "segmenter=sammed3d vs phase6-sammed3d")
    # meaningful: the swap path really ran (prompt-mapping log in the trajectory) and reviewed
    assert any("segmenter_interactions" in rec for rec in a["_trajectory"])
    assert len(a["finish"]["png"]) == 2 and a["_meta"]["done"] is True


def test_v3_path_equals_phase5_branch(tmp_path):
    ref = _materialize(PHASE5_V3, tmp_path / "ref")
    a = _run_driver(WORKTREE_SERVER / "src", "v3", tmp_path, "integrated")
    b = _run_driver(ref, "v3", tmp_path, "phase5")
    _assert_identical(a, b, "v3 flags vs phase5-v3")
    # meaningful: pointer proposals, the localize/review/candidate notes and v3 trajectory
    assert a["locate"]["text"].startswith("locate (gui_actor)")
    assert "Medical second reader" in json.dumps(a)
    calls = [r["call"] for rec in a["_trajectory"] for r in rec.get("v3", [])]
    assert "locate" in calls and calls.count("checkpoint") >= 3
    assert {"localize"} <= set(a["_v3_fired"]) and a["_meta"]["done"] is True


def test_beamster_path_equals_phase5_branch(tmp_path):
    ref = _materialize(PHASE5_V3, tmp_path / "ref")
    a = _run_driver(WORKTREE_SERVER / "src", "beamster", tmp_path, "integrated")
    b = _run_driver(ref, "beamster", tmp_path, "phase5")
    _assert_identical(a, b, "dataset=beamster vs phase5-v3")
    # meaningful: multi-component observation text and the component review sheet
    assert "components 2 (mL:" in a["point#1"]["text"]
    assert len(a["finish"]["png"]) == 3 and a["_meta"]["done"] is True
