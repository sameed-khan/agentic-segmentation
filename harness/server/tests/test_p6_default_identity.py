"""Cross-revision Phase-6 byte identity: the SAME scripted op sequence (open, sheet, view,
point, box, edit, undo, finish review INCLUDING the candidate sheet, finish confirm) run
against (a) THIS worktree's segd with the DEFAULT segmenter (nninteractive; no segmenter
argument) and (b) a PINNED pre-Phase-6 baseline of segd must produce IDENTICAL observation
text and IDENTICAL PNG bytes for every render: swapping the segmenter code in must not
change what the agent sees on the default path.

The baseline is a recorded revision, never the worktree itself:
- default: ``tests/fixtures/segd_baseline/`` = ``git archive 1dcc622 harness/server/src/segd``
  (the pre-Phase-6 main segd source) with ``BASELINE.json`` (commit + sha256 of every file,
  verified before use);
- override: env ``SEGD_BASELINE_SRC`` = a directory containing the ``segd`` package of the
  baseline revision (e.g. a ``git archive`` extract); ``SEGD_BASELINE_COMMIT`` names it.
If neither exists the test SKIPS with an explicit reason — it never compares the worktree to
itself (a separate determinism self-test below is named as such).

FakeNNI (nni_url=None) is the segmentation backend, exactly like the existing v2 core
tests. One subprocess per tree keeps the two ``segd`` packages from sharing sys.modules.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

WORKTREE_SERVER = Path(__file__).resolve().parents[1]        # harness/server
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "segd_baseline"
DRIVER = Path(__file__).resolve().parent / "_p6_identity_driver.py"


def _run_driver(src_root: Path, out_json: Path, case_dir: Path) -> dict:
    res = subprocess.run(
        [sys.executable, str(DRIVER), str(src_root), str(out_json), str(case_dir)],
        capture_output=True, text=True, timeout=280,
        env={**os.environ, "NNI_URL": ""}, cwd=str(WORKTREE_SERVER))
    assert res.returncode == 0, f"driver failed on {src_root}:\n{res.stdout}\n{res.stderr}"
    return json.loads(out_json.read_text())


def _pinned_baseline() -> tuple[Path, str]:
    """(directory containing the baseline ``segd`` package, commit id) or skip."""
    env = os.environ.get("SEGD_BASELINE_SRC")
    if env:
        src = Path(env)
        if not (src / "segd" / "case_state.py").is_file():
            pytest.skip(f"SEGD_BASELINE_SRC={env} has no segd package")
        return src, os.environ.get("SEGD_BASELINE_COMMIT", "unrecorded (SEGD_BASELINE_SRC)")
    manifest = FIXTURE / "BASELINE.json"
    if not manifest.is_file():
        pytest.skip("no pinned baseline: tests/fixtures/segd_baseline missing and "
                    "SEGD_BASELINE_SRC not set — cross-revision identity NOT checked")
    info = json.loads(manifest.read_text())
    for rel, want in info["files"].items():
        got = hashlib.sha256((FIXTURE / rel).read_bytes()).hexdigest()
        assert got == want, f"baseline fixture {rel} was modified (pinned to {info['commit']})"
    return FIXTURE, info["commit"]


def test_default_segmenter_byte_identity_with_pinned_pre_phase6_baseline(tmp_path):
    base_src, commit = _pinned_baseline()
    worktree_src = WORKTREE_SERVER / "src"
    assert base_src.resolve() != worktree_src.resolve()
    # the baseline must really be the pre-swap code: no sammed3d bridge, old CaseState
    assert not (base_src / "segd" / "sammed3d.py").exists(), "baseline is not pre-Phase-6"
    assert "segmenter" not in (base_src / "segd" / "case_state.py").read_text()
    assert (worktree_src / "segd" / "sammed3d.py").is_file()
    a = _run_driver(worktree_src, tmp_path / "worktree.json", tmp_path / "wt" / "run" / "phantom")
    b = _run_driver(base_src, tmp_path / "baseline.json", tmp_path / "base" / "run" / "phantom")
    differing = [k for k in a if a.get(k) != b.get(k)] + [k for k in b if k not in a]
    assert not differing, (f"default segmenter diverged from pinned baseline {commit} in: "
                           + ", ".join(sorted(set(differing))))
    # meaningful only when the sequence covered the review WITH the candidate sheet
    assert len(a["finish"]["png"]) == 2
    assert a["_meta"]["done"] is True


def test_driver_is_deterministic_worktree_self_check(tmp_path):
    """Determinism of the driver itself (NOT a cross-revision check)."""
    src = WORKTREE_SERVER / "src"
    a = _run_driver(src, tmp_path / "a.json", tmp_path / "a" / "run" / "phantom")
    b = _run_driver(src, tmp_path / "b.json", tmp_path / "b" / "run" / "phantom")
    assert a == b
