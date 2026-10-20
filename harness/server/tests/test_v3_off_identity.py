"""Cross-revision v3-OFF byte identity (review round-2): the SAME scripted op sequence
(open, sheet, view, point, finish review INCLUDING the candidate sheet, finish confirm)
run against (a) THIS worktree's segd with v3 off and (b) the pre-v3 segd source of the
reference main tree (/home/ubuntu/agentic-segmentation, read-only: imported from a
temporary copy under /tmp-style tmp_path, never modified) must produce IDENTICAL
observation text and IDENTICAL PNG bytes for every render.

FakeNNI (nni_url=None) is the segmentation backend, exactly like the existing v2 core
tests. Comparison happens on the full JSON dump of the driver run: one subprocess per
tree keeps the two ``segd`` packages from sharing sys.modules.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

WORKTREE_SERVER = Path(__file__).resolve().parents[1]        # harness/server
MAIN_SEGD_SRC = Path("/home/ubuntu/agentic-segmentation/harness/server/src")


def _run_driver(src_root: Path, out_json: Path, case_dir: Path, driver: Path) -> dict:
    res = subprocess.run(
        [sys.executable, str(driver), str(src_root), str(out_json), str(case_dir)],
        capture_output=True, text=True, timeout=280,
        env={**os.environ, "NNI_URL": ""}, cwd=str(WORKTREE_SERVER))
    assert res.returncode == 0, f"driver failed on {src_root}:\n{res.stdout}\n{res.stderr}"
    return json.loads(out_json.read_text())


def test_v3_off_byte_identity_with_pre_v3_main_tree(tmp_path):
    driver = Path(__file__).resolve().parent / "_v3_identity_driver.py"
    main_src_copy = tmp_path / "main_src"        # temporary copy of the READ-ONLY tree
    if MAIN_SEGD_SRC.is_dir():
        shutil.copytree(MAIN_SEGD_SRC, main_src_copy,
                        ignore=shutil.ignore_patterns("__pycache__"))
        srcs = {"worktree_v3_off": WORKTREE_SERVER / "src", "pre_v3_main": main_src_copy}
    else:                                        # reference tree absent (CI): self-check
        srcs = {"worktree_v3_off": WORKTREE_SERVER / "src",
                "worktree_repeat": WORKTREE_SERVER / "src"}
    names = list(srcs)
    runs = {name: _run_driver(src, tmp_path / f"{name}.json",
                              tmp_path / name / "run" / "phantom", driver)
            for name, src in srcs.items()}
    a, b = runs[names[0]], runs[names[1]]
    differing = [k for k in a if a.get(k) != b.get(k)]
    assert not differing, (
        "v3-off diverged from the reference tree in: "
        + ", ".join(differing))
    # the comparison is meaningful only when the sequence covered the review WITH the
    # candidate sheet and the confirmation
    assert len(a["finish"]["png"]) == 2
    assert a["_meta"]["done"] is True
