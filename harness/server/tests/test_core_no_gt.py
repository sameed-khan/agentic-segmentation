"""GT isolation: no `load_mask` (or any mask loading) in agent-reachable segd code.

CONTRACTS §0: only segd.data.load_mask reads masks, and only offline code may call it.
app.py / case_state.py / ops.py (plus the other agent-reachable modules) must never
import or call it.
"""
from __future__ import annotations

from pathlib import Path

import segd

AGENT_REACHABLE = [
    "app.py", "case_state.py", "ops.py", "observe.py", "config.py",
    "fake_nni.py", "log.py", "segctl.py", "marks.py", "render.py", "candidates.py",
]


def _src(name: str) -> str:
    return (Path(segd.__file__).parent / name).read_text()


def test_no_load_mask_in_agent_reachable_sources():
    for name in AGENT_REACHABLE:
        src = _src(name)
        assert "load_mask" not in src, f"{name} references load_mask"
        assert "mask_paths" not in src, f"{name} references GT mask paths"


def test_agent_reachable_modules_do_not_read_nifti_masks():
    # nibabel is allowed only for WRITING the final mask (case_state.finish)
    for name in AGENT_REACHABLE:
        src = _src(name)
        if "nibabel" in src or "nib." in src:
            assert "nib.load" not in src, f"{name} loads a NIfTI file"
