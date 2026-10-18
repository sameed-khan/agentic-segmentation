"""Live test against a real nnInteractive server (skipped unless NNI_URL is set).

Run once during bring-up:  NNI_URL=http://127.0.0.1:1527 uv run pytest -q tests/test_core_live.py
The server allows max 4 sessions and is shared with other agents, so this test uses
exactly ONE session and closes it in a finally block.
"""
from __future__ import annotations

import os

import pytest

import segd.ops
from _helpers import add_ring_marks, dice, install_stubs, lesion_center, make_case_state

NNI_URL = os.environ.get("NNI_URL")


@pytest.mark.skipif(not NNI_URL, reason="live test: set NNI_URL to a running nnInteractive server")
def test_live_point_and_lasso_on_phantom(tmp_path, monkeypatch):
    install_stubs(monkeypatch, force=True)   # deterministic lesion marks; the model is the test subject
    cs, lesion = make_case_state(tmp_path, monkeypatch, nni_url=NNI_URL)
    corners, centre = add_ring_marks(cs, lesion_center(lesion))
    k = lesion_center(lesion)[2]
    try:
        segd.ops.run_op(cs, "view", {"axis": "axial", "slice": k, "marks": "fine"})
        segd.ops.run_op(cs, "point", {"ids": [centre]})
        d_point = dice(cs.current_mask(), lesion)
        print(f"live point dice = {d_point:.3f}")
        assert d_point > 0.9, f"live point Dice {d_point:.3f} <= 0.9"

        # back to empty, then lasso around the lesion (one session reused)
        segd.ops.run_op(cs, "checkout", {"version": "v0"})
        segd.ops.run_op(cs, "lasso", {"ids": corners})
        d_lasso = dice(cs.current_mask(), lesion)
        print(f"live lasso dice = {d_lasso:.3f}")
        assert d_lasso > 0.9, f"live lasso Dice {d_lasso:.3f} <= 0.9"

        # sync rule with a NON-empty mask against the real server: paint (dirty) -> point
        # must reset interactions and re-feed the painted mask as the initial seg
        segd.ops.run_op(cs, "paint", {"ids": corners[:4], "mode": "add", "slice_range": [k, k]})
        assert (cs.current_mask() > 0).any()
        out = segd.ops.run_op(cs, "point", {"ids": [centre]})
        assert out.text.startswith("version v4")   # paint made v3; the point commits v4
        assert (cs.current_mask() > 0).any()
    finally:
        cs.close()   # release the lease immediately; the server is shared
