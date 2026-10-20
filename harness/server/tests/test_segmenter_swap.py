"""Phase 6 segmenter swap: segd's CaseState / HTTP API with ``segmenter="sammed3d"`` against
the sammed3d service's ``--fake`` backend over real localhost HTTP (no torch, no GPU).

The fake backend is the service's FakePatchPredictor (same session state, accumulation,
patch placement, stitching and grid mapping as the real engine; only the per-patch model
is a stand-in). The default (nninteractive/FakeNNI) path must stay unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import segd.ops
from segd import sammed3d as sammed3d_bridge
from segd.app import create_app
from segd.case_state import CaseState, OpError
from _fake_services import FakeSamMed3DServer
from _helpers import (
    add_ring_marks, dice, install_fake_data, install_stubs, lesion_center, make_case_state,
    make_phantom,
)

SAMED3D_SRC = Path(__file__).resolve().parents[2] / "local_models" / "sammed3d" / "src"
pytestmark = pytest.mark.skipif(not SAMED3D_SRC.is_dir(),
                                reason="harness/local_models/sammed3d not present")


@pytest.fixture()
def fake_server(monkeypatch):
    monkeypatch.setenv("SAMED3D_SRC", str(SAMED3D_SRC))
    with FakeSamMed3DServer() as srv:
        yield srv


def _swap_case(tmp_path, monkeypatch, url, **kw):
    install_stubs(monkeypatch, force=True)
    img, lesion, _body = make_phantom(kw.pop("shape", (120, 120, 80)), z_halfwidth=0.25)
    cs = CaseState("phantom", img, np.eye(4), (1.0, 1.0, 1.0), "bright lesion in the limb", "T2",
                   tmp_path / "results" / "run1" / "phantom", segmenter="sammed3d",
                   segmenter_url=url, **kw)
    cs.initialize()
    return cs, lesion


def _trajectory(cs) -> list[dict]:
    path = Path(cs.out_dir) / "trajectory.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_default_segmenter_is_nninteractive_and_untouched(tmp_path, monkeypatch):
    install_stubs(monkeypatch, force=True)
    sys.modules.pop("sammed3d.client", None)
    cs, lesion = make_case_state(tmp_path, monkeypatch)
    try:
        assert cs.segmenter == "nninteractive" and cs.segmenter_url is None
        assert type(cs.nni).__name__ == "FakeNNI"
        meta = json.loads((Path(cs.out_dir) / "case_meta.json").read_text())
        assert meta["segmenter"] == "nninteractive" and meta["segmenter_url"] is None
        corners, centre = add_ring_marks(cs, lesion_center(lesion))
        segd.ops.run_op(cs, "view", {"axis": "axial", "slice": lesion_center(lesion)[2],
                                     "marks": "fine"})
        segd.ops.run_op(cs, "point", {"ids": [centre]})
        rec = _trajectory(cs)[-1]
        assert "segmenter_interactions" not in rec      # nnInteractive records are unchanged
    finally:
        cs.close()


def test_unknown_segmenter_is_rejected():
    with pytest.raises(ValueError):
        CaseState("c", np.zeros((4, 4, 4), np.float32), np.eye(4), (1.0, 1.0, 1.0), "t", "T2",
                  "/tmp/never", segmenter="medsam2")


def test_swap_path_point_lasso_box_paint_refine_and_trajectory_mapping(
        tmp_path, monkeypatch, fake_server):
    cs, lesion = _swap_case(tmp_path, monkeypatch, fake_server.url)
    try:
        assert type(cs.nni).__name__ == "SamMed3DSession"
        meta = json.loads((Path(cs.out_dir) / "case_meta.json").read_text())
        assert meta["segmenter"] == "sammed3d" and meta["segmenter_url"] == fake_server.url
        centre = lesion_center(lesion)
        corners, centre_id = add_ring_marks(cs, centre)
        k = centre[2]
        segd.ops.run_op(cs, "view", {"axis": "axial", "slice": k, "marks": "fine"})

        out = segd.ops.run_op(cs, "point", {"ids": [centre_id]})
        assert out.text.startswith("version v1")
        assert dice(cs.current_mask(), lesion) > 0.6
        rec = [r for r in _trajectory(cs) if r["op"] == "point"][-1]
        mapped = rec["segmenter_interactions"]
        point_log = [e for e in mapped if "mapped_points" in e]
        assert point_log and point_log[0]["op"] == "point"
        assert point_log[0]["mapped_points"][0][3] == 1           # label 1 = include
        summary = mapped[-1]
        assert summary["patches_used"] == 1 and summary["n_points_total"] == 1

        # lasso (drawn region -> <= 5 sampled points) and box (centre + quarter points)
        segd.ops.run_op(cs, "checkout", {"version": "v0"})
        segd.ops.run_op(cs, "lasso", {"ids": corners})
        lasso = [r for r in _trajectory(cs) if r["op"] == "lasso"][-1]["segmenter_interactions"]
        assert lasso[0]["op"] == "lasso" and 1 <= lasso[0]["n_points"] <= 5
        segd.ops.run_op(cs, "checkout", {"version": "v0"})
        segd.ops.run_op(cs, "box", {"id_a": corners[0], "id_b": corners[2]})
        box = [r for r in _trajectory(cs) if r["op"] == "box"][-1]["segmenter_interactions"]
        assert box[0]["op"] == "bbox" and box[0]["n_points"] in (5, 9)

        # paint (non-model edit -> dirty) then point: the sync rule re-sends the current
        # mask as the initial seg (low-res mask prompt) and the point still works
        segd.ops.run_op(cs, "paint", {"ids": corners[:4], "mode": "add", "slice_range": [k, k]})
        painted = int((cs.current_mask() > 0).sum())
        assert painted > 0
        segd.ops.run_op(cs, "point", {"ids": [centre_id]})
        assert (cs.current_mask() > 0).sum() > 0

        # refine = initial seg + prediction; the mapping log shows the interior point
        before = cs.model_prompts
        segd.ops.run_op(cs, "refine", {})
        assert cs.model_prompts == before + 1
        ref = [r for r in _trajectory(cs) if r["op"] == "refine"][-1]["segmenter_interactions"]
        assert any(e.get("op") == "initial_seg" for e in ref)
    finally:
        cs.close()
    assert fake_server.n_sessions == 0                              # close() freed the session


def test_swap_path_matches_nni_call_surface_in_observation_format(tmp_path, monkeypatch, fake_server):
    """Observation text/format is segmenter-independent: same grammar for both arms."""
    install_stubs(monkeypatch, force=True)
    texts = {}
    for name in ("nninteractive", "sammed3d"):
        if name == "sammed3d":
            cs, lesion = _swap_case(tmp_path / name, monkeypatch, fake_server.url)
        else:
            cs, lesion = make_case_state(tmp_path / name, monkeypatch)
        try:
            centre = lesion_center(lesion)
            _corners, centre_id = add_ring_marks(cs, centre)
            segd.ops.run_op(cs, "view", {"axis": "axial", "slice": centre[2], "marks": "fine"})
            texts[name] = segd.ops.run_op(cs, "point", {"ids": [centre_id]}).text
        finally:
            cs.close()
    import re

    def grammar(t):   # numbers differ (different segmenter); the field structure must not
        return re.sub(r"[-+]?\d+(\.\d+)?", "N", t)
    assert grammar(texts["nninteractive"]) == grammar(texts["sammed3d"])


def test_swap_path_reconnects_when_the_service_lost_the_session(tmp_path, monkeypatch, fake_server):
    cs, lesion = _swap_case(tmp_path, monkeypatch, fake_server.url)
    try:
        centre = lesion_center(lesion)
        _corners, centre_id = add_ring_marks(cs, centre)
        segd.ops.run_op(cs, "view", {"axis": "axial", "slice": centre[2], "marks": "fine"})
        fake_server.app.state.sammed3d["sessions"].clear()          # service "restarted"
        out = segd.ops.run_op(cs, "point", {"ids": [centre_id]})    # reconnect + retry
        assert out.text.startswith("version v1") and (cs.current_mask() > 0).any()
    finally:
        cs.close()


def test_http_open_with_segmenter_field(tmp_path, monkeypatch, fake_server):
    install_stubs(monkeypatch, force=True)
    install_fake_data(monkeypatch)
    c = TestClient(create_app())
    r = c.post("/cases", json={"case_id": "phantom", "run_id": "r1",
                               "out_dir": str(tmp_path / "results"),
                               "segmenter": "sammed3d", "segmenter_url": fake_server.url})
    assert r.status_code == 200 and r.json()["ok"] is True
    st = c.get("/cases/phantom/status").json()
    assert st["segmenter"] == "sammed3d" and st["nni"] == "sammed3d"
    meta = json.loads((tmp_path / "results" / "r1" / "phantom" / "case_meta.json").read_text())
    assert meta["segmenter"] == "sammed3d" and meta["segmenter_url"] == fake_server.url
    c.delete("/cases/phantom")
    assert fake_server.n_sessions == 0
    # unknown segmenter -> 400 (programmer error); default stays nninteractive
    bad = c.post("/cases", json={"case_id": "phantom", "run_id": "r2",
                                 "out_dir": str(tmp_path / "results"), "segmenter": "x"})
    assert bad.status_code == 400
    ok = c.post("/cases", json={"case_id": "phantom", "run_id": "r3",
                                "out_dir": str(tmp_path / "results")})
    assert ok.status_code == 200
    assert c.get("/cases/phantom/status").json()["segmenter"] == "nninteractive"
    c.delete("/cases/phantom")


def test_sammed3d_default_url_is_8095():
    assert sammed3d_bridge.default_url() == "http://127.0.0.1:8095"
