"""The three optional feature sets of the integrated segd — harness v3 (``v3``), Phase 4
``dataset`` (mstt | beamster) and Phase 6 ``segmenter`` (nninteractive | sammed3d) — are
orthogonal case-open flags whose defaults are harness-v2.0.1 behaviour. Byte identity of each
single flag path with its reference revision lives in test_identity.py; this file pins the
defaults, the combination rules and smoke-tests the combinations nobody ran before:

- defaults recorded in case_meta.json and GET status;
- v3 pointer/second-reader + dataset beamster: refused (HTTP 400 / ValueError) — the
  multi-lesion and seg_locate prompt/observation changes were never combined;
- v3 + sammed3d and beamster + sammed3d: mechanically supported (never run as experiments).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import segd.ops
from segd.app import create_app
from segd.case_state import CaseState
from _fake_services import (FakeSamMed3DServer, FixedPointerBackend, ScriptedMedGemmaBackend,
                            _load_module, _serve)
from _helpers import install_fake_data, lesion_center, make_phantom, make_two_lesion_phantom

HARNESS_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "harness")
needs_sammed3d = pytest.mark.skipif(
    not (HARNESS_DIR / "local_models" / "sammed3d" / "src").is_dir(),
    reason="harness/local_models/sammed3d not present")


def _meta(cs) -> dict:
    return json.loads((Path(cs.out_dir) / "case_meta.json").read_text())


def test_defaults_are_v2_behaviour_in_case_meta(tmp_path):
    img, _lesion, _body = make_phantom()
    cs = CaseState("phantom", img, np.eye(4), (1.0, 1.0, 1.0), "bright lesion in the limb",
                   "T2", tmp_path / "run" / "phantom")
    cs.initialize()
    meta = _meta(cs)
    assert (meta["v3"], meta["dataset"], meta["segmenter"], meta["segmenter_url"], meta["nni"]) == \
        (None, "mstt", "nninteractive", None, "fake")
    assert (cs.v3, cs.dataset, cs.segmenter) == (None, "mstt", "nninteractive")
    cs.close()


def test_http_defaults_and_combination_rules(tmp_path, monkeypatch):
    install_fake_data(monkeypatch)
    c = TestClient(create_app())
    body = {"case_id": "phantom", "run_id": "r1", "out_dir": str(tmp_path / "results")}
    assert c.post("/cases", json=body).status_code == 200
    st = c.get("/cases/phantom/status").json()
    assert st["segmenter"] == "nninteractive"
    meta = json.loads((tmp_path / "results" / "r1" / "phantom" / "case_meta.json").read_text())
    assert meta["dataset"] == "mstt" and meta["v3"] is None
    # bad flag values are runner misconfiguration -> 400
    assert c.post("/cases", json={**body, "segmenter": "medsam2"}).status_code == 400
    assert c.post("/cases", json={**body, "v3": {"pointer": "nope"}}).status_code == 400
    # v3 components + beamster: refused with a clear message (never run, no combined prompt)
    r = c.post("/cases", json={**body, "dataset": "beamster",
                               "v3": {"pointer": "gui_actor", "medgemma": "q4"}})
    assert r.status_code == 400 and "not supported with dataset 'beamster'" in r.json()["detail"]
    # an all-off v3 object is still v2 behaviour, so it composes with beamster
    assert c.post("/cases", json={**body, "dataset": "beamster",
                                  "v3": {"checkpoints": []}}).status_code == 200


def test_v3_with_beamster_raises_value_error(tmp_path):
    img, _lesion, _body = make_phantom()
    with pytest.raises(ValueError, match="not supported with dataset 'beamster'"):
        CaseState("phantom", img, np.eye(4), (1.0, 1.0, 1.0), "t", "T1c", tmp_path / "x",
                  dataset="beamster", v3={"medgemma": "q4"})


@needs_sammed3d
def test_sammed3d_with_v3_flags_smoke(tmp_path, monkeypatch):
    """v3 pointer + MedGemma checkpoints on top of the sammed3d segmenter: the agent-facing
    additions (seg_locate image, second-reader notes) work independently of the segmenter."""
    monkeypatch.setenv("SAMED3D_SRC", str(HARNESS_DIR / "local_models" / "sammed3d" / "src"))
    lm = HARNESS_DIR / "local_models"
    pointer = FixedPointerBackend([])
    pointer_url, stop_p = _serve(_load_module("flagm_pointd", lm / "pointd_gui_actor" / "server.py")
                                 .create_app(pointer))
    med_url, stop_m = _serve(_load_module("flagm_medgemma", lm / "medgemma" / "start_bf16_server.py")
                             .create_app(ScriptedMedGemmaBackend(), "google/medgemma-1.5-4b-it"))
    img, lesion, _body = make_phantom()
    centre = lesion_center(lesion)
    try:
        with FakeSamMed3DServer() as srv:
            cs = CaseState("phantom", img, np.eye(4), (1.0, 1.0, 1.0), "bright lesion in the limb",
                           "T2", tmp_path / "run" / "phantom", segmenter="sammed3d",
                           segmenter_url=srv.url,
                           v3={"pointer": "gui_actor", "pointer_url": pointer_url,
                               "medgemma": "q4", "medgemma_url": med_url,
                               "checkpoints": ["localize", "review", "candidates"]})
            cs.initialize()
            try:
                segd.ops.run_op(cs, "open", {})
                view = segd.ops.run_op(cs, "view", {"axis": "axial", "slice": centre[2],
                                                    "marks": "fine"})
                assert "Medical second reader" in view.text            # localize checkpoint
                drawn = sorted((int(m), float(x), float(y))
                               for m, _p, x, y in cs.active_view.render_info.marks)
                pointer.points = [(x, y, 0.9) for _m, x, y in drawn[:2]]
                loc = segd.ops.run_op(cs, "locate", {})
                assert loc.text.startswith("locate (gui_actor)")
                segd.ops.run_op(cs, "sheet", {"axis": "axial", "start": 10, "stop": 70})
                vis = cs.markset.visible("axial", centre[2], None, "fine")
                near = min(vis, key=lambda m: sum((a - b) ** 2 for a, b in zip(m.ijk, centre))).id
                segd.ops.run_op(cs, "point", {"ids": [near]})
                fin = segd.ops.run_op(cs, "finish", {"confidence": "medium", "reason": "check"})
                assert re.search(r'review_code "\d{4}"', fin.text)
                meta = _meta(cs)
                assert meta["segmenter"] == "sammed3d" and meta["v3"]["pointer"] == "gui_actor"
                assert type(cs.nni).__name__ == "SamMed3DSession"
            finally:
                cs.close()
    finally:
        stop_p()
        stop_m()


@needs_sammed3d
def test_sammed3d_with_beamster_smoke(tmp_path, monkeypatch):
    """dataset beamster (dense marks, multi-component text, component review sheet) on top of
    the sammed3d segmenter."""
    monkeypatch.setenv("SAMED3D_SRC", str(HARNESS_DIR / "local_models" / "sammed3d" / "src"))
    img, lesion_a, lesion_b = make_two_lesion_phantom()
    ca, cb = lesion_center(lesion_a), lesion_center(lesion_b)
    with FakeSamMed3DServer() as srv:
        cs = CaseState("phantom", img, np.eye(4), (1.0, 1.0, 1.0), "brain metastases", "T1c",
                       tmp_path / "run" / "phantom", dataset="beamster", segmenter="sammed3d",
                       segmenter_url=srv.url)
        cs.initialize()
        try:
            segd.ops.run_op(cs, "open", {})
            segd.ops.run_op(cs, "view", {"axis": "axial", "slice": ca[2], "marks": "fine"})
            vis = cs.markset.visible("axial", ca[2], None, "fine")
            text = ""
            for centre in (ca, cb):
                mark = min(vis, key=lambda m: sum((a - b) ** 2 for a, b in zip(m.ijk, centre))).id
                text = segd.ops.run_op(cs, "point", {"ids": [mark]}).text
            assert re.search(r"components \d+ \(mL:", text)           # multi-component text
            fin = segd.ops.run_op(cs, "finish", {"confidence": "medium", "reason": "check"})
            assert re.search(r'review_code "\d{4}"', fin.text)
            meta = _meta(cs)
            assert meta["dataset"] == "beamster" and meta["segmenter"] == "sammed3d"
        finally:
            cs.close()
