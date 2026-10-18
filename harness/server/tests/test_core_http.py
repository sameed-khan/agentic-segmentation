"""HTTP API round trip (FastAPI TestClient) and cross-case concurrency."""
from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from segd.app import create_app
from _helpers import add_ring_marks, dice, install_fake_data, install_stubs, lesion_center, make_phantom


def _open(client, tmp_path, case_id="phantom"):
    r = client.post("/cases", json={"case_id": case_id, "run_id": "r1", "out_dir": str(tmp_path / "results")})
    assert r.status_code == 200 and r.json() == {"ok": True, "handle": case_id}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    install_stubs(monkeypatch, force=True)
    install_fake_data(monkeypatch)
    app = create_app()
    return TestClient(app), tmp_path, app


def test_healthz(client):
    c, _, _app = client
    assert c.get("/healthz").json() == {"ok": True}


def test_http_round_trip(client):
    c, tmp, _app = client
    _open(c, tmp)

    r = c.post("/cases/phantom/op", json={"op": "open", "args": {}})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["done"] is False
    assert "case phantom" in body["text"] and "target:" in body["text"]
    assert len(body["images"]) == 1
    assert body["images"][0].startswith("iVBOR")       # base64 PNG

    r = c.post("/cases/phantom/op", json={"op": "view", "args": {"axis": "axial", "slice": 40, "marks": "fine"}})
    assert r.json()["ok"] is True

    st = c.get("/cases/phantom/status").json()
    assert st["ok"] and st["tool_calls"] == 2 and st["done"] is False and st["current_version"] == "v0"
    assert st["active_view"] == {"axis": "axial", "slice": 40}

    r = c.post("/cases/phantom/op", json={"op": "finish", "args": {"confidence": "high", "reason": "t"}})
    assert r.json()["ok"] is True and r.json()["done"] is False       # first finish = review
    assert "Final review of v0" in r.json()["text"] and len(r.json()["images"]) == 1
    import re as _re
    code = _re.search(r'review_code "(\d{4})"', r.json()["text"]).group(1)
    r = c.post("/cases/phantom/op", json={"op": "finish", "args": {"confidence": "high", "reason": "t", "review_code": code}})
    assert r.json()["ok"] is True and r.json()["done"] is True
    out_dir = tmp / "results" / "r1" / "phantom"
    assert (out_dir / "final_mask.nii.gz").exists()
    assert (out_dir / "versions.npz").exists()
    assert (out_dir / "trajectory.jsonl").exists()
    assert (out_dir / "case_meta.json").exists()
    assert len(list((out_dir / "renders").glob("*.png"))) == 3

    r = c.post("/cases/phantom/op", json={"op": "view", "args": {"axis": "axial", "slice": 40}})
    assert r.status_code == 200 and r.json()["ok"] is False and "finished" in r.json()["error"]

    r = c.delete("/cases/phantom")
    assert r.json()["ok"] is True
    assert c.post("/cases/phantom/op", json={"op": "view", "args": {"axis": "axial", "slice": 40}}).status_code == 404


def test_http_agent_facing_error_is_200(client):
    c, _, _app = client
    c.post("/cases", json={"case_id": "phantom", "run_id": "r1", "out_dir": "unused"})
    r = c.post("/cases/phantom/op", json={"op": "view", "args": {"axis": "sideways"}})
    assert r.status_code == 200
    b = r.json()
    assert b["ok"] is False and "axis" in b["error"] and b["text"] == b["error"]
    # programmer errors: unknown handle -> 404; missing body field -> 400
    assert c.post("/cases/nope/op", json={"op": "open"}).status_code == 404
    assert c.post("/cases", json={"case_id": "phantom"}).status_code == 400


def test_http_unknown_case_404(client):
    c, _, _app = client
    r = c.post("/cases", json={"case_id": "no-such-case", "run_id": "r1", "out_dir": "x"})
    assert r.status_code == 404


def test_http_point_dice(client):
    """Mark resolution + FakeNNI growth through the full HTTP stack (Dice > 0.9)."""
    c, tmp, app = client
    _open(c, tmp)
    cs = app.state.segd["cases"]["phantom"]
    lesion = make_phantom()[1]
    corners, centre = add_ring_marks(cs, lesion_center(lesion))
    c.post("/cases/phantom/op", json={"op": "view", "args": {"axis": "axial", "slice": lesion_center(lesion)[2], "marks": "fine"}})
    r = c.post("/cases/phantom/op", json={"op": "point", "args": {"ids": [centre]}})
    assert r.json()["ok"] is True, r.json()
    assert dice(cs.current_mask(), lesion) > 0.9
    st = c.get("/cases/phantom/status").json()
    assert st["model_prompts"] == 1 and st["current_version"] == "v1"


def test_cross_case_concurrency(tmp_path, monkeypatch):
    """Ops on different cases run concurrently; two 1 s prompts overlap."""
    install_stubs(monkeypatch, force=True)
    install_fake_data(monkeypatch, case_ids=("phantomA", "phantomB"))
    app = create_app()
    from segd.fake_nni import FakeNNI
    orig = FakeNNI._do_point

    def slow_point(self, coord, include):
        time.sleep(1.0)
        return orig(self, coord, include)

    monkeypatch.setattr(FakeNNI, "_do_point", slow_point)

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
            for cid in ("phantomA", "phantomB"):
                r = await ac.post("/cases", json={"case_id": cid, "run_id": "r1", "out_dir": str(tmp_path / "results")})
                assert r.json()["ok"]
            await asyncio.gather(
                ac.post("/cases/phantomA/op", json={"op": "view", "args": {"axis": "axial", "slice": 12, "marks": "fine"}}),
                ac.post("/cases/phantomB/op", json={"op": "view", "args": {"axis": "axial", "slice": 12, "marks": "fine"}}),
            )
            t0 = time.monotonic()
            rs = await asyncio.gather(
                ac.post("/cases/phantomA/op", json={"op": "point", "args": {"ids": [1]}}),
                ac.post("/cases/phantomB/op", json={"op": "point", "args": {"ids": [1]}}),
            )
            wall = time.monotonic() - t0
            assert all(r.json()["ok"] for r in rs)
            return wall

    wall = asyncio.run(run())
    # two 1 s point prompts overlapped: well under the 2 s serial time
    assert wall < 1.95, f"ops did not run concurrently ({wall:.2f}s)"


def test_same_case_ops_serialize(tmp_path, monkeypatch):
    install_stubs(monkeypatch, force=True)
    install_fake_data(monkeypatch)
    app = create_app()
    events = []
    from segd.fake_nni import FakeNNI
    orig = FakeNNI._do_point

    def recording_point(self, coord, include):
        events.append("start")
        time.sleep(0.05)
        out = orig(self, coord, include)
        events.append("end")
        return out

    monkeypatch.setattr(FakeNNI, "_do_point", recording_point)

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
            await ac.post("/cases", json={"case_id": "phantom", "run_id": "r1", "out_dir": str(tmp_path / "results")})
            cs = app.state.segd["cases"]["phantom"]
            corners, _centre = add_ring_marks(cs, lesion_center(make_phantom()[1]))
            k = lesion_center(make_phantom()[1])[2]
            r = await ac.post("/cases/phantom/op", json={"op": "view", "args": {"axis": "axial", "slice": k, "marks": "fine"}})
            assert r.json()["ok"], r.json()
            rs = await asyncio.gather(*[
                ac.post("/cases/phantom/op", json={"op": "point", "args": {"ids": [cid], "polarity": "add"}})
                for cid in corners
            ])
            assert all(r.json()["ok"] for r in rs), [r.json() for r in rs]

    asyncio.run(run())
    assert len(events) == 8
    # no interleaving: every "start" is immediately followed by its "end"
    assert all(events[2 * k] == "start" and events[2 * k + 1] == "end" for k in range(4))


def test_internal_error_on_op_returns_200_generic(client, monkeypatch):
    """Finding #2: an unexpected (programmer) exception on an agent op returns HTTP 200
    ok:false with a generic message + budget counters; the full traceback goes to the
    trajectory with error_kind 'internal' (scoring tags `harness`). 5xx stays reserved
    for non-op endpoints."""
    import sys
    c, tmp, app = client
    _open(c, tmp)

    def boom(*a, **k):
        raise RuntimeError("unexpected segd bug")

    monkeypatch.setattr(sys.modules["segd.render"], "render_overview", boom)
    r = c.post("/cases/phantom/op", json={"op": "open", "args": {}})
    assert r.status_code == 200
    b = r.json()
    assert b["ok"] is False and b["text"] == b["error"]
    assert b["error"].startswith("internal harness error")
    assert "tool calls 1/40" in b["error"]          # counters visible on errors too
    recs = [json.loads(l) for l in open(tmp / "results" / "r1" / "phantom" / "trajectory.jsonl")]
    assert recs[-1]["error_kind"] == "internal"
    assert "Traceback" in recs[-1]["error"] and "unexpected segd bug" in recs[-1]["error"]


def test_failed_open_closes_claimed_nni_session(client, monkeypatch):
    """Finding #5: if initialize fails after the nnInteractive session is claimed, the
    session is closed (slot freed) and the case is not registered."""
    c, tmp, app = client
    from segd.case_state import CaseState
    from segd.fake_nni import FakeNNI
    closed = []
    monkeypatch.setattr(FakeNNI, "close", lambda self: closed.append(1))

    def boom_meta(self, meta):
        raise RuntimeError("meta boom")

    monkeypatch.setattr("segd.log.TrajectoryLogger.write_case_meta", boom_meta)
    with pytest.raises(RuntimeError):
        c.post("/cases", json={"case_id": "phantom", "run_id": "r1",
                               "out_dir": str(tmp / "results")})
    assert closed == [1]                             # the claimed session was released
    assert app.state.segd["cases"].get("phantom") is None


def test_finish_closes_nni_session(client):
    """Finding #5: finish frees the nnInteractive session slot immediately."""
    c, tmp, app = client
    _open(c, tmp)
    cs = app.state.segd["cases"]["phantom"]
    c.post("/cases/phantom/op", json={"op": "open", "args": {}})
    assert cs.nni is not None
    r = c.post("/cases/phantom/op", json={"op": "finish", "args": {"confidence": "high", "reason": "x", "skip_review": True}})
    assert r.json()["ok"] is True and r.json()["done"] is True
    assert cs.nni is None
