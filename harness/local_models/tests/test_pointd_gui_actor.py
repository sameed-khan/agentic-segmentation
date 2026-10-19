"""GUI-Actor service API contract and region->point conversion (fake backend, no torch).

The real backend's coordinate rule is mirrored here: the service multiplies the
official helper's NORMALIZED region centres (pred["topk_points"], one (x, y) per
connected region, best first) by the sent image's width/height.
"""
from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from conftest import b64png, png_bytes  # noqa: F401


# ------------------------------------------------------------------- API contract
def test_healthz(gui_actor_client):
    r = gui_actor_client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["backend"] == "fake"
    assert body["model"] == "microsoft/GUI-Actor-2B-Qwen2-VL"
    assert "version" in body


def test_point_contract(gui_actor_client):
    r = gui_actor_client.post("/point", json={"image_png_b64": b64png(64, 48), "query": "the tumor", "k": 3})
    assert r.status_code == 200
    body = r.json()
    # response contract: {points:[{x,y,score}], raw, latency_ms, model, version}
    assert len(body["points"]) == 3
    for rank, p in enumerate(body["points"]):
        assert set(("x", "y", "score", "rank")) <= set(p)
        assert p["rank"] == rank
        assert 0 <= p["x"] <= 64 and 0 <= p["y"] <= 48  # pixel coords of the sent image
    assert isinstance(body["raw"], str) and body["raw"]
    assert isinstance(body["latency_ms"], float)
    assert body["model"] == "microsoft/GUI-Actor-2B-Qwen2-VL"
    assert body["version"]
    assert body["image_w"] == 64 and body["image_h"] == 48


def test_point_k_caps_results(gui_actor_client):
    r = gui_actor_client.post("/point", json={"image_png_b64": b64png(), "query": "the tumor", "k": 1})
    assert len(r.json()["points"]) == 1


def test_point_deterministic(gui_actor_client):
    a = gui_actor_client.post("/point", json={"image_png_b64": b64png(80, 60), "query": "q", "k": 3}).json()
    b = gui_actor_client.post("/point", json={"image_png_b64": b64png(80, 60), "query": "q", "k": 3}).json()
    assert a["points"] == b["points"]


def test_point_rejects_bad_input(gui_actor_client):
    assert gui_actor_client.post("/point", json={"image_png_b64": "not-base64-!!!", "query": "q"}).status_code == 400
    assert gui_actor_client.post("/point", json={"image_png_b64": base64.b64encode(b"junk").decode(),
                                                 "query": "q"}).status_code == 400
    assert gui_actor_client.post("/point", json={"image_png_b64": b64png(), "query": ""}).status_code == 422
    assert gui_actor_client.post("/point", json={"image_png_b64": b64png(), "query": "q", "k": 0}).status_code == 422


# ------------------------------------------------- real-backend coordinate rule
def test_real_region_to_point_conversion(gui_actor_server_mod):
    """The real backend must map official normalized region centres -> pixel points."""
    gui_server = gui_actor_server_mod

    class StubBackend(gui_server.GuiActorRealBackend):
        model_id = "microsoft/GUI-Actor-2B-Qwen2-VL"  # read by the /point response builder

        def __init__(self):  # skip __init__ (no torch)
            self._loaded = True

        def predict(self, png_bytes, query, k, verify):
            return {"points": [{"x_norm": 0.25, "y_norm": 0.5, "score": 0.9},
                               {"x_norm": 0.75, "y_norm": 0.125, "score": 0.4}],
                    "raw": "pyautogui.click(<|pointer_start|><|pointer_pad|><|pointer_end|>)",
                    "verifier": None}

    app = gui_server.create_app(StubBackend())
    from fastapi.testclient import TestClient

    with TestClient(app) as client:  # startup hook no-ops (load() short-circuits)
        body = client.post("/point", json={"image_png_b64": b64png(200, 100), "query": "q", "k": 2}).json()
    assert body["points"][0]["x"] == pytest.approx(0.25 * 200, abs=0.02)
    assert body["points"][0]["y"] == pytest.approx(0.5 * 100, abs=0.02)
    assert body["points"][1]["x"] == pytest.approx(0.75 * 200, abs=0.02)
    assert body["points"][1]["y"] == pytest.approx(0.125 * 100, abs=0.02)


def test_real_predict_uses_official_inference(gui_actor_server_mod, monkeypatch):
    """The real backend calls the official repo `inference` with the official system
    message and use_placeholder=True, then converts topk_points (normalized) + topk_values."""
    import sys
    import types

    gui_server = gui_actor_server_mod
    captured = {}

    def fake_inference(conversation, model, tokenizer, processor, use_placeholder, topk):
        captured["conversation"] = conversation
        captured["use_placeholder"] = use_placeholder
        captured["topk"] = topk
        return {"output_text": "pred", "topk_points": [(0.5, 0.5), (0.1, 0.1)],
                "topk_values": [0.8, 0.2], "n_width": 36, "n_height": 36}

    fake_gui_actor = types.ModuleType("gui_actor")
    fake_inference_mod = types.ModuleType("gui_actor.inference")
    fake_inference_mod.inference = fake_inference
    fake_constants = types.ModuleType("gui_actor.constants")
    fake_constants.grounding_system_message = "You are a GUI agent."
    fake_gui_actor.inference = fake_inference_mod
    fake_gui_actor.constants = fake_constants
    for name, mod in [("gui_actor", fake_gui_actor),
                      ("gui_actor.inference", fake_inference_mod),
                      ("gui_actor.constants", fake_constants)]:
        monkeypatch.setitem(sys.modules, name, mod)

    backend = gui_server.GuiActorRealBackend()
    backend._loaded = True  # skip load()
    backend._gui_inference = fake_inference
    backend._system_message = lambda: "You are a GUI agent."
    backend._model = object()
    backend._tokenizer = object()
    backend._processor = object()

    img = Image.new("RGB", (100, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    out = backend.predict(buf.getvalue(), "the OK button", k=2, verify=False)

    assert captured["use_placeholder"] is True
    assert captured["topk"] == 2
    assert captured["conversation"][1]["content"][0]["type"] == "image"
    assert captured["conversation"][1]["content"][1]["text"] == "the OK button"
    assert out["points"][0] == {"x_norm": 0.5, "y_norm": 0.5, "score": 0.8}
    assert out["raw"] == "pred"
