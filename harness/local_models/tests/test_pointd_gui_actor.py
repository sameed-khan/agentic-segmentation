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


# ------------------------------------------------ official verifier path (review P1-11)
def test_verifier_uses_official_multimodal_classes(gui_actor_server_mod):
    """The verifier loader must declare the OFFICIAL classes: a multimodal
    Qwen2VLForConditionalGeneration + AutoProcessor — never a causal LM with a plain
    tokenizer as image processor. (Source-level check: the real load needs a GPU; this
    regression test would have caught the AutoModelForCausalLM/AutoTokenizer bug.)"""
    import inspect

    src = inspect.getsource(gui_actor_server_mod.GuiActorRealBackend._load_verifier)
    assert "Qwen2VLForConditionalGeneration.from_pretrained" in src
    assert "AutoProcessor.from_pretrained" in src
    assert "AutoModelForCausalLM" not in src
    assert "AutoTokenizer.from_pretrained" not in src


def test_verify_true_fails_explicitly_without_verifier(gui_actor_server_mod):
    """verify=true with an unavailable verifier must RAISE — silently returning
    verifier:null (running without verification) is the bug under review."""
    gui_server = gui_actor_server_mod
    backend = gui_server.GuiActorRealBackend()
    backend._loaded = True  # skip load()
    backend._gui_inference = lambda *a, **k: {"output_text": "x", "topk_points": [(0.5, 0.5)],
                                              "topk_values": [0.9]}
    backend._system_message = lambda: "s"
    backend._model, backend._tokenizer, backend._processor = object(), object(), object()
    backend._verifier = None
    buf = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buf, format="PNG")
    with pytest.raises(RuntimeError, match="verifier"):
        backend.predict(buf.getvalue(), "q", k=1, verify=True)


def _verifier_backend(gui_server, monkeypatch, rows):
    """Real backend with a STUBBED official verifier (no torch/weights): call i scores
    point i with the logits row (z_true, z_false) = rows[i] placed at the fake
    tokenizer ids 7/9 ("True"/"False"). Returns (backend, captured)."""
    import sys
    import types
    from types import SimpleNamespace

    import numpy as np

    captured = {}

    def fake_process_vision_info(messages):
        captured["messages"] = messages
        return [messages[0]["content"][0]["image"]], None

    fake_qvu = types.ModuleType("qwen_vl_utils")
    fake_qvu.process_vision_info = fake_process_vision_info
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", fake_qvu)

    # fake token ids: "True" -> 7, "False" -> 9 (the tests put their logits there)
    tok = types.SimpleNamespace(encode=lambda s, add_special_tokens=False:
                                {"True": [7], "False": [9]}[s])

    class FakeInputs(dict):
        """A mapping so the fake model can be called with **inputs."""

        def to(self, device):
            captured["device"] = device
            return self

    class FakeProc:
        tokenizer = tok

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "TEMPLATE"

        def __call__(self, text=None, images=None, videos=None, padding=True, return_tensors="pt"):
            captured["images"] = images
            return FakeInputs()

    class FakeModel:
        device = "cuda:0"

        def __init__(self):
            self.calls = 0

        def __call__(self, **inputs):
            z_true, z_false = rows[self.calls]
            self.calls += 1
            arr = np.zeros(16, dtype=float)
            arr[7], arr[9] = z_true, z_false
            return SimpleNamespace(logits=arr.reshape(1, 1, 16))

    class FakeTorch:
        class inference_mode:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

    backend = gui_server.GuiActorRealBackend()
    backend._torch = FakeTorch()
    backend._verifier = FakeModel()
    backend._verifier_processor = FakeProc()
    return backend, captured


def _verify_all(backend, rows):
    """One _verify() call over len(rows) points; returns (order, out)."""
    img = Image.new("RGB", (100, 80), (5, 5, 5))
    pts = [(0.1 + 0.4 * i, 0.5) for i in range(len(rows))]
    return backend._verify(img, "the OK button", pts, 100, 80)


def test_real_verifier_official_protocol(gui_actor_server_mod, monkeypatch):
    """The verify path follows verifier/verifier_model.py: red hollow circle (radius 7,
    width 3) + 500 px half-size crop on a copy at (x*W, y*H), the exact prompt via the
    chat template + process_vision_info image protocol, score = the softmax True/False
    ratio = sigmoid(z_true - z_false), candidates reranked by score."""
    import numpy as np

    gui_server = gui_actor_server_mod
    rows = [(0.0, 1.0), (2.0, 1.0)]
    backend, captured = _verifier_backend(gui_server, monkeypatch, rows)

    order, out = _verify_all(backend, rows)
    # OFFICIAL scoring: full-vocab softmax then p_true/(p_true+p_false). Because
    # softmax is shift invariant that equals sigmoid(z_true - z_false):
    #   sigmoid(-1) = 0.2689, sigmoid(1) = 0.7311.
    # (The old expectation 0.0 = logit 0/(0+1) and 0.6667 = logit 2/(2+1) encoded the
    # logit-ratio regression; see verifier_model.py:153-156.)
    assert out["p_true"] == [0.2689, 0.7311]
    assert order == [1, 0]
    # official prompt text in the user message
    content = captured["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert "hollow red circle" in content[1]["text"] and "the OK button" in content[1]["text"]
    assert content[1]["text"].rstrip().endswith("Answer True or False.")
    # the marked COPY differs from the input (circle drawn) and is cropped to the
    # official min(w-1, x+500) convention: a 100x80 image crops to 99x79
    marked = captured["images"][0]
    assert marked.size == (99, 79)
    assert np.asarray(marked).sum() != np.asarray(Image.new("RGB", (100, 80), (5, 5, 5))).sum()


def test_verifier_score_negative_logits(gui_actor_server_mod, monkeypatch):
    """Negative True/False logits must be scored by the official softmax ratio, i.e.
    sigmoid(z_true - z_false). z_true=-2, z_false=-1 -> sigmoid(-1) = 0.2689; the old
    logit-ratio bug returned -2/-3 = 0.6667 here."""
    gui_server = gui_actor_server_mod
    backend, _ = _verifier_backend(gui_server, monkeypatch, [(-2.0, -1.0)])
    order, out = _verify_all(backend, [(-2.0, -1.0)])
    assert out["p_true"] == [0.2689]
    assert order == [0]


def test_verifier_score_opposite_signed_logits_sum_zero(gui_actor_server_mod, monkeypatch):
    """Opposite-signed logits summing to zero (z_true=1, z_false=-1): the raw logit
    ratio divides by zero; the official full-vocab-softmax ratio is sigmoid(2) = 0.8808
    and must not raise."""
    gui_server = gui_actor_server_mod
    backend, _ = _verifier_backend(gui_server, monkeypatch, [(1.0, -1.0)])
    order, out = _verify_all(backend, [(1.0, -1.0)])
    assert out["p_true"] == [0.8808]
    assert order == [0]


def test_verifier_score_invariant_to_constant_logit_shift(gui_actor_server_mod, monkeypatch):
    """A constant shift of both logits cannot change the softmax ratio: (0,1) and (2,3)
    have the same z_true - z_false = -1 and must score identically (0.2689). The old
    logit-ratio bug gave 0.0 vs 0.4 — a spurious reranking."""
    gui_server = gui_actor_server_mod
    rows = [(0.0, 1.0), (2.0, 3.0)]
    backend, _ = _verifier_backend(gui_server, monkeypatch, rows)
    order, out = _verify_all(backend, rows)
    assert out["p_true"] == [0.2689, 0.2689]
    assert order == [0, 1]


def test_predict_verify_reranks_points(gui_actor_server_mod, monkeypatch):
    """predict(verify=True) reorders the returned points by the verifier scores."""
    gui_server = gui_actor_server_mod
    backend = gui_server.GuiActorRealBackend()
    backend._loaded = True
    backend._gui_inference = lambda *a, **k: {"output_text": "x",
                                              "topk_points": [(0.5, 0.5), (0.1, 0.1)],
                                              "topk_values": [0.9, 0.8]}
    backend._system_message = lambda: "s"
    backend._model, backend._tokenizer, backend._processor = object(), object(), object()
    backend._verifier, backend._verifier_processor = object(), object()
    monkeypatch.setattr(backend, "_verify",
                        lambda image, query, pts, w, h: ([1, 0], {"used": True, "p_true": [0.1, 0.9]}))
    buf = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buf, format="PNG")
    out = backend.predict(buf.getvalue(), "q", k=2, verify=True)
    assert out["points"][0]["x_norm"] == 0.1 and out["points"][1]["x_norm"] == 0.5
    assert out["verifier"]["used"] is True


def test_point_verify_unavailable_is_http_500(gui_actor_server_mod):
    """The HTTP layer surfaces an unavailable verifier as an explicit 500, not a
    verifier:null success."""
    gui_server = gui_actor_server_mod

    class VerifyUnavailableBackend(gui_server.GuiActorRealBackend):
        model_id = "microsoft/GUI-Actor-2B-Qwen2-VL"

        def __init__(self):  # no torch
            self._loaded = True

        def predict(self, png_bytes, query, k, verify):
            if verify:
                raise RuntimeError("verify=true but the official GUI-Actor verifier is unavailable")
            return {"points": [{"x_norm": 0.5, "y_norm": 0.5, "score": 0.9}], "raw": "x",
                    "verifier": None}

    from fastapi.testclient import TestClient

    app = gui_server.create_app(VerifyUnavailableBackend())
    with TestClient(app, raise_server_exceptions=False) as client:
        r = client.post("/point", json={"image_png_b64": b64png(), "query": "q", "k": 1,
                                        "verify": True})
    assert r.status_code == 500 and "verifier" in r.json()["detail"]


# ------------------------------------------------ official system prompt (review P2-1)
def test_system_message_matches_official_2b_card(gui_actor_server_mod):
    """The default (2B Qwen2-VL) checkpoint must use the SHORT official prompt from the
    model card / eval/screenSpot.py qwen2vl branch — not the longer constants variant."""
    gui_server = gui_actor_server_mod
    backend = gui_server.GuiActorRealBackend()  # unloaded; the qwen2vl branch needs no import
    assert backend._system_message() == (
        "You are a GUI agent. You are given a task and a screenshot of the screen. "
        "You need to perform a series of pyautogui actions to complete the task.")


def test_system_message_qwen25_checkpoint_uses_constants_variant(gui_actor_server_mod, monkeypatch):
    import sys
    import types

    gui_server = gui_actor_server_mod
    long_prompt = ("You are a GUI agent. Given a screenshot of the current GUI and a human "
                   "instruction, your task is to locate the screen element.")
    fake_gui_actor = types.ModuleType("gui_actor")
    fake_constants = types.ModuleType("gui_actor.constants")
    fake_constants.grounding_system_message = long_prompt
    fake_gui_actor.constants = fake_constants
    monkeypatch.setitem(sys.modules, "gui_actor", fake_gui_actor)
    monkeypatch.setitem(sys.modules, "gui_actor.constants", fake_constants)
    backend = gui_server.GuiActorRealBackend(model_id="microsoft/GUI-Actor-2B-Qwen2.5-VL")
    assert backend._system_message() == long_prompt


# ----------------------------------------------------------------- inference lock (M4/m6)
def test_inference_lock_serializes_concurrent_requests(gui_actor_server_mod):
    """Two concurrent /point requests NEVER overlap inside backend.predict: the
    process-wide _INFER_LOCK serialises inference (HF tokenizers are not thread-safe;
    'Already borrowed' under concurrency). Uses the FakeBackend test-only delay hook so
    the second request is still in flight while the first predicts."""
    import asyncio
    import threading
    import time

    import httpx

    class RecordingBackend(gui_actor_server_mod.FakeBackend):
        def __init__(self):
            super().__init__(delay_s=0.05)
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def predict(self, png_bytes, query, k, verify):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                return super().predict(png_bytes, query, k, verify)
            finally:
                with self.lock:
                    self.active -= 1

    backend = RecordingBackend()
    app = gui_actor_server_mod.create_app(backend)

    async def fire():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://testserver") as client:
            async def one(i):
                r = await client.post("/point", json={"image_png_b64": b64png(8, 8, i),
                                                      "query": "q", "k": 2})
                return r.status_code
            return await asyncio.gather(*(one(i) for i in range(3)))

    statuses = asyncio.run(fire())
    assert statuses == [200, 200, 200]
    assert backend.max_active == 1  # never two predictions inside the critical section
