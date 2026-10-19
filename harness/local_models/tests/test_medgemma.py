"""MedGemma thin-client + BF16-server API tests (fake backend, no torch).

The same client targets both MedGemma arms: llama.cpp `llama-server` (port 8091,
Q4_K_M) and the Transformers BF16 server (port 8094) — both OpenAI-compatible.
"""
from __future__ import annotations

import json

from medgemma_client import MedGemmaClient, extract_json_object, image_to_data_url

from conftest import b64png, png_bytes


# ----------------------------------------------------------------- JSON extraction
def test_extract_json_whole_text():
    assert extract_json_object('{"answer": "lesion", "confidence": 0.9}') == \
        {"answer": "lesion", "confidence": 0.9}


def test_extract_json_fenced():
    text = "Reviewing the image:\n```json\n{\"answer\": \"not_lesion\"}\n```\nDone."
    assert extract_json_object(text) == {"answer": "not_lesion"}


def test_extract_json_balanced_block():
    text = 'The finding: {"answer": "uncertain", "note": "no {clear} rim"} — end.'
    assert extract_json_object(text) == {"answer": "uncertain", "note": "no {clear} rim"}


def test_extract_json_rejects_prose():
    import pytest

    for bad in ("no json at all", "", "[1, 2, 3]"):  # non-object JSON is also a failure
        with pytest.raises(ValueError):
            extract_json_object(bad)


# ----------------------------------------------------------------- image payloads
def test_image_to_data_url_from_bytes_and_path(tmp_path):
    url = image_to_data_url(png_bytes(8, 8))
    assert url.startswith("data:image/png;base64,")
    p = tmp_path / "x.png"
    p.write_bytes(png_bytes(8, 8, seed=1))
    assert image_to_data_url(str(p)).startswith("data:image/png;base64,")


# ----------------------------------------------------------------- client vs fake server
def test_client_full_http_path(medgemma_bf16_app, monkeypatch):
    """End-to-end: real client over real HTTP against the fake server (no torch).

    Uses a uvicorn thread on a random port -- the same path the study runner takes,
    including /healthz and the base64 image payload.
    """
    import threading
    import time
    import urllib.request

    import uvicorn

    config = uvicorn.Config(medgemma_bf16_app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started
    host, port = server.servers[0].sockets[0].getsockname()[:2]
    base_url = f"http://{host}:{port}"

    from medgemma_client import healthz

    assert healthz(base_url)["status"] == "ok"

    client = MedGemmaClient(base_url=base_url, model="google/medgemma-1.5-4b-it")
    result = client.chat_json('Reply with JSON only: {"answer": "lesion"|"no_lesion"|"uncertain"}.',
                              [png_bytes(16, 16)], max_tokens=64)
    assert result.parse_error is None
    assert result.parsed["answer"] == "uncertain"  # deterministic fake reply
    assert result.uncertain is True
    assert result.latency_ms > 0
    server.should_exit = True
    thread.join(timeout=10)


def test_client_chat_json_with_uncertain(medgemma_client_fixture, monkeypatch):
    client = MedGemmaClient(base_url="http://testserver", model="google/medgemma-1.5-4b-it")

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def read(self):
            return json.dumps(self.payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout):
        body = json.loads(request.data.decode())
        assert body["temperature"] == 0.0  # deterministic decoding is the default contract
        content = body["messages"][0]["content"]
        assert content[0]["type"] == "text"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        return Response({"choices": [{"message": {"role": "assistant", "content":
                                                  '{"answer": "uncertain", "confidence": 0.42}'}}],
                        "model": "google/medgemma-1.5-4b-it", "usage": {"total_tokens": 10}})

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = client.chat_json("Is the marked tissue lesion?", [png_bytes(16, 16)])
    assert result.parse_error is None
    assert result.parsed == {"answer": "uncertain", "confidence": 0.42}
    assert result.uncertain is True
    assert result.usage == {"total_tokens": 10}


def test_client_parse_failure_is_returned_not_raised(monkeypatch):
    client = MedGemmaClient(base_url="http://testserver")

    class Response:
        def read(self):
            return json.dumps({"choices": [{"message": {"role": "assistant",
                                                        "content": "I think it is a lesion, probably."}}],
                               "model": "m"}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: Response())
    result = client.chat_json("prompt")
    assert result.parsed is None
    assert result.parse_error
    assert "lesion" in result.text  # raw reply always kept for the log


# ----------------------------------------------------------------- server contract
def test_bf16_server_openai_shape(medgemma_client_fixture):
    body = medgemma_client_fixture.post("/v1/chat/completions", json={
        "model": "google/medgemma-1.5-4b-it",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "q"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64png(8, 8)}},
        ]}],
        "temperature": 0,
        "max_tokens": 32,
    }).json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "google/medgemma-1.5-4b-it"
    assert body["choices"][0]["message"]["role"] == "assistant"
    json.loads(body["choices"][0]["message"]["content"])  # fake replies are valid JSON
    assert "usage" in body


def test_bf16_server_rejects_bad_image(medgemma_client_fixture):
    r = medgemma_client_fixture.post("/v1/chat/completions", json={
        "model": "google/medgemma-1.5-4b-it",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "q"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!not-base64!!!"}},
        ]}],
    })
    assert r.status_code == 400


def test_bf16_server_healthz(medgemma_client_fixture):
    body = medgemma_client_fixture.get("/healthz").json()
    assert body["status"] == "ok" and body["backend"] == "fake"
    assert body["model"] == "google/medgemma-1.5-4b-it"
