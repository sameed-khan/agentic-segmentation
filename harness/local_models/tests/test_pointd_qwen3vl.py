"""Qwen3-VL service API contract + grounding-output parsing (fake backend, no torch).

Covers the official Qwen3-VL grounding conventions (card + 2d_grounding cookbook):
JSON {"point_2d": [x, y], "label": ...} arrays on the 0-1000 relative scale,
bbox_2d -> centre fallback, code-fence stripping, out-of-range drops, 0-1000 ->
pixel conversion, pointd API contract (score=None, k cap), the appended
FORMAT_INSTRUCTION, and the process-wide inference lock.
"""
from __future__ import annotations

import json

from qwen3vl_points import parse_qwen3vl_points

from conftest import b64png


# ------------------------------------------------------------------- parser
def test_single_point_0_1000_to_pixels():
    raw = '[{"point_2d": [500, 250], "label": "tumor"}]'
    pts = parse_qwen3vl_points(raw, 200, 100)
    assert len(pts) == 1
    assert pts[0].x == 100.0 and pts[0].y == 25.0  # x/1000*W, y/1000*H
    assert pts[0].rank == 0 and pts[0].label == "tumor"


def test_label_optional():
    pts = parse_qwen3vl_points('[{"point_2d": [100, 100]}]', 100, 100)
    assert [(p.x, p.y, p.label) for p in pts] == [(10.0, 10.0, None)]


def test_multiple_points_in_order():
    raw = json.dumps([{"point_2d": [0, 0], "label": "a"}, {"point_2d": [1000, 1000], "label": "b"},
                      {"point_2d": [500, 500], "label": "c"}])
    pts = parse_qwen3vl_points(raw, 1000, 1000)
    assert [(p.x, p.y) for p in pts] == [(0.0, 0.0), (1000.0, 1000.0), (500.0, 500.0)]
    assert [p.rank for p in pts] == [0, 1, 2]


def test_code_fences_and_surrounding_prose():
    raw = 'Here are the points:\n```json\n[{"point_2d": [250, 500], "label": "lesion"}]\n```\nDone.'
    assert [(p.x, p.y) for p in parse_qwen3vl_points(raw, 100, 100)] == [(25.0, 50.0)]


def test_single_bare_object_wrapped():
    raw = '{"point_2d": [100, 200], "label": "x"}'
    assert [(p.x, p.y) for p in parse_qwen3vl_points(raw, 200, 100)] == [(20.0, 20.0)]


def test_bbox_2d_reduced_to_centre():
    """bbox_2d [x1, y1, x2, y2] (x first, per the cookbook) -> centre -> pixels."""
    raw = '[{"bbox_2d": [100, 200, 300, 400], "label": "tumor"}]'
    pts = parse_qwen3vl_points(raw, 1000, 1000)
    assert len(pts) == 1 and (pts[0].x, pts[0].y) == (200.0, 300.0)


def test_bbox_2d_swapped_corners_same_centre():
    raw = '[{"bbox_2d": [300, 400, 100, 200]}]'
    assert [(p.x, p.y) for p in parse_qwen3vl_points(raw, 1000, 1000)] == [(200.0, 300.0)]


def test_out_of_range_dropped():
    raw = '[{"point_2d": [1200, 500], "label": "off"}, {"point_2d": [300, 400]}]'
    assert [(p.x, p.y) for p in parse_qwen3vl_points(raw, 1000, 1000)] == [(300.0, 400.0)]


def test_negative_dropped():
    raw = '[{"point_2d": [-10, 50]}, {"point_2d": [10, 50]}]'
    assert [(p.x, p.y) for p in parse_qwen3vl_points(raw, 1000, 1000)] == [(10.0, 50.0)]


def test_floats_tolerated():
    raw = '[{"point_2d": [123.9, 456.1]}]'
    assert [(p.x, p.y) for p in parse_qwen3vl_points(raw, 1000, 1000)] == [(123.9, 456.1)]


def test_malformed_items_skipped_not_fatal():
    raw = ('[{"bbox_2d": "wrong"}, {"point_2d": [5]}, {"point_2d": [true, 3]}, '
           '{"point_2d": [50, 60], "label": "ok"}]')
    assert [(p.x, p.y) for p in parse_qwen3vl_points(raw, 1000, 1000)] == [(50.0, 60.0)]


def test_empty_and_non_json():
    assert parse_qwen3vl_points(None, 100, 100) == []
    assert parse_qwen3vl_points("", 100, 100) == []
    assert parse_qwen3vl_points("The tumor is visible near the centre.", 1000, 1000) == []
    assert parse_qwen3vl_points('[{"point_2d": [50, 60]', 100, 100) == []  # unparseable


# ------------------------------------------------------------------- API contract
def test_healthz(qwen3vl_client):
    body = qwen3vl_client.get("/healthz").json()
    assert body == {"status": "ok", "backend": "fake", "model": "Qwen/Qwen3-VL-8B-Instruct",
                    "version": body["version"]}
    assert body["version"]


def test_point_contract_score_is_null(qwen3vl_client):
    body = qwen3vl_client.post("/point", json={"image_png_b64": b64png(300, 200), "query": "Point to the fibrous soft-tissue tumor.", "k": 3}).json()
    assert len(body["points"]) >= 1
    for p in body["points"]:
        assert set(("x", "y", "score", "rank", "point_id")) <= set(p)
        assert p["score"] is None  # no calibrated scores: rank only
        assert 0 <= p["x"] <= 300 and 0 <= p["y"] <= 200
    assert body["model"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert "raw" in body and "latency_ms" in body and "version" in body


def test_user_text_appends_format_instruction(qwen3vl_server_mod, qwen3vl_client):
    text = qwen3vl_server_mod.user_text("Point to the fibrous soft-tissue tumor.")
    assert text.startswith("Point to the fibrous soft-tissue tumor.")
    assert text[len("Point to the fibrous soft-tissue tumor."):] == qwen3vl_server_mod.FORMAT_INSTRUCTION
    assert "point_2d" in text and "0-1000" in text
    body = qwen3vl_client.post("/point", json={"image_png_b64": b64png(), "query": "q", "k": 1}).json()
    assert body["format_instruction"] == qwen3vl_server_mod.FORMAT_INSTRUCTION


def test_point_k_caps_and_preserves_order(qwen3vl_client):
    body = qwen3vl_client.post("/point", json={"image_png_b64": b64png(), "query": "the lesion", "k": 2}).json()
    assert len(body["points"]) == 2
    assert [p["rank"] for p in body["points"]] == [0, 1]


def test_point_deterministic(qwen3vl_client):
    req = {"image_png_b64": b64png(77, 55), "query": "same question", "k": 3}
    a = qwen3vl_client.post("/point", json=req).json()
    b = qwen3vl_client.post("/point", json=req).json()
    assert a["points"] == b["points"] and a["raw"] == b["raw"]


def test_fake_raw_is_qwen_json_array(qwen3vl_client):
    """The fake backend emits the native point_2d JSON (fenced) so the parser is on
    the served path (0-1000 -> pixel conversion exercised end-to-end)."""
    body = qwen3vl_client.post("/point", json={"image_png_b64": b64png(200, 100), "query": "q", "k": 3}).json()
    assert "point_2d" in body["raw"]
    assert body["image_w"] == 200 and body["image_h"] == 100


def test_invalid_png_400(qwen3vl_client):
    import base64
    r = qwen3vl_client.post("/point", json={"image_png_b64": base64.b64encode(b"notapng").decode(),
                                            "query": "q", "k": 3})
    assert r.status_code == 400


# ----------------------------------------------------------------- inference lock
def test_inference_lock_serializes_concurrent_requests(qwen3vl_server_mod):
    """Two concurrent /point requests NEVER overlap inside backend.predict (the
    process-wide _INFER_LOCK serialises inference)."""
    import asyncio
    import threading

    import httpx

    class RecordingBackend(qwen3vl_server_mod.FakeBackend):
        def __init__(self):
            super().__init__(delay_s=0.05)
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def predict(self, png_bytes, query, k, max_new_tokens):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                return super().predict(png_bytes, query, k, max_new_tokens)
            finally:
                with self.lock:
                    self.active -= 1

    backend = RecordingBackend()
    app = qwen3vl_server_mod.create_app(backend)

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
