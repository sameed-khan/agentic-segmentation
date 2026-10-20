"""MolmoPoint service API contract + grounding-token decode (fake backend, no torch).

Covers the card's pointing decode (modeling_molmo_point.py mirror in
molmopoint_points.py): EXTRACT_POINT_TRIPLE token text, patch->grid-cell lookup,
3x3 location refinement, original-image pixel scaling, emission-order ranks,
object-id passthrough, and the pointd API contract (score=None, k cap).
"""
from __future__ import annotations

import numpy as np

from molmopoint_points import decode_image_points, parse_point_triples

from conftest import b64png

# Same synthetic geometry as the fake backend: 2x3 grid, one subpatch per patch row.
POOLING = [[i] for i in range(6)]
MAPPING = [[0, 1, 2], [3, 4, 5]]


def _point_token_text(vit_id: int, location_id: int, object_id: int,
                      no_more_points_class: bool = True, n_pool_rows: int = 6) -> str:
    """Raw token text for one point (server token numbering: subpatch numbers start
    after the patch rows, +1 when the no_more_points class exists)."""
    n_patches = n_pool_rows + (1 if no_more_points_class else 0)
    return (f"<POINT_{vit_id}> <POINT_{n_patches}> "
            f"<POINT_{location_id + n_patches + 1}> {object_id}")


# ------------------------------------------------------------------- parser
def test_parse_point_triples_emission_order():
    raw = "<POINT_3> <POINT_7> <POINT_10> 0 <POINT_5> <POINT_7> <POINT_12> 2"
    assert parse_point_triples(raw) == [(3, 7, 10, 0), (5, 7, 12, 2)]


def test_parse_point_triples_no_match():
    assert parse_point_triples(None) == []
    assert parse_point_triples("") == []
    assert parse_point_triples("The fibrous tumor is visible. <POINT_1>") == []
    assert parse_point_triples("no tokens at all") == []


def test_decode_single_point_with_location_refinement():
    """Cell (row 1, col 2) + location 4 (centre of the 3x3): mirrors the official math
    p_x = 2 + (1+0.5)*0.33, p_y = 1 + (1+0.5)*0.33 -> scaled to the ORIGINAL image."""
    raw = _point_token_text(vit_id=5, location_id=4, object_id=0)
    pts = decode_image_points(raw, POOLING, [MAPPING], [(300, 100)])
    assert len(pts) == 1
    p = pts[0]
    assert p.x == (2 + 1.5 * 0.33) / 3 * 300
    assert p.y == (1 + 1.5 * 0.33) / 2 * 100
    assert p.point_id == 0 and p.rank == 0


def test_decode_location_offset_zero_is_corner():
    """Location 0 -> +(0.5)*0.33 in both axes (loc_x = loc_y = 0)."""
    raw = _point_token_text(vit_id=0, location_id=0, object_id=1)
    p = decode_image_points(raw, POOLING, [MAPPING], [(300, 100)])[0]
    assert np.isclose(p.x, (0 + 0.5 * 0.33) / 3 * 300)
    assert np.isclose(p.y, (0 + 0.5 * 0.33) / 2 * 100)


def test_decode_scales_to_original_image_size():
    raw = _point_token_text(vit_id=0, location_id=4, object_id=0)
    small = decode_image_points(raw, POOLING, [MAPPING], [(150, 50)])[0]
    big = decode_image_points(raw, POOLING, [MAPPING], [(600, 200)])[0]
    assert (big.x, big.y) == (small.x * 4, small.y * 4)  # same cell fractions, 4x image


def test_decode_multiple_points_across_images_keeps_emission_order():
    """Two points in image 0 and one in image 1; the decoder returns all with
    (object_id, image_index, x, y) exactly like the card (the single-image service
    filters image_index 0). Image 1's grid holds ViT ids 6-9 (not in image 0's)."""
    raw = (_point_token_text(5, 4, 0, n_pool_rows=10) + " "
           + _point_token_text(0, 0, 1, n_pool_rows=10) + " "
           + _point_token_text(6, 8, 2, n_pool_rows=10))
    pooling = [[i] for i in range(10)]
    pts = decode_image_points(raw, pooling, [MAPPING, [[6, 7], [8, 9]]],
                              [(300, 100), (100, 100)])
    assert len(pts) == 3
    assert [(p.point_id, p.image_ix, p.rank) for p in pts] == [(0, 0, 0), (1, 0, 1), (2, 1, 2)]
    # 2nd point is in image 0 cell (0,0) at location 0; 3rd point is in image 1 cell (0,0) at location 8
    assert np.isclose(pts[1].x, (0 + 0.5 * 0.33) / 3 * 300)
    assert np.isclose(pts[2].x, (0 + 2.5 * 0.33) / 2 * 100)


def test_decode_invalid_patch_id_skipped():
    """A generated patch row whose ViT id is absent from the mapping is skipped
    (official 'Invalid patch id encountered' behaviour); later points keep going."""
    pooling = [[i] for i in range(7)]  # row 6 pools ViT id 6, absent from MAPPING
    raw = _point_token_text(6, 4, 0, n_pool_rows=7) + " " + _point_token_text(2, 4, 1, n_pool_rows=7)
    pts = decode_image_points(raw, pooling, [MAPPING], [(300, 100)])
    assert len(pts) == 1 and pts[0].point_id == 1 and pts[0].rank == 0


def test_decode_mismatched_mappings_and_sizes_raises():
    raw = _point_token_text(5, 4, 0)
    try:
        decode_image_points(raw, POOLING, [MAPPING], [(300, 100), (300, 100)])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_token_numbering_without_no_more_points_class():
    """no_more_points_class=False: subpatch numbers start one lower (no +1)."""
    raw = _point_token_text(5, 4, 0, no_more_points_class=False)
    pts = decode_image_points(raw, POOLING, [MAPPING], [(300, 100)], no_more_points_class=False)
    assert len(pts) == 1
    assert np.isclose(pts[0].x, (2 + 1.5 * 0.33) / 3 * 300)


def test_no_more_points_stop_token_ignored():
    """The trailing <POINT_6> (no_more_points token) must not become a point."""
    raw = _point_token_text(5, 4, 0) + " <POINT_6>"
    assert len(decode_image_points(raw, POOLING, [MAPPING], [(300, 100)])) == 1


# ------------------------------------------------------------------- API contract
def test_healthz(molmopoint_client):
    body = molmopoint_client.get("/healthz").json()
    assert body == {"status": "ok", "backend": "fake", "model": "allenai/MolmoPoint-8B",
                    "version": body["version"]}
    assert body["version"]


def test_point_contract_score_is_null(molmopoint_client):
    body = molmopoint_client.post("/point", json={"image_png_b64": b64png(300, 200), "query": "Point to the fibrous soft-tissue tumor.", "k": 3}).json()
    assert len(body["points"]) >= 1
    for p in body["points"]:
        assert set(("x", "y", "score", "rank", "point_id")) <= set(p)
        assert p["score"] is None  # MolmoPoint has no calibrated scores: rank only
        assert 0 <= p["x"] <= 300 and 0 <= p["y"] <= 200
    assert body["model"] == "allenai/MolmoPoint-8B"
    assert "raw" in body and "latency_ms" in body and "version" in body


def test_point_k_caps_and_preserves_order(molmopoint_client):
    body = molmopoint_client.post("/point", json={"image_png_b64": b64png(), "query": "the lesion", "k": 2}).json()
    assert len(body["points"]) == 2
    assert [p["rank"] for p in body["points"]] == [0, 1]


def test_point_deterministic(molmopoint_client):
    req = {"image_png_b64": b64png(77, 55), "query": "same question", "k": 3}
    a = molmopoint_client.post("/point", json=req).json()
    b = molmopoint_client.post("/point", json=req).json()
    assert a["points"] == b["points"] and a["raw"] == b["raw"]


def test_fake_raw_uses_grounding_token_format(molmopoint_client):
    """The fake backend emits real <POINT_N> grounding-token text so the official
    decode math (mirror) is on the served path end-to-end."""
    body = molmopoint_client.post("/point", json={"image_png_b64": b64png(200, 100), "query": "q", "k": 3}).json()
    assert "<POINT_" in body["raw"]
    assert "raw" in body and body["image_w"] == 200 and body["image_h"] == 100


# ----------------------------------------------------------------- inference lock
def test_inference_lock_serializes_concurrent_requests(molmopoint_server_mod):
    """Two concurrent /point requests NEVER overlap inside backend.predict (the
    process-wide _INFER_LOCK serialises inference)."""
    import asyncio
    import threading

    import httpx

    class RecordingBackend(molmopoint_server_mod.FakeBackend):
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
    app = molmopoint_server_mod.create_app(backend)

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
