"""Molmo2 service API contract + point parsing (fake backend, no torch).

Covers the research-note requirements: 0-1000 normalized -> pixel conversion,
multiple points in emission order, score=None (rank only), out-of-range filtering,
official `id x y` triplet grammar with a bare-pairs tolerance.
"""
from __future__ import annotations

from molmo_points import parse_molmo_points

from conftest import b64png


# ------------------------------------------------------------------- parser
def test_single_bare_pair():
    """No triplet (x,y must be 3-4 digits) -> bare-pairs tolerance."""
    pts = parse_molmo_points('<points coords="500 250">the lesion</points>', 200, 100)
    assert len(pts) == 1
    assert pts[0].x == 100.0 and pts[0].y == 25.0  # x/1000*W, y/1000*H
    assert pts[0].rank == 0 and pts[0].point_id is None


def test_multiple_points_in_order_with_ids():
    raw = ('Here are the points. <points coords="7 050 100 8 900 500">lesions</points> '
           'and also <points coords="9 010 200">other</points>')
    pts = parse_molmo_points(raw, 1000, 1000)
    assert len(pts) == 3
    assert [(p.x, p.y) for p in pts] == [(50.0, 100.0), (900.0, 500.0), (10.0, 200.0)]
    assert [p.rank for p in pts] == [0, 1, 2]
    assert [p.point_id for p in pts] == [7, 8, 9]  # ids are point ids, NOT ranks


def test_triplet_scan_is_sequential():
    """Official POINTS_REGEX consumes non-overlapping id-x-y triplets."""
    raw = '<points coords="0 400 200 1 900 400">two objects</points>'
    pts = parse_molmo_points(raw, 1000, 1000)
    assert [(p.x, p.y, p.point_id) for p in pts] == [(400.0, 200.0, 0), (900.0, 400.0, 1)]


def test_degenerate_coords_follow_official_strictness():
    """A coords stream whose 3-4 digit x/y alignment starts late only keeps the
    aligned triplet (official regex behaviour; out-of-frame numbers are not points)."""
    raw = '<points coords="0 100 1 900 500">x</points>'
    assert [(p.x, p.y, p.point_id) for p in parse_molmo_points(raw, 1000, 1000)] == [(900.0, 500.0, 1)]


def test_out_of_range_dropped():
    raw = '<points coords="5 1200 500 6 300 400">x</points>'
    # triplet (5, 1200, 500): x = 1200/1000*1000 px > image_w -> officially dropped;
    # the next triplet is kept.
    assert [(p.x, p.y, p.point_id) for p in parse_molmo_points(raw, 1000, 1000)] == [(300.0, 400.0, 6)]


def test_out_of_range_dropped_exact():
    pts = parse_molmo_points('<points coords="1 100 999 900">x</points>', 1000, 1000)
    # one triplet (id=1, x=100, y=999); the trailing 900 has no aligned x y pair
    assert [(p.x, p.y) for p in pts] == [(100.0, 999.0)]


def test_clamp_option_moves_instead_of_dropping():
    pts = parse_molmo_points('<points coords="5 1200 500">x</points>', 1000, 1000, clamp=True)
    assert (pts[0].x, pts[0].y) == (1000.0, 500.0)  # 1200/1000*1000 clamped to the border


def test_empty_and_bare_text():
    assert parse_molmo_points(None, 100, 100) == []
    assert parse_molmo_points("", 100, 100) == []
    assert parse_molmo_points("no points here", 100, 100) == []
    assert parse_molmo_points("mark 1 of 5 slices", 1000, 1000) == []  # prose numbers: no x y pair


def test_wrapped_in_code_fence():
    raw = "```xml\n<points coords=\"0 030 456\">lesion</points>\n```"
    assert [(p.x, p.y) for p in parse_molmo_points(raw, 100, 100)] == [(3.0, 45.6)]


# ------------------------------------------------------------------- API contract
def test_healthz(molmo2_client):
    body = molmo2_client.get("/healthz").json()
    assert body == {"status": "ok", "backend": "fake", "model": "allenai/Molmo2-8B",
                    "version": body["version"]}
    assert body["version"]


def test_point_contract_score_is_null(molmo2_client):
    body = molmo2_client.post("/point", json={"image_png_b64": b64png(300, 200), "query": "the lesion", "k": 3}).json()
    assert len(body["points"]) >= 1
    for p in body["points"]:
        assert set(("x", "y", "score", "rank", "point_id")) <= set(p)
        assert p["score"] is None  # Molmo2 has no calibrated scores: rank only
        assert 0 <= p["x"] <= 300 and 0 <= p["y"] <= 200
    assert body["model"] == "allenai/Molmo2-8B"
    assert "raw" in body and "latency_ms" in body and "version" in body


def test_point_k_caps_and_preserves_order(molmo2_client):
    body = molmo2_client.post("/point", json={"image_png_b64": b64png(), "query": "the lesion", "k": 2}).json()
    assert len(body["points"]) == 2
    assert [p["rank"] for p in body["points"]] == [0, 1]


def test_point_deterministic(molmo2_client):
    req = {"image_png_b64": b64png(77, 55), "query": "same question", "k": 3}
    a = molmo2_client.post("/point", json=req).json()
    b = molmo2_client.post("/point", json=req).json()
    assert a["points"] == b["points"] and a["raw"] == b["raw"]


def test_fake_raw_uses_molmo2_format(molmo2_client):
    """The fake backend emits a real <points coords=...> triplet span so the parser is
    on the served path (the 0-1000 -> pixel conversion is exercised end-to-end)."""
    body = molmo2_client.post("/point", json={"image_png_b64": b64png(200, 100), "query": "q", "k": 3}).json()
    assert "<points" in body["raw"] and "coords=" in body["raw"]
