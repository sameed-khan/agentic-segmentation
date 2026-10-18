"""Contract tests for segd.marks + segd.render (CONTRACTS §1–3, lane B).

Synthetic phantoms only (fast, well under 60 s; no segd.data, no GT loading).
Covers: compute_marks determinism, save/load round trip (kinds incl. the v<7
'regio' truncation), one mark ID resolving to the same voxel from all three
views, voxel_to_panel/panel_to_voxel round trips on full + cropped + scaled
views for every axis, visible() cap/level semantics (coarse ⊂ fine), render
size budgets, edge marks inside the crop box and inside the body, raw-pointing
ruler geometry, label placement inside panel bounds, and the additive
title_suffix / single_panel kwargs of render_view.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from segd import marks as M
from segd import render as R

# ---------------------------------------------------------------------------
# phantoms (module-scope fixtures: compute_marks is the only non-trivial cost)
# ---------------------------------------------------------------------------


def _phantom(shape=(96, 96, 72), seed=0) -> np.ndarray:
    """Noisy cylinder limb (~300) with bright/dark/soft internal structures so
    LoG extrema of both signs and supervoxels all find candidates."""
    rng = np.random.default_rng(seed)
    x, y, z = np.meshgrid(*(np.linspace(-1.0, 1.0, n, dtype=np.float32)
                            for n in shape), indexing="ij")
    limb = x ** 2 + y ** 2 < 0.75
    img = np.where(limb, 300.0, 0.0)
    img += np.where(((x - 0.3) / 0.18) ** 2 + ((y + 0.2) / 0.16) ** 2
                    + ((z - 0.3) / 0.3) ** 2 < 1, 220.0, 0.0)
    img -= np.where(((x + 0.25) / 0.2) ** 2 + ((y - 0.1) / 0.14) ** 2
                    + ((z + 0.2) / 0.3) ** 2 < 1, 250.0, 0.0)
    img += np.where(((x - 0.1) / 0.4) ** 2 + ((y + 0.4) / 0.22) ** 2
                    + ((z + 0.4) / 0.25) ** 2 < 1, 120.0, 0.0)
    img += rng.normal(0, 30, shape) * limb
    return img.astype(np.float32)


@pytest.fixture(scope="module")
def img_ms() -> tuple[np.ndarray, M.MarkSet]:
    img = _phantom((96, 96, 72), seed=0)
    return img, M.compute_marks(img, (1.0, 1.0, 1.0))


@pytest.fixture(scope="module")
def img_ms2() -> tuple[np.ndarray, M.MarkSet]:
    img = _phantom((120, 110, 64), seed=7)
    return img, M.compute_marks(img, (1.0, 1.0, 1.0))


# ---------------------------------------------------------------------------
# determinism + persistence
# ---------------------------------------------------------------------------


def test_compute_marks_deterministic():
    img = _phantom((72, 72, 48), seed=5)
    ms1 = M.compute_marks(img, (1.0, 1.0, 1.0))
    ms2 = M.compute_marks(img, (1.0, 1.0, 1.0))
    assert len(ms1) == len(ms2) > 0
    assert sorted(ms1.marks) == sorted(ms2.marks)
    for i in ms1.marks:
        a, b = ms1.marks[i], ms2.marks[i]
        assert (a.kind, a.ijk, a.level, a.scale_mm) == (b.kind, b.ijk, b.level, b.scale_mm)
    assert ms1.strengths == ms2.strengths
    assert ms1.window == ms2.window


def test_save_load_roundtrip_incl_kinds(img_ms, tmp_path):
    img, ms = img_ms
    kinds_present = {m.kind for m in ms.marks.values()}
    assert {"blob+", "blob-", "region"} <= kinds_present     # all kinds survive
    path = tmp_path / "marks.npz"
    ms.save(path)
    ms2 = M.MarkSet.load(path)
    assert len(ms2) == len(ms)
    assert tuple(ms2.shape or ()) == img.shape
    for mid, mk in ms.marks.items():
        b = ms2.marks[mid]
        assert (b.kind, b.ijk, b.level) == (mk.kind, mk.ijk, mk.level)
        assert b.scale_mm == pytest.approx(mk.scale_mm)
    assert ms2.strengths == ms.strengths
    assert ms2.window == ms.window

    d = np.load(path, allow_pickle=False)
    assert set(str(k) for k in d["kinds"]) == kinds_present
    assert "region" in set(str(k) for k in d["kinds"])       # not truncated by dtype
    params = json.loads(str(d["params"]))
    assert params["version"] == M.MARKS_VERSION


def test_legacy_regio_kind_maps_to_region_on_load(img_ms, tmp_path):
    """MARKS_VERSION <= 6 caches stored kinds as '<U5', which truncates 'region'
    to 'regio'; load must map the legacy spelling back."""
    img, ms = img_ms
    path = tmp_path / "legacy.npz"
    ms.save(path)
    raw = dict(np.load(path, allow_pickle=False))
    raw["kinds"] = np.asarray(["regio" if str(k) == "region" else str(k)
                               for k in raw["kinds"]], dtype="<U5")
    np.savez_compressed(path, **raw)

    ms2 = M.MarkSet.load(path)
    assert "regio" not in {m.kind for m in ms2.marks.values()}
    for mid, mk in ms.marks.items():
        assert ms2.marks[mid].kind == mk.kind


def test_compute_marks_tiny_volume_does_not_crash():
    """Volumes far below a supervoxel must not crash the level-assignment step
    (v6 dereferenced a missing label grid in _supervoxel_at)."""
    img = _phantom((24, 24, 18), seed=1)
    ms = M.compute_marks(img, (1.0, 1.0, 1.0))
    assert isinstance(ms, M.MarkSet)          # empty or degenerate is fine, no raise


# ---------------------------------------------------------------------------
# cross-view mark identity
# ---------------------------------------------------------------------------


def test_mark_id_resolves_to_same_voxel_from_all_views(img_ms):
    img, ms = img_ms
    coarse = [m for m in ms.marks.values() if m.level == "coarse"]
    assert coarse
    pick = max(coarse, key=lambda m: ms.strengths.get(m.id, -1.0))
    for axis in M.AXES:
        slice_idx = pick.ijk[M.AXIS_INDEX[axis]]          # view slice through the mark
        rows, cols = M.slice_shape(img.shape, axis)
        r, c = M.inplane_of(pick.ijk, axis, rows, cols)
        box = (max(0, r - 6), max(0, c - 6),              # tight crop, display coords
               min(rows, r + 7), min(cols, c + 7))
        _, ri = R.render_view(img, axis, slice_idx, box, "none", "coarse", ms,
                              single_panel=True)
        drawn = {mid: (x, y) for mid, _p, x, y in ri.marks}
        assert pick.id in drawn, f"mark {pick.id} missing on {axis} view"
        geom = ri.views[0]
        assert R.panel_to_voxel(axis, slice_idx, drawn[pick.id], geom) == pick.ijk
        ref = R.voxel_to_panel(axis, slice_idx, pick.ijk, geom)
        assert drawn[pick.id] == pytest.approx((ref[0] + geom.scale / 2.0,
                                                ref[1] + geom.scale / 2.0))


# ---------------------------------------------------------------------------
# pure coordinate round trips (CONTRACTS §1: voxel_to_panel / panel_to_voxel)
# ---------------------------------------------------------------------------


def _voxel_roundtrip(geoms, axis, slice_idx, rows, cols, rng, n=50) -> None:
    for geom in geoms:
        for _ in range(n):
            r = int(rng.integers(0, rows))
            c = int(rng.integers(0, cols))
            ijk = M.voxel_of_inplane(r, c, slice_idx, axis, rows, cols)
            x, y = R.voxel_to_panel(axis, slice_idx, ijk, geom)
            got = R.panel_to_voxel(axis, slice_idx, (x, y), geom)
            assert got == ijk                  # full inversion, no drift
            pr, pc = M.inplane_of(ijk, axis, rows, cols)
            gr, gc = M.inplane_of(got, axis, rows, cols)
            assert np.hypot(gr - pr, gc - pc) <= 0.5


def _cropped_geom(img, axis, slice_idx, rng) -> R.ViewGeom:
    rows, cols = M.slice_shape(img.shape, axis)
    r0 = int(rng.integers(0, rows // 2))
    r1 = min(rows, r0 + int(rng.integers(5, rows // 2 + 1)))
    c0 = int(rng.integers(0, cols // 2))
    c1 = min(cols, c0 + int(rng.integers(5, cols // 2 + 1)))
    _, ri = R.render_view(img, axis, slice_idx, (r0, c0, r1, c1))
    return ri.views[0]


def test_voxel_panel_roundtrip_full_cropped_scaled(img_ms, img_ms2):
    """Full + cropped (auto scaled) + synthetic re-scaled views, all three axes."""
    rng = np.random.default_rng(42)
    for img in (img_ms[0], img_ms2[0]):
        for axis in M.AXES:
            rows, cols = M.slice_shape(img.shape, axis)
            mid = int(img.shape[M.AXIS_INDEX[axis]] // 2)
            _, ri_full = R.render_view(img, axis, mid, "full")
            g_full = ri_full.views[0]
            g_crop = _cropped_geom(img, axis, mid, rng)
            g_up = R.ViewGeom(g_crop.axis, g_crop.slice_idx, g_crop.rows, g_crop.cols,
                              g_crop.crop, g_crop.scale * 2.0,
                              int(g_crop.panel_w * 2), int(g_crop.panel_h * 2))
            g_down = R.ViewGeom(g_crop.axis, g_crop.slice_idx, g_crop.rows, g_crop.cols,
                                g_crop.crop, g_crop.scale * 0.5,
                                max(1, int(g_crop.panel_w * 0.5)),
                                max(1, int(g_crop.panel_h * 0.5)))
            _voxel_roundtrip([g_full, g_crop, g_up, g_down],
                             axis, mid, rows, cols, rng)


def test_voxel_panel_roundtrip_through_other_renders(img_ms, img_ms2):
    for img, ms in (img_ms, img_ms2):
        rng = np.random.default_rng(11)
        _, ri_ov = R.render_overview(img, (1.0, 1.0, 1.0), marks=ms)
        for geom in ri_ov.views:
            _voxel_roundtrip([geom], geom.axis, geom.slice_idx, geom.rows, geom.cols,
                             rng, n=20)
        _, ri_sheet = R.render_sheet(img, "axial", 0, img.shape[2] - 1, 4, markset=ms)
        for geom in ri_sheet.views:
            _voxel_roundtrip([geom], geom.axis, geom.slice_idx, geom.rows, geom.cols,
                             rng, n=15)


# ---------------------------------------------------------------------------
# visible(): cap + level semantics (CONTRACTS §2)
# ---------------------------------------------------------------------------


def test_visible_level_semantics_coarse_subset_of_fine(img_ms, img_ms2):
    for img, ms in (img_ms, img_ms2):
        rng = np.random.default_rng(3)
        for axis in M.AXES:
            n_ax = img.shape[M.AXIS_INDEX[axis]]
            rows, cols = M.slice_shape(img.shape, axis)
            slices = {int(v) for v in (n_ax // 4, n_ax // 2, (3 * n_ax) // 4)} | \
                {int(s) for s in rng.integers(0, n_ax, size=6)}
            for s in slices:
                co = {m.id for m in ms.visible(axis, s, (0, 0, rows, cols), "coarse")}
                fi = {m.id for m in ms.visible(axis, s, (0, 0, rows, cols), "fine")}
                assert len(co) <= M.VISIBLE_CAP and len(fi) <= M.VISIBLE_CAP
                assert co <= fi, (axis, s, sorted(co - fi))
                assert all(ms.marks[i].level == "coarse" for i in co)


def test_visible_matches_render_drawn_marks(img_ms):
    img, ms = img_ms
    axis, k = "axial", img.shape[2] // 2
    _, ri = R.render_view(img, axis, k, "full", "none", "coarse", ms, single_panel=True)
    drawn_ids = {mid for mid, _p, _x, _y in ri.marks}
    vis_ids = {m.id for m in ms.visible(axis, k, None, "coarse")}
    assert drawn_ids <= vis_ids, "the render must draw only visible() marks"


# ---------------------------------------------------------------------------
# render outputs (CONTRACTS §3 budgets)
# ---------------------------------------------------------------------------


def test_render_view_size_budget(img_ms):
    img, ms = img_ms
    for axis in M.AXES:
        mid = int(img.shape[M.AXIS_INDEX[axis]] // 2)
        im, ri = R.render_view(img, axis, mid, "full", "outline", "coarse", ms)
        assert len(ri.views) == 2
        assert all(max(g.panel_w, g.panel_h) <= R.PANEL_LONG_PX for g in ri.views)
        assert im.width <= R.TOTAL_WIDTH_PX and max(im.size) <= R.TOTAL_WIDTH_PX


def test_render_other_kinds_size_budget(img_ms):
    img, ms = img_ms
    k = img.shape[2] // 2
    im, ri = R.render_overview(img, (1.0, 1.0, 1.0), marks=ms)
    assert im.width <= R.TOTAL_WIDTH_PX and im.height <= R.TOTAL_WIDTH_PX
    assert all(max(g.panel_w, g.panel_h) <= R.PANEL_LONG_PX for g in ri.views)

    im_s, ri_s = R.render_sheet(img, "axial", 0, img.shape[2] - 1, 9, markset=ms)
    assert im_s.width <= R.TOTAL_WIDTH_PX and im_s.height <= R.TOTAL_WIDTH_PX
    assert all(max(g.panel_w, g.panel_h) <= R.PANEL_LONG_PX for g in ri_s.views)

    body = M.body_mask(img)
    prev = np.roll(body, 2, axis=1).astype(np.uint8)
    im_o, ri_o = R.render_outline_change(img, prev, body.astype(np.uint8), "axial",
                                         markset=ms)
    assert max(im_o.size) <= R.TOTAL_WIDTH_PX
    assert max(ri_o.views[0].panel_w, ri_o.views[0].panel_h) <= R.PANEL_LONG_PX

    im_r, ri_r = R.render_raw_pointing(img, "axial", k, "full")
    g_r = ri_r.views[0]
    assert im_r.size == (g_r.panel_w + R.RAW_MARGIN_LEFT_PX,
                         g_r.panel_h + R.RAW_MARGIN_TOP_PX + R.RAW_MARGIN_BOTTOM_PX)
    assert max(ri_r.views[0].panel_w, ri_r.views[0].panel_h) <= R.PANEL_LONG_PX


def test_single_panel_mode_and_title_suffix(img_ms):
    img, ms = img_ms
    axis, k = "axial", img.shape[2] // 2
    kw = dict(overlay="outline", marks_level="coarse", markset=ms)
    pair, ri_pair = R.render_view(img, axis, k, "full", **kw)
    single, ri_single = R.render_view(img, axis, k, "full", single_panel=True, **kw)
    # single_panel output == the pair's right (marked) panel, pixel-exactly
    marked_x = pair.width - ri_pair.views[1].panel_w
    assert np.array_equal(np.asarray(single), np.asarray(pair)[:, marked_x:])
    assert len(ri_single.views) == 1
    assert all(p == 0 for _mid, p, _x, _y in ri_single.marks)
    # determinism + explicit default suffix equivalence
    pair2, _ = R.render_view(img, axis, k, "full", **kw)
    assert np.array_equal(np.asarray(pair2), np.asarray(pair))
    pair_aff, _ = R.render_view(img, axis, k, "full", title_suffix="| marks", **kw)
    assert np.array_equal(np.asarray(pair_aff), np.asarray(pair))
    # a custom suffix changes the chip, nothing else
    pair_suf, _ = R.render_view(img, axis, k, "full", title_suffix="| seg v3", **kw)
    assert not np.array_equal(np.asarray(pair_suf), np.asarray(pair))
    # clean example image: single panel with outline, no marks chip, no marks
    solo, _ = R.render_view(img, axis, k, "full", overlay="outline",
                            marks_level="none", single_panel=True)
    assert solo.size == (ri_pair.views[0].panel_w, ri_pair.views[0].panel_h)
    solo_n, ri_n = R.render_view(img, axis, k, "full", overlay="outline",
                                 marks_level="none", single_panel=True)
    assert ri_n.marks == [] and ri_n.edge_marks == []
    assert np.array_equal(np.asarray(solo), np.asarray(solo_n))


def test_render_overview_three_axes(img_ms):
    img, ms = img_ms
    im, ri = R.render_overview(img, (1.0, 1.0, 1.0), marks=ms)
    assert len(ri.views) == 3
    assert {g.axis for g in ri.views} == {"axial", "coronal", "sagittal"}
    assert len(ri.marks) > 0                            # coarse marks on all panels
    assert all(panel in (0, 1, 2) for _mid, panel, _x, _y in ri.marks)


# ---------------------------------------------------------------------------
# edge marks (CONTRACTS §2: E ids, box + body containment)
# ---------------------------------------------------------------------------


def test_edge_marks_inside_box_and_body(img_ms):
    img, _ = img_ms
    body = M.body_mask(img)
    k = img.shape[2] // 2
    rows, cols = M.slice_shape(img.shape, "axial")
    box = (rows // 6, cols // 6, (5 * rows) // 6, (5 * cols) // 6)
    _, ri = R.render_view(img, "axial", k, box, "none", "edges")
    body2d = M.extract_slice(body.astype(np.uint8), "axial", k).astype(bool)
    assert ri.edge_marks, "expected edge marks on the phantom limb slice"
    assert len(ri.edge_marks) <= 40
    ids = [lab for lab, _p, _r, _c in ri.edge_marks]
    assert len(set(ids)) == len(ids)
    for lab, panel, row, col in ri.edge_marks:
        assert panel == 1
        assert box[0] <= row < box[2] and box[1] <= col < box[3]
        assert body2d[row, col]


def _no_colored_pixels_in_border_ring(im: Image.Image, views) -> None:
    """No glyph/label-colored pixel within 2 px of a panel edge (label margin)."""
    arr = np.asarray(im.convert("RGB")).astype(int)
    x_off = 0
    colors = (R.KIND_COLOR["blob+"], R.KIND_COLOR["blob-"], R.KIND_COLOR["region"],
              R.EDGE_COLOR)
    for geom in views:
        w, h = geom.panel_w, geom.panel_h
        panel = arr[:, x_off:x_off + w]
        for color in colors:
            hit = np.abs(panel - np.asarray(color)).sum(axis=2) < 90
            ring = np.zeros_like(hit)
            ring[:, :2] = True
            ring[:, w - 2:] = True
            ring[:2, :] = True
            ring[h - 2:, :] = True
            assert int((hit & ring).sum()) == 0, \
                f"colored pixels within 2 px of panel edge ({geom.axis})"
        x_off += w + R.PANEL_GAP_PX


def test_label_placement_inside_panel_bounds(img_ms):
    """Stress _place_label with crops putting marks flush against panel edges.

    The crops place the picked mark at the last in-panel column while keeping it
    clear of the frame furniture (title chip, orientation letters, scale plate);
    labels at edge-flush positions must fall back to interior offsets and never
    touch the panel border.  Marks whose glyph would sit on frame furniture are
    dropped entirely (drawn marks stay referenceable).
    """
    img, ms = img_ms
    coarse = [m for m in ms.marks.values() if m.level == "coarse"]
    drawn_stress = 0
    for pick in coarse[:8]:
        rows, cols = M.slice_shape(img.shape, "axial")
        r, c = M.inplane_of(pick.ijk, "axial", rows, cols)
        if not (6 <= r <= rows - 3 and 6 <= c <= cols - 3):
            continue
        slice_idx = pick.ijk[2]
        for box in ((r - 4, c - 6, r + 2, c + 1), (r - 1, c - 3, r + 2, c + 1)):
            im, ri = R.render_view(img, "axial", slice_idx, box, "none", "fine", ms)
            drawn_stress += len(ri.marks)
            _no_colored_pixels_in_border_ring(im, ri.views)
    assert drawn_stress > 0, "stress configurations must have drawn marks"


# ---------------------------------------------------------------------------
# raw pointing ruler geometry (CONTRACTS §1 pure pair on panel-px coordinates)
# ---------------------------------------------------------------------------


def test_raw_pointing_ruler_geometry_consistent(img_ms):
    img, _ = img_ms
    axis = "axial"
    k = img.shape[2] // 2
    _, ri = R.render_raw_pointing(img, axis, k, "full")
    geom = ri.views[0]
    rows, cols = M.slice_shape(img.shape, axis)
    # every tick position maps through exactly the crop + floor(xy/scale) rule
    # that panel_to_voxel implements, on both the top and the left ruler
    for y in range(50, geom.panel_h, 50):
        exp = M.voxel_of_inplane(
            min(int(np.floor(y / geom.scale)) + geom.crop[0], rows - 1),
            geom.crop[1], k, axis, rows, cols)
        assert R.panel_to_voxel(axis, k, (0, y), geom) == exp
    for x in range(50, geom.panel_w, 50):
        exp = M.voxel_of_inplane(
            geom.crop[0],
            min(int(np.floor(x / geom.scale)) + geom.crop[1], cols - 1),
            k, axis, rows, cols)
        assert R.panel_to_voxel(axis, k, (x, 0), geom) == exp
    # and the same geometry round-trips voxels
    _voxel_roundtrip([geom], axis, k, rows, cols, np.random.default_rng(9), n=20)


def test_raw_pointing_rulers_sit_outside_the_image_area(img_ms):
    """Rulers/title/scale bar live in margins: the image area pasted at the margin offset
    is pixel-identical to the bare panel, so no image content is hidden."""
    img, _ = img_ms
    k = img.shape[2] // 2
    im, ri = R.render_raw_pointing(img, "axial", k, "full")
    g = ri.views[0]
    window = R._window_for(img, None)
    bare = R._panel_from_slice(M.extract_slice(img, "axial", k), window, g.crop, g.scale,
                               g.panel_w, g.panel_h)
    L, T = R.RAW_MARGIN_LEFT_PX, R.RAW_MARGIN_TOP_PX
    area = np.asarray(im)[T:T + g.panel_h, L:L + g.panel_w]
    assert np.array_equal(area, np.asarray(bare))
    assert max(g.panel_w, g.panel_h) <= R.PANEL_LONG_PX
    # the margins do carry the rulers
    assert (np.asarray(im)[:T] > 100).any() and (np.asarray(im)[:, :L] > 100).any()


# ---------------------------------------------------------------------------
# label boxes never cover glyph footprints (review #1/#10)
# ---------------------------------------------------------------------------
def _spy_labels(monkeypatch):
    boxes: list[tuple] = []
    orig = R._place_label

    def spy(draw, x, y, text, color, pw, ph, occupied):
        p = orig(draw, x, y, text, color, pw, ph, occupied)
        if p is not None:
            boxes.append((text, p[0]))
        return p

    monkeypatch.setattr(R, "_place_label", spy)
    return boxes


def _assert_no_label_over_glyph(boxes, glyph_centres, half=7):
    assert boxes and glyph_centres
    for text, box in boxes:
        for gx, gy in glyph_centres:
            fp = (gx - half, gy - half, gx + half, gy + half)
            assert not R._boxes_overlap(box, fp), f"label {text} box {box} covers glyph at {(gx, gy)}"


def test_no_label_box_overlaps_any_glyph_footprint(img_ms, monkeypatch):
    img, ms = img_ms
    k = img.shape[2] // 2
    rows, cols = M.slice_shape(img.shape, "axial")
    tight = (rows // 4, cols // 4, rows // 4 + 40, cols // 4 + 40)      # dense, large-scale crop
    for crop in ("full", tight):
        boxes = _spy_labels(monkeypatch)
        _, ri = R.render_view(img, "axial", k, crop, "none", "edges", ms)
        geom = ri.views[1]
        centres = [((c - geom.crop[1] + 0.5) * geom.scale, (r - geom.crop[0] + 0.5) * geom.scale)
                   for _lab, _p, r, c in ri.edge_marks]
        _assert_no_label_over_glyph(boxes, centres, half=7)
        assert len(boxes) == len(ri.edge_marks)
        boxes.clear()
        _, ri = R.render_view(img, "axial", k, crop, "none", "fine", ms)
        centres = [(x, y) for _m, _p, x, y in ri.marks]
        _assert_no_label_over_glyph(boxes, centres, half=7)
    # left-hand fallbacks exist and are right-aligned (they end before the glyph)
    assert any(side == "r" for side, _dx, _dy in R.LABEL_OFFSETS)


# ---------------------------------------------------------------------------
# edge marks are computed in the body, never restricted to the agent's mask (review #2)
# ---------------------------------------------------------------------------
def test_edge_marks_not_restricted_to_mask_and_prefer_boundary(img_ms):
    img, _ = img_ms
    body = M.body_mask(img)
    k = img.shape[2] // 2
    mask = np.zeros(img.shape, np.uint8)
    mask[40:56, 40:56, k - 2:k + 3] = 1                       # small agent mask inside the limb
    mask2d = M.extract_slice(mask, "axial", k).astype(bool)
    body2d = M.extract_slice(body.astype(np.uint8), "axial", k).astype(bool)
    _, ri = R.render_view(img, "axial", k, "full", "outline", "edges", None, mask=mask, body=body)
    pts = [(r, c) for _l, _p, r, c in ri.edge_marks]
    assert 0 < len(pts) <= 40
    assert any(not mask2d[r, c] for r, c in pts), "edge marks must also appear outside the mask"
    assert all(body2d[r, c] for r, c in pts)
    # candidates within 15 mm of the mask boundary rank first (lowest E numbers)
    from scipy import ndimage as ndi
    bnd = mask2d & ~ndi.binary_erosion(mask2d, structure=np.ones((3, 3), bool))
    dist = ndi.distance_transform_edt(~bnd)
    near = [dist[r, c] <= 15.0 for r, c in pts]
    assert near[0]
    n_near = sum(near)
    assert all(near[:n_near]), "near-boundary marks come before all others"
    # deterministic, and a mask does not shrink the candidate pool to the mask
    _, ri2 = R.render_view(img, "axial", k, "full", "outline", "edges", None, mask=mask, body=body)
    assert ri2.edge_marks == ri.edge_marks
    _, ri_nomask = R.render_view(img, "axial", k, "full", "none", "edges", None, body=body)
    assert any(not mask2d[r, c] for _l, _p, r, c in ri_nomask.edge_marks)


def test_edge_marks_without_body_keep_all_components():
    """Per-slice Otsu fallback keeps every component (two limbs -> marks on both)."""
    rng = np.random.default_rng(3)
    img = rng.normal(0, 5, (96, 96, 20)).astype(np.float32)
    ii, jj = np.meshgrid(np.arange(96), np.arange(96), indexing="ij")
    for cj in (24, 72):                                       # two separate cylinders
        img += np.where((ii - 48) ** 2 + (jj - cj) ** 2 < 14 ** 2, 300.0, 0.0)[:, :, None]
    _, ri = R.render_view(img, "axial", 10, "full", "none", "edges", None)
    js = np.asarray([c for _l, _p, _r, c in ri.edge_marks])
    # axial display: col = cols-1-j, so the two limbs land in the two column halves
    assert (js < 48).any() and (js >= 48).any()


# ---------------------------------------------------------------------------
# sheets crop to the whole-volume body bbox (review: sheets mostly air)
# ---------------------------------------------------------------------------
def test_sheet_cells_cropped_to_body_bbox(img_ms):
    img, _ = img_ms
    body = M.body_mask(img)
    for axis in ("axial", "coronal", "sagittal"):
        im, ri = R.render_sheet(img, axis, 0, img.shape[M.AXIS_INDEX[axis]] - 1, 4, body=body)
        g = ri.views[0]
        rows, cols = M.slice_shape(img.shape, axis)
        assert (g.crop[2] - g.crop[0]) * (g.crop[3] - g.crop[1]) < rows * cols
        proj = body.any(axis=M.AXIS_INDEX[axis])
        disp = np.flipud(np.fliplr(proj.T))
        rr, cc = np.nonzero(disp)
        assert g.crop[0] <= rr.min() and rr.max() < g.crop[2]          # whole body fits
        assert g.crop[1] <= cc.min() and cc.max() < g.crop[3]
        assert all(v.crop == g.crop for v in ri.views)
        assert im.width <= R.TOTAL_WIDTH_PX and im.height <= R.TOTAL_WIDTH_PX


# ---------------------------------------------------------------------------
# three-plane change feedback
# ---------------------------------------------------------------------------
def _expected_plane_box(axis, lo, hi, shape):
    corners = [(i, j, k) for i in (lo[0], hi[0] - 1) for j in (lo[1], hi[1] - 1)
               for k in (lo[2], hi[2] - 1)]
    rows, cols = M.slice_shape(shape, axis)
    rc = [M.inplane_of(c, axis, rows, cols) for c in corners]
    return (min(r for r, _ in rc), min(c for _, c in rc),
            max(r for r, _ in rc) + 1, max(c for _, c in rc) + 1)


@pytest.mark.parametrize("box", [((40, 56), (40, 52), (30, 40)),      # small: 30 mm margin rule
                                 ((10, 90), (10, 90), (5, 65))])      # large: 50% rule, clamped
def test_change_3plane_geometry_outlines_and_size(img_ms, box):
    img, ms = img_ms
    mask = np.zeros(img.shape, np.uint8)
    mask[tuple(slice(a, b) for a, b in box)] = 1
    prev = mask.copy()
    prev[box[0][1]:box[0][1] + 3, box[1][0]:box[1][1], box[2][0]:box[2][1]] = 1   # removed slab
    im, ri = R.render_change_3plane(img, prev, mask, markset=ms)
    assert ri.kind == "change3" and ri.note == "mask centre"
    assert [v.axis for v in ri.views] == ["axial", "coronal", "sagittal"]
    assert im.width <= R.TOTAL_WIDTH_PX
    assert all(max(v.panel_w, v.panel_h) <= R.PANEL_LONG_PX for v in ri.views)
    com = [int(np.floor(a.mean() + 0.5)) for a in np.nonzero(mask)]
    assert [v.slice_idx for v in ri.views] == [com[2], com[1], com[0]]
    lo = [b[0] for b in box]
    hi = [b[1] for b in box]
    grown_lo, grown_hi = [], []
    for a in range(3):
        m = int(np.ceil(max(30.0, 0.5 * (hi[a] - lo[a]))))
        grown_lo.append(max(0, lo[a] - m))
        grown_hi.append(min(img.shape[a], hi[a] + m))
    for v in ri.views:
        assert v.crop == _expected_plane_box(v.axis, grown_lo, grown_hi, img.shape)
    # new mask green and removed region red are visible in the panels
    arr = np.asarray(im).astype(int)
    green = (np.abs(arr - np.asarray(R.OUTLINE_COLOR)).sum(axis=2) < 30).sum()
    red = (np.abs(arr - np.asarray(R.REMOVED_COLOR)).sum(axis=2) < 30).sum()
    assert green > 100 and red > 50


def test_change_3plane_empty_mask_uses_changed_region_then_volume_centre(img_ms):
    img, _ = img_ms
    prev = np.zeros(img.shape, np.uint8)
    prev[20:30, 30:40, 40:50] = 1
    empty = np.zeros_like(prev)
    im, ri = R.render_change_3plane(img, prev, empty)
    assert ri.note == "changed region centre"
    assert [v.slice_idx for v in ri.views] == [45, 35, 25]    # floor(mean + 0.5) of 40..49, 30..39, 20..29
    im2, ri2 = R.render_change_3plane(img, empty, empty)
    assert ri2.note == "volume centre"
    assert [v.slice_idx for v in ri2.views] == [img.shape[2] // 2, img.shape[1] // 2, img.shape[0] // 2]
    assert im2.width <= R.TOTAL_WIDTH_PX
