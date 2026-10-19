"""Shared test helpers: synthetic phantom + contract stubs for segd.marks / segd.render.

The stubs implement CONTRACTS §2–3 exactly (Mark/MarkSet/render signatures) and are
installed via monkeypatch ONLY when the real modules are missing (or explicitly forced
for determinism). We never create real marks.py/render.py files from here.
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import numpy as np

from segd import config
from segd.case_state import (
    body_mask_fallback, inplane_to_voxel, slice2d_displayed, voxel_to_inplane,
)

try:  # real modules, if the concurrent agent has written them
    from segd import marks as real_marks
    from segd import render as real_render
except ImportError:
    real_marks = None
    real_render = None


# ------------------------------------------------------------------ phantom
def make_phantom(shape=(120, 120, 80), seed=0, z_halfwidth=0.25):
    """Noisy cylinder 'limb' (intensity ~300) with a bright ellipsoid lesion (~500).

    z_halfwidth shrinks the lesion along k for lasso/scribble tests (2D lasso covers
    a thin lesion once propagated). Returns (image float32, lesion bool, body bool).
    """
    rng = np.random.default_rng(seed)
    X, Y, Z = shape
    xx, yy, zz = np.meshgrid(*(np.linspace(-1, 1, n, dtype=np.float32) for n in shape), indexing="ij")
    body = (xx ** 2 + yy ** 2) < 0.8
    lesion = ((xx - 0.2) / 0.15) ** 2 + ((yy + 0.1) / 0.12) ** 2 + (zz / z_halfwidth) ** 2 < 1
    img = np.where(body, 300.0, 0.0) + np.where(lesion, 500.0, 0.0)
    img = img + rng.normal(0, 40, shape) * body
    return img.astype(np.float32), lesion, body


def lesion_center(lesion: np.ndarray) -> tuple:
    """Voxel at the lesion centroid (test-side GT knowledge; tests are offline)."""
    idx = np.nonzero(lesion)
    return tuple(int(round(float(a.mean()))) for a in idx)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * float((a & b).sum()) / float(s)


# ------------------------------------------------------------- stub marks
@dataclass
class StubMark:
    id: int
    kind: str                 # "blob+" | "blob-" | "region"
    ijk: tuple
    scale_mm: float
    level: str                # "coarse" | "fine"
    strength: float = 0.5     # priority for deterministic thinning (real MarkSet: .strengths)


class StubMarkSet:
    """Contract MarkSet: integer global marks, visibility window max(scale_mm/2, 2),
    strength-ordered greedy in-plane thinning, box2d as (r0, c0, r1, c1) display-oriented
    — mirroring the real segd.marks behavior."""

    def __init__(self, marks, shape=None, strengths=None):
        self.marks = {m.id: m for m in marks}
        if strengths:
            for mid, s in strengths.items():
                if mid in self.marks:
                    self.marks[mid].strength = float(s)
        self.shape = tuple(shape) if shape is not None else None

    def visible(self, axis: str, slice_idx: int, box2d, level: str):
        if level not in ("coarse", "fine") or self.shape is None:
            return []
        a = config.FIXED_AXIS[axis]
        cand = []
        for m in self.marks.values():
            if level == "coarse" and m.level != "coarse":
                continue
            if abs(m.ijk[a] - slice_idx) > max(m.scale_mm / 2.0, 2):
                continue
            r, c = voxel_to_inplane(axis, m.ijk, self.shape)
            if box2d is not None and not (box2d[0] <= r < box2d[2] and box2d[1] <= c < box2d[3]):
                continue
            cand.append((m, r, c))
        cand.sort(key=lambda t: (-t[0].strength, t[0].id))
        kept = []
        for m, r, c in cand:                       # greedy 6 mm in-plane thinning
            if all((r - r2) ** 2 + (c - c2) ** 2 > 36.0 for _m, r2, c2 in kept):
                kept.append((m, r, c))
        return [m for m, _r, _c in kept[:50]]

    def save(self, path) -> None:
        arr = np.asarray([[m.id, *m.ijk, m.scale_mm, 0 if m.level == "coarse" else 1, m.strength]
                          for m in self.marks.values()])
        np.savez(path, marks=arr, shape=np.asarray(self.shape or (0, 0, 0)))

    @classmethod
    def load(cls, path) -> "StubMarkSet":
        z = np.load(path)
        arr = z["marks"]
        marks = [StubMark(int(r[0]), "region", (int(r[1]), int(r[2]), int(r[3])), float(r[4]),
                          "coarse" if r[5] == 0 else "fine", float(r[6])) for r in arr]
        shape = tuple(int(v) for v in z["shape"]) if "shape" in z.files else None
        return cls(marks, shape)


def stub_compute_marks(image: np.ndarray, spacing) -> StubMarkSet:
    """Deterministic marks: coarse/fine grid points inside the body (step 24/6 vox)."""
    body = body_mask_fallback(image)
    marks: list[StubMark] = []
    nid = 1
    shape = image.shape
    for step, offset, level, scale, strength in ((24, 12, "coarse", 12.0, 0.9), (6, 3, "fine", 5.0, 0.4)):
        for i in range(offset, shape[0], step):
            for j in range(offset, shape[1], step):
                for k in range(offset, shape[2], step):
                    if body[i, j, k]:
                        marks.append(StubMark(nid, "region", (i, j, k), scale, level, strength))
                        nid += 1
    return StubMarkSet(marks, shape)


def stub_edge_marks_2d(slice2d: np.ndarray, mask2d_body, box2d=None, max_n=40):
    """Deterministic in-plane edge marks: body-boundary extremes (top/bottom/left/right)."""
    if not mask2d_body.any():
        return []
    rows = np.nonzero(mask2d_body.any(axis=1))[0]
    cols = np.nonzero(mask2d_body.any(axis=0))[0]
    pts = [(int(rows[0]), int(np.nonzero(mask2d_body[rows[0]])[0][0])),
           (int(rows[-1]), int(np.nonzero(mask2d_body[rows[-1]])[0][0])),
           (int(np.nonzero(mask2d_body[:, cols[0]])[0][0]), int(cols[0])),
           (int(np.nonzero(mask2d_body[:, cols[-1]])[0][0]), int(cols[-1]))]
    if box2d is not None:
        r0, c0, r1, c1 = (int(v) for v in box2d)
        pts = [(r, c) for r, c in pts if r0 <= r < r1 and c0 <= c < c1]
    return pts[:max_n]


# ------------------------------------------------------------ stub render
@dataclass
class StubRenderInfo:
    axis: str
    slice_idx: int
    view_geom: dict
    marks: list                  # [(id, (x, y))] panel positions of drawn global marks
    edge_marks: list             # [("E1", (row, col))] in-plane coords of drawn edge marks
    views: list | None = None    # per-panel geoms (added for v3); default [view_geom]
    panel_rects: list | None = None   # per-panel/row paste rects in the combined image (v3)

    def __post_init__(self):
        if self.views is None:
            self.views = [self.view_geom]


def _stub_geom(image, axis, slice_idx, crop_box2d=None):
    """view_geom for the stub: identity scale, no offset; panel px == displayed pixel."""
    a1, a2 = config.ROW_AXIS[axis], config.COL_AXIS[axis]
    return {"shape": tuple(image.shape), "axis": axis, "slice": int(slice_idx),
            "crop": crop_box2d, "scale": 1.0, "panel_size": (image.shape[a2], image.shape[a1])}


def stub_voxel_to_panel(axis: str, slice_idx: int, ijk, view_geom: dict):
    if view_geom.get("crop"):
        raise NotImplementedError("stub supports uncropped panels only")
    # panel px (x, y) = displayed (col, row)
    ijk = list(ijk)
    ijk[config.FIXED_AXIS[axis]] = slice_idx
    r, c = voxel_to_inplane(axis, tuple(ijk), tuple(view_geom["shape"]))
    return (float(c), float(r))


def stub_panel_to_voxel(axis: str, slice_idx: int, xy, view_geom: dict):
    if view_geom.get("crop"):
        raise NotImplementedError("stub supports uncropped panels only")
    a1, a2 = config.ROW_AXIS[axis], config.COL_AXIS[axis]
    geom_shape = tuple(view_geom["shape"])
    ijk = inplane_to_voxel(axis, int(round(xy[1])), int(round(xy[0])), slice_idx, geom_shape)
    return tuple(int(v) for v in ijk)


def _panel_image(slice2d: np.ndarray, marks=None, max_px=64):
    from PIL import Image, ImageDraw

    win_lo, win_hi = np.percentile(slice2d, [1, 99])
    arr = np.clip((slice2d - win_lo) / max(win_hi - win_lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    h, w = arr.shape
    scale = min(1.0, max_px / max(h, w))
    small = np.asarray(Image.fromarray(arr).resize((max(1, int(w * scale)), max(1, int(h * scale)))))
    img = Image.fromarray(small).convert("RGB")
    if marks:
        d = ImageDraw.Draw(img)
        for _mid, (x, y) in marks:
            d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 0, 0))
    return img


def _stub_slice_body(disp: np.ndarray) -> np.ndarray:
    """Per-slice body proxy when no mask slice is available (mirrors render.py)."""
    from skimage.filters import threshold_otsu
    if disp.max() <= disp.min():
        return np.ones(disp.shape, dtype=bool)
    return disp > threshold_otsu(disp)


def stub_render_overview(image, spacing, mask=None, marks=None):
    shape = image.shape
    panels = []
    recs = []
    for axis, s in (("axial", shape[2] // 2), ("coronal", shape[1] // 2), ("sagittal", shape[0] // 2)):
        disp = slice2d_displayed(image, axis, s)
        recs.append(StubRenderInfo(axis, s, _stub_geom(image, axis, s), [], []))
        panels.append(_panel_image(disp))
    from PIL import Image
    h = max(p.height for p in panels)
    w = sum(p.width for p in panels) + 4
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    x = 0
    rects = []
    for p in panels:
        canvas.paste(p, (x, 0))
        rects.append((x, 0, x + p.width, p.height))
        x += p.width + 4
    ri = StubRenderInfo("overview", shape[2] // 2, _stub_geom(image, "axial", shape[2] // 2),
                        [], [], views=[r.view_geom for r in recs], panel_rects=rects)
    return canvas, ri


def stub_render_sheet(image, axis, start, stop, n, mask=None, markset=None,
                      spacing=(1.0, 1.0, 1.0), body=None):
    from PIL import Image
    step = max(1, (stop - start) // max(n - 1, 1))
    slices = [start + k * step for k in range(n) if start + k * step <= stop]
    panels = [_panel_image(slice2d_displayed(image, axis, s)) for s in slices]
    h = max(p.height for p in panels)
    w = sum(p.width for p in panels) + 2 * len(panels)
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    x = 0
    rects = []
    views = []
    for p, s in zip(panels, slices):
        canvas.paste(p, (x, 0))
        rects.append((x, 0, x + p.width, p.height))
        views.append(_stub_geom(image, axis, s))
        x += p.width + 2
    ri = StubRenderInfo(axis, start, _stub_geom(image, axis, start), [], [],
                        views=views, panel_rects=rects)
    return canvas, ri


def stub_render_view(image, axis, slice_idx, crop, overlay, marks_level, markset,
                     mask=None, mark_boxes=None, spacing=(1.0, 1.0, 1.0), body=None):
    a1, a2 = config.ROW_AXIS[axis], config.COL_AXIS[axis]
    disp = slice2d_displayed(image, axis, slice_idx)
    body2d = None
    if mask is not None and (np.asarray(mask) > 0).any():
        fixed_axis = config.FIXED_AXIS[axis]
        body2d = _display_2d(np.take(np.asarray(mask > 0, dtype=np.uint8) * 255.0, int(slice_idx), axis=fixed_axis)) > 0
    drawn, edges = [], []
    if markset is not None and marks_level in ("coarse", "fine"):
        for m in markset.visible(axis, slice_idx, None, marks_level):
            ijk = list(m.ijk)
            ijk[config.FIXED_AXIS[axis]] = slice_idx
            r, c = voxel_to_inplane(axis, tuple(ijk), tuple(image.shape))
            drawn.append((m.id, (float(c), float(r))))  # panel px == displayed px (identity geom)
    if marks_level == "edges":
        b2d = body2d if (body2d is not None and body2d.any()) else _stub_slice_body(disp)
        for n_i, (r, c) in enumerate(stub_edge_marks_2d(disp, b2d, None)):
            edges.append((f"E{n_i + 1}", (r, c)))
    img = _panel_image(disp, marks=drawn)
    ri = StubRenderInfo(axis, int(slice_idx), _stub_geom(image, axis, slice_idx, None), drawn, edges,
                        panel_rects=[(0, 0, img.width, img.height)])
    return img, ri


def _display_2d(plane2d: np.ndarray) -> np.ndarray:
    """Voxel-order 2D plane -> display orientation (T then flip both)."""
    return np.ascontiguousarray(plane2d.T[::-1, ::-1])


def stub_render_outline_change(image, prev_mask, new_mask, axis, markset=None,
                               spacing=(1.0, 1.0, 1.0)):
    prev = prev_mask > 0
    new = new_mask > 0
    added = new & ~prev
    removed = prev & ~new
    both = new | prev
    if both.any():
        fixed = config.FIXED_AXIS[axis]
        counts = both.any(axis=tuple(i for i in range(3) if i != fixed))
        s = int(np.argmax(counts))
        sl = [slice(None)] * 3
        sl[fixed] = s
        disp_added = _display_2d(added[tuple(sl)])
        disp_removed = _display_2d(removed[tuple(sl)])
        img = _panel_image(_display_2d(np.take(image, s, axis=fixed)))
        from PIL import ImageDraw
        d = ImageDraw.Draw(img)
        ys, xs = np.nonzero(disp_added)
        for y, x in zip(ys[::7], xs[::7]):
            d.point((x, y), fill=(0, 255, 0))
        ys, xs = np.nonzero(disp_removed)
        for y, x in zip(ys[::7], xs[::7]):
            d.point((x, y), fill=(255, 0, 0))
    else:
        img = _panel_image(slice2d_displayed(image, axis, image.shape[config.FIXED_AXIS[axis]] // 2))
    return img, StubRenderInfo(axis, image.shape[config.FIXED_AXIS[axis]] // 2,
                               _stub_geom(image, axis, image.shape[config.FIXED_AXIS[axis]] // 2), [], [])


# ------------------------------------------------------------ case construction
def make_case_state(tmp_path, monkeypatch, case_id="phantom", shape=(120, 120, 80), budgets=None,
                    nni_url=None, crop_forced=False, z_halfwidth=0.25, seed=0):
    """Build an initialized CaseState on the phantom. install_stubs() must have been
    called first (marks module needed by initialize). crop_forced=True forces the
    body-bbox upload path by shrinking the packed-size limit to 1 byte."""
    from segd import config as cfg
    from segd.case_state import CaseState

    if crop_forced:
        monkeypatch.setattr(cfg, "UPLOAD_MAX_PACKED_BYTES", 1)
    img, lesion, _body = make_phantom(shape, seed=seed, z_halfwidth=z_halfwidth)
    cs = CaseState(case_id, img, np.eye(4), (1.0, 1.0, 1.0), "bright lesion in the limb", "T2",
                   tmp_path / "results" / "run1" / case_id, nni_url=nni_url, budgets=budgets)
    cs.initialize()
    return cs, lesion


def _mark_cls(markset):
    if isinstance(markset, StubMarkSet):
        return StubMark
    from segd.marks import Mark   # contract Mark dataclass (real module present)
    return Mark


def add_ring_marks(cs, center, dx=10, dy=8, scale_mm=5.0, axis="axial"):
    """Extend the case's MarkSet with 4 corner marks (polygon around the lesion, test-side
    GT knowledge) plus a centre mark, rebuilding the MarkSet (the real one caches parallel
    arrays). The ring lies in the DISPLAYED in-plane axes of ``axis`` (axial: i/j ring at
    the centre k; coronal: i/k ring at the centre j; sagittal: j/k ring at the centre i)
    so a view on that axis resolves all ring marks. ``dx`` offsets the column axis, ``dy``
    the row axis (CONTRACTS §1). Returns (corner_ids, centre_id)."""
    ms = cs.markset
    cls = _mark_cls(ms)
    a_row, a_col = config.ROW_AXIS[axis], config.COL_AXIS[axis]
    new_marks = dict(ms.marks)
    base = (max(new_marks) + 1) if new_marks else 1
    corner_ids = []
    for drow, dcol in ((-dy, -dx), (dy, -dx), (dy, dx), (-dy, dx)):
        ijk = [int(v) for v in center]
        ijk[a_row] += drow
        ijk[a_col] += dcol
        if isinstance(ms, StubMarkSet):
            new_marks[base] = StubMark(base, "region", tuple(ijk), scale_mm, "fine", strength=2.0)
        else:
            new_marks[base] = cls(id=base, kind="region", ijk=tuple(ijk), scale_mm=scale_mm, level="fine")
        corner_ids.append(base)
        base += 1
    centre_id = base
    if isinstance(ms, StubMarkSet):
        new_marks[base] = StubMark(base, "region", tuple(int(v) for v in center), scale_mm, "fine", strength=2.0)
    else:
        new_marks[base] = cls(id=base, kind="region", ijk=tuple(int(v) for v in center),
                              scale_mm=scale_mm, level="fine")
    if isinstance(ms, StubMarkSet):
        cs.markset = StubMarkSet(list(new_marks.values()), shape=ms.shape)
    else:
        from segd.marks import MarkSet
        strengths = dict(ms.strengths)
        for i in corner_ids + [centre_id]:
            strengths[i] = 2.0          # above the real prio range (max 1.0): never thinned away
        cs.markset = MarkSet(new_marks, ms.spacing, ms.window, strengths, shape=ms.shape)
    return corner_ids, centre_id


# --------------------------------------------------------- fake data loading
def install_fake_data(monkeypatch, shape=(120, 120, 80), z_halfwidth=0.25, case_ids=("phantom",)):
    """Replace segd.data case loading with a phantom-backed loader for HTTP tests.
    Returns {case_id: (image, lesion)}; GT stays in the test process only."""
    from dataclasses import dataclass as _dc
    from segd import data as data_mod

    @_dc
    class FakeCaseRow:
        case: str
        sequence: str = "T2"
        target_text: str = "bright lesion in the limb"

    phantoms = {}
    for i, cid in enumerate(case_ids):
        img, lesion, _body = make_phantom(shape, seed=i, z_halfwidth=z_halfwidth)
        phantoms[cid] = (img, lesion)

    def fake_get_case(case):
        if case not in phantoms:
            raise KeyError(case)
        return FakeCaseRow(case)

    def fake_load_image(row):
        img = phantoms[row.case][0]
        return img, np.eye(4), (1.0, 1.0, 1.0)

    def fake_load_case_geometry(row):
        img = phantoms[row.case][0]
        return data_mod.CaseGeometry(img, np.eye(4), (1.0, 1.0, 1.0), None, None)

    monkeypatch.setattr(data_mod, "get_case", fake_get_case)
    monkeypatch.setattr(data_mod, "load_image", fake_load_image)
    monkeypatch.setattr(data_mod, "load_case_geometry", fake_load_case_geometry)
    return phantoms


@dataclass
class _StubPlaneView:
    axis: str
    slice_idx: int


@dataclass
class StubChange3Info:
    kind: str
    views: list
    note: str


def stub_render_change_3plane(image, prev_mask, new_mask, markset=None, spacing=(1.0, 1.0, 1.0)):
    """Stub 3-plane feedback: three small panels through the new mask's centre."""
    from PIL import Image
    new = np.asarray(new_mask) > 0
    if new.any():
        centre = tuple(int(round(float(a.mean()))) for a in np.nonzero(new))
        note = "mask centre"
    else:
        centre = tuple(n // 2 for n in image.shape)
        note = "volume centre"
    views = [_StubPlaneView("axial", centre[2]), _StubPlaneView("coronal", centre[1]),
             _StubPlaneView("sagittal", centre[0])]
    panels = [_panel_image(slice2d_displayed(image, v.axis, v.slice_idx)) for v in views]
    canvas = Image.new("RGB", (sum(p.width for p in panels) + 8, max(p.height for p in panels)))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + 4
    return canvas, StubChange3Info("change3", views, note)


def stub_render_candidates(image, mask, candidates, spacing=(1.0, 1.0, 1.0), markset=None):
    """Stub candidate sheet: one small panel row per candidate (contract signature of
    segd.render.render_candidates; ops uses the image and v3 uses the row rects)."""
    from PIL import Image
    n_rows = max(1, len(list(candidates)))
    canvas = Image.new("RGB", (48, 16 * n_rows), (24, 24, 24))
    rows = [(0, 16 * i, 48, 16 * (i + 1)) for i in range(n_rows)]
    return canvas, StubRenderInfo("candidates", image.shape[config.FIXED_AXIS["axial"]] // 2,
                                  _stub_geom(image, "axial", image.shape[config.FIXED_AXIS["axial"]] // 2),
                                  [], [], panel_rects=rows)


# ------------------------------------------------------- stub installation
def _module(name: str) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    return mod


def install_stubs(monkeypatch, force: bool = False) -> None:
    """Install contract stubs for segd.marks / segd.render via monkeypatch.

    force=True always replaces the real modules (deterministic core tests);
    force=False only fills in missing modules (integration runs use the real ones).
    """
    need_marks = force or real_marks is None
    need_render = force or real_render is None
    if need_marks:
        mod = _module("segd.marks")
        monkeypatch.setitem(sys.modules, "segd.marks", mod)
        monkeypatch.setattr(mod, "Mark", StubMark, raising=False)
        monkeypatch.setattr(mod, "MarkSet", StubMarkSet, raising=False)
        monkeypatch.setattr(mod, "compute_marks", stub_compute_marks, raising=False)
        monkeypatch.setattr(mod, "body_mask", lambda image: body_mask_fallback(image), raising=False)
        monkeypatch.setattr(mod, "edge_marks_2d", stub_edge_marks_2d, raising=False)
    if need_render:
        mod = _module("segd.render")
        monkeypatch.setitem(sys.modules, "segd.render", mod)
        monkeypatch.setattr(mod, "RenderInfo", StubRenderInfo, raising=False)
        monkeypatch.setattr(mod, "voxel_to_panel", stub_voxel_to_panel, raising=False)
        monkeypatch.setattr(mod, "panel_to_voxel", stub_panel_to_voxel, raising=False)
        monkeypatch.setattr(mod, "render_overview", stub_render_overview, raising=False)
        monkeypatch.setattr(mod, "render_sheet", stub_render_sheet, raising=False)
        monkeypatch.setattr(mod, "render_view", stub_render_view, raising=False)
        monkeypatch.setattr(mod, "render_outline_change", stub_render_outline_change, raising=False)
        monkeypatch.setattr(mod, "render_change_3plane", stub_render_change_3plane, raising=False)
        monkeypatch.setattr(mod, "render_candidates", stub_render_candidates, raising=False)
