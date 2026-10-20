"""Panel rendering for the segd harness (design §5.5, CONTRACTS §1 and §3).

Every render function returns ``(PIL.Image, RenderInfo)``.  ``RenderInfo`` carries
the geometry of each panel plus the drawn marks (with panel positions) and edge
marks (with in-plane coordinates) so ``CaseState`` can resolve agent references
back to voxels without raw coordinates appearing in text.

Display convention (radiological, CONTRACTS §1) is defined once in
:mod:`segd.marks` (``extract_slice`` & co) and imported here.  MSTT data is
1 mm isotropic; ``spacing`` kwargs default to that.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

from segd import marks as M

__all__ = [
    "ViewGeom",
    "RenderInfo",
    "render_view",
    "render_overview",
    "render_sheet",
    "render_outline_change",
    "render_change_3plane",
    "render_review_grid",
    "render_review_components",
    "render_candidates",
    "render_raw_pointing",
    "draw_proposed_points",
    "voxel_to_panel",
    "panel_to_voxel",
]

PANEL_LONG_PX = 768          # long side of each panel (CONTRACTS §3)
TOTAL_WIDTH_PX = 1568        # side-by-side budget (Claude image limit)
PANEL_GAP_PX = 16
LABEL_FONT_PX = 14
SMALL_FONT_PX = 11
LABEL_MIN_DIST_PX = 26.0     # panel-space greedy thinning between glyphs
LABEL_OFFSETS = (            # (anchor, dx, dy) label placements from the glyph centre (px)
    ("l", 15, 0), ("r", -15, 0),          # 'l': text starts at x+dx; 'r': text ENDS at x+dx
    ("l", 20, -10), ("r", -20, -10), ("l", 20, 10), ("r", -20, 10),
    ("c", 0, -20), ("c", 0, 20),          # 'c': text centred on x+dx
    ("l", 26, 0), ("r", -26, 0),
)

KIND_COLOR = {               # bright fills + black halo stay legible on any tissue
    "blob+": (255, 214, 0),    # amber  ○
    "blob-": (0, 229, 255),    # cyan   □
    "region": (255, 61, 143),  # pink   △
}
EDGE_COLOR = (167, 111, 255)   # purple ◇
OUTLINE_COLOR = (0, 230, 118)  # green (mask outline, new mask)
REMOVED_COLOR = (255, 59, 48)  # red (removed region)
BOX_COLOR = (255, 152, 0)      # orange (mark_boxes)
ORIENTATION = {                # left, right, top, bottom letters per axis
    "axial": ("R", "L", "A", "P"),
    "coronal": ("R", "L", "S", "I"),
    "sagittal": ("A", "P", "S", "I"),
}
_FONT_PATH = Path(__file__).parent / "fonts" / "DejaVuSans-Bold.ttf"
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}

_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)


def _font(size: int) -> ImageFont.FreeTypeFont:
    f = _FONT_CACHE.get(size)
    if f is None:
        f = ImageFont.truetype(str(_FONT_PATH), size)
        _FONT_CACHE[size] = f
    return f


# ---------------------------------------------------------------------------
# Geometry containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ViewGeom:
    """Panel geometry: crop box (display in-plane, half-open) and panel scale."""
    axis: str
    slice_idx: int
    rows: int                     # display rows of the full slice (voxels)
    cols: int
    crop: tuple[int, int, int, int]   # (r0, c0, r1, c1) in-plane, display-oriented
    scale: float                  # panel px per voxel
    panel_w: int
    panel_h: int


@dataclass
class RenderInfo:
    """What was drawn, for ID resolution by CaseState (CONTRACTS §3).

    ``views`` holds one ``ViewGeom`` per panel (a seg_view pair shares one crop:
    panel 0 = clean, panel 1 = marked).  ``marks`` entries are
    ``(mark_id, panel_idx, x_px, y_px)`` with the glyph-centre position;
    ``edge_marks`` entries are ``("E1", panel_idx, row, col)`` with in-plane
    display coordinates of the 2D slice array.

    ``panel_rects`` (optional, added for harness v3): the paste rectangle
    ``(x0, y0, x1, y1)`` of every recorded panel in the combined image — per panel
    for overview/sheet/view, per candidate ROW for render_candidates — so offline
    components (seg_locate clean-panel crop, MedGemma second-reader checkpoints) can
    map combined-image pixels back to a panel/voxel. None when not recorded.
    """
    kind: str
    views: list[ViewGeom]
    marks: list[tuple[int, int, float, float]] = field(default_factory=list)
    edge_marks: list[tuple[str, int, int, int]] = field(default_factory=list)
    window: tuple[float, float] | None = None
    note: str = ""                # render_change_3plane: what the slices pass through
    panel_rects: list[tuple[int, int, int, int]] | None = None

    def geom(self, panel_idx: int = 0) -> ViewGeom:
        return self.views[panel_idx]


# ---------------------------------------------------------------------------
# Pure coordinate mapping (CONTRACTS §1; used by rendering and the pointing study)
# ---------------------------------------------------------------------------
def voxel_to_panel(axis: str, slice_idx: int, ijk: tuple[int, int, int],
                   view_geom: ViewGeom) -> tuple[float, float]:
    """Voxel -> panel pixel of the voxel footprint's top-left corner.

    Pure function of ``(axis, slice_idx, view_geom)``.  The glyph centre used for
    drawing is this plus half a voxel footprint (``scale/2``).
    """
    r, c = M.inplane_of(tuple(int(v) for v in ijk), axis, view_geom.rows, view_geom.cols)
    x = (c - view_geom.crop[1]) * view_geom.scale
    y = (r - view_geom.crop[0]) * view_geom.scale
    return (float(x), float(y))


def panel_to_voxel(axis: str, slice_idx: int, xy: tuple[float, float],
                   view_geom: ViewGeom) -> tuple[int, int, int]:
    """Panel pixel -> voxel whose footprint contains it (clamped to the slice).

    Inverse of :func:`voxel_to_panel`: for ``xy = voxel_to_panel(...)`` this
    returns exactly the input voxel.
    """
    x, y = float(xy[0]), float(xy[1])
    c = int(np.floor(x / view_geom.scale + 1e-6)) + view_geom.crop[1]
    r = int(np.floor(y / view_geom.scale + 1e-6)) + view_geom.crop[0]
    return M.voxel_of_inplane(r, c, slice_idx, axis, view_geom.rows, view_geom.cols)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
_WINDOW_CACHE: dict[tuple, tuple[float, float]] = {}


def _window_for(image: np.ndarray, markset: M.MarkSet | None) -> tuple[float, float]:
    """0.5/99.5 percentiles inside the body (CONTRACTS §3).

    Prefers the window cached on ``markset`` (computed at seg_open); otherwise
    computes :func:`segd.marks.body_mask`, cached by content hash so repeated
    calls on the same volume stay cheap.
    """
    if markset is not None and getattr(markset, "window", None):
        return markset.window
    key = (image.shape, hashlib.sha1(np.ascontiguousarray(image[::8, ::8, ::8])).digest())
    win = _WINDOW_CACHE.get(key)
    if win is None:
        win = M.window_params(image, M.body_mask(image))
        if len(_WINDOW_CACHE) > 3:
            _WINDOW_CACHE.clear()
        _WINDOW_CACHE[key] = win
    return win


def _resolve_crop(crop, rows: int, cols: int, mask2d: np.ndarray | None
                  ) -> tuple[int, int, int, int]:
    """crop spec -> (r0, c0, r1, c1) in-plane display box, clamped to the slice."""
    if crop == "full":
        return (0, 0, rows, cols)
    if crop in ("mask", "marks"):
        if mask2d is None or not mask2d.any():
            return (0, 0, rows, cols)
        rr, cc = np.nonzero(mask2d)
        r0, r1 = int(rr.min()), int(rr.max()) + 1
        c0, c1 = int(cc.min()), int(cc.max()) + 1
        return _grow_box((r0, c0, r1, c1), 15, rows, cols)   # +15 mm margin (CONTRACTS §3)
    if isinstance(crop, (tuple, list, np.ndarray)) and len(crop) == 4:
        r0, c0, r1, c1 = (int(round(float(v))) for v in crop)
        return (max(0, r0), max(0, c0), min(rows, r1), min(cols, c1))
    raise ValueError(f"crop must be 'full', 'mask'/'marks' or a 4-tuple, got {crop!r}")


def _grow_box(box: tuple[int, int, int, int], margin: int, rows: int, cols: int
              ) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return (max(0, r0 - margin), max(0, c0 - margin), min(rows, r1 + margin), min(cols, c1 + margin))


def _fit_panel(crop_box: tuple[int, int, int, int]) -> tuple[float, int, int]:
    """Scale so the long side is ~768 px (upscale small crops, downscale large)."""
    r0, c0, r1, c1 = crop_box
    ch, cw = max(r1 - r0, 1), max(c1 - c0, 1)
    scale = min(PANEL_LONG_PX / max(ch, cw), 32.0)   # cap extreme upscales
    panel_w = max(1, int(round(cw * scale)))
    panel_h = max(1, int(round(ch * scale)))
    return scale, panel_w, panel_h


def _to_rgb(arr2d: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    lo, hi = window
    w = np.clip((arr2d.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    u8 = (w * 255.0).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


def _mask_boundary(mask2d: np.ndarray) -> np.ndarray:
    m = mask2d.astype(bool)
    if not m.any():
        return m
    return m & ~ndi.binary_erosion(m, structure=np.ones((3, 3), bool))


def _panel_from_slice(slice2d: np.ndarray, window: tuple[float, float],
                      crop_box: tuple[int, int, int, int], scale: float,
                      panel_w: int, panel_h: int) -> Image.Image:
    """Windowed crop, upscaled to panel size (bilinear)."""
    r0, c0, r1, c1 = crop_box
    sub = slice2d[r0:r1, c0:c1]
    im = Image.fromarray(_to_rgb(sub, window), mode="RGB")
    return im.resize((panel_w, panel_h), Image.Resampling.BILINEAR)


def _overlay_mask(im: Image.Image, mask2d: np.ndarray, crop_box, scale: float,
                  mode: str, color=OUTLINE_COLOR) -> Image.Image:
    """Outline (default) and/or translucent fill of a 2D mask on a panel copy."""
    if mask2d is None or not np.asarray(mask2d).any() or mode == "none":
        return im
    r0, c0, r1, c1 = crop_box
    sub = np.asarray(mask2d)[r0:r1, c0:c1].astype(bool)
    mh = max(1, int(round((r1 - r0) * scale)))
    mw = max(1, int(round((c1 - c0) * scale)))
    sub_img = Image.fromarray((sub * 255).astype(np.uint8), mode="L")
    sub_img = sub_img.resize((mw, mh), Image.Resampling.NEAREST)
    mup = np.asarray(sub_img) > 127
    out = im.copy()
    if mode == "fill":
        arr = np.asarray(out).astype(np.float32)
        fill = np.zeros_like(arr)
        fill[..., 0], fill[..., 1], fill[..., 2] = color
        arr[mup] = 0.62 * arr[mup] + 0.38 * fill[mup]
        return Image.fromarray(arr.astype(np.uint8), mode="RGB")
    bnd = _mask_boundary(sub)
    draw = ImageDraw.Draw(out)
    for rr, cc in zip(*np.nonzero(bnd)):
        x0, y0 = cc * scale, rr * scale
        draw.rectangle((x0, y0, min((cc + 1) * scale, mw) - 0.02,
                        min((rr + 1) * scale, mh) - 0.02), fill=color)
    return out


def _draw_boxes(im: Image.Image, boxes, crop_box, scale: float) -> Image.Image:
    if not boxes:
        return im
    draw = ImageDraw.Draw(im)
    r0, c0 = crop_box[0], crop_box[1]
    for box in boxes:
        br0, bc0, br1, bc1 = (int(v) for v in box)
        draw.rectangle(((bc0 - c0) * scale, (br0 - r0) * scale,
                        (bc1 - c0) * scale - 0.02, (br1 - r0) * scale - 0.02),
                       outline=BOX_COLOR, width=2)
    return im


def _glyph(draw: ImageDraw.ImageDraw, x: float, y: float, kind: str, color) -> None:
    """Kind glyph centred at (x, y): blob+ ○, blob- □, region △ (CONTRACTS §3)."""
    s = 6.0
    if kind == "blob+":
        draw.ellipse((x - s, y - s, x + s, y + s), outline=_BLACK, width=4)
        draw.ellipse((x - s, y - s, x + s, y + s), outline=color, width=2)
    elif kind == "blob-":
        draw.rectangle((x - s, y - s, x + s, y + s), outline=_BLACK, width=4)
        draw.rectangle((x - s, y - s, x + s, y + s), outline=color, width=2)
    else:  # region
        pts = [(x, y - s - 1), (x - s - 1, y + s), (x + s + 1, y + s)]
        draw.polygon(pts, outline=_BLACK, width=3)
        draw.polygon(pts, outline=color, width=1)


def _diamond(draw: ImageDraw.ImageDraw, x: float, y: float, color) -> None:
    s = 7.0
    pts = [(x, y - s), (x + s, y), (x, y + s), (x - s, y)]
    draw.polygon(pts, outline=_BLACK, width=3)
    draw.polygon(pts, outline=color, width=1)


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str,
                color, font_size: int = LABEL_FONT_PX) -> None:
    draw.text(xy, text, font=_font(font_size), fill=color,
              stroke_width=2, stroke_fill=_BLACK)


def _boxes_overlap(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _place_label(draw, x: float, y: float, text: str, color, panel_w: int, panel_h: int,
                 occupied: list[tuple]) -> tuple[tuple, tuple] | None:
    """First non-overlapping label box around (x, y); None when nothing fits.

    The box is the exact drawn text extent plus 1 px halo, and the stroke/halo is
    included via ``stroke_width=2`` + the pad in ``box`` so the drawn text can
    never extend past its box.  Positions within 2 px of a panel edge are
    rejected (no half-clipped labels at a panel border); all other offsets are
    tried before the mark is dropped entirely.  Left-side placements are
    right-aligned (text ends at the offset), so they clear the glyph's own
    footprint instead of overlapping it.
    """
    font = _font(LABEL_FONT_PX)
    text_w = float(draw.textlength(text, font=font))
    for side, dx, dy in LABEL_OFFSETS:
        ax = x + dx - (text_w if side == "r" else text_w / 2.0 if side == "c" else 0.0)
        anchor = (ax, y + dy - LABEL_FONT_PX / 2.0 - 1)
        l, t, r, b = draw.textbbox(anchor, text, font=font, stroke_width=2)
        box = (l - 1, t - 1, r + 1, b + 1)
        if box[0] < 2 or box[1] < 2 or box[2] > panel_w - 2 or box[3] > panel_h - 2:
            continue
        if any(_boxes_overlap(box, ob) for ob in occupied):
            continue
        return (box, anchor)
    return None


def _draw_marks(panel: Image.Image, marks_list, geom: ViewGeom,
                occupied: list[tuple] | None = None) -> list[tuple[int, float, float]]:
    """Glyphs + numbered labels with collision avoidance (CONTRACTS §3).

    ``marks_list`` is strongest-first (order from ``MarkSet.visible``).  Glyphs
    are thinned in panel space; a mark whose label cannot be placed is dropped
    entirely (glyph included, lowest priority first) so the agent never sees an
    unreferenceable glyph.  Returns ``[(mark_id, x, y), ...]`` for RenderInfo.
    """
    occupied = list(occupied or [])
    if not marks_list:
        return []
    draw = ImageDraw.Draw(panel)

    # 1) panel-space greedy thinning, strongest first
    kept: list[tuple] = []
    for m in marks_list:
        x, y = voxel_to_panel(geom.axis, geom.slice_idx, m.ijk, geom)
        x += geom.scale / 2.0
        y += geom.scale / 2.0
        if not (0 <= x < geom.panel_w and 0 <= y < geom.panel_h):
            continue
        if any((x - px) ** 2 + (y - py) ** 2 < LABEL_MIN_DIST_PX ** 2 for _, px, py in kept):
            continue
        kept.append((m, x, y))

    # 2) glyph footprints first: drop marks whose glyph would sit on frame
    #    furniture (title chip / orientation letters / scale-bar plate are
    #    already in `occupied`) and reserve every surviving footprint BEFORE
    #    any label is placed — then no later glyph can paint over an earlier
    #    label, and each placed label's box enters `occupied` immediately so
    #    the next label cannot overlap it
    kept_frame: list[tuple] = []
    for m, x, y in kept:
        footprint = (x - 7, y - 7, x + 7, y + 7)
        if any(_boxes_overlap(footprint, ob) for ob in occupied):
            continue
        occupied.append(footprint)
        kept_frame.append((m, x, y))
    placements: list[tuple] = []
    for m, x, y in kept_frame:
        p = _place_label(draw, x, y, str(m.id), KIND_COLOR.get(m.kind, _WHITE),
                         geom.panel_w, geom.panel_h, occupied)
        if p is None:
            continue   # drop the whole mark; weakest are tried last
        placements.append((m, x, y, p))
        occupied.append(p[0])

    # 3) draw glyphs (successes only), then labels
    for m, x, y, _ in placements:
        _glyph(draw, x, y, m.kind, KIND_COLOR.get(m.kind, _WHITE))
    for m, x, y, (box, text_xy) in placements:
        _draw_label(draw, text_xy, str(m.id), KIND_COLOR.get(m.kind, _WHITE))
    return [(m.id, x, y) for m, x, y, _ in placements]


def _draw_edge_marks(panel: Image.Image, edge_pts, geom: ViewGeom,
                     occupied: list[tuple] | None = None) -> list[tuple[str, int, int]]:
    """Diamond glyphs with 'E<n>' labels; returns [(label, row, col), ...].

    Like :func:`_draw_marks`: every surviving diamond footprint is reserved in
    ``occupied`` BEFORE any label is placed, so no label can cover any diamond
    (its own or a neighbour's) and no diamond can paint over a placed label.  A
    mark whose label cannot be placed is dropped together with its glyph.
    """
    occupied = list(occupied or [])
    if not edge_pts:
        return []
    draw = ImageDraw.Draw(panel)
    r0, c0 = geom.crop[0], geom.crop[1]
    kept: list[tuple] = []
    for n, (row, col) in enumerate(edge_pts, start=1):
        x = (col - c0) * geom.scale + geom.scale / 2.0
        y = (row - r0) * geom.scale + geom.scale / 2.0
        if not (7 <= x < geom.panel_w - 7 and 7 <= y < geom.panel_h - 7):
            continue
        diamond = (x - 8, y - 8, x + 8, y + 8)     # glyph footprint + 1 px pad
        if any(_boxes_overlap(diamond, ob) for ob in occupied):
            continue                               # glyph would cover frame furniture / marks
        occupied.append(diamond)
        kept.append((n, row, col, x, y))
    out = []
    placed = []
    for n, row, col, x, y in kept:
        p = _place_label(draw, x, y, f"E{n}", EDGE_COLOR, geom.panel_w, geom.panel_h, occupied)
        if p is None:
            continue
        occupied.append(p[0])
        placed.append((n, row, col, x, y, p))
    for n, row, col, x, y, p in placed:
        _diamond(draw, x, y, EDGE_COLOR)
    for n, row, col, x, y, p in placed:
        _draw_label(draw, p[1], f"E{n}", EDGE_COLOR)
        out.append((f"E{n}", int(row), int(col)))
    return out


def _chip(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str,
          font_size: int = SMALL_FONT_PX) -> tuple[float, float, float, float]:
    """White text on a black chip; returns its bounding box (for label avoidance)."""
    f = _font(font_size)
    l, t, r, b = draw.textbbox(xy, text, font=f)
    pad = 3
    draw.rectangle((l - pad, t - pad, r + pad, b + pad), fill=_BLACK)
    draw.text(xy, text, font=f, fill=_WHITE)
    return (l - pad, t - pad, r + pad, b + pad)


def _draw_frame(panel: Image.Image, geom: ViewGeom, spacing, title: str,
                letters: bool = True, scale_bar: bool = True,
                chip_pos: str = "tl") -> list[tuple]:
    """Orientation letters, 10 mm scale bar, slice-index chip (CONTRACTS §1).

    Returns the occupied boxes so mark labels can avoid them.
    """
    draw = ImageDraw.Draw(panel)
    occupied: list[tuple] = []
    fs = SMALL_FONT_PX
    if chip_pos == "tl":
        occupied.append(_chip(draw, (4, 4), title, fs))
    else:
        # 'bl': bottom-left — the scale-bar plate owns the bottom-right corner
        # (the chip used to be drawn bottom-right and was then covered by it)
        occupied.append(_chip(draw, (4, geom.panel_h - fs - 10), title, fs))
    if letters:
        l, r, t, b = ORIENTATION[geom.axis]
        top_x = (geom.panel_w - 18) / 2
        if chip_pos == "tl":
            # narrow panels: keep the top letter clear of the title chip
            top_x = max(top_x, occupied[0][2] + 3)
        bot_x = (geom.panel_w - 18) / 2
        if chip_pos != "tl":
            bot_x = max(bot_x, occupied[0][2] + 3)
        for letter, (x, y) in zip((l, r, t, b),
                                  ((4, (geom.panel_h - 18) / 2),
                                   (geom.panel_w - 18, (geom.panel_h - 18) / 2),
                                   (top_x, 4),
                                   (bot_x, geom.panel_h - 18))):
            occupied.append(_chip(draw, (x, y), letter, 12))
    if scale_bar:
        ra, ca = M._INPLANE[geom.axis][0], M._INPLANE[geom.axis][2]
        mm_per_vox = float(np.mean((spacing[ra], spacing[ca])))
        bar_px = 10.0 / mm_per_vox * geom.scale
        # the bar and its label share one black plate fully inside the panel:
        # the label used to start at the bar's left end and overflow the right
        # panel edge on full views ('10 mm' printed as '10 mr')
        f_bar = _font(fs)
        tw = float(draw.textlength("10 mm", font=f_bar))
        right = geom.panel_w - 4
        x_r = right                                 # right end of the bar line
        x_l = x_r - min(bar_px, max(4.0, right - tw - 16))
        y = geom.panel_h - 22
        label_l = x_l - tw - 8
        if label_l < 4:                             # tiny panel: bar without label
            label_l = x_l
            draw.rectangle((x_l - 4, y - 10, x_r + 4, y + 12), fill=_BLACK)
            occupied.append((x_l - 4, y - 10, x_r + 4, y + 12))
        else:
            draw.rectangle((label_l - 3, y - 10, x_r + 4, y + 12), fill=_BLACK)
            draw.text((label_l, y), "10 mm", font=f_bar, fill=_WHITE, anchor="lm")
            occupied.append((label_l - 3, y - 10, x_r + 4, y + 12))
        draw.line((x_l, y, x_r, y), fill=_WHITE, width=3)
        draw.line((x_l, y - 4, x_l, y + 4), fill=_WHITE, width=2)
        draw.line((x_r, y - 4, x_r, y + 4), fill=_WHITE, width=2)
    return occupied


def _combine(panels: list[Image.Image]) -> Image.Image:
    gap = PANEL_GAP_PX
    w = sum(p.width for p in panels) + gap * (len(panels) - 1)
    h = max(p.height for p in panels)
    out = Image.new("RGB", (w, h), (24, 24, 24))
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += p.width + gap
    return out


# ---------------------------------------------------------------------------
# Public render functions
# ---------------------------------------------------------------------------
def render_view(image: np.ndarray, axis: str, slice_idx: int, crop="full",
                overlay: str = "outline", marks_level: str = "none",
                markset: M.MarkSet | None = None, mask: np.ndarray | None = None,
                mark_boxes=None, spacing=(1.0, 1.0, 1.0),
                title_suffix: str = "| marks", single_panel: bool = False,
                body: np.ndarray | None = None
                ) -> tuple[Image.Image, RenderInfo]:
    """Clean + marked panels side by side (CONTRACTS §3).

    ``crop``: ``"full"``, ``"mask"`` (mask bbox + 15 mm margin; needs ``mask``),
    ``"marks"`` (same resolution as ``"mask"``; mark-driven crops are computed by
    the caller and passed as an explicit box) or an explicit ``(r0, c0, r1, c1)``
    in-plane box in display coordinates.  ``marks_level``: ``"none" | "coarse" |
    "fine" | "edges"``.  Panel 0 is clean, panel 1 carries overlay + marks and
    the title chip ``{axis} {slice} {title_suffix}`` (default ``" | marks"``,
    unchanged behaviour).

    Optional additions (recorded under CONTRACTS Deviations):
    ``title_suffix`` lets callers change the marked panel's chip without touching
    the contract defaults; ``single_panel=True`` returns ONE panel (image +
    overlay + marks, marks recorded with panel index 0) instead of the pair —
    for example figures that only need a clean panel with a mask outline.
    ``body`` (optional 3D body mask) is the search region for ``marks_level="edges"``
    (all body components on the slice); without it a per-slice Otsu foreground
    (all components) is used.  The agent's mask never restricts edge marks; it
    only ranks candidates near its boundary first.
    """
    if axis not in M.AXIS_INDEX:
        raise ValueError(f"axis must be one of {M.AXES}, got {axis!r}")
    image = np.asarray(image)
    spacing = tuple(float(s) for s in spacing)
    slice_idx = int(slice_idx)
    rows, cols = M.slice_shape(image.shape, axis)
    window = _window_for(image, markset)
    slice2d = M.extract_slice(image, axis, slice_idx)
    mask2d = M.extract_slice(mask, axis, slice_idx).astype(bool) if mask is not None else None
    crop_box = _resolve_crop(crop, rows, cols, mask2d)
    scale, panel_w, panel_h = _fit_panel(crop_box)
    geom = ViewGeom(axis, slice_idx, rows, cols, crop_box, scale, panel_w, panel_h)
    marked_panel = 0 if single_panel else 1

    marked = _panel_from_slice(slice2d, window, crop_box, scale, panel_w, panel_h)
    marked = _overlay_mask(marked, mask2d, crop_box, scale, overlay)
    if single_panel and marks_level == "none":
        title = f"{axis} {slice_idx}"          # clean example figure: no '| marks' chip
    else:
        title = f"{axis} {slice_idx} {title_suffix}"
    occupied = _draw_frame(marked, geom, spacing, title)

    info_marks: list[tuple[int, int, float, float]] = []
    edge_list: list[tuple[str, int, int, int]] = []
    if marks_level in ("coarse", "fine") and markset is not None:
        vis = markset.visible(axis, slice_idx, crop_box, marks_level)
        info_marks = [(mid, marked_panel, x, y)
                      for mid, x, y in _draw_marks(marked, vis, geom, occupied)]
    elif marks_level == "edges":
        body2d = _edge_body2d(slice2d, M.extract_slice(body, axis, slice_idx)
                              if body is not None else None)
        edge_pts = M.edge_marks_2d(slice2d, body2d, crop_box, max_n=40,
                                   prefer_mask2d=mask2d)
        edge_list = [(lab, marked_panel, row, col)
                     for lab, row, col in _draw_edge_marks(marked, edge_pts, geom, occupied)]
    marked = _draw_boxes(marked, mark_boxes, crop_box, scale)

    if single_panel:
        out = marked
        views: list[ViewGeom] = [geom]
        rects: list[tuple[int, int, int, int]] = [(0, 0, panel_w, panel_h)]
    else:
        clean = _panel_from_slice(slice2d, window, crop_box, scale, panel_w, panel_h)
        _draw_frame(clean, geom, spacing, f"{axis} {slice_idx}")
        out = _combine([clean, marked])
        views = [geom, geom]
        rects = [(0, 0, panel_w, panel_h),
                 (panel_w + PANEL_GAP_PX, 0, 2 * panel_w + PANEL_GAP_PX, panel_h)]
    info = RenderInfo("view", views, info_marks, edge_list, window, panel_rects=rects)
    return out, info


def _edge_body2d(slice2d: np.ndarray, body2d: np.ndarray | None) -> np.ndarray:
    """2D body region for edge marks: the body mask's slice when given, else a
    per-slice Otsu foreground keeping ALL components (never the agent's mask)."""
    if body2d is not None and np.asarray(body2d).any():
        return np.asarray(body2d).astype(bool)
    s = np.asarray(slice2d, dtype=np.float32)
    if s.max() <= s.min():
        return np.zeros(s.shape, bool)
    m = s > threshold_otsu(s[s > s.min()])
    return ndi.binary_fill_holes(m)


def render_overview(image: np.ndarray, spacing, mask: np.ndarray | None = None,
                    marks: M.MarkSet | None = None) -> tuple[Image.Image, RenderInfo]:
    """Three orthogonal panels through the volume/mask centre (CONTRACTS §3)."""
    image = np.asarray(image)
    spacing = tuple(float(s) for s in spacing)
    if mask is not None and np.asarray(mask).any():
        idx = np.argwhere(np.asarray(mask) > 0).mean(axis=0)
        centre = tuple(int(v) for v in np.clip(np.floor(idx + 0.5), [0, 0, 0],
                                               np.asarray(image.shape) - 1))
    else:
        centre = tuple(d // 2 for d in image.shape)
    window = _window_for(image, marks)
    panel_long = (TOTAL_WIDTH_PX - 2 * PANEL_GAP_PX) // 3   # 512 each -> 1568 total
    views: list[ViewGeom] = []
    panels: list[Image.Image] = []
    info_marks: list[tuple[int, int, float, float]] = []
    for panel_idx, (axis, slice_idx) in enumerate(
            (("axial", centre[2]), ("coronal", centre[1]), ("sagittal", centre[0]))):
        rows, cols = M.slice_shape(image.shape, axis)
        crop_box = (0, 0, rows, cols)
        sc = min(panel_long / max(rows, cols), 32.0)
        pw = max(1, int(round(cols * sc)))
        ph = max(1, int(round(rows * sc)))
        geom = ViewGeom(axis, slice_idx, rows, cols, crop_box, sc, pw, ph)
        slice2d = M.extract_slice(image, axis, slice_idx)
        panel = _panel_from_slice(slice2d, window, crop_box, sc, pw, ph)
        if mask is not None:
            panel = _overlay_mask(panel, M.extract_slice(mask, axis, slice_idx),
                                  crop_box, sc, "outline")
        occupied = _draw_frame(panel, geom, spacing, f"{axis} {slice_idx}")
        if marks is not None:
            vis = marks.visible(axis, slice_idx, None, "coarse", max_n=40)
            info_marks += [(mid, panel_idx, x, y)
                           for mid, x, y in _draw_marks(panel, vis, geom, occupied)]
        views.append(geom)
        panels.append(panel)
    combined = _combine(panels)
    rects: list[tuple[int, int, int, int]] = []
    x = 0
    for p in panels:
        rects.append((x, 0, x + p.width, p.height))
        x += p.width + PANEL_GAP_PX
    return combined, RenderInfo("overview", views, info_marks, [], window, panel_rects=rects)


def _body_crop_box(image: np.ndarray, axis: str, body: np.ndarray | None,
                   rows: int, cols: int, margin: int = 4) -> tuple[int, int, int, int]:
    """Display-oriented in-plane bbox of the whole-volume body on ``axis``."""
    if body is None:
        body = M.body_mask(image)
    proj = np.asarray(body).astype(bool).any(axis=M.AXIS_INDEX[axis])   # [a, b] voxel order
    if not proj.any():
        return (0, 0, rows, cols)
    disp = np.flipud(np.fliplr(proj.T))
    rr, cc = np.nonzero(disp)
    return _grow_box((int(rr.min()), int(cc.min()), int(rr.max()) + 1, int(cc.max()) + 1),
                     margin, rows, cols)


def render_sheet(image: np.ndarray, axis: str, start: int, stop: int, n: int,
                 mask: np.ndarray | None = None, markset: M.MarkSet | None = None,
                 spacing=(1.0, 1.0, 1.0), body: np.ndarray | None = None
                 ) -> tuple[Image.Image, RenderInfo]:
    """Contact sheet of ``n`` slices labelled with slice indices (CONTRACTS §3).

    Every cell is cropped to the whole-volume body bounding box on this axis (+4
    voxel margin) so cells are not mostly air; ``body`` is the 3D body mask
    (computed from the image when omitted).
    """
    if axis not in M.AXIS_INDEX:
        raise ValueError(f"axis must be one of {M.AXES}, got {axis!r}")
    image = np.asarray(image)
    spacing = tuple(float(s) for s in spacing)
    n_ax = image.shape[M.AXIS_INDEX[axis]]
    start, stop = max(0, int(start)), min(n_ax - 1, int(stop))
    if stop < start:
        start, stop = 0, n_ax - 1
    idxs = sorted(set(np.linspace(start, stop, max(1, int(n))).round().astype(int).tolist()))
    window = _window_for(image, markset)
    rows, cols = M.slice_shape(image.shape, axis)
    crop_box = _body_crop_box(image, axis, body, rows, cols)
    r0, c0, r1, c1 = crop_box
    ncols = int(np.ceil(np.sqrt(len(idxs))))
    nrows = int(np.ceil(len(idxs) / ncols))
    cell_long = min(384, (TOTAL_WIDTH_PX - (ncols - 1) * 4) // max(ncols, 1),
                    (TOTAL_WIDTH_PX - (nrows - 1) * 4) // max(nrows, 1))
    scale = min(cell_long / max(r1 - r0, c1 - c0), 32.0)
    cw = max(1, int(round((c1 - c0) * scale)))
    ch = max(1, int(round((r1 - r0) * scale)))
    pad = 4
    out = Image.new("RGB", (ncols * cw + (ncols - 1) * pad, nrows * ch + (nrows - 1) * pad),
                    (24, 24, 24))
    views: list[ViewGeom] = []
    rects: list[tuple[int, int, int, int]] = []
    for n_i, si in enumerate(idxs):
        geom = ViewGeom(axis, int(si), rows, cols, crop_box, scale, cw, ch)
        panel = _panel_from_slice(M.extract_slice(image, axis, int(si)), window,
                                  crop_box, scale, cw, ch)
        if mask is not None:
            panel = _overlay_mask(panel, M.extract_slice(mask, axis, int(si)),
                                  crop_box, scale, "outline")
        _draw_frame(panel, geom, spacing, f"{axis} {si}", letters=False,
                    scale_bar=(n_i == len(idxs) - 1), chip_pos="tl")
        r_i, c_i = divmod(n_i, ncols)
        px0, py0 = c_i * (cw + pad), r_i * (ch + pad)
        out.paste(panel, (px0, py0))
        rects.append((px0, py0, px0 + cw, py0 + ch))
        views.append(geom)
    return out, RenderInfo("sheet", views, [], [], window, panel_rects=rects)


def render_outline_change(image: np.ndarray, prev_mask: np.ndarray | None,
                          new_mask: np.ndarray, axis: str,
                          markset: M.MarkSet | None = None,
                          spacing=(1.0, 1.0, 1.0)) -> tuple[Image.Image, RenderInfo]:
    """Slice with the largest changed area, cropped: new mask green, removed red."""
    if axis not in M.AXIS_INDEX:
        raise ValueError(f"axis must be one of {M.AXES}, got {axis!r}")
    image = np.asarray(image)
    spacing = tuple(float(s) for s in spacing)
    new_mask = np.asarray(new_mask) > 0
    prev_mask = None if prev_mask is None else (np.asarray(prev_mask) > 0)
    diff = new_mask if prev_mask is None else (new_mask ^ prev_mask)
    other_axes = tuple(i for i in range(3) if i != M.AXIS_INDEX[axis])
    if diff.any():
        areas = diff.sum(axis=other_axes)
    else:
        areas = new_mask.sum(axis=other_axes)
    sl = int(np.argmax(areas)) if areas.any() else image.shape[M.AXIS_INDEX[axis]] // 2
    window = _window_for(image, markset)
    rows, cols = M.slice_shape(image.shape, axis)
    new2d = M.extract_slice(new_mask.astype(np.uint8), axis, sl).astype(bool)
    prev2d = (M.extract_slice(prev_mask.astype(np.uint8), axis, sl).astype(bool)
              if prev_mask is not None else None)
    show = new2d | (prev2d if prev2d is not None else np.zeros_like(new2d))
    crop_box = _resolve_crop("mask", rows, cols, show)
    scale, pw, ph = _fit_panel(crop_box)
    geom = ViewGeom(axis, sl, rows, cols, crop_box, scale, pw, ph)
    panel = _panel_from_slice(M.extract_slice(image, axis, sl), window, crop_box, scale, pw, ph)
    if prev2d is not None:
        removed = prev2d & ~new2d
        if removed.any():
            panel = _overlay_mask(panel, removed, crop_box, scale, "fill", REMOVED_COLOR)
    panel = _overlay_mask(panel, new2d, crop_box, scale, "outline")
    _draw_frame(panel, geom, spacing, f"{axis} {sl} | change", chip_pos="bl")
    return panel, RenderInfo("outline_change", [geom], [], [], window)


CHANGE_MARGIN_MM = 30.0      # 3-plane crop: max(30 mm, 50% of the extent) per side
CHANGE_MARGIN_FRAC = 0.5
CHANGE_MAX_SCALE = 12.0      # px per voxel cap for tiny masks


def _overlay_outline_px(im: Image.Image, mask2d: np.ndarray, crop_box, color,
                        width: int = 2, tint: float = 0.0) -> Image.Image:
    """Thin (``width`` px) outline of a 2D mask at PANEL resolution (nearest upscale),
    optionally with a faint fill; the crop border itself never gets an outline."""
    if mask2d is None or not np.asarray(mask2d).any():
        return im
    r0, c0, r1, c1 = crop_box
    sub = np.asarray(mask2d)[r0:r1, c0:c1].astype(bool)
    if not sub.any():
        return im
    up = np.asarray(Image.fromarray((sub * 255).astype(np.uint8), mode="L")
                    .resize(im.size, Image.Resampling.NEAREST)) > 127
    inner = ndi.binary_erosion(up, structure=np.ones((3, 3), bool), iterations=width,
                               border_value=1)
    ring = up & ~inner
    arr = np.asarray(im).astype(np.float32)
    if tint > 0:
        arr[up] = (1 - tint) * arr[up] + tint * np.asarray(color, np.float32)
    arr[ring] = np.asarray(color, np.float32)
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def _display_interval(lo: int, hi: int, n: int, flipped: bool) -> tuple[int, int]:
    """Half-open voxel interval on one axis -> half-open display interval."""
    return (n - hi, n - lo) if flipped else (lo, hi)


def render_change_3plane(image: np.ndarray, prev_mask: np.ndarray | None,
                         new_mask: np.ndarray, markset: M.MarkSet | None = None,
                         spacing=(1.0, 1.0, 1.0)) -> tuple[Image.Image, RenderInfo]:
    """Axial + coronal + sagittal feedback after a mask-changing op.

    The three slices pass through the new mask's centre of mass (empty new mask:
    the centre of the changed region, else the volume centre).  Each panel is
    cropped to the 3D bbox of the new mask (changed region when the mask is
    empty) grown by ``max(30 mm, 50% of its extent)`` per side and clamped to the
    volume; all panels share one scale so sizes are comparable, each panel's long
    side is <= ``PANEL_LONG_PX`` and the image is <= ``TOTAL_WIDTH_PX`` wide.  New
    mask: green outline; removed region (previous minus new): red outline + tint.
    ``RenderInfo.views`` = [axial, coronal, sagittal] geometries; ``note`` says
    what the slices pass through.
    """
    image = np.asarray(image)
    spacing = tuple(float(v) for v in spacing)
    new_mask = np.asarray(new_mask) > 0
    prev_mask = None if prev_mask is None else (np.asarray(prev_mask) > 0)
    removed = None if prev_mask is None else (prev_mask & ~new_mask)
    diff = new_mask if prev_mask is None else (new_mask ^ prev_mask)
    if new_mask.any():
        focus, basis = new_mask, "mask centre"
    elif diff.any():
        focus, basis = diff, "changed region centre"
    else:
        focus, basis = None, "volume centre"
    shape = image.shape
    if focus is not None:
        idx = np.nonzero(focus)
        centre = tuple(int(np.clip(np.floor(a.mean() + 0.5), 0, n - 1)) for a, n in zip(idx, shape))
        lo = [int(a.min()) for a in idx]
        hi = [int(a.max()) + 1 for a in idx]
        for a in range(3):
            ext_mm = (hi[a] - lo[a]) * spacing[a]
            m_vox = int(np.ceil(max(CHANGE_MARGIN_MM, CHANGE_MARGIN_FRAC * ext_mm) / spacing[a]))
            lo[a], hi[a] = max(0, lo[a] - m_vox), min(shape[a], hi[a] + m_vox)
    else:
        centre = tuple(d // 2 for d in shape)
        lo, hi = [0, 0, 0], list(shape)

    window = _window_for(image, markset)
    plan = []
    for axis, sl_idx in (("axial", centre[2]), ("coronal", centre[1]), ("sagittal", centre[0])):
        ra, rflip, ca, cflip = M._INPLANE[axis]
        rows, cols = M.slice_shape(shape, axis)
        r_lo, r_hi = _display_interval(lo[ra], hi[ra], shape[ra], rflip)
        c_lo, c_hi = _display_interval(lo[ca], hi[ca], shape[ca], cflip)
        plan.append((axis, sl_idx, rows, cols, (r_lo, c_lo, r_hi, c_hi)))
    long_side = max(max(b[2] - b[0], b[3] - b[1]) for *_x, b in plan)
    sum_w = sum(b[3] - b[1] for *_x, b in plan)
    scale = min(PANEL_LONG_PX / long_side,
                (TOTAL_WIDTH_PX - 2 * PANEL_GAP_PX) / sum_w, CHANGE_MAX_SCALE)

    views: list[ViewGeom] = []
    panels: list[Image.Image] = []
    for axis, sl_idx, rows, cols, box in plan:
        pw = max(1, int(np.floor((box[3] - box[1]) * scale)))
        ph = max(1, int(np.floor((box[2] - box[0]) * scale)))
        geom = ViewGeom(axis, sl_idx, rows, cols, box, scale, pw, ph)
        panel = _panel_from_slice(M.extract_slice(image, axis, sl_idx), window, box, scale, pw, ph)
        if removed is not None and removed.any():
            panel = _overlay_outline_px(panel, M.extract_slice(removed, axis, sl_idx), box,
                                        REMOVED_COLOR, tint=0.25)
        panel = _overlay_outline_px(panel, M.extract_slice(new_mask, axis, sl_idx), box,
                                    OUTLINE_COLOR)
        _draw_frame(panel, geom, spacing, f"{axis} {sl_idx}")
        views.append(geom)
        panels.append(panel)
    return _combine(panels), RenderInfo("change3", views, [], [], window, note=basis)


REVIEW_FRACTIONS = (0.25, 0.5, 0.75)
REVIEW_PAST_MM = 8.0
REVIEW_MARGIN_MM = 20.0
REVIEW_MARGIN_FRAC = 0.3
REVIEW_GAP_PX = 8
REVIEW_MAX_SCALE = 12.0
REVIEW_CELL_MAX_PX = 268      # 5 cells + gaps ~1.4k px wide, 3 rows ~0.8k high: ~1.1 MP
REVIEW_PAST_COLOR = (255, 140, 0)  # orange frame/chip of the past-the-mask cells
# patient direction of decreasing / increasing voxel index per axis (CONTRACTS §1: +i Right, +j Anterior, +k Superior)
_REVIEW_PAST_WORDS = {
    "axial": ("below", "above"),
    "coronal": ("posterior to", "anterior to"),
    "sagittal": ("left of", "right of"),
}


def _draw_dashed_frame(panel: Image.Image, color, width: int = 3, dash: int = 10) -> None:
    draw = ImageDraw.Draw(panel)
    w, h = panel.size
    for x in range(0, w, 2 * dash):
        for y0 in (0, h - width):
            draw.rectangle((x, y0, min(x + dash, w) - 1, y0 + width - 1), fill=color)
    for y in range(0, h, 2 * dash):
        for x0 in (0, w - width):
            draw.rectangle((x0, y, x0 + width - 1, min(y + dash, h) - 1), fill=color)


def _draw_past_chip(panel: Image.Image, head: str, past: str) -> None:
    """Orange chip at the top-left: slice id, then the past-the-mask text word-wrapped to the panel width."""
    d = ImageDraw.Draw(panel)
    f = _font(SMALL_FONT_PX)
    max_w = panel.width - 14
    lines, cur = [head + " \u00b7"], ""
    for word in past.split():
        trial = f"{cur} {word}".strip()
        if cur and d.textlength(trial, font=f) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    lines.append(cur)
    lh = SMALL_FONT_PX + 4
    w = int(max(d.textlength(t, font=f) for t in lines)) + 8
    d.rectangle((4, 4, 4 + w, 4 + lh * len(lines) + 4), fill=_BLACK)
    for i, t in enumerate(lines):
        d.text((8, 7 + i * lh), t, font=f, fill=REVIEW_PAST_COLOR)


def _review_slices(axis: str, lo: int, hi: int, n: int, step_mm: float) -> list[tuple[int, str | None]]:
    """(slice, past-label or None) for the five review columns of one row.  ``lo``/``hi``
    are the mask's first / one-past-last slice on the row axis, ``n`` the volume size."""
    first, last = lo, hi - 1
    k = int(np.ceil(REVIEW_PAST_MM / step_mm))
    low_word, high_word = _REVIEW_PAST_WORDS[axis]
    before = max(0, first - k)
    after = min(n - 1, last + k)
    cells: list[tuple[int, str | None]] = [
        (before, _past_label(before, first, step_mm, low_word, first - k < 0))]
    for frac in REVIEW_FRACTIONS:
        cells.append((int(round(first + frac * (last - first))), None))
    cells.append((after, _past_label(after, last, step_mm, high_word, last + k > n - 1)))
    return cells


def _past_label(sl: int, end: int, step_mm: float, word: str, clamped: bool) -> str:
    if sl == end:
        return "volume edge, mask reaches it"
    text = f"{abs(sl - end) * step_mm:.0f} mm {word} mask"
    return text + (" (volume edge)" if clamped else "")


def render_review_grid(image: np.ndarray, mask: np.ndarray, spacing=(1.0, 1.0, 1.0),
                       markset: M.MarkSet | None = None) -> tuple[Image.Image, RenderInfo]:
    """3x5 final-review grid: rows axial/coronal/sagittal; columns are the slice
    8 mm before the mask's first slice on the row axis (orange dashed frame, chip
    'N mm below|posterior to|left of mask'), 25/50/75% of the mask extent (green
    outline), and the slice 8 mm after the last mask slice (orange).  Past-the-mask
    slices are clamped to the volume; a cell clamped onto the mask's own end slice
    says 'volume edge'.  Every cell is cropped to the mask's in-plane bbox grown by
    ``max(20 mm, 30% of the extent)``; crop and scale are shared by the five cells
    of a row.  Width and height are <= ``TOTAL_WIDTH_PX`` (~1.1 MP).  An empty mask
    falls back to the overview (three planes through the volume centre)."""
    image = np.asarray(image)
    mask = np.asarray(mask) > 0
    spacing = tuple(float(v) for v in spacing)
    if not mask.any():
        return render_overview(image, spacing, mask=None, marks=markset)
    shape = image.shape
    idx = np.nonzero(mask)
    lo = [int(a.min()) for a in idx]
    hi = [int(a.max()) + 1 for a in idx]
    window = _window_for(image, markset)
    cell_max = REVIEW_CELL_MAX_PX
    views: list[ViewGeom] = []
    row_imgs: list[Image.Image] = []
    for axis in ("axial", "coronal", "sagittal"):
        ra, rflip, ca, cflip = M._INPLANE[axis]
        fixed = M.AXIS_INDEX[axis]
        rows, cols = M.slice_shape(shape, axis)
        box_vox = {}
        for a in (ra, ca):
            ext_mm = (hi[a] - lo[a]) * spacing[a]
            m_vox = int(np.ceil(max(REVIEW_MARGIN_MM, REVIEW_MARGIN_FRAC * ext_mm) / spacing[a]))
            box_vox[a] = (max(0, lo[a] - m_vox), min(shape[a], hi[a] + m_vox))
        r_lo, r_hi = _display_interval(*box_vox[ra], shape[ra], rflip)
        c_lo, c_hi = _display_interval(*box_vox[ca], shape[ca], cflip)
        box = (r_lo, c_lo, r_hi, c_hi)
        scale = min(cell_max / (c_hi - c_lo), cell_max / (r_hi - r_lo), REVIEW_MAX_SCALE)
        pw = max(1, int(np.floor((c_hi - c_lo) * scale)))
        ph = max(1, int(np.floor((r_hi - r_lo) * scale)))
        panels = []
        for sl, past in _review_slices(axis, lo[fixed], hi[fixed], shape[fixed], spacing[fixed]):
            geom = ViewGeom(axis, sl, rows, cols, box, scale, pw, ph)
            panel = _panel_from_slice(M.extract_slice(image, axis, sl), window, box, scale, pw, ph)
            panel = _overlay_outline_px(panel, M.extract_slice(mask, axis, sl), box, OUTLINE_COLOR)
            _draw_frame(panel, geom, spacing, f"{axis[:3]} {sl}", scale_bar=False)  # short chip: narrow cells
            if past is not None:
                _draw_dashed_frame(panel, REVIEW_PAST_COLOR)
                _draw_past_chip(panel, f"{axis[:3]} {sl}", past)
            views.append(geom)
            panels.append(panel)
        row = Image.new("RGB", (5 * pw + 4 * REVIEW_GAP_PX, ph), (24, 24, 24))
        for i, p in enumerate(panels):
            row.paste(p, (i * (pw + REVIEW_GAP_PX), 0))
        row_imgs.append(row)
    w = max(r.width for r in row_imgs)
    h = sum(r.height for r in row_imgs) + REVIEW_GAP_PX * 2
    out = Image.new("RGB", (w, h), (24, 24, 24))
    y = 0
    for r in row_imgs:
        out.paste(r, ((w - r.width) // 2, y))
        y += r.height + REVIEW_GAP_PX
    return out, RenderInfo("review", views, [], [], window)


REVIEW_COMPONENT_MAX = 5     # components per component-review sheet (largest first;
                             # more sheets are returned in batches so every current-mask
                             # component is represented in the finish review)


def render_review_components(image: np.ndarray, mask: np.ndarray,
                             spacing=(1.0, 1.0, 1.0),
                             markset: M.MarkSet | None = None,
                             max_components: int = REVIEW_COMPONENT_MAX,
                             start: int = 0,
                             ) -> tuple[Image.Image, RenderInfo]:
    """Per-component review sheet for multi-lesion targets (Phase 4, non-MSTT datasets).

    The union-extent review grid (render_review_grid) inspects the union of all lesions;
    with several separated lesions its 25/50/75% columns can fall between them and each
    lesion's own boundary is never seen. This sheet adds what the grid cannot show: one
    row per connected component of ``mask`` (largest first), columns axial/coronal/sagittal
    through the component's centre of mass, every cell cropped to THAT component's bbox
    grown by ``max(20 mm, 30% of its extent)`` per side (one shared scale per row). Green
    outline; chip 'Ck ax 123' names the row's component rank (volumes are in the
    observation text). Components are rendered largest first; the sheet covers ranks
    ``start+1 .. start+max_components`` — callers batch through ALL components (one sheet
    per ``max_components`` ranks, as many sheets as needed) so no current-mask component
    is omitted. Derived from the current mask only.
    """
    image = np.asarray(image)
    mask = np.asarray(mask) > 0
    spacing = tuple(float(s) for s in spacing)
    labels, n = ndi.label(mask, structure=ndi.generate_binary_structure(3, 1))
    if n == 0:
        return render_overview(image, spacing, mask=None, marks=markset)
    shape = image.shape
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    order = [int(i) for i in np.argsort(sizes)[::-1] if i != 0]
    batch = order[start:start + max_components]
    if not batch:
        raise ValueError(f"component sheet batch is empty: {len(order)} components, "
                         f"start={start}, max_components={max_components}")
    window = _window_for(image, markset)
    views: list[ViewGeom] = []
    row_imgs: list[Image.Image] = []
    for rank, lab in enumerate(batch, start=start + 1):
        comp = labels == lab
        idx = np.nonzero(comp)
        centre = tuple(int(np.clip(np.floor(a.mean() + 0.5), 0, sn - 1))
                       for a, sn in zip(idx, shape))
        lo = [int(a.min()) for a in idx]
        hi = [int(a.max()) + 1 for a in idx]
        plan = []
        for axis, sl_idx in (("axial", centre[2]), ("coronal", centre[1]), ("sagittal", centre[0])):
            ra, rflip, ca, cflip = M._INPLANE[axis]
            rows, cols = M.slice_shape(shape, axis)
            box_vox = {}
            for a in (ra, ca):
                ext_mm = (hi[a] - lo[a]) * spacing[a]
                m_vox = int(np.ceil(max(REVIEW_MARGIN_MM, REVIEW_MARGIN_FRAC * ext_mm) / spacing[a]))
                box_vox[a] = (max(0, lo[a] - m_vox), min(shape[a], hi[a] + m_vox))
            r_lo, r_hi = _display_interval(*box_vox[ra], shape[ra], rflip)
            c_lo, c_hi = _display_interval(*box_vox[ca], shape[ca], cflip)
            plan.append((axis, sl_idx, rows, cols, (r_lo, c_lo, r_hi, c_hi)))
        long_side = max(max(b[2] - b[0], b[3] - b[1]) for *_x, b in plan)
        sum_w = sum(b[3] - b[1] for *_x, b in plan)
        scale = min(REVIEW_CELL_MAX_PX / long_side,
                    (TOTAL_WIDTH_PX - 2 * PANEL_GAP_PX) / sum_w, REVIEW_MAX_SCALE)
        panels = []
        for axis, sl_idx, rows, cols, box in plan:
            pw = max(1, int(np.floor((box[3] - box[1]) * scale)))
            ph = max(1, int(np.floor((box[2] - box[0]) * scale)))
            geom = ViewGeom(axis, sl_idx, rows, cols, box, scale, pw, ph)
            panel = _panel_from_slice(M.extract_slice(image, axis, sl_idx), window, box, scale, pw, ph)
            panel = _overlay_outline_px(panel, M.extract_slice(comp.astype(np.uint8), axis, sl_idx),
                                        box, OUTLINE_COLOR)
            _draw_frame(panel, geom, spacing, f"C{rank} {axis[:3]} {sl_idx}", scale_bar=False)
            views.append(geom)
            panels.append(panel)
        row = Image.new("RGB", (sum(p.width for p in panels) + 2 * REVIEW_GAP_PX,
                                max(p.height for p in panels)), (24, 24, 24))
        x = 0
        for p in panels:
            row.paste(p, (x, 0))
            x += p.width + REVIEW_GAP_PX
        row_imgs.append(row)
    w = max(r.width for r in row_imgs)
    h = sum(r.height for r in row_imgs) + REVIEW_GAP_PX * (len(row_imgs) - 1)
    out = Image.new("RGB", (w, h), (24, 24, 24))
    y = 0
    for r in row_imgs:
        out.paste(r, ((w - r.width) // 2, y))
        y += r.height + REVIEW_GAP_PX
    return out, RenderInfo("review_components", views, [], [], window)


CANDIDATE_COLOR = (255, 64, 255)     # magenta outline/tint of a candidate (green = mask, red = removed)
CANDIDATE_TINT = 0.2                 # <= 20% fill so the tissue inside the outline stays visible
CANDIDATE_MARGIN_MM = 20.0           # cell crop: bbox(mask U candidate) + this margin per side
CANDIDATE_GAP_PX = 8
CANDIDATE_MAX_SCALE = 8.0            # px per voxel cap (tiny candidates stay cell-sized)
CANDIDATE_MAX_MP = 1.1               # pixel budget of the whole image (<= ~1.2 MP with gaps)
CANDIDATE_LABEL_PX = 17


def _draw_candidate_label(panel: Image.Image, label: str, occupied: list[tuple]) -> tuple:
    """Candidate id chip ('C1'): magenta text on a black plate, placed in the top margin
    WITHOUT covering the frame furniture (slice chip, orientation letters) passed in
    ``occupied``: the first left-to-right x whose plate clears every occupied rectangle
    wins; on a panel too narrow for the top row, the free spot of the bottom row is tried;
    as a last resort the chip is clamped into the top row (right-aligned) so it is always
    drawn.  Returns the chip's plate rectangle."""
    d = ImageDraw.Draw(panel)
    f = _font(CANDIDATE_LABEL_PX)
    l0, t0, r0, b0 = d.textbbox((0, 0), label, font=f)   # text extent relative to the anchor

    def plate(x, y):
        return (x + l0 - 4, y + t0 - 3, x + r0 + 4, y + b0 + 3)

    w, h = panel.width, panel.height
    y_top = 4
    y_bot = h - 4 - b0                    # plate flush with the bottom margin
    best = None
    for y in (y_top, y_bot):
        x = 4
        while x + (r0 - l0) + 8 <= w and any(_boxes_overlap(plate(x, y), ob) for ob in occupied):
            x += 4
        if x + (r0 - l0) + 8 <= w and not any(_boxes_overlap(plate(x, y), ob) for ob in occupied):
            best = (x, y)
            break
    if best is None:                      # nothing fits without a collision: clamp top-right
        best = (max(4, w - (r0 - l0) - 8), y_top)
    x, y = best
    l, t, r, b = plate(x, y)
    d.rectangle((l, t, r, b), fill=_BLACK, outline=CANDIDATE_COLOR, width=2)
    d.text((x, y), label, font=f, fill=CANDIDATE_COLOR)
    return (l, t, r, b)


def render_candidates(image: np.ndarray, mask: np.ndarray, candidates,
                      spacing=(1.0, 1.0, 1.0), markset: M.MarkSet | None = None
                      ) -> tuple[Image.Image, RenderInfo]:
    """Candidate-correction sheet: one row per candidate (max 5), columns axial / coronal /
    sagittal through the candidate's centre of mass (moved to the nearest slice that still
    contains candidate voxels).  Every cell is cropped to the bbox of (mask U candidate) on
    that plane + ``CANDIDATE_MARGIN_MM``; the current mask is a green outline, the candidate
    a magenta outline with a ``CANDIDATE_TINT`` fill and its id chip next to the slice chip.
    Cells shrink with the number of rows so the image stays <= ``TOTAL_WIDTH_PX`` per side
    and <= ~1.2 MP.  ``candidates`` need ``id``/``bbox``/``crop`` (segd.candidates.Candidate)."""
    image = np.asarray(image)
    mask = np.asarray(mask) > 0
    spacing = tuple(float(v) for v in spacing)
    cands = list(candidates)[:5]
    if not cands:
        raise ValueError("render_candidates needs at least one candidate")
    n = len(cands)
    gap = CANDIDATE_GAP_PX
    cell = int(min((TOTAL_WIDTH_PX - 2 * gap) // 3,                       # 3 columns per row
                   (TOTAL_WIDTH_PX - (n - 1) * gap) // n,                 # n rows high
                   np.sqrt(CANDIDATE_MAX_MP * 1e6 / (3 * n))))            # total pixel budget
    shape = image.shape
    window = _window_for(image, markset)
    midx = np.nonzero(mask)
    mlo = [int(a.min()) for a in midx] if midx[0].size else None
    mhi = [int(a.max()) + 1 for a in midx] if midx[0].size else None
    views: list[ViewGeom] = []
    row_imgs: list[Image.Image] = []
    row_rects: list[tuple[int, int, int, int]] = []
    y = 0
    for cand in cands:
        region = np.zeros(shape, dtype=bool)
        region[cand.bbox] = cand.crop
        cidx = np.nonzero(cand.crop)
        com = [float(a.mean()) + s.start for a, s in zip(cidx, cand.bbox)]   # candidate centre of mass
        lo = [s.start for s in cand.bbox]
        hi = [s.stop for s in cand.bbox]
        if mlo is not None:                    # crop = bbox(mask U candidate)
            lo = [min(a, b) for a, b in zip(lo, mlo)]
            hi = [max(a, b) for a, b in zip(hi, mhi)]
        panels: list[Image.Image] = []
        for axis in ("axial", "coronal", "sagittal"):
            ra, rflip, ca, cflip = M._INPLANE[axis]
            fixed = M.AXIS_INDEX[axis]
            rows, cols = M.slice_shape(shape, axis)
            sl = int(np.clip(round(com[fixed]), 0, shape[fixed] - 1))
            has = np.nonzero(cand.crop.any(axis=tuple(i for i in range(3) if i != fixed)))[0] \
                + cand.bbox[fixed].start
            if sl not in has:                  # the COM of a non-convex candidate can miss it
                sl = int(has[np.argmin(np.abs(has - sl))])
            box_vox = {}
            for a in (ra, ca):
                m_vox = int(np.ceil(CANDIDATE_MARGIN_MM / spacing[a]))
                box_vox[a] = (max(0, lo[a] - m_vox), min(shape[a], hi[a] + m_vox))
            r_lo, r_hi = _display_interval(*box_vox[ra], shape[ra], rflip)
            c_lo, c_hi = _display_interval(*box_vox[ca], shape[ca], cflip)
            box = (r_lo, c_lo, r_hi, c_hi)
            scale = min(cell / (c_hi - c_lo), cell / (r_hi - r_lo), CANDIDATE_MAX_SCALE)
            pw = max(1, int(np.floor((c_hi - c_lo) * scale)))
            ph = max(1, int(np.floor((r_hi - r_lo) * scale)))
            geom = ViewGeom(axis, sl, rows, cols, box, scale, pw, ph)
            panel = _panel_from_slice(M.extract_slice(image, axis, sl), window, box, scale, pw, ph)
            panel = _overlay_outline_px(panel, M.extract_slice(region, axis, sl), box,
                                        CANDIDATE_COLOR, tint=CANDIDATE_TINT)
            panel = _overlay_outline_px(panel, M.extract_slice(mask, axis, sl), box, OUTLINE_COLOR)
            occupied = _draw_frame(panel, geom, spacing, f"{axis[:3]} {sl}", scale_bar=False)
            _draw_candidate_label(panel, cand.id, occupied)   # collision-aware: clears chip + letters
            views.append(geom)
            panels.append(panel)
        row_h = max(p.height for p in panels)
        row = Image.new("RGB", (3 * cell + 2 * gap, row_h), (24, 24, 24))
        for i, p in enumerate(panels):
            row.paste(p, (i * (cell + gap) + (cell - p.width) // 2, (row_h - p.height) // 2))
        row_imgs.append(row)
        row_rects.append((0, y, 3 * cell + 2 * gap, y + row_h))
        y += row_h + gap
    h = sum(r.height for r in row_imgs) + gap * (n - 1)
    out = Image.new("RGB", (3 * cell + 2 * gap, h), (24, 24, 24))
    y = 0
    for r in row_imgs:
        out.paste(r, (0, y))
        y += r.height + gap
    return out, RenderInfo("candidates", views, [], [], window, panel_rects=row_rects)


RAW_MARGIN_TOP_PX = 22       # raw-pointing rulers/title live in margins around the image
RAW_MARGIN_LEFT_PX = 40
RAW_MARGIN_BOTTOM_PX = 26


def render_raw_pointing(image: np.ndarray, axis: str, slice_idx: int, crop="full",
                        markset: M.MarkSet | None = None,
                        spacing=(1.0, 1.0, 1.0)) -> tuple[Image.Image, RenderInfo]:
    """Image with pixel rulers in a margin OUTSIDE the image area (raw-pointing study).

    The returned picture is the image area (``geom.panel_w`` x ``geom.panel_h``)
    pasted at (``RAW_MARGIN_LEFT_PX``, ``RAW_MARGIN_TOP_PX``) on a dark canvas;
    nothing is drawn over the image itself.  Ruler ticks every 50 px along the top
    and left margins are numbered in IMAGE-AREA pixels (0 = the image's top-left
    corner), the coordinates the model answers in; ``panel_to_voxel`` maps those
    back, so ``ViewGeom`` (panel_w/panel_h/scale/crop) describes the image area
    only.  The scale bar and title sit in the bottom margin.
    """
    if axis not in M.AXIS_INDEX:
        raise ValueError(f"axis must be one of {M.AXES}, got {axis!r}")
    image = np.asarray(image)
    spacing = tuple(float(s) for s in spacing)
    slice_idx = int(slice_idx)
    rows, cols = M.slice_shape(image.shape, axis)
    window = _window_for(image, markset)
    slice2d = M.extract_slice(image, axis, slice_idx)
    crop_box = _resolve_crop(crop, rows, cols, None)
    scale, pw, ph = _fit_panel(crop_box)
    geom = ViewGeom(axis, slice_idx, rows, cols, crop_box, scale, pw, ph)
    img_area = _panel_from_slice(slice2d, window, crop_box, scale, pw, ph)

    L, T, B = RAW_MARGIN_LEFT_PX, RAW_MARGIN_TOP_PX, RAW_MARGIN_BOTTOM_PX
    canvas = Image.new("RGB", (pw + L, ph + T + B), (24, 24, 24))
    canvas.paste(img_area, (L, T))
    draw = ImageDraw.Draw(canvas)
    f = _font(SMALL_FONT_PX)
    for x in range(0, pw, 50):                       # top ruler: tick down from the label row
        draw.line((L + x, T - 7, L + x, T - 1), fill=_WHITE, width=2)
        txt = str(x)
        tw = draw.textlength(txt, font=f)
        tx = min(max(L + x - tw / 2, 1), pw + L - tw - 1)
        draw.text((tx, 2), txt, font=f, fill=_WHITE)
    for y in range(0, ph, 50):                       # left ruler: right-aligned labels
        draw.line((L - 7, T + y, L - 1, T + y), fill=_WHITE, width=2)
        txt = str(y)
        tw = draw.textlength(txt, font=f)
        draw.text((L - 9 - tw, T + y - 6), txt, font=f, fill=_WHITE)
    ra, ca = M._INPLANE[axis][0], M._INPLANE[axis][2]
    mm_per_vox = float(np.mean((spacing[ra], spacing[ca])))
    bar_px = min(10.0 / mm_per_vox * scale, pw / 2.0)
    yb = T + ph + B // 2
    draw.line((L + 2, yb, L + 2 + bar_px, yb), fill=_WHITE, width=3)
    draw.line((L + 2, yb - 4, L + 2, yb + 4), fill=_WHITE, width=2)
    draw.line((L + 2 + bar_px, yb - 4, L + 2 + bar_px, yb + 4), fill=_WHITE, width=2)
    draw.text((L + 8 + bar_px, yb), "10 mm", font=f, fill=_WHITE, anchor="lm")
    title = f"{axis} {slice_idx} | image px: x right, y down"
    tw = draw.textlength(title, font=f)
    draw.text((max(L + pw - tw - 4, L + 8 + bar_px + 50), yb), title, font=f, fill=_WHITE,
              anchor="lm")
    return canvas, RenderInfo("raw", [geom], [], [], window)


PROPOSED_COLOR = (56, 189, 248)    # bright sky blue; distinct from marks/mask/candidates


def draw_proposed_points(view_img, panel_rects, points) -> Image.Image:
    """seg_locate feedback: draw proposed points (panel-0 pixel coordinates) with
    numbered P labels on every panel of the view image (clean + marked share one
    geometry, so the same panel-relative position is valid on both).

    ``panel_rects`` are the RenderInfo paste rects (panel 0 first); a single-panel
    image passes ``[(0, 0, w, h)]``. ``points``: [(x, y, label)]. Returns a COPY; the
    input image is never modified.
    """
    out = view_img.copy()
    draw = ImageDraw.Draw(out)
    for x, y, label in points:
        for x0, y0, _x1, _y1 in panel_rects:
            px, py = x0 + x, y0 + y
            draw.ellipse((px - 7, py - 7, px + 7, py + 7), outline=_BLACK, width=4)
            draw.ellipse((px - 7, py - 7, px + 7, py + 7), outline=PROPOSED_COLOR, width=2)
            placed = False
            for ax, ay in ((px + 9, py - 24), (px + 9, py + 8), (px - 30, py - 24)):
                if 2 <= ax and ax + 24 < out.width - 2 and 2 <= ay and ay + 16 < out.height - 2:
                    _chip(draw, (ax, ay), label)
                    placed = True
                    break
            if not placed:
                _chip(draw, (max(2, px - 30), max(2, py - 24)), label)
    return out
