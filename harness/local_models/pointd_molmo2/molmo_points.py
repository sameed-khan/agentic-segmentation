"""Pure Molmo2 point-output parser (no torch): text -> pixel points.

Implements the official model-card parsing convention (allenai/Molmo2-8B README,
COORD_REGEX/POINTS_REGEX/_points_from_num_str), adapted to single-image pointing:

- Point spans look like ``<points coords="...">label</points>`` / ``<tracks .../>``,
  optionally inside code fences; the official COORD_REGEX captures the coords string
  of every <points>/<tracks> tag.
- Official grammar (coords payload): space-separated ``id x y`` triplets, x/y 3-4
  digit integers on a 0-1000 scale (official POINTS_REGEX). As a documented tolerance
  for the less-documented single-image grammar, bare ``x y`` pairs are accepted when
  no triplet matches anywhere in the payload.
- Coordinates map to pixels via x_px = x/1000*image_w, y_px = y/1000*image_h.
- In-image filter mirrors the official check (0 <= x <= image_w, 0 <= y <= image_h);
  `clamp=True` moves out-of-range points onto the border instead (not default).
- Rank = emission order in the raw text. Molmo2 point ids (counting/tracking) are
  NOT ranks, and Molmo2 has no calibrated scores — the service reports score=None.

Validate this parser against actual Molmo2 output on the A100 before any study run
(single-image grammar is less documented than the video one — research note).

No torch import; used by the server and the offline tests.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Official model-card regexes: coords payload of any <points>/<tracks ...> tag, then
# "id x y" with x/y 3-4 digits (POINTS_REGEX), single-space separated.
COORD_REGEX = re.compile(r"<(?:points|tracks)[^>]*? coords=\"([0-9\t:;, .]+)\"/?>")
TRIPLET_REGEX = re.compile(r"(\d+) (\d{3,4}) (\d{3,4})")
# Single-image tolerance: bare "x y" pairs (1-4 digit) when no triplet is present.
PAIR_REGEX = re.compile(r"(\d{1,4}) (\d{1,4})")


@dataclass
class MolmoPoint:
    x: float              # pixel x in [0, image_w]
    y: float              # pixel y in [0, image_h]
    point_id: int | None  # Molmo2 point id when present (NOT a rank)
    rank: int             # emission order, 0-based


def _norm_to_px(value: float, size: float) -> float:
    return value / 1000.0 * float(size)


def _entries_from_payload(payload: str) -> list[tuple[int | None, float, float]]:
    """(point_id | None, x_norm, y_norm) entries of one coords payload, in order."""
    triplets = [(int(pid), float(xs), float(ys))
                for pid, xs, ys in TRIPLET_REGEX.findall(payload)]
    if triplets:
        return triplets
    return [(None, float(xs), float(ys)) for xs, ys in PAIR_REGEX.findall(payload)]


def parse_molmo_points(raw_text: str | None, image_w: float, image_h: float,
                       clamp: bool = False) -> list[MolmoPoint]:
    """All points in the raw text, in emission order, mapped to pixels.

    `clamp=False` mirrors the official in-image filter (out-of-range entries are
    dropped, never silently moved). Returns [] for empty/point-less text.
    """
    points: list[MolmoPoint] = []
    if not raw_text:
        return points
    spans = [m.group(1) for m in COORD_REGEX.finditer(raw_text)]
    # Tolerate models that emit bare coordinates without the <points> wrapper.
    payloads = spans if spans else ([raw_text] if TRIPLET_REGEX.search(raw_text) or
                                    PAIR_REGEX.search(raw_text) else [])
    for payload in payloads:
        for pid, x_norm, y_norm in _entries_from_payload(payload):
            x, y = _norm_to_px(x_norm, image_w), _norm_to_px(y_norm, image_h)
            if clamp:
                x = min(max(x, 0.0), float(image_w))
                y = min(max(y, 0.0), float(image_h))
            elif not (0 <= x <= image_w and 0 <= y <= image_h):
                continue  # official: drop out-of-image points
            points.append(MolmoPoint(x=x, y=y, point_id=pid, rank=len(points)))
    return points
