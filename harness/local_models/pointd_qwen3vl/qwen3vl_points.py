"""Pure Qwen3-VL grounding-output parser (no torch): text -> pixel points.

Implements the official Qwen3-VL grounding conventions (HF card +
QwenLM/Qwen3-VL cookbook `2d_grounding.ipynb`):

- Coordinate system: RELATIVE coordinates 0-1000 ("Qwen3-VL's default coordinate
  system has been changed from the absolute coordinates used in Qwen2.5-VL to
  relative coordinates ranging from 0 to 1000"). Pixels: x_px = x/1000*image_w,
  y_px = y/1000*image_h (exactly the cookbook conversion).
- Point format: a JSON array of {"point_2d": [x, y], "label": ...} objects
  (cookbook point-grounding examples; first coordinate is x).
- Box format (tolerated fallback): {"bbox_2d": [x1, y1, x2, y2], "label": ...} —
  reduced to its centre ((x1+x2)/2, (y1+y2)/2) before conversion, so boxes and
  points are interchangeable downstream.
- Tolerances: ```json code fences are stripped; a single bare JSON object is
  accepted (wrapped); one malformed item is skipped (not fatal); non-finite or
  out-of-range (>1000) coordinates are dropped (mirrors the official
  in-image behaviour of keeping only valid coordinates).

No torch import; used by the server and the offline tests.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


@dataclass
class Qwen3VLPoint:
    x: float               # pixel x in [0, image_w]
    y: float               # pixel y in [0, image_h]
    label: str | None      # model label for the point ("" / missing -> None)
    rank: int              # emission order, 0-based


def _json_candidates(raw_text: str) -> list:
    """JSON arrays parsed from the text: ```json fences first, else the whole text;
    a single bare JSON object is wrapped into a one-element list."""
    chunks = [m.group(1) for m in _FENCE_RE.finditer(raw_text)] or [raw_text]
    out: list = []
    for chunk in chunks:
        start, end = chunk.find("["), chunk.rfind("]")
        if start != -1 and end > start:
            try:
                value = json.loads(chunk[start:end + 1])
            except json.JSONDecodeError:
                continue
            if isinstance(value, list):
                out.extend(v for v in value if isinstance(v, dict))
        obj_start, obj_end = chunk.find("{"), chunk.rfind("}")
        if not out and obj_start != -1 and obj_end > obj_start:
            try:
                value = json.loads(chunk[obj_start:obj_end + 1])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                out.append(value)
    return out


def _finite(v) -> float | None:
    """Finite number (bools excluded) or None."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def _xy_from_item(item: dict) -> tuple[float, float] | None:
    """(x, y) on the 0-1000 scale from a point_2d item, or its bbox_2d centre."""
    if isinstance(item.get("point_2d"), (list, tuple)) and len(item["point_2d"]) >= 2:
        x, y = _finite(item["point_2d"][0]), _finite(item["point_2d"][1])
    elif isinstance(item.get("bbox_2d"), (list, tuple)) and len(item["bbox_2d"]) >= 4:
        x1, y1, x2, y2 = (_finite(v) for v in item["bbox_2d"][:4])
        if None in (x1, y1, x2, y2):
            return None
        x, y = (x1 + x2) / 2.0, (y1 + y2) / 2.0  # centre; x1>x2 swaps are irrelevant
    else:
        return None
    if x is None or y is None:
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        return None  # out of the relative coordinate range: not a valid point
    return x, y


def parse_qwen3vl_points(raw_text: str | None, image_w: float, image_h: float) -> list[Qwen3VLPoint]:
    """All points in the raw text, in emission order, mapped from the 0-1000
    relative scale to input-image pixels. Returns [] for empty/non-JSON text."""
    points: list[Qwen3VLPoint] = []
    if not raw_text:
        return points
    for item in _json_candidates(raw_text):
        xy = _xy_from_item(item)
        if xy is None:
            continue
        x_norm, y_norm = xy
        label = item.get("label")
        points.append(Qwen3VLPoint(
            x=x_norm / 1000.0 * float(image_w),
            y=y_norm / 1000.0 * float(image_h),
            label=label if isinstance(label, str) and label else None,
            rank=len(points)))
    return points
