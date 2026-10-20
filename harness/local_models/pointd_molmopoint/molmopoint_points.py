"""Pure MolmoPoint grounding-token decoder (no torch): token text -> pixel points.

MolmoPoint-8B does not emit text coordinates. A point is generated as a series of
special <POINT_N> grounding tokens: one ViT-patch token, one subpatch token and one
3x3 location token, followed by the object id as plain digits. Decoding on the real
service path uses the OFFICIAL model method (`model.extract_image_points`, per the
allenai/MolmoPoint-8B card) which needs inference-time preprocessing metadata
(token pooling + subpatch mapping). This module mirrors that official decode —
`EXTRACT_POINT_TRIPLE`, `get_subpatch_ids` and `extract_image_points` from the
snapshot's modeling_molmo_point.py, exact same regex and arithmetic — for the
--fake backend and the offline tests (numpy only, torch never imported):

- Triple: ``<POINT_a> ?<POINT_b> ?<POINT_c> ?<digits>`` (a=patch row, b=subpatch
  number, c=location number, digits=object id) — official EXTRACT_POINT_TRIPLE.
- Token numbering (official get_subpatch_ids): n_patches = pooling rows
  (+1 when the no_more_points class exists), subpatch_id = b - n_patches,
  location_id = c - n_patches - n_subpatches.
- Patch row -> ViT patch id via the pooling matrix; the grid cell whose mapping
  equals that id is the coarse point; the 3x3 location refines it by
  +(loc_x+0.5)*0.33, +(loc_y+0.5)*0.33 patch units; scaled to original-image
  pixels by (p_x / mapping_w) * image_w, (p_y / mapping_h) * image_h
  (official extract_image_points).

Validate against real MolmoPoint output on the A100 before a study run (the fake
geometry is synthetic; the real path calls the official method directly).

No torch import; used by the --fake backend and the offline tests.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# Official modeling_molmo_point.EXTRACT_POINT_TRIPLE (verbatim).
EXTRACT_POINT_TRIPLE = re.compile(r"<POINT_(\d+)> ?<POINT_(\d+)> ?<POINT_(\d+)> ?([0-9]+)")


@dataclass
class MolmoGroundingPoint:
    x: float              # pixel x in the original image
    y: float              # pixel y in the original image
    image_ix: int         # index of the image the point falls in (single-image service: 0)
    point_id: int         # object id (example_id) from the token stream, NOT a rank
    rank: int             # emission order, 0-based


def parse_point_triples(raw_text: str | None) -> list[tuple[int, int, int, int]]:
    """(patch_row, subpatch_num, location_num, object_id) triples, in emission order."""
    if not raw_text:
        return []
    return [(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
            for m in EXTRACT_POINT_TRIPLE.finditer(raw_text)]


def decode_image_points(raw_text: str | None, pooling, mappings, image_sizes,
                        no_more_points_class: bool = True,
                        patch_location: str | None = "3x3") -> list[MolmoGroundingPoint]:
    """Official token-text decode (get_subpatch_ids + extract_image_points math).

    `pooling` [n_patch_rows, n_subpatches] maps a generated patch token to the ViT
    patch id; `mappings` is one 2-D grid of ViT patch ids per image; `image_sizes`
    is one (w, h) per image (original pixels). Out-of-grid patch ids are skipped
    (official logs "Invalid patch id encountered" and continues).
    """
    if len(mappings) != len(image_sizes):
        raise ValueError("Mapping and image sizes must have the same length")
    pooling = np.asarray(pooling)
    n_patches, n_subpatches = pooling.shape[-2:]
    if no_more_points_class:
        n_patches += 1
    n_locations = 9 if patch_location else 0

    points: list[MolmoGroundingPoint] = []
    for patch_row, subpatch_num, location_num, object_id in parse_point_triples(raw_text):
        subpatch_id = subpatch_num - n_patches
        location_id = location_num - n_patches - n_subpatches
        vit_patch_id = pooling[patch_row, subpatch_id]
        for image_ix, (mapping, (w, h)) in enumerate(zip(mappings, image_sizes)):
            mapping = np.asarray(mapping)
            patch_coords = np.argwhere(mapping == int(vit_patch_id))
            if len(patch_coords) == 1:
                p_y, p_x = patch_coords[0]
                if location_id is not None and n_locations:
                    loc_x = location_id // 3
                    loc_y = location_id % 3
                    p_x += (loc_x + 0.5) * 0.33
                    p_y += (loc_y + 0.5) * 0.33
                else:
                    p_x += 0.5
                    p_y += 0.5
                points.append(MolmoGroundingPoint(
                    x=(p_x / mapping.shape[1]) * w,
                    y=(p_y / mapping.shape[0]) * h,
                    image_ix=image_ix,
                    point_id=object_id,
                    rank=len(points)))
                break
        # else: invalid patch id (no grid cell holds it) -> official behaviour: skip
    return points
