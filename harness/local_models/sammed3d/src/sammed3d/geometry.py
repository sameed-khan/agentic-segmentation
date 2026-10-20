"""SAM-Med3D session geometry (Phase 6 segmenter swap): patch placement/tiling,
prompt mapping, stitching — the pure numpy/scipy core of the local segmenter.

No torch: the ``--fake`` service, the sammed3d tests and the segd tests run the SAME
math as the real torch engine (CONTRACTS.md "Phase 6"). The one torchio-dependent step
of the official preprocessing (resample to 1.5 mm, ``tio.Resample``) is reimplemented
here with scipy ``map_coordinates`` on the exact same centred grid; the engine uses
this implementation and a test asserts bit-equality against torchio.

Coordinate spaces:
- session space: voxels of the image uploaded via ``set_image`` — the crop CaseState
  uploaded ([i, j, k], spacing from ``image_properties["spacing"]`` in mm, identity
  direction; segd convention CONTRACTS §1).
- model space: the same volume resampled to ``TARGET_SPACING_MM`` isotropic (official
  SAM-Med3D preprocessing step 1; grid centred exactly like torchio's
  ``Resample.get_reference_image``).
- patch space: a ``PATCH_VOX``^3 crop of model space, identified by its half-open
  window ``(lo, hi)`` in model-space voxels; windows may extend past the volume
  (zero padding, as the official ``CropOrPad`` pads).

Ground truth is never used: patch placement derives from prompt points and the
model's own previous prediction only (CONTRACTS §0).
"""
from __future__ import annotations

from itertools import product
from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

TARGET_SPACING_MM = 1.5        # official SAM-Med3D target spacing (utils/infer_utils.py)
PATCH_VOX = 128                # official crop_size -> 192 mm at 1.5 mm
LOWRES_VOX = 32                # decoder / mask-prompt resolution (128 // 4)
PATCH_OVERLAP_VOX = 32         # 25 % overlap between tiled patches (stride 96)
TILE_MARGIN_VOX = 16           # padding (24 mm) of a tiled axis's box so prompts/mask edges stay off patch borders
MAX_PATCHES = 27               # 3^3 cap; the tiler coarsens stride before dropping tiles
MAX_DRAWN_POINTS = 5           # lasso/scribble -> at most 5 sampled points
LOGIT_EPS = 1e-4               # clip before logit() when a mask becomes logits
MASK_PROMPT_LOGIT = 9.21024    # logit(1 - LOGIT_EPS): the +/- encoding of a binary mask


# ------------------------------------------------------------------ spaces
def transform_params(session_shape, spacing, target: float = TARGET_SPACING_MM) -> tuple:
    """torchio Resample's reference grid for identity-direction data: new size =
    floor(old_size * old_spacing / target), new origin centred on the old image
    (physical mm of model voxel 0). The one transform all conversions share."""
    shape = np.asarray(session_shape, dtype=float)
    sp = np.asarray(spacing, dtype=float)
    new_size = np.floor(shape * sp / target).astype(int)
    new_origin = ((shape - 1.0) * sp - (new_size - 1) * target) / 2.0
    return new_origin, new_size


def resampled_shape(session_shape, spacing, target: float = TARGET_SPACING_MM) -> tuple:
    """Model-space shape (torchio Resample's floor formula)."""
    return tuple(int(n) for n in transform_params(session_shape, spacing, target)[1])


def resample_np(image: np.ndarray, spacing, target: float = TARGET_SPACING_MM) -> np.ndarray:
    """Resample to ``target`` mm isotropic, linear interpolation, centred grid.

    Reimplements torchio ``Resample.get_reference_image`` (verified bit-identical for
    identity-direction data by test_geometry.py); out-of-volume samples are 0.
    """
    image = np.ascontiguousarray(image, dtype=np.float32)
    new_origin, new_size = transform_params(image.shape, spacing, target)
    coords = (np.indices(tuple(new_size), dtype=np.float64) * target
              + new_origin[:, None, None, None]) / np.asarray(spacing, dtype=float)[:, None, None, None]
    out = map_coordinates(image, coords.reshape(3, -1), order=1, mode="constant", cval=0.0)
    return out.reshape(tuple(new_size)).astype(np.float32)


def session_to_model(p, session_shape, spacing, target: float = TARGET_SPACING_MM) -> np.ndarray:
    """Session-space voxel -> model-space voxel (float, fractional); the SAME centred
    transform_params grid resample_np samples."""
    new_origin, _size = transform_params(session_shape, spacing, target)
    return (np.asarray(p, dtype=np.float64) * np.asarray(spacing, dtype=np.float64)
            - new_origin) / target


def model_to_session(p, session_shape, spacing, target: float = TARGET_SPACING_MM) -> np.ndarray:
    """Model-space voxel -> session-space voxel (float, fractional)."""
    new_origin, _size = transform_params(session_shape, spacing, target)
    return (np.asarray(p, dtype=np.float64) * target + new_origin) \
        / np.asarray(spacing, dtype=np.float64)


# ------------------------------------------------------------------- windows
def bbox_of_points(points) -> Optional[tuple]:
    """Half-open (lo, hi) model-space bbox of a list of (N,3) points, or None if empty."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return None
    lo = np.floor(pts.min(axis=0)).astype(int)
    hi = np.ceil(pts.max(axis=0)).astype(int) + 1
    return lo, hi


def bbox_union(a: Optional[tuple], b: Optional[tuple]) -> Optional[tuple]:
    if a is None:
        return b
    if b is None:
        return a
    return (np.minimum(a[0], b[0]), np.maximum(a[1], b[1]))


def window_for_extent(centre, box_lo, box_hi, volume_shape,
                      patch: int = PATCH_VOX) -> tuple:
    """A single patch window covering ``[box_lo, box_hi)``.

    Starts centred on ``centre`` (the first positive prompt) and shifts the minimum
    amount per axis so the window still covers the box. The window may extend past the
    volume (zero padding, as the official CropOrPad). Requires the box extent <= patch per axis (callers tile otherwise).
    """
    c = np.array(centre, dtype=np.float64)   # copy: callers may hold the same array
    for a in range(3):
        if box_lo[a] < c[a] - patch / 2.0:        # window must reach further low
            c[a] = box_lo[a] + patch / 2.0
        if box_hi[a] > c[a] + patch / 2.0:        # ... and further high
            c[a] = box_hi[a] - patch / 2.0
    # no clamp into the volume: like the official CropOrPad, a window centred near an
    # image edge is zero-padded (crop_patch) rather than shifted off the prompt
    lo = np.rint(c - patch / 2.0).astype(int)
    return lo, lo + patch


def _tiles_per_axis(extent: int, stride: int) -> int:
    if extent <= PATCH_VOX:
        return 1
    return int(np.ceil((extent - PATCH_VOX) / stride)) + 1


def choose_patches(box_lo, box_hi, seed_point, volume_shape,
                   max_patches: int = MAX_PATCHES,
                   overlap_vox: int = PATCH_OVERLAP_VOX) -> tuple[list, dict]:
    """Patches covering the union bbox of (positive prompts ∪ current mask).

    - extent <= PATCH_VOX on every axis: ONE window, started at ``seed_point`` (the
      first positive prompt) and minimally shifted to cover the box
      (window_for_extent); ``seed_point=None`` centres on the box.
    - otherwise: per axis longer than a window, overlapping tiles with ``overlap_vox``
      overlap (stride PATCH_VOX - overlap_vox, end tile aligned to the box end) over
      the box padded by TILE_MARGIN_VOX each side; axes that fit one window keep a
      single tile centred on the seed (as above). If that needs more
      than ``max_patches`` windows, the per-axis tile count is coarsened (largest
      first) until it fits — full box coverage is kept, the overlap shrinks and the
      choice is logged in the returned ``info``. Windows are sorted lexicographically
      (deterministic raster order).

    Returns (windows, info); info carries tiles/strides/capped/n_windows/note for the
    trajectory log.
    """
    box_lo = np.asarray(box_lo, dtype=int)
    box_hi = np.asarray(box_hi, dtype=int)
    volume_shape = np.asarray(volume_shape, dtype=int)
    ext = box_hi - box_lo
    if np.all(ext <= PATCH_VOX):
        seed = None if seed_point is None else np.asarray(seed_point, dtype=np.float64)
        lo, hi = window_for_extent(seed if seed is not None else (box_lo + box_hi) / 2.0,
                                   box_lo, box_hi, volume_shape)
        windows = [(lo, hi)]
        info = {"tiles": [1, 1, 1], "strides": [0, 0, 0], "capped": False,
                "n_windows": 1, "note": "single patch (box fits in one window)"}
        return windows, info

    # Layout. An axis whose extent fits one window keeps ONE tile, placed like the
    # single-patch case (centred on the first positive prompt, shifted minimally to
    # cover the box); an axis longer than a window is tiled over the box padded by
    # TILE_MARGIN_VOX on both sides, so prompts/mask edges are not on a patch border.
    seed = None if seed_point is None else np.asarray(seed_point, dtype=np.float64)
    centre = seed if seed is not None else (box_lo + box_hi) / 2.0
    one_lo, _ = window_for_extent(centre, box_lo, box_hi, volume_shape)
    lo_pad = np.where(ext > PATCH_VOX, box_lo - TILE_MARGIN_VOX, box_lo)
    ext_pad = np.where(ext > PATCH_VOX, ext + 2 * TILE_MARGIN_VOX, ext)
    raw_tiles = np.array([_tiles_per_axis(int(e), PATCH_VOX - overlap_vox) for e in ext_pad])
    min_tiles = np.array([_tiles_per_axis(int(e), PATCH_VOX) for e in ext_pad])  # zero overlap
    tiles = raw_tiles.copy()
    capped = False
    # coarsen only while the stride stays <= PATCH_VOX (no uncovered gaps) — the largest
    # tile count first; below that, tiles would have to be DROPPED (holes) and the cap
    # is logged loudly instead
    while int(np.prod(tiles)) > max_patches:
        cand = [a for a in range(3) if tiles[a] > min_tiles[a]]
        if not cand:
            break
        tiles[int(max(cand, key=lambda a: tiles[a]))] -= 1
        capped = True
    strides = np.zeros(3, dtype=int)
    starts_per_axis = []
    for a in range(3):
        if tiles[a] <= 1:
            starts_per_axis.append([int(one_lo[a])])
            continue
        stride = int(np.ceil((ext_pad[a] - PATCH_VOX) / (tiles[a] - 1)))
        strides[a] = stride
        starts_per_axis.append([int(lo_pad[a]) + i * stride for i in range(tiles[a])])
    windows = []
    for sx, sy, sz in product(*starts_per_axis):
        lo = np.array([sx, sy, sz], dtype=int)
        windows.append((lo, lo + PATCH_VOX))
    windows.sort(key=lambda w: tuple(int(v) for v in w[0]))     # deterministic raster order
    note = ""
    if int(len(windows)) > max_patches:
        # even zero-overlap tiling exceeds the cap: drop the tiles farthest from the
        # seed (or box centre) — the mask can end up with holes; the log says so
        capped = True
        ref = np.asarray(seed_point if seed_point is not None else (box_lo + box_hi) / 2.0,
                         dtype=np.float64)
        keyed = sorted(windows,
                       key=lambda w: (float(np.linalg.norm((w[0] + w[1]) / 2.0 - ref)),
                                      tuple(int(v) for v in w[0])))
        dropped = len(windows) - max_patches
        windows = keyed[:max_patches]
        note = (f"patch budget: {len(keyed)} zero-overlap tiles needed for the box, "
                f"capped at {max_patches}; dropped {dropped} farthest tile(s) "
                "(possible coverage holes)")
    elif capped:
        note = (f"patch budget: {int(np.prod(raw_tiles))} tiles needed for a "
                f"{int(100 * overlap_vox / PATCH_VOX)}%-overlap cover of the box, "
                f"capped at {max_patches} by coarsening the per-axis stride to "
                f"{[int(s) for s in strides]} vox "
                f"(overlap {[int(PATCH_VOX - s) for s in strides]} vox)")
    info = {"tiles": [int(t) for t in tiles], "strides": [int(s) for s in strides],
            "capped": capped, "n_windows": len(windows), "note": note}
    return windows, info


def points_inside_window(points, lo, hi, labels=None):
    """Points (N,3) inside the half-open window, in patch-local coordinates.
    With ``labels`` (N,), returns (local_points, local_labels)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        local = np.zeros((0, 3), dtype=np.float64)
        return (local, np.zeros((0,), dtype=np.int64)) if labels is not None else local
    inside = np.all((pts >= np.asarray(lo)) & (pts < np.asarray(hi)), axis=1)
    local = pts[inside] - np.asarray(lo, dtype=np.float64)
    if labels is None:
        return local
    return local, np.asarray(labels, dtype=np.int64)[inside]


def crop_patch(volume: np.ndarray, lo, hi, patch: int = PATCH_VOX) -> np.ndarray:
    """Crop the half-open window from ``volume`` with zero padding to exactly
    ``patch``^3 (windows may stick out of the volume; official CropOrPad pads 0)."""
    out = np.zeros((patch, patch, patch), dtype=np.float32)
    vshape = np.asarray(volume.shape)
    lo = np.asarray(lo, dtype=int)
    hi = np.asarray(hi, dtype=int)
    clo = np.maximum(lo, 0)
    chi = np.minimum(hi, vshape)
    if np.any(clo >= chi):
        return out
    src = volume[tuple(slice(int(l), int(h)) for l, h in zip(clo, chi))]
    dst = tuple(slice(int(l - lo[a]), int(l - lo[a]) + int(h - l))
                for a, (l, h) in enumerate(zip(clo, chi)))
    out[dst] = src
    return out


# ------------------------------------------------------- normalization, logits
def znorm_masked(patch: np.ndarray) -> np.ndarray:
    """Official ROI normalization (tio.ZNormalization(masking_method=lambda x: x > 0)):
    mean/sample-std (ddof=1) over voxels > 0, applied to ALL voxels. Degenerate
    patches (no positive voxel, or zero std — an all-zero edge patch) stay all-zero
    instead of raising as torchio would (documented deviation, CONTRACTS Phase 6)."""
    x = np.asarray(patch, dtype=np.float32)
    vals = x[x > 0]
    if vals.size < 2:
        return np.zeros_like(x)
    std = float(vals.std(ddof=1))
    if std == 0.0:
        return np.zeros_like(x)
    return ((x - float(vals.mean())) / std).astype(np.float32)


def prob_to_logits(p: np.ndarray) -> np.ndarray:
    """Probability map -> logits. The +/- encoding of binary masks (spec) and the
    reconstruction of the previous prediction's logits both go through this."""
    p = np.clip(np.asarray(p, dtype=np.float64), LOGIT_EPS, 1.0 - LOGIT_EPS)
    return np.log(p / (1.0 - p)).astype(np.float32)


def block_mean_to_32(patch: np.ndarray) -> np.ndarray:
    """128^3 patch -> 32^3 by exact 4^3 block means (a patch is always exactly 128^3)."""
    x = np.asarray(patch, dtype=np.float32)
    n = x.shape[0] // LOWRES_VOX
    if x.shape != (LOWRES_VOX * n,) * 3:
        raise ValueError(f"expected a {LOWRES_VOX * n}^3 patch, got {x.shape}")
    return x.reshape(LOWRES_VOX, n, LOWRES_VOX, n, LOWRES_VOX, n).mean(axis=(1, 3, 5))


def mask_prompt_for_window(prev_state, window) -> np.ndarray:
    """The low-res mask prompt (logits, 32^3) for one patch window.

    ``prev_state`` is (lo, prob_crop float16, prev_lowres) — the previous prediction's
    probability map in model space, cropped to its union window, plus that
    prediction's per-window raw 32^3 logits. Patches whose window exactly matches one
    of those windows reuse the RAW low-res logits (official iterative carry-over);
    everything else derives its prompt from the previous probability map: crop to the
    window (zero padding), block-mean to 32^3, prob_to_logits (a 0/1 mask becomes
    +/- MASK_PROMPT_LOGIT). No previous state -> all-zero prompt, exactly the
    official first call.
    """
    if prev_state is None:
        return np.zeros((LOWRES_VOX, LOWRES_VOX, LOWRES_VOX), dtype=np.float32)
    prev_lo, prev_prob, prev_lowres = prev_state
    prev_lo = np.asarray(prev_lo, dtype=int)
    w_lo, w_hi = (np.asarray(w, dtype=int) for w in window)
    for (p_lo, p_hi), logits in (prev_lowres or []):
        if (np.asarray(p_lo, dtype=int) == w_lo).all() and (np.asarray(p_hi, dtype=int) == w_hi).all():
            return np.asarray(logits, dtype=np.float32)
    lo = np.maximum(w_lo, prev_lo)
    hi = np.minimum(w_hi, prev_lo + np.asarray(prev_prob.shape, dtype=int))
    patch = np.zeros(tuple(int(v) for v in (w_hi - w_lo)), dtype=np.float32)
    if not np.any(lo >= hi):
        dst = tuple(slice(int(a - w_lo[i]), int(b - w_lo[i])) for i, (a, b) in enumerate(zip(lo, hi)))
        src = tuple(slice(int(l - prev_lo[i]), int(h - prev_lo[i])) for i, (l, h) in enumerate(zip(lo, hi)))
        patch[dst] = prev_prob[src]
    if patch.shape != (PATCH_VOX, PATCH_VOX, PATCH_VOX):
        patch = crop_patch(patch, [0, 0, 0], [PATCH_VOX] * 3)
    return prob_to_logits(block_mean_to_32(patch))


# ------------------------------------------------------------------ stitching
def stitch_logits(patch_logits: list, model_shape) -> tuple:
    """Average the per-patch LOGITS in overlaps (spec); sigmoid/threshold happens on
    the stitched array. ``patch_logits`` is a list of (((lo, hi), logits 128^3)).
    Returns (logits_crop float32 over the union bbox, union_lo)."""
    items = [((np.asarray(w[0], dtype=int), np.asarray(w[1], dtype=int)), np.asarray(l))
             for w, l in patch_logits]
    union_lo = np.maximum(np.min([b[0] for b, _l in items], axis=0), 0)
    union_hi = np.minimum(np.max([b[1] for b, _l in items], axis=0),
                          np.asarray(model_shape, dtype=int))
    size = tuple(int(h - l) for l, h in zip(union_lo, union_hi))
    total = np.zeros(size, dtype=np.float64)
    count = np.zeros(size, dtype=np.float64)
    for (lo, hi), logits in items:
        clo = np.maximum(lo, union_lo)
        chi = np.minimum(hi, union_hi)
        if np.any(clo >= chi):
            continue
        dst = tuple(slice(int(l - union_lo[i]), int(h - union_lo[i]))
                    for i, (l, h) in enumerate(zip(clo, chi)))
        src = tuple(slice(int(l - lo[i]), int(h - lo[i])) for i, (l, h) in enumerate(zip(clo, chi)))
        total[dst] += logits[src]
        count[dst] += 1.0
    return (total / np.maximum(count, 1.0)).astype(np.float32), union_lo


# ------------------------------------------------------- grid mapping (back)
def _grid_session_to_model(session_shape, spacing, target: float = TARGET_SPACING_MM) -> np.ndarray:
    """Coordinates (3, I, J, K) of every session voxel in model space (float)."""
    new_origin, _size = transform_params(session_shape, spacing, target)
    return (np.indices(tuple(session_shape), dtype=np.float64)
            * np.asarray(spacing, dtype=np.float64)[:, None, None, None]
            - new_origin[:, None, None, None]) / target


def _grid_model_to_session(model_shape, session_shape, spacing,
                           target: float = TARGET_SPACING_MM) -> np.ndarray:
    """Coordinates (3, D, H, W) of every model voxel in session space (float)."""
    new_origin, _size = transform_params(session_shape, spacing, target)
    return (np.indices(tuple(model_shape), dtype=np.float64) * target
            + new_origin[:, None, None, None]) \
        / np.asarray(spacing, dtype=np.float64)[:, None, None, None]


def map_binary_model_to_session(binary_model: np.ndarray, spacing, session_shape) -> np.ndarray:
    """Binary model-space mask -> session-space uint8 mask (official
    ``data_postprocess``: nearest-neighbour resampling of the binary mask)."""
    m = np.asarray(binary_model, dtype=np.float32)
    out = map_coordinates(m, _grid_session_to_model(session_shape, spacing),
                          order=0, mode="constant", cval=0.0)
    return (out > 0).astype(np.uint8).reshape(tuple(session_shape))


def map_mask_session_to_model(mask_session: np.ndarray, spacing, model_shape) -> np.ndarray:
    """Binary session-space mask -> model-space float mask (nearest, official label
    handling; a mask given by the harness becomes the previous-state probability map)."""
    m = np.asarray(mask_session, dtype=np.float32)
    out = map_coordinates(m, _grid_model_to_session(model_shape, m.shape, spacing),
                          order=0, mode="constant", cval=0.0)
    return out.reshape(tuple(model_shape))


# --------------------------------------------------------- prompt point maps
def bbox_quarter_points(bbox_lohi, include: bool) -> list[tuple]:
    """A box prompt -> native SAM-Med3D points: the box centre plus one point at each
    quarter position (0.25/0.75 of the extent per axis with extent > 1; a size-1 axis
    of a 2D box contributes its single voxel centre). All points carry the box's
    include label (spec: positive/negative per include flag). A 2D box -> 5 points,
    a 3D box -> 9. Coordinates are half-open-bbox space (callers translate)."""
    lo, hi = (np.asarray(v, dtype=np.float64) for v in bbox_lohi)
    ext = hi - lo
    levels = []
    for e in ext:
        levels.append([0.5] if e <= 1.0 else [0.25, 0.75])
    centre = lo + np.array([0.5, 0.5, 0.5]) * ext
    pts = [centre]
    for fracs in product(*levels):
        if all(f == 0.5 for f in fracs):
            continue                                     # that is the centre point
        pts.append(lo + np.asarray(fracs) * ext)
    return [(tuple(float(v) for v in p), 1 if include else 0) for p in pts]


def _edt_interior_seed(mask: np.ndarray) -> np.ndarray:
    """Voxel of the region's interior-most point (max of the Euclidean distance
    transform inside the region); ties resolved lexicographically (deterministic)."""
    dist = distance_transform_edt(np.asarray(mask) > 0)
    flat = int(np.argmax(dist))                          # first max in C order
    return np.asarray(np.unravel_index(flat, dist.shape), dtype=np.float64)


def edt_max_voxel(mask: np.ndarray) -> tuple:
    """Public form of _edt_interior_seed: the region's interior-most voxel (EDT max),
    used for the initial-seg/refine prompt point (spec)."""
    return tuple(int(v) for v in _edt_interior_seed(mask))


def fps_points(region_mask: np.ndarray, max_points: int = MAX_DRAWN_POINTS) -> list:
    """Up to ``max_points`` points inside a filled region: start at the interior-most
    voxel (EDT max) and add farthest-point-sampling picks (Euclidean, voxel units,
    lexicographic tie-break). Deterministic; GT-free. Coordinates are region-local."""
    idx = np.nonzero(np.asarray(region_mask) > 0)
    pts = np.stack(idx, axis=1).astype(np.float64)
    if pts.shape[0] == 0:
        return []
    if pts.shape[0] == 1:
        return [tuple(float(v) for v in pts[0])]
    seed = _edt_interior_seed(region_mask)
    chosen = [seed]
    dist = np.linalg.norm(pts - seed, axis=1)
    for _ in range(max_points - 1):
        k = int(np.argmax(dist))
        if dist[k] <= 0.0:
            break
        chosen.append(pts[k])
        dist = np.minimum(dist, np.linalg.norm(pts - pts[k], axis=1))
    return [tuple(float(v) for v in p) for p in chosen]


def _neighbours(voxel, shape, connectivity: int) -> list:
    offs = [o for o in product((-1, 0, 1), repeat=3)
            if any(o) and sum(abs(c) for c in o) <= (1 if connectivity == 6 else 2 if connectivity == 18 else 3)]
    out = []
    for o in offs:
        n = tuple(int(v) + d for v, d in zip(voxel, o))
        if all(0 <= c < s for c, s in zip(n, shape)):
            out.append(n)
    return out


def stroke_points(stroke_mask: np.ndarray, max_points: int = MAX_DRAWN_POINTS) -> list:
    """Up to ``max_points`` points along a 1-voxel stroke: order the stroke pixels by
    BFS hop distance from a deterministic start (a lexicographically smallest
    endpoint = pixel with exactly one stroke neighbour; fallback: lexicographically
    smallest pixel, e.g. a closed stroke), then pick evenly spaced fractions of the
    maximum distance. Deterministic; GT-free. Region-local coordinates."""
    m = np.asarray(stroke_mask) > 0
    idx = np.nonzero(m)
    coords = [tuple(int(v) for v in p) for p in np.stack(idx, axis=1)]
    if not coords:
        return []
    coord_set = set(coords)
    shape = m.shape
    endpoints = sorted(c for c in coords
                       if sum(1 for n in _neighbours(c, shape, 18) if n in coord_set) == 1)
    start = endpoints[0] if endpoints else sorted(coords)[0]
    order: list = []
    visited = {start}
    frontier = [start]
    dist_of = {start: 0}
    while frontier:
        nxt = []
        for c in sorted(frontier):                       # deterministic expansion
            order.append(c)
            for n in _neighbours(c, shape, 18):
                if n in coord_set and n not in visited:
                    visited.add(n)
                    dist_of[n] = dist_of[c] + 1
                    nxt.append(n)
        frontier = nxt
    if len(order) <= max_points:
        return [(float(a), float(b), float(c)) for a, b, c in order]
    max_d = max(dist_of.values())
    picks = []
    for t in np.linspace(0.0, 1.0, max_points):
        target = int(round(t * max_d))
        at = [c for c in order if dist_of[c] == target]   # discovery order = deterministic
        if at:
            picks.append(at[0])
    out = [(float(a), float(b), float(c)) for a, b, c in picks]
    return out or [(float(v) for v in start)]
