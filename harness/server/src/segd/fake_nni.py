"""Offline stand-in for the nnInteractive remote session (same method surface).

Used when ``nni_url=None`` (tests and segd without a GPU server; CONTRACTS §4).
It receives exactly the same calls CaseState would send to a real session, in the
same (session/crop) coordinate space, so coordinate translation and the sync rule
are testable offline.

Growth model:
- point: intensity-similar connected region grown from the seed (3D flood fill over
  ``|I - ref| <= tol`` with ``tol = 0.2 * (p99 - p1)`` of the uploaded image), capped at
  ``POINT_RADIUS_VOX`` from the seed, holes filled. This tracks a homogeneous lesion and
  stays off lower-intensity tissue.
- lasso: the drawn (filled) polygon is taken as-is on its slice and propagated a few
  slices in each direction, gated by the same intensity-similarity test, so a 2D lasso
  roughly fills a homogeneous 3D object. (The real model propagates properly in 3D.)
- scribble: the rasterized polyline is grown into the intensity-similar region
  (connected-component growth seeded from the line) on its slice, then propagated like lasso.
- bbox: the box is filled where intensity is similar to the box centre.
- initial seg: target buffer is overwritten with the given mask; with ``run_prediction``
  a light morphological clean-up (closing + hole fill) stands in for model refinement.

Prompts added with ``run_prediction=False`` are queued (they do not touch the target
buffer) and are applied by the next ``run_prediction=True`` prompt or ``_predict()``,
mirroring the real accumulate-then-predict semantics.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy import ndimage

POINT_RADIUS_VOX = 64       # cap on the grown-region radius from a point seed
PROPAGATION_SLICES = 3      # lasso/scribble slice propagation distance in each direction


def _sl(axis: int, idx: int) -> tuple:
    """Index tuple selecting slice ``idx`` along ``axis`` (other axes whole)."""
    out = [slice(None)] * 3
    out[axis] = int(idx)
    return tuple(out)


def _bbox_of(region: np.ndarray) -> Optional[List[List[int]]]:
    """Half-open [[x0,x1],[y0,y1],[z0,z1]] bbox of a boolean region, or None if empty."""
    idx = np.nonzero(region)
    if idx[0].size == 0:
        return None
    return [[int(a.min()), int(a.max()) + 1] for a in idx]


def _similarity_tol(image: np.ndarray) -> float:
    lo, hi = np.percentile(image, [1, 99])
    return max(0.2 * float(hi - lo), 5.0)


def _apply(buf: np.ndarray, region: np.ndarray, include: bool) -> None:
    if include:
        buf[region] = 1
    else:
        buf[region & (buf > 0)] = 0


class FakeNNI:
    """Drop-in for ``nnInteractiveRemoteInferenceSession`` with identical call signatures."""

    def __init__(self, verbose: bool = False):
        self.image: Optional[np.ndarray] = None          # 3D (session space), as uploaded
        self.target_buffer: Optional[np.ndarray] = None  # uint8 (I,J,K), updated in place
        self._last_paste_bbox: Optional[List[List[int]]] = None
        self.supported_interactions = {
            "points": True, "bbox2d": True, "bbox3d": True,
            "lasso": True, "scribble": True, "initial_label": True,
        }
        self.supports_undo = False
        self.supports_initial_label = True
        self.preferred_scribble_thickness = 1
        self.do_autozoom = True
        self._pending: list = []  # queued (method, kwargs) interactions awaiting a prediction
        self._verbose = verbose

    # ------------------------------------------------------------ setup
    def set_image(self, image: np.ndarray, image_properties: Optional[dict] = None) -> None:
        assert image.ndim == 4, f"expected a 4d image as input, got {image.ndim}d. Shape {image.shape}"
        self.image = np.asarray(image[0], dtype=np.float32)
        self._pending = []

    def set_target_buffer(self, target_buffer: np.ndarray) -> None:
        assert target_buffer.ndim == 3
        self.target_buffer = target_buffer

    def set_do_autozoom(self, do_autozoom: bool) -> None:
        self.do_autozoom = bool(do_autozoom)

    # --------------------------------------------------------- interactions
    def _flush_pending(self) -> None:
        for method, kwargs in self._pending:
            method(**kwargs)
        self._pending = []

    def _run(self, region_fn, include: bool, run_prediction: bool) -> None:
        if not run_prediction:
            self._pending.append((region_fn, {"include": include}))
            return
        self._flush_pending()
        region_fn(include=include)

    def add_point_interaction(self, coordinates, include_interaction: bool, run_prediction: bool = True,
                              override_capability_checks: bool = False) -> Optional[List[List[int]]]:
        self._run(lambda include: self._do_point(tuple(coordinates), include), include_interaction, run_prediction)
        return self._last_paste_bbox

    def add_lasso_interaction(self, lasso_image: np.ndarray, include_interaction: bool, run_prediction: bool = True,
                              override_capability_checks: bool = False,
                              interaction_bbox: Optional[List[List[int]]] = None) -> Optional[List[List[int]]]:
        self._run(lambda include: self._do_drawn(lasso_image, interaction_bbox, include, mode="lasso"),
                  include_interaction, run_prediction)
        return self._last_paste_bbox

    def add_scribble_interaction(self, scribble_image: np.ndarray, include_interaction: bool, run_prediction: bool = True,
                                 override_capability_checks: bool = False,
                                 interaction_bbox: Optional[List[List[int]]] = None) -> Optional[List[List[int]]]:
        self._run(lambda include: self._do_drawn(scribble_image, interaction_bbox, include, mode="scribble"),
                  include_interaction, run_prediction)
        return self._last_paste_bbox

    def add_bbox_interaction(self, bbox_coords, include_interaction: bool, run_prediction: bool = True,
                             override_capability_checks: bool = False) -> Optional[List[List[int]]]:
        self._run(lambda include: self._do_bbox(bbox_coords, include), include_interaction, run_prediction)
        return self._last_paste_bbox

    def add_initial_seg_interaction(self, initial_seg: np.ndarray, run_prediction: bool = False,
                                    override_capability_checks: bool = False) -> Optional[List[List[int]]]:
        assert self.target_buffer is not None
        assert initial_seg.shape == self.target_buffer.shape, \
            f"initial seg shape {initial_seg.shape} != target buffer {self.target_buffer.shape}"
        self._pending = []
        self.target_buffer[:] = initial_seg.astype(self.target_buffer.dtype, copy=False)
        if run_prediction:
            cleaned = ndimage.binary_closing(self.target_buffer > 0, structure=ndimage.generate_binary_structure(3, 1))
            cleaned = ndimage.binary_fill_holes(cleaned)
            self.target_buffer[:] = cleaned.astype(self.target_buffer.dtype)
        self._last_paste_bbox = _bbox_of(self.target_buffer > 0)
        return self._last_paste_bbox

    def _predict(self, force_full_refine: bool = False) -> Optional[List[List[int]]]:
        self._flush_pending()
        self._last_paste_bbox = _bbox_of(self.target_buffer > 0)
        return self._last_paste_bbox

    # -------------------------------------------------------- state control
    def reset_interactions(self) -> None:
        if self.target_buffer is not None:
            self.target_buffer.fill(0)
        self._pending = []
        self._last_paste_bbox = None

    def ping(self, timeout: float = 5.0) -> bool:
        return self.image is not None

    def close(self) -> None:
        pass

    # -------------------------------------------------------------- growth
    def _do_point(self, coord, include: bool) -> None:
        assert self.image is not None and self.target_buffer is not None
        shape = self.image.shape
        seed = tuple(int(min(max(round(float(c)), 0), n - 1)) for c, n in zip(coord, shape))
        r = POINT_RADIUS_VOX
        lo = [max(s - r, 0) for s in seed]
        hi = [min(s + r + 1, n) for s, n in zip(seed, shape)]
        crop = self.image[tuple(slice(a, b) for a, b in zip(lo, hi))]
        local_seed = tuple(s - a for s, a in zip(seed, lo))
        tol = _similarity_tol(self.image)
        ref = float(np.median(crop[tuple(slice(max(0, s - 2), s + 3) for s in local_seed)]))
        similar = np.abs(crop - ref) <= tol
        labels, _ = ndimage.label(similar, structure=ndimage.generate_binary_structure(3, 1))
        comp = labels == labels[local_seed]
        # cap by distance from the seed
        dist_from_seed = np.ones(comp.shape, dtype=np.float32)
        dist_from_seed[local_seed] = 0
        dist = ndimage.distance_transform_edt(dist_from_seed)
        region = np.zeros(shape, dtype=bool)
        region[tuple(slice(a, b) for a, b in zip(lo, hi))] = comp & (dist <= r)
        region = ndimage.binary_fill_holes(region)
        _apply(self.target_buffer, region, include)
        self._last_paste_bbox = _bbox_of(self.target_buffer > 0)

    def _do_drawn(self, drawn: np.ndarray, interaction_bbox, include: bool, mode: str) -> None:
        assert self.image is not None and self.target_buffer is not None
        drawn = np.asarray(drawn) > 0
        if interaction_bbox is None:
            interaction_bbox = [[0, s] for s in self.image.shape]
        placed = np.zeros(self.image.shape, dtype=bool)
        placed[tuple(slice(int(lb), int(ub)) for lb, ub in interaction_bbox)] = drawn
        # the interaction's fixed (2D plane) axis: the bbox axis of extent 1 (a 2D prompt
        # drawn on the axial plane has size 1 on k); fall back to axis 2 for 3D regions
        sizes = [int(ub) - int(lb) for lb, ub in interaction_bbox]
        fixed = sizes.index(1) if 1 in sizes else 2
        slices_with_content = sorted({int(k) for k in np.nonzero(placed)[fixed]})
        if not slices_with_content:
            self._last_paste_bbox = None
            return
        s0 = max(slices_with_content, key=lambda s: int(np.take(placed, s, axis=fixed).sum()))
        poly = np.take(placed, s0, axis=fixed)
        region = np.zeros(self.image.shape, dtype=bool)
        region[_sl(fixed, s0)] = poly
        if mode == "scribble":
            # grow the polyline into the intensity-similar region on its slice, then
            # propagate that grown region (the drawn line alone is too thin)
            poly = self._grow_2d(poly, s0, fixed)
            region[_sl(fixed, s0)] = poly
        tol = _similarity_tol(self.image)
        # propagation reference: intensity at the drawn region's centroid (the object the
        # agent drew around), not the region median which a loose polygon would dominate
        ys, xs = np.nonzero(poly)
        cy = int(round(ys.mean())) if ys.size else 0
        cx = int(round(xs.mean())) if xs.size else 0
        nb = np.take(self.image, s0, axis=fixed)[max(0, cy - 1):cy + 2, max(0, cx - 1):cx + 2]
        ref = float(np.median(nb)) if nb.size else 0.0
        # propagation gate reference: the drawn region's own similar-part area (a lasso
        # around the object includes a non-similar margin ring; comparing against the
        # full polygon would stop propagation after one slice)
        sim0 = np.abs(np.take(self.image, s0, axis=fixed) - ref) <= tol
        ref_area = int((poly & sim0).sum()) or int(poly.sum())
        for direction in (+1, -1):
            for d in range(1, PROPAGATION_SLICES + 1):
                s = s0 + direction * d
                if not (0 <= s < self.image.shape[fixed]):
                    break
                cand = poly & (np.abs(np.take(self.image, s, axis=fixed) - ref) <= tol)
                if cand.sum() < 0.5 * max(ref_area, 1):
                    break
                region[_sl(fixed, s)] = cand
        region = ndimage.binary_fill_holes(region)
        _apply(self.target_buffer, region, include)
        self._last_paste_bbox = _bbox_of(self.target_buffer > 0)

    def _grow_2d(self, seed2d: np.ndarray, s: int, fixed: int = 2) -> np.ndarray:
        tol = _similarity_tol(self.image)
        sl = np.take(self.image, s, axis=fixed)
        similar = np.abs(sl - np.median(sl[seed2d])) <= tol
        grown = ndimage.binary_propagation(seed2d, mask=similar,
                                           structure=ndimage.generate_binary_structure(2, 1))
        return grown & similar

    def _do_bbox(self, bbox_coords, include: bool) -> None:
        assert self.image is not None and self.target_buffer is not None
        shape = self.image.shape
        box = [[int(round(b[0])), int(round(b[1]))] for b in bbox_coords]
        box = [[max(0, lb), min(n, ub)] for (lb, ub), n in zip(box, shape)]
        center = tuple((lb + ub) // 2 for lb, ub in box)
        tol = _similarity_tol(self.image)
        # seed intensity: the centre voxel of the box (median of a 3^3 neighbourhood)
        sub = self.image[tuple(slice(max(0, c - 1), c + 2) for c in center)]
        ref = float(np.median(sub))
        box_arr = self.image[tuple(slice(lb, ub) for lb, ub in box)]
        similar = np.abs(box_arr - ref) <= tol
        labels, _ = ndimage.label(similar, structure=ndimage.generate_binary_structure(3, 1))
        local_center = tuple(min(max(c - lb, 0), n - 1) for c, (lb, ub), n in zip(center, box, box_arr.shape))
        if labels[local_center] == 0:
            comp = np.zeros_like(similar)
        else:
            comp = labels == labels[local_center]
        region = np.zeros(shape, dtype=bool)
        region[tuple(slice(lb, ub) for lb, ub in box)] = comp
        region = ndimage.binary_fill_holes(region)
        _apply(self.target_buffer, region, include)
        self._last_paste_bbox = _bbox_of(self.target_buffer > 0)
