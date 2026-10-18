"""Set-of-marks generation for the segd harness.

Implements research/harness_design_v2.md §5.4 and harness/server/CONTRACTS.md §2:
body mask, Laplacian-of-Gaussian blob extrema (both signs), SLIC supervoxel region
marks, coarse/fine thinning with minimum spacing, global stable integer IDs, and
the per-slice visibility rule.  All tunables are module constants so the Phase 1
tuning pass (eval/v2/mark_coverage.py) has one place to turn knobs.

Conventions
-----------
- Voxel space is the canonical array ``[i, j, k]`` of :mod:`segd.data` (MSTT-199 is
  1 mm isotropic with identity affine; ``spacing`` is honoured where cheap).
- In-plane ``(row, col)`` coordinates of a 2D slice always use the radiological
  display orientation of CONTRACTS §1: for ``axial`` row 0 = max j (anterior at
  top) and col 0 = max i (patient right on image left); ``coronal`` row 0 = max k
  (superior at top); ``sagittal`` col 0 = max j (anterior on image left).  The
  functions :func:`slice_shape`, :func:`extract_slice`, :func:`inplane_of` and
  :func:`voxel_of_inplane` are the single definition of that mapping;
  ``segd.render`` imports them instead of duplicating it.
- GT is never touched here (CONTRACTS §0): :func:`compute_marks` sees only the
  image volume.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.filters import threshold_multiotsu, threshold_otsu
from skimage.segmentation import slic

__all__ = [
    "Mark",
    "MarkSet",
    "compute_marks",
    "body_mask",
    "edge_marks_2d",
    "window_params",
    "normalize",
    "slice_shape",
    "extract_slice",
    "inplane_of",
    "voxel_of_inplane",
    "AXIS_INDEX",
]

# ---------------------------------------------------------------------------
# Tunable parameters (frozen after the design-case tuning pass; provenance is
# stored inside every saved MarkSet).
# ---------------------------------------------------------------------------
DOWNSAMPLE_MM = 2.0        # blob marks are computed on a ~2 mm downsampled volume
REGION_DOWNSAMPLE_MM = 4.0  # supervoxel grid (coarser: SLIC + EDT dominate the runtime)
LOG_SIGMAS_MM = (2.0, 4.0, 8.0, 16.0)
LOG_REL_THRESHOLD = 0.08   # keep |response| >= max(this * p99.9|response|, abs floor)
LOG_ABS_THRESHOLD = 0.05
BODY_ERODE_MM = 4.0        # blob candidates must be >= this far inside the body surface
SUPERVOXEL_MM = 20.0       # target supervoxel equivalent-sphere diameter
SUPERVOXEL_COMPACTNESS = 0.5
SUPERVOXEL_MAX_ITER = 5
SUPERVOXEL_MIN_MM3 = 1000.0  # drop supervoxel fragments below ~12 mm equivalent diameter
BODY_GRID_MAX_DIM = 240     # body mask computed on a grid with <= this voxels per axis
REGION_STRENGTH_MM = 10.0  # region strength = clip(max-dist-to-supervoxel-boundary / this)
FINE_SPACING_MM = 5.0
FINE_CLEAR_COARSE_MM = 3.0  # fine marks also keep this clearance from coarse marks
REGION_SCALE_CLAMP_MM = (8.0, 40.0)
MIN_VISIBLE_MM = 2.0       # visibility half-thickness floor (CONTRACTS §2)
VISIBLE_CAP = 50           # ~50 labels per panel; thinning is by strength priority
VISIBLE_THIN_MM = {        # greedy in-plane thinning inside visible(): keeps labels apart
    "coarse": 10.0,
    "fine": 6.0,
}

AXIS_INDEX = {"axial": 2, "coronal": 1, "sagittal": 0}  # fixed voxel axis per view
AXES = ("axial", "coronal", "sagittal")

# Bump when the mark algorithm changes so stale caches (CaseState + mark_coverage)
# are recomputed instead of trusted.  v7: kinds are saved with a wide enough dtype
# ('region' was truncated to 'regio' by the '<U5' dtype of v6 caches), legacy
# 'regio' strings are mapped back on load, and the region-mark grid positions no
# longer collapse toward the volume corner (the clamp used the downsampled shape).
MARKS_VERSION = 7

# In-plane display mapping (CONTRACTS §1): (row axis index, row flipped, col axis index,
# col flipped).  For axial the slice is img[:, :, k] = [i, j]; display needs
# row = J-1-j, col = I-1-i -> row axis 1 flipped, col axis 0 flipped, etc.
_INPLANE = {
    "axial": (1, True, 0, True),
    "coronal": (2, True, 0, True),
    "sagittal": (2, True, 1, True),
}


# ---------------------------------------------------------------------------
# Display-orientation geometry (single definition shared with segd.render)
# ---------------------------------------------------------------------------
def slice_shape(shape: tuple[int, int, int], axis: str) -> tuple[int, int]:
    """(rows, cols) of the display-oriented 2D slice for ``axis``."""
    ra, _, ca, _ = _INPLANE[axis]
    return int(shape[ra]), int(shape[ca])


def extract_slice(vol: np.ndarray, axis: str, idx: int) -> np.ndarray:
    """Display-oriented 2D slice ``vol[..., axis-slice idx]`` (radiological layout)."""
    arr = np.take(vol, int(idx), axis=AXIS_INDEX[axis])          # [a, b] in voxel order
    s = np.flipud(np.fliplr(arr.T))                              # flip both display axes
    return np.ascontiguousarray(s)


def inplane_of(ijk: tuple[int, int, int], axis: str, rows: int, cols: int) -> tuple[int, int]:
    """Voxel (i, j, k) -> (row, col) of the display-oriented slice for ``axis``."""
    ra, rflip, ca, cflip = _INPLANE[axis]
    r = int(ijk[ra])
    c = int(ijk[ca])
    return (rows - 1 - r if rflip else r, cols - 1 - c if cflip else c)


def voxel_of_inplane(row: int, col: int, slice_idx: int, axis: str,
                     rows: int, cols: int) -> tuple[int, int, int]:
    """Inverse of :func:`inplane_of`; clamps into the slice bounds."""
    ra, rflip, ca, cflip = _INPLANE[axis]
    row = int(np.clip(row, 0, rows - 1))
    col = int(np.clip(col, 0, cols - 1))
    r = rows - 1 - row if rflip else row
    c = cols - 1 - col if cflip else col
    ijk = [0, 0, 0]
    ijk[ra] = r
    ijk[ca] = c
    ijk[AXIS_INDEX[axis]] = int(slice_idx)
    return (int(ijk[0]), int(ijk[1]), int(ijk[2]))


# ---------------------------------------------------------------------------
# Marks
# ---------------------------------------------------------------------------
@dataclass
class Mark:
    id: int                  # global, stable, 1..N; coarse marks get the lowest IDs
    kind: str                # "blob+" | "blob-" | "region"
    ijk: tuple[int, int, int]  # voxel (never shown to the agent)
    scale_mm: float
    level: str               # "coarse" | "fine"  (fine revealed in crops)


class MarkSet:
    """Global 3D mark set with the CONTRACTS §2 visibility rule.

    ``strengths`` maps mark id -> priority in [0, 1] used for deterministic
    thinning when a panel would exceed ``max_n`` labels.  ``shape`` is the voxel
    shape the marks live on (needed for in-plane display mapping).
    """

    def __init__(self, marks: dict[int, Mark], spacing: tuple[float, float, float],
                 window: tuple[float, float], strengths: dict[int, float] | None = None,
                 params: dict | None = None, shape: tuple[int, int, int] | None = None):
        self.spacing = tuple(float(s) for s in spacing)
        self.window = (float(window[0]), float(window[1]))
        self.marks: dict[int, Mark] = dict(marks)
        self.shape = tuple(int(s) for s in shape) if shape else None
        self.strengths: dict[int, float] = dict(strengths or {})
        self.params = params or {}
        # id-sorted parallel arrays for vectorised visible()
        ids = sorted(self.marks)
        self._ids = np.asarray(ids, dtype=np.int32)
        # dtype taken from the data ('coarse' is 6 chars); explicit capture only for
        # the empty case so no kind/level string is ever truncated by a fixed dtype.
        self._level = np.asarray([self.marks[i].level for i in ids]) if ids else np.empty(0, dtype="<U6")
        self._ijk = (np.asarray([self.marks[i].ijk[0] for i in ids], dtype=np.int32),
                     np.asarray([self.marks[i].ijk[1] for i in ids], dtype=np.int32),
                     np.asarray([self.marks[i].ijk[2] for i in ids], dtype=np.int32))
        self._scale = np.asarray([self.marks[i].scale_mm for i in ids], dtype=np.float64)
        self._strength = np.asarray([self.strengths.get(i, 0.0) for i in ids], dtype=np.float64)

    def __len__(self) -> int:
        return len(self.marks)

    # -- persistence --------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # atomic: concurrent runs share results/cache/marks, so never expose a
        # half-written file (tmp name ends in .npz so numpy does not append it)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp.npz")
        try:
            self._write_npz(tmp)
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def _write_npz(self, tmp: Path) -> None:
        ijk = np.stack(self._ijk, axis=1) if len(self._ids) else np.zeros((0, 3), np.int32)
        kinds = [self.marks[i].kind for i in self._ids]
        # np.asarray sizes the dtype from the data (no fixed-width truncation,
        # unlike the v6 '<U5' which cut 'region' to 'regio')
        kinds_arr = np.asarray(kinds) if kinds else np.empty(0, dtype="<U6")
        np.savez_compressed(
            tmp,
            ids=self._ids, level=self._level, ijk=ijk, scale_mm=self._scale,
            strength=self._strength, kinds=kinds_arr,
            spacing=np.asarray(self.spacing, np.float64),
            window=np.asarray(self.window, np.float64),
            shape=np.asarray(self.shape or (0, 0, 0), np.int64),
            params=np.asarray(json.dumps(self.params)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "MarkSet":
        d = np.load(Path(path), allow_pickle=False)
        ids = d["ids"]
        ijk = d["ijk"]
        # MARKS_VERSION <= 6 caches stored kinds as '<U5', truncating 'region' to
        # 'regio'; map the legacy spelling back so loaders never see it.
        kinds = ["region" if str(k) == "regio" else str(k) for k in d["kinds"]]
        levels = [str(l) for l in d["level"]]
        scales = d["scale_mm"]
        strengths = d["strength"]
        marks = {int(i): Mark(int(i), kinds[n], (int(ijk[n, 0]), int(ijk[n, 1]), int(ijk[n, 2])),
                              float(scales[n]), levels[n])
                 for n, i in enumerate(ids)}
        st = {int(i): float(strengths[n]) for n, i in enumerate(ids)}
        shape = tuple(int(v) for v in d["shape"])
        return cls(marks, tuple(d["spacing"]), tuple(d["window"]), st,
                   params=json.loads(str(d["params"])), shape=shape)

    # -- visibility ---------------------------------------------------------
    def visible(self, axis: str, slice_idx: int, box2d: tuple | None = None,
                level: str = "coarse", max_n: int = VISIBLE_CAP) -> list[Mark]:
        """Marks shown on one slice of ``axis`` (CONTRACTS §2).

        A mark is visible if ``|mark_slice_coord - slice_idx| <= max(scale_mm/2, 2)``
        (in voxels; marks are stored on the 1 mm isotropic grid).  ``box2d`` is an
        optional ``(r0, c0, r1, c1)`` in-plane box in display-oriented coordinates
        restricting the result to a crop.  ``level="coarse"`` returns coarse marks
        only, ``"fine"`` coarse + fine — and "fine" always contains everything
        "coarse" returns on the same slice/box: the coarse pass (thinned at 10 mm,
        capped at ``max_n``) is computed identically for both levels and kept as
        the core of the fine set, while fine candidates (thinned at 6 mm, 3 mm
        clearance from the kept coarse marks) only fill the remaining label
        budget.  Selection: greedy priority thinning for in-plane separation,
        then — if a set still exceeds ``max_n`` — a deterministic
        farthest-point-sampling cap so the drawn labels spread across the whole
        panel instead of clumping on the strongest structures (a purely
        strength-ordered cap measurably loses in-slice coverage on dense slices,
        e.g. L43).  The returned list is ordered strongest-first.
        """
        if axis not in AXIS_INDEX or level not in ("coarse", "fine") or len(self._ids) == 0:
            return []
        sl = self._ijk[AXIS_INDEX[axis]].astype(np.float64)
        r_vox = np.maximum(self._scale / (2.0 * self.spacing[AXIS_INDEX[axis]]),
                           MIN_VISIBLE_MM / self.spacing[AXIS_INDEX[axis]])
        sel = np.abs(sl - float(slice_idx)) <= r_vox
        is_coarse = self._level == "coarse"
        if level == "coarse":
            sel &= is_coarse
        rows, cols = slice_shape(self.shape, axis)
        rr, cc = _inplane_rows_cols(self._ijk, axis, rows, cols)
        if box2d is not None:
            r0, c0, r1, c1 = (int(v) for v in box2d)
            sel &= (rr >= r0) & (rr < r1) & (cc >= c0) & (cc < c1)
        idxs = np.nonzero(sel)[0]
        if idxs.size == 0:
            return []
        pts = np.stack([rr[idxs], cc[idxs]], axis=1).astype(np.float64)
        prio = self._strength[idxs]
        inplane_mm = np.asarray(self.spacing)[[_INPLANE[axis][0], _INPLANE[axis][2]]]
        pts_mm = pts * inplane_mm[None, :]
        capped = bool(max_n and max_n > 0)

        def _cap(rel: np.ndarray) -> np.ndarray:
            if capped and rel.size > max_n:
                return rel[_fps_select(pts_mm[rel], prio[rel], int(max_n))]
            return rel

        co = np.nonzero(is_coarse[idxs])[0]          # indices relative to ``idxs``
        fi = np.nonzero(~is_coarse[idxs])[0]
        kept_co = np.nonzero(_greedy_spatial_thin(pts_mm[co], prio[co],
                                                  radius_mm=VISIBLE_THIN_MM["coarse"]))[0]
        co_capped = _cap(kept_co)
        if level == "coarse":
            order = co_capped
        else:
            fine_pool = fi
            extra = (pts_mm[co[co_capped]], FINE_CLEAR_COARSE_MM) if co_capped.size else ()
            kept_fi = np.nonzero(_greedy_spatial_thin(pts_mm[fine_pool], prio[fine_pool],
                                                      radius_mm=VISIBLE_THIN_MM["fine"],
                                                      extra=extra if len(extra) == 2 else None))[0]
            n_co = co_capped.size
            if capped:
                slots = int(max_n) - n_co
                if slots <= 0:
                    order = co_capped          # coarse alone fills the budget
                elif kept_fi.size > slots:
                    order = np.concatenate([co_capped,
                                            fine_pool[kept_fi][_fps_select(
                                                pts_mm[fine_pool[kept_fi]],
                                                prio[fine_pool[kept_fi]], slots)]])
                else:
                    order = np.concatenate([co_capped, fine_pool[kept_fi]])
            else:
                order = np.concatenate([co_capped, fine_pool[kept_fi]])
        # strongest first, stable tie-break on position
        sub = idxs[order]
        strength = self._strength[sub]
        r_, c_ = pts[order, 0], pts[order, 1]
        srt = np.lexsort((c_, r_, -strength))
        return [self.marks[int(self._ids[i])] for i in sub[srt]]


def _fps_select(pts_mm: np.ndarray, priority: np.ndarray, max_n: int) -> np.ndarray:
    """Deterministic farthest-point selection of ``max_n`` points (cap step).

    Seeds with the strongest point; then repeatedly adds the remaining point
    with the largest distance to everything already accepted (ties: stronger
    first, then lowest row/col).  Spreads capped labels over the panel instead
    of spending the budget on the strongest cluster.
    """
    n = len(pts_mm)
    if n <= max_n:
        return np.ones(n, dtype=bool)
    seed = int(np.lexsort((np.arange(n), -priority))[0])
    keep = np.zeros(n, dtype=bool)
    keep[seed] = True
    diff = pts_mm - pts_mm[seed]
    min_d2 = np.einsum("ij,ij->i", diff, diff)
    for _ in range(max_n - 1):
        cand = np.nonzero(~keep)[0]
        if cand.size == 0:
            break
        pick = cand[np.lexsort((pts_mm[cand, 1], pts_mm[cand, 0],
                                -priority[cand], -min_d2[cand]))[0]]
        keep[pick] = True
        diff = pts_mm - pts_mm[pick]
        np.minimum(min_d2, np.einsum("ij,ij->i", diff, diff), out=min_d2)
    return keep


def _inplane_rows_cols(ijk_arrays, axis: str, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    ra, rflip, ca, cflip = _INPLANE[axis]
    r = ijk_arrays[ra].astype(np.int64)
    c = ijk_arrays[ca].astype(np.int64)
    if rflip:
        r = (rows - 1) - r
    if cflip:
        c = (cols - 1) - c
    return r, c


# ---------------------------------------------------------------------------
# Greedy deterministic spatial thinning (priority order, minimum distance)
# ---------------------------------------------------------------------------
def _greedy_spatial_thin(pts_mm: np.ndarray, priority: np.ndarray, radius_mm: float,
                         extra: tuple[np.ndarray, float] | None = None) -> np.ndarray:
    """Greedy thinning in priority order with a minimum distance.

    Returns a boolean mask of accepted points.  ``extra`` optionally forbids
    points within a second radius of a fixed reference set (e.g. accepted coarse
    marks during the fine pass).  Chunked cKDTree queries keep this fast.
    """
    n = len(pts_mm)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(-priority, kind="stable")
    keep = np.zeros(n, dtype=bool)
    acc: list[np.ndarray] = []          # accepted points of this set
    tree = None
    CH = 1024
    for start in range(0, n, CH):
        chunk = order[start:start + CH]
        P = pts_mm[chunk]
        ok = np.ones(len(chunk), dtype=bool)
        if tree is not None:
            d, _ = tree.query(P, k=1, distance_upper_bound=radius_mm)
            ok &= ~(d <= radius_mm)
        if extra is not None:
            ref, rad = extra
            if len(ref):
                d2 = _min_dist2(P, ref)
                ok &= d2 > rad * rad
        local: list[np.ndarray] = []
        for t in range(len(chunk)):
            if not ok[t]:
                continue
            if local:
                d2 = _min_dist2(P[t:t + 1], np.asarray(local))
                if d2[0] <= radius_mm * radius_mm:
                    continue
            ok[t] = True
            keep[chunk[t]] = True
            local.append(P[t])
        if local:
            acc.append(np.asarray(local))
            tree = cKDTree(np.concatenate(acc))
    return keep


def _min_dist2(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """min over rows of Q of squared distance for each row of P."""
    d2 = np.empty(len(P), dtype=np.float64)
    CH = 4096
    for s in range(0, len(P), CH):
        block = P[s:s + CH]
        diff = block[:, None, :] - Q[None, :, :]
        d2[s:s + CH] = np.min(np.einsum("ijk,ijk->ij", diff, diff), axis=1)
    return d2


# ---------------------------------------------------------------------------
# Body mask and intensity normalisation
# ---------------------------------------------------------------------------
def body_mask(image: np.ndarray) -> np.ndarray:
    """Multi-Otsu foreground (lowest threshold = air/tissue split) -> largest 3D
    component -> holes filled per slice and in 3D.

    The lowest of three multi-Otsu thresholds is used rather than 2-class Otsu:
    several design volumes have dark limbs or dark/different-intensity tumors and
    2-class Otsu splits bright tissue from the rest, which drops the tumour (and
    sometimes whole muscle groups) from the mask (measured on M07/M09/V65).
    Computed on a downsampled grid (``BODY_GRID_MAX_DIM`` voxels per axis) and
    nearest-neighbour-upsampled to ``image.shape``: each full-resolution voxel maps
    to one grid voxel, so a 6-connected grid component stays 6-connected and the
    result is a single component without a full-resolution relabel pass.
    """
    image = np.asarray(image)
    step = max(1, int(np.ceil(max(image.shape) / BODY_GRID_MAX_DIM)))
    small = image[::step, ::step, ::step]
    small = small.astype(np.float64)
    try:
        thr = float(threshold_multiotsu(small, classes=3)[0])
    except ValueError:
        thr = float(threshold_otsu(small))   # degenerate histogram: fall back
    body = _largest_component(small > thr)
    filled = _fill_per_slice(body, axis=2)
    filled |= _fill_per_slice(body, axis=1)
    filled |= _fill_per_slice(body, axis=0)
    filled = ndi.binary_fill_holes(filled)
    body = _largest_component(filled)
    if not body.any():
        return np.zeros(image.shape, dtype=bool)
    idx = [np.minimum((np.arange(n) * ns) // n, ns - 1)
           for n, ns in zip(image.shape, body.shape)]
    return body[np.ix_(*idx)]


def _largest_component(fg: np.ndarray) -> np.ndarray:
    if not fg.any():
        return np.zeros_like(fg)
    lab, n = ndi.label(fg)
    if n <= 1:
        out = fg.copy()
        del lab
        return out
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = sizes.argmax()
    return lab == keep


def _fill_per_slice(mask3d: np.ndarray, axis: int) -> np.ndarray:
    out = mask3d.copy()
    n = mask3d.shape[axis]
    for s in range(n):
        sl1 = [slice(None)] * 3
        sl1[axis] = s
        slab = mask3d[tuple(sl1)]
        out[tuple(sl1)] = ndi.binary_fill_holes(slab)
    return out


def window_params(image: np.ndarray, body: np.ndarray) -> tuple[float, float]:
    """(lo, hi): 0.5/99.5 percentiles inside the body (CONTRACTS §3 window)."""
    vals = image[body.astype(bool)]
    if vals.size == 0:
        vals = np.asarray(image).ravel()
    lo, hi = np.percentile(vals.astype(np.float64), (0.5, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = float(lo) + 1.0
    return float(lo), float(hi)


def normalize(image: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    lo, hi = window
    out = np.clip(image.astype(np.float32), lo, hi)
    out -= np.float32(lo)
    out /= np.float32(hi - lo)
    return out


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------
def _downsample_factors(spacing: tuple[float, float, float],
                        target_mm: float = DOWNSAMPLE_MM) -> tuple[int, int, int]:
    return tuple(max(1, int(round(target_mm / float(s)))) for s in spacing)


def _blob_candidates(dnorm: np.ndarray, dbody: np.ndarray, ds_mm: float
                     ) -> list[tuple[tuple[int, int, int], float, float, str]]:
    """(ijk_ds, sigma_mm, |scale-normalised response|, sign) for both signs."""
    eroded_iter = max(1, int(round(BODY_ERODE_MM / ds_mm)))
    inner = ndi.binary_erosion(dbody, iterations=eroded_iter) if dbody.any() else dbody
    out: list[tuple[tuple[int, int, int], float, float, str]] = []
    for sigma_mm in LOG_SIGMAS_MM:
        sv = sigma_mm / ds_mm
        sm = ndi.gaussian_filter(dnorm, sv)
        lg = -ndi.laplace(sm) * (sv * sv)      # scale-normalised LoG
        vals = np.abs(lg)[inner]
        if vals.size == 0:
            continue
        p999 = float(np.percentile(vals[::13], 99.9)) if vals.size > 64 else float(vals.max())
        thr = max(LOG_ABS_THRESHOLD, LOG_REL_THRESHOLD * p999)
        for sign, kind in ((+1, "blob+"), (-1, "blob-")):
            filt = ndi.maximum_filter(lg, size=3) if sign > 0 else ndi.minimum_filter(lg, size=3)
            ext = (lg == filt) & (lg * sign >= thr) & inner
            for ijk in np.argwhere(ext):
                out.append((tuple(int(v) for v in ijk), sigma_mm, float(abs(lg[tuple(ijk)])), kind))
    return out


def _supervoxel_at(ijk: tuple[int, int, int], labels_ds: np.ndarray,
                   factors: tuple[int, int, int]) -> int:
    """Supervoxel id containing full-resolution voxel ``ijk`` (0 = outside body)."""
    pos = tuple(min(int(ijk[a]) // factors[a], labels_ds.shape[a] - 1) for a in range(3))
    return int(labels_ds[pos])


def _per_group_strongest(group: np.ndarray, prio: np.ndarray) -> np.ndarray:
    """Index of the strongest candidate within each group (ties: lowest index)."""
    out: list[int] = []
    for g in np.unique(group[group > 0]):
        idx = np.nonzero(group == g)[0]
        out.append(int(idx[np.lexsort((idx, -prio[idx]))[0]]))
    return np.asarray(sorted(out), dtype=np.int64)


def _label_boundary(labels: np.ndarray) -> np.ndarray:
    b = np.zeros(labels.shape, dtype=bool)
    a = labels
    for ax in range(3):
        s1 = [slice(None)] * 3
        s2 = [slice(None)] * 3
        s1[ax] = slice(0, -1)
        s2[ax] = slice(1, None)
        diff = a[tuple(s1)] != a[tuple(s2)]
        b[tuple(s1)] |= diff
        b[tuple(s2)] |= diff
    return b


def _region_candidates(dnorm: np.ndarray, dbody: np.ndarray, ds_mm: float,
                       factors: tuple[int, int, int]
                       ) -> tuple[list[tuple[tuple[int, int, int], float, float, str]], np.ndarray]:
    """SLIC supervoxels -> (candidates, label grid).

    For every supervoxel: the point furthest from the supervoxel boundary (EDT
    maximum).  Returned voxel indices are already on the full-resolution grid
    (``ijk_ds * factors`` — safe without a clamp: ``(ceil(S/f) - 1) * f <= S - 1``,
    and :func:`compute_marks` clamps against the true full shape anyway).  The
    label grid is the downsampled supervoxel partition (0 outside the body).
    """
    n_body = int(dbody.sum())
    if n_body < 40:
        return [], None   # too few body voxels for a supervoxel partition
    vox_per_sv = (SUPERVOXEL_MM ** 3) * np.pi / 6.0 / (ds_mm ** 3)
    n_seg = int(np.clip(n_body / max(vox_per_sv, 1.0), 8, 60000))
    labels = slic(dnorm, n_segments=n_seg, compactness=SUPERVOXEL_COMPACTNESS,
                  max_num_iter=SUPERVOXEL_MAX_ITER, start_label=1, mask=dbody,
                  channel_axis=None)
    nl = int(labels.max())
    if nl < 1:
        return [], None
    boundary = _label_boundary(labels)
    dist = ndi.distance_transform_edt(~boundary, sampling=(ds_mm, ds_mm, ds_mm))
    ids = np.arange(1, nl + 1)
    sizes = np.bincount(labels.ravel(), minlength=nl + 1)[1:]
    pos = ndi.maximum_position(dist, labels, ids)
    min_vox = max(4.0, SUPERVOXEL_MIN_MM3 / (ds_mm ** 3))
    out = []
    for n_lab, p in zip(ids, pos):
        if sizes[n_lab - 1] < min_vox:
            continue
        d_mm = float(dist[int(p[0]), int(p[1]), int(p[2])])
        vol_mm3 = float(sizes[n_lab - 1]) * ds_mm ** 3
        eq_diam = 2.0 * (3.0 * vol_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0)
        # full-resolution position: p * factors (no clamp — the previous clamp
        # against the DOWNSAMPLED shape collapsed ~3/4 of region marks into the
        # volume's low-index corner; compute_marks clamps against image.shape)
        ijk_full = (int(p[0]) * factors[0],
                    int(p[1]) * factors[1],
                    int(p[2]) * factors[2])
        out.append((ijk_full, float(np.clip(eq_diam, *REGION_SCALE_CLAMP_MM)), d_mm, "region"))
    return out, labels


# ---------------------------------------------------------------------------
# compute_marks
# ---------------------------------------------------------------------------
def compute_marks(image: np.ndarray, spacing: tuple[float, float, float],
                  timings: dict | None = None) -> MarkSet:
    """Deterministic mark set for one volume (CONTRACTS §2).

    Pipeline: body mask -> intensity window -> 2 mm blob grid (LoG extrema, both
    signs, 4 scales) + ~20 mm SLIC supervoxels on a coarser grid (region points =
    the position furthest from the supervoxel boundary) -> joint priority ranking
    -> coarse level = the strongest candidate within each supervoxel (so every
    ~20 mm region of the body, including tumour interiors, contributes one
    coarse mark), fine level = the rest thinned at 5 mm with 3 mm clearance from
    coarse marks -> stable IDs (coarse first, each sorted by k, then j, then i).
    ``timings``, when given, receives per-stage wall seconds for the offline
    diagnostic.
    """
    def _t(tag: str, t0: float) -> None:
        if timings is not None:
            timings[tag] = timings.get(tag, 0.0) + (time.monotonic() - t0)

    image = np.asarray(image, dtype=np.float32)
    spacing = tuple(float(s) for s in spacing)
    t0 = time.monotonic()
    body = body_mask(image)
    _t("body_mask", t0)

    t0 = time.monotonic()
    win = window_params(image, body)
    norm = normalize(image, win)
    _t("normalize", t0)

    f = _downsample_factors(spacing)
    dnorm = norm[::f[0], ::f[1], ::f[2]].copy()
    dbody = body[::f[0], ::f[1], ::f[2]].copy()
    ds_mm = (spacing[0] * f[0], spacing[1] * f[1], spacing[2] * f[2])
    ds_mean_mm = float(np.mean(ds_mm))

    t0 = time.monotonic()
    blobs = _blob_candidates(dnorm, dbody, ds_mean_mm)
    _t("log_blobs", t0)

    t0 = time.monotonic()
    fr = _downsample_factors(spacing, REGION_DOWNSAMPLE_MM)
    dnorm_r = norm[::fr[0], ::fr[1], ::fr[2]].copy()
    dbody_r = body[::fr[0], ::fr[1], ::fr[2]].copy()
    ds_r_mm = float(np.mean((spacing[0] * fr[0], spacing[1] * fr[1], spacing[2] * fr[2])))
    regions, labels_r = _region_candidates(dnorm_r, dbody_r, ds_r_mm, fr)
    _t("supervoxels", t0)
    has_sv = labels_r is not None and int(labels_r.max()) >= 1

    # to full-resolution voxel indices and mm positions (blobs are on the 2 mm grid,
    # regions are already full-resolution from _region_candidates)
    shape = image.shape
    cands = []  # (ijk_full, scale_mm, kind, raw_strength)
    for ijk_ds, scale_mm, raw, kind in blobs:
        ijk = (min(int(ijk_ds[0]) * f[0], shape[0] - 1),
               min(int(ijk_ds[1]) * f[1], shape[1] - 1),
               min(int(ijk_ds[2]) * f[2], shape[2] - 1))
        cands.append((ijk, scale_mm, kind, raw))
    for ijk_full, scale_mm, raw, kind in regions:
        ijk = (min(int(ijk_full[0]), shape[0] - 1),
               min(int(ijk_full[1]), shape[1] - 1),
               min(int(ijk_full[2]), shape[2] - 1))
        cands.append((ijk, scale_mm, kind, raw))
    if not cands:
        return MarkSet({}, spacing, win, {}, params=_params_snapshot())

    # joint priority: percentile rank of normalised strength across all candidates
    t0 = time.monotonic()
    raw = np.asarray([c[3] for c in cands], dtype=np.float64)
    kinds = [c[2] for c in cands]
    blob_mask = np.asarray([k != "region" for k in kinds])
    s_norm = np.zeros(len(cands), dtype=np.float64)
    if blob_mask.any():
        ref = float(np.percentile(np.abs(raw[blob_mask]), 99.9)) or 1.0
        s_norm[blob_mask] = np.clip(np.abs(raw[blob_mask]) / ref, 0.0, 1.2)
    if (~blob_mask).any():
        s_norm[~blob_mask] = np.clip(raw[~blob_mask] / REGION_STRENGTH_MM, 0.0, 1.0)
    rank = np.argsort(s_norm, kind="stable")
    prio = np.empty(len(cands), dtype=np.float64)
    prio[rank] = np.arange(len(cands), dtype=np.float64) / max(len(cands) - 1, 1)

    pts_mm = np.asarray([(c[0][0] * spacing[0], c[0][1] * spacing[1], c[0][2] * spacing[2])
                        for c in cands], dtype=np.float64)

    # Level assignment: within each supervoxel the single strongest candidate
    # (any kind) becomes coarse; that tiles the body with one coarse mark per
    # ~SUPERVOXEL_MM region and keeps tumour interiors represented.  A coarse
    # mark stands for its whole supervoxel, so its scale (and thus its slice
    # visibility half-width max(scale/2, 2)) is at least the supervoxel's
    # equivalent diameter — otherwise abundant strong small-σ blobs win
    # supervoxels and vanish from every slice but their own (measured on M33).
    # All other candidates are fine-thinned (FINE_SPACING_MM mutual, clearance
    # from coarse).
    t0 = time.monotonic()
    # supervoxel membership for level assignment; sv id 0 (= outside/none) is
    # ignored by the grouping, so volumes without supervoxels (tiny bodies) fall
    # back to all-fine marks instead of crashing the partition step
    sv_ids = np.asarray([_supervoxel_at(c[0], labels_r, fr) if has_sv else 0
                         for c in cands], dtype=np.int64)
    coarse_idx = _per_group_strongest(sv_ids, prio)
    coarse_scale: dict[int, float] = {}
    if has_sv:
        sizes_ds = np.bincount(labels_r.ravel())
        sv_eq_mm = 2.0 * (3.0 * sizes_ds * ds_r_mm ** 3 / (4.0 * np.pi)) ** (1.0 / 3.0)
        coarse_scale = {int(sv): float(np.clip(sv_eq_mm[sv], *REGION_SCALE_CLAMP_MM))
                        for sv in sv_ids[coarse_idx] if sv > 0}
    fine_pool = np.setdiff1d(np.arange(len(cands), dtype=np.int64), coarse_idx)
    keep_fine_local = _greedy_spatial_thin(
        pts_mm[fine_pool], prio[fine_pool], FINE_SPACING_MM,
        extra=(pts_mm[coarse_idx], FINE_CLEAR_COARSE_MM) if len(coarse_idx) else None)
    fine_idx = fine_pool[keep_fine_local]
    _t("thin", t0)

    # stable IDs: coarse first, each sorted by k, then j, then i
    def _key(ci: int) -> tuple:
        return (cands[ci][0][2], cands[ci][0][1], cands[ci][0][0])

    coarse_sorted = sorted(coarse_idx.tolist(), key=_key)
    fine_sorted = sorted(fine_idx.tolist(), key=_key)
    marks: dict[int, Mark] = {}
    strengths: dict[int, float] = {}
    nid = 0
    for level, group in (("coarse", coarse_sorted), ("fine", fine_sorted)):
        for ci in group:
            nid += 1
            ijk, scale_mm, kind, _ = cands[ci]
            if level == "coarse":
                scale_mm = max(scale_mm, coarse_scale.get(int(sv_ids[ci]), 0.0))
            marks[nid] = Mark(nid, kind, ijk, float(scale_mm), level)
            strengths[nid] = float(prio[ci])

    return MarkSet(marks, spacing, win, strengths, params=_params_snapshot(), shape=shape)


def _params_snapshot() -> dict:
    return {
        "version": MARKS_VERSION,
        "downsample_mm": DOWNSAMPLE_MM, "region_downsample_mm": REGION_DOWNSAMPLE_MM,
        "log_sigmas_mm": list(LOG_SIGMAS_MM),
        "log_rel_thr": LOG_REL_THRESHOLD, "log_abs_thr": LOG_ABS_THRESHOLD,
        "body_erode_mm": BODY_ERODE_MM, "supervoxel_mm": SUPERVOXEL_MM,
        "supervoxel_compactness": SUPERVOXEL_COMPACTNESS,
        "supervoxel_max_iter": SUPERVOXEL_MAX_ITER,
        "supervoxel_min_mm3": SUPERVOXEL_MIN_MM3,
        "fine_spacing_mm": FINE_SPACING_MM,
        "fine_clear_coarse_mm": FINE_CLEAR_COARSE_MM,
        "region_strength_mm": REGION_STRENGTH_MM,
        "region_scale_clamp_mm": list(REGION_SCALE_CLAMP_MM),
        "body_grid_max_dim": BODY_GRID_MAX_DIM,
    }


def cache_matches(ms: "MarkSet", shape) -> bool:
    """True when a loaded mark cache was computed with the CURRENT version/params
    (full fingerprint) on a volume of ``shape`` — otherwise it must be recomputed."""
    current = json.loads(json.dumps(_params_snapshot()))
    return ms.params == current and tuple(ms.shape or ()) == tuple(int(v) for v in shape)


# ---------------------------------------------------------------------------
# Edge marks (per-render, in-plane)
# ---------------------------------------------------------------------------
EDGE_PREFER_MM = 15.0   # candidates this close to the mask boundary rank first


def edge_marks_2d(slice2d: np.ndarray, mask2d_body: np.ndarray, box2d: tuple | None = None,
                  max_n: int = 40, min_spacing_mm: float = 5.0,
                  prefer_mask2d: np.ndarray | None = None,
                  prefer_mm: float = EDGE_PREFER_MM) -> list[tuple[int, int]]:
    """In-plane (row, col) gradient-magnitude maxima inside the body and ``box2d``.

    ``prefer_mask2d`` (the agent's current mask on this slice, when non-empty)
    only re-ranks: candidates within ``prefer_mm`` of its boundary (either side)
    come first, the rest still qualify.  The mask never restricts the search.

    ``slice2d``/``mask2d_body`` are display-oriented 2D arrays (CONTRACTS §2).
    Deterministic: candidates are local maxima of the Sobel magnitude, thinned
    greedily at ``min_spacing_mm`` (1 mm isotropic in-plane), strongest first.
    """
    s = np.asarray(slice2d, dtype=np.float32)
    m = np.asarray(mask2d_body).astype(bool)
    r0, c0, r1, c1 = (0, 0, s.shape[0], s.shape[1]) if box2d is None else \
        (max(0, int(box2d[0])), max(0, int(box2d[1])), min(s.shape[0], int(box2d[2])),
         min(s.shape[1], int(box2d[3])))
    if r1 <= r0 or c1 <= c0 or not m.any():
        return []
    vals = s[m]
    lo, hi = np.percentile(vals, (1.0, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    w = np.clip((s - lo) / (hi - lo), 0.0, 1.0)
    gy, gx = np.gradient(ndi.gaussian_filter(w, 0.8))
    gm = np.hypot(gx, gy)
    allowed = m & (gm >= max(0.04, 0.25 * float(np.percentile(gm[m], 95.0))))
    box = np.zeros(s.shape, dtype=bool)
    box[r0:r1, c0:c1] = True
    cand = allowed & (gm == ndi.maximum_filter(gm, size=3)) & box
    idx = np.argwhere(cand)
    if len(idx) == 0:
        return []
    strength = gm[cand]
    near = np.zeros(len(idx), dtype=bool)
    if prefer_mask2d is not None:
        pm = np.asarray(prefer_mask2d).astype(bool)
        if pm.any():
            bnd = pm & ~ndi.binary_erosion(pm, structure=np.ones((3, 3), bool))
            dist = ndi.distance_transform_edt(~bnd)
            near = dist[idx[:, 0], idx[:, 1]] <= prefer_mm
    # near-boundary candidates outrank every other candidate (gm is < 2 everywhere)
    rank = strength + 2.0 * near
    keep = _greedy_spatial_thin(idx.astype(np.float64), rank, min_spacing_mm)
    sel = idx[keep]
    st = rank[keep]
    srt = np.lexsort((sel[:, 1], sel[:, 0], -st))  # near first, strongest first, row/col tie-break
    return [(int(r), int(c)) for r, c in sel[srt][:max_n]]
