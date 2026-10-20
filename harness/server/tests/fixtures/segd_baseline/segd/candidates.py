"""Candidate corrections for a segmentation mask (GT-free).

``propose(image, mask, spacing, method=...)`` returns regions the agent may ADD to or REMOVE from its
current mask. Everything is computed from the image, the current mask and (for ``probe``) nnInteractive;
this module never reads ground truth (see tests/test_candidates.py::test_no_gt_access).

Methods
-------
``intensity``   robust intensity band of the mask interior; grow outside the mask (<= D mm, inside the
                body, connected to the mask), split into components and distance shells.
``supervoxel``  SLIC supervoxels; uncovered part of partly covered supervoxels + adjacent supervoxels whose
                mean/sd match the interior (add); mask parts in barely covered supervoxels (remove).
``ends``        per axis end of the mask: does the footprint's intensity band continue on the next slices?
                -> continuation slab (add).
``combo``       intensity + supervoxel + ends pooled, de-duplicated, ranked by volume.
``probe``       nnInteractive what-if: feed the current mask as initial segmentation, click a probe seed,
                region = prediction - mask (add) or mask - prediction (remove). Needs ``nni`` (see NNIProbe).

All array indices are [i, j, k] voxels; ``spacing`` in mm. Deterministic (probe: up to GPU nondeterminism).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import slic

from segd import config
from segd import marks as M

PROBE_UPLOAD_MAX_PACKED_BYTES = 60 * 1024 * 1024   # same rule as config.UPLOAD_MAX_PACKED_BYTES
PROBE_CROP_MARGIN_VOX = 5

Box = tuple[slice, slice, slice]


# ----------------------------------------------------------------------------------------------
# Candidate
# ----------------------------------------------------------------------------------------------
@dataclass
class Candidate:
    id: str
    kind: str                    # 'add' | 'remove'
    bbox: Box                    # full-volume slices that contain ``crop``
    crop: np.ndarray             # bool, region inside ``bbox``
    shape: tuple[int, int, int]  # full-volume shape
    volume_ml: float
    how: dict[str, Any]          # method + seed info (probe: seed voxel + polarity, reproducible as a click)
    score: float = 0.0           # ranking score (higher = shown first)

    @property
    def region(self) -> np.ndarray:
        """Full-volume bool mask of the voxels that would change."""
        out = np.zeros(self.shape, dtype=bool)
        out[self.bbox] = self.crop
        return out


def _box_of(region: np.ndarray, pad: int = 0) -> Box | None:
    idx = np.nonzero(region.any(axis=(1, 2)))[0]
    if idx.size == 0:
        return None
    jdx = np.nonzero(region.any(axis=(0, 2)))[0]
    kdx = np.nonzero(region.any(axis=(0, 1)))[0]
    return tuple(slice(max(int(a[0]) - pad, 0), min(int(a[-1]) + 1 + pad, n))
                 for a, n in zip((idx, jdx, kdx), region.shape))


def _make(kind: str, region_full_or_crop: np.ndarray, offset: tuple[int, int, int], shape, spacing,
          how: dict, score: float | None = None) -> Candidate | None:
    """Build a Candidate from a bool array living at ``offset`` (crop -> full voxel offset)."""
    sub = _box_of(region_full_or_crop)
    if sub is None:
        return None
    crop = region_full_or_crop[sub]
    bbox = tuple(slice(s.start + o, s.stop + o) for s, o in zip(sub, offset))
    vox = float(np.prod(spacing))
    vol = float(crop.sum()) * vox / 1000.0
    return Candidate("", kind, bbox, np.ascontiguousarray(crop), tuple(int(n) for n in shape), vol,
                     how, vol if score is None else float(score))


# ----------------------------------------------------------------------------------------------
# Working context (crop around the mask, smoothed image, body, interior band)
# ----------------------------------------------------------------------------------------------
class _Ctx:
    def __init__(self, image: np.ndarray, mask: np.ndarray, spacing, cache: dict | None):
        self.image = image
        self.mask = np.asarray(mask).astype(bool)
        self.spacing = tuple(float(s) for s in spacing)
        self.shape = tuple(int(n) for n in image.shape)
        self.cache = cache if cache is not None else {}
        self.mask_ml = float(self.mask.sum()) * float(np.prod(self.spacing)) / 1000.0
        self._body = None

    @property
    def body(self) -> np.ndarray:
        if self._body is None:
            if "body" not in self.cache:
                b = M.body_mask(self.image)
                self.cache["body"] = b if b.any() else np.ones(self.shape, dtype=bool)
            self._body = self.cache["body"]
        return self._body

    def crop_box(self, margin_mm: float) -> Box:
        bb = _box_of(self.mask)
        out = []
        for s, n, sp in zip(bb, self.shape, self.spacing):
            m = int(np.ceil(margin_mm / sp))
            out.append(slice(max(s.start - m, 0), min(s.stop + m, n)))
        return tuple(out)

    def smoothed(self, box: Box, sigma_mm: float = 1.0) -> np.ndarray:
        """Gaussian-smoothed image on ``box`` (computed on box + 3 sigma halo to avoid edge effects)."""
        sig = [sigma_mm / sp for sp in self.spacing]
        halo = [int(np.ceil(3 * s)) for s in sig]
        hb = tuple(slice(max(s.start - h, 0), min(s.stop + h, n))
                   for s, h, n in zip(box, halo, self.shape))
        sm = ndi.gaussian_filter(self.image[hb].astype(np.float32), sig)
        inner = tuple(slice(s.start - h.start, s.stop - h.start) for s, h in zip(box, hb))
        return sm[inner]


def _ball(radius_vox: tuple[float, float, float]) -> np.ndarray:
    r = [max(int(np.floor(v)), 0) for v in radius_vox]
    grids = np.ogrid[tuple(slice(-n, n + 1) for n in r)]
    d = sum((g / max(v, 1e-6)) ** 2 for g, v in zip(grids, radius_vox))
    st = d <= 1.0 + 1e-9
    return st


def _erode_mm(mask: np.ndarray, spacing, mm: float) -> np.ndarray:
    st = _ball(tuple(mm / sp for sp in spacing))
    if st.sum() <= 1:
        return mask.copy()
    return ndi.binary_erosion(mask, structure=st)


def _interior(mask_c: np.ndarray, spacing) -> np.ndarray:
    for mm, need in ((2.0, 200), (1.0, 50)):
        e = _erode_mm(mask_c, spacing, mm)
        if int(e.sum()) >= need:
            return e
    return mask_c


def _band(sm_c: np.ndarray, interior: np.ndarray, lo: float, hi: float, pad: float
          ) -> tuple[float, float, float, float]:
    """(band_lo, band_hi, median, sd) of the smoothed interior; percentile band widened by ``pad`` x width."""
    v = sm_c[interior].astype(np.float64)
    blo, bhi = np.percentile(v, (lo, hi))
    w = bhi - blo
    return float(blo - pad * w), float(bhi + pad * w), float(np.median(v)), float(v.std())


def _interior_point(region: np.ndarray, spacing) -> tuple[int, int, int]:
    """Voxel of a bool array that is deepest inside it (EDT max)."""
    bb = _box_of(region)
    sub = np.pad(region[bb], 1)
    dt = ndi.distance_transform_edt(sub, sampling=spacing)
    p = np.unravel_index(int(np.argmax(dt)), dt.shape)
    return tuple(int(p[a] - 1 + bb[a].start) for a in range(3))


def _components(region: np.ndarray, min_vox: int, limit: int | None = None) -> list[np.ndarray]:
    """Connected components of ``region`` above ``min_vox``, LARGEST FIRST.  Only the top
    ``limit`` are materialized: every materialization is a region-sized allocation, so with
    many qualifying components the small ones (which can never make the final max_n cut)
    must not be built at all (fix-round performance bound)."""
    lab, n = ndi.label(region)
    if n == 0:
        return []
    sizes = np.bincount(lab.ravel())
    objs = ndi.find_objects(lab)
    out = []
    for i in np.argsort(-sizes[1:], kind="stable"):     # component labels by size, descending
        if sizes[i + 1] < min_vox:
            break                                       # sorted: everything after is smaller
        if limit is not None and len(out) >= limit:
            break
        sl = objs[i]
        comp = np.zeros(region.shape, dtype=bool)
        comp[sl] = lab[sl] == i + 1
        out.append(comp)
    return out


def _split_shells(comp: np.ndarray, dist_mm: np.ndarray, shell_mm: float, min_vox: int,
                  min_extent_factor: float = 1.5) -> list[tuple[np.ndarray, int]]:
    """Split a component into connected pieces per distance-from-mask shell when it reaches further than
    ``min_extent_factor * shell_mm``. Returns [(piece, shell_index)]."""
    dmax = float(dist_mm[comp].max())
    if dmax <= min_extent_factor * shell_mm:
        return [(comp, 0)]
    shell = np.floor(dist_mm / shell_mm).astype(np.int32)
    out = []
    for s in range(int(shell[comp].max()) + 1):
        for piece in _components(comp & (shell == s), min_vox):
            out.append((piece, s))
    return out


# ----------------------------------------------------------------------------------------------
# (a) intensity growth
# ----------------------------------------------------------------------------------------------
# Bound for many-component regions (fix round #7): rank by size first and materialize only
# the top few — 2 x the ops-layer candidate cap (config.CANDIDATE_MAX_N) is generous, since
# _finalize never returns more than max_n.
_MATERIALIZE_TOP = 2 * config.CANDIDATE_MAX_N


def _gen_intensity(ctx: _Ctx, *, D_mm: float = 30.0, lo: float = 2.0, hi: float = 98.0, pad: float = 0.3,
                   shell_mm: float = 10.0, open_mm: float = 0.0, min_piece_ml: float = 0.2, **_) -> list[Candidate]:
    box = ctx.crop_box(D_mm + 5.0)
    off = tuple(s.start for s in box)
    mask_c = ctx.mask[box]
    body_c = ctx.body[box]
    sm = ctx.smoothed(box)
    blo, bhi, _, _ = _band(sm, _interior(mask_c, ctx.spacing), lo, hi, pad)
    dist = ndi.distance_transform_edt(~mask_c, sampling=ctx.spacing)
    cand = (sm >= blo) & (sm <= bhi) & body_c & ~mask_c & (dist <= D_mm)
    if open_mm > 0:
        st = _ball(tuple(open_mm / sp for sp in ctx.spacing))
        if st.sum() > 1:
            cand = ndi.binary_opening(cand, structure=st)
    min_vox = max(int(min_piece_ml * 1000.0 / np.prod(ctx.spacing)), 1)
    touch = ndi.binary_dilation(mask_c)
    out = []
    for comp in _components(cand, min_vox, limit=_MATERIALIZE_TOP):
        if not (comp & touch).any():
            continue
        for piece, shell in _split_shells(comp, dist, shell_mm, min_vox):
            seed = _interior_point(piece, ctx.spacing)
            c = _make("add", piece, off, ctx.shape, ctx.spacing,
                      {"method": "intensity", "seed_ijk": tuple(int(a + o) for a, o in zip(seed, off)),
                       "shell": shell, "band": (round(blo, 2), round(bhi, 2))})
            if c is not None:
                out.append(c)
    return out


# ----------------------------------------------------------------------------------------------
# (b) supervoxels
# ----------------------------------------------------------------------------------------------
def _supervoxels(ctx: _Ctx, box: Box, sm: np.ndarray, sv_mm: float, ds_mm: float, compactness: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """SLIC on a downsampled grid. Returns (labels at full crop resolution, labels at ds resolution)."""
    key = ("sv", tuple((s.start, s.stop) for s in box), sv_mm, ds_mm, compactness)
    if key in ctx.cache:
        return ctx.cache[key]
    f = tuple(max(1, int(round(ds_mm / sp))) for sp in ctx.spacing)
    body_c = ctx.body[box]
    dsl = tuple(slice(0, None, a) for a in f)
    dbody = body_c[dsl]
    dimg = sm[dsl]
    body_vals = dimg[dbody]
    lo, hi = np.percentile(body_vals, (0.5, 99.5)) if body_vals.size else (0.0, 1.0)
    dnorm = np.clip((dimg - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
    ds_vox_mm = float(np.mean([sp * a for sp, a in zip(ctx.spacing, f)]))
    vox_per_sv = (sv_mm ** 3) * np.pi / 6.0 / ds_vox_mm ** 3
    n_seg = int(np.clip(dbody.sum() / max(vox_per_sv, 1.0), 4, 60000))
    labels = slic(dnorm, n_segments=n_seg, compactness=compactness, max_num_iter=5, start_label=1,
                  mask=dbody, channel_axis=None)
    idx = [np.minimum(np.arange(n) // a, ns - 1) for n, a, ns in zip(body_c.shape, f, labels.shape)]
    full = labels[np.ix_(*idx)]
    full = np.where(body_c, full, 0)
    ctx.cache[key] = (full, labels)
    return full, labels


def _adjacent_pairs(labels: np.ndarray) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for ax in range(3):
        a = np.moveaxis(labels, ax, 0)
        x, y = a[:-1].ravel(), a[1:].ravel()
        d = (x != y) & (x > 0) & (y > 0)
        if d.any():
            pr = np.unique(np.stack([np.minimum(x[d], y[d]), np.maximum(x[d], y[d])], 1), axis=0)
            pairs.update((int(p), int(q)) for p, q in pr)
    return pairs


def _gen_supervoxel(ctx: _Ctx, *, sv_mm: float = 14.0, ds_mm: float = 3.0, compactness: float = 0.1,
                    cov_lo: float = 0.15, cov_hi: float = 0.85, margin_mm: float = 40.0, hops: int = 2,
                    lo: float = 2.0, hi: float = 98.0, pad: float = 0.3, sd_ratio: float = 1.5,
                    shell_mm: float = 10.0, min_piece_ml: float = 0.2, **_) -> list[Candidate]:
    box = ctx.crop_box(margin_mm)
    off = tuple(s.start for s in box)
    mask_c = ctx.mask[box]
    sm = ctx.smoothed(box)
    labels, labels_ds = _supervoxels(ctx, box, sm, sv_mm, ds_mm, compactness)
    nl = int(labels.max())
    if nl < 1:
        return []
    lf = labels.ravel()
    tot = np.bincount(lf, minlength=nl + 1).astype(np.float64)
    ins = np.bincount(lf[mask_c.ravel()], minlength=nl + 1).astype(np.float64)
    cov = np.divide(ins, tot, out=np.zeros_like(ins), where=tot > 0)
    cov[0] = 1.0
    ids = np.arange(nl + 1)
    mean = ndi.mean(sm, labels, ids)
    sd = np.sqrt(np.maximum(ndi.mean(sm * sm, labels, ids) - np.asarray(mean) ** 2, 0.0))
    inter = _interior(mask_c, ctx.spacing)
    blo, bhi, _, isd = _band(sm, inter, lo, hi, pad)

    partial = (cov >= cov_lo) & (cov <= cov_hi) & (tot > 0)
    picked = np.zeros(nl + 1, dtype=bool)
    support = (cov >= cov_lo) & (tot > 0)          # supervoxels the mask 'owns' (>= cov_lo covered)
    similar = (mean >= blo) & (mean <= bhi) & (sd <= sd_ratio * max(isd, 1e-6)) & (cov < cov_lo) & (tot > 0)
    similar[0] = False
    pairs = _adjacent_pairs(labels_ds)
    nbr: dict[int, set[int]] = {}
    for p, q in pairs:
        nbr.setdefault(p, set()).add(q)
        nbr.setdefault(q, set()).add(p)
    frontier = set(np.nonzero(support)[0].tolist())
    for _ in range(max(hops, 0)):
        new = {q for p in frontier for q in nbr.get(p, ()) if similar[q] and not picked[q]}
        if not new:
            break
        picked[list(new)] = True
        frontier = new

    add_region = (partial | picked)[labels] & ~mask_c
    add_region &= ctx.body[box]
    dist = ndi.distance_transform_edt(~mask_c, sampling=ctx.spacing)
    min_vox = max(int(min_piece_ml * 1000.0 / np.prod(ctx.spacing)), 1)
    out = []
    for comp in _components(add_region, min_vox, limit=_MATERIALIZE_TOP):
        for piece, shell in _split_shells(comp, dist, shell_mm, min_vox):
            seed = _interior_point(piece, ctx.spacing)
            c = _make("add", piece, off, ctx.shape, ctx.spacing,
                      {"method": "supervoxel", "seed_ijk": tuple(int(a + o) for a, o in zip(seed, off)),
                       "shell": shell})
            if c is not None:
                out.append(c)
    weak = (cov < cov_lo) & (tot > 0)
    rem_region = mask_c & weak[labels]
    for comp in _components(rem_region, min_vox, limit=_MATERIALIZE_TOP):
        seed = _interior_point(comp, ctx.spacing)
        c = _make("remove", comp, off, ctx.shape, ctx.spacing,
                  {"method": "supervoxel", "seed_ijk": tuple(int(a + o) for a, o in zip(seed, off))})
        if c is not None:
            out.append(c)
    return out


# ----------------------------------------------------------------------------------------------
# (e) ends: continuation of the end-slice footprint
# ----------------------------------------------------------------------------------------------
_AXIS_NAME = {0: ("left", "right"), 1: ("posterior", "anterior"), 2: ("inferior", "superior")}


def _gen_ends(ctx: _Ctx, *, max_ext_mm: float = 40.0, lo: float = 2.0, hi: float = 98.0, pad: float = 0.3,
              seed_slab_mm: float = 3.0, grow_mm: float = 2.0, min_area_frac: float = 0.3,
              min_ext_mm: float = 2.0, min_piece_ml: float = 0.2, axes: tuple = (0, 1, 2), **_) -> list[Candidate]:
    box = ctx.crop_box(max_ext_mm + 5.0)
    off = tuple(s.start for s in box)
    mask_c = ctx.mask[box]
    body_c = ctx.body[box]
    sm = ctx.smoothed(box)
    blo, bhi, _, _ = _band(sm, _interior(mask_c, ctx.spacing), lo, hi, pad)
    inband = (sm >= blo) & (sm <= bhi) & body_c
    out = []
    for ax in axes:
        sp_ax = ctx.spacing[ax]
        others = [a for a in range(3) if a != ax]
        in_sp = tuple(ctx.spacing[a] for a in others)
        grow_st = _ball(tuple(max(grow_mm * max(1.0, 1.0) / s, 1.0) for s in in_sp))
        n = mask_c.shape[ax]
        occ = np.nonzero(mask_c.any(axis=tuple(others)))[0]
        if occ.size == 0:
            continue
        for side, direction in ((0, -1), (1, +1)):
            end = int(occ[-1] if direction > 0 else occ[0])
            k = max(int(round(seed_slab_mm / sp_ax)), 1)
            slab = [end - direction * t for t in range(k)]
            slab = [s for s in slab if 0 <= s < n]
            foot = np.zeros([mask_c.shape[a] for a in others], dtype=bool)
            for s in slab:
                foot |= np.take(mask_c, s, axis=ax)
            ref_area = float(foot.sum())
            if ref_area == 0:
                continue
            prev = foot
            slices: list[tuple[int, np.ndarray]] = []
            for t in range(1, int(max_ext_mm / sp_ax) + 1):
                s = end + direction * t
                if not 0 <= s < n:
                    break
                allowed = np.take(inband, s, axis=ax) & ~np.take(mask_c, s, axis=ax)
                reach = ndi.binary_dilation(prev, structure=grow_st)
                reg = allowed & reach
                reg = ndi.binary_opening(reg) if reg.sum() > 20 else reg
                if reg.sum() < min_area_frac * ref_area:
                    break
                slices.append((s, reg))
                prev = reg
            if len(slices) * sp_ax < min_ext_mm:
                continue
            vol = np.zeros(mask_c.shape, dtype=bool)
            for s, reg in slices:
                idx = [slice(None)] * 3
                idx[ax] = s
                vol[tuple(idx)] = reg
            min_vox = max(int(min_piece_ml * 1000.0 / np.prod(ctx.spacing)), 1)
            comps = _components(vol, min_vox, limit=1)   # only the largest piece is used
            if not comps:
                continue
            comp = comps[0]                       # connected through the slices; keep the largest piece
            seed = _interior_point(comp, ctx.spacing)
            c = _make("add", comp, off, ctx.shape, ctx.spacing,
                      {"method": "ends", "axis": ax, "side": _AXIS_NAME[ax][side],
                       "extent_mm": round(len(slices) * sp_ax, 1),
                       "seed_ijk": tuple(int(a + o) for a, o in zip(seed, off))})
            if c is not None:
                out.append(c)
    return out


# ----------------------------------------------------------------------------------------------
# (d) probe: nnInteractive what-if
# ----------------------------------------------------------------------------------------------
class NNIProbe:
    """Scratch nnInteractive remote session for what-if prompts (one ``set_image`` per case).

    ``probe(mask, seed_ijk, include)`` resets the session, feeds ``mask`` as the initial segmentation,
    places one point prompt and returns the predicted mask (bool, full volume) plus wall seconds.
    Uploads the full float16 volume, or the body bbox + margin if its packed size exceeds
    PROBE_UPLOAD_MAX_PACKED_BYTES (same rule as segd.case_state).
    """

    def __init__(self, url: str, image: np.ndarray, spacing, body: np.ndarray | None = None):
        from nnInteractive.inference.remote import nnInteractiveRemoteInferenceSession
        from nnInteractive.inference.remote.serialization import pack_array
        self.shape = tuple(int(n) for n in image.shape)
        img16 = np.ascontiguousarray(image, dtype=np.float16)
        self.offset = (0, 0, 0)
        if len(pack_array(img16[None])) > PROBE_UPLOAD_MAX_PACKED_BYTES and body is not None and body.any():
            bb = _box_of(body, PROBE_CROP_MARGIN_VOX)
            self.offset = tuple(s.start for s in bb)
            img16 = np.ascontiguousarray(img16[bb])
        self.crop_shape = tuple(int(n) for n in img16.shape)
        self.sl = tuple(slice(o, o + n) for o, n in zip(self.offset, self.crop_shape))
        self.session = nnInteractiveRemoteInferenceSession(url, read_timeout=300, write_timeout=600)
        self.session.set_image(img16[None], image_properties={"spacing": [float(s) for s in spacing]})
        self.session.set_target_buffer(np.zeros(self.crop_shape, dtype=np.uint8))
        self.set_image_seconds = 0.0

    def probe(self, mask: np.ndarray, seed_ijk, include: bool) -> tuple[np.ndarray, float]:
        t0 = time.monotonic()
        s = self.session
        s.reset_interactions()
        m = np.ascontiguousarray(mask[self.sl], dtype=np.uint8)
        if m.any():
            s.add_initial_seg_interaction(m, run_prediction=False)
        p = [int(c) - int(o) for c, o in zip(seed_ijk, self.offset)]
        s.add_point_interaction(p, include_interaction=bool(include))
        out = np.zeros(self.shape, dtype=bool)
        out[self.sl] = np.asarray(s.target_buffer) > 0
        return out, time.monotonic() - t0

    def close(self) -> None:
        try:
            self.session.close()
        except Exception:
            pass


def _beyond_end_seeds(ctx: _Ctx, dist_mm=(5.0, 10.0, 15.0)) -> list[tuple[tuple, dict]]:
    """For each of the 6 mask ends: first of (10, 5, 15) mm beyond the end, at the in-plane centroid of the
    end footprint, that lies inside the body."""
    out = []
    bb = _box_of(ctx.mask)
    for ax in range(3):
        others = [a for a in range(3) if a != ax]
        for side, direction in ((0, -1), (1, +1)):
            end = bb[ax].stop - 1 if direction > 0 else bb[ax].start
            idx = [slice(None)] * 3
            idx[ax] = end
            foot = ctx.mask[tuple(idx)]
            c = ndi.center_of_mass(foot)
            for d in (dist_mm[1], dist_mm[0], dist_mm[2]):
                s = end + direction * int(round(d / ctx.spacing[ax]))
                if not 0 <= s < ctx.shape[ax]:
                    continue
                p = [0, 0, 0]
                p[ax] = s
                p[others[0]], p[others[1]] = int(round(c[0])), int(round(c[1]))
                if ctx.body[tuple(p)]:
                    out.append((tuple(p), {"src": f"beyond:{_AXIS_NAME[ax][side]}", "dist_mm": d}))
                    break
    return out


def _remove_seeds(ctx: _Ctx, max_n: int = 4, min_ml: float = 1.0) -> list[tuple[tuple, dict]]:
    """Interior points of mask parts far from the core: secondary connected components and protrusions
    (mask minus its opening with radius max(3 mm, 0.35 x max EDT))."""
    box = ctx.crop_box(2.0)
    off = tuple(s.start for s in box)
    m = ctx.mask[box]
    min_vox = max(int(min_ml * 1000.0 / np.prod(ctx.spacing)), 1)
    parts = []
    lab, n = ndi.label(m)
    if n > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        main = int(np.argmax(sizes))
        for comp in _components(m & (lab != main), min_vox, limit=_MATERIALIZE_TOP):
            parts.append((comp, "component"))
    dt = ndi.distance_transform_edt(m, sampling=ctx.spacing)
    r = max(3.0, 0.35 * float(dt.max()))
    core = ndi.binary_opening(m, structure=_ball(tuple(r / sp for sp in ctx.spacing)))
    for comp in _components(m & ~core & (lab == (int(np.argmax(np.bincount(lab.ravel())[1:])) + 1)), min_vox,
                            limit=_MATERIALIZE_TOP):
        parts.append((comp, "protrusion"))
    parts.sort(key=lambda p: -int(p[0].sum()))
    out = []
    for comp, src in parts[:max_n]:
        seed = _interior_point(comp, ctx.spacing)
        out.append((tuple(int(a + o) for a, o in zip(seed, off)), {"src": f"lobe:{src}"}))
    return out


def _gen_probe(ctx: _Ctx, *, nni, max_probes: int = 16, seeds_per_method: int = 3, min_seed_gap_mm: float = 5.0,
               seed_methods: tuple = ("intensity", "supervoxel", "ends"), do_remove: bool = True,
               **params) -> list[Candidate]:
    if nni is None:
        raise ValueError("method='probe' needs nni=NNIProbe(...) (or an object with .probe(mask, ijk, include))")
    seeds: list[tuple[tuple, bool, dict]] = []
    for meth in seed_methods:
        pm = _GENERATORS[meth](ctx, **params)
        adds = sorted([c for c in pm if c.kind == "add"], key=lambda c: -c.score)[:seeds_per_method]
        for c in adds:
            seeds.append((tuple(c.how["seed_ijk"]), True, {"src": f"{meth}"}))
    for p, info in _beyond_end_seeds(ctx):
        seeds.append((p, True, info))
    if do_remove:
        for p, info in _remove_seeds(ctx):
            seeds.append((p, False, info))
    kept: list[tuple[tuple, bool, dict]] = []
    for p, inc, info in seeds:                      # greedy de-duplication of nearby same-polarity seeds
        if any(inc == q[1] and np.linalg.norm((np.asarray(p) - np.asarray(q[0])) * np.asarray(ctx.spacing))
               < min_seed_gap_mm for q in kept):
            continue
        kept.append((p, inc, info))
    kept = kept[:max_probes]
    out = []
    for p, inc, info in kept:
        pred, secs = nni.probe(ctx.mask, p, inc)
        delta = (pred & ~ctx.mask) if inc else (ctx.mask & ~pred)
        bb = _box_of(delta)
        if bb is None:
            continue
        c = _make("add" if inc else "remove", delta[bb], tuple(s.start for s in bb), ctx.shape, ctx.spacing,
                  {"method": "probe", "seed_ijk": tuple(int(a) for a in p), "include": bool(inc),
                   "latency_s": round(secs, 3), **info})
        if c is not None:
            out.append(c)
    return out


# ----------------------------------------------------------------------------------------------
# post-processing and entry point
# ----------------------------------------------------------------------------------------------
def _overlap_vox(a: Candidate, b: Candidate) -> int:
    lo = [max(x.start, y.start) for x, y in zip(a.bbox, b.bbox)]
    hi = [min(x.stop, y.stop) for x, y in zip(a.bbox, b.bbox)]
    if any(h <= l for l, h in zip(lo, hi)):
        return 0
    ia = tuple(slice(l - x.start, h - x.start) for l, h, x in zip(lo, hi, a.bbox))
    ib = tuple(slice(l - y.start, h - y.start) for l, h, y in zip(lo, hi, b.bbox))
    return int((a.crop[ia] & b.crop[ib]).sum())


def _finalize(cands: list[Candidate], ctx: _Ctx, max_n: int, min_ml: float | None, huge_factor: float,
              overlap: float) -> list[Candidate]:
    min_ml = min(0.5, 0.05 * ctx.mask_ml) if min_ml is None else min_ml
    keep = [c for c in cands if c.volume_ml >= min_ml and c.volume_ml <= huge_factor * ctx.mask_ml]
    # deterministic order: score desc, then position
    keep.sort(key=lambda c: (-c.score, c.kind, tuple(s.start for s in c.bbox)))
    out: list[Candidate] = []
    for c in keep:
        dup = False
        for k in out:
            if k.kind != c.kind:
                continue
            ov = _overlap_vox(c, k)
            if ov > overlap * min(c.crop.sum(), k.crop.sum()):
                dup = True
                break
        if not dup:
            out.append(c)
    out = out[:max_n]
    for i, c in enumerate(out, 1):
        c.id = f"C{i}"
    return out


def _gen_combo(ctx: _Ctx, **params) -> list[Candidate]:
    """intensity + supervoxel + ends pooled (de-duplicated and ranked by the common _finalize step)."""
    out: list[Candidate] = []
    for name in ("intensity", "supervoxel", "ends"):
        out += _GENERATORS[name](ctx, **params)
    return out


_GENERATORS: dict[str, Callable[..., list[Candidate]]] = {
    "intensity": _gen_intensity,
    "supervoxel": _gen_supervoxel,
    "ends": _gen_ends,
    "combo": _gen_combo,
}


def propose(image: np.ndarray, mask: np.ndarray, spacing, *, method: str, nni=None, markset=None,
            max_n: int = 8, min_ml: float | None = None, huge_factor: float = 3.0, overlap: float = 0.5,
            **params) -> list[Candidate]:
    """Candidate corrections for ``mask`` on ``image`` (see module docstring).

    ``markset`` is accepted for API symmetry; the supervoxel method computes its own SLIC grid (the marks code
    does not retain its labels). ``cache`` (dict, optional, in ``params``) memoises the body mask and
    supervoxels across calls on the same image. ``min_ml`` defaults to min(0.5 mL, 5 % of the mask volume);
    candidates larger than ``huge_factor`` x mask volume are dropped; candidates of the same kind overlapping
    > ``overlap`` (fraction of the smaller) are merged keeping the higher score. Empty mask -> [].
    """
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return []
    cache = params.pop("cache", None)
    ctx = _Ctx(np.asarray(image), mask, spacing, cache)
    if method == "probe":
        cands = _gen_probe(ctx, nni=nni, **params)
    elif method in _GENERATORS:
        cands = _GENERATORS[method](ctx, **params)
    else:
        raise ValueError(f"unknown method {method!r}")
    return _finalize(cands, ctx, max_n, min_ml, huge_factor, overlap)


def apply_candidates(mask: np.ndarray, cands: list[Candidate]) -> np.ndarray:
    """Union of 'add' regions, difference of 'remove' regions (an approximation of applying probe candidates
    as sequential clicks)."""
    out = np.asarray(mask).astype(bool).copy()
    for c in cands:
        if c.kind == "add":
            out[c.bbox] |= c.crop
    for c in cands:
        if c.kind == "remove":
            out[c.bbox] &= ~c.crop
    return out


# ----------------------------------------------------------------------------------------------
# Agent-facing description (patient directions only, no coordinates)
# ----------------------------------------------------------------------------------------------
# voxel axis -> (word for decreasing index, word for increasing index); +i Right, +j Anterior, +k Superior
_DIRECTION_WORDS = (("left", "right"), ("posterior", "anterior"), ("inferior", "superior"))
_SIDE_MIN_FRAC = 0.25      # a side is listed when its mask-contact count is >= this fraction of the largest
_SIDE_MAX = 3
_LEVEL_MM = 2.0            # centre offset below this on every axis: "level with the mask"


def describe_candidate(cand: Candidate, mask: np.ndarray, spacing) -> str:
    """'C1 add 52 mL, touches mask on its inferior and posterior sides, mostly superior of the mask'.

    Sides = faces of the candidate whose neighbouring voxel is mask (counted over the candidate's voxels);
    direction = axis with the largest offset between the candidate's and the mask's centre of mass."""
    mask = np.asarray(mask).astype(bool)
    pad = 1
    lo = [max(s.start - pad, 0) for s in cand.bbox]
    hi = [min(s.stop + pad, n) for s, n in zip(cand.bbox, cand.shape)]
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    reg = np.zeros([b - a for a, b in zip(lo, hi)], dtype=bool)
    reg[tuple(slice(s.start - a, s.stop - a) for s, a in zip(cand.bbox, lo))] = cand.crop
    m = mask[box]
    counts = []
    for axis in range(3):
        for step, word in ((-1, _DIRECTION_WORDS[axis][0]), (1, _DIRECTION_WORDS[axis][1])):
            shifted = np.zeros_like(m)
            sl_dst = [slice(None)] * 3
            sl_src = [slice(None)] * 3
            if step > 0:                       # neighbour at +1: shifted[x] = m[x+1]
                sl_dst[axis], sl_src[axis] = slice(0, -1), slice(1, None)
            else:
                sl_dst[axis], sl_src[axis] = slice(1, None), slice(0, -1)
            shifted[tuple(sl_dst)] = m[tuple(sl_src)]
            counts.append((int((reg & shifted).sum()), word))
    top = max(c for c, _ in counts)
    if top == 0:
        touch = "does not touch the mask"
    else:
        sides = [w for c, w in sorted(counts, key=lambda t: -t[0]) if c >= _SIDE_MIN_FRAC * top][:_SIDE_MAX]
        touch = ("touches mask on its " + " and ".join(sides if len(sides) < 3 else [", ".join(sides[:-1]), sides[-1]])
                 + (" side" if len(sides) == 1 else " sides"))
    idx = np.nonzero(cand.crop)
    cc = np.array([float(a.mean()) + s.start for a, s in zip(idx, cand.bbox)])
    midx = np.nonzero(mask)
    mc = np.array([float(a.mean()) for a in midx])
    delta_mm = (cc - mc) * np.asarray(spacing, dtype=float)
    ax = int(np.argmax(np.abs(delta_mm)))
    if abs(delta_mm[ax]) < _LEVEL_MM:
        where = "level with the mask"
    else:
        where = f"mostly {_DIRECTION_WORDS[ax][int(delta_mm[ax] > 0)]} of the mask"
    vol = cand.volume_ml
    vol_txt = f"{vol:.0f}" if vol >= 10 else f"{vol:.1f}"
    return f"{cand.id} {cand.kind} {vol_txt} mL, {touch}, {where}"
