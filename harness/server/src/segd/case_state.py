"""CaseState: per-case daemon state — image, versions (immutable uint8 DAG), active view,
mark resolution, nnInteractive session + sync rule, upload-crop handling.

Array conventions follow segd/data.py: [i, j, k] voxel indices of the canonical image,
1 mm isotropic for MSTT. This module never touches ground truth (CONTRACTS §0).
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx
import nibabel as nib
import numpy as np
from nnInteractive.inference.remote import (
    SessionExpiredError, nnInteractiveRemoteInferenceSession,
)
from nnInteractive.inference.remote.serialization import pack_array

from segd import config


class OpError(Exception):
    """Agent-facing error: shown to the agent as ok:false text (HTTP 200)."""


MODEL_PROMPT_ERROR = "segmentation model error; the mask is unchanged, try again"


def _is_transport_failure(exc: BaseException) -> bool:
    """True for session expiry or connection-level failures: worth one reconnect + retry
    in model_prompt (reviewer finding #2)."""
    return isinstance(exc, (SessionExpiredError, httpx.HTTPError))


def body_mask_fallback(image: np.ndarray) -> np.ndarray:
    """Otsu foreground, largest connected component, holes filled. Local fallback used
    only when `segd.marks.body_mask` is unavailable (e.g. marks module not written yet)."""
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    sub = image[::2, ::2, ::2]
    try:
        t = float(threshold_otsu(sub))
    except ValueError:          # constant image
        return np.ones(image.shape, dtype=bool)
    fg = image > t
    if not fg.any():
        return np.ones(image.shape, dtype=bool)
    labels, n = ndimage.label(fg, structure=ndimage.generate_binary_structure(3, 1))
    if n > 1:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        fg = labels == sizes.argmax()
    return ndimage.binary_fill_holes(fg)


def _load_body_mask(image: np.ndarray) -> np.ndarray:
    try:
        from segd.marks import body_mask  # contract import; marks.py may not exist yet
        return np.asarray(body_mask(image), dtype=bool)
    except ImportError:
        return body_mask_fallback(image)


# ------------------------------------------------------------------ versions
def pack_region(mask: np.ndarray) -> tuple[Optional[tuple], bytes, int]:
    """Compact a full-volume uint8 {0,1} mask: crop to its half-open bbox, packbits.

    Volumes can be 460x460x442 (94 M voxels); bbox cropping plus bit packing keeps an
    all-zero version at 0 bytes and a typical lesion version orders of magnitude below
    the full volume. Returns (bbox ((3,2) or None), packed bytes, nvox inside bbox).
    """
    idx = np.nonzero(mask)
    if idx[0].size == 0:
        return None, b"", 0
    bbox = tuple((int(a.min()), int(a.max()) + 1) for a in idx)
    crop = mask[tuple(slice(lo, hi) for lo, hi in bbox)]
    return bbox, np.packbits(crop.astype(bool), axis=None).tobytes(), int(crop.size)


def unpack_region(bbox, packed: bytes, nvox: int, shape: tuple) -> np.ndarray:
    """Inverse of pack_region: full-volume uint8 mask."""
    out = np.zeros(shape, dtype=np.uint8)
    if bbox is None:
        return out
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))[:nvox]
    out[tuple(slice(lo, hi) for lo, hi in bbox)] = bits.reshape(tuple(hi - lo for lo, hi in bbox))
    return out


@dataclass
class VersionRecord:
    id: int
    parent: Optional[int]
    bbox: Optional[tuple]
    packed: bytes
    nvox: int


@dataclass
class ActiveView:
    axis: str
    slice_idx: int
    view_geom: Any = None
    render_info: Any = None       # segd.render RenderInfo of the last `view` render
    marks_level: str = "none"
    box2d: Optional[tuple] = None  # in-plane (row, col) half-open box when crop == "marks"
    image: Any = None              # the rendered PIL image, kept for v3 (seg_locate uses
    #                                the CURRENT VIEW exactly as the agent saw it); None
    #                                for views created before this field existed


@dataclass
class ResolvedMark:
    id: Any
    ijk: tuple  # full-volume voxel


# ---------------------------------------------------------------- case state
class CaseState:
    """One open case. Ops on one case are serialized by app.py's per-case lock (this
    class carries no lock of its own); nni sessions are closed on finish/close/failure."""

    def __init__(self, case_id: str, image: np.ndarray, affine: np.ndarray,
                 spacing: tuple[float, float, float], target_text: str, sequence: str,
                 out_dir, nni_url: Optional[str] = None, budgets: Optional[dict] = None,
                 orig_geometry: Optional[tuple] = None, v3: Optional[dict] = None,
                 dataset: str = "mstt"):
        self.case_id = case_id
        self.dataset = dataset          # "mstt" (default, unchanged behaviour) or "beamster"
        self.image = np.ascontiguousarray(image, dtype=np.float32)
        self.affine = np.asarray(affine, dtype=np.float64)
        self.spacing = tuple(float(s) for s in spacing)
        self.target_text = target_text
        self.sequence = sequence
        self.out_dir = Path(out_dir)
        self.nni_url = nni_url
        self.budgets = {**config.DEFAULT_BUDGETS, **(budgets or {})}
        # (source-file affine, canonical->source orientation transform) when the source
        # NIfTI needed canonicalisation at load (finding #14); None for RAS sources.
        self.orig_geometry = orig_geometry
        # Harness v3 flags (segd.v3): normalized dict or None (v2 behaviour). Local
        # import: segd.v3 imports case_state helpers at module level.
        from segd.v3 import normalize_v3
        self.v3 = normalize_v3(v3)
        self.v3_fired: set[str] = set()   # checkpoints/injections already delivered
        # ("localize", f"review:{vid}", f"candidates:{vid}:{cid}"; snapshotted and
        # restored by run_op for EVERY op so a failed op can deliver its note on the
        # retry — the note never reached the agent when the op failed)

        self.shape: tuple = self.image.shape
        self.done = False
        self.tool_calls = 0
        self.reviewed_version: int | None = None   # version shown in the last final review
        self.review_code: str | None = None        # confirmation code of that review (ops.op_finish)
        self.model_prompts = 0
        self.versions: dict[int, VersionRecord] = {0: VersionRecord(0, None, None, b"", 0)}
        self.current = 0                      # v0 = empty
        self.markset: Any = None
        self.active_view: Optional[ActiveView] = None
        self.nni: Any = None
        self._nni_dirty = False
        self._nni_seconds = 0.0
        self.crop_offset: tuple = (0, 0, 0)   # session-space origin in full-volume voxels
        self.crop_shape: tuple = self.shape
        self.body: Optional[np.ndarray] = None
        self.body_bbox: Optional[tuple] = None
        self.upload_packed_bytes = 0
        self.log = None                       # TrajectoryLogger, created in initialize()
        self._versions_flushed_through = -1   # highest version id already in versions.npz
        # candidate corrections (segd.candidates), computed GT-free per version id; any mask
        # change to that version invalidates them (ops.op_accept enforces the staleness)
        self.candidate_cache: dict[int, list] = {}   # version id -> proposed add-candidates
        self.candidates_shown: int | None = None     # version whose candidates the agent last saw
        self.accept_rounds = 0                       # successful seg_accept calls (review cap)
        self._timings_candidates: float | None = None   # seconds spent computing candidates in this op
        self._log_extra: dict = {}                   # op-specific fields for the trajectory record
        self._last_render_info: Any = None           # RenderInfo of the last op's render (v3 hooks)
        self.created_at = time.time()

    # ----------------------------------------------------------------- setup
    def initialize(self) -> None:
        """Heavy one-off work (marks, upload) — called by POST /cases in the threadpool."""
        from segd.log import TrajectoryLogger

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log = TrajectoryLogger(self.out_dir)

        self.body = _load_body_mask(self.image)
        idx = np.nonzero(self.body)
        if idx[0].size:
            self.body_bbox = tuple(
                (int(max(a.min() - config.CROP_MARGIN_VOX, 0)), int(min(a.max() + 1 + config.CROP_MARGIN_VOX, n)))
                for a, n in zip(idx, self.shape)
            )

        self.markset = self._load_or_compute_marks()
        self._nni_connect()

        self.log.write_case_meta({
            "case_id": self.case_id, "shape": list(self.shape), "spacing": list(self.spacing),
            "sequence": self.sequence, "target_text": self.target_text, "budgets": self.budgets,
            "nni": "remote" if self.nni_url else "fake",
            "v3": self.v3,
            "dataset": self.dataset,
            "upload_packed_bytes": self.upload_packed_bytes,
            "crop_offset": list(self.crop_offset), "crop_shape": list(self.crop_shape),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.created_at)),
            "source_orientation": self._source_orientation(),
        })

    def _source_orientation(self) -> str:
        """Axcode letters of the source NIfTI ('RAS' for the common canonical files)."""
        if self.orig_geometry is None:
            return "RAS"
        from nibabel.orientations import io_orientation, ornt2axcodes
        return "".join(ornt2axcodes(io_orientation(self.orig_geometry[0])))

    def _load_or_compute_marks(self) -> Any:
        """Compute once per volume and cache at <results root>/cache/marks/<case>.npz."""
        cache_path = self.out_dir.parent.parent / "cache" / "marks" / f"{self.case_id}.npz"
        try:
            from segd.marks import MarkSet, cache_matches, compute_marks  # contract import
        except ImportError:
            return None  # marks module not written yet; ops needing marks will fail clearly
        if cache_path.exists():
            try:
                ms = MarkSet.load(str(cache_path))
                # fingerprint includes the density profile of THIS dataset (decision 5)
                if cache_matches(ms, self.shape,
                                 profile=config.MARKS_PROFILE.get(self.dataset, "default")):
                    return ms
            except Exception:
                pass  # corrupt/incompatible cache file: recompute
        # dataset-level marks density (Phase 4 decision 5): "mstt" keeps the default profile
        ms = compute_marks(self.image, self.spacing,
                           profile=config.MARKS_PROFILE.get(self.dataset, "default"))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            ms.save(str(cache_path))   # atomic (tmp + os.replace)
        except Exception:
            pass  # cache write is an optimization, never a failure
        return ms

    def _nni_connect(self) -> None:
        """Upload the image (float16 (1,I,J,K)) and register the target buffer.

        If the packed size exceeds config.UPLOAD_MAX_PACKED_BYTES, upload the body bbox
        (+config.CROP_MARGIN_VOX) instead and keep `crop_offset`; every prompt coordinate
        and mask is translated between full and session space in both directions.
        """
        img16 = np.ascontiguousarray(self.image, dtype=np.float16)
        full_packed = pack_array(img16[None])
        if len(full_packed) <= config.UPLOAD_MAX_PACKED_BYTES or self.body_bbox is None:
            session_img, self.crop_offset = img16, (0, 0, 0)
            self.upload_packed_bytes = len(full_packed)
        else:
            self.crop_offset = tuple(lo for lo, _ in self.body_bbox)
            self.crop_shape = tuple(hi - lo for lo, hi in self.body_bbox)
            session_img = img16[tuple(slice(lo, hi) for lo, hi in self.body_bbox)]
            # size of what is actually uploaded (the packed cropped volume)
            self.upload_packed_bytes = len(pack_array(np.ascontiguousarray(session_img)[None]))
        self.crop_shape = tuple(int(n) for n in session_img.shape)

        if self.nni_url:
            self.nni = nnInteractiveRemoteInferenceSession(self.nni_url, read_timeout=300, write_timeout=600)
        else:
            from segd.fake_nni import FakeNNI
            self.nni = FakeNNI()
        self.nni.set_image(session_img[None], image_properties={"spacing": list(self.spacing)})
        self.nni.set_target_buffer(np.zeros(self.crop_shape, dtype=np.uint8))

    def close(self) -> None:
        try:
            if self.nni is not None:
                self.nni.close()
        except Exception:
            pass
        self.nni = None

    # -------------------------------------------------------------- versions
    def version_records(self) -> list[dict]:
        return [r.__dict__ for r in (self.versions[i] for i in sorted(self.versions))]

    def _clear_authorization(self) -> None:
        """Revoke everything the agent was shown/authorized for on the CURRENT version:
        the candidate sheet (seg_accept) and the review confirmation (review_code).
        Called by every path that changes ``current`` (commit/checkout/undo/rollback):
        authorization must never survive a mask change or a version round trip, while the
        per-version computation cache (candidate_cache) is kept — re-proposing a cached
        version is cheap, only the authorization is revoked."""
        self.candidates_shown = None
        self.reviewed_version = None
        self.review_code = None

    def commit_mask(self, mask_full: np.ndarray, note: str = "") -> int:
        """Store a new immutable version from a full-volume mask; returns its int id."""
        vid = len(self.versions)
        bbox, packed, nvox = pack_region(mask_full)
        self.versions[vid] = VersionRecord(vid, self.current, bbox, packed, nvox)
        self.current = vid
        self._clear_authorization()
        # versions.npz is written at finish and after every VERSIONS_NPZ_EVERY committed
        # versions (CONTRACTS §4/§5.7); an explicit flush_versions() covers the tail.
        if config.VERSIONS_NPZ_EVERY and vid % config.VERSIONS_NPZ_EVERY == 0:
            self.flush_versions()
        return vid

    @contextmanager
    def rollback_on_error(self, parent_vid: int):
        """Make a mutating op all-or-nothing after its commit/switch: when the body raises
        (render or observation failure), versions created inside it are dropped, ``current``
        returns to ``parent_vid`` and the nnInteractive state is marked stale, so an error
        reply ("mask unchanged") is true."""
        n_before = len(self.versions)
        try:
            yield
        except BaseException:
            for vid in [v for v in self.versions if v >= n_before]:
                del self.versions[vid]
                self.candidate_cache.pop(vid, None)   # version ids are reused after a rollback
            self.current = parent_vid
            # fail closed: whatever authorization the op already revoked stays revoked
            self._clear_authorization()
            self._nni_dirty = True
            self._versions_flushed_through = min(self._versions_flushed_through, n_before - 1)
            raise

    def flush_versions(self) -> None:
        if self.log is None:
            return
        self.log.flush_versions(self.version_records(), self.shape)
        self._versions_flushed_through = len(self.versions) - 1

    def current_mask(self) -> np.ndarray:
        r = self.versions[self.current]
        return unpack_region(r.bbox, r.packed, r.nvox, self.shape)

    def version_mask(self, vid: int) -> np.ndarray:
        r = self.versions[vid]
        return unpack_region(r.bbox, r.packed, r.nvox, self.shape)

    def checkout(self, version: str | int) -> int:
        """Switch the current version; returns the new version id (int; label via vid_label)."""
        vid = parse_version(version, len(self.versions))
        if vid != self.current:
            self.current = vid
            self._clear_authorization()   # undo/checkout round trips must not revive authorization
        self._nni_dirty = True
        return vid

    def undo(self) -> int:
        parent = self.versions[self.current].parent
        if parent is None:
            raise OpError(f"cannot undo: {vid_label(self.current)} has no parent")
        return self.checkout(parent)

    # ------------------------------------------- nnInteractive sync (CONTRACTS §4)
    def _sync_nni(self) -> None:
        """Before a model prompt: after any non-model change (paint/edit/checkout/undo)
        the server's interaction state is stale, so reset it and, if the current mask is
        non-empty, feed it as the initial seg (no prediction)."""
        if not self._nni_dirty:
            return
        self.nni.reset_interactions()  # zeroes the target buffer client- and server-side
        mask = self.current_mask_in_session_space()
        if mask.any():
            self.nni.add_initial_seg_interaction(mask, run_prediction=False)
        self._nni_dirty = False

    def _reconnect(self) -> None:
        """Drop and rebuild the nnInteractive session (image re-uploaded, fresh target
        buffer). The caller keeps ``_nni_dirty=True`` so the next prompt re-sends the
        current mask (finding #2)."""
        try:
            if self.nni is not None:
                self.nni.close()
        except Exception:
            pass
        self.nni = None
        self._nni_connect()

    def _prompt_once(self, fn) -> None:
        self._sync_nni()
        t0 = time.time()
        fn(self.nni)
        self._nni_seconds = time.time() - t0
        self.commit_mask(self.session_mask_to_full(self.nni.target_buffer))
        self.model_prompts += 1

    def model_prompt(self, fn) -> None:
        """Run one model prompt (fn(session)) and store the resulting target buffer as a
        new version. Callers must have enforced the prompt budget.

        Any failure leaves the current mask unchanged: nni is marked dirty so the next
        prompt re-syncs the server state (a partially-applied multi-point prompt is
        discarded). Session-expiry/connection failures reconnect once (fresh session,
        current mask re-sent by the sync rule) and retry; every other exception raises
        the agent-facing MODEL_PROMPT_ERROR OpError (finding #2)."""
        if self.nni is None:
            # an earlier reconnect failed and left no session: try to rebuild it now
            # (the dirty flag makes the sync rule re-send the current mask)
            self._nni_dirty = True
            try:
                self._reconnect()
            except Exception as e:
                self.nni = None
                raise OpError(MODEL_PROMPT_ERROR) from e
        try:
            self._prompt_once(fn)
            return
        except Exception as e:
            self._nni_dirty = True
            if not _is_transport_failure(e):
                raise OpError(MODEL_PROMPT_ERROR) from e
        try:
            self._reconnect()
            self._prompt_once(fn)
        except Exception as e:
            self._nni_dirty = True
            raise OpError(MODEL_PROMPT_ERROR) from e

    # ------------------------------------------------- coordinate translation
    def full_to_session_ijk(self, ijk) -> tuple:
        """Full-volume voxel -> session (crop) voxel; OpError when the point falls outside
        the uploaded crop (e.g. a mark projected onto a slice outside the scanned body
        region — finding #6; the real server would reject it with a 500)."""
        s = tuple(int(c) - int(o) for c, o in zip(ijk, self.crop_offset))
        if any(not (0 <= c < n) for c, n in zip(s, self.crop_shape)):
            raise OpError("that mark lies outside the scanned body region on this slice")
        return s

    def session_to_full_ijk(self, ijk) -> tuple:
        return tuple(int(c) + int(o) for c, o in zip(ijk, self.crop_offset))

    def session_mask_to_full(self, buf) -> np.ndarray:
        full = np.zeros(self.shape, dtype=np.uint8)
        sl = tuple(slice(o, o + n) for o, n in zip(self.crop_offset, self.crop_shape))
        full[sl] = np.asarray(buf, dtype=np.uint8)
        return full

    def current_mask_in_session_space(self) -> np.ndarray:
        sl = tuple(slice(o, o + n) for o, n in zip(self.crop_offset, self.crop_shape))
        return self.current_mask()[sl]

    # ------------------------------------------------------------ mark resolution
    def _make_active_view(self, axis: str, slice_idx: int, render_info, marks_level: str,
                          box2d) -> ActiveView:
        return ActiveView(axis, slice_idx, getattr(render_info, "view_geom", None),
                          render_info, marks_level, box2d)

    def require_active_view(self) -> ActiveView:
        if self.active_view is None:
            raise OpError("no active view: call seg_view before any spatial op")
        return self.active_view

    def resolve_ids(self, ids) -> list[ResolvedMark]:
        """Resolve mark IDs in the active view (CONTRACTS §4). Global integer IDs must be
        shown in the active view; edge IDs 'E<n>' come only from the active view's
        RenderInfo. Returns ResolvedMark entries with full-volume voxels."""
        av = self.require_active_view()
        drawn = drawn_global_ids(av.render_info)
        edges = edge_mark_map(av.render_info)
        out: list[ResolvedMark] = []
        for raw in ids:
            mid = normalize_mark_id(raw)
            if isinstance(mid, int):
                if self.markset is None:
                    raise OpError("marks are not available for this case (marks module missing?)")
                mark = self.markset.marks.get(mid)
                if mark is None:
                    raise OpError(f"mark {mid} does not exist")
                if mid not in drawn:
                    raise OpError(f"mark {mid} is not shown in the current view")
                ijk = project_to_slice(mark.ijk, av.axis, av.slice_idx)
            else:
                if mid not in edges:
                    raise OpError(f"edge mark {mid} is not shown in the current view")
                row, col = edges[mid]
                ijk = inplane_to_voxel(av.axis, row, col, av.slice_idx, self.shape)
            out.append(ResolvedMark(mid, tuple(int(c) for c in ijk)))
        return out

    # -------------------------------------------------------------- finish
    def save_final_nifti(self) -> Path:
        """final_mask.nii.gz in the ORIGINAL source-file geometry (CONTRACTS §4, finding
        #14): when the source NIfTI needed canonicalisation at load, the mask is mapped
        back through the inverse orientation (nibabel ornt) and saved with the source
        affine; RAS sources keep the previous behaviour (canonical affine)."""
        mask = self.current_mask().astype(np.uint8)
        if self.orig_geometry is not None:
            orig_affine, inv_ornt = self.orig_geometry
            from nibabel.orientations import apply_orientation
            mask = np.ascontiguousarray(apply_orientation(mask, inv_ornt))
            affine = orig_affine
        else:
            affine = self.affine
        path = self.out_dir / "final_mask.nii.gz"
        nib.save(nib.Nifti1Image(mask, affine), str(path))
        return path

    def finish(self, confidence: str, reason: str) -> str:
        if confidence not in ("high", "medium", "low"):
            raise OpError("confidence must be high|medium|low")
        self.flush_versions()
        self.save_final_nifti()
        self.close()   # free the nnInteractive session slot as soon as the case ends (finding #5)
        self.log.write_case_meta({
            "case_id": self.case_id, "shape": list(self.shape), "spacing": list(self.spacing),
            "sequence": self.sequence, "target_text": self.target_text, "budgets": self.budgets,
            "nni": "remote" if self.nni_url else "fake",
            "v3": self.v3,
            "dataset": self.dataset,
            "upload_packed_bytes": self.upload_packed_bytes,
            "crop_offset": list(self.crop_offset), "crop_shape": list(self.crop_shape),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.created_at)),
            "source_orientation": self._source_orientation(),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "final_version": vid_label(self.current), "confidence": confidence, "reason": reason,
            "tool_calls": self.tool_calls, "model_prompts": self.model_prompts,
        })
        self.done = True
        return f"case {self.case_id} finished (confidence {confidence}); final mask saved from {vid_label(self.current)}"


# ----------------------------------------------------------------- helpers
def vid_label(vid: int) -> str:
    return f"v{vid}"


def parse_version(version, n_versions: int) -> int:
    if isinstance(version, str):
        s = version.strip().lower()
        if not (s.startswith("v") and s[1:].isdigit()):
            raise OpError(f"unknown version {version!r}")
        vid = int(s[1:])
    elif isinstance(version, (int, np.integer)):
        vid = int(version)
    else:
        raise OpError(f"unknown version {version!r}")
    if vid < 0 or vid >= n_versions:
        raise OpError(f"unknown version {version!r}")
    return vid


def normalize_mark_id(raw):
    """Global marks are integers, edge marks are strings 'E<n>'."""
    if isinstance(raw, bool):
        raise OpError(f"invalid mark id {raw!r}")
    if isinstance(raw, (int, np.integer)):
        return int(raw)
    if isinstance(raw, str):
        s = raw.strip()
        if s.upper().startswith("E") and s[1:].isdigit():
            return s.upper()
        if s.isdigit():
            return int(s)
    raise OpError(f"invalid mark id {raw!r}")


def project_to_slice(ijk, axis: str, slice_idx: int) -> tuple:
    """Project a 3D mark voxel onto a slice: the fixed-axis coordinate becomes slice_idx."""
    out = list(int(c) for c in ijk)
    out[config.FIXED_AXIS[axis]] = int(slice_idx)
    return tuple(out)


def inplane_to_voxel(axis: str, row: int, col: int, slice_idx: int, shape: tuple) -> tuple:
    """Displayed in-plane (row, col) on the 2D slice -> full voxel (CONTRACTS §1).

    Delegates to segd.marks.voxel_of_inplane (the single definition of the mapping);
    the local fallback below is identical to it for when marks.py is absent.
    """
    try:
        from segd.marks import slice_shape, voxel_of_inplane
        rows, cols = slice_shape(shape, axis)
        return voxel_of_inplane(row, col, slice_idx, axis, rows, cols)
    except ImportError:
        ijk = [0, 0, 0]
        ijk[config.FIXED_AXIS[axis]] = int(slice_idx)
        ijk[config.ROW_AXIS[axis]] = shape[config.ROW_AXIS[axis]] - 1 - int(row)
        ijk[config.COL_AXIS[axis]] = shape[config.COL_AXIS[axis]] - 1 - int(col)
        return tuple(ijk)


def voxel_to_inplane(axis: str, ijk, shape: tuple) -> tuple:
    """Full voxel -> displayed in-plane (row, col) of the slice it lies on."""
    try:
        from segd.marks import inplane_of, slice_shape
        rows, cols = slice_shape(shape, axis)
        return inplane_of(tuple(int(c) for c in ijk), axis, rows, cols)
    except ImportError:
        return (shape[config.ROW_AXIS[axis]] - 1 - int(ijk[config.ROW_AXIS[axis]]),
                shape[config.COL_AXIS[axis]] - 1 - int(ijk[config.COL_AXIS[axis]]))


def slice2d_displayed(image: np.ndarray, axis: str, slice_idx: int) -> np.ndarray:
    """The 2D slice in display orientation (rows/cols flipped), CONTRACTS §1.

    This is the array convention for `edge_marks_2d` in-plane coordinates and for
    mask crops passed to rendering. Delegates to segd.marks.extract_slice when
    available (single definition); local fallback identical to it.
    """
    try:
        from segd.marks import extract_slice
        return extract_slice(image, axis, slice_idx)
    except ImportError:
        raw = np.take(image, int(slice_idx), axis=config.FIXED_AXIS[axis])
        return np.ascontiguousarray(raw.T[::-1, ::-1])


# --------- adaptation layer for segd.render.RenderInfo (CONTRACTS §3) ---------
def drawn_global_ids(render_info) -> set[int]:
    """Global mark IDs shown in the render. RenderInfo records 'the list of marks drawn
    with their panel positions' — accept (id, (x, y)) tuples, mark objects, or an id->xy
    dict."""
    if render_info is None:
        return set()
    items = getattr(render_info, "marks", None)
    if items is None:
        raise OpError("RenderInfo does not record drawn marks (render contract mismatch)")
    if isinstance(items, dict):
        return {int(k) for k in items}
    ids = set()
    for item in items:
        if isinstance(item, (tuple, list)):
            ids.add(int(item[0]))
        else:
            ids.add(int(getattr(item, "id")))
    return ids


def edge_mark_map(render_info) -> dict[str, tuple]:
    """Edge marks E<n> -> in-plane (row, col) recorded by the render.

    Accepts the real RenderInfo 4-tuples (label, panel_idx, row, col) and the plain
    (label, (row, col)) form."""
    if render_info is None:
        return {}
    items = getattr(render_info, "edge_marks", None)
    if items is None:
        raise OpError("RenderInfo does not record edge marks (render contract mismatch)")
    if isinstance(items, dict):
        return {str(k): (int(v[0]), int(v[1])) for k, v in items.items()}
    out = {}
    for item in items:
        if isinstance(item[-1], (tuple, list)):      # (label, (row, col))
            eid, rc = item[0], item[-1]
        else:                                        # (label, panel_idx, row, col)
            eid, rc = item[0], (item[-2], item[-1])
        out[str(eid)] = (int(rc[0]), int(rc[1]))
    return out


def primary_view_geom(render_info):
    """The geometry of a RenderInfo's first panel (real RenderInfo has ``views``)."""
    if render_info is None:
        return None
    g = getattr(render_info, "view_geom", None)
    if g is not None:
        return g
    views = getattr(render_info, "views", None)
    if views:
        return views[0]
    return None
