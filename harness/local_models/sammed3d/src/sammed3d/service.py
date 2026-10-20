"""SAM-Med3D segmenter service (Phase 6 segmenter swap).

FastAPI on 127.0.0.1:8095 (``sammed3d-service``). It exposes a session API mirroring
exactly the nnInteractive remote-session methods segd's CaseState calls — set_image,
set_target_buffer, add_{point,bbox,lasso,scribble,initial_seg}_interaction,
reset_interactions, predict, close — so the Python client (``sammed3d.client.SamMed3DSession``)
is a call-signature drop-in for ``nnInteractiveRemoteInferenceSession``
(CONTRACTS.md "Phase 6"). Each prediction returns the FULL session-space mask
(packbits of its bbox crop in the response body, meta in the ``X-Sam-Meta`` header)
together with the prompt-mapping log, patch info and timings.

Two backends:
- real (default): the official SAM-Med3D-turbo checkpoint via ``medim.create_model``,
  official preprocessing (resample 1.5 mm, masked z-normalisation, 128^3 patches),
  patch tiling/stitching per geometry.py; needs torch + a GPU.
- ``--fake``: a deterministic numpy stand-in (no torch): the SAME session state,
  accumulation, patch placement, stitching and grid mapping, with a per-patch
  intensity-similar flood fill standing in for the model (used by the offline tests).

The orchestrator (``SamSession``) is backend-agnostic; a predictor implements
``prepare(image, spacing)`` (resample to model space) and ``predict_patch(...)``
returning ``(logits_128, logits_lowres)`` for one patch. The fake therefore exercises
the identical geometry the real engine uses.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from starlette.concurrency import run_in_threadpool

from sammed3d import client as _client
from sammed3d import geometry

SERVICE_VERSION = "0.1.0"
DEFAULT_PORT = 8095
DEFAULT_URL = f"http://127.0.0.1:{DEFAULT_PORT}"
POINT_RADIUS_VOX = 64       # fake flood-growth cap from the seed (model vox)


class FakePatchPredictor:
    """``--fake`` mode's numpy stand-in for the torch engine (deterministic, no torch).

    Per-patch 'model' on the z-normalized patch:
    - positive points grow an intensity-similar flood fill (tolerance from the patch's
      own percentiles, capped at POINT_RADIUS_VOX from the seed);
    - negative points grow the same region, which is removed;
    - the mask prompt (>0 region, the previous mask or initial seg) is kept where no
      point overrides it — so a refine keeps the mask and a mask-prompt-only patch
      contributes its prompt slice (the documented tiling behavior).

    This is a smoke/API stand-in, NOT a SAM-Med3D quality model; the trajectory of
    real-model behaviour is measured in the GPU smoke test (research/phase6_sammed3d.md).
    """

    kind = "fake"

    def prepare(self, image: np.ndarray, spacing) -> np.ndarray:
        return geometry.resample_np(image, spacing)

    def predict_patch(self, patch_img, points_local, labels, mask_prompt) -> tuple:
        """Return (logits_128, logits_32) for one z-normalized patch."""
        logits_lo = np.asarray(mask_prompt, dtype=np.float32)   # (32,32,32) logits
        x = np.asarray(patch_img, dtype=np.float32)
        if points_local.shape[0]:
            # z-normalized data: "similar" is a fixed tolerance in std units (the
            # phantom's noise is ~0.7 std, the lesion-body separation ~3.5 std)
            tol = 1.5
            pos_region = np.zeros(x.shape, dtype=bool)
            neg_region = np.zeros(x.shape, dtype=bool)
            for p, lab in zip(points_local, labels):
                seed = tuple(int(np.clip(round(float(c)), 0, n - 1))
                             for c, n in zip(p, x.shape))
                comp = self._flood(x, seed, tol)
                if not comp.any():
                    comp = np.zeros(x.shape, dtype=bool)
                    comp[seed] = True
                if int(lab) == 1:
                    pos_region |= comp
                else:
                    neg_region |= comp
            pos_lo = geometry.block_mean_to_32(pos_region.astype(np.float32)) > 0.5
            neg_lo = geometry.block_mean_to_32(neg_region.astype(np.float32)) > 0.5
            # removal wins in a conflicting block (erase must be able to shrink the
            # mask; the official model decides from all prompts at once)
            logits_lo = np.where(neg_lo, -geometry.MASK_PROMPT_LOGIT,
                                 np.where(pos_lo, geometry.MASK_PROMPT_LOGIT, logits_lo))
        logits128 = self._upsample(logits_lo)
        return logits128.astype(np.float32), logits_lo.astype(np.float32)

    @staticmethod
    def _upsample(logits_lo: np.ndarray) -> np.ndarray:
        r = geometry.PATCH_VOX // geometry.LOWRES_VOX
        return np.repeat(np.repeat(np.repeat(logits_lo, r, axis=0), r, axis=1), r, axis=2)

    @staticmethod
    def _flood(x, seed, tol) -> np.ndarray:
        from scipy import ndimage
        shape = x.shape
        lo = [max(s - POINT_RADIUS_VOX, 0) for s in seed]
        hi = [min(s + POINT_RADIUS_VOX + 1, n) for s, n in zip(seed, shape)]
        crop = x[tuple(slice(a, b) for a, b in zip(lo, hi))]
        local_seed = tuple(s - a for s, a in zip(seed, lo))
        nb = crop[tuple(slice(max(0, s - 2), s + 3) for s in local_seed)]
        ref = float(np.median(nb))
        similar = np.abs(crop - ref) <= tol
        labels_, _n = ndimage.label(similar, structure=ndimage.generate_binary_structure(3, 1))
        lab = int(labels_[local_seed])
        if lab == 0:
            # noise: the seed voxel itself failed the tolerance — grow from the nearest
            # similar voxel instead (deterministic); without one, an empty region
            nz = np.nonzero(similar)
            if nz[0].size:
                d2 = ((nz[0] - local_seed[0]) ** 2 + (nz[1] - local_seed[1]) ** 2
                      + (nz[2] - local_seed[2]) ** 2)
                k = int(np.argmin(d2))
                lab = int(labels_[nz[0][k], nz[1][k], nz[2][k]])
        comp = (labels_ == lab) if lab else np.zeros(crop.shape, dtype=bool)
        out = np.zeros(shape, dtype=bool)
        out[tuple(slice(a, b) for a, b in zip(lo, hi))] = comp
        return out


class SamSession:
    """One open case session: interaction state + shared prediction orchestration.

    Accumulation semantics (official iterative clicking): ALL point prompts are kept
    in order; every prediction recomputes the 3D mask from all points so far plus the
    previous prediction's low-res mask (per patch; geometry.mask_prompt_for_window).
    Prompts added with ``run_prediction=False`` only accumulate (no model call),
    mirroring the nnInteractive session.
    """

    def __init__(self, session_id: str, predictor, model_lock: threading.Lock):
        self.session_id = session_id
        self.predictor = predictor
        self.model_lock = model_lock
        self.image: Optional[np.ndarray] = None          # float32 (I,J,K) session space
        self.spacing: Optional[tuple] = None
        self.target_shape: Optional[tuple] = None
        self.image_model: Optional[np.ndarray] = None    # float32, model space (1.5 mm)
        self.points: list = []                           # [((i,j,k) session vox, label)]
        self.first_positive: Optional[np.ndarray] = None # first positive since last reset
        self.prev_state = None                           # (lo, prob f16 crop, lowres list)
        self.pending_log: list = []                      # mapping-log entries of queued prompts

    # ---------------------------------------------------------------- setup
    def set_image(self, image: np.ndarray, image_properties: Optional[dict]) -> dict:
        assert image.ndim == 4, f"expected a 4d image as input, got {image.ndim}d. Shape {image.shape}"
        if image.shape[0] != 1:
            raise HTTPException(status_code=400, detail="expected a single-channel 4d image")
        spacing = (image_properties or {}).get("spacing")
        if spacing is None:
            raise HTTPException(status_code=400, detail="image_properties.spacing is required")
        self.image = np.ascontiguousarray(image[0], dtype=np.float32)
        self.spacing = tuple(float(s) for s in spacing)
        self.points = []
        self.first_positive = None
        self.prev_state = None
        self.pending_log = []
        t0 = time.time()
        self.image_model = self.predictor.prepare(self.image, self.spacing)
        return {"original_image_shape": list(self.image.shape),
                "model_shape": list(self.image_model.shape),
                "prepare_s": round(time.time() - t0, 3)}

    def set_target_buffer(self, shape, dtype: str) -> dict:
        self.target_shape = tuple(int(n) for n in shape)
        return {"ok": True}

    # ----------------------------------------------------------- interactions
    def _register_point(self, p_session, label: int) -> None:
        """Store a point in SESSION-space voxels (converted to model space once, in
        _predict); the mapping log reports session coordinates."""
        self.points.append((np.asarray(p_session, dtype=np.float64), int(label)))
        if int(label) == 1 and self.first_positive is None:
            self.first_positive = np.asarray(p_session, dtype=np.float64)

    def add_point_interaction(self, coordinates, include_interaction: bool,
                              run_prediction: bool, override_capability_checks: bool) -> dict:
        coord = np.asarray([float(c) for c in coordinates], dtype=np.float64)
        if not np.all(np.isfinite(coord)):
            raise HTTPException(status_code=400, detail="coordinates must be finite")
        self._register_point(coord, 1 if include_interaction else 0)
        return self._maybe_predict(run_prediction, "point",
                                   coord=[float(c) for c in coord],
                                   label=1 if include_interaction else 0,
                                   mapped_points=_point_log(
                                       [(coord, 1 if include_interaction else 0)]))

    def add_bbox_interaction(self, bbox_coords, include_interaction: bool,
                             run_prediction: bool, override_capability_checks: bool) -> dict:
        try:
            lo = np.asarray([float(b[0]) for b in bbox_coords])
            hi = np.asarray([float(b[1]) for b in bbox_coords])
        except (TypeError, IndexError, ValueError):
            raise HTTPException(status_code=400, detail="bbox_coords must be [[lo,hi] x 3]")
        mapped = geometry.bbox_quarter_points((lo, hi), include_interaction)
        for p, lab in mapped:                     # box coords are session-space voxels
            self._register_point(p, lab)
        return self._maybe_predict(run_prediction, "bbox",
                                   n_points=len(mapped), include=bool(include_interaction),
                                   mapped_points=_point_log(mapped))

    def _add_drawn_interaction(self, drawn: np.ndarray, mode: str,
                               include_interaction: bool, run_prediction: bool,
                               interaction_bbox, override_capability_checks: bool) -> dict:
        if interaction_bbox is None:
            raise HTTPException(status_code=400, detail=f"{mode} needs interaction_bbox")
        box = [[int(b[0]), int(b[1])] for b in interaction_bbox]
        if drawn.shape != tuple(hi - lo for lo, hi in box):
            raise HTTPException(
                status_code=400,
                detail=f"drawn mask shape {tuple(drawn.shape)} != interaction box extents "
                       f"{tuple(hi - lo for lo, hi in box)}")
        local = geometry.fps_points(drawn) if mode == "lasso" else geometry.stroke_points(drawn)
        lo = np.asarray([b[0] for b in box], dtype=np.float64)
        label = 1 if include_interaction else 0
        pts = [lo + np.asarray(p) for p in local]   # drawn crop-local -> session voxels
        for p in pts:
            self._register_point(p, label)
        return self._maybe_predict(run_prediction, mode,
                                   n_points=len(pts), include=bool(include_interaction),
                                   mapped_points=_point_log([(p, label) for p in pts]))

    def add_initial_seg_interaction(self, initial_seg: np.ndarray, run_prediction: bool,
                                    override_capability_checks: bool) -> dict:
        mask = np.asarray(initial_seg)
        if mask.shape != self.image.shape:
            raise HTTPException(status_code=400,
                                detail=f"initial seg shape {tuple(mask.shape)} != image "
                                       f"{tuple(self.image.shape)}")
        mask = mask > 0
        # the given mask becomes the previous-state mask prompt (mapped to model space,
        # nearest like the official label resampling); the raw low-res logits of any
        # earlier prediction are dropped — the given mask is authoritative now
        mask_model = geometry.map_mask_session_to_model(mask, self.spacing,
                                                        self.image_model.shape)
        self.prev_state = (np.zeros(3, dtype=int), mask_model.astype(np.float16), [])
        interior_added = False
        mapped = []
        if not self.points:
            interior = geometry.edt_max_voxel(mask)
            self._register_point(np.asarray(interior, dtype=np.float64), 1)
            interior_added = True
            mapped = _point_log([(np.asarray(interior, dtype=np.float64), 1)])
        return self._maybe_predict(run_prediction, "initial_seg",
                                   interior_point=interior_added, mapped_points=mapped,
                                   mask_prompt_vox=int(mask.sum()))

    def reset_interactions(self) -> dict:
        self.points = []
        self.first_positive = None
        self.prev_state = None
        self.pending_log = []
        return {"ran_prediction": False, "interactions": []}

    def _maybe_predict(self, run_prediction: bool, op: str, **detail) -> dict:
        if not run_prediction:
            self.pending_log.append({"op": op, **detail})   # logged with the next prediction
            return {"ran_prediction": False, "op": op, **detail}
        with self.model_lock:
            return self._predict(op, **detail)

    # ------------------------------------------------------------- prediction
    def _predict(self, op: str, **detail) -> dict:
        """Recompute the 3D mask from ALL accumulated prompts + the previous mask
        (official iterative clicking) and return it in session space."""
        spacing = self.spacing
        # mapping log: every prompt call queued since the previous prediction (a
        # multi-mark seg_point op is several calls) + the call that triggered this one
        call_log = self.pending_log + ([{"op": op, **detail}] if detail else [])
        self.pending_log = []
        pts = np.asarray([p for p, _l in self.points], dtype=np.float64) if self.points \
            else np.zeros((0, 3))
        labels = np.asarray([l for _p, l in self.points], dtype=np.int64) if self.points \
            else np.zeros((0,), dtype=np.int64)
        pts_model = np.zeros((0, 3)) if pts.size == 0 else np.stack(
            [geometry.session_to_model(p, self.image.shape, spacing) for p in pts])
        pos = pts_model[labels == 1]
        prev_bbox = None
        if self.prev_state is not None:
            prev_lo, prev_prob = np.asarray(self.prev_state[0], dtype=int), self.prev_state[1]
            prev_mask = np.asarray(prev_prob) > 0.5          # the thresholded previous mask
            if prev_mask.any():
                prev_bbox = geometry.bbox_of_points(
                    np.stack(np.nonzero(prev_mask), axis=1) + prev_lo)
        box = geometry.bbox_union(geometry.bbox_of_points(pos), prev_bbox)
        if box is None:
            return {"ran_prediction": False, "op": op, **detail,
                    "interactions": call_log + [{"note": "no prompts; empty mask"}]}
        t0 = time.time()
        seed_model = None if self.first_positive is None else geometry.session_to_model(
            self.first_positive, self.image.shape, spacing)
        windows, info = geometry.choose_patches(box[0], box[1], seed_model,
                                                self.image_model.shape)
        patch_logits, lowres_list, per_patch = [], [], []
        for w_idx, (w_lo, w_hi) in enumerate(windows):
            local_pts, local_labels = geometry.points_inside_window(
                pts_model, w_lo, w_hi, labels)
            mask_prompt = geometry.mask_prompt_for_window(self.prev_state, (w_lo, w_hi))
            if local_pts.shape[0] == 0 and not (np.asarray(mask_prompt) > 0).any():
                per_patch.append({"idx": w_idx, "skipped": "no prompt inside"})
                continue                      # nothing to condition on: log and skip
            patch_img = geometry.znorm_masked(geometry.crop_patch(self.image_model, w_lo, w_hi))
            t_p = time.time()
            logits128, logits32 = self.predictor.predict_patch(
                patch_img, local_pts, local_labels, mask_prompt)
            patch_logits.append(((w_lo, w_hi), np.asarray(logits128, dtype=np.float32)))
            lowres_list.append(((w_lo, w_hi), np.asarray(logits32, dtype=np.float32)))
            per_patch.append({"idx": w_idx, "n_points": int(local_pts.shape[0]),
                              "predict_s": round(time.time() - t_p, 3)})
        if not patch_logits:
            return {"ran_prediction": False, "op": op, **detail, "patches": info,
                    "per_patch": per_patch,
                    "interactions": call_log + [{"note": "no patch had a prompt; mask unchanged"}]}
        stitched, union_lo = geometry.stitch_logits(patch_logits, self.image_model.shape)
        binary_crop = stitched > 0.0               # sigmoid > 0.5  <=>  logits > 0
        binary = np.zeros(self.image_model.shape, dtype=bool)   # paste into model space
        binary[tuple(slice(int(l), int(l) + n) for l, n in zip(union_lo, binary_crop.shape))] = binary_crop
        mask_session = geometry.map_binary_model_to_session(binary, spacing, self.image.shape)
        prob = (1.0 / (1.0 + np.exp(-stitched.astype(np.float32)))).astype(np.float16)
        self.prev_state = (union_lo, prob, lowres_list)
        mask_meta, mask_bytes = _client.encode_mask(mask_session.astype(np.uint8))
        return {"ran_prediction": True, "op": op, **detail, "mask": mask_meta,
                "bbox": mask_meta["bbox"], "n_vox": int(mask_session.sum()),
                "patches": info, "per_patch": per_patch,
                "interactions": call_log + [
                                 {"patches_used": len(patch_logits),
                                  "patches_skipped": len(per_patch) - len(patch_logits),
                                  "windows_model_vox": [[[int(v) for v in w[0]], [int(v) for v in w[1]]]
                                                        for w in windows],
                                  "n_points_total": int(len(self.points)),
                                  "patch_note": info.get("note", "")}],
                "timings": {"total_s": round(time.time() - t0, 3)}}, mask_bytes

    def status(self) -> dict:
        return {"session_id": self.session_id, "image": self.image is not None,
                "spacing": self.spacing, "points": len(self.points),
                "target_shape": self.target_shape}


def _point_log(points) -> list:
    """Trajectory-log form of mapped prompt points: [[i, j, k, label], ...] in session
    voxels (2-decimal)."""
    return [[round(float(c), 2) for c in p] + [int(lab)] for p, lab in points]


def _json_header(obj: dict) -> str:
    return json.dumps(_client._to_jsonable(obj), separators=(",", ":"))


def create_app(predictor, max_sessions: int = 8, model_info: Optional[dict] = None) -> FastAPI:
    """Build the FastAPI app around a predictor ('real' engine or 'fake' stand-in)."""
    app = FastAPI(title="sammed3d", version=SERVICE_VERSION)
    st = {"sessions": {}, "glock": threading.Lock(),
          "model_lock": threading.Lock(), "max_sessions": max_sessions,
          "predictor": predictor, "info": dict(model_info or {})}
    app.state.sammed3d = st

    def _resolve(request: Request) -> SamSession:
        sid = request.headers.get("x-sam-session")
        if not sid:
            raise HTTPException(status_code=400, detail="missing X-Sam-Session header")
        sess = st["sessions"].get(sid)
        if sess is None:
            raise HTTPException(status_code=404, detail="unknown session")
        return sess

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True, "service": "sammed3d", "version": SERVICE_VERSION, **st["info"]}

    @app.post("/sessions")
    def create_session() -> dict:
        with st["glock"]:
            if len(st["sessions"]) >= st["max_sessions"]:
                raise HTTPException(status_code=503, detail="server is at capacity")
            sid = uuid.uuid4().hex
            st["sessions"][sid] = SamSession(sid, st["predictor"], st["model_lock"])
        return {"session_id": sid}

    @app.delete("/sessions/{sid}")
    def close_session(sid: str) -> dict:
        with st["glock"]:
            st["sessions"].pop(sid, None)
        return {"ok": True, "closed": sid}

    @app.get("/capabilities")
    def capabilities() -> dict:
        # mirror the nnInteractive capability keys; SAM-Med3D supports points natively;
        # every other interaction is the documented prompt mapping (CONTRACTS Phase 6)
        return {
            "supported_interactions": {
                "points": True, "bbox2d": True, "bbox3d": True, "lasso": True,
                "scribble": True, "initial_label": True,
            },
            "channel_mapping": {"points": [0, 1]},
            "num_interaction_channels": 2,
            "supports_initial_label": True,
            "supports_zero_shot_label_refinement": True,
            "preferred_scribble_thickness": 1,
            "interaction_decay": 0,
            "inference_session_version": 1,
            "license": st["info"].get("license"),
            "supports_undo": False,
            "do_autozoom": True,
        }

    @app.post("/set_image")
    async def set_image(request: Request) -> dict:
        sess = _resolve(request)
        meta = json.loads(request.headers.get(_client.META_HEADER, "{}"))
        image = _client.decode_npy(await request.body())
        return await run_in_threadpool(sess.set_image, image, meta.get("image_properties") or {})

    @app.post("/set_target_buffer")
    async def set_target_buffer(request: Request) -> dict:
        sess = _resolve(request)
        body = await request.json()
        return sess.set_target_buffer(body.get("shape") or (), body.get("dtype") or "uint8")

    @app.post("/set_do_autozoom")
    async def set_do_autozoom(request: Request) -> dict:
        _resolve(request)
        return {"ok": True}          # the service always places its own patches

    @app.post("/reset_interactions")
    def reset_interactions(request: Request) -> Response:
        return _respond(_resolve(request).reset_interactions())

    @app.post("/predict")
    def predict(request: Request) -> Response:
        sess = _resolve(request)
        with sess.model_lock:
            out = sess._predict("predict")
        return _respond(out)

    @app.post("/add_point_interaction")
    async def add_point_interaction(request: Request) -> Response:
        sess = _resolve(request)
        body = await request.json()
        out = await run_in_threadpool(
            sess.add_point_interaction, body["coordinates"],
            body["include_interaction"], body.get("run_prediction", True),
            body.get("override_capability_checks", False))
        return _respond(out)

    @app.post("/add_bbox_interaction")
    async def add_bbox_interaction(request: Request) -> Response:
        sess = _resolve(request)
        body = await request.json()
        out = await run_in_threadpool(
            sess.add_bbox_interaction, body["bbox_coords"],
            body["include_interaction"], body.get("run_prediction", True),
            body.get("override_capability_checks", False))
        return _respond(out)

    def _register_drawn(op_name: str, mode: Optional[str] = None) -> None:
        async def handler(request: Request) -> Response:
            sess = _resolve(request)
            meta = json.loads(request.headers.get(_client.META_HEADER, "{}"))
            drawn = _client.decode_npy(await request.body()).astype(np.uint8)
            if op_name == "add_initial_seg_interaction":
                out = await run_in_threadpool(
                    sess.add_initial_seg_interaction, drawn,
                    meta.get("run_prediction", False),
                    meta.get("override_capability_checks", False))
            else:
                out = await run_in_threadpool(
                    sess._add_drawn_interaction, drawn, mode,
                    meta["include_interaction"], meta.get("run_prediction", True),
                    meta.get("interaction_bbox"),
                    meta.get("override_capability_checks", False))
            return _respond(out)
        app.post(f"/{op_name}")(handler)

    _register_drawn("add_lasso_interaction", "lasso")
    _register_drawn("add_scribble_interaction", "scribble")
    _register_drawn("add_initial_seg_interaction")

    def _respond(out) -> Response:
        """Meta header always; packbits body when a full mask is returned."""
        if isinstance(out, tuple):                 # (meta, mask_bytes) from _predict
            meta, mask_bytes = out
            meta = {k: v for k, v in meta.items()}
        else:
            meta, mask_bytes = out, b""
        return Response(
            content=mask_bytes,
            headers={_client.META_HEADER: _json_header(meta)},
            media_type="application/octet-stream",
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAM-Med3D segmenter service (Phase 6 segmenter swap).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--fake", action="store_true",
                        help="deterministic numpy stand-in (no torch); for offline tests")
    parser.add_argument("--checkpoint", default=None,
                        help="path to sam_med3d_turbo.pth (default: the project cache)")
    args = parser.parse_args()

    info = {}
    if args.fake:
        predictor = FakePatchPredictor()
        info = {"mode": "fake"}
    else:
        from sammed3d.engine import SamMed3DEngine   # torch import stays lazy (--fake)
        predictor = SamMed3DEngine(args.checkpoint)
        info = predictor.info
    import uvicorn
    uvicorn.run(create_app(predictor, model_info=info), host=args.host,
                port=args.port, log_level="info")


if __name__ == "__main__":
    main()
