"""HTTP client for the SAM-Med3D local segmenter service (Phase 6 segmenter swap).

``SamMed3DSession`` mirrors ``nnInteractive.inference.remote.nnInteractiveRemoteInferenceSession``'s
public call signatures exactly (asserted by tests/test_client_parity.py) so segd's
CaseState can swap it in behind the identical call surface: the agent sees the same
tools and prompts; only the segmenter changes (CONTRACTS.md "Phase 6").

Differences from the nnInteractive remote session, by design (documented there):
- the service returns the FULL predicted mask per prediction (not a bbox diff); it is
  written into ``self.target_buffer`` as a whole, and ``_last_paste_bbox`` mirrors the
  predicted mask's bbox so callers see the same shape of return value;
- no undo on the server (``supports_undo`` False; ``undo()`` returns False without a
  call — CaseState's undo is a version checkout + sync rule, not a session undo);
- every prediction response's meta carries the prompt-mapping log the service built
  (what each harness prompt became in SAM-Med3D terms); it is kept in
  ``self.last_interactions`` for the trajectory record.

Wire format: request body = raw .npy bytes (np.save), request/response meta = JSON in
the ``X-Sam-Meta`` header. numpy 2 packs this fine and keeps the client torch-free.
"""
from __future__ import annotations

import io
import json
import os
import warnings
from typing import List, Optional, Union

import httpx
import numpy as np

META_HEADER = "x-sam-meta"
CONTENT_TYPE_NPY = "application/x-npy"


class SessionExpiredError(RuntimeError):
    """The service no longer knows this session (restart, or session was closed).

    Mirrors nnInteractive's SessionExpiredError role: CaseState's reconnect logic
    (``_is_transport_failure``) treats it as a transport failure worth one
    reconnect + retry."""


class ServerAtCapacityError(RuntimeError):
    """The service refused a new session (at capacity)."""


def encode_npy(arr: np.ndarray) -> bytes:
    """Raw .npy bytes of ``arr`` (the service wire format for images/masks)."""
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr))
    return buf.getvalue()


def decode_npy(data: bytes) -> np.ndarray:
    """Inverse of encode_npy."""
    return np.load(io.BytesIO(data), allow_pickle=False)


def encode_mask(mask: np.ndarray) -> tuple[dict, bytes]:
    """Full uint8 {0,1} mask -> (meta, packbits-of-bbox-crop bytes). Keeps a 94 M-voxel
    lesion mask at a few hundred kB on the wire (same idea as CaseState.pack_region)."""
    mask = np.asarray(mask, dtype=np.uint8)
    idx = np.nonzero(mask)
    if idx[0].size == 0:
        return {"shape": list(mask.shape), "bbox": None, "nvox": 0}, b""
    bbox = [[int(a.min()), int(a.max()) + 1] for a in idx]
    crop = mask[tuple(slice(lo, hi) for lo, hi in bbox)]
    meta = {"shape": list(mask.shape), "bbox": bbox, "nvox": int(crop.size)}
    return meta, np.packbits(crop.astype(bool), axis=None).tobytes()


def decode_mask(meta: dict, data: bytes, dtype=np.uint8) -> np.ndarray:
    """Inverse of encode_mask."""
    shape = tuple(int(n) for n in meta["shape"])
    bbox, nvox = meta["bbox"], int(meta["nvox"])
    out = np.zeros(shape, dtype=dtype)
    if bbox is None:
        return out
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))[:nvox]
    out[tuple(slice(int(lo), int(hi)) for lo, hi in bbox)] = bits.reshape(
        tuple(int(hi - lo) for lo, hi in bbox))
    return out


def _to_jsonable(obj):
    """numpy scalars/arrays -> JSON builtins (mirrors the nnInteractive client)."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _open_http_client(*args, **kwargs) -> httpx.Client:
    """Factory hook for tests (inject an httpx.ASGITransport to run the service app
    in-process without a real HTTP server). Production: plain httpx.Client."""
    return httpx.Client(*args, **kwargs)


class SamMed3DSession:
    """Client-side session for the sammed3d service; call-signature drop-in for
    ``nnInteractiveRemoteInferenceSession`` (which CaseState uses for nnInteractive).

    All model state lives on the service; this object holds the caller's
    target_buffer (mirrored from responses), the capability metadata fetched at
    construction, and the last response's prompt-mapping log.
    """

    def __init__(
        self,
        server_url: str,
        api_key: Optional[str] = None,
        connect_timeout: float = 10.0,
        read_timeout: float = 60.0,
        set_image_read_timeout: float = 600.0,
        write_timeout: float = 120.0,
        pool_timeout: float = 10.0,
    ):
        if api_key is None:
            api_key = os.environ.get("SAMMED3D_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = _open_http_client(
            base_url=str(server_url).rstrip("/"),
            timeout=httpx.Timeout(
                connect=connect_timeout, read=read_timeout, write=write_timeout,
                pool=pool_timeout,
            ),
            headers=headers,
        )
        self._set_image_timeout = httpx.Timeout(
            connect=connect_timeout, read=set_image_read_timeout,
            write=write_timeout, pool=pool_timeout,
        )
        self._session_id: Optional[str] = None
        try:
            resp = self._http.post("/sessions")
            resp.raise_for_status()
            self._session_id = resp.json()["session_id"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 503:
                raise ServerAtCapacityError(_detail(e.response)) from e
            raise
        self._http.headers["X-Sam-Session"] = self._session_id

        caps = self._get_json("/capabilities")
        # attributes mirroring the nnInteractive session so callers can introspect
        # the session the same way regardless of the segmenter behind it
        self.supported_interactions: dict = caps["supported_interactions"]
        self.channel_mapping: dict = {
            k: (tuple(v) if isinstance(v, list) else v) for k, v in caps["channel_mapping"].items()
        }
        self.num_interaction_channels: int = caps["num_interaction_channels"]
        self.supports_initial_label: bool = caps["supports_initial_label"]
        self.supports_zero_shot_label_refinement: bool = caps["supports_zero_shot_label_refinement"]
        self.preferred_scribble_thickness = caps["preferred_scribble_thickness"]
        self.interaction_decay = caps["interaction_decay"]
        self.INFERENCE_SESSION_VERSION = caps["inference_session_version"]
        self.license: Optional[str] = caps.get("license")
        self.supports_undo: bool = bool(caps.get("supports_undo", False))
        self.do_autozoom: bool = bool(caps.get("do_autozoom", True))

        self.original_image_shape: Optional[tuple] = None
        self.target_buffer: Union[np.ndarray, None] = None
        self._last_paste_bbox: Optional[List[List[int]]] = None
        self.last_interactions: Optional[list] = None     # prompt-mapping log (Phase 6)
        self.last_timings: Optional[dict] = None

    # --------------------------------------------------------------- helpers
    def _get_json(self, path: str) -> dict:
        resp = self._http.get(path)
        self._raise_for_session_errors(resp)
        resp.raise_for_status()
        return resp.json()

    def _post_json(self, path: str, body: dict) -> dict:
        payload = json.dumps(_to_jsonable(body), separators=(",", ":"))
        resp = self._http.post(path, content=payload,
                               headers={"Content-Type": "application/json"})
        self._raise_for_session_errors(resp)
        resp.raise_for_status()
        return resp

    def _post_npy(self, path: str, meta: dict, arr: np.ndarray,
                  timeout: Optional[httpx.Timeout] = None) -> httpx.Response:
        headers = {
            META_HEADER: json.dumps(_to_jsonable(meta), separators=(",", ":")),
            "Content-Type": CONTENT_TYPE_NPY,
        }
        kwargs = {} if timeout is None else {"timeout": timeout}
        resp = self._http.post(path, content=encode_npy(arr), headers=headers, **kwargs)
        self._raise_for_session_errors(resp)
        resp.raise_for_status()
        return resp

    @staticmethod
    def _raise_for_session_errors(resp: httpx.Response) -> None:
        if resp.status_code == 404:
            raise SessionExpiredError(_detail(resp) or "unknown session")
        if resp.status_code == 503:
            raise ServerAtCapacityError(_detail(resp) or "server is at capacity")

    def _apply_prediction(self, resp: httpx.Response) -> None:
        """Write the returned FULL mask into the target buffer; mirror the meta."""
        self._last_paste_bbox = None
        self.last_interactions = None
        self.last_timings = None
        meta_raw = resp.headers.get(META_HEADER)
        if meta_raw is None:
            return
        meta = json.loads(meta_raw)
        self.last_interactions = meta.get("interactions")
        self.last_timings = meta.get("timings")
        self._last_paste_bbox = meta.get("bbox")
        if not meta.get("ran_prediction") or meta.get("mask") is None:
            return
        if self.target_buffer is not None:
            mask = decode_mask(meta["mask"], resp.content)
            self.target_buffer[:] = mask.astype(self.target_buffer.dtype, copy=False)

    # ------------------------------------------------------------------ API
    def initialize_from_trained_model_folder(
        self,
        model_training_output_dir: str,
        use_fold=None,
        checkpoint_name: str = "checkpoint_final.pth",
    ):
        """The service loaded its checkpoint at startup; this call is a no-op."""
        warnings.warn(
            "SamMed3DSession ignores initialize_from_trained_model_folder(): the "
            "service loaded its checkpoint at startup.",
            RuntimeWarning, stacklevel=2,
        )

    def set_image(self, image: np.ndarray, image_properties: Optional[dict] = None) -> None:
        assert image.ndim == 4, f"expected a 4d image as input, got {image.ndim}d. Shape {image.shape}"
        meta = {"image_properties": image_properties or {}}
        resp = self._post_npy("/set_image", meta, image, timeout=self._set_image_timeout)
        info = resp.json()
        self.original_image_shape = tuple(info["original_image_shape"])

    def set_target_buffer(self, target_buffer: np.ndarray) -> None:
        self.target_buffer = target_buffer
        self._post_json("/set_target_buffer",
                        {"shape": list(target_buffer.shape), "dtype": str(target_buffer.dtype)})

    def set_do_autozoom(self, do_autozoom: bool) -> None:
        self.do_autozoom = bool(do_autozoom)   # the service always places its own patches
        self._post_json("/set_do_autozoom", {"do_autozoom": bool(do_autozoom)})

    def reset_interactions(self) -> None:
        if isinstance(self.target_buffer, np.ndarray):
            self.target_buffer.fill(0)
        self._last_paste_bbox = None
        self.last_interactions = None
        self._post_json("/reset_interactions", {})

    def undo(self) -> bool:
        """The service has no interaction undo (supports_undo=False)."""
        return False

    def _predict(self, force_full_refine: bool = False) -> Optional[List[List[int]]]:
        resp = self._post_json("/predict", {"force_full_refine": bool(force_full_refine)})
        self._apply_prediction(resp)
        return self._last_paste_bbox

    def add_point_interaction(
        self,
        coordinates,
        include_interaction: bool,
        run_prediction: bool = True,
        override_capability_checks: bool = False,
    ) -> Optional[List[List[int]]]:
        resp = self._post_json("/add_point_interaction", {
            "coordinates": [float(c) for c in coordinates],
            "include_interaction": bool(include_interaction),
            "run_prediction": bool(run_prediction),
            "override_capability_checks": bool(override_capability_checks),
        })
        self._apply_prediction(resp)
        return self._last_paste_bbox

    def add_bbox_interaction(
        self,
        bbox_coords,
        include_interaction: bool,
        run_prediction: bool = True,
        override_capability_checks: bool = False,
    ) -> Optional[List[List[int]]]:
        resp = self._post_json("/add_bbox_interaction", {
            "bbox_coords": [list(b) for b in bbox_coords],
            "include_interaction": bool(include_interaction),
            "run_prediction": bool(run_prediction),
            "override_capability_checks": bool(override_capability_checks),
        })
        self._apply_prediction(resp)
        return self._last_paste_bbox

    def add_scribble_interaction(
        self,
        scribble_image: np.ndarray,
        include_interaction: bool,
        run_prediction: bool = True,
        override_capability_checks: bool = False,
        interaction_bbox: Optional[List[List[int]]] = None,
    ) -> Optional[List[List[int]]]:
        return self._post_mask_interaction(
            "/add_scribble_interaction", scribble_image, include_interaction,
            run_prediction, override_capability_checks, interaction_bbox)

    def add_lasso_interaction(
        self,
        lasso_image: np.ndarray,
        include_interaction: bool,
        run_prediction: bool = True,
        override_capability_checks: bool = False,
        interaction_bbox: Optional[List[List[int]]] = None,
    ) -> Optional[List[List[int]]]:
        return self._post_mask_interaction(
            "/add_lasso_interaction", lasso_image, include_interaction,
            run_prediction, override_capability_checks, interaction_bbox)

    def _post_mask_interaction(
        self,
        path: str,
        mask_image: np.ndarray,
        include_interaction: bool,
        run_prediction: bool,
        override_capability_checks: bool,
        interaction_bbox: Optional[List[List[int]]],
    ) -> Optional[List[List[int]]]:
        meta = {
            "include_interaction": bool(include_interaction),
            "run_prediction": bool(run_prediction),
            "override_capability_checks": bool(override_capability_checks),
            "interaction_bbox": ([list(b) for b in interaction_bbox]
                                 if interaction_bbox is not None else None),
        }
        resp = self._post_npy(path, meta, mask_image)
        self._apply_prediction(resp)
        return self._last_paste_bbox

    def add_initial_seg_interaction(
        self,
        initial_seg: np.ndarray,
        run_prediction: bool = False,
        override_capability_checks: bool = False,
    ) -> Optional[List[List[int]]]:
        # Mirror the nnInteractive client: the caller's target buffer is overwritten
        # with initial_seg before any prediction runs.
        if self.target_buffer is not None:
            self.target_buffer[:] = np.asarray(initial_seg).astype(
                self.target_buffer.dtype, copy=False)
        meta = {
            "run_prediction": bool(run_prediction),
            "override_capability_checks": bool(override_capability_checks),
        }
        resp = self._post_npy("/add_initial_seg_interaction", meta, initial_seg)
        self._apply_prediction(resp)
        return self._last_paste_bbox

    def ping(self, timeout: float = 5.0) -> bool:
        try:
            resp = self._http.get("/healthz", timeout=timeout)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def close(self) -> None:
        if self._session_id is not None:
            try:
                self._http.delete("/sessions/" + self._session_id,
                                  timeout=httpx.Timeout(5.0))
            except httpx.HTTPError:
                pass
            self._session_id = None
            self._http.headers.pop("X-Sam-Session", None)
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _detail(resp: httpx.Response) -> Optional[str]:
    try:
        data = resp.json()
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(data, dict):
        d = data.get("detail")
        if isinstance(d, str):
            return d
    return None
