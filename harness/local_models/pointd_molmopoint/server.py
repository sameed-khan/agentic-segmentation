#!/usr/bin/env python3
"""pointd-molmopoint: HTTP pointer service around AI2 MolmoPoint-8B (Phase 5, Study A).

Same API as pointd-molmo2 / pointd-gui-actor (port 8095):

    POST /point  {image_png_b64, query, k=3, max_new_tokens?}
      -> {"points": [{"x": <px>, "y": <px>, "score": null, "rank": 0, "point_id": <int>}, ...],
          "raw": <decoded grounding-token text>, "latency_ms": <float>, "model": <model id>,
          "version": <service version>, "image_w": <int>, "image_h": <int>, "verifier": null}
    GET /healthz {status, backend, model, version}

MolmoPoint does not emit text coordinates: each point is generated as grounding
tokens (<POINT_a> ViT-patch, <POINT_b> subpatch, <POINT_c> 3x3 location, then the
object id as digits) and must be decoded with the OFFICIAL pointing API
(allenai/MolmoPoint-8B card, followed exactly):

    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True, padding=True,
                                           return_pointing_metadata=True)
    metadata = inputs.pop("metadata")
    output = model.generate(**inputs, logits_processor=model.build_logit_processor_from_inputs(inputs),
                            max_new_tokens=..., do_sample=False)   # greedy
    text = processor.post_process_image_text_to_text(tokens, skip_special_tokens=False, ...)
    points = model.extract_image_points(text, metadata["token_pooling"],
                                        metadata["subpatch_mapping"], metadata["image_sizes"])
    # -> [[object_id, image_index, x_px, y_px], ...] in ORIGINAL-image pixels; this single-image
    #    service serves image_index 0.

The study `query` is already a full instruction (e.g. 'Point to the fibrous
soft-tissue tumor.') and is used verbatim — no output-format suffix (the card's
prompts are bare instructions; the grounding-token constraint comes from the
logit processor). MolmoPoint emits no calibrated confidence: `score` is always
null; `k` caps the RETURNED points only (rank = emission order).

Loading: card path with one documented override — dtype=torch.bfloat16 EXPLICIT
(the checkpoint's config dtype is float32, so the card's `dtype="auto"` would need
~35 GB; same rationale as pointd-molmo2), while generation still runs under the
card's `torch.autocast("cuda", dtype=torch.bfloat16)`. sdpa attention
(the checkpoint config already records sdpa); device_map pinned to --device
(card says "auto"; single-GPU box -> explicit placement).

--fake mode: deterministic dummy grounding-token output decoded by the pure parser
molmopoint_points.py (mirrors the official decode without torch; no torch import).

Run:  uv run --project harness/local_models/pointd_molmopoint python server.py [--fake]
      (real: python server.py --model-id <weights dir> [--port 8095])
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from molmopoint_points import MolmoGroundingPoint, decode_image_points

SERVICE_VERSION = "1.0.1"   # matches the other pointd services (process-wide inference lock)
DEFAULT_MODEL_ID = "allenai/MolmoPoint-8B"
DEFAULT_MAX_NEW_TOKENS = 200  # official card example

import threading

_INFER_LOCK = threading.Lock()

log = logging.getLogger("pointd_molmopoint")


class PointRequest(BaseModel):
    image_png_b64: str = Field(..., description="base64 of one RGB PNG")
    query: str = Field(..., min_length=1, description="pointing instruction (used verbatim)")
    k: int = Field(3, ge=1, le=20, description="max points returned (emission-order cap)")
    max_new_tokens: int = Field(DEFAULT_MAX_NEW_TOKENS, ge=8, le=2048)


# --------------------------------------------------------------------------- backends
class FakeBackend:
    """Deterministic dummy grounding-token output; no torch import anywhere in this class."""

    name = "fake"

    # Synthetic pointing geometry (shared by raw emission and decode): a 2x3 grid of
    # ViT patches (ids 0-5), one subpatch per patch token, no_more_points class on.
    N_COLS, N_ROWS = 3, 2
    POOLING = [[i] for i in range(N_ROWS * N_COLS)]        # [n_patch_rows][n_subpatches]
    MAPPING = [[0, 1, 2], [3, 4, 5]]                       # grid cell -> ViT patch id
    NO_MORE_POINTS_CLASS = True
    PATCH_LOCATION = "3x3"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, delay_s: float = 0.0, **_: Any) -> None:
        self.model_id = model_id
        self.delay_s = delay_s  # test-only hook: slow predict() so lock tests can overlap

    def load(self) -> None:
        pass

    def predict(self, png_bytes: bytes, query: str, k: int, max_new_tokens: int) -> dict[str, Any]:
        if self.delay_s:
            time.sleep(self.delay_s)
        seed = int.from_bytes(hashlib.sha256(b"molmopoint|" + query.encode() + b"|" + png_bytes).digest()[:8], "big")
        n_patches = self.N_ROWS * self.N_COLS
        # Token numbering mirrors the official decoder: patch rows 0..5, then the
        # no_more_points token (<POINT_6>), then subpatch numbers (7..), then locations.
        n_after_patches = n_patches + (1 if self.NO_MORE_POINTS_CLASS else 0)
        spans, spans_metadata = [], []
        for rank in range(max(k, 1)):
            h = hashlib.sha256(f"{seed}-{rank}".encode()).digest()
            fx = int.from_bytes(h[0:4], "big") / 2**32
            fy = int.from_bytes(h[4:8], "big") / 2**32
            col, row = int(fx * self.N_COLS), int(fy * self.N_ROWS)  # coarse grid cell
            vit_id = row * self.N_COLS + col
            subpatch_num = 0 + n_after_patches                       # first subpatch of the row
            location_num = int(h[8] % 9) + n_after_patches + 1       # one of the 9 locations
            spans.append(f"<POINT_{vit_id}> <POINT_{subpatch_num}> <POINT_{location_num}> 0")
            spans_metadata.append((vit_id, subpatch_num, location_num, 0))
        raw = "".join(s + " " for s in spans) + f"<POINT_{n_after_patches}>"  # no_more_points stop
        return {"raw": raw, "image_size": None,
                "extra": {"pointing_metadata": {
                    "pooling": self.POOLING, "mappings": [self.MAPPING],
                    "no_more_points_class": self.NO_MORE_POINTS_CLASS,
                    "patch_location": self.PATCH_LOCATION}}}


class MolmoPointRealBackend:
    """Official HF-card inference path (AutoModelForImageTextToText, trust_remote_code)."""

    name = "real"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = "cuda:0",
                 max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, **_: Any) -> None:
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        import torch  # noqa: PLC0415 -- real mode only
        from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

        self._torch = torch
        # Card: AutoProcessor(..., trust_remote_code=True, padding_side="left").
        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True,
                                                        padding_side="left")
        # Explicit bfloat16: the checkpoint's config dtype is float32 (the card's
        # dtype="auto" would need ~35 GB); generation runs under the card's autocast below.
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, trust_remote_code=True, dtype=torch.bfloat16,
            device_map=self.device, attn_implementation="sdpa").eval()
        log.info("loaded %s (BF16, sdpa) on %s", self.model_id, self.device)
        self._loaded = True

    def predict(self, png_bytes: bytes, query: str, k: int, max_new_tokens: int) -> dict[str, Any]:
        import PIL.Image  # noqa: PLC0415

        image = PIL.Image.open(io.BytesIO(png_bytes)).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "text", "text": query},
            {"type": "image", "image": image},
        ]}]
        # Official card decode path: pointing metadata from the preprocessor, constrained
        # generation via the model's logit processor, special tokens KEPT in the text.
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
            return_dict=True, padding=True, return_pointing_metadata=True)
        metadata = inputs.pop("metadata")
        inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
        with self._torch.inference_mode(), self._torch.autocast("cuda", dtype=self._torch.bfloat16):
            output = self._model.generate(
                **inputs,
                logits_processor=self._model.build_logit_processor_from_inputs(inputs),
                max_new_tokens=max_new_tokens,
                do_sample=False)
        generated_tokens = output[:, inputs["input_ids"].size(1):]
        raw = self._processor.post_process_image_text_to_text(
            generated_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
        decoded = self._model.extract_image_points(
            raw, metadata["token_pooling"], metadata["subpatch_mapping"], metadata["image_sizes"])
        return {"raw": raw, "image_size": image.size, "extra": {"decoded_points": decoded}}


# --------------------------------------------------------------------------- app
def create_app(backend: FakeBackend | MolmoPointRealBackend) -> FastAPI:
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        backend.load()  # load weights once (real backend); no-op for --fake
        yield

    app = FastAPI(title="pointd-molmopoint", version=SERVICE_VERSION, lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "backend": backend.name, "model": backend.model_id,
                "version": SERVICE_VERSION}

    @app.post("/point")
    def point(req: PointRequest) -> dict[str, Any]:
        try:
            png = base64.b64decode(req.image_png_b64, validate=False)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"invalid base64 image: {exc}") from exc
        import PIL.Image  # noqa: PLC0415

        try:
            img = PIL.Image.open(io.BytesIO(png))
            img.verify()
            w, h = PIL.Image.open(io.BytesIO(png)).size
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid PNG: {exc}") from exc
        if w <= 0 or h <= 0:
            raise HTTPException(status_code=400, detail="degenerate image size")

        # One inference at a time (same rationale as pointd-molmo2): the model/logit
        # processor are not thread-safe; latency_ms is measured inside the lock =
        # pure inference time (queue wait excluded).
        with _INFER_LOCK:
            t0 = time.perf_counter()
            try:
                out = backend.predict(png, req.query, req.k, req.max_new_tokens)
            except HTTPException:
                raise
            except Exception as exc:
                log.exception("prediction failed")
                raise HTTPException(status_code=500, detail=f"prediction failed: {exc}") from exc
            latency_ms = (time.perf_counter() - t0) * 1000.0

        extra = out.get("extra") or {}
        decoded = extra.get("decoded_points")
        if decoded is None:
            meta = extra["pointing_metadata"]  # fake backend: decode via the pure parser
            decoded = [(p.point_id, p.image_ix, p.x, p.y) for p in decode_image_points(
                out["raw"], meta["pooling"], meta["mappings"], [(w, h)],
                no_more_points_class=meta["no_more_points_class"],
                patch_location=meta["patch_location"])]
        # Single-image service: keep image_index 0 points in emission order.
        points: list[dict[str, Any]] = []
        for p_id, image_ix, x, y in decoded:
            if image_ix != 0:
                continue
            points.append({"x": round(float(x), 2), "y": round(float(y), 2), "score": None,
                           "rank": len(points), "point_id": int(p_id)})
        return {"points": points[: req.k], "raw": out["raw"], "latency_ms": round(latency_ms, 3),
                "model": backend.model_id, "version": SERVICE_VERSION,
                "image_w": w, "image_h": h, "verifier": None}

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fake", action="store_true", help="deterministic dummy backend; no torch import")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if args.fake:
        backend: FakeBackend | MolmoPointRealBackend = FakeBackend(model_id=args.model_id)
    else:
        backend = MolmoPointRealBackend(model_id=args.model_id, device=args.device,
                                        max_new_tokens=args.max_new_tokens)
    import uvicorn  # noqa: PLC0415

    uvicorn.run(create_app(backend), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
