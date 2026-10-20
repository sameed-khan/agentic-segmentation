#!/usr/bin/env python3
"""pointd-molmo2: HTTP pointer service around AI2 Molmo2-8B (Phase 5, Study A).

Same API as pointd-gui-actor:

    POST /point  {image_png_b64, query, k=3}
      -> {"points": [{"x": <px>, "y": <px>, "score": null, "rank": 0, "point_id": <int|null>}, ...],
          "raw": <model raw output>, "latency_ms": <float>, "model": <model id>,
          "version": <service version>, "image_w": <int>, "image_h": <int>, "verifier": null}

- Points are PIXEL coordinates of the image that was sent, parsed from Molmo2's
  `<points coords="...">` output (0-1000 normalized) by molmo_points.py.
- Molmo2 emits no calibrated confidence: `score` is always null (use `rank` =
  emission order). `k` caps the RETURNED points only — it does not control
  generation (no documented request-time k constraint).
- Transformers 4.57.1, trust_remote_code=True, EXPLICIT torch_dtype bfloat16
  (the checkpoint is FP32; `dtype="auto"` would need ~35 GB — see the research note).

`--fake` mode: deterministic dummy points, no torch import (offline tests).

Run:  uv run --project harness/local_models/pointd_molmo2 python server.py [--fake]
      (real: python server.py --model-id allenai/Molmo2-8B [--port 8093])
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import logging
import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from molmo_points import MolmoPoint, parse_molmo_points

SERVICE_VERSION = "1.0.0"
DEFAULT_MODEL_ID = "allenai/Molmo2-8B"
DEFAULT_MAX_NEW_TOKENS = 256

import threading

_INFER_LOCK = threading.Lock()

log = logging.getLogger("pointd_molmo2")


class PointRequest(BaseModel):
    image_png_b64: str = Field(..., description="base64 of one RGB PNG")
    query: str = Field(..., min_length=1, description="pointing instruction / target text")
    k: int = Field(3, ge=1, le=20, description="max points returned (emission-order cap)")
    max_new_tokens: int = Field(DEFAULT_MAX_NEW_TOKENS, ge=8, le=2048)


# --------------------------------------------------------------------------- backends
class FakeBackend:
    """Deterministic dummy outputs; no torch import anywhere in this class."""

    name = "fake"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, **_: Any) -> None:
        self.model_id = model_id

    def load(self) -> None:
        pass

    def predict(self, png_bytes: bytes, query: str, k: int, max_new_tokens: int) -> dict[str, Any]:
        seed = int.from_bytes(hashlib.sha256(b"molmo2|" + query.encode() + b"|" + png_bytes).digest()[:8], "big")
        # Raw text in Molmo2's own triplet format (`id x y`, 0-1000 scale) so the official
        # parser path is exercised end-to-end by every request.
        coords = []
        for rank in range(max(k, 1)):
            h = hashlib.sha256(f"{seed}-{rank}".encode()).digest()
            fx = int.from_bytes(h[0:4], "big") / 2**32
            fy = int.from_bytes(h[4:8], "big") / 2**32
            coords.append((fx, fy))
        raw = "Here are the points. <points coords=\"" + " ".join(
            f"{i} {min(round(fx * 1000), 999):03d} {min(round(fy * 1000), 999):03d}" for i, (fx, fy) in enumerate(coords)
        ) + "\">target</points>"
        return {"raw": raw, "image_size": None, "extra": {}}


class Molmo2RealBackend:
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
        # Explicit bfloat16: checkpoint weights are FP32 (dtype "auto" would need ~35 GB);
        # Ai2 confirms BF16 loading is safe (research note, pointer_models doc §Molmo2).
        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, trust_remote_code=True, torch_dtype=torch.bfloat16,
            device_map=self.device, attn_implementation="sdpa").eval()
        log.info("loaded %s (BF16, sdpa) on %s", self.model_id, self.device)
        self._loaded = True

    def predict(self, png_bytes: bytes, query: str, k: int, max_new_tokens: int) -> dict[str, Any]:
        import PIL.Image  # noqa: PLC0415

        image = PIL.Image.open(io.BytesIO(png_bytes)).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": query},
        ]}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(images=[image], text=[text], return_tensors="pt")
        inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            generated = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        raw = self._processor.tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:],
                                               skip_special_tokens=True)
        return {"raw": raw, "image_size": image.size, "extra": {}}


# --------------------------------------------------------------------------- app
def create_app(backend: FakeBackend | Molmo2RealBackend) -> FastAPI:
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        backend.load()  # load weights once (real backend); no-op for --fake
        yield

    app = FastAPI(title="pointd-molmo2", version=SERVICE_VERSION, lifespan=lifespan)

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

        # One inference at a time: HF tokenizers/models are not thread-safe (FastAPI runs
        # sync endpoints in a threadpool; concurrent requests raised 'Already borrowed').
        # latency_ms is measured inside the lock = pure inference time (queue wait excluded).
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

        parsed: list[MolmoPoint] = parse_molmo_points(out["raw"], w, h)
        points = [{"x": round(p.x, 2), "y": round(p.y, 2), "score": None,
                   "rank": p.rank, "point_id": p.point_id} for p in parsed[: req.k]]
        return {"points": points, "raw": out["raw"], "latency_ms": round(latency_ms, 3),
                "model": backend.model_id, "version": SERVICE_VERSION,
                "image_w": w, "image_h": h, "verifier": None}

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fake", action="store_true", help="deterministic dummy backend; no torch import")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if args.fake:
        backend: FakeBackend | Molmo2RealBackend = FakeBackend(model_id=args.model_id)
    else:
        backend = Molmo2RealBackend(model_id=args.model_id, device=args.device,
                                    max_new_tokens=args.max_new_tokens)
    import uvicorn  # noqa: PLC0415

    uvicorn.run(create_app(backend), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
