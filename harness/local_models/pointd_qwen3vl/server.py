#!/usr/bin/env python3
"""pointd-qwen3vl: HTTP pointer service around Qwen/Qwen3-VL-8B-Instruct (Phase 5, Study A).

Same API as pointd-molmo2 / pointd-gui-actor (port 8096), with one documented
addition (see `format_instruction` below):

    POST /point  {image_png_b64, query, k=3, max_new_tokens?}
      -> {"points": [{"x": <px>, "y": <px>, "score": null, "rank": 0, "point_id": null}, ...],
          "raw": <model raw output>, "latency_ms": <float>, "model": <model id>,
          "version": <service version>, "image_w": <int>, "image_h": <int>, "verifier": null,
          "format_instruction": <the fixed suffix appended to every query>}
    GET /healthz {status, backend, model, version}

Output format (native Qwen3-VL grounding convention, HF card + QwenLM/Qwen3-VL
cookbook 2d_grounding.ipynb): a JSON array of {"point_2d": [x, y], "label": ...}
objects with coordinates on the RELATIVE 0-1000 scale (Qwen3-VL's default
coordinate system); parsed by the pure parser qwen3vl_points.py (no torch) and
converted to input-image pixels (x/1000*W, y/1000*H). A {"bbox_2d": [x1, y1, x2, y2]}
answer is tolerated and reduced to its centre. The study `query` is already a full
instruction (e.g. 'Point to the fibrous soft-tissue tumor.'); the service APPENDS
a short fixed output-format instruction (FORMAT_INSTRUCTION below — request the
JSON point_2d array on the 0-1000 scale) and echoes it in every response as
`format_instruction`. Qwen3-VL emits no calibrated confidence: `score` is always
null; `k` caps the RETURNED points only (rank = emission order).

Loading (card path): Qwen3VLForConditionalGeneration, dtype="auto" on the card is
BF16 here — the explicit dtype=torch.bfloat16 required by the harness Study A
protocol matches the checkpoint dtype; attn_implementation="sdpa" (the card
recommends flash_attention_2, which would need a source build); device_map pinned
to --device (card says "auto"; single-GPU box -> explicit placement). Decoding is
deterministic (do_sample=False), overriding the card's sampling suggestion —
required by the study's determinism/resume protocol.

--fake mode: deterministic dummy JSON point output parsed by qwen3vl_points.py;
no torch import.

Run:  uv run --project harness/local_models/pointd_qwen3vl python server.py [--fake]
      (real: python server.py --model-id <weights dir> [--port 8096])
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import json
import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from qwen3vl_points import Qwen3VLPoint, parse_qwen3vl_points

SERVICE_VERSION = "1.0.1"   # matches the other pointd services (process-wide inference lock)
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 256

# Fixed output-format instruction appended to EVERY query (native Qwen3-VL point
# convention: JSON array of {"point_2d": [x, y], "label": ...}, 0-1000 relative
# coordinates; wording follows the official cookbook's point-grounding prompts).
FORMAT_INSTRUCTION = (
    ' Report the point(s) as a JSON array only, one object per point, exactly like '
    '{"point_2d": [x, y], "label": "target"}: x and y are integer coordinates on a '
    '0-1000 scale relative to the image width and height.'
)

import threading

_INFER_LOCK = threading.Lock()

log = logging.getLogger("pointd_qwen3vl")


class PointRequest(BaseModel):
    image_png_b64: str = Field(..., description="base64 of one RGB PNG")
    query: str = Field(..., min_length=1, description="pointing instruction (FORMAT_INSTRUCTION is appended)")
    k: int = Field(3, ge=1, le=20, description="max points returned (emission-order cap)")
    max_new_tokens: int = Field(DEFAULT_MAX_NEW_TOKENS, ge=8, le=2048)


def user_text(query: str) -> str:
    """The full user prompt: the study query + the fixed output-format instruction."""
    return query + FORMAT_INSTRUCTION


# --------------------------------------------------------------------------- backends
class FakeBackend:
    """Deterministic dummy outputs; no torch import anywhere in this class."""

    name = "fake"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, delay_s: float = 0.0, **_: Any) -> None:
        self.model_id = model_id
        self.delay_s = delay_s  # test-only hook: slow predict() so lock tests can overlap

    def load(self) -> None:
        pass

    def predict(self, png_bytes: bytes, query: str, k: int, max_new_tokens: int) -> dict[str, Any]:
        if self.delay_s:
            time.sleep(self.delay_s)
        seed = int.from_bytes(hashlib.sha256(b"qwen3vl|" + query.encode() + b"|" + png_bytes).digest()[:8], "big")
        # Raw text in the NATIVE Qwen3-VL point-grounding JSON format (0-1000 scale)
        # wrapped in a ```json fence so the official fence-stripping path is exercised.
        items = []
        for rank in range(max(k, 1)):
            h = hashlib.sha256(f"{seed}-{rank}".encode()).digest()
            fx = int.from_bytes(h[0:4], "big") / 2**32
            fy = int.from_bytes(h[4:8], "big") / 2**32
            items.append({"point_2d": [min(round(fx * 1000), 1000), min(round(fy * 1000), 1000)],
                          "label": "target"})
        raw = "```json\n" + json.dumps(items) + "\n```"
        return {"raw": raw, "image_size": None, "extra": {}}


class Qwen3VLRealBackend:
    """Official HF-card inference path (Qwen3VLForConditionalGeneration + AutoProcessor)."""

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
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # noqa: PLC0415

        self._torch = torch
        # Card quickstart (Qwen3-VL-8B-Instruct): explicit bfloat16 == the checkpoint's
        # dtype ("auto" on the card); sdpa instead of the card's flash_attention_2
        # suggestion (no source build on the A100).
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map=self.device,
            attn_implementation="sdpa").eval()
        log.info("loaded %s (BF16, sdpa) on %s", self.model_id, self.device)
        self._loaded = True

    def predict(self, png_bytes: bytes, query: str, k: int, max_new_tokens: int) -> dict[str, Any]:
        import PIL.Image  # noqa: PLC0415

        image = PIL.Image.open(io.BytesIO(png_bytes)).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text(query)},
        ]}]
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt")
        inputs = inputs.to(self._model.device)
        with self._torch.inference_mode():
            generated = self._model.generate(**inputs, max_new_tokens=max_new_tokens,
                                             do_sample=False)
        generated_ids = generated[0, inputs["input_ids"].shape[1]:]
        raw = self._processor.batch_decode([generated_ids], skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0]
        return {"raw": raw, "image_size": image.size, "extra": {}}


# --------------------------------------------------------------------------- app
def create_app(backend: FakeBackend | Qwen3VLRealBackend) -> FastAPI:
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        backend.load()  # load weights once (real backend); no-op for --fake
        yield

    app = FastAPI(title="pointd-qwen3vl", version=SERVICE_VERSION, lifespan=lifespan)

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

        # One inference at a time (same rationale as pointd-molmo2): the model/tokenizer
        # are not thread-safe; latency_ms is measured inside the lock = pure inference
        # time (queue wait excluded).
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

        parsed: list[Qwen3VLPoint] = parse_qwen3vl_points(out["raw"], w, h)
        points = [{"x": round(p.x, 2), "y": round(p.y, 2), "score": None,
                   "rank": p.rank, "point_id": None} for p in parsed[: req.k]]
        return {"points": points, "raw": out["raw"], "latency_ms": round(latency_ms, 3),
                "model": backend.model_id, "version": SERVICE_VERSION,
                "image_w": w, "image_h": h, "verifier": None,
                "format_instruction": FORMAT_INSTRUCTION}

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fake", action="store_true", help="deterministic dummy backend; no torch import")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8096)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if args.fake:
        backend: FakeBackend | Qwen3VLRealBackend = FakeBackend(model_id=args.model_id)
    else:
        backend = Qwen3VLRealBackend(model_id=args.model_id, device=args.device,
                                     max_new_tokens=args.max_new_tokens)
    import uvicorn  # noqa: PLC0415

    uvicorn.run(create_app(backend), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
