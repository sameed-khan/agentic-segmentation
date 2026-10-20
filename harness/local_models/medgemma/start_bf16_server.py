#!/usr/bin/env python3
"""OpenAI-compatible MedGemma server: Transformers BF16 reference runtime (port 8094).

Implements the small OpenAI surface the harness needs — `POST /v1/chat/completions`
(text + base64 `image_url` content, `temperature`, `max_tokens`) and `GET /healthz` —
on top of the official HF-card loading path for google/medgemma-1.5-4b-it:

    AutoModelForImageTextToText.from_pretrained(..., torch_dtype=bfloat16)  (896 px, 256 tokens/image)
    processor.apply_chat_template(...); model.generate(..., do_sample=False)

This is the full-precision REFERENCE arm for the Q4_K_M llama.cpp deployment arm
(port 8091); run both on the design set to quantify the quantization effect.
MedGemma has no system instruction: every message is passed through as given.

`--fake` mode: deterministic canned JSON replies, no torch import (offline tests).

Run:  uv run --project harness/local_models/medgemma python start_bf16_server.py [--fake]
      (real: python start_bf16_server.py --model-id google/medgemma-1.5-4b-it --port 8094)
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
import uuid
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

SERVICE_VERSION = "1.0.1"   # 1.0.1: process-wide inference lock (see /v1/chat/completions below)
DEFAULT_MODEL_ID = "google/medgemma-1.5-4b-it"

import threading

_INFER_LOCK = threading.Lock()

log = logging.getLogger("medgemma_bf16_server")


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    temperature: float | None = 0.0
    max_tokens: int | None = 256


# --------------------------------------------------------------------------- backends
class FakeBackend:
    """Deterministic canned replies; no torch import anywhere in this class."""

    name = "fake"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, delay_s: float = 0.0, **_: Any) -> None:
        self.model_id = model_id
        self.delay_s = delay_s  # test-only hook: slow generate() so lock tests can overlap

    def load(self) -> None:
        pass

    def generate(self, prompt_text: str, images_b64: list[str], temperature: float,
                 max_tokens: int) -> str:
        if self.delay_s:
            time.sleep(self.delay_s)
        seed = int.from_bytes(hashlib.sha256(b"medgemma|" + prompt_text.encode()
                                             + b"|".join(i.encode() for i in images_b64)).digest()[:8], "big")
        # A valid JSON reply with the explicit `uncertain` option, deterministic per input.
        return f'{{"answer": "uncertain", "confidence": {seed % 100 / 100:.2f}, "reason": "fake"}}'


class MedGemmaRealBackend:
    """Official HF-card path: AutoModelForImageTextToText, BF16, deterministic decoding."""

    name = "real"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = "cuda:0", **_: Any) -> None:
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        import torch  # noqa: PLC0415 -- real mode only
        from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map=self.device).eval()
        log.info("loaded %s (BF16) on %s", self.model_id, self.device)
        self._loaded = True

    def generate(self, prompt_text: str, images_b64: list[str], temperature: float,
                 max_tokens: int) -> str:
        import PIL.Image  # noqa: PLC0415

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        pil_images = []
        for b64 in images_b64:
            img = PIL.Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            pil_images.append(img)
            content.append({"type": "image", "image": img})
        messages = [{"role": "user", "content": content}]
        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self._model.device)
        do_sample = bool(temperature and temperature > 0)
        with self._torch.inference_mode():
            out = self._model.generate(**inputs, max_new_tokens=max_tokens, do_sample=do_sample)
        return self._processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:],
                                                skip_special_tokens=True)


# --------------------------------------------------------------------------- app
def _content_parts(message: dict[str, Any]) -> tuple[str, list[str]]:
    """OpenAI message -> (text, [base64 png]) across str / part-list content."""
    content = message.get("content")
    if isinstance(content, str):
        return content, []
    text_parts: list[str] = []
    images: list[str] = []
    for part in content or []:
        ptype = part.get("type")
        if ptype == "text":
            text_parts.append(str(part.get("text", "")))
        elif ptype == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            b64 = url.split("base64,", 1)[1] if "base64," in url else ""
            if not b64:
                raise ValueError("image_url without base64 data is not supported")
            images.append(b64)
        elif ptype == "image":
            images.append(str(part.get("b64_json", "")))  # convenience alias
    return "\n".join(text_parts), images


def create_app(backend: FakeBackend | MedGemmaRealBackend, served_model_id: str) -> FastAPI:
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        backend.load()  # load weights once (real backend); no-op for --fake
        yield

    app = FastAPI(title="medgemma-bf16-server", version=SERVICE_VERSION, lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "backend": backend.name, "model": served_model_id,
                "version": SERVICE_VERSION}

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest) -> JSONResponse:
        if req.temperature and req.temperature > 0:
            log.warning("non-zero temperature %s: local study calls are deterministic", req.temperature)
        texts: list[str] = []
        all_images: list[str] = []
        for message in req.messages:
            text, images = _content_parts(message)
            texts.append(text)
            all_images.extend(images)
        prompt_text = "\n".join(t for t in texts if t)
        try:
            base64.b64decode(all_images[0], validate=False) if all_images else None
        except (binascii.Error, ValueError) as exc:
            return JSONResponse(status_code=400, content={"error": {"message": f"invalid base64 image: {exc}"}})
        # One inference at a time: HF tokenizers/models are not thread-safe (FastAPI runs
        # sync endpoints in a threadpool; concurrent requests raised 'Already borrowed').
        # latency_ms is measured inside the lock = pure inference time (queue wait excluded).
        with _INFER_LOCK:
            t0 = time.perf_counter()
            try:
                out_text = backend.generate(prompt_text, all_images, req.temperature or 0.0,
                                            req.max_tokens or 256)
            except Exception as exc:
                log.exception("generation failed")
                return JSONResponse(status_code=500, content={"error": {"message": f"generation failed: {exc}"}})
            latency_ms = (time.perf_counter() - t0) * 1000.0
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        payload = {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": served_model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": out_text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": len(prompt_text) // 4, "completion_tokens": len(out_text) // 4,
                      "total_tokens": (len(prompt_text) + len(out_text)) // 4},
            "latency_ms": round(latency_ms, 3),   # extension field, logged by the client
        }
        return JSONResponse(content=payload)

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fake", action="store_true", help="deterministic dummy backend; no torch import")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8094)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    backend: FakeBackend | MedGemmaRealBackend = (
        FakeBackend(model_id=args.model_id) if args.fake
        else MedGemmaRealBackend(model_id=args.model_id, device=args.device))
    import uvicorn  # noqa: PLC0415

    uvicorn.run(create_app(backend, args.model_id), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
