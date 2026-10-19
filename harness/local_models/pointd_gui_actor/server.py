#!/usr/bin/env python3
"""pointd-gui-actor: HTTP pointer service around Microsoft GUI-Actor-2B (Phase 5, Study A).

Wraps the OFFICIAL GUI-Actor inference (repo-local Transformers code with the custom
pointer head — a generic Qwen2-VL endpoint would NOT run the attention head):

    POST /point  {image_png_b64, query, k=3, verify=false}
      -> {"points": [{"x": <px>, "y": <px>, "score": <region activation|null>, "rank": 0}, ...],
          "raw": <model raw output / prediction text>, "latency_ms": <float>,
          "model": <model id>, "version": <service version>,
          "image_w": <int>, "image_h": <int>, "verifier": <null | diagnostic object>}

Points are PIXEL coordinates of the image that was sent (the official helper returns
normalized region centres; x*W, y*H here). `score` is the official region mean activation
(a ranking, not a calibrated probability). `k` = number of connected regions requested
(top of the region ranking; the first region is the official `best_point`).

    GET /healthz -> {"status": "ok", "backend": "fake"|"real", "model": ..., "version": ...}

`--fake` mode: deterministic dummy points from a hash of (query, image bytes); torch is
never imported (used by the offline tests and to check the HTTP plumbing).

Real mode expects the official repo checkout (pinned commit, see setup_a100.sh) at
--gui-actor-repo (default $GUI_ACTOR_REPO): imports gui_actor.modeling /
gui_actor.inference and runs Qwen2VLForConditionalGenerationWithPointer in BF16 on the
pinned transformers==4.51.3 + qwen-vl-utils==0.0.8 stack.

Run:  uv run --project harness/local_models/pointd_gui_actor python server.py [--fake]
      (real: python server.py --model-id microsoft/GUI-Actor-2B-Qwen2-VL [--port 8092])
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import logging
import os
import sys
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

SERVICE_VERSION = "1.0.0"
DEFAULT_MODEL_ID = "microsoft/GUI-Actor-2B-Qwen2-VL"
DEFAULT_VERIFIER_MODEL_ID = "microsoft/GUI-Actor-Verifier-2B"
DEFAULT_REPO = os.environ.get("GUI_ACTOR_REPO", "")

log = logging.getLogger("pointd_gui_actor")


# --------------------------------------------------------------------------- schemas
class PointRequest(BaseModel):
    image_png_b64: str = Field(..., description="base64 of one RGB PNG")
    query: str = Field(..., min_length=1, description="grounding instruction / target text")
    k: int = Field(3, ge=1, le=20, description="number of regions -> points")
    verify: bool = Field(False, description="rerank candidates with GUI-Actor-Verifier-2B (real mode)")


# --------------------------------------------------------------------------- backends
class FakeBackend:
    """Deterministic dummy outputs; no torch import anywhere in this class."""

    name = "fake"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, **_: Any) -> None:
        self.model_id = model_id

    def load(self) -> None:  # symmetry with RealBackend.load
        pass

    def predict(self, png_bytes: bytes, query: str, k: int, verify: bool) -> dict[str, Any]:
        seed = int.from_bytes(hashlib.sha256(b"gui-actor|" + query.encode() + b"|" + png_bytes).digest()[:8], "big")
        raw = f"fake gui-actor prediction for query={query!r} (sha256 seed {seed >> 32})"
        # Three pseudo-random but deterministic centres spread across the image; the
        # runner/scorer only needs in-bounds pixels, determinism and ordering.
        points = []
        for rank in range(k):
            h = hashlib.sha256(f"{seed}-{rank}".encode()).digest()
            fx = int.from_bytes(h[0:4], "big") / 2**32
            fy = int.from_bytes(h[4:8], "big") / 2**32
            score = round(0.9 - 0.1 * rank - int.from_bytes(h[8:10], "big") % 10 / 200.0, 4)
            points.append({"x_norm": fx, "y_norm": fy, "score": score})
        return {"points": points, "raw": raw, "verifier": None}


class GuiActorRealBackend:
    """Official GUI-Actor-2B inference (custom pointer head) in BF16.

    Mirrors the official README snippet exactly:
      Qwen2VLForConditionalGenerationWithPointer.from_pretrained(..., torch_dtype=bfloat16)
      inference(conversation, model, tokenizer, processor, use_placeholder=True, topk=k)
    where pred["topk_points"] are the official connected-region centres, normalized (x, y),
    and pred["topk_values"] the region mean activations.
    """

    name = "real"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, repo_dir: str = "", device: str = "cuda:0",
                 attn_implementation: str = "sdpa", verifier_model_id: str = DEFAULT_VERIFIER_MODEL_ID,
                 verifier_enabled: bool = True, **_: Any) -> None:
        self.model_id = model_id
        self.device = device
        self.attn_implementation = attn_implementation
        self.repo_dir = repo_dir
        self.verifier_model_id = verifier_model_id
        self.verifier_enabled = verifier_enabled
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._verifier = None
        self._verifier_processor = None
        self._loaded = False

    def load(self) -> None:
        """Load weights once at startup (called from the FastAPI startup hook)."""
        if self._loaded:
            return
        if self.repo_dir:
            src = os.path.join(self.repo_dir, "src")
            if os.path.isdir(src):
                sys.path.insert(0, src)  # official repo layout: src/gui_actor/...
            elif os.path.isdir(self.repo_dir):
                sys.path.insert(0, self.repo_dir)
        import torch  # noqa: PLC0415 -- real mode only

        from gui_actor.inference import inference as gui_inference  # noqa: PLC0415
        from gui_actor.modeling import Qwen2VLForConditionalGenerationWithPointer  # noqa: PLC0415
        from transformers import Qwen2VLProcessor  # noqa: PLC0415

        self._torch = torch
        self._gui_inference = gui_inference
        self._processor = Qwen2VLProcessor.from_pretrained(self.model_id)
        self._tokenizer = self._processor.tokenizer
        self._model = Qwen2VLForConditionalGenerationWithPointer.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map=self.device,
            attn_implementation=self.attn_implementation,
        ).eval()
        log.info("loaded %s (BF16, attn=%s) on %s", self.model_id, self.attn_implementation, self.device)
        if self.verifier_enabled:
            self._load_verifier(torch)
        self._loaded = True

    def _load_verifier(self, torch) -> None:  # noqa: ANN001
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

            self._verifier = AutoModelForCausalLM.from_pretrained(
                self.verifier_model_id, torch_dtype=torch.bfloat16, device_map=self.device).eval()
            self._verifier_processor = AutoTokenizer.from_pretrained(self.verifier_model_id)
            log.info("loaded verifier %s", self.verifier_model_id)
        except Exception as exc:  # verifier is optional; actor keeps serving without it
            log.warning("verifier load failed (%s); continuing without reranking", exc)
            self._verifier = None

    # -- inference ----------------------------------------------------------
    def predict(self, png_bytes: bytes, query: str, k: int, verify: bool) -> dict[str, Any]:
        import PIL.Image  # noqa: PLC0415

        image = PIL.Image.open(io.BytesIO(png_bytes)).convert("RGB")
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": self._system_message()}]},
            {"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": query}]},
        ]
        pred = self._gui_inference(conversation, self._model, self._tokenizer, self._processor,
                                   use_placeholder=True, topk=k)
        w, h = image.size
        norm_points = [(float(p[0]), float(p[1])) for p in (pred.get("topk_points") or [])]
        scores = [float(v) for v in (pred.get("topk_values") or [])]
        verifier_out = None
        if verify and self._verifier is not None and norm_points:
            order, verifier_out = self._verify(image, query, norm_points, w, h)
            norm_points = [norm_points[i] for i in order]
            scores = [scores[i] for i in order] if scores else scores
        points = [{"x_norm": px, "y_norm": py,
                   "score": (scores[i] if i < len(scores) else None)}
                  for i, (px, py) in enumerate(norm_points)]
        return {"points": points, "raw": str(pred.get("output_text")), "verifier": verifier_out}

    def _system_message(self) -> str:
        from gui_actor.constants import grounding_system_message  # noqa: PLC0415

        return grounding_system_message

    def _verify(self, image, query: str, norm_points: list, w: int, h: int):  # noqa: ANN001
        """Official verifier protocol: red hollow circle at the candidate point ->
        "exame whether the hollow red circle accurately placed ... Answer True or False"."""
        import io as _io  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415
        from PIL import ImageDraw  # noqa: PLC0415

        probs: list[float] = []
        for (px, py) in norm_points:
            img = image.copy()
            draw = ImageDraw.Draw(img)
            radius = 8 * max(img.width, img.height) / 1000.0  # official radius 8 at 1000 px scale
            cx, cy = px * w, py * h
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                         outline="red", width=4)
            buf = _io.BytesIO()
            img.save(buf, format="PNG")
            prompt = ("Please observe the screenshot and exame whether the hollow red circle "
                      f"accurately placed on the intended position in the image: '{query}'. "
                      "Answer True or False.")
            text = self._verifier_processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
                tokenize=False, add_generation_prompt=True)
            inputs = self._verifier_processor(images=[PIL.Image.open(_io.BytesIO(buf.getvalue()))],
                                              text=[text], return_tensors="pt").to(self._verifier.device)
            with self._torch.inference_mode():
                logits = self._verifier(**inputs).logits[0, -1]
            true_id = self._verifier_processor.encode("True", add_special_tokens=False)[0]
            false_id = self._verifier_processor.encode("False", add_special_tokens=False)[0]
            pair = np.array([logits[false_id].item(), logits[true_id].item()], dtype=np.float64)
            pair -= pair.max()
            exp = np.exp(pair)
            probs.append(float(exp[1] / exp.sum()))  # P(True)
        order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        return order, {"used": True, "p_true": [round(p, 4) for p in probs]}


# --------------------------------------------------------------------------- app
def create_app(backend: FakeBackend | GuiActorRealBackend) -> FastAPI:
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        backend.load()  # load weights once (real backend); no-op for --fake
        yield

    app = FastAPI(title="pointd-gui-actor", version=SERVICE_VERSION, lifespan=lifespan)

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

        t0 = time.perf_counter()
        try:
            out = backend.predict(png, req.query, req.k, req.verify)
        except HTTPException:
            raise
        except Exception as exc:
            log.exception("prediction failed")
            raise HTTPException(status_code=500, detail=f"prediction failed: {exc}") from exc
        latency_ms = (time.perf_counter() - t0) * 1000.0

        points = []
        for rank, p in enumerate(out["points"][: req.k]):
            x = round(float(p["x_norm"]) * w, 2)
            y = round(float(p["y_norm"]) * h, 2)
            score = p.get("score")
            points.append({"x": x, "y": y, "score": score, "rank": rank})
        return {"points": points, "raw": out["raw"], "latency_ms": round(latency_ms, 3),
                "model": backend.model_id, "version": SERVICE_VERSION,
                "image_w": w, "image_h": h, "verifier": out.get("verifier")}

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fake", action="store_true", help="deterministic dummy backend; no torch import")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--attn-implementation", default="sdpa",
                    help="sdpa (default, validated) or flash_attention_2 (official snippet; "
                         "needs the flash-attn pin)")
    ap.add_argument("--gui-actor-repo", default=DEFAULT_REPO,
                    help="official microsoft/GUI-Actor checkout (pinned commit, setup_a100.sh); "
                         "default $GUI_ACTOR_REPO")
    ap.add_argument("--verifier-model-id", default=DEFAULT_VERIFIER_MODEL_ID)
    ap.add_argument("--no-verifier", action="store_true", help="do not load GUI-Actor-Verifier-2B")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if args.fake:
        backend: FakeBackend | GuiActorRealBackend = FakeBackend(model_id=args.model_id)
    else:
        backend = GuiActorRealBackend(model_id=args.model_id, repo_dir=args.gui_actor_repo,
                                      device=args.device, attn_implementation=args.attn_implementation,
                                      verifier_model_id=args.verifier_model_id,
                                      verifier_enabled=not args.no_verifier)
    import uvicorn  # noqa: PLC0415

    uvicorn.run(create_app(backend), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
