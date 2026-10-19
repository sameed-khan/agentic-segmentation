#!/usr/bin/env python3
"""Thin MedGemma client (Phase 5): OpenAI chat format, base64 images, temperature 0.

Talks to any OpenAI-compatible server hosting google/medgemma-1.5-4b-it:
llama.cpp `llama-server` (Q4_K_M GGUF, port 8091, via start_llama_server.sh) or the
Transformers BF16 server (port 8094, via start_bf16_server.py / start_bf16_vllm.sh).

- One user message per call: prompt text + N images as base64 `image_url` content.
  MedGemma has NO system instruction (technical report App. B) — everything goes in
  the user text.
- temperature 0 / deterministic decoding by default so local results are reproducible.
- `chat` returns the raw completion; `chat_json` STRICTLY parses the reply as JSON
  with an explicit `uncertain` option (the harness output contract: strict JSON with
  an explicit uncertain/abstain value; parse failures are counted as errors, never
  retried). The raw reply is always kept on the result for logging.

Standard library only (urllib) so it runs in any venv (study runner, tests, shell).
"""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TIMEOUT_S = 600.0


def image_to_data_url(image) -> str:
    """PIL Image | bytes | path -> `data:image/png;base64,...` for OpenAI image_url."""
    buf = None
    if isinstance(image, (bytes, bytearray)):
        buf = bytes(image)
    elif isinstance(image, (str, Path)):
        buf = Path(image).read_bytes()
    else:  # PIL Image (import lazily; PIL is optional for text-only use)
        import io

        import PIL.Image as PILImage  # noqa: PLC0415

        out = io.BytesIO()
        pil = PILImage.fromarray(image) if hasattr(image, "__array__") else image
        pil.convert("RGB").save(out, format="PNG")
        buf = out.getvalue()
    return "data:image/png;base64," + base64.b64encode(buf).decode("ascii")


@dataclass
class ChatResult:
    """One completion: raw reply text, parsed JSON (when requested), and the log."""

    text: str
    raw_body: str                    # full HTTP JSON body (for the run log)
    latency_ms: float
    model: str
    usage: dict | None = None
    parsed: dict | None = None       # chat_json only: parsed JSON, None on parse failure
    parse_error: str | None = None   # chat_json only: why parsing failed
    uncertain: bool = False          # chat_json only: explicit "uncertain" reply
    extra: dict = field(default_factory=dict)


def extract_json_object(text: str) -> dict:
    """Strictly extract a JSON object from a model reply.

    Order: whole text -> ```json fenced block -> first balanced {...} block. The
    fenced/block fallbacks absorb formatting noise; anything non-JSON fails (the
    scorer counts it as a parse failure — never retried).
    """
    if text is None:
        raise ValueError("empty reply")
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
        raise ValueError(f"reply is JSON but not an object: {type(obj).__name__}")
    except json.JSONDecodeError:
        pass
    import re

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fence:
        return _load_object(fence.group(1), "fenced block")
    start = stripped.find("{")
    if start >= 0:
        depth = 0
        in_str = False
        esc = False
        for idx in range(start, len(stripped)):
            ch = stripped[idx]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return _load_object(stripped[start:idx + 1], "balanced block")
    raise ValueError("no JSON object found in reply")


def _load_object(candidate: str, source: str) -> dict:
    obj = json.loads(candidate)
    if not isinstance(obj, dict):
        raise ValueError(f"{source} is JSON but not an object")
    return obj


class MedGemmaClient:
    """Minimal OpenAI-compatible chat client for the two MedGemma endpoints."""

    def __init__(self, base_url: str = "http://127.0.0.1:8091", model: str | None = None,
                 api_key: str | None = None, timeout_s: float = DEFAULT_TIMEOUT_S):
        self.base_url = base_url.rstrip("/")
        self.model = model  # llama-server ignores this; the BF16 server expects the id
        self.api_key = api_key
        self.timeout_s = timeout_s

    # -- raw chat -----------------------------------------------------------
    def chat(self, prompt: str, images=None, *, temperature: float = 0.0,
             max_tokens: int = 256) -> ChatResult:
        """One user turn: prompt text + optional images (paths/bytes/PIL Images)."""
        content: list[dict] = [{"type": "text", "text": prompt}]
        for image in images or []:
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image)}})
        body = {
            "model": self.model or "medgemma",
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        request = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})},
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"MedGemma server HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"MedGemma server unreachable at {self.base_url}: {exc.reason}") from exc
        latency_ms = (time.perf_counter() - t0) * 1000.0
        message = (payload.get("choices") or [{}])[0].get("message", {})
        text = message.get("content") or ""  # OpenAI chat completion: message.content
        return ChatResult(text=text, raw_body=json.dumps(payload), latency_ms=round(latency_ms, 3),
                          model=payload.get("model", body["model"]),
                          usage=payload.get("usage"))

    # -- strict JSON --------------------------------------------------------
    def chat_json(self, prompt: str, images=None, *, temperature: float = 0.0,
                  max_tokens: int = 256, allow_uncertain: bool = True,
                  uncertain_value: str = "uncertain") -> ChatResult:
        """chat + strict JSON parse with the explicit `uncertain` option.

        Parse failures are RETURNED (parsed=None, parse_error set), never retried —
        the scorer counts them as errors. `uncertain` is True when the parsed reply's
        `answer` (or the whole reply, for bare strings) equals `uncertain_value`.
        """
        result = self.chat(prompt, images, temperature=temperature, max_tokens=max_tokens)
        try:
            data = extract_json_object(result.text)
        except (ValueError, json.JSONDecodeError) as exc:
            result.parse_error = str(exc)
            return result
        result.parsed = data
        answer = data.get("answer")
        result.uncertain = bool(allow_uncertain and isinstance(answer, str)
                                and answer.strip().lower() == uncertain_value)
        return result


def healthz(base_url: str, timeout_s: float = 10.0) -> dict:
    """GET /healthz of a local MedGemma endpoint (llama-server has no /healthz root path
    variants; both our servers expose it)."""
    with urllib.request.urlopen(base_url.rstrip("/") + "/healthz", timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))
