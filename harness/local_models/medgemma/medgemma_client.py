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
    """One completion: raw reply text, parsed JSON (when requested), and the log.

    Latency semantics (review M3): ``wall_ms`` is the client wall clock around the whole
    request (includes queue wait behind the server's inference lock);
    ``service_latency_ms`` is the SERVER-reported inference latency when the payload
    exposes it, else None."""

    text: str
    raw_body: str                    # full HTTP JSON body (for the run log)
    wall_ms: float
    model: str
    service_latency_ms: float | None = None
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


def server_reported_latency_ms(payload) -> float | None:
    """Server-reported inference latency from known extension fields, else None.

    Our BF16 Transformers server returns ``latency_ms`` (measured inside its inference
    lock, queue wait excluded); llama-server returns the llama.cpp ``timings`` object —
    the generation cost it reports is ``prompt_ms + predicted_ms``. Any other/absent
    field -> None (the caller keeps its own wall time; the scorer must not silently
    compare wall time against other models' service latency).
    """
    if not isinstance(payload, dict):
        return None
    lat = payload.get("latency_ms")
    if isinstance(lat, (int, float)) and not isinstance(lat, bool):
        return round(float(lat), 3)
    timings = payload.get("timings")
    if isinstance(timings, dict):
        prompt_ms, predicted_ms = timings.get("prompt_ms"), timings.get("predicted_ms")
        if (isinstance(prompt_ms, (int, float)) and not isinstance(prompt_ms, bool)
                and isinstance(predicted_ms, (int, float)) and not isinstance(predicted_ms, bool)):
            return round(float(prompt_ms) + float(predicted_ms), 3)
    return None


def _clamp_read_to_deadline(resp, deadline: float) -> None:
    """Bound the BODY READ to the caller's absolute wall-clock deadline: urlopen's
    ``timeout`` only applies to individual socket operations, so a slow-drip reply
    could otherwise keep the request alive past the op's HTTP budget. Clamps the
    response socket's timeout to the remaining budget; raises TimeoutError when the
    budget is already spent."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("reply exceeded the wall-clock deadline")
    sock = getattr(getattr(getattr(resp, "fp", None), "raw", None), "_sock", None)
    if sock is not None:
        sock.settimeout(max(remaining, 0.001))


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
             max_tokens: int = 256, deadline: float | None = None) -> ChatResult:
        """One user turn: prompt text + optional images (paths/bytes/PIL Images).

        ``deadline``: optional ABSOLUTE wall-clock deadline (``time.monotonic()``
        timestamp, e.g. from a per-operation budget). It bounds the WHOLE request —
        connection, headers and body read: ``urlopen(timeout=...)`` only bounds
        individual socket operations, so a slow-drip reply could otherwise exceed the
        caller's op deadline. Raises RuntimeError when the budget is already spent
        before the request or when the reply exceeds it."""
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
            timeout = self._remaining_s(deadline)
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                if deadline is not None:
                    _clamp_read_to_deadline(resp, deadline)
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"MedGemma server HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"MedGemma server unreachable at {self.base_url}: {exc.reason}") from exc
        except TimeoutError as exc:   # incl. socket.timeout: the deadline clamp fired
            raise RuntimeError(f"MedGemma request timed out: wall-clock deadline "
                               f"exceeded at {self.base_url}") from exc
        wall_ms = (time.perf_counter() - t0) * 1000.0
        message = (payload.get("choices") or [{}])[0].get("message", {})
        text = message.get("content") or ""  # OpenAI chat completion: message.content
        return ChatResult(text=text, raw_body=json.dumps(payload), wall_ms=round(wall_ms, 3),
                          model=payload.get("model", body["model"]),
                          usage=payload.get("usage"),
                          service_latency_ms=server_reported_latency_ms(payload))

    # -- strict JSON --------------------------------------------------------
    def _remaining_s(self, deadline: float | None) -> float:
        """Effective socket timeout for the remaining wall-clock budget (the client's
        base timeout when no deadline was given); raises when it is already spent."""
        if deadline is None:
            return self.timeout_s
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("MedGemma deadline already spent; request not sent")
        return min(self.timeout_s, remaining)

    def chat_json(self, prompt: str, images=None, *, temperature: float = 0.0,
                  max_tokens: int = 256, allow_uncertain: bool = True,
                  uncertain_value: str = "uncertain",
                  deadline: float | None = None) -> ChatResult:
        """chat + strict JSON parse with the explicit `uncertain` option.

        Parse failures are RETURNED (parsed=None, parse_error set), never retried —
        the scorer counts them as errors. `uncertain` is True when the parsed reply's
        `answer` (or the whole reply, for bare strings) equals `uncertain_value`.
        ``deadline`` (see chat) bounds the whole request wall-clock as well.
        """
        result = self.chat(prompt, images, temperature=temperature, max_tokens=max_tokens,
                           deadline=deadline)
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
