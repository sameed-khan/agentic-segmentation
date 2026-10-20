"""Harness v3: optional local-model components (research/harness_v3_methodology.md §3
Study C), enabled per case through the `v3` flags of POST /cases (all OFF by default, so
v2 behaviour is byte-identical when disabled).

Two independent components, switchable by case flags:

- **Pointer tool** (`v3.pointer` = "gui_actor"|"molmo2" + `v3.pointer_url`): the
  `seg_locate` tool (op `locate`). Sends the CURRENT VIEW image exactly as the agent saw
  it (the clean panel of the last seg_view render) to the pointer service (POST /point,
  k=5), maps the returned pixel points to the nearest drawn marks (dedup, rank order)
  and shows an image with the proposed points drawn (P1..P5). Counts as a tool call, not
  a model prompt; failures are agent-facing OpErrors like other failed calls.
- **Medical second reader** (`v3.medgemma` = "q4"|"bf16" + `v3.medgemma_url`, with
  `v3.checkpoints` ⊆ {localize, review, candidates}): MedGemma 1.5 4B consulted at fixed
  checkpoints and injected into tool observations (NOT an optional tool — exposure is
  controlled): (a) localize = the first seg_sheet (or the seg_open overview when the
  agent skips sheets: the first successful post-open op) — lesion box -> nearest marks;
  (b) review = the first-stage seg_finish review — completeness/problems JSON;
  (c) candidates = every candidate of the finish review — lesion/normal/uncertain.
  Deterministic (temperature 0); any failure yields a short "second reader unavailable"
  note and never crashes the op. Every request/response (prompt hash, image hash, raw
  output, latency) is appended to the trajectory record under the `v3` key.

This module never touches ground truth (CONTRACTS §0) and appends no raw coordinates to
agent-facing text: pointer points and MedGemma boxes are mapped to mark ids only.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from segd import config
from segd.case_state import OpError
from segd.observe import budget_line

MEDGEMMA_UNAVAILABLE = ("Medical second reader (MedGemma): unavailable this checkpoint "
                        "(service or parse error); continuing without it.")
LOCATE_PROPOSAL_NOTE = ("Proposals are hints from a pointing specialist: check each mark "
                        "on the image before pointing.")


# ------------------------------------------------------------------ case flags
def normalize_v3(spec: Any) -> dict | None:
    """Validate/normalize the case's v3 flags; None/absent -> None (v2 behaviour).

    Schema (POST /cases body key `v3`; echoed into case_meta.json and runner meta):
      {pointer: null|"gui_actor"|"molmo2", pointer_url: str?,
       medgemma: null|"q4"|"bf16", medgemma_url: str?,
       checkpoints: list of "localize"|"review"|"candidates"}
    URLs default to the standard loopback ports (config). Raises ValueError (programmer
    error -> HTTP 400) for unknown keys/values; idempotent on an already-normalized dict.
    """
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError("v3 must be an object")
    unknown = set(spec) - {"pointer", "pointer_url", "medgemma", "medgemma_url", "checkpoints"}
    if unknown:
        raise ValueError(f"unknown v3 key(s): {sorted(unknown)}")
    pointer = spec.get("pointer")
    if pointer is not None and pointer not in config.V3_POINTER_DEFAULT_URLS:
        raise ValueError("v3.pointer must be null|gui_actor|molmo2")
    medgemma = spec.get("medgemma")
    if medgemma is not None and medgemma not in config.V3_MEDGEMMA_DEFAULT_URLS:
        raise ValueError("v3.medgemma must be null|q4|bf16")
    pointer_url, medgemma_url = spec.get("pointer_url"), spec.get("medgemma_url")
    if pointer_url is not None and pointer is None:
        raise ValueError("v3.pointer_url set but v3.pointer is null")
    if medgemma_url is not None and medgemma is None:
        raise ValueError("v3.medgemma_url set but v3.medgemma is null")
    if pointer_url is not None and not isinstance(pointer_url, str):
        raise ValueError("v3.pointer_url must be a string")
    if medgemma_url is not None and not isinstance(medgemma_url, str):
        raise ValueError("v3.medgemma_url must be a string")
    cps = spec.get("checkpoints")
    if cps is None:
        cps = list(config.V3_CHECKPOINTS)
    if not isinstance(cps, list) or any(c not in config.V3_CHECKPOINTS for c in cps):
        raise ValueError("v3.checkpoints must be a list drawn from "
                         f"{list(config.V3_CHECKPOINTS)}")
    return {
        "pointer": pointer,
        "pointer_url": str(pointer_url or config.V3_POINTER_DEFAULT_URLS[pointer])
        if pointer else None,
        "medgemma": medgemma,
        "medgemma_url": str(medgemma_url or config.V3_MEDGEMMA_DEFAULT_URLS[medgemma])
        if medgemma else None,
        "checkpoints": [c for c in config.V3_CHECKPOINTS if c in cps],   # canonical order
    }


def medgemma_on(cs) -> bool:
    """MedGemma second reader enabled for this case."""
    v3 = getattr(cs, "v3", None)
    return bool(v3 and v3["medgemma"])


def checkpoint_due(cs, checkpoint: str) -> bool:
    """True when the named MedGemma checkpoint is enabled and has not fired yet."""
    v3 = getattr(cs, "v3", None)
    return bool(v3 and v3["medgemma"] and checkpoint in v3["checkpoints"])


# ------------------------------------------------------------ trajectory records
def record_call(cs, **fields) -> None:
    """Append one v3 request/response record to the current op's trajectory entry
    (run_op merges cs._log_extra into the record; CONTRACTS §4)."""
    cs._log_extra.setdefault("v3", []).append(fields)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _png_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------- clients
def pointer_request(url: str, image_png_b64: str, query: str, k: int) -> dict:
    """POST /point to a pointd service (pointd_gui_actor/pointd_molmo2 Phase 5 API).

    Returns the parsed body; raises RuntimeError on transport/HTTP errors (the caller
    turns that into an agent-facing OpError, which counts as a tool call)."""
    r = httpx.post(f"{url.rstrip('/')}/point",
                   json={"image_png_b64": image_png_b64, "query": query, "k": k,
                         "verify": False},
                   timeout=config.V3_POINTER_TIMEOUT_S)
    r.raise_for_status()
    return r.json()


_MEDGEMMA_CLIENT_CACHE: dict[str, Any] = {}


def _medgemma_client(url: str, timeout_s: float):
    """MedGemmaClient from harness/local_models/medgemma/medgemma_client.py (single
    source; stdlib-only module loaded by path, cached per path)."""
    path = str(config.V3_MEDGEMMA_CLIENT_FILE)
    cls = _MEDGEMMA_CLIENT_CACHE.get(path)
    if cls is None:
        if not Path(path).is_file():
            raise RuntimeError(f"medgemma client module missing: {path}")
        spec = importlib.util.spec_from_file_location("segd_medgemma_client", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["segd_medgemma_client"] = mod
        spec.loader.exec_module(mod)
        cls = mod.MedGemmaClient
        _MEDGEMMA_CLIENT_CACHE[path] = cls
    return cls(base_url=url, timeout_s=timeout_s)


def _op_group_deadline(cs) -> float:
    """Absolute deadline (perf_counter) shared by ALL MedGemma calls of the current op.

    Set at run_op ENTRY (ops.run_op, only when v3 medgemma is on) so pre-hook work
    (renders, candidate generation) counts against the same budget the extension's
    HTTP deadline sees; calls outside run_op initialize it on first use."""
    deadline = getattr(cs, "_v3_group_deadline", None)
    if deadline is None:
        deadline = time.perf_counter() + config.V3_MEDGEMMA_GROUP_TIMEOUT_S
        cs._v3_group_deadline = deadline
    return deadline


def _prompt(name: str, **variables: str) -> str:
    """A MedGemma checkpoint prompt template (harness/prompts/medgemma/<name>.md);
    {{var}} placeholders are replaced verbatim (no escaping needed: they are our texts)."""
    path = config.V3_PROMPT_DIR / f"{name}.md"
    text = path.read_text()
    for key, value in variables.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text


def medgemma_call(cs, checkpoint: str, image, prompt: str, *, requested_id: str | None = None,
                  allowed_verdicts: tuple[str, ...] | None = None) -> dict | None:
    """One deterministic MedGemma call; returns the parsed JSON object or None.

    Never raises: transport/HTTP/parse failures are logged under `v3` and return None
    (the caller appends MEDGEMMA_UNAVAILABLE; fixed checkpoints must never crash an op).
    The record carries prompt/image hashes, the raw reply and the latency.

    Result binding at this boundary (the advice only counts once it passes):
      - ``requested_id``: the record is bound to the candidate row the call was made for;
        a reply naming a DIFFERENT candidate is unusable (a verdict must never attach to
        a row the model was not asked about).
      - ``allowed_verdicts``: a reply without one of the contract's verdicts, or without
        a usable confidence, is unusable.
    Unusable replies return None with the reason in the record (``parsed`` stays None, so
    offline adherence can never attribute them).

    Deadline: the call shares the op's ABSOLUTE MedGemma group deadline (config.
    V3_MEDGEMMA_GROUP_TIMEOUT_S, set at run_op entry). The per-call allowance is the
    remaining group budget minus config.V3_MEDGEMMA_RESPONSE_RESERVE_S, so note
    formatting, render saving and response transport still fit before the extension's
    HTTP deadline. When no allowance is left the call is NOT sent and the record says
    so; otherwise the request carries that ABSOLUTE wall-clock deadline (the client
    bounds connect AND body read, not only individual socket operations) capped by
    V3_MEDGEMMA_TIMEOUT_S, so one slow call cannot consume the extension's whole
    HTTP deadline."""
    rec = {"call": "checkpoint", "checkpoint": checkpoint,
           "service": f"medgemma/{cs.v3['medgemma']}", "url": cs.v3["medgemma_url"],
           "prompt_sha256": _sha256(prompt.encode("utf-8")), "image_sha256": None,
           "candidate_id": requested_id, "returned_candidate_id": None,
           "raw": None, "latency_ms": None, "ok": False, "error": None, "parsed": None,
           "model": None}
    t0 = time.perf_counter()
    try:
        remaining = _op_group_deadline(cs) - t0
        allowance = remaining - config.V3_MEDGEMMA_RESPONSE_RESERVE_S
        if allowance <= 0:
            rec["error"] = (f"medgemma op deadline exhausted (group budget "
                            f"{config.V3_MEDGEMMA_GROUP_TIMEOUT_S}s, "
                            f"reserve {config.V3_MEDGEMMA_RESPONSE_RESERVE_S}s); "
                            "call not sent")
            rec["latency_ms"] = 0.0
            return None
        png = _png_bytes(image)                     # image serialization inside the guard:
        rec["image_sha256"] = _sha256(png)          # a bad image is a recorded failure too
        client = _medgemma_client(cs.v3["medgemma_url"],
                                  timeout_s=min(config.V3_MEDGEMMA_TIMEOUT_S, allowance))
        res = client.chat_json(prompt, images=[png], temperature=0.0,
                               max_tokens=config.V3_MEDGEMMA_MAX_TOKENS,
                               deadline=time.monotonic() + allowance)
        rec.update(raw=res.text, latency_ms=round(res.wall_ms, 3), model=res.model)
        parsed = res.parsed
        if parsed is None:
            rec["error"] = f"unusable reply: {res.parse_error}"
            return None
        rec["returned_candidate_id"] = (str(parsed["candidate_id"])
                                        if "candidate_id" in parsed else None)
        problems: list[str] = []
        if (requested_id is not None and rec["returned_candidate_id"] is not None
                and rec["returned_candidate_id"] != requested_id):
            problems.append(f"candidate id mismatch: requested {requested_id}, "
                            f"reply names {rec['returned_candidate_id']}")
        if allowed_verdicts is not None:
            verdict = parsed.get("verdict")
            if not (isinstance(verdict, str)
                    and verdict.strip().lower() in allowed_verdicts):
                problems.append(f"invalid verdict {verdict!r}")
            c = parsed.get("confidence")
            try:
                valid_conf = (isinstance(c, (int, float)) and not isinstance(c, bool)
                              and 0.0 <= float(c) <= 1.0)
            except (OverflowError, ValueError):     # e.g. int 10**309 in the reply
                valid_conf = False
            if not valid_conf:
                problems.append(f"invalid confidence {c!r}")
        if problems:
            rec["error"] = "unusable reply: " + "; ".join(problems)
            return None
        # parsed is exposed ONLY after every check passed: a reply that fails
        # validation (or whose validation raised) must never reach the trajectory as
        # usable advice (offline adherence counts ``ok`` records only)
        rec["parsed"] = parsed
        rec["ok"] = True
        return parsed
    except Exception as e:                                  # noqa: BLE001 (by design)
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
        rec["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        rec["parsed"] = None                                # never a half-validated reply
        return None
    finally:
        record_call(cs, **rec)


# ------------------------------------------------------------- panel geometry
def _panel_rects(ri) -> list[tuple[int, int, int, int]]:
    """Paste rectangles of the panels/rows recorded by a RenderInfo (added for v3;
    [] when the render does not record them)."""
    return [tuple(int(v) for v in r) for r in (getattr(ri, "panel_rects", None) or [])]


def _geom_field(geom, name: str, dict_key: str | None = None):
    """Read a ViewGeom field from the real ViewGeom (attributes) or the test-stub dict
    geom (keys axis/slice/crop/scale/panel_size/shape; _helpers.py stubs)."""
    v = getattr(geom, name, None)
    if v is None and isinstance(geom, dict):
        v = geom.get(dict_key or name)
    return v


def panel_voxel(ri, px: float, py: float, shape: tuple) -> tuple | None:
    """A pixel of the combined render -> full-volume voxel, via the panel whose paste
    rect contains it. When the pixel falls between panels (a sheet grid gap), the
    NEAREST panel is used and the point clamped into it (deterministic snap, like the
    mark snapping). None when the render records no panels/rows."""
    rects = _panel_rects(ri)
    views = getattr(ri, "views", None) or []
    hit = None
    for i, rect in enumerate(rects):
        if rect[0] <= px < rect[2] and rect[1] <= py < rect[3]:
            hit = (i, px, py)
            break
    if hit is None and rects:
        # nearest panel centre (combined-image px), point clamped into its rect
        def centre(r):
            return ((r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0)

        i = min(range(len(rects)), key=lambda j: (centre(rects[j])[0] - px) ** 2
                + (centre(rects[j])[1] - py) ** 2)
        x0, y0, x1, y1 = rects[i]
        hit = (i, min(max(px, x0), x1 - 1.0), min(max(py, y0), y1 - 1.0))
    if hit is None:
        return None
    i, px2, py2 = hit
    if i >= len(views):
        return None
    geom = views[i]
    scale = float(_geom_field(geom, "scale") or 1.0)
    crop = _geom_field(geom, "crop")
    axis = _geom_field(geom, "axis")
    slice_idx = int(_geom_field(geom, "slice_idx", "slice"))
    from segd.case_state import inplane_to_voxel
    if crop:
        c = (px2 - rects[i][0]) / scale + crop[1]
        r = (py2 - rects[i][1]) / scale + crop[0]
    else:
        c, r = px2 - rects[i][0], py2 - rects[i][1]
    return inplane_to_voxel(axis, int(r), int(c), slice_idx, tuple(shape))


def clean_panel(img, ri):
    """The clean panel of a rendered view image exactly as the agent saw it: the first
    paste rect when recorded, else the whole image (single-panel renders/stubs)."""
    rects = _panel_rects(ri)
    if len(rects) >= 1:
        x0, y0, x1, y1 = rects[0]
        return img.crop((x0, y0, x1, y1))
    return img


def nearest_marks(cs, voxel: tuple, n: int = 3) -> list[tuple[str, float]]:
    """The n nearest global marks to a full-volume voxel, by mm distance: [(label, mm)]."""
    if cs.markset is None or not cs.markset.marks:
        return []
    out = []
    for mid, m in cs.markset.marks.items():
        d = sum((float(a - b) * s) ** 2 for a, b, s in zip(voxel, m.ijk, cs.spacing)) ** 0.5
        out.append((str(mid), d))
    out.sort(key=lambda t: (t[1], int(t[0])))
    return out[:n]


# ------------------------------------------------------------------ seg_locate
def _mark_positions_in_panel_px(ri, geom) -> list[tuple[str, float, float]]:
    """Global + edge mark positions in PANEL pixels of a view render (both panels of a
    seg_view share one geometry, so panel-relative positions are the same).

    Global marks: RenderInfo.marks entries (mark_id, panel_idx, x_px, y_px) with glyph
    centres; edge marks: edge_marks entries ("E1", panel_idx, row, col) with in-plane
    display coordinates -> panel px via the geom's crop/scale (top-left voxel corner +
    half footprint, like the glyphs)."""
    marks: list[tuple[str, float, float]] = []
    for item in (getattr(ri, "marks", None) or []):
        if isinstance(item, (tuple, list)):
            mid, x, y = item[0], item[-2], item[-1]          # (id, panel, x, y)
            if isinstance(x, (tuple, list)):                 # (id, (x, y)) stub form
                x, y = x[0], x[1]
            marks.append((str(int(mid)), float(x), float(y)))
    crop = _geom_field(geom, "crop")
    scale = float(_geom_field(geom, "scale") or 1.0)
    for item in (getattr(ri, "edge_marks", None) or []):
        if isinstance(item[-1], (tuple, list)):              # (label, (row, col))
            label, (row, col) = item[0], item[-1]
        else:                                                # (label, panel, row, col)
            label, (row, col) = item[0], (item[-2], item[-1])
        if crop:
            x = (col - crop[1]) * scale + scale / 2.0
            y = (row - crop[0]) * scale + scale / 2.0
        else:
            x, y = col + scale / 2.0, row + scale / 2.0
        marks.append((str(label), float(x), float(y)))
    return marks


def snap_points_to_marks(points: list[dict], mark_positions: list[tuple[str, float, float]]
                         ) -> list[dict]:
    """Map proposed pixel points to their nearest drawn marks (CONTRACTS §0: marks are
    the only spatial interface). Points arrive in service rank order; a mark keeps its
    FIRST (best-ranked) point only (dedup). Distance ties go to the lowest numeric mark
    id (integer ids before edges), matching the predeclared study snap rule."""

    def order_key(mid: str) -> tuple:
        digits = mid[1:] if mid.upper().startswith("E") else mid
        return (0 if digits.isdigit() else 1, int(digits) if digits.isdigit() else 10**9)

    out: list[dict] = []
    used: set[str] = set()
    for i, p in enumerate(points):
        label = f"P{i + 1}"
        best, best_d = None, float("inf")
        for mid, mx, my in mark_positions:
            d = ((p["x"] - mx) ** 2 + (p["y"] - my) ** 2) ** 0.5
            if d < best_d or (d == best_d and best is not None
                              and order_key(mid) < order_key(best)):
                best, best_d = mid, d
        if best is None or best in used:
            out.append({"point": label, "mark": None, "score": p.get("score"),
                        "dist_px": None})
            continue
        used.add(best)
        out.append({"point": label, "mark": best, "score": p.get("score"),
                    "dist_px": round(best_d, 1)})
    return out


def op_locate(cs, query: str | None = None):
    """seg_locate body: call the pointer service on the current view, return ranked
    marks + an image with the proposed points drawn. Raises OpError when disabled, when
    the view shows no marks, or when the pointer service fails (still a tool call)."""
    v3 = cs.v3
    if not (v3 and v3["pointer"]):
        raise OpError("seg_locate is disabled for this case (no v3 pointer configured)")
    av = cs.require_active_view()
    if av.image is None or av.render_info is None:
        raise OpError("no current view image: call seg_view before seg_locate")
    query_text = (str(query).strip() if query is not None else "") or cs.target_text
    geom = av.view_geom if av.view_geom is not None else av.render_info
    rects = _panel_rects(av.render_info)
    panel = clean_panel(av.image, av.render_info)
    mark_positions = _mark_positions_in_panel_px(av.render_info, geom)
    if not mark_positions:
        raise OpError("the current view shows no marks; call seg_view (marks coarse/fine/"
                      "edges) before seg_locate")

    buf = io.BytesIO()
    panel.save(buf, format="PNG")
    png = buf.getvalue()
    t0 = time.perf_counter()
    rec = {"call": "locate", "service": f"pointer/{v3['pointer']}", "url": v3["pointer_url"],
           "query": query_text, "k": config.V3_LOCATE_K,
           "prompt_sha256": None, "image_sha256": _sha256(png),
           "raw": None, "latency_ms": None, "ok": False, "error": None, "parsed": None,
           "snapped": []}
    try:
        resp = pointer_request(v3["pointer_url"], base64.b64encode(png).decode("ascii"),
                               query_text, config.V3_LOCATE_K)
        points = [{"x": float(p["x"]), "y": float(p["y"]), "score": p.get("score")}
                  for p in (resp.get("points") or [])][:config.V3_LOCATE_K]
        rec["raw"] = str(resp.get("raw"))[:2000]
        rec["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        rec["ok"] = True
        # structured response (audit: returned points, model/service provenance); the raw
        # text stays truncated — coordinates stay out of agent-facing text (CONTRACTS §0)
        rec["parsed"] = {"points": points, "model": resp.get("model"),
                         "version": resp.get("version"), "verifier": resp.get("verifier"),
                         "image_w": resp.get("image_w"), "image_h": resp.get("image_h")}
    except Exception as e:                                  # noqa: BLE001 -> OpError
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
        rec["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        record_call(cs, **rec)
        raise OpError(f"seg_locate: pointer service unavailable ({v3['pointer']} at "
                      f"{v3['pointer_url']}); the view is unchanged, try again later or "
                      "continue without it") from e

    snapped = snap_points_to_marks(points, mark_positions)
    rec["snapped"] = snapped
    record_call(cs, **rec)

    from segd import render
    overlay = render.draw_proposed_points(
        av.image, rects or [(0, 0, av.image.width, av.image.height)],
        [(p["x"], p["y"], f"P{i + 1}") for i, p in enumerate(points)])

    lines = [f"locate ({v3['pointer']}): {len(points)} proposed point(s), nearest marks:"]
    for s in snapped:
        if s["mark"] is None:
            continue   # no mark in the view (or deduped: an earlier point proposed it)
        lines.append(f"{s['point']} mark {s['mark']} score {s['score']} "
                     f"({s['dist_px']} px)")
    text = "\n".join(lines) + f"\n{LOCATE_PROPOSAL_NOTE} | {budget_line(cs)}"
    return text, [overlay]


# -------------------------------------------------------- checkpoint injection
def _fmt_conf(value) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "?"


def localize_note(cs, image, ri) -> str:
    """Run the localize checkpoint on the given sheet/overview image; append-ready note."""
    cs.v3_fired.add("localize")
    prompt = _prompt("localize", target=cs.target_text, sequence=cs.sequence)
    parsed = medgemma_call(cs, "localize", image, prompt)
    if parsed is None:
        return MEDGEMMA_UNAVAILABLE
    box = parsed.get("box")
    conf = _fmt_conf(parsed.get("confidence"))
    if not (isinstance(box, (list, tuple)) and len(box) == 4
            and all(isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0 for v in box)):
        return (f"Medical second reader (MedGemma): no usable lesion box "
                f"(confidence {conf}); treat its reply as 'cannot locate'.")
    y0, x0, y1, x1 = (float(v) for v in box)
    px, py = (x0 + x1) / 2.0 * image.width, (y0 + y1) / 2.0 * image.height
    try:
        voxel = panel_voxel(ri, px, py, cs.shape)
    except Exception:                                       # noqa: BLE001 (by design)
        voxel = None
    if voxel is None:
        return (f"Medical second reader (MedGemma): it returned a lesion location but "
                f"it could not be mapped to marks (confidence {conf}).")
    near = nearest_marks(cs, voxel)
    if not near:
        return (f"Medical second reader (MedGemma): it returned a lesion location but "
                f"no marks exist to map it to (confidence {conf}).")
    others = "".join(f", then {m}" for m, _ in near[1:])
    return (f"Medical second reader (MedGemma): the lesion is nearest mark {near[0][0]}"
            f"{others}; box confidence {conf}. Treat as a hint; verify on the images "
            "before pointing.")


def review_note(cs, review_img, vid: int) -> str:
    """Run the review checkpoint on the finish review grid; append-ready note."""
    cs.v3_fired.add(f"review:{vid}")
    prompt = _prompt("review")
    parsed = medgemma_call(cs, "review", review_img, prompt)
    if parsed is None:
        return MEDGEMMA_UNAVAILABLE
    complete = str(parsed.get("complete", "uncertain"))
    conf = _fmt_conf(parsed.get("confidence"))
    problems = parsed.get("problems")
    prob_txt = ""
    if isinstance(problems, list):
        issues = [f"{p.get('panel', '?')}: {p.get('issue', '?')}" for p in problems
                  if isinstance(p, dict) and p.get("issue") not in (None, "ok")]
        if issues:
            prob_txt = "; problems " + "; ".join(issues)
    return (f"Medical second reader (MedGemma): complete {complete} (confidence {conf})"
            f"{prob_txt}. It is a hint; the decision is yours.")


CANDIDATE_VERDICTS = ("lesion", "normal", "uncertain")   # the checkpoint's JSON contract


def candidates_note(cs, sheet_img, row_rects, cands, vid: int) -> str:
    """Run the candidates checkpoint: one MedGemma call per candidate row of the sheet;
    each (version, candidate) fires at most once.

    Every reply is bound to the REQUESTED candidate id (the row image it saw): a reply
    naming another candidate is unusable and that row shows as unavailable — so advice
    can never attach to, or be attributed to, a row the model was not asked about.

    The WHOLE per-candidate iteration (prompt load, row crop, request, formatting) is
    guarded: an earlier row's delivered advice is kept, a failing row gets its
    unavailable entry + diagnostic, and later rows still run."""
    entries: list[str] = []
    for i, cand in enumerate(cands):
        key = f"candidates:{vid}:{cand.id}"
        if key in cs.v3_fired:
            continue
        cs.v3_fired.add(key)
        try:
            prompt = _prompt("candidates", candidate_id=cand.id)
            row = sheet_img
            if row_rects and i < len(row_rects):
                x0, y0, x1, y1 = row_rects[i]
                row = sheet_img.crop((x0, y0, x1, y1))
            parsed = medgemma_call(cs, "candidates", row, prompt,
                                   requested_id=str(cand.id),
                                   allowed_verdicts=CANDIDATE_VERDICTS)
        except Exception as e:                              # noqa: BLE001 (by design)
            record_call(cs, call="checkpoint", checkpoint="candidates", ok=False,
                        candidate_id=str(cand.id), parsed=None,
                        error=f"row: {type(e).__name__}: {e}"[:300])
            entries.append(f"{cand.id} unavailable")
            continue
        if parsed is None:
            entries.append(f"{cand.id} unavailable")
            continue
        verdict = str(parsed["verdict"]).strip().lower()
        entries.append(f"{cand.id} {verdict} ({_fmt_conf(parsed.get('confidence'))})")
    if not entries:
        return ""
    return ("Medical second reader (MedGemma) candidate verdicts: "
            + ", ".join(entries) + ". Hints only; judge each on the images.")


def after_op_hook(cs, op_name: str, out):
    """run_op hook (called after the op body succeeded): inject the localize checkpoint.

    Fires once per case: on the first seg_sheet result; when the agent never calls a
    sheet (its first successful post-open op is something else), on the seg_open
    overview rendered at that moment. Never raises; appends the note to out.text."""
    try:
        if not (medgemma_on(cs) and checkpoint_due(cs, "localize")):
            return out
        if "localize" in cs.v3_fired or op_name == "open":
            return out                    # once per case; open itself never fires
        if op_name == "sheet" and out.images and cs._last_render_info is not None:
            img, ri = out.images[0], cs._last_render_info
        else:
            from segd import render
            img, ri = render.render_overview(cs.image, cs.spacing, mask=cs.current_mask(),
                                             marks=None)
        note = localize_note(cs, img, ri)
        if note:
            out.text = out.text + "\n" + note
    except Exception as e:                                  # noqa: BLE001 (by design)
        record_call(cs, call="checkpoint", checkpoint="localize", ok=False,
                    error=f"hook: {type(e).__name__}: {e}"[:300])
        try:
            out.text = out.text + "\n" + MEDGEMMA_UNAVAILABLE
            cs.v3_fired.add("localize")
        except Exception:                                   # noqa: BLE001
            pass
    return out


# --------------------------------------------------------- finish-review hooks
def finish_review_notes(cs, review_img, cand_img, cand_row_rects, cands, vid: int) -> str:
    """Both finish-review checkpoints (review grid + per-candidate rows); each fires at
    most once per version (candidates_note also skips already-fired candidate ids).
    Returns the combined note text ('' when nothing fired).

    Each checkpoint is protected SEPARATELY — including prompt loading and row cropping,
    which sit outside medgemma_call's request guard: a checkpoint's failure becomes a
    trajectory diagnostic plus the standard unavailable note, notes produced by earlier
    checkpoints are kept, and the other enabled checkpoints still run (never a blanket
    empty-note fallback that silently drops mandatory checkpoints)."""
    notes: list[str] = []
    if medgemma_on(cs) and checkpoint_due(cs, "review") and f"review:{vid}" not in cs.v3_fired:
        try:
            notes.append(review_note(cs, review_img, vid))
        except Exception as e:                              # noqa: BLE001 (by design)
            record_call(cs, call="checkpoint", checkpoint="review", ok=False,
                        error=f"checkpoint: {type(e).__name__}: {e}"[:300])
            notes.append(MEDGEMMA_UNAVAILABLE)
    if medgemma_on(cs) and checkpoint_due(cs, "candidates") and cands:
        try:
            note = candidates_note(cs, cand_img, cand_row_rects, cands, vid)
            if note:
                notes.append(note)
        except Exception as e:                              # noqa: BLE001 (by design)
            record_call(cs, call="checkpoint", checkpoint="candidates", ok=False,
                        error=f"checkpoint: {type(e).__name__}: {e}"[:300])
            notes.append(MEDGEMMA_UNAVAILABLE)
    return "\n".join(n for n in notes if n)
