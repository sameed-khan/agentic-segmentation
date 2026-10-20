"""segd HTTP API (CONTRACTS §5): FastAPI on 127.0.0.1:8765, one CaseState per open case.

Responses (agent-facing): {"ok": true, "text": ..., "images": [<base64 png>...], "done": false}
or {"ok": false, "error": "...", "text": "..."} with HTTP 200 for agent-facing errors
(OpError) AND for unexpected exceptions on agent ops (generic message; the full traceback
is in trajectory.jsonl with error_kind "internal" — reviewer finding #2). 4xx/5xx are
kept for programmer errors on non-op endpoints.

Concurrency: all endpoints are sync `def`, so starlette dispatches each request to its
threadpool; heavy ops (image upload, nnInteractive calls) block only their worker thread.
One lock per case serializes ops on the same case (created atomically and re-checked
under the lock, finding #11); different cases run concurrently.
"""
from __future__ import annotations

import base64
import io
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from segd import config
from segd.case_state import CaseState, OpError
from segd.observe import budget_line

# Generic agent-facing message for unexpected exceptions on agent ops; never leaks
# internals — the traceback lives in trajectory.jsonl (finding #2).
INTERNAL_OP_ERROR = ("internal harness error; the operation failed and the mask is unchanged; "
                     "try again")


def _image_to_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def create_app() -> FastAPI:
    app = FastAPI(title="segd", version="0.1.0")
    st = {"cases": {}, "locks": {}, "glock": threading.Lock()}
    app.state.segd = st   # exposed for the runner/tests; endpoints use the closure

    def _lock(handle: str) -> threading.Lock:
        # create per-case locks under a global lock (finding #11: racy setdefault)
        with st["glock"]:
            if handle not in st["locks"]:
                st["locks"][handle] = threading.Lock()
            return st["locks"][handle]

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.post("/cases")
    def open_case(body: dict):
        for field in ("case_id", "run_id", "out_dir"):
            if not body.get(field):
                raise HTTPException(status_code=400, detail=f"missing field: {field}")
        case_id = str(body["case_id"])
        # Logging layout is results/<run_id>/<case>/ (CONTRACTS §5.7): out_dir is the
        # results root, run_id and case_id name the subdirectories.
        out_dir = Path(body["out_dir"]) / str(body["run_id"]) / case_id
        nni_url = body.get("nni_url") or None
        budgets = body.get("budgets") or None
        # Hold the case lock for the whole open (finding #11: two concurrent POSTs for
        # the same id used to leak one nnInteractive session; also prevents an op from
        # running against a half-initialized case).
        with _lock(case_id):
            existing = st["cases"].get(case_id)
            if existing is not None:
                existing.close()
                st["cases"].pop(case_id, None)
            try:
                from segd import data
                row = data.get_case(case_id)
                geom = data.load_case_geometry(row)
            except KeyError:
                raise HTTPException(status_code=404, detail=f"unknown case {case_id!r}")
            cs = CaseState(case_id, geom.image, geom.affine, geom.spacing, row.target_text,
                           row.sequence, out_dir, nni_url=nni_url, budgets=budgets,
                           orig_geometry=(geom.orig_affine, geom.inv_ornt)
                           if geom.orig_affine is not None else None)
            try:
                cs.initialize()   # marks + nnInteractive upload; runs in the request's worker thread
            except Exception:
                cs.close()        # free a claimed nnInteractive session slot (finding #5)
                raise
            st["cases"][case_id] = cs
        return {"ok": True, "handle": case_id}

    @app.post("/cases/{handle}/op")
    def op(handle: str, body: dict):
        with _lock(handle):
            cs = st["cases"].get(handle)   # re-check under the lock (finding #11)
            if cs is None:
                raise HTTPException(status_code=404, detail=f"no open case {handle!r}")
            op_name = body.get("op")
            args = body.get("args") or {}
            try:
                from segd.ops import run_op
                out = run_op(cs, op_name, args)
            except OpError as e:
                # agent-facing error: HTTP 200 with ok:false (CONTRACTS §5)
                return JSONResponse(content={"ok": False, "error": str(e), "text": str(e)},
                                    status_code=200)
            except Exception:
                # unexpected error on an agent op: still HTTP 200/ok:false so the agent
                # keeps working; traceback already logged by run_op (finding #2)
                msg = f"{INTERNAL_OP_ERROR} | {budget_line(cs)}"
                return JSONResponse(content={"ok": False, "error": msg, "text": msg},
                                    status_code=200)
        return {
            "ok": True, "text": out.text, "images": [_image_to_b64(i) for i in out.images],
            "done": bool(out.done),
        }

    @app.get("/cases/{handle}/status")
    def status(handle: str):
        with _lock(handle):
            cs = st["cases"].get(handle)   # re-check under the lock (finding #11)
            if cs is None:
                raise HTTPException(status_code=404, detail=f"no open case {handle!r}")
            av = cs.active_view
            parent = cs.versions[cs.current].parent
            return {
                "ok": True, "handle": cs.case_id, "current_version": f"v{cs.current}",
                "parent": None if parent is None else f"v{parent}",
                "tool_calls": cs.tool_calls, "model_prompts": cs.model_prompts,
                "budgets": dict(cs.budgets), "done": cs.done,
                "active_view": None if av is None else {"axis": av.axis, "slice": av.slice_idx},
                "nni": "remote" if cs.nni_url else "fake",
                "versions": len(cs.versions),
            }

    @app.delete("/cases/{handle}")
    def close_case(handle: str):
        with _lock(handle):
            cs = st["cases"].get(handle)   # re-check under the lock (finding #11)
            if cs is None:
                raise HTTPException(status_code=404, detail=f"no open case {handle!r}")
            try:
                cs.flush_versions()
            except Exception:
                pass
            cs.close()
            st["cases"].pop(handle, None)
        return {"ok": True, "closed": handle}

    return app


app = create_app()


def main() -> None:
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")
