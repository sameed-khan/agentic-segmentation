"""Fake local-model services shared by test_v3.py and the cross-revision identity driver
(_identity_driver.py): a loader for the service modules under harness/local_models, an
uvicorn-in-a-thread server on a random loopback port, a fixed-output pointd backend and a
scripted MedGemma backend (no torch, no GPU, no network)."""
from __future__ import annotations

import importlib.util
import io
import json
import re
import sys
import threading
import time

from pathlib import Path

import uvicorn
from PIL import Image


def _load_module(name: str, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod   # before exec: pydantic resolves the deferred annotations
    spec.loader.exec_module(mod)
    return mod


def _serve(app):
    """uvicorn on an ephemeral loopback port in a daemon thread -> (base url, stop fn)."""
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started
    host, port = server.servers[0].sockets[0].getsockname()[:2]
    return f"http://{host}:{port}", (lambda: (setattr(server, "should_exit", True),
                                              thread.join(timeout=10)))



class FixedPointerBackend:
    """Deterministic pointd backend: answers the desired PANEL-pixel points."""

    name = "fixed"
    model_id = "test/pointer"

    def __init__(self, points):
        self.points = [(float(x), float(y), score) for x, y, score in points]

    def load(self):
        pass

    def predict(self, png_bytes, query, k, verify):
        w, h = Image.open(io.BytesIO(png_bytes)).size
        out = []
        for x, y, score in self.points[: max(1, int(k))]:
            out.append({"x_norm": x / w, "y_norm": y / h, "score": score})
        return {"points": out, "raw": f"fixed {query!r}", "verifier": None}


class ScriptedMedGemmaBackend:
    """Replies shaped to each checkpoint's JSON contract, chosen by a marker in the
    prompt text; records every call (prompt text, image count) for assertions."""

    name = "scripted"
    model_id = "google/medgemma-1.5-4b-it"

    def __init__(self, box=(0.4, 0.4, 0.6, 0.6), complete="no",
                 problems=(("axial 999", "missed_lesion"),), verdict="lesion"):
        self.box, self.complete = box, complete
        self.problems, self.verdict = problems, verdict
        self.calls: list[tuple[str, int]] = []

    def load(self):
        pass

    def generate(self, prompt_text: str, images_b64, temperature: float, max_tokens: int) -> str:
        self.calls.append((prompt_text, len(images_b64)))
        if '"box": [y0' in prompt_text:
            return json.dumps({"box": list(self.box) if self.box else None,
                               "confidence": 0.7})
        if '"complete"' in prompt_text:
            return json.dumps({"complete": self.complete,
                               "problems": [{"panel": p, "issue": i} for p, i in self.problems],
                               "confidence": 0.6})
        if '"verdict"' in prompt_text:
            m = re.search(r"candidate (C\d+)", prompt_text)
            return json.dumps({"candidate_id": m.group(1) if m else "?",
                               "verdict": self.verdict, "confidence": 0.8})
        return json.dumps({"answer": "uncertain", "confidence": 0.5})




SAMED3D_SRC = Path(__file__).resolve().parents[2] / "local_models" / "sammed3d" / "src"


class FakeSamMed3DServer:
    """The sammed3d --fake service on a random localhost port (real HTTP)."""

    def __init__(self):
        if str(SAMED3D_SRC) not in sys.path:
            sys.path.insert(0, str(SAMED3D_SRC))
        from sammed3d import service
        self.app = service.create_app(service.FakePatchPredictor(), model_info={"mode": "fake"})
        self.server = uvicorn.Server(uvicorn.Config(self.app, host="127.0.0.1", port=0,
                                                    log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        while not self.server.started:
            assert self.thread.is_alive(), "fake sammed3d service failed to start"
            time.sleep(0.02)
        self.url = f"http://127.0.0.1:{self.server.servers[0].sockets[0].getsockname()[1]}"
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=5)

    @property
    def n_sessions(self) -> int:
        return len(self.app.state.sammed3d["sessions"])
