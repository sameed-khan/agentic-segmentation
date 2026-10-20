"""Shared test fixtures for the sammed3d project (offline, no GPU)."""
from __future__ import annotations

import threading

import numpy as np
import pytest
import uvicorn

from sammed3d import client, service


def make_phantom(shape=(120, 120, 80), seed=0):
    """Noisy cylinder 'limb' (intensity ~300) with a bright ellipsoid lesion (~500):
    the same phantom as harness/server/tests/_helpers.py (tests must stay comparable
    across projects; the lesion mask is GT and stays in the test process)."""
    rng = np.random.default_rng(seed)
    X, Y, Z = shape
    xx, yy, zz = np.meshgrid(*(np.linspace(-1, 1, n, dtype=np.float32) for n in shape),
                             indexing="ij")
    body = (xx ** 2 + yy ** 2) < 0.8
    lesion = ((xx - 0.2) / 0.15) ** 2 + ((yy + 0.1) / 0.12) ** 2 + (zz / 0.25) ** 2 < 1
    img = np.where(body, 300.0, 0.0) + np.where(lesion, 500.0, 0.0)
    img = img + rng.normal(0, 40, shape) * body
    return img.astype(np.float32), lesion, body


def lesion_centre(lesion: np.ndarray) -> tuple:
    idx = np.nonzero(lesion)
    return tuple(int(round(float(a.mean()))) for a in idx)


class _FakeService:
    """A fake-mode sammed3d service on a random localhost port (real HTTP, no torch)."""

    def __init__(self):
        self.app = service.create_app(service.FakePatchPredictor())
        config = uvicorn.Config(self.app, host="127.0.0.1", port=0, log_level="error")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        while not self.server.started:
            if not self.thread.is_alive():
                raise RuntimeError("fake service failed to start")
            import time
            time.sleep(0.02)
        self.port = self.server.servers[0].sockets[0].getsockname()[1]
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=5)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def make_fake_session(monkeypatch, img, spacing=(1.0, 1.0, 1.0), shape=None):
    """A SamMed3DSession against a fresh fake service on a random port; set_image /
    set_target_buffer done, ready for interactions. Caller must close() it (which only
    closes the session; the server thread is a daemon and dies with the process)."""
    svc = _FakeService()
    svc.__enter__()
    sess = client.SamMed3DSession(svc.url)
    sess._fake_service = svc        # keep the server thread referenced (daemon anyway)
    if shape is not None:
        sess.set_target_buffer(np.zeros(shape, dtype=np.uint8))
    sess.set_image(np.ascontiguousarray(img, dtype=np.float16)[None],
                   image_properties={"spacing": list(spacing)})
    if shape is None:
        sess.set_target_buffer(np.zeros(img.shape, dtype=np.uint8))
    return sess, svc.app


@pytest.fixture
def phantom():
    return make_phantom()
