"""Shared fixtures for the local-model tests: fake services via fastapi TestClient.

These tests run WITHOUT a GPU and without torch: the fake backends are deterministic
and import nothing heavy. Everything runs inside the harness/server environment:

    uv run --project harness/server python -m pytest harness/local_models/tests -q
"""
from __future__ import annotations

import base64
import importlib.util
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

LOCAL_MODELS_DIR = Path(__file__).resolve().parents[1]
GUI_ACTOR_DIR = LOCAL_MODELS_DIR / "pointd_gui_actor"
MOLMO2_DIR = LOCAL_MODELS_DIR / "pointd_molmo2"
MEDGEMMA_DIR = LOCAL_MODELS_DIR / "medgemma"

for _d in (GUI_ACTOR_DIR, MOLMO2_DIR, MEDGEMMA_DIR):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))  # molmo_points / medgemma_client imports


def _load_module(name: str, path: Path):
    """Load a service module under a UNIQUE name (two services ship `server.py`)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def png_bytes(width: int = 64, height: int = 48, seed: int = 0) -> bytes:
    """Deterministic small PNG for API tests."""
    img = Image.new("RGB", (width, height), (10 + seed, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def b64png(width: int = 64, height: int = 48, seed: int = 0) -> str:
    return base64.b64encode(png_bytes(width, height, seed)).decode("ascii")


@pytest.fixture(scope="session")
def gui_actor_server_mod():
    return _load_module("phase5_gui_actor_server", GUI_ACTOR_DIR / "server.py")


@pytest.fixture(scope="session")
def molmo2_server_mod():
    return _load_module("phase5_molmo2_server", MOLMO2_DIR / "server.py")


@pytest.fixture(scope="session")
def medgemma_server_mod():
    return _load_module("phase5_medgemma_server", MEDGEMMA_DIR / "start_bf16_server.py")


@pytest.fixture(scope="session")
def gui_actor_app(gui_actor_server_mod):
    return gui_actor_server_mod.create_app(
        gui_actor_server_mod.FakeBackend(model_id="microsoft/GUI-Actor-2B-Qwen2-VL"))


@pytest.fixture(scope="session")
def molmo2_app(molmo2_server_mod):
    return molmo2_server_mod.create_app(
        molmo2_server_mod.FakeBackend(model_id="allenai/Molmo2-8B"))


@pytest.fixture(scope="session")
def medgemma_bf16_app(medgemma_server_mod):
    return medgemma_server_mod.create_app(medgemma_server_mod.FakeBackend(),
                                          "google/medgemma-1.5-4b-it")


@pytest.fixture()
def gui_actor_client(gui_actor_app):
    from fastapi.testclient import TestClient

    with TestClient(gui_actor_app) as client:  # context manager runs startup hooks
        yield client


@pytest.fixture()
def molmo2_client(molmo2_app):
    from fastapi.testclient import TestClient

    with TestClient(molmo2_app) as client:
        yield client


@pytest.fixture()
def medgemma_client_fixture(medgemma_bf16_app):
    from fastapi.testclient import TestClient

    with TestClient(medgemma_bf16_app) as client:
        yield client
