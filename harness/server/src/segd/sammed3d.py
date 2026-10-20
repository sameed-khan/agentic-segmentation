"""Bridge to the SAM-Med3D local-model project (Phase 6 segmenter swap).

The sammed3d project (``harness/local_models/sammed3d``) owns the HTTP client
(``SamMed3DSession``, a call-signature drop-in for the nnInteractive remote session)
and the service; this module only locates and loads it, so the segd environment never
needs torch/medim. Path resolution walks up from this file to
``<repo>/harness/local_models/sammed3d/src``; the environment variable ``SAMED3D_SRC``
overrides it (used when a copied segd tree has no sibling project).

``CLIENT_FACTORY`` lets tests inject a session built on a test transport (no real
HTTP server); production code never sets it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from segd import config

CLIENT_FACTORY = None        # tests: factory(url) -> session-like object


def _src_root() -> Path:
    env = os.environ.get("SAMED3D_SRC")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "local_models" / "sammed3d" / "src"


def _load_client_module():
    root = _src_root()
    if not root.is_dir():
        raise ImportError(
            f"SAM-Med3D client not found at {root} (set SAMED3D_SRC to the "
            "sammed3d project's src directory)")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import sammed3d.client as module
    return module


def open_session(url: str, read_timeout: float = 300.0, write_timeout: float = 600.0):
    """Build a SamMed3DSession (or a CLIENT_FACTORY-provided test double)."""
    if CLIENT_FACTORY is not None:
        return CLIENT_FACTORY(url)
    return _load_client_module().SamMed3DSession(url, read_timeout=read_timeout,
                                                 write_timeout=write_timeout)


def session_expired_types() -> tuple:
    """Exception types a sammed3d session raises when the service lost the session;
    CaseState's reconnect logic treats them like nnInteractive session expiry. Empty
    until the sammed3d client module has been loaded (i.e. on the default nnInteractive
    path nothing from this bridge is imported or touched)."""
    module = sys.modules.get("sammed3d.client")
    return () if module is None else (module.SessionExpiredError,)


def default_url() -> str:
    return config.SAMED3D_URL
