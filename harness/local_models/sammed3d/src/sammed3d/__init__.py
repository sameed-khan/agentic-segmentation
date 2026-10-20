"""sammed3d: SAM-Med3D local segmenter service + client (Phase 6 segmenter swap).

Public surface:
- ``client.SamMed3DSession``: call-signature drop-in for the nnInteractive remote
  session, pointing at the ``sammed3d-service`` (default 127.0.0.1:8095).
- ``service.create_app``: the FastAPI app (``--fake`` backend for offline tests);
  imported lazily so the client keeps the segd env fastapi-import-free at call time.
- ``geometry``: pure numpy/scipy patch/tiling/prompt-mapping/stitching math.
- ``engine.SamMed3DEngine``: the real torch backend (official SAM-Med3D-turbo);
  always imported lazily (torch).
"""
from sammed3d.client import (
    SamMed3DSession, SessionExpiredError, ServerAtCapacityError, decode_mask, encode_mask,
)

DEFAULT_PORT = 8095
DEFAULT_URL = "http://127.0.0.1:8095"
SERVICE_VERSION = "0.1.0"

__all__ = [
    "SamMed3DSession", "SessionExpiredError", "ServerAtCapacityError",
    "decode_mask", "encode_mask", "DEFAULT_PORT", "DEFAULT_URL", "SERVICE_VERSION",
]


def __getattr__(name):
    if name == "create_app":
        from sammed3d.service import create_app
        return create_app
    raise AttributeError(name)
