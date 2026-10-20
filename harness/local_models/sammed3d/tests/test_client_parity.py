"""Client signature parity: SamMed3DSession must expose exactly the nnInteractive
remote-session call signatures segd relies on (inspect-signature comparison)."""
from __future__ import annotations

import inspect

from nnInteractive.inference.remote import nnInteractiveRemoteInferenceSession

from sammed3d.client import SamMed3DSession

METHODS = [
    "set_image", "set_target_buffer", "set_do_autozoom", "reset_interactions",
    "add_point_interaction", "add_bbox_interaction", "add_lasso_interaction",
    "add_scribble_interaction", "add_initial_seg_interaction", "_predict",
    "initialize_from_trained_model_folder", "ping", "close", "undo",
]


def _call_shape(sig: inspect.Signature) -> list:
    return [(p.name, p.kind, p.default) for p in sig.parameters.values()]


def test_public_method_signatures_match_nninteractive():
    for name in METHODS:
        mine = inspect.signature(getattr(SamMed3DSession, name))
        theirs = inspect.signature(getattr(nnInteractiveRemoteInferenceSession, name))
        # what callers depend on: parameter names, kinds and defaults (annotation text may
        # differ: the sammed3d client has no torch, so `Union[np.ndarray, torch.Tensor]`
        # is spelled `np.ndarray` there)
        assert _call_shape(mine) == _call_shape(theirs), (
            f"{name}: {mine} != {theirs}")


def test_constructor_takes_the_callers_arguments():
    # CaseState builds the session as (url, read_timeout=300, write_timeout=600)
    sig = inspect.signature(SamMed3DSession.__init__)
    theirs = inspect.signature(nnInteractiveRemoteInferenceSession.__init__)
    assert [p.name for p in sig.parameters.values()] == \
        [p.name for p in theirs.parameters.values()]
    assert [p.default for p in sig.parameters.values()] == \
        [p.default for p in theirs.parameters.values()]
