"""Cross-revision Phase-6 default-segmenter identity driver (support module for test_p6_default_identity.py,
not collected by pytest): runs the SAME scripted op sequence on ONE segd source tree
(whichever root is passed as argv[1]) and dumps every observation text plus the sha256
of every rendered PNG to JSON — the offline stand-in for the extension's op deadline
view: the agent only ever sees the observation text and the images.

Usage:
    python _p6_identity_driver.py <segd_src_root> <out_json> <case_out_dir>

The sequence (open, sheet, view, point, box, edit, undo, finish review incl. candidates,
finish confirm) uses FakeNNI (nni_url=None) exactly like the existing v2 core tests and
the DEFAULT segmenter (no segmenter argument), so the driver runs on the pre-Phase-6 main
tree unchanged.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])          # the segd tree under test resolves first

import numpy as np                        # noqa: E402
from segd.case_state import CaseState     # noqa: E402
import segd.ops                           # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))   # worktree tests dir: _helpers
from _helpers import lesion_center, make_phantom   # noqa: E402


def _png_hashes(images) -> list:
    out = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append(hashlib.sha256(buf.getvalue()).hexdigest())
    return out


def main() -> None:
    img, lesion, _body = make_phantom()
    centre = lesion_center(lesion)
    case = CaseState("phantom", img, np.eye(4), (1.0, 1.0, 1.0),
                     "bright lesion in the limb", "T2", sys.argv[3])
    case.initialize()
    # a fine mark far from the lesion but VISIBLE on the axial slice we view: pointing
    # there adds a body-tissue blob, so the first finish review also renders the
    # candidate sheet (the candidates come from the GT-free intensity generator; the
    # GT knowledge stays in this driver, as in the existing v2 tests)
    vis = case.markset.visible("axial", centre[2], None, "fine")
    far = max(vis, key=lambda m: sum((a - b) ** 2 for a, b in zip(m.ijk, centre))).id

    out: dict = {}
    near = sorted(vis, key=lambda m: sum((a - b) ** 2 for a, b in zip(m.ijk, centre)))[:2]
    for op, args in (("open", {}), ("sheet", {"axis": "axial", "start": 10, "stop": 70}),
                     ("view", {"axis": "axial", "slice": centre[2], "marks": "fine"}),
                     ("point", {"ids": [near[0].id]}),
                     ("box", {"id_a": near[0].id, "id_b": near[1].id}),
                     ("edit", {"op": "dilate", "radius_mm": 2.0}),
                     ("undo", {}),
                     ("point", {"ids": [far]}),
                     ("finish", {"confidence": "medium", "reason": "check"})):
        o = segd.ops.run_op(case, op, args)     # raises on failure: identity needs a
        n = sum(1 for k in out if k.split("#")[0] == op)
        out[op if n == 0 else f"{op}#{n}"] = {"text": o.text, "png": _png_hashes(o.images)}
        if op == "finish":
            break
    code = re.search(r'review_code "(\d{4})"', out["finish"]["text"]).group(1)
    o = segd.ops.run_op(case, "finish", {"confidence": "high", "reason": "ok",
                                         "review_code": code})
    out["finish_confirm"] = {"text": o.text, "png": _png_hashes(o.images)}
    out["_meta"] = {"current": case.current, "done": case.done,
                    "tool_calls": case.tool_calls, "model_prompts": case.model_prompts}
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"candidate sheet rendered: {len(out['finish']['png']) == 2}; "
          f"v{case.current}, done={case.done}")


if __name__ == "__main__":
    main()
