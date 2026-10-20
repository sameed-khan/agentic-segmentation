"""Cross-revision identity driver (support module for test_identity.py, not collected by
pytest). Runs ONE scripted op sequence on ONE segd source tree (the root passed as argv[1])
and dumps, per op, the observation text and the sha256 of every rendered PNG plus the
normalized trajectory records to JSON — the offline stand-in for what the agent sees.
test_identity.py runs it on the integrated tree and on a pinned reference tree and
requires identical JSON.

Usage:
    python _identity_driver.py <segd_src_root> <out_json> <case_dir> <scenario>

Scenarios (the segd tree is imported from <segd_src_root>; FakeNNI / fake services only):
- ``default``   no optional argument at all (also runs on the pre-v3/pre-Phase-6 tag
                harness-v2.0.1): open, sheet, view, point, box, edit, undo, point, finish
                review INCLUDING the candidate sheet, finish confirm.
- ``sammed3d``  ``segmenter="sammed3d"`` against the sammed3d service's ``--fake`` backend
                (FakePatchPredictor) on a localhost port; same op sequence as ``default``.
- ``v3``        harness v3: fake pointd + scripted MedGemma services (localhost), checkpoints
                localize/review/candidates, seg_locate; open, locate, sheet, view, point,
                finish review (+candidate sheet), finish confirm.
- ``beamster``  ``dataset="beamster"`` on a two-lesion phantom: dense marks profile, multi-
                component observation text, component review sheet.

Environment: ``SEGD_IDENT_SERVICES_ROOT`` = the harness/ dir holding local_models/ (fake
service modules; both trees under comparison use the SAME services), ``SAMED3D_SRC`` for the
sammed3d bridge of a tree that has no sibling local_models project.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])          # the segd tree under test resolves first

import numpy as np                        # noqa: E402
from segd.case_state import CaseState     # noqa: E402
import segd.ops                           # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))   # worktree tests dir: _helpers
from _fake_services import _serve                       # noqa: E402
from _helpers import lesion_center, make_phantom, make_two_lesion_phantom   # noqa: E402

VOLATILE_KEYS = {"ts", "timings", "latency_ms", "url", "wall_ms", "service_latency_ms", "elapsed_ms"}


def _png_hashes(images) -> list:
    out = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append(hashlib.sha256(buf.getvalue()).hexdigest())
    return out


def _scrub(obj):
    """Drop wall-clock/port dependent keys from a trajectory record."""
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items() if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    return obj


def _trajectory(case) -> list:
    path = Path(case.out_dir) / "trajectory.jsonl"
    return [_scrub(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def _record(out: dict, key: str, o) -> None:
    n = sum(1 for k in out if k.split("#")[0] == key)
    out[key if n == 0 else f"{key}#{n}"] = {"text": o.text, "png": _png_hashes(o.images)}


def _finish_and_confirm(case, out: dict) -> None:
    code = re.search(r'review_code "(\d{4})"', out["finish"]["text"]).group(1)
    _record(out, "finish_confirm", segd.ops.run_op(
        case, "finish", {"confidence": "high", "reason": "ok", "review_code": code}))


def _dump(case, out: dict, out_json: str) -> None:
    out["_meta"] = {"current": case.current, "done": case.done,
                    "tool_calls": case.tool_calls, "model_prompts": case.model_prompts}
    out["_trajectory"] = _trajectory(case)
    Path(out_json).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"finish sheets: {len(out['finish']['png'])}; v{case.current}, done={case.done}")


def _services_root() -> Path:
    return Path(os.environ["SEGD_IDENT_SERVICES_ROOT"])


def _run_v2_sequence(case, centre, out: dict) -> None:
    """The v2 sequence shared by ``default`` and ``sammed3d``."""
    vis = case.markset.visible("axial", centre[2], None, "fine")
    # a fine mark far from the lesion but visible on the viewed slice: pointing there adds a
    # body-tissue blob, so the first finish review also renders the candidate sheet
    far = max(vis, key=lambda m: sum((a - b) ** 2 for a, b in zip(m.ijk, centre))).id
    near = sorted(vis, key=lambda m: sum((a - b) ** 2 for a, b in zip(m.ijk, centre)))[:2]
    for op, args in (("open", {}), ("sheet", {"axis": "axial", "start": 10, "stop": 70}),
                     ("view", {"axis": "axial", "slice": centre[2], "marks": "fine"}),
                     ("point", {"ids": [near[0].id]}),
                     ("box", {"id_a": near[0].id, "id_b": near[1].id}),
                     ("edit", {"op": "dilate", "radius_mm": 2.0}),
                     ("undo", {}),
                     ("point", {"ids": [far]}),
                     ("finish", {"confidence": "medium", "reason": "check"})):
        _record(out, op, segd.ops.run_op(case, op, args))   # raises on failure
    _finish_and_confirm(case, out)


def scenario_default(case_dir: str, out_json: str, **case_kw) -> None:
    img, lesion, _body = make_phantom()
    case = CaseState("phantom", img, np.eye(4), (1.0, 1.0, 1.0),
                     "bright lesion in the limb", "T2", case_dir, **case_kw)
    case.initialize()
    out: dict = {}
    _run_v2_sequence(case, lesion_center(lesion), out)
    _dump(case, out, out_json)


def scenario_sammed3d(case_dir: str, out_json: str) -> None:
    root = os.environ.get("SAMED3D_SRC") or str(_services_root() / "local_models" / "sammed3d" / "src")
    os.environ["SAMED3D_SRC"] = root
    sys.path.insert(0, root)
    from sammed3d import service
    url, _stop = _serve(service.create_app(service.FakePatchPredictor(), model_info={"mode": "fake"}))
    img, lesion, _body = make_phantom((120, 120, 80), z_halfwidth=0.25)
    case = CaseState("phantom", img, np.eye(4), (1.0, 1.0, 1.0),
                     "bright lesion in the limb", "T2", case_dir,
                     segmenter="sammed3d", segmenter_url=url)
    case.initialize()
    out: dict = {}
    try:
        _run_v2_sequence(case, lesion_center(lesion), out)
        _dump(case, out, out_json)
    finally:
        case.close()


def scenario_v3(case_dir: str, out_json: str) -> None:
    from _fake_services import FixedPointerBackend, ScriptedMedGemmaBackend, _load_module
    lm = _services_root() / "local_models"
    pointer = FixedPointerBackend([])
    pointer_url, _ = _serve(_load_module("ident_pointd", lm / "pointd_gui_actor" / "server.py")
                            .create_app(pointer))
    medgemma_url, _ = _serve(_load_module("ident_medgemma", lm / "medgemma" / "start_bf16_server.py")
                             .create_app(ScriptedMedGemmaBackend(), "google/medgemma-1.5-4b-it"))
    img, lesion, _body = make_phantom()
    centre = lesion_center(lesion)
    case = CaseState("phantom", img, np.eye(4), (1.0, 1.0, 1.0),
                     "bright lesion in the limb", "T2", case_dir,
                     v3={"pointer": "gui_actor", "pointer_url": pointer_url,
                         "medgemma": "q4", "medgemma_url": medgemma_url,
                         "checkpoints": ["localize", "review", "candidates"]})
    case.initialize()
    out: dict = {}
    _record(out, "open", segd.ops.run_op(case, "open", {}))
    # the localize checkpoint fires on the first sheet; the pointer proposes drawn marks of
    # the CURRENT view (panel-pixel points), so view first
    _record(out, "view", segd.ops.run_op(
        case, "view", {"axis": "axial", "slice": centre[2], "marks": "fine"}))
    drawn = sorted((int(mid), float(x), float(y))
                   for mid, _panel, x, y in case.active_view.render_info.marks)
    pointer.points = [(x, y, 0.9 - 0.1 * i) for i, (_m, x, y) in enumerate(drawn[:3])]
    _record(out, "locate", segd.ops.run_op(case, "locate", {}))
    _record(out, "sheet", segd.ops.run_op(case, "sheet", {"axis": "axial", "start": 10, "stop": 70}))
    vis = case.markset.visible("axial", centre[2], None, "fine")
    near = min(vis, key=lambda m: sum((a - b) ** 2 for a, b in zip(m.ijk, centre))).id
    far = max(vis, key=lambda m: sum((a - b) ** 2 for a, b in zip(m.ijk, centre))).id
    _record(out, "view", segd.ops.run_op(
        case, "view", {"axis": "axial", "slice": centre[2], "marks": "fine"}))
    for ids in ([near], [far]):
        _record(out, "point", segd.ops.run_op(case, "point", {"ids": ids}))
    _record(out, "finish", segd.ops.run_op(case, "finish", {"confidence": "medium", "reason": "check"}))
    _finish_and_confirm(case, out)
    out["_v3_fired"] = sorted(case.v3_fired)
    _dump(case, out, out_json)


def scenario_beamster(case_dir: str, out_json: str) -> None:
    img, lesion_a, lesion_b = make_two_lesion_phantom()
    ca, cb = lesion_center(lesion_a), lesion_center(lesion_b)
    case = CaseState("phantom", img, np.eye(4), (1.0, 1.0, 1.0),
                     "brain metastases", "T1c", case_dir, dataset="beamster")
    case.initialize()
    out: dict = {}
    _record(out, "open", segd.ops.run_op(case, "open", {}))
    _record(out, "sheet", segd.ops.run_op(case, "sheet", {"axis": "axial", "start": 10, "stop": 70}))
    _record(out, "view", segd.ops.run_op(
        case, "view", {"axis": "axial", "slice": ca[2], "marks": "fine"}))
    vis = case.markset.visible("axial", ca[2], None, "fine")
    for centre in (ca, cb):
        mark = min(vis, key=lambda m: sum((a - b) ** 2 for a, b in zip(m.ijk, centre))).id
        _record(out, "point", segd.ops.run_op(case, "point", {"ids": [mark]}))
    _record(out, "finish", segd.ops.run_op(case, "finish", {"confidence": "medium", "reason": "check"}))
    _finish_and_confirm(case, out)
    _dump(case, out, out_json)


SCENARIOS = {"default": scenario_default, "sammed3d": scenario_sammed3d, "v3": scenario_v3,
             "beamster": scenario_beamster}


if __name__ == "__main__":
    SCENARIOS[sys.argv[4]](sys.argv[3], sys.argv[2])
