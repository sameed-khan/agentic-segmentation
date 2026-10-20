"""Harness v3 (seg_locate + MedGemma second reader) on fake services.

Fake services run over REAL HTTP (uvicorn threads on random ports, --fake backends from
harness/local_models — no torch, no GPU): a scripted MedGemma OpenAI server and a
fixed-output pointd pointer. Core tests use the real segd.marks/segd.render (deterministic
on the phantom; the stub path is covered by the existing v2 suites).

Covered: flag plumbing/validation, observation byte-identity when v3 is off, seg_locate
declaration-gating (server-side guard), pixel->mark mapping/dedup/ranking (global and
edge marks), budget counting, error paths, checkpoint injection exactly once per
checkpoint per version, trajectory v3 logging, review_code/authorization unaffected,
case_meta recording, full-flow with pointer + MedGemma together, the MedGemma ABSOLUTE
per-op deadline (set at run_op entry: pre-hook delay + several individually successful
delayed replies still return the op in time with unavailable notes; per-call clamp;
skipped checkpoints), candidate-id binding (a reply naming another candidate is
unusable), replies whose validation raises (confidence 10**309) stay unavailable AND
unattributed, whole-iteration per-candidate protection (a row crop failure keeps
earlier advice and later rows still run), per-checkpoint finish-hook protection with
notes retained, and checkpoint-delivery state restore so a failed persistence retry
still delivers.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import segd.ops
from segd import config
from segd.app import create_app
from segd.case_state import OpError, edge_mark_map
from _fake_services import FixedPointerBackend, ScriptedMedGemmaBackend, _load_module, _serve
from _helpers import install_fake_data, lesion_center, make_phantom

HARNESS_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "harness")


def make_case_state_v3(tmp_path, case_id="phantom", v3=None, root="run1"):
    """Initialized CaseState on the phantom with the REAL marks/render (deterministic)."""
    from segd.case_state import CaseState
    img, lesion, _body = make_phantom()
    cs = CaseState(case_id, img, np.eye(4), (1.0, 1.0, 1.0), "bright lesion in the limb", "T2",
                   tmp_path / "results" / root / case_id, v3=v3)
    cs.initialize()
    return cs, lesion


def _partial_lesion_mask(lesion) -> np.ndarray:
    """Superior half of the lesion as a version: the missing adjacent half is proposed
    by the intensity candidate generator (test-side GT knowledge, as in the E14/E15 tests)."""
    kk = np.indices(lesion.shape)[2]
    return (lesion & (kk >= lesion_center(lesion)[2])).astype(np.uint8)


def _drawn_positions(cs) -> dict:
    """Drawn global marks of the active view: id -> (x, y) panel-0 pixel."""
    ri = cs.active_view.render_info
    return {int(mid): (float(x), float(y)) for mid, _panel, x, y in ri.marks}


# ---------------------------------------------------------------- fake services
class ShiftIdBackend:
    """Malicious/off-contract candidate backend: answers every row with the NEXT
    candidate id (a C1 request gets candidate_id C2) — regression for review P1 #2."""

    name = "scripted"
    model_id = "google/medgemma-1.5-4b-it"

    def __init__(self):
        self.calls: list[str] = []

    def load(self):
        pass

    def generate(self, prompt_text: str, images_b64, temperature: float, max_tokens: int) -> str:
        m = re.search(r"candidate (C\d+)", prompt_text)
        requested = m.group(1) if m else "C?"
        self.calls.append(requested)
        return json.dumps({"candidate_id": f"C{int(requested[1:]) + 1}",
                           "verdict": "lesion", "confidence": 0.9})


class DelayedBackend:
    """MedGemma backend that sleeps before every reply (group-deadline regression,
    review P1 #1): delayed responses, not only immediate connection refusal."""

    name = "delayed"
    model_id = "google/medgemma-1.5-4b-it"

    def __init__(self, delay: float):
        self.delay = delay
        self.calls: list[str] = []

    def load(self):
        pass

    def generate(self, prompt_text: str, images_b64, temperature: float, max_tokens: int) -> str:
        self.calls.append(prompt_text[:60])
        time.sleep(self.delay)
        if '"complete"' in prompt_text:
            return json.dumps({"complete": "no", "problems": [], "confidence": 0.6})
        if '"verdict"' in prompt_text:
            m = re.search(r"candidate (C\d+)", prompt_text)
            return json.dumps({"candidate_id": m.group(1) if m else "?",
                               "verdict": "lesion", "confidence": 0.8})
        return json.dumps({"box": None, "confidence": 0.5})


@pytest.fixture()
def pointer_url():
    mod = _load_module("v3_test_pointd",
                       HARNESS_DIR / "local_models" / "pointd_gui_actor" / "server.py")
    holder = {}

    def make(points):
        url, stop = _serve(mod.create_app(FixedPointerBackend(points)))
        holder["stop"] = stop
        return url

    yield make
    if holder.get("stop"):
        holder["stop"]()


@pytest.fixture()
def medgemma_url():
    mod = _load_module("v3_test_medgemma",
                       HARNESS_DIR / "local_models" / "medgemma" / "start_bf16_server.py")
    holder = {}

    def make(backend=None):
        b = backend if backend is not None else mod.FakeBackend()
        url, stop = _serve(mod.create_app(b, "google/medgemma-1.5-4b-it"))
        holder["stop"] = stop
        return url

    yield make
    if holder.get("stop"):
        holder["stop"]()


DEAD_URL = "http://127.0.0.1:1"   # nothing listens there; connection refused immediately


# ------------------------------------------------------------------- flag plumbing
def test_normalize_v3_flags():
    from segd.v3 import normalize_v3
    assert config.V3_CHECKPOINTS == ("localize", "review", "candidates")
    assert normalize_v3(None) is None
    assert normalize_v3({}) == {"pointer": None, "pointer_url": None, "medgemma": None,
                                "medgemma_url": None,
                                "checkpoints": ["localize", "review", "candidates"]}
    spec = normalize_v3({"pointer": "molmo2", "medgemma": "bf16",
                         "checkpoints": ["review", "review", "localize"]})
    assert spec["pointer_url"] == "http://127.0.0.1:8093"
    assert spec["medgemma_url"] == "http://127.0.0.1:8094"
    assert spec["checkpoints"] == ["localize", "review"]      # canonical order, dedup
    for bad in ({"pointer": "nonsense"}, {"medgemma": "q8"}, {"checkpoints": ["nope"]},
                {"pointer_url": "http://x"}, {"unknown": 1}, "v3"):
        with pytest.raises(ValueError):
            normalize_v3(bad)


def test_http_case_open_records_v3_and_rejects_bad(tmp_path, monkeypatch):
    install_fake_data(monkeypatch)
    app = create_app()
    c = TestClient(app)
    r = c.post("/cases", json={"case_id": "phantom", "run_id": "r1",
                               "out_dir": str(tmp_path / "results"),
                               "v3": {"pointer": "gui_actor", "medgemma": "q4"}})
    assert r.status_code == 200
    cs = app.state.segd["cases"]["phantom"]
    assert cs.v3["pointer"] == "gui_actor" and cs.v3["medgemma"] == "q4"
    assert cs.v3["pointer_url"] == "http://127.0.0.1:8092"
    meta = json.loads((tmp_path / "results" / "r1" / "phantom" / "case_meta.json").read_text())
    assert meta["v3"] == cs.v3
    bad = {"case_id": "phantom", "run_id": "r1", "out_dir": str(tmp_path / "results")}
    assert c.post("/cases", json={**bad, "v3": {"pointer": "nope"}}).status_code == 400
    assert c.post("/cases", json={**bad, "v3": {"pointer_url": "http://x"}}).status_code == 400
    assert "v3" in c.post("/cases", json={**bad, "v3": {"pointer": "nope"}}).json()["detail"]


def test_v3_off_observations_identical(tmp_path):
    """No-v3 and all-off v3 produce byte-identical observations and counters; seg_locate
    is rejected server-side when no pointer is configured."""
    runs = []
    for i, v3 in enumerate((None, {"pointer": None, "medgemma": None, "checkpoints": []})):
        cs, _lesion = make_case_state_v3(tmp_path, root=f"r{i}", v3=v3)
        with pytest.raises(OpError, match="seg_locate is disabled"):
            segd.ops.run_op(cs, "locate", {})
        texts = []
        for op, args in (("open", {}), ("sheet", {"axis": "axial", "start": 10, "stop": 70}),
                         ("view", {"axis": "axial", "slice": 40, "marks": "fine"}),
                         ("finish", {"confidence": "high", "reason": "r"})):
            texts.append(segd.ops.run_op(cs, op, args).text)
        code = re.search(r'review_code "(\d{4})"', texts[-1]).group(1)
        texts.append(segd.ops.run_op(cs, "finish", {"confidence": "high", "reason": "r",
                                                    "review_code": code}).text)
        runs.append(texts)
        assert cs.done and len(cs.v3_fired) == 0
    assert runs[0] == runs[1] and len(runs[0]) == 5


# ------------------------------------------------------------------------- seg_locate
def test_seg_locate_mapping_dedup_ranking(tmp_path, pointer_url):
    from segd.v3 import normalize_v3
    cs, lesion = make_case_state_v3(tmp_path)
    k = lesion_center(lesion)[2]
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": k, "marks": "fine"})
    drawn = _drawn_positions(cs)
    ma, mb, mc = sorted(drawn)[:3]                       # drawn marks are >=26 px apart
    (ax, ay), (bx, by), (cx, cy) = drawn[ma], drawn[mb], drawn[mc]
    cs.v3 = normalize_v3({"pointer": "gui_actor",
                          "pointer_url": pointer_url([(ax + 0.5, ay + 0.5, 0.91),
                                                      (ax + 2.0, ay + 2.0, 0.80),
                                                      (bx, by, 0.75), (cx, cy, 0.60),
                                                      (5.0, 5.0, 0.55)]),
                          "checkpoints": []})
    calls, prompts = cs.tool_calls, cs.model_prompts
    out = segd.ops.run_op(cs, "locate", {})
    assert cs.tool_calls == calls + 1 and cs.model_prompts == prompts    # call, not prompt
    lines = out.text.splitlines()
    assert lines[0] == "locate (gui_actor): 5 proposed point(s), nearest marks:"
    assert f"P1 mark {ma} score 0.91 (0.7 px)" in lines[1]
    assert f"P3 mark {mb} score 0.75 (0.0 px)" in lines[2]
    assert f"P4 mark {mc} score 0.6 (0.0 px)" in lines[3]
    assert not any(line.startswith("P2 ") for line in lines)             # deduped away
    assert re.match(r"P5 mark \d+ score 0.55 ", lines[4])                # nearest real mark
    assert "Proposals are hints" in out.text and "tool calls" in out.text
    assert len(out.images) == 1                                          # proposed-points image
    # trajectory: the v3 record with hashes + the per-point snap list
    rec = [json.loads(l) for l in (cs.out_dir / "trajectory.jsonl").read_text().splitlines()][-1]
    assert rec["op"] == "locate" and rec["error"] is None
    v3rec = rec["v3"][0]
    assert v3rec["call"] == "locate" and v3rec["ok"] is True
    assert v3rec["raw"] == "fixed 'bright lesion in the limb'"
    assert v3rec["latency_ms"] > 0 and len(v3rec["image_sha256"]) == 64
    assert [s["point"] for s in v3rec["snapped"]] == ["P1", "P2", "P3", "P4", "P5"]
    assert [s["mark"] for s in v3rec["snapped"]] == [str(ma), None, str(mb), str(mc), re.match(r"P5 mark (\d+)", lines[4]).group(1)]


def test_seg_locate_default_query_and_disabled(tmp_path, pointer_url):
    from segd.v3 import normalize_v3
    cs, lesion = make_case_state_v3(tmp_path)
    with pytest.raises(OpError, match="seg_locate is disabled"):
        segd.ops.run_op(cs, "locate", {})
    cs.v3 = normalize_v3({"pointer": "gui_actor", "pointer_url": DEAD_URL, "checkpoints": []})
    with pytest.raises(OpError, match="no active view"):
        segd.ops.run_op(cs, "locate", {})
    k = lesion_center(lesion)[2]
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": k})   # marks coarse (default)
    cs.v3["pointer_url"] = pointer_url([(300.0, 300.0, 0.9)])
    out = segd.ops.run_op(cs, "locate", {"query": "  "})        # empty -> target text
    assert "locate (gui_actor)" in out.text and "mark " in out.text


def test_seg_locate_edge_marks_and_error_path(tmp_path, pointer_url):
    from segd.v3 import normalize_v3
    cs, lesion = make_case_state_v3(tmp_path)
    k = lesion_center(lesion)[2]
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": k, "marks": "edges"})
    edges = edge_mark_map(cs.active_view.render_info)
    assert edges
    eid = sorted(edges)[0]
    erow, ecol = edges[eid]
    geom = cs.active_view.view_geom
    ex = (ecol - geom.crop[1]) * geom.scale + geom.scale / 2.0
    ey = (erow - geom.crop[0]) * geom.scale + geom.scale / 2.0
    cs.v3 = normalize_v3({"pointer": "molmo2",
                          "pointer_url": pointer_url([(ex, ey, 0.99)]), "checkpoints": []})
    out = segd.ops.run_op(cs, "locate", {})
    assert f"P1 mark {eid} score 0.99" in out.text and "molmo2" in out.text

    # error path: dead service -> agent-facing OpError, counts as a tool call, logged
    cs.v3["pointer_url"] = DEAD_URL
    calls, prompts = cs.tool_calls, cs.model_prompts
    with pytest.raises(OpError, match="pointer service unavailable"):
        segd.ops.run_op(cs, "locate", {})
    assert cs.tool_calls == calls + 1 and cs.model_prompts == prompts
    rec = [json.loads(l) for l in (cs.out_dir / "trajectory.jsonl").read_text().splitlines()][-1]
    assert rec["op"] == "locate" and rec["error_kind"] == "op"
    assert rec["v3"][0]["ok"] is False and rec["v3"][0]["error"]


def test_locate_record_keeps_structured_response(tmp_path, pointer_url):
    """Review P2 #9: the trajectory keeps the pointer service's structured response
    (points + model/service provenance), not only a truncated raw string."""
    from segd.v3 import normalize_v3
    cs, lesion = make_case_state_v3(tmp_path, root="locateparse")
    k = lesion_center(lesion)[2]
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": k, "marks": "fine"})
    drawn = _drawn_positions(cs)
    mid = sorted(drawn)[0]
    cs.v3 = normalize_v3({"pointer": "gui_actor",
                          "pointer_url": pointer_url([(drawn[mid][0], drawn[mid][1], 0.95)]),
                          "checkpoints": []})
    out = segd.ops.run_op(cs, "locate", {})
    assert f"P1 mark {mid}" in out.text
    recs = [json.loads(l) for l in (cs.out_dir / "trajectory.jsonl").read_text().splitlines()]
    v3rec = [r for r in recs if r["op"] == "locate"][-1]["v3"][0]
    pts = v3rec["parsed"]["points"]
    assert len(pts) == 1 and pts[0]["x"] == pytest.approx(drawn[mid][0])
    assert v3rec["parsed"]["model"] == "test/pointer"
    assert v3rec["parsed"]["version"] and v3rec["parsed"]["image_w"] > 0


# --------------------------------------------------------- medgemma second reader
def test_localize_checkpoint_on_first_sheet_only(tmp_path, medgemma_url):
    backend = ScriptedMedGemmaBackend()
    url = medgemma_url(backend)
    cs, lesion = make_case_state_v3(tmp_path, v3={"medgemma": "q4", "medgemma_url": url,
                                                  "checkpoints": ["localize"]})
    out = segd.ops.run_op(cs, "open", {})
    assert "MedGemma" not in out.text                    # open itself never fires
    out = segd.ops.run_op(cs, "sheet", {"axis": "axial", "start": 10, "stop": 70})
    assert "Medical second reader (MedGemma): the lesion is nearest mark" in out.text
    assert "confidence 0.70" in out.text
    out = segd.ops.run_op(cs, "sheet", {"axis": "coronal", "start": 10, "stop": 70})
    assert "MedGemma" not in out.text                    # exactly once per case
    assert len(backend.calls) == 1
    prompt, n_images = backend.calls[0]
    assert n_images == 1 and '"box": [y0' in prompt and "bright lesion" in prompt
    # trajectory: prompt + image hashes + parsed reply + latency under v3
    recs = [json.loads(l) for l in (cs.out_dir / "trajectory.jsonl").read_text().splitlines()]
    sheet_recs = [r for r in recs if r["op"] == "sheet"]
    v3rec = sheet_recs[0]["v3"][0]
    assert v3rec["checkpoint"] == "localize" and v3rec["ok"] is True
    assert v3rec["parsed"] == {"box": [0.4, 0.4, 0.6, 0.6], "confidence": 0.7}
    assert len(v3rec["prompt_sha256"]) == 64 and len(v3rec["image_sha256"]) == 64
    assert len(sheet_recs[1].get("v3") or []) == 0
    assert cs.v3_fired == {"localize"}


def test_localize_fallback_to_open_overview_when_no_sheet(tmp_path, medgemma_url):
    backend = ScriptedMedGemmaBackend()
    url = medgemma_url(backend)
    cs, _lesion = make_case_state_v3(tmp_path, v3={"medgemma": "bf16", "medgemma_url": url,
                                                   "checkpoints": ["localize"]})
    segd.ops.run_op(cs, "open", {})
    out = segd.ops.run_op(cs, "view", {"axis": "axial", "slice": 40})   # skipped the sheet
    assert "Medical second reader (MedGemma): the lesion is nearest mark" in out.text
    out = segd.ops.run_op(cs, "sheet", {"axis": "axial", "start": 10, "stop": 70})
    assert "MedGemma" not in out.text
    assert len(backend.calls) == 1


def test_review_and_candidates_checkpoints(tmp_path, medgemma_url):
    backend = ScriptedMedGemmaBackend(box=None)
    url = medgemma_url(backend)
    cs, lesion = make_case_state_v3(tmp_path, v3={"medgemma": "q4", "medgemma_url": url,
                                                  "checkpoints": ["review", "candidates"]})
    cs.commit_mask(_partial_lesion_mask(lesion))            # v1: candidates will be found
    out = segd.ops.run_op(cs, "finish", {"confidence": "medium", "reason": "check"})
    assert not out.done and len(out.images) == 2
    assert "Medical second reader (MedGemma): complete no (confidence 0.60)" in out.text
    assert "problems axial 999: missed_lesion" in out.text
    assert re.search(r"candidate verdicts: C\d+ lesion \(0\.80\)", out.text)
    code = re.search(r'review_code "(\d{4})"', out.text).group(1)
    n_calls = len(backend.calls)

    out = segd.ops.run_op(cs, "finish", {"confidence": "medium", "reason": "again"})
    assert "Not saved." in out.text and "MedGemma" not in out.text   # no re-fire on confirm
    assert len(backend.calls) == n_calls

    out = segd.ops.run_op(cs, "finish", {"confidence": "high", "reason": "save",
                                         "review_code": code})
    assert out.done and cs.done
    assert len(backend.calls) == n_calls


def test_review_checkpoint_fires_again_on_new_version(tmp_path, medgemma_url):
    backend = ScriptedMedGemmaBackend(box=None)
    url = medgemma_url(backend)
    cs, _lesion = make_case_state_v3(tmp_path, v3={"medgemma": "q4", "medgemma_url": url,
                                                   "checkpoints": ["review"]})
    segd.ops.run_op(cs, "finish", {"confidence": "medium", "reason": "r"})    # v0 review
    assert len(backend.calls) == 1
    cs.commit_mask(cs.body.astype(np.uint8))                 # v1: a new version
    out = segd.ops.run_op(cs, "finish", {"confidence": "medium", "reason": "r"})
    assert "Medical second reader (MedGemma): complete" in out.text   # new version: fires
    assert len(backend.calls) == 2 and cs.v3_fired == {"review:0", "review:1"}
    code = re.search(r'review_code "(\d{4})"', out.text).group(1)
    segd.ops.run_op(cs, "finish", {"confidence": "high", "reason": "save", "review_code": code})
    assert cs.done


def test_second_reader_off_contract_reply_never_crashes(tmp_path, medgemma_url):
    """Valid JSON but off-contract (stock fake backend): no crash, honest note text."""
    url = medgemma_url()
    cs, lesion = make_case_state_v3(tmp_path, v3={"medgemma": "bf16", "medgemma_url": url,
                                                  "checkpoints": ["localize", "review"]})
    cs.commit_mask(_partial_lesion_mask(lesion))
    out = segd.ops.run_op(cs, "sheet", {"axis": "axial", "start": 10, "stop": 70})
    assert "Medical second reader (MedGemma): no usable lesion box" in out.text
    out = segd.ops.run_op(cs, "finish", {"confidence": "medium", "reason": "r"})
    assert "Medical second reader (MedGemma): complete uncertain" in out.text


def test_second_reader_unavailable_never_crashes(tmp_path, medgemma_url):
    """Transport failure (dead port): the short unavailable note, the op succeeds."""
    cs, lesion = make_case_state_v3(tmp_path, v3={"medgemma": "q4", "medgemma_url": DEAD_URL,
                                                  "checkpoints": ["localize", "review"]})
    cs.commit_mask(_partial_lesion_mask(lesion))
    out = segd.ops.run_op(cs, "sheet", {"axis": "axial", "start": 10, "stop": 70})
    assert "Medical second reader (MedGemma): unavailable this checkpoint" in out.text
    out = segd.ops.run_op(cs, "finish", {"confidence": "medium", "reason": "r"})
    assert "unavailable this checkpoint" in out.text and not cs.done
    recs = [json.loads(l) for l in (cs.out_dir / "trajectory.jsonl").read_text().splitlines()]
    v3rec = [r for r in recs if r["op"] == "sheet"][0]["v3"][0]
    assert v3rec["ok"] is False and v3rec["error"] and v3rec["latency_ms"] is not None


def test_missing_prompt_file_notes_unavailable(tmp_path, monkeypatch, medgemma_url):
    url = medgemma_url()
    monkeypatch.setattr(config, "V3_PROMPT_DIR", tmp_path / "no-prompts-here")
    cs, _lesion = make_case_state_v3(tmp_path, v3={"medgemma": "q4", "medgemma_url": url,
                                                   "checkpoints": ["localize"]})
    segd.ops.run_op(cs, "open", {})
    out = segd.ops.run_op(cs, "sheet", {"axis": "axial", "start": 10, "stop": 70})
    assert "unavailable this checkpoint" in out.text
    assert "the lesion is nearest mark" not in out.text


def test_review_code_and_authorization_unaffected_by_v3(tmp_path, medgemma_url):
    backend = ScriptedMedGemmaBackend(box=None)
    url = medgemma_url(backend)
    cs, lesion = make_case_state_v3(tmp_path, v3={"medgemma": "q4", "medgemma_url": url,
                                                  "checkpoints": ["review", "candidates"]})
    cs.commit_mask(_partial_lesion_mask(lesion))
    segd.ops.run_op(cs, "candidates", {})
    assert cs.candidates_shown == cs.current
    out = segd.ops.run_op(cs, "finish", {"confidence": "medium", "reason": "r"})
    code = re.search(r'review_code "(\d{4})"', out.text).group(1)
    out = segd.ops.run_op(cs, "finish", {"confidence": "high", "reason": "r",
                                         "review_code": "0000"})
    assert "Not saved." in out.text and "MedGemma" not in out.text   # wrong code: not saved
    cs.commit_mask(cs.body.astype(np.uint8))                 # mask change -> authorization gone
    assert cs.candidates_shown is None and cs.reviewed_version is None
    out = segd.ops.run_op(cs, "finish", {"confidence": "high", "reason": "r",
                                         "review_code": code})
    assert "Final review of v2" in out.text                  # new version: new review
    assert "review_code" in out.text


def test_all_v3_on_full_flow(tmp_path, pointer_url, medgemma_url):
    """Pointer + MedGemma together: view -> locate -> point -> finish review."""
    from segd.v3 import normalize_v3
    cs, lesion = make_case_state_v3(
        tmp_path, v3={"pointer": "gui_actor", "medgemma": "q4",
                      "checkpoints": ["localize", "review", "candidates"]})
    segd.ops.run_op(cs, "open", {})
    k = lesion_center(lesion)[2]
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": k, "marks": "fine"})
    drawn = _drawn_positions(cs)
    mid0 = sorted(drawn)[0]
    mx, my = drawn[mid0]
    cs.v3 = normalize_v3({"pointer": "gui_actor", "pointer_url": pointer_url([(mx, my, 0.95)]),
                          "medgemma": "q4", "medgemma_url": medgemma_url(ScriptedMedGemmaBackend()),
                          "checkpoints": ["localize", "review", "candidates"]})
    out = segd.ops.run_op(cs, "locate", {})                  # tool call, no prompt
    assert "MedGemma" not in out.text
    assert cs.model_prompts == 0
    out = segd.ops.run_op(cs, "point", {"ids": [mid0]})      # the model prompt still works
    assert out.text and not out.done
    out = segd.ops.run_op(cs, "finish", {"confidence": "medium", "reason": "r"})
    assert "review_code" in out.text


# ----------------------------------------------------- group deadline (review P1 #1)
def test_medgemma_call_skips_when_group_deadline_exhausted(tmp_path):
    """Group budget spent -> the call is NOT sent and the record says why (the caller
    appends the unavailable note)."""
    from segd.v3 import medgemma_call
    cs, _lesion = make_case_state_v3(tmp_path, root="deadgroup",
                                     v3={"medgemma": "q4", "medgemma_url": DEAD_URL,
                                         "checkpoints": ["review"]})
    cs._log_extra = {}
    cs._v3_group_deadline = time.perf_counter() - 1.0      # budget already spent
    parsed = medgemma_call(cs, "review", Image.new("RGB", (4, 4)), "prompt")
    assert parsed is None
    rec = cs._log_extra["v3"][0]
    assert rec["ok"] is False and rec["parsed"] is None and rec["latency_ms"] == 0.0
    assert "deadline" in rec["error"] and "not sent" in rec["error"]


def test_medgemma_call_deadline_clamps_single_slow_call(tmp_path, medgemma_url, monkeypatch):
    """A single delayed reply is cut off at the group budget, so the unavailable note is
    delivered while the op is still inside the extension's HTTP deadline (previously a
    180 s call raced the 180 s op abort and the note was lost)."""
    monkeypatch.setattr(config, "V3_MEDGEMMA_GROUP_TIMEOUT_S", 1.0)
    monkeypatch.setattr(config, "V3_MEDGEMMA_RESPONSE_RESERVE_S", 0.5)
    backend = DelayedBackend(delay=6.0)
    url = medgemma_url(backend)
    cs, lesion = make_case_state_v3(tmp_path, root="slowcall",
                                    v3={"medgemma": "q4", "medgemma_url": url,
                                        "checkpoints": ["localize"]})
    out = segd.ops.run_op(cs, "sheet", {"axis": "axial", "start": 10, "stop": 70})
    assert "Medical second reader (MedGemma): unavailable this checkpoint" in out.text
    assert len(backend.calls) == 1                        # the call reached the service once
    recs = [json.loads(l) for l in (cs.out_dir / "trajectory.jsonl").read_text().splitlines()]
    v3rec = [r for r in recs if r["op"] == "sheet"][0]["v3"][0]
    assert v3rec["ok"] is False and "timed out" in (v3rec["error"] or "")
    assert v3rec["latency_ms"] < 2 * 1000 * config.V3_MEDGEMMA_GROUP_TIMEOUT_S


def test_medgemma_group_deadline_bounds_whole_checkpoint_group(tmp_path, medgemma_url,
                                                              monkeypatch):
    """Delayed fake replies whose aggregate duration exceeds the client deadline: the op
    still returns its review code with unavailable notes (never a lost response), and
    checkpoints that cannot finish are skipped with an explicit not-sent record."""
    monkeypatch.setattr(config, "V3_MEDGEMMA_GROUP_TIMEOUT_S", 1.0)
    monkeypatch.setattr(config, "V3_MEDGEMMA_RESPONSE_RESERVE_S", 0.5)
    backend = DelayedBackend(delay=6.0)
    url = medgemma_url(backend)
    cs, lesion = make_case_state_v3(tmp_path, root="groupdead",
                                    v3={"medgemma": "q4", "medgemma_url": url,
                                        "checkpoints": ["review", "candidates"]})
    cs.commit_mask(_partial_lesion_mask(lesion))          # v1: candidates will be found
    out = segd.ops.run_op(cs, "finish", {"confidence": "medium", "reason": "r"})   # no raise
    assert not out.done and 'review_code "' in out.text   # the op itself stays intact
    assert out.text.count("unavailable") >= 2             # review + candidates notes
    assert len(backend.calls) == 1                        # only the review call was sent
    recs = [json.loads(l) for l in (cs.out_dir / "trajectory.jsonl").read_text().splitlines()]
    v3 = [r for r in recs if r["op"] == "finish" and r.get("error") is None][-1]["v3"]
    review = [e for e in v3 if e["checkpoint"] == "review"]
    cand = [e for e in v3 if e["checkpoint"] == "candidates"]
    assert len(review) == 1 and "timed out" in (review[0]["error"] or "")
    assert len(cand) == len(cs.candidate_cache[cs.current])   # one record per candidate
    assert all(e["ok"] is False and ("deadline" in (e["error"] or "")
                                     or "timed out" in (e["error"] or "")) for e in cand)


def test_op_deadline_covers_pre_hook_delay_and_aggregate_replies(tmp_path, medgemma_url,
                                                                monkeypatch):
    """Regression (review round-2 finding #1): the ABSOLUTE MedGemma deadline starts at
    run_op ENTRY. A delay in the op body BEFORE the checkpoint hook plus several
    individually successful delayed replies whose sum exceeds the remaining allowance
    must still return the whole op in time, with unavailable entries for the rows that
    no longer fit (previously the budget only started at the first call, so pre-hook
    work + per-call success could push the op past the extension's HTTP deadline)."""
    from segd import render
    monkeypatch.setattr(config, "V3_MEDGEMMA_GROUP_TIMEOUT_S", 5.0)
    monkeypatch.setattr(config, "V3_MEDGEMMA_RESPONSE_RESERVE_S", 1.0)
    backend = DelayedBackend(delay=2.0)
    url = medgemma_url(backend)
    cs, lesion = make_case_state_v3(tmp_path, root="opdead",
                                    v3={"medgemma": "q4", "medgemma_url": url,
                                        "checkpoints": ["review", "candidates"]})
    cs.commit_mask(_partial_lesion_mask(lesion))
    real_grid = render.render_review_grid

    def slow_grid(*a, **k):
        time.sleep(1.0)                       # pre-hook work before the FIRST call
        return real_grid(*a, **k)

    monkeypatch.setattr(render, "render_review_grid", slow_grid)
    t0 = time.time()
    out = segd.ops.run_op(cs, "finish", {"confidence": "medium", "reason": "r"})
    elapsed = time.time() - t0
    assert not out.done and 'review_code "' in out.text   # the op itself stays intact
    # the review call finished INSIDE its (post-pre-hook) allowance: its advice arrived
    assert "Medical second reader (MedGemma): complete no (confidence 0.60)" in out.text
    # the candidate rows did not fit anymore: unavailable entries, no delivered verdict
    med_lines = [l for l in out.text.splitlines() if l.startswith("Medical second reader")]
    assert len(med_lines) == 2
    assert "candidate verdicts" in med_lines[1]
    assert not re.search(r"C\d+ lesion", med_lines[1])
    assert f"{cs.candidate_cache[cs.current][0].id} unavailable" in med_lines[1]
    # the op returned well inside the pre-hook delay + per-op budget + reserve + slack
    assert elapsed < 1.0 + config.V3_MEDGEMMA_GROUP_TIMEOUT_S \
        + config.V3_MEDGEMMA_RESPONSE_RESERVE_S + 3.0
    assert len(backend.calls) == 2        # review + one candidate attempt, nothing more
    recs = [json.loads(l) for l in (cs.out_dir / "trajectory.jsonl").read_text().splitlines()]
    v3 = [r for r in recs if r["op"] == "finish" and r.get("error") is None][-1]["v3"]
    review = [e for e in v3 if e["checkpoint"] == "review"]
    assert len(review) == 1 and review[0]["ok"] is True
    cand = [e for e in v3 if e["checkpoint"] == "candidates"]
    assert len(cand) == len(cs.candidate_cache[cs.current])   # one record per row
    assert all(e["ok"] is False and e["parsed"] is None
               and ("deadline" in (e["error"] or "") or "timed out" in (e["error"] or ""))
               for e in cand)


# --------------------------------------------- candidate id binding (review P1 #2)
def test_candidates_checkpoint_binds_requested_candidate_id(tmp_path, medgemma_url):
    """A reply that names ANOTHER candidate (C2 for a C1 request) is unusable: no advice
    under either id, and the trajectory record stays bound to the requested id so
    offline adherence can never attribute it."""
    backend = ShiftIdBackend()
    url = medgemma_url(backend)
    cs, lesion = make_case_state_v3(tmp_path, root="idbind",
                                    v3={"medgemma": "q4", "medgemma_url": url,
                                        "checkpoints": ["candidates"]})
    cs.commit_mask(_partial_lesion_mask(lesion))
    out = segd.ops.run_op(cs, "finish", {"confidence": "medium", "reason": "r"})
    line = next(l for l in out.text.splitlines() if l.startswith("Medical second reader"))
    verdicts = line.split("candidate verdicts: ")[1].split(". ")[0].split(", ")
    assert verdicts == [f"{c.id} unavailable" for c in cs.candidate_cache[cs.current]]
    med_lines = [l for l in out.text.splitlines() if l.startswith("Medical second reader")]
    assert med_lines and not any("lesion" in l for l in med_lines)   # no advice delivered
    recs = [json.loads(l) for l in (cs.out_dir / "trajectory.jsonl").read_text().splitlines()]
    v3 = [r for r in recs if r["op"] == "finish" and r.get("error") is None][-1]["v3"]
    for e in v3:
        assert e["ok"] is False and e["parsed"] is None
        assert e["returned_candidate_id"] == f"C{int(e['candidate_id'][1:]) + 1}"
        assert "candidate id mismatch" in e["error"] and f"{e['candidate_id']}" in e["error"]


class HugeConfidenceBackend:
    """Valid JSON with the matching candidate id, but the confidence is the integer
    10**309: ``float(c)`` raises OverflowError during validation (review round-2
    finding: the parsed verdict used to stay attributable despite the unusable reply).
    """

    name = "hugeconf"
    model_id = "google/medgemma-1.5-4b-it"

    def load(self):
        pass

    def generate(self, prompt_text: str, images_b64, temperature: float, max_tokens: int) -> str:
        m = re.search(r"candidate (C\d+)", prompt_text)
        cid = m.group(1) if m else "C1"
        return (f'{{"candidate_id": "{cid}", "verdict": "lesion", '
                f'"confidence": {10**309}}}')


def test_huge_confidence_reply_unavailable_and_unattributed(tmp_path, medgemma_url):
    """Regression (review round-2): a reply whose VALIDATION raises (confidence 10**309
    -> OverflowError in float()) leaves the row unavailable, the trajectory record with
    parsed=None (the verdict is never exposed offline), and the op succeeding."""
    backend = HugeConfidenceBackend()
    url = medgemma_url(backend)
    cs, lesion = make_case_state_v3(tmp_path, root="hugeconf",
                                    v3={"medgemma": "q4", "medgemma_url": url,
                                        "checkpoints": ["candidates"]})
    cs.commit_mask(_partial_lesion_mask(lesion))
    out = segd.ops.run_op(cs, "finish", {"confidence": "medium", "reason": "r"})
    line = next(l for l in out.text.splitlines() if "candidate verdicts" in l)
    assert f"{cs.candidate_cache[cs.current][0].id} unavailable" in line
    assert "lesion" not in line.split("candidate verdicts: ")[1]     # no verdict exposed
    recs = [json.loads(l) for l in (cs.out_dir / "trajectory.jsonl").read_text().splitlines()]
    v3 = [r for r in recs if r["op"] == "finish" and r.get("error") is None][-1]["v3"]
    e = v3[0]
    assert e["ok"] is False and e["parsed"] is None and e["raw"] is not None
    assert "invalid confidence" in e["error"]


# ----------------------------- checkpoint-delivery state restore (review P1 #3)
def test_localize_checkpoint_retries_after_failed_persistence(tmp_path, medgemma_url,
                                                              monkeypatch):
    """save_render failure AFTER the localize call: the op fails, but the checkpoint is
    NOT consumed (delivery state restored) — the retry delivers the advice. The error
    record keeps the v3 diagnostics."""
    backend = ScriptedMedGemmaBackend()
    url = medgemma_url(backend)
    cs, _lesion = make_case_state_v3(tmp_path, root="persistfail",
                                     v3={"medgemma": "q4", "medgemma_url": url,
                                         "checkpoints": ["localize"]})
    segd.ops.run_op(cs, "open", {})
    real_save = cs.log.save_render
    attempts = {"n": 0}

    def flaky_save(image, op_name):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise OSError("disk full")
        return real_save(image, op_name)

    monkeypatch.setattr(cs.log, "save_render", flaky_save)
    with pytest.raises(OSError, match="disk full"):
        segd.ops.run_op(cs, "sheet", {"axis": "axial", "start": 10, "stop": 70})
    assert "localize" not in cs.v3_fired                  # delivery state restored
    recs = [json.loads(l) for l in (cs.out_dir / "trajectory.jsonl").read_text().splitlines()]
    err = [r for r in recs if r.get("error_kind") == "internal"][-1]
    assert err["op"] == "sheet"
    assert err["v3"][0]["checkpoint"] == "localize" and err["v3"][0]["ok"] is True

    out = segd.ops.run_op(cs, "sheet", {"axis": "axial", "start": 10, "stop": 70})   # retry
    assert "Medical second reader (MedGemma): the lesion is nearest mark" in out.text
    assert cs.v3_fired == {"localize"}
    assert len(backend.calls) == 2                        # fired again on the retry


# ------------------------------ per-checkpoint finish-hook guard (review P1 #4)
def _prompt_dir_with_only(tmp_path, keep, name="prompts_only", src_dir=None):
    pdir = tmp_path / name
    pdir.mkdir(exist_ok=True)
    for f in keep:                                        # full file names, e.g. review.md
        shutil.copy((src_dir or config.V3_PROMPT_DIR) / f, pdir / f)
    return pdir


def test_finish_hook_protects_each_checkpoint(tmp_path, medgemma_url, monkeypatch):
    """A failed checkpoint must not blank the other one's advice: each finish checkpoint
    is guarded separately (prompt loading included) and produces its own unavailable
    note + diagnostic; earlier successful notes are kept."""
    orig_prompt_dir = config.V3_PROMPT_DIR

    # (a) review.md missing: unavailable note for review, candidates still run
    monkeypatch.setattr(config, "V3_PROMPT_DIR", _prompt_dir_with_only(tmp_path, ["candidates.md"]))
    cs, lesion = make_case_state_v3(tmp_path, root="hookguard-a",
                                    v3={"medgemma": "q4",
                                        "medgemma_url": medgemma_url(ScriptedMedGemmaBackend(box=None)),
                                        "checkpoints": ["review", "candidates"]})
    cs.commit_mask(_partial_lesion_mask(lesion))
    out = segd.ops.run_op(cs, "finish", {"confidence": "medium", "reason": "r"})
    assert "Medical second reader (MedGemma): unavailable this checkpoint" in out.text
    assert re.search(r"candidate verdicts: C\d+ lesion \(0\.80\)", out.text)
    recs = [json.loads(l) for l in (cs.out_dir / "trajectory.jsonl").read_text().splitlines()]
    v3 = [r for r in recs if r["op"] == "finish" and r.get("error") is None][-1]["v3"]
    assert [e for e in v3 if e["checkpoint"] == "review" and e["ok"] is False]
    assert "FileNotFoundError" in [e for e in v3 if e["checkpoint"] == "review"
                                  and e["ok"] is False][0]["error"]

    # (b) candidates.md missing: the successful review note is KEPT, candidates show
    # their per-row unavailable entries
    monkeypatch.setattr(config, "V3_PROMPT_DIR",
                        _prompt_dir_with_only(tmp_path, ["review.md"], name="prompts_b",
                                              src_dir=orig_prompt_dir))
    cs, lesion = make_case_state_v3(tmp_path, root="hookguard-b",
                                    v3={"medgemma": "q4",
                                        "medgemma_url": medgemma_url(ScriptedMedGemmaBackend(box=None)),
                                        "checkpoints": ["review", "candidates"]})
    cs.commit_mask(_partial_lesion_mask(lesion))
    out = segd.ops.run_op(cs, "finish", {"confidence": "medium", "reason": "r"})
    assert "Medical second reader (MedGemma): complete no (confidence 0.60)" in out.text
    line = next(l for l in out.text.splitlines() if "candidate verdicts" in l)
    assert f"{cs.candidate_cache[cs.current][0].id} unavailable" in line
    assert "lesion (0.80)" not in out.text
    recs = [json.loads(l) for l in (cs.out_dir / "trajectory.jsonl").read_text().splitlines()]
    v3 = [r for r in recs if r["op"] == "finish" and r.get("error") is None][-1]["v3"]
    assert [e for e in v3 if e["checkpoint"] == "candidates" and e["ok"] is False
            and "FileNotFoundError" in e["error"]]
    assert [e for e in v3 if e["checkpoint"] == "review" and e["ok"] is True]


def test_candidate_row_crop_failure_keeps_earlier_and_later_rows(tmp_path, medgemma_url,
                                                                monkeypatch):
    """Regression (review round-2): the WHOLE per-candidate iteration is guarded, not
    only prompt loading. Deterministic fault injection: C1 succeeds, C2's row crop
    raises, C3 still runs. The note keeps C1's delivered verdict, marks C2 unavailable
    (with the diagnostic), and delivers C3 (previously the outer catch replaced every
    entry with one unavailable note and the later rows were skipped)."""
    from segd import render
    backend = ScriptedMedGemmaBackend()
    url = medgemma_url(backend)
    cs, lesion = make_case_state_v3(tmp_path, root="rowcrop",
                                    v3={"medgemma": "q4", "medgemma_url": url,
                                        "checkpoints": ["candidates"]})
    segd.ops.run_op(cs, "open", {})
    segd.ops.run_op(cs, "sheet", {"axis": "axial", "start": 10, "stop": 70})
    segd.ops.run_op(cs, "view", {"axis": "axial", "slice": lesion_center(lesion)[2],
                                 "marks": "fine"})
    # a fine mark far from the lesion (visible on the viewed slice): pointing there adds
    # a body-tissue blob whose intensity shell produces SEVERAL candidates
    vis = cs.markset.visible("axial", lesion_center(lesion)[2], None, "fine")
    centre = lesion_center(lesion)
    far = max(vis, key=lambda m: sum((a - b) ** 2 for a, b in zip(m.ijk, centre))).id
    segd.ops.run_op(cs, "point", {"ids": [far]})
    real_rc = render.render_candidates

    def exploding_second_row(image, mask, cands, spacing, **k):
        img, ri = real_rc(image, mask, cands, spacing, **k)
        orig_crop, n = img.crop, {"n": 0}

        def crop(box):
            n["n"] += 1
            if n["n"] == 2:                   # the SECOND row's crop
                raise RuntimeError("row 2 crop exploded")
            return orig_crop(box)

        img.crop = crop
        return img, ri

    monkeypatch.setattr(render, "render_candidates", exploding_second_row)
    out = segd.ops.run_op(cs, "finish", {"confidence": "medium", "reason": "r"})
    cands = cs.candidate_cache[cs.current]
    assert len(cands) >= 3                       # C1 ok, C2 crop failure, C3+ continue
    line = next(l for l in out.text.splitlines() if "candidate verdicts" in l)
    verdicts = line.split("candidate verdicts: ")[1].split(". ")[0].split(", ")
    assert verdicts[0] == f"{cands[0].id} lesion (0.80)"      # earlier advice KEPT
    assert verdicts[1] == f"{cands[1].id} unavailable"
    assert verdicts[2] == f"{cands[2].id} lesion (0.80)"      # later row still ran
    assert len(backend.calls) == 2                           # rows 1 and 3 sent, not row 2
    recs = [json.loads(l) for l in (cs.out_dir / "trajectory.jsonl").read_text().splitlines()]
    v3 = [r for r in recs if r["op"] == "finish" and r.get("error") is None][-1]["v3"]
    ok_rows = [e for e in v3 if e["ok"] is True]
    assert [e["candidate_id"] for e in ok_rows] == [cands[0].id, cands[2].id]
    bad = [e for e in v3 if e["candidate_id"] == cands[1].id]
    assert len(bad) == 1 and bad[0]["ok"] is False and bad[0]["parsed"] is None
    assert "row 2 crop exploded" in bad[0]["error"]


# ------------------------------------------------- pure mapping/geometry helpers
def test_snap_points_to_marks_tiebreak_and_dedup():
    from segd.v3 import snap_points_to_marks
    # two marks equidistant from the point -> lowest numeric id wins
    pts = snap_points_to_marks([{"x": 5.0, "y": 5.0, "score": 0.9}],
                               [("7", 3.0, 5.0), ("3", 7.0, 5.0)])
    assert pts[0]["mark"] == "3"
    # rank order preserved; dedup keeps the FIRST (best-ranked) point per mark
    pts = snap_points_to_marks([{"x": 0, "y": 0, "score": 0.9},
                                {"x": 1, "y": 0, "score": 0.8},
                                {"x": 50, "y": 50, "score": 0.7}],
                               [("9", 0.5, 0.0), ("2", 50.0, 50.0)])
    assert [p["mark"] for p in pts] == ["9", None, "2"]
    assert pts[0]["dist_px"] == 0.5 and pts[2]["dist_px"] == 0.0


def test_panel_voxel_math_and_nearest_marks():
    """panel_voxel round-trip on a synthetic geom (real inplane mapping), and the
    nearest-mark ranking by mm distance (duck-typed CaseState/markset)."""
    from segd.v3 import nearest_marks, panel_voxel

    class Geom:                                  # minimal ViewGeom-like object
        axis, slice_idx, scale = "axial", 3, 2.0
        crop = (0, 0, 4, 4)

    class RI:
        views = [Geom()]
        panel_rects = [(0, 0, 8, 8)]

    # displayed (r, c) = ((py-y0)/scale, (px-x0)/scale) = (2, 1) -> axial display flip
    ijk = panel_voxel(RI(), 3.0, 5.0, (8, 8, 4))
    assert ijk == (8 - 1 - 1, 8 - 1 - 2, 3)      # CONTRACTS §1
    # a pixel outside every rect snaps to the NEAREST panel and is clamped into it
    # (same snap philosophy as the mark snapping; deterministic, never a crash)
    assert panel_voxel(RI(), 100.0, 100.0, (8, 8, 4)) == (4, 4, 3)

    class FakeMark:
        def __init__(self, mid, ijk):
            self.id, self.ijk = mid, ijk

    class FakeMarkSet:
        marks = {1: FakeMark(1, (0, 0, 0)), 2: FakeMark(2, (10, 0, 0))}

    class FakeCS:
        markset = FakeMarkSet()
        spacing = (1.0, 1.0, 1.0)

    assert nearest_marks(FakeCS(), (0, 0, 0)) == [("1", 0.0), ("2", 10.0)]
    assert nearest_marks(FakeCS(), (20, 0, 0), n=1) == [("2", 10.0)]
