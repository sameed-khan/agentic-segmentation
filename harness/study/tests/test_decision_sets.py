"""Study B unit tests: decision-set label logic on a tiny synthetic run tree
(synthetic trajectories + versions.npz in the segd packbits format), decision
scorer math (strict JSON, sens/spec, AUROC, threshold-at-specificity), paired
McNemar, and replay resumability.

Run:  uv run --project harness/server python -m pytest harness/study/tests -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

STUDY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_DIR))

import build_decision_sets as bds  # noqa: E402
import replay_local as replay  # noqa: E402
import score_decision as D  # noqa: E402
from segd.case_state import pack_region  # noqa: E402

SHAPE = (10, 10, 4)


# ------------------------------------------------------------------ synthetic runs
def write_versions_npz(path: Path, versions: dict[int, np.ndarray | None]) -> None:
    """versions.npz in the segd.case_state pack_region format (ids/parents/bbox/packed)."""
    payload = {"shape": np.asarray(SHAPE),
               "ids": np.asarray(sorted(versions)),
               "parents": np.asarray([-1] + sorted(versions)[:-1])}
    for vid, mask in versions.items():
        bbox, packed, nvox = pack_region(np.zeros(SHAPE) if mask is None else mask.astype(np.uint8))
        payload[f"v{vid}_bbox"] = np.full((3, 2), -1) if bbox is None else np.asarray(bbox)
        payload[f"v{vid}_packed"] = np.frombuffer(packed, dtype=np.uint8)
        payload[f"v{vid}_nvox"] = nvox
    np.savez(path, **payload)


def gt_volume() -> np.ndarray:
    gt = np.zeros(SHAPE, dtype=bool)
    gt[2:8, 2:8, :] = True  # 288 voxels
    return gt


def blob(center=(5, 5), half=2, depth=4) -> np.ndarray:
    m = np.zeros(SHAPE, dtype=bool)
    m[center[0] - half:center[0] + half, center[1] - half:center[1] + half, :depth] = True
    return m


def make_case_dir(runs_root: Path, run: str, case: str, versions: dict[int, np.ndarray | None],
                  traj: list[dict]) -> Path:
    case_dir = runs_root / run / case
    (case_dir / "renders").mkdir(parents=True)
    write_versions_npz(case_dir / "versions.npz", versions)
    for render in [op for t in traj for op in (t.get("renders") or [])]:
        (case_dir / render).parent.mkdir(parents=True, exist_ok=True)
        (case_dir / render).write_bytes(b"png")
    with open(case_dir / "trajectory.jsonl", "w") as f:
        for t in traj:
            f.write(json.dumps(t) + "\n")
    return case_dir


@pytest.fixture()
def eval_score():
    return bds.load_eval_score()


@pytest.fixture()
def cache(monkeypatch):
    """CaseCache with synthetic image/GT (no real data touched)."""
    gt = gt_volume()
    data = bds.CaseData("C11", img=np.zeros(SHAPE, dtype=np.float32), gt=gt,
                        spacing=(1.0, 1.0, 1.0), body=None)

    class FakeCache:
        def get(self, case):
            return data

        def gt_only(self, case):
            return data.gt, SHAPE, data

    return FakeCache()


def meta_of(case: str) -> tuple[str, str]:
    return {"C11": ("F", "T1"), "C22": ("N", "T2")}.get(case, ("M", "T1"))


# ------------------------------------------------------------------ label logic
def test_done_label_not_done_on_version_gain(tmp_path, cache, eval_score):
    """First finish review on v1 (Dice 0.2); a later version reaches 0.6 -> not_done."""
    gt = gt_volume()
    make_case_dir(tmp_path, "phase1-06-high", "C11",
                  {1: blob(half=1, depth=1), 2: gt},  # dice(v1) small, v2 perfect
                  [{"op": "point", "parent": "v0", "child": "v1", "error": None, "args": {}},
                   {"op": "finish", "parent": "v1", "renders": ["renders/010_finish.png"],
                    "error": None, "candidates": None, "observation": "Final review of v1"}])
    done, wrong = bds.build_done_and_wrong(tmp_path, tmp_path, cache, eval_score,
                                           runs=("phase1-06-high",), meta_of=meta_of)
    assert len(done) == 1 and done[0]["label"] == "not_done"
    assert done[0]["dice_state"] < 0.3 and done[0]["dice_best"] > 0.9
    assert done[0]["images"] == ["phase1-06-high/C11/renders/010_finish.png"]


def test_done_label_done_when_state_is_best(tmp_path, cache, eval_score):
    gt = gt_volume()
    make_case_dir(tmp_path, "phase1-06-high", "C11",
                  {1: gt},  # the reviewed state IS the best version
                  [{"op": "point", "parent": "v0", "child": "v1", "error": None, "args": {}},
                   {"op": "finish", "parent": "v1", "renders": ["renders/010_finish.png"],
                    "error": None, "candidates": None, "observation": "Final review of v1"}])
    done, _ = bds.build_done_and_wrong(tmp_path, tmp_path, cache, eval_score,
                                       runs=("phase1-06-high",), meta_of=meta_of)
    assert done[0]["label"] == "done" and done[0]["version_dice_gain"] == 0.0


def test_done_label_not_done_on_candidate_oracle(tmp_path, cache, eval_score, monkeypatch):
    """Oracle-accepting a GT-positive candidate improves Dice by >= 0.05 -> not_done."""
    gt = gt_volume()
    mask = blob(center=(4, 4), half=2, depth=4)  # reviewed state: 64 vox, all inside GT
    make_case_dir(tmp_path, "phase1-10-high-cand", "C11",
                  {1: mask},
                  [{"op": "point", "parent": "v0", "child": "v1", "error": None, "args": {}},
                   {"op": "finish", "parent": "v1", "renders": ["renders/010_finish.png"],
                    "error": None, "observation": "review",
                    "candidates": {"version": "v1", "method": "intensity",
                                   "items": [{"id": "C1", "volume_ml": 12.0}]}}])
    # candidate add-region DISJOINT from the mask (the E16 invariant), inside GT
    good = np.zeros(SHAPE, dtype=bool)
    good[6:8, 2:8, :] = True  # 48 voxels, precision 1.0 -> lesion
    monkeypatch.setattr("segd.candidates.propose",
                        lambda img, m, spacing, method, max_n, cache: [
                            SimpleNamespace(id="C1", region=good, volume_ml=12.0)])
    done, _ = bds.build_done_and_wrong(tmp_path, tmp_path, cache, eval_score,
                                       runs=("phase1-10-high-cand",), meta_of=meta_of)
    assert done[0]["label"] == "not_done"
    assert done[0]["oracle_dice_gain"] is not None and done[0]["oracle_dice_gain"] >= 0.05
    # sanity: gt = 6*6*4 = 144 vox; dice_state = 2*64/(64+144); dice_after = 2*(64+48)/(64+48+144)
    assert done[0]["dice_state"] == pytest.approx(2 * 64 / (64 + 144), abs=1e-3)
    assert done[0]["oracle_dice_gain"] == pytest.approx(2 * 112 / 256 - 2 * 64 / 208, abs=1e-3)


def test_done_candidate_ids_must_match(tmp_path, cache, eval_score, monkeypatch):
    gt = gt_volume()
    make_case_dir(tmp_path, "phase1-10-high-cand", "C11",
                  {1: gt},
                  [{"op": "point", "parent": "v0", "child": "v1", "error": None, "args": {}},
                   {"op": "finish", "parent": "v1", "renders": ["renders/010_finish.png"],
                    "error": None, "observation": "review",
                    "candidates": {"version": "v1", "method": "intensity",
                                   "items": [{"id": "C9", "volume_ml": 1.0}]}}])
    monkeypatch.setattr("segd.candidates.propose",
                        lambda img, mask, spacing, method, max_n, cache: [])
    with pytest.raises(AssertionError):
        bds.build_done_and_wrong(tmp_path, tmp_path, cache, eval_score,
                                 runs=("phase1-10-high-cand",), meta_of=meta_of)


def test_wrong_structure_label(tmp_path, cache, eval_score):
    """State after the first model prompt: label = Dice < 0.1."""
    gt = gt_volume()
    tiny = blob(center=(2, 2), half=1, depth=1)   # dice ~ 0.02
    decent = blob(center=(5, 5), half=2, depth=4)  # dice 4/9 vs GT [2:8, 2:8, :] -> ~0.66
    make_case_dir(tmp_path, "phase1-06-high", "C11", {1: tiny, 2: decent},
                  [{"op": "view", "parent": "v0", "child": None, "renders": ["renders/004_view.png"],
                    "error": None, "args": {}},
                   {"op": "point", "parent": "v0", "child": "v1", "error": None,
                    "renders": ["renders/007_point.png"], "args": {}, "observation": "version v1"}])
    make_case_dir(tmp_path, "phase1-06-high", "C22", {1: decent},
                  [{"op": "point", "parent": "v0", "child": "v1", "error": None,
                    "renders": ["renders/007_point.png"], "args": {}, "observation": "version v1"}])
    _done, wrong = bds.build_done_and_wrong(tmp_path, tmp_path, cache, eval_score,
                                            runs=("phase1-06-high",), meta_of=meta_of)
    by_case = {r["case"]: r for r in wrong}
    assert by_case["C11"]["label"] == "wrong_structure" and by_case["C11"]["dice_state"] < 0.1
    assert by_case["C22"]["label"] == "structures_ok" and by_case["C22"]["dice_state"] >= 0.1
    # the first point op is the state (view/renders before it do not create versions)
    assert by_case["C11"]["version"] == "v1"
    assert by_case["C11"]["images"] == ["phase1-06-high/C11/renders/007_point.png"]


def test_failed_first_point_is_not_the_state(tmp_path, cache, eval_score):
    """An errored first point op is skipped; the first SUCCESSFUL one is the state."""
    gt = gt_volume()
    make_case_dir(tmp_path, "phase1-06-high", "C11", {1: gt},
                  [{"op": "point", "parent": "v0", "child": None, "error": "mark 1485 is not shown",
                    "args": {}},
                   {"op": "point", "parent": "v0", "child": "v1", "error": None, "args": {},
                    "renders": ["renders/007_point.png"], "observation": "version v1"}])
    _done, wrong = bds.build_done_and_wrong(tmp_path, tmp_path, cache, eval_score,
                                            runs=("phase1-06-high",), meta_of=meta_of)
    assert wrong[0]["version"] == "v1"


# ------------------------------------------------------------------ candidate set build
def test_candidate_judgement_expansion(tmp_path, monkeypatch):
    e16 = tmp_path / "candidate_replay" / "manifest.jsonl"
    e16.parent.mkdir(parents=True)
    e16.write_text(json.dumps({
        "key": "F14_v1", "case": "F14", "histotype": "F", "sequence": "T2", "version": "v1",
        "dice_reviewed": 0.7, "images": ["results/phase1-10-high-cand/F14/renders/015_finish.png"],
        "review_text": "Final review of v1 ...", "source_run": "results/phase1-10-high-cand",
        "candidates": [
            {"id": "C1", "volume_ml": 1.0, "precision": 0.8, "label": "lesion",
             "region_vox": 10, "inter_vox": 8, "overlap_mask_vox": 0},
            {"id": "C2", "volume_ml": 2.0, "precision": 0.1, "label": "normal",
             "region_vox": 20, "inter_vox": 2, "overlap_mask_vox": 0},
        ]}) + "\n")
    rows = bds.build_candidate_judgement(tmp_path, tmp_path)
    assert [r["key"] for r in rows] == ["F14_v1:C1", "F14_v1:C2"]
    assert rows[0]["label"] == "lesion" and rows[1]["label"] == "normal"
    assert rows[0]["candidate"]["id"] == "C1"
    assert rows[0]["images_abs"][0].endswith("015_finish.png")


# ------------------------------------------------------------------ decision scorer
def _row(set_name, key, label, histotype="F"):
    return {"key": key, "set": set_name, "case": key.split(":")[0], "histotype": histotype,
            "sequence": "T1", "label": label}


def test_parse_answer_strict():
    spec = D.SET_SPEC["candidate_judgement"]["answers"]
    p = D.parse_answer('{"answer": "lesion", "confidence": 0.8}', spec)
    assert p == {"status": "ok", "answer": "lesion", "confidence": 0.8}
    p = D.parse_answer('The tissue is lesion.\n{"answer": "not_lesion"}', spec)
    assert p["status"] == "ok" and p["answer"] == "not_lesion"
    p = D.parse_answer('{"answer": "uncertain"}', spec)
    assert p["status"] == "uncertain"
    for bad in ('{"answer": "cancer"}', "it is a lesion", '{"confidence": 0.9}', ""):
        assert D.parse_answer(bad, spec)["status"] == "parse_fail"


def test_score_call_candidate_confusion():
    pos = _row("candidate_judgement", "A:v1:C1", "lesion")
    neg = _row("candidate_judgement", "B:v1:C1", "normal")

    def call(row, raw):
        return D.score_call({"run": "r", "key": row["key"], "case": row["case"], "rep": 0,
                             "model": "medgemma_q4", "raw_text": raw, "error": None,
                             "latency_ms": 5.0, "usage": None}, row)

    calls = [call(pos, '{"answer": "lesion", "confidence": 0.9}'),      # tp
             call(pos, '{"answer": "not_lesion", "confidence": 0.7}'),  # fn
             call(neg, '{"answer": "not_lesion", "confidence": 0.8}'),  # tn
             call(neg, '{"answer": "lesion", "confidence": 0.6}'),      # fp
             call(pos, '{"answer": "uncertain"}'),                      # abstain
             call(pos, "prose only")]                                   # parse fail
    m = D.confusion(calls)
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (1, 1, 1, 1)
    assert m["sensitivity"] == 0.5 and m["specificity"] == 0.5
    assert m["n_uncertain"] == 1 and m["n_parse_fail"] == 1
    assert calls[4]["counted"] is False and calls[5]["status"] == "parse_fail"


def test_auroc_perfect_and_with_ties():
    assert D.auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert D.auroc([0.1, 0.2], [0.9, 0.8]) == 0.0
    assert D.auroc([0.9, 0.5], [0.1, 0.5]) == 0.875  # tie-corrected Mann-Whitney
    assert D.auroc([], [0.1]) is None


def test_threshold_at_specificity():
    neg = [0.1, 0.2, 0.3, 0.4]
    thr = D.threshold_at_specificity(neg, 0.95)
    assert thr is not None and thr > 0.4
    thr5 = D.threshold_at_specificity(neg, 0.5)
    assert 0.2 < thr5 <= 0.2 + 1e-6  # smallest threshold with spec >= 0.5


def test_metrics_for_sens_at_spec():
    rows = [_row("candidate_judgement", f"K{i}:v1:C1", "lesion" if i < 3 else "normal")
            for i in range(6)]
    calls = []
    for i, row in enumerate(rows):
        answer = "lesion" if i < 2 else "not_lesion"   # 2/3 lesions found
        conf = 0.9 - 0.1 * i
        p = 1.0 - conf if answer == "not_lesion" else conf
        calls.append({"run": "r", "key": row["key"], "case": row["case"], "rep": 0,
                      "model": "m", "set": "candidate_judgement", "histotype": row["histotype"],
                      "label": row["label"], "is_positive": int(row["label"] == "lesion"),
                      "latency_ms": 1.0, "cost_usd": None, "response_error": False,
                      "status": "ok", "answer": answer, "confidence": conf,
                      "p_positive": p, "counted": True,
                      "tp": int(row["label"] == "lesion" and answer == "lesion"),
                      "fp": int(row["label"] == "normal" and answer == "lesion"),
                      "tn": int(row["label"] == "normal" and answer == "not_lesion"),
                      "fn": int(row["label"] == "lesion" and answer != "lesion")})
    m = D.metrics_for(calls, n_boot=50, seed=1)
    assert m["sensitivity"] == pytest.approx(2 / 3, abs=1e-3)
    assert m["specificity"] == 1.0
    # AUROC: positives p = [0.9, 0.8, 0.3] (third answers not_lesion, conf 0.7),
    # negatives p = [0.4, 0.5, 0.6] -> only 6 of 9 pairs ordered correctly.
    assert m["auroc"] == pytest.approx(6 / 9, abs=1e-3)
    assert m["sens_ci95"][0] <= 2 / 3 <= m["sens_ci95"][1] or m["sens_ci95"] == (1.0, 1.0)


def test_done_set_positive_class_is_not_done():
    assert D.SET_SPEC["done_check"]["positive"] == "not_done"
    row = _row("done_check", "A:v1:done", "not_done")
    call = D.score_call({"run": "r", "key": row["key"], "case": "A", "rep": 0, "model": "m",
                         "raw_text": '{"answer": "done", "confidence": 0.9}', "error": None,
                         "latency_ms": 1.0, "usage": None}, row)
    assert call["fn"] == 1 and call["p_positive"] == pytest.approx(0.1)


def test_paired_compare_mcnamar():
    keys = [f"K{i}:v1:C1" for i in range(4)]
    rows = [_row("done_check", k, "not_done" if i % 2 else "done") for i, k in enumerate(keys)]
    def run_calls(decisions):
        calls = []
        for row, dec in zip(rows, decisions):
            base = {"run": "r", "key": row["key"], "case": row["case"], "rep": 0, "model": "m",
                    "set": "done_check", "histotype": row["histotype"], "label": row["label"],
                    "is_positive": int(row["label"] == "not_done"), "latency_ms": 1.0,
                    "cost_usd": None, "response_error": False, "status": "ok",
                    "confidence": 0.9, "counted": True}
            answer = dec
            p = 0.9 if answer == "not_done" else 0.1
            calls.append(dict(base, answer=answer, p_positive=p,
                              tp=int(row["label"] == "not_done" and answer == "not_done"),
                              fp=int(row["label"] == "done" and answer == "not_done"),
                              tn=int(row["label"] == "done" and answer == "done"),
                              fn=int(row["label"] == "not_done" and answer != "not_done")))
        return calls
    a = run_calls(["not_done", "not_done", "done", "done"])
    b = run_calls(["not_done", "done", "not_done", "not_done"])
    comp = D.paired_compare(a, b, "not_done")
    assert comp["n_paired_keys"] == 4
    assert comp["n_b_hit_a_miss"] == 1 and comp["n_a_hit_b_miss"] == 1
    assert comp["mcnemar_p"] == 1.0


# ------------------------------------------------------------------ replay resumability
def test_replay_resume_keeps_ok_drops_errors(tmp_path):
    resp = tmp_path / "responses.jsonl"
    ok_line = json.dumps({"run": "r", "key": "A:v1:C1", "model": "medgemma_q4", "rep": 0, "ok": True})
    err_line = json.dumps({"run": "r", "key": "B:v1:C1", "model": "medgemma_q4", "rep": 0,
                           "ok": False, "error": "timeout"})
    resp.write_text(ok_line + "\n" + err_line + "\n")
    done = replay.load_done_and_prune(resp)
    assert done == {("A:v1:C1", "medgemma_q4", 0)}
    kept = [json.loads(l) for l in resp.read_text().splitlines()]
    assert len(kept) == 1 and kept[0]["key"] == "A:v1:C1"


def test_replay_template_rendering(tmp_path):
    from medgemma_client import extract_json_object  # noqa: F401  (client import path)

    template = (Path(STUDY_DIR / "prompts" / "decision_wrong_structure.md")).read_text()
    text = replay.render_template(template, {"text": "version v1 | volume 1 mL",
                                             "case": "C1", "target": "fibrous soft-tissue tumor",
                                             "sequence_text": "T1-weighted", "candidate_id": ""})
    assert "version v1 | volume 1 mL" in text
    assert "fibrous soft-tissue tumor" in text
    assert "{{" not in text
