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
import local_point_study as runner  # noqa: E402
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


def test_done_oracle_accepts_only_gt_positive_candidates(tmp_path, cache, eval_score, monkeypatch):
    """The oracle accepts EXACTLY the GT-positive (precision > 0.5) candidates: a large
    normal candidate must not erase a positive one's gain (review P1-2)."""
    gt = gt_volume()
    mask = blob(center=(4, 4), half=2, depth=4)          # reviewed state: 64 vox, in GT
    lesion = np.zeros(SHAPE, dtype=bool)
    lesion[6:7, 2:8, :] = True                           # 24 vox, precision 1.0
    normal = np.zeros(SHAPE, dtype=bool)
    normal[0:2, 0:10, :] = True                          # 80 vox, precision 0.0
    make_case_dir(tmp_path, "phase1-10-high-cand", "C11",
                  {1: mask},
                  [{"op": "point", "parent": "v0", "child": "v1", "error": None, "args": {}},
                   {"op": "finish", "parent": "v1", "renders": ["renders/010_finish.png"],
                    "error": None, "observation": "review",
                    "candidates": {"version": "v1", "method": "intensity",
                                   "items": [{"id": "C1", "volume_ml": 2.4},
                                             {"id": "C2", "volume_ml": 8.0}]}}])
    monkeypatch.setattr("segd.candidates.propose",
                        lambda img, m, spacing, method, max_n, cache: [
                            SimpleNamespace(id="C1", region=lesion, volume_ml=2.4),
                            SimpleNamespace(id="C2", region=normal, volume_ml=8.0)])
    done, _ = bds.build_done_and_wrong(tmp_path, tmp_path, cache, eval_score,
                                       runs=("phase1-10-high-cand",), meta_of=meta_of)
    # accepting only the lesion: dice 2*(64+24)/(64+24+144) = 0.7586 (+0.143) -> not_done
    assert done[0]["label"] == "not_done"
    assert done[0]["oracle_dice_gain"] == pytest.approx(2 * 88 / 232 - 2 * 64 / 208, abs=1e-3)
    # counterfactual (the OLD bug): accepting BOTH candidates erases the gain —
    # dice 2*(64+24)/(64+24+80+144) = 0.564 < dice_state 0.615 -> improvement 0 -> done
    assert 2 * 88 / 312 < 2 * 64 / 208


def test_done_oracle_volume_drift_detected(tmp_path, cache, eval_score, monkeypatch):
    """Matching checks logged VOLUMES, not just the always-C1..Cn ids (review P2-2)."""
    gt = gt_volume()
    mask = blob(center=(4, 4), half=2, depth=4)
    region = np.zeros(SHAPE, dtype=bool)
    region[6:7, 2:8, :] = True
    make_case_dir(tmp_path, "phase1-10-high-cand", "C11",
                  {1: mask},
                  [{"op": "point", "parent": "v0", "child": "v1", "error": None, "args": {}},
                   {"op": "finish", "parent": "v1", "renders": ["renders/010_finish.png"],
                    "error": None, "observation": "review",
                    "candidates": {"items": [{"id": "C1", "volume_ml": 99.0}]}}])
    monkeypatch.setattr("segd.candidates.propose",
                        lambda img, m, spacing, method, max_n, cache: [
                            SimpleNamespace(id="C1", region=region, volume_ml=2.4)])
    with pytest.raises(AssertionError, match="volume drifted"):
        bds.build_done_and_wrong(tmp_path, tmp_path, cache, eval_score,
                                 runs=("phase1-10-high-cand",), meta_of=meta_of)


def test_wrong_structure_empty_version_labelled_wrong_structure(tmp_path, cache, eval_score):
    """An empty first version records dice_state 0.0 and is labelled by the SAME Dice < 0.1
    rule (never structures_ok with a 0.0 Dice — review P1-9)."""
    make_case_dir(tmp_path, "phase1-06-high", "C11", {1: None},   # v1 = empty mask
                  [{"op": "point", "parent": "v0", "child": "v1", "error": None,
                    "renders": ["renders/007_point.png"], "args": {}, "observation": "version v1"}])
    _done, wrong = bds.build_done_and_wrong(tmp_path, tmp_path, cache, eval_score,
                                            runs=("phase1-06-high",), meta_of=meta_of)
    assert wrong[0]["dice_state"] == 0.0
    assert wrong[0]["label"] == "wrong_structure"


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
    # parse failures are terminal WRONG answers: the malformed positive adds a fn
    # (review P1-4: failures stay in the denominators with explicit counts)
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (1, 2, 1, 1)
    assert m["sensitivity"] == pytest.approx(1 / 3, abs=1e-3)
    assert m["specificity"] == 0.5
    assert m["n_uncertain"] == 1 and m["n_parse_fail"] == 1
    assert calls[4]["counted"] is False and calls[5]["counted"] is True
    assert calls[5]["fn"] == 1


def test_parse_failures_cannot_inflate_sensitivity():
    """One correct positive + nine malformed positive replies -> sens 0.1, never 1.0."""
    pos = _row("candidate_judgement", "A:v1:C1", "lesion")
    calls = [D.score_call({"run": "r", "key": pos["key"], "case": "A", "rep": 0,
                           "model": "m", "raw_text": '{"answer": "lesion"}',
                           "error": None, "latency_ms": 1.0, "usage": None}, pos)]
    calls += [D.score_call({"run": "r", "key": pos["key"], "case": "A", "rep": i + 1,
                            "model": "m", "raw_text": f"malformed reply {i}",
                            "error": None, "latency_ms": 1.0, "usage": None}, pos)
              for i in range(9)]
    m = D.confusion(calls)
    assert m["sensitivity"] == 0.1
    assert m["n_parse_fail"] == 9 and m["n_counted"] == 10 and m["fn"] == 9


def test_response_error_counts_as_error():
    pos = _row("done_check", "A:v1:done", "not_done")
    neg = _row("done_check", "B:v1:done", "done")
    err_p = D.score_call({"run": "r", "key": pos["key"], "case": "A", "rep": 0, "model": "m",
                          "raw_text": "", "error": "timeout", "latency_ms": None,
                          "usage": None}, pos)
    err_n = D.score_call({"run": "r", "key": neg["key"], "case": "B", "rep": 0, "model": "m",
                          "raw_text": "", "error": "timeout", "latency_ms": None,
                          "usage": None}, neg)
    assert err_p["status"] == "response_error" and err_p["fn"] == 1 and err_p["counted"] is True
    assert err_n["fp"] == 1


def test_uncertain_as_negative_secondary_metrics():
    """The uncertain-as-negative policy only ADDITIONALLY maps the abstentions: every
    primary counted row keeps its assignment, so parse/transport/missing failures stay
    FN/FP in the secondary denominators, and with zero uncertain answers the secondary
    metrics equal the primary ones (review fix round: the old filter dropped failures
    and could report secondary sens 1.0 while the primary said 0.1)."""
    pos = _row("candidate_judgement", "A:v1:C1", "lesion")
    neg = _row("candidate_judgement", "B:v1:C1", "normal")

    def call(row, raw=None, error=None, rep=0):
        return D.score_call({"run": "r", "key": row["key"], "case": row["case"], "rep": rep,
                             "model": "m", "raw_text": raw or "", "error": error,
                             "latency_ms": None if error is not None else 1.0,
                             "usage": None}, row)

    def missing(row, rep):
        c = call(row, error="missing response: planned call never recorded", rep=rep)
        c["missing"] = True
        return c

    calls = [call(pos, '{"answer": "uncertain"}'),                       # abstain on positive
             call(neg, '{"answer": "uncertain"}'),                       # abstain on negative
             call(pos, '{"answer": "lesion", "confidence": 0.9}'),       # tp
             call(neg, '{"answer": "not_lesion", "confidence": 0.9}'),   # tn
             call(pos, "prose only"),                                    # parse fail -> fn
             call(pos, error="timeout"),                                 # transport -> fn
             missing(pos, 6),                                            # missing -> fn
             missing(neg, 7)]                                            # missing -> fp
    m = D.metrics_for(calls, n_boot=20, seed=1)
    assert m["n_uncertain"] == 2
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (1, 3, 1, 1)
    # primary omits the abstains: sens = 1/(1+3) = 0.25, spec = 1/(1+1) = 0.5
    assert m["sensitivity"] == 0.25 and m["specificity"] == 0.5
    # secondary additionally maps uncertain positive -> FN, uncertain negative -> TN:
    # sens = 1/5, spec = 2/3 — the failures stay wrong in the denominators
    assert m["sensitivity_uncertain_as_negative"] == pytest.approx(1 / 5, abs=1e-3)
    assert m["specificity_uncertain_as_negative"] == pytest.approx(2 / 3, abs=1e-3)

    # zero uncertain answers, failures included: the secondary policy must change NOTHING
    calls_zu = [call(pos, '{"answer": "lesion", "confidence": 0.9}'),    # tp
                call(pos, "prose only"),                                 # parse fail -> fn
                call(neg, '{"answer": "not_lesion", "confidence": 0.9}'),  # tn
                missing(neg, 3)]                                         # missing -> fp
    mz = D.metrics_for(calls_zu, n_boot=20, seed=1)
    assert mz["n_uncertain"] == 0
    assert mz["sensitivity"] == mz["sensitivity_uncertain_as_negative"] == 0.5
    assert mz["specificity"] == mz["specificity_uncertain_as_negative"] == 0.5


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


def test_stored_threshold_is_the_exact_operational_value():
    """The persisted threshold must reproduce the metric (review P2-3): a 4-decimal
    round of 0.4+1e-9 applied with >= would reach only 0.75 specificity."""
    rows = [_row("candidate_judgement", f"K{i}:v1:C1", "lesion" if i < 2 else "normal")
            for i in range(6)]
    calls = []
    for i, row in enumerate(rows):
        p = (0.9, 0.8, 0.1, 0.2, 0.3, 0.4)[i]
        pos = int(row["label"] == "lesion")
        calls.append({"run": "r", "key": row["key"], "case": row["case"], "rep": 0,
                      "model": "m", "set": "candidate_judgement", "histotype": "F",
                      "label": row["label"], "is_positive": pos, "latency_ms": 1.0,
                      "cost_usd": None, "response_error": False, "status": "ok",
                      "answer": "lesion" if pos else "not_lesion", "confidence": p if pos else 1 - p,
                      "p_positive": p, "counted": True,
                      "tp": int(pos and p >= 0), "fp": int(not pos and False),
                      "tn": int(not pos), "fn": 0, "missing": False})
    m = D.metrics_for(calls, n_boot=10, seed=1)
    thr = m["threshold_at_spec_0.95"]
    assert thr == pytest.approx(0.4 + 1e-9, abs=1e-12)   # exact, not rounded to 0.4
    assert m["threshold_convention"].startswith("positive iff p_positive >=")
    assert m["spec_at_threshold"] == 1.0                 # >= convention on the exact value
    # applying the ROUNDED 0.4 with >= would misclassify the 0.4 negative -> spec 0.75
    assert sum(1 for c in calls if not c["is_positive"] and c["p_positive"] >= 0.4) == 1


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


def test_cluster_bootstrap_resamples_cases_not_decisions():
    """CI resampling carries every decision of a case together (review P1-5): with two
    internally consistent cases the statistic can only be 0.0 or 1.0, never an
    in-between row-mixture."""
    def tp_rows(case, n, value):
        return [{"run": "r", "key": f"{case}:v1:C{i}", "case": case, "rep": 0, "model": "m",
                 "set": "candidate_judgement", "histotype": "F", "label": "lesion",
                 "is_positive": 1, "status": "ok", "counted": True, "tp": value,
                 "missing": False} for i in range(n)]

    rows = tp_rows("A", 10, 1) + tp_rows("B", 10, 0)
    lo, hi = D.cluster_bootstrap_ci(rows, lambda c: c["tp"], n_boot=300, seed=3)
    assert (lo, hi) == (0.0, 1.0)  # a flat row bootstrap gives an in-between CI here
    # single case: cluster bootstrap degrades to that case's constant statistic
    lo1, hi1 = D.cluster_bootstrap_ci(tp_rows("A", 5, 1), lambda c: c["tp"], n_boot=50, seed=3)
    assert (lo1, hi1) == (1.0, 1.0)
    assert D.cluster_bootstrap_ci([], lambda c: 0, 10, 1) == (None, None)


def test_paired_compare_uses_correctness_both_classes():
    """McNemar pairs per-decision correctness (tp+tn): arms differing ONLY on negative
    rows must produce discordances (review P1-6; the old tp-only rule saw none)."""
    def calls(answers):
        out = []
        for i, (label, ans) in enumerate(zip(["normal", "normal", "lesion", "lesion"], answers)):
            row = _row("candidate_judgement", f"K{i}:v1:C1", label)
            out.append({"run": "r", "key": row["key"], "case": row["case"], "rep": 0,
                        "model": "m", "set": "candidate_judgement", "histotype": "F",
                        "label": label, "is_positive": int(label == "lesion"),
                        "latency_ms": 1.0, "cost_usd": None, "response_error": False,
                        "status": "ok", "answer": ans, "confidence": 0.9,
                        "p_positive": 0.9 if ans == "lesion" else 0.1,
                        "counted": True,
                        "tp": int(label == "lesion" and ans == "lesion"),
                        "fp": int(label == "normal" and ans == "lesion"),
                        "tn": int(label == "normal" and ans == "not_lesion"),
                        "fn": int(label == "lesion" and ans != "lesion"),
                        "missing": False})
        return out

    # both arms agree on positives; differ ONLY on the two negative rows
    a = calls(["not_lesion", "not_lesion", "lesion", "lesion"])   # all correct
    b = calls(["lesion", "not_lesion", "lesion", "not_lesion"])    # fp on K0, fn on K3
    comp = D.paired_compare(a, b, "lesion")
    assert comp is not None
    assert comp["n_paired_keys"] == 4
    assert comp["n_b_hit_a_miss"] == 0 and comp["n_a_hit_b_miss"] == 2
    assert comp["outcome"].startswith("per-decision correctness")
    # sanity: the old tp-only rule aggregated b/a tp flags (0/0) and would have seen NO
    # discordance at all on these two negative-row differences.


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
    # per-decision correctness (tp+tn): a is wrong on K0 (fp) and K3 (fn); b is wrong on
    # K1/K2 — a discordance on a NEGATIVE row (K2: a tn, b fp) now counts (review P1-6).
    assert comp["n_b_hit_a_miss"] == 1 and comp["n_a_hit_b_miss"] == 2
    assert comp["mcnemar_p"] == pytest.approx(1.0)  # exact binomial n=3, min(1,2)=1 -> 2*4/8


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


def _replay_row(tmp_path, images_file="a.png"):
    png = tmp_path / images_file
    png.write_bytes(b"png")
    return {"key": "A:v1:done", "set": "done_check", "case": "A", "histotype": "F",
            "sequence": "T1", "source_run": "r", "version": "v1", "text": "review v1",
            "images": [images_file], "images_abs": [str(png)],
            "candidate": {"id": "C1", "volume_ml": 1.0, "overlap_mask_vox": 0}}


def test_replay_parse_failure_is_final_outcome_not_retried(tmp_path, monkeypatch):
    """A malformed model reply is a COMPLETED outcome (ok=True, parse_error set): the row
    is retained verbatim on resume and never retried (review P1-3)."""
    class FakeResult:
        text = "oops not json"
        parsed = None
        parse_error = "no JSON object found"
        uncertain = None
        usage = None
        latency_ms = 3.0
        model = "medgemma-1.5-4b-it"
        raw_body = "oops not json"

    class FakeClient:
        def __init__(self, base_url=None, timeout_s=None):
            pass

        def chat_json(self, *args, **kwargs):
            return FakeResult()

    monkeypatch.setattr(replay, "MedGemmaClient", FakeClient)
    row = _replay_row(tmp_path)
    opts = replay.parse_args(["--set", "done_check", "--run", "x"])
    line = replay.build_call(row, 0, opts, Path("decision_done.md"), "template {{case}}")
    assert line["ok"] is True and line["error"] is None
    assert line["parse_error"] == "no JSON object found" and line["parsed"] is None
    resp = tmp_path / "responses.jsonl"
    resp.write_text(json.dumps(line) + "\n")
    done = replay.load_done_and_prune(resp)
    assert done == {("A:v1:done", opts.model, 0)}
    kept = [json.loads(l) for l in resp.read_text().splitlines()]
    assert len(kept) == 1 and kept[0]["parse_error"] == "no JSON object found"


def test_replay_transport_error_is_retried_on_resume(tmp_path, monkeypatch):
    """Only transport/HTTP failures are ok=False and get dropped for retry on resume."""
    class FailingClient:
        def __init__(self, base_url=None, timeout_s=None):
            pass

        def chat_json(self, *args, **kwargs):
            raise RuntimeError("connection refused")

    monkeypatch.setattr(replay, "MedGemmaClient", FailingClient)
    row = _replay_row(tmp_path)
    opts = replay.parse_args(["--set", "done_check", "--run", "x"])
    line = replay.build_call(row, 0, opts, Path("decision_done.md"), "t")
    assert line["ok"] is False and "connection refused" in line["error"]
    resp = tmp_path / "responses.jsonl"
    resp.write_text(json.dumps(line) + "\n")
    assert replay.load_done_and_prune(resp) == set()      # dropped -> retried next run
    assert resp.read_text().strip() == ""


def test_replay_refuses_config_drift_on_resume(tmp_path):
    """Resuming with a drifted prompt/decoding config is refused (review P2-4)."""
    run_dir = tmp_path / "done_check" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "responses.jsonl").write_text(
        json.dumps({"key": "k", "model": "medgemma_q4", "rep": 0, "ok": True}) + "\n")
    prior = {"set": "done_check", "model": "medgemma_q4", "reps": 1, "max_tokens": 256,
             "manifest": "m", "images_root": None, "prompt_template": "p.md", "prompt_sha256": "x"}
    (run_dir / "meta.json").write_text(json.dumps(prior))
    drifted = dict(prior, max_tokens=999)
    with pytest.raises(SystemExit, match="drifted"):
        replay.check_immutable_config(run_dir, drifted, {("k", "medgemma_q4", 0)})
    replay.check_immutable_config(run_dir, dict(prior), {("k", "medgemma_q4", 0)})  # same config ok
    replay.check_immutable_config(run_dir, drifted, set())                          # fresh run ok


def test_runner_refuses_config_drift_on_resume(tmp_path):
    run_dir = tmp_path / "studyA" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "responses.jsonl").write_text(
        json.dumps({"case": "A", "model": "gui_actor", "condition": "raw", "rep": 0, "ok": True}) + "\n")
    prior = {"models": ["gui_actor"], "conditions": "per-model defaults", "reps": 1, "k": 3,
             "max_tokens": 256, "prepare_manifest": "/m/manifest.jsonl", "images_root": None,
             "prompts": {"local_point_raw": {"sha256": "abc"}}}
    (run_dir / "meta.json").write_text(json.dumps(prior))
    drifted = dict(prior, prompts={"local_point_raw": {"sha256": "zzz"}})
    with pytest.raises(SystemExit, match="drifted"):
        runner.check_immutable_config(run_dir, drifted, {("A", "gui_actor", "raw", 0)})
    runner.check_immutable_config(run_dir, dict(prior), {("A", "gui_actor", "raw", 0)})


def test_score_coverage_missing_calls_counted_as_errors(tmp_path, monkeypatch):
    """Planned calls with no response row become explicit error rows in the denominators
    and are listed in score.json coverage (review P1-4: calls never silently vanish)."""
    set_dir = tmp_path / "done_check"
    set_dir.mkdir(parents=True)
    rows = [_row("done_check", "K0:v1:done", "not_done"), _row("done_check", "K1:v1:done", "done")]
    (set_dir / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    run_dir = set_dir / "r1"
    run_dir.mkdir()
    (run_dir / "meta.json").write_text(json.dumps({"reps": 1}))
    resp = [{"run": "r1", "key": "K0:v1:done", "case": "K0", "rep": 0, "model": "m",
             "raw_text": '{"answer": "not_done", "confidence": 0.9}', "error": None,
             "usage": None, "latency_ms": 1.0}]
    (run_dir / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in resp))
    monkeypatch.setattr(sys, "argv", ["score_decision.py", "--set", "done_check", "--run", "r1",
                                      "--results-root", str(tmp_path), "--n-boot", "10"])
    D.main()
    score = json.loads((run_dir / "score.json").read_text())
    m = score["metrics"]
    assert m["n"] == 2 and m["n_missing"] == 1 and m["n_response_error"] == 1
    # the recorded call is correct on the positive row (tp=1); the missing negative call
    # counts as fp -> specificity 0.0, accuracy 0.5 — NOT a 1.0/1.0 headline
    assert m["tp"] == 1 and m["fp"] == 1
    assert m["sensitivity"] == 1.0 and m["specificity"] == 0.0 and m["accuracy"] == 0.5
    assert score["coverage"] == {"planned": 2, "recorded": 1, "missing": 1,
                                 "missing_keys": ["K1:v1:done/r0"]}
    report = (run_dir / "report.md").read_text()
    assert "never recorded" in report and "K1:v1:done/r0" in report

    # a run that recorded its --keys subset validates only its planned calls: selecting
    # exactly the recorded key yields complete coverage (subset runs stay scoreable)
    (run_dir / "meta.json").write_text(json.dumps({"reps": 1, "selected_keys": ["K0:v1:done"]}))
    D.main()
    score = json.loads((run_dir / "score.json").read_text())
    assert score["coverage"] == {"planned": 1, "recorded": 1, "missing": 0, "missing_keys": []}


def test_score_compare_run_coverage_mirror_is_invariant(tmp_path, monkeypatch):
    """The comparison run must be coverage-normalized under the SAME missing-as-error
    policy before pairing: the paired count and McNemar p are invariant to the
    --run/--compare-run order (review finding: with runA perfect and runB missing 9 of
    10 calls, the old code paired 1 key with p=1.0 in one order and 10 keys with
    p~0.00391 in the other)."""
    set_dir = tmp_path / "candidate_judgement"
    set_dir.mkdir(parents=True)
    rows = [_row("candidate_judgement", f"K{i}:v1:C0", "lesion") for i in range(10)]
    (set_dir / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for run, keys in (("runA", range(10)), ("runB", [0])):
        run_dir = set_dir / run
        run_dir.mkdir(parents=True)
        (run_dir / "meta.json").write_text(json.dumps({"reps": 1}))
        (run_dir / "responses.jsonl").write_text("".join(
            json.dumps({"run": run, "key": f"K{i}:v1:C0", "case": f"K{i}", "rep": 0,
                        "model": "m", "error": None, "usage": None, "latency_ms": 1.0,
                        "raw_text": '{"answer": "lesion", "confidence": 0.9}'}) + "\n"
            for i in keys))

    def paired_vs(main, compare):
        monkeypatch.setattr(sys, "argv", ["score_decision.py", "--set", "candidate_judgement",
                                          "--run", main, "--compare-run", compare,
                                          "--results-root", str(tmp_path), "--n-boot", "10"])
        D.main()
        return json.loads((set_dir / main / "score.json").read_text())["paired_vs"]

    ab = paired_vs("runA", "runB")
    ba = paired_vs("runB", "runA")
    for comp in (ab, ba):
        assert comp["n_paired_keys"] == 10
        assert [comp["n_b_hit_a_miss"], comp["n_a_hit_b_miss"]] in ([0, 9], [9, 0])
        assert comp["mcnemar_p"] == pytest.approx(0.00391)
    # runB's scored rows contain the 9 synthesized missing-call error rows (same policy
    # as the main run's coverage normalization)
    b_scored = [json.loads(l) for l in
                (set_dir / "runB" / "scored.jsonl").read_text().splitlines() if l.strip()]
    assert sum(1 for c in b_scored if c["missing"]) == 9
    assert all(c["status"] == "response_error" and c["fn"] == 1  # positive rows -> fn
               for c in b_scored if c["missing"])
    # the old 'those keys do not pair' note is gone; missing compare calls are scored
    assert "same missing-as-error policy" in (set_dir / "runA" / "report.md").read_text()
    assert "do not pair" not in (set_dir / "runA" / "report.md").read_text()


def test_replay_template_rendering(tmp_path):
    from medgemma_client import extract_json_object  # noqa: F401  (client import path)

    template = (Path(STUDY_DIR / "prompts" / "decision_wrong_structure.md")).read_text()
    text = replay.render_template(template, {"text": "version v1 | volume 1 mL",
                                             "case": "C1", "target": "fibrous soft-tissue tumor",
                                             "sequence_text": "T1-weighted", "candidate_id": ""})
    assert "version v1 | volume 1 mL" in text
    assert "fibrous soft-tissue tumor" in text
    assert "{{" not in text
