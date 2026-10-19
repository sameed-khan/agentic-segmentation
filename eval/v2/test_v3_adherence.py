"""Unit checks for eval/v2/v3_adherence.py (RQ3 follow/override analysis) on synthetic
trajectories + a synthetic run dir (GT monkeypatched, no real data needed).

Run:  uv run --project harness/server python -m pytest eval/v2/test_v3_adherence.py -q
  or  uv run --project harness/server python eval/v2/test_v3_adherence.py  (self-runner)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3_adherence as VA  # noqa: E402


def test_score_module_loaded_by_path():
    """Regression (review P2 #8): the helper imports eval/v2/score.py by explicit path
    under a unique module name, so a combined pytest run (harness/study/score.py owning
    sys.modules['score']) cannot hijack it."""
    assert (Path(VA.S.__file__).resolve()
            == (Path(VA.__file__).resolve().parent / "score.py").resolve())
    assert sys.modules.get("eval_v2_score") is VA.S


def rec(op, resolved=None, error=None, child=None, parent=None, polarity=None,
        accepted=None, v3=None, **args):
    """A trajectory.jsonl-shaped record (mirrors eval/v2/test_score.py)."""
    args = dict(args)
    if polarity is not None:
        args["polarity"] = polarity
    r = {"op": op, "args": args, "resolved": resolved or [], "error": error,
         "child": child, "parent": parent}
    if accepted is not None:
        r["accepted"] = accepted
    if v3 is not None:
        r["v3"] = v3
    return r


def sphere(shape, center, radius) -> np.ndarray:
    g = np.indices(shape, dtype=float)
    for a in range(3):
        g[a] -= center[a]
    return (np.sqrt((g ** 2).sum(0)) <= radius).astype(np.uint8)


GT = sphere((40, 40, 40), (20, 20, 20), 12)
INSIDE = (20, 20, 20)
OUTSIDE = (5, 5, 5)
SPACING = (1.0, 1.0, 1.0)


def locate_v3(marks):
    """A seg_locate trajectory record with the given proposed mark ids."""
    snapped = [{"point": f"P{i + 1}", "mark": m, "score": 0.9 - i * 0.1, "dist_px": 1.0}
               for i, m in enumerate(marks)]
    return rec("locate", v3=[{"call": "locate", "ok": True, "snapped": snapped,
                              "image_sha256": "x" * 64}])


def test_collect_v3_events_ordering():
    traj = [
        rec("open", {}),
        locate_v3(["10", "11"]),
        rec("point", resolved=[{"id": 10, "ijk": INSIDE}], child="v1", polarity="add"),
        locate_v3(["20"]),                                   # after the click
        rec("sheet", {}),
    ]
    events = VA.collect_v3_events(traj)
    assert [l["marks"] for l in events["locates"]] == [["10", "11"], ["20"]]
    assert events["first_positive"]["ids"] == ["10"] and events["first_positive"]["order"] == 2
    assert VA._latest_locate_before(events) == ["10", "11"]  # the locate BEFORE the click


def test_pointer_adherence_followed_and_correct():
    traj = [
        locate_v3(["10", "11"]),
        rec("point", resolved=[{"id": 10, "ijk": INSIDE}], child="v1", polarity="add"),
    ]
    events = VA.collect_v3_events(traj)
    row = VA.pointer_adherence(events, True, VA.followed_click_inside(events, GT, SPACING))
    assert row["pointer_followed"] is True
    assert row["first_click_inside"] is True
    assert row["pointer_followed_inside"] is True
    assert row["locate_proposed_marks"] == "10;11"


def test_pointer_adherence_override():
    traj = [
        locate_v3(["10", "11"]),
        rec("point", resolved=[{"id": 99, "ijk": OUTSIDE}], child="v1", polarity="add"),
    ]
    events = VA.collect_v3_events(traj)
    row = VA.pointer_adherence(events, False)
    assert row["pointer_followed"] is False
    assert row["pointer_followed_inside"] is False
    assert row["first_click_inside"] is False


def test_pointer_adherence_without_locate_or_click():
    events = VA.collect_v3_events([rec("open", {}), rec("sheet", {})])
    row = VA.pointer_adherence(events, None)
    assert row == {"locate_proposed_marks": "", "first_positive_ids": "",
                   "pointer_followed": "", "first_click_inside": "",
                   "pointer_followed_inside": ""}


def test_pointer_adherence_inside_is_about_followed_points_only():
    """Regression (review P1 #6): propose mark 10 outside GT; the first positive point
    submits [10, 20] with an independently chosen mark 20 inside GT. Following 10 was
    WRONG even though some submitted point landed inside GT."""
    traj = [
        locate_v3(["10"]),
        rec("point", resolved=[{"id": 10, "ijk": OUTSIDE}, {"id": 20, "ijk": INSIDE}],
            child="v1", polarity="add"),
    ]
    events = VA.collect_v3_events(traj)
    row = VA.pointer_adherence(events, True,
                               VA.followed_click_inside(events, GT, SPACING))
    assert row["pointer_followed"] is True                # 10 was proposed and used
    assert row["pointer_followed_inside"] is False        # following it was wrong
    assert row["first_click_inside"] is True              # v2 whole-click metric unchanged


def test_pointer_adherence_edge_proposals_not_across_views():
    """Regression (review P1 #6, view identity): edge labels repeat per view — E1 on a
    later view is not the E1 proposed on an earlier view."""
    click = rec("point", resolved=[{"id": "E1", "ijk": INSIDE}], child="v1",
                polarity="add")
    events = VA.collect_v3_events([locate_v3(["E1"]), click])   # same view: follows
    row = VA.pointer_adherence(events, True, VA.followed_click_inside(events, GT, SPACING))
    assert row["pointer_followed"] is True and row["pointer_followed_inside"] is True
    events = VA.collect_v3_events([locate_v3(["E1"]),            # a successful seg_view
                                   rec("view", {"axis": "axial", "slice": 5}),   # in between
                                   click])
    row = VA.pointer_adherence(events, True, VA.followed_click_inside(events, GT, SPACING))
    assert row["pointer_followed"] == ""                  # E1 identity unverifiable:
    assert row["pointer_followed_inside"] == ""           # unusable advice -> undefined
    assert row["first_positive_ids"] == "E1"
    assert row["locate_proposed_marks"] == ""


def test_pointer_adherence_undefined_without_preceding_advice():
    """Regression (review P1 #6): a first positive click with no preceding locate advice
    leaves follow/override UNDEFINED ("") instead of recording a spurious override."""
    events = VA.collect_v3_events([
        rec("point", resolved=[{"id": 7, "ijk": INSIDE}], child="v1", polarity="add")])
    row = VA.pointer_adherence(events, True, None)
    assert row["pointer_followed"] == "" and row["pointer_followed_inside"] == ""
    assert row["locate_proposed_marks"] == ""
    assert row["first_click_inside"] is True              # the click metric is still set


def _mk_versions(path, shape, dice_versions: dict[str, tuple]):
    """Write a versions.npz in score.py's format; dice_versions: vid -> (bbox, crop);
    bbox None = empty version (all -1 + nvox 0, as score.py unpacks)."""
    arrays = {"shape": np.asarray(shape, dtype=np.int64)}
    ids, parents = [], []
    for vid, (bbox, crop) in dice_versions.items():
        n = int(vid[1:])
        ids.append(n)
        parents.append(n - 1)
        arrays[f"v{n}_bbox"] = (np.full((3, 2), -1, dtype=np.int64) if bbox is None
                                else np.asarray(bbox, dtype=np.int64))
        arrays[f"v{n}_nvox"] = np.asarray(0 if bbox is None else int(crop.size), dtype=np.int64)
        packed = b"" if bbox is None else np.packbits(crop.astype(bool), axis=None).tobytes()
        arrays[f"v{n}_packed"] = np.frombuffer(packed, dtype=np.uint8)
    arrays["ids"] = np.asarray(ids, dtype=np.int64)
    arrays["parents"] = np.asarray(parents, dtype=np.int64)
    np.savez_compressed(path, **arrays)


def test_candidate_adherence_verdicts_and_dice_delta():
    events = {
        "medgemma": [
            {"call": "checkpoint", "checkpoint": "review", "order": 0, "version": "v1",
             "parsed": {"complete": "no", "confidence": 0.6}},
            {"call": "checkpoint", "checkpoint": "candidates", "order": 1, "version": "v1",
             "candidate_id": "C1",
             "parsed": {"candidate_id": "C1", "verdict": "lesion", "confidence": 0.8}},
            {"call": "checkpoint", "checkpoint": "candidates", "order": 2, "version": "v1",
             "candidate_id": "C2",
             "parsed": {"candidate_id": "C2", "verdict": "normal", "confidence": 0.9}},
        ],
        "accepts": [{"ids": ["C1"], "from": "v1", "to": "v2", "order": 3}],
    }
    row = VA.candidate_adherence(events, None, SPACING, {"v1": 0.1, "v2": 0.2})
    assert row["review_complete"] == "no" and row["review_confidence"] == 0.6
    assert row["candidate_verdicts"] == "v1:C1:lesion;v1:C2:normal"
    assert row["accepted_ids"] == "C1" and row["n_accepts"] == 1
    assert row["n_accepts_medgemma_lesion"] == 1            # followed the verdict
    assert row["n_accepts_overridden"] == 0 and row["n_accepts_unexposed"] == 0
    assert row["accept_adherence"] == "v1->v2:followed"
    assert row["accept_dice_delta"] == "v1->v2:+0.1000"     # and it was right (+dice);
    # one entry for the whole accept OPERATION, labelled with its version pair


def test_candidate_adherence_version_and_order_aware():
    """Regression (review P1 #5): candidate ids restart at C1 for every proposal set.
    An accept matches only advice already delivered for ITS from-version and candidate
    id, BEFORE the accept — a later, conflicting C1 verdict must not re-label the
    earlier acceptance, and an accept without advice stays unexposed."""
    events = {
        "medgemma": [
            {"call": "checkpoint", "checkpoint": "candidates", "order": 0, "version": "v1",
             "candidate_id": "C1",
             "parsed": {"candidate_id": "C1", "verdict": "lesion", "confidence": 0.8}},
            # v2's review: a DIFFERENT C1 (normal) — never advice about the v1 set
            {"call": "checkpoint", "checkpoint": "candidates", "order": 3, "version": "v2",
             "candidate_id": "C1",
             "parsed": {"candidate_id": "C1", "verdict": "normal", "confidence": 0.9}},
        ],
        "accepts": [
            {"ids": ["C1"], "from": "v1", "to": "v2", "order": 1},   # followed v1 advice
            {"ids": ["C1"], "from": "v2", "to": "v3", "order": 4},   # overrode v2 advice
            {"ids": ["C1"], "from": "v3", "to": "v4", "order": 6},   # no advice for v3
        ],
    }
    row = VA.candidate_adherence(events, None, SPACING,
                                 {"v1": 0.1, "v2": 0.2, "v3": 0.2, "v4": 0.3})
    assert row["candidate_verdicts"] == "v1:C1:lesion;v2:C1:normal"
    assert row["accepted_ids"] == "C1;C1;C1" and row["n_accepts"] == 3
    assert row["n_accepts_medgemma_lesion"] == 1            # only the v1 accept followed
    assert row["n_accepts_overridden"] == 1 and row["n_accepts_unexposed"] == 1
    assert row["accept_adherence"] == ("v1->v2:followed;v2->v3:override;"
                                       "v3->v4:unexposed")
    assert row["accept_dice_delta"] == "v1->v2:+0.1000;v2->v3:+0.0000;v3->v4:+0.1000"


def test_candidate_adherence_accept_operation_delta_once():
    """Regression (review P2 #7): for seg_accept([C1, C2]) the operation-level Dice delta
    is reported ONCE with the op's adherence status, never once per candidate."""
    events = {
        "medgemma": [
            {"call": "checkpoint", "checkpoint": "candidates", "order": 0, "version": "v1",
             "candidate_id": "C1",
             "parsed": {"candidate_id": "C1", "verdict": "lesion", "confidence": 0.8}},
            {"call": "checkpoint", "checkpoint": "candidates", "order": 1, "version": "v1",
             "candidate_id": "C2",
             "parsed": {"candidate_id": "C2", "verdict": "normal", "confidence": 0.9}},
        ],
        "accepts": [{"ids": ["C1", "C2"], "from": "v1", "to": "v2", "order": 2}],
    }
    row = VA.candidate_adherence(events, None, SPACING, {"v1": 0.1, "v2": 0.2})
    assert row["accepted_ids"] == "C1;C2" and row["n_accepts"] == 2
    assert row["n_accepts_medgemma_lesion"] == 1 and row["n_accepts_overridden"] == 1
    assert row["accept_adherence"] == "v1->v2:followed-override"
    assert row["accept_dice_delta"] == "v1->v2:+0.1000"     # once, not per candidate


def test_candidate_adherence_ignores_mismatched_reply():
    """Regression (review P1 #2, attribution side): the server binds each record to the
    REQUESTED candidate id and nulls unusable replies — a reply naming C2 for a C1
    request produces no C2 advice and no attribution."""
    events = {
        "medgemma": [{"call": "checkpoint", "checkpoint": "candidates", "order": 0,
                      "version": "v1", "candidate_id": "C1", "parsed": None,
                      "error": "unusable reply: candidate id mismatch: requested C1, "
                               "reply names C2"}],
        "accepts": [{"ids": ["C1"], "from": "v1", "to": "v2", "order": 1}],
    }
    row = VA.candidate_adherence(events, None, SPACING, {"v1": 0.0, "v2": 0.2})
    assert row["candidate_verdicts"] == ""                 # no C2 advice anywhere
    assert row["n_accepts_medgemma_lesion"] == 0 and row["n_accepts_overridden"] == 0
    assert row["n_accepts_unexposed"] == 1                 # the accept stays unexposed
    assert row["accept_adherence"] == "v1->v2:unexposed"


def test_candidate_adherence_reject_ignored():
    events = {"medgemma": [{"call": "checkpoint", "checkpoint": "candidates", "order": 0,
                            "version": "v1", "candidate_id": "C2",
                            "parsed": {"candidate_id": "C2", "verdict": "normal"}}],
              "accepts": []}
    row = VA.candidate_adherence(events, None, SPACING, None)
    assert row["candidate_verdicts"] == "v1:C2:normal"
    assert row["accepted_ids"] == "" and row["accept_dice_delta"] == ""


def test_versions_dice_map(tmp_path):
    shape = (40, 40, 40)
    empty = np.zeros(shape, bool)
    v1 = np.zeros(shape, bool)
    v1[20:26, 20:26, 14:20] = True
    _mk_versions(tmp_path / "versions.npz", shape,
                 {"v0": (None, empty), "v1": (v1_bbox, v1_crop)})
    z = np.load(tmp_path / "versions.npz")
    dice = VA.versions_dice_map(z, GT)
    assert dice["v0"] == 0.0
    assert 0.0 < dice["v1"] < 1.0
    assert VA.versions_dice_map(None, GT) == {}


def test_adherence_case_end_to_end(tmp_path, monkeypatch):
    """Full case-dir path: trajectory + versions.npz + monkeypatched GT loaders."""
    shape = (40, 40, 40)
    case_dir = tmp_path / "run" / "X01"
    case_dir.mkdir(parents=True)
    empty = np.zeros(shape, bool)
    v1 = np.zeros(shape, bool)
    v1[18:24, 18:24, 18:24] = True
    traj = [
        rec("open", {}),
        locate_v3(["10"]),
        rec("view", {"axis": "axial", "slice": 20}),
        rec("point", resolved=[{"id": 10, "ijk": (20, 20, 20)}], child="v1", polarity="add"),
        rec("sheet", {}),
        rec("finish", {"confidence": "high", "reason": "r"}),
    ]
    (case_dir / "trajectory.jsonl").write_text(
        "\n".join(json.dumps(r) for r in traj) + "\n")
    _mk_versions(case_dir / "versions.npz", shape, {"v0": (None, empty), "v1": (v1_bbox, v1_crop)})

    import segd.data as data_mod

    class FakeRow:
        histotype = "F"
        sequence = "T2"
        image_path = str(case_dir / "image.nii")

    import nibabel as nib
    img = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4))
    nib.save(img, str(case_dir / "image.nii"))
    monkeypatch.setattr(data_mod, "get_case", lambda case: FakeRow())
    monkeypatch.setattr(data_mod, "load_mask", lambda row, affine, shape_ref: GT)

    row = VA.adherence_case(case_dir)
    assert row["case"] == "X01" and row["histotype"] == "F"
    assert row["status"] == "ok"
    assert row["n_locate_calls"] == 1
    assert row["medgemma_checkpoints_seen"] == ""
    assert row["pointer_followed"] is True and row["first_click_inside"] is True
    assert row["pointer_followed_inside"] is True
    assert row["accepted_ids"] == ""          # no accepts in this trajectory


def test_adherence_case_v2_run_degrades(tmp_path, monkeypatch):
    """A v2 (no v3 records) trajectory: every v3 column empty, no crash."""
    case_dir = tmp_path / "run" / "X02"
    case_dir.mkdir(parents=True)
    (case_dir / "trajectory.jsonl").write_text(
        json.dumps(rec("open", {})) + "\n" + json.dumps(rec("finish", {"confidence": "high"})) + "\n")
    import segd.data as data_mod

    class FakeRow:
        histotype = "M"
        sequence = "T1"
        image_path = "unused"

    def boom(case):
        raise KeyError(case)
    monkeypatch.setattr(data_mod, "get_case", boom)
    monkeypatch.setattr(data_mod, "load_mask", lambda *a, **k: GT)
    row = VA.adherence_case(case_dir)
    assert row["status"] == "missing_gt"
    assert row["n_locate_calls"] == 0
    assert row["medgemma_checkpoints_seen"] == ""
    assert row["pointer_followed"] == ""


# ---------------------------------------------------------------------- bbox/crop
v1_bbox = ((18, 24), (18, 24), (18, 24))
v1_crop = np.ones((6, 6, 6), bool)


# ----------------------------------------------------------------- self-runner
if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {name}: {e}")
        except Exception as e:                             # noqa: BLE001
            failed += 1
            print(f"ERROR {name}: {e!r}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
