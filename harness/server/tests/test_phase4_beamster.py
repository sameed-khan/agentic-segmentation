"""Phase 4 (BEAMSTER, decision-tested): manifest reader + SEGD_DATA_ROOT remap,
dataset-aware multi-lesion observation text, per-component review sheet, MSTT defaults
untouched, and a FakeNNI smoke path on one REAL design case (read-only main-tree data).

The real-data smoke test is skipped when /home/ubuntu/agentic-segmentation/data is absent
(CI); everything else runs offline on synthetic data.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pytest

import segd.data
from segd.data import (
    CaseRow, get_case, read_cases, read_cases_phase4, remap_data_path,
)
from segd.observe import components_text, components_text_for, mask_stats

from _helpers import install_stubs, make_case_state, make_phantom

MAIN_DATA = Path("/home/ubuntu/agentic-segmentation/data")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_COLUMNS = ["case", "target_text", "modality", "sequence", "image_path", "mask_paths",
                    "n_lesions", "lesion_volume_ml", "split", "image_path_src", "mask_paths_src"]


def _write_manifest(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def _manifest_row(case="Dataset_A_001", n_lesions=2, split="design",
                  root="/home/ubuntu/agentic-segmentation/data"):
    return {
        "case": case, "target_text": "brain metastases (all enhancing lesions on CE-T1 MRI)",
        "modality": "MRI", "sequence": "CE-T1",
        "image_path": f"{root}/phase4/beamster/nifti_iso/image/{case}.nii.gz",
        "mask_paths": f"{root}/phase4/beamster/nifti_iso/mask/{case}.nii.gz",
        "n_lesions": str(n_lesions), "lesion_volume_ml": "2.22", "split": split,
        "image_path_src": f"{root}/phase4/beamster/{case}.nii.gz",
        "mask_paths_src": f"{root}/phase4/beamster/{case}_segm.nii.gz",
    }


# --------------------------------------------------------------------- manifest reader
def test_phase4_manifest_reader_and_remap(tmp_path, monkeypatch):
    monkeypatch.setattr(segd.data, "PHASE4_CSV", str(tmp_path / "{dataset}_cases.csv"))
    _write_manifest(tmp_path / "beamster_cases.csv",
                    [_manifest_row(), _manifest_row("Dataset_A_002", 1, "test")])
    monkeypatch.setenv("SEGD_DATA_ROOT", "/mnt/a100data")     # the A100 mount point

    rows = read_cases_phase4("beamster", "design")
    assert len(rows) == 1
    r = rows[0]
    assert r.case == "Dataset_A_001" and r.modality == "MRI" and r.n_lesions == 2
    assert r.split == "design" and r.sequence == "CE-T1"
    # manifest absolute paths remapped onto $SEGD_DATA_ROOT
    assert str(r.image_path) == "/mnt/a100data/phase4/beamster/nifti_iso/image/Dataset_A_001.nii.gz"
    assert r.mask_paths == [Path("/mnt/a100data/phase4/beamster/nifti_iso/mask/Dataset_A_001.nii.gz")]
    assert r.target_text == "brain metastases (all enhancing lesions on CE-T1 MRI)"
    # split filter
    assert [x.case for x in read_cases_phase4("beamster", "test")] == ["Dataset_A_002"]
    # get_case dispatch (dataset arg) + unknown case/dataset
    assert get_case("Dataset_A_001", "beamster").n_lesions == 2
    assert get_case("Dataset_A_002", "beamster").split == "test"
    with pytest.raises(KeyError):
        get_case("nope", "beamster")
    with pytest.raises(FileNotFoundError):
        read_cases_phase4("nosuch")


def test_phase4_remap_rules():
    # only MANIFEST_DATA_ROOT-prefixed paths are remapped
    assert str(remap_data_path("/home/ubuntu/agentic-segmentation/data/phase4/x.nii.gz")) == \
        str(segd.data.data_root() / "phase4/x.nii.gz")
    assert remap_data_path("/other/mount/x.nii.gz") == Path("/other/mount/x.nii.gz")
    assert remap_data_path("rel/x.nii.gz") == Path("rel/x.nii.gz")


def test_mstt_rows_keep_defaults():
    """MSTT behaviour untouched: CaseRow defaults, histotype target text, get_case."""
    rows = read_cases("design")
    assert rows
    for r in rows[:5]:
        assert r.modality == "" and r.n_lesions is None and r.manifest_target_text is None
        assert r.target_text == segd.data.HISTOTYPE_TEXT[r.histotype]
    assert get_case("F14").case == "F14"
    assert get_case("F14", "mstt").case == "F14"
    with pytest.raises(KeyError):
        get_case("Dataset_A_001", "mstt")     # beamster cases are not in the mstt manifest


# ----------------------------------------------------------------- observation text
class _Cs:
    def __init__(self, dataset):
        self.dataset = dataset


def test_multi_components_text_and_mstt_identity():
    m = np.zeros((50, 50, 50), np.uint8)
    m[5:15, 5:15, 5:15] = 1            # 1.0 mL
    m[30:36, 30:36, 30:36] = 1         # 0.216 mL
    m[40:41, 10:11, 10:12] = 1         # 0.002 mL
    st = mask_stats(m, (1.0, 1.0, 1.0))
    assert st["n_components"] == 3
    assert st["component_ml"][0] == 1.0 and len(st["component_ml"]) == 3
    # MSTT: byte-identical legacy line
    assert components_text_for(_Cs("mstt"), st) == components_text(st)
    assert "largest 82.1%" in components_text_for(_Cs("mstt"), st)
    # Phase 4: component count + volumes of up to the 10 largest components
    multi = components_text_for(_Cs("beamster"), st)
    assert multi == "components 3 (mL: 1.00, 0.22, 0.002)"
    # >10 components: the tail is counted, never any GT count
    big = np.zeros((100, 100, 100), np.uint8)
    for g in range(13):
        z = 5 + g * 7
        big[z:z + 2, z:z + 2, z:z + 2] = 1
    st13 = mask_stats(big, (1.0, 1.0, 1.0))
    t13 = components_text_for(_Cs("beamster"), st13)
    assert t13.startswith("components 13 (mL: ") and t13.endswith("; 3 smaller)")
    assert len(st13["component_ml"]) == 10


def test_default_case_state_is_mstt(tmp_path, monkeypatch):
    """CaseState without the dataset arg behaves as mstt (identity-test contract)."""
    install_stubs(monkeypatch, force=True)
    cs, _lesion = make_case_state(tmp_path, monkeypatch)
    assert cs.dataset == "mstt"
    assert (tmp_path / "results" / "run1" / "phantom" / "case_meta.json").read_text() \
        .count('"dataset": "mstt"') == 1


# ------------------------------------------------------- per-component review sheet
def _dilate3d(m, it=1):
    from scipy import ndimage as ndi
    return ndi.binary_dilation(m, iterations=it)


def test_render_review_components():
    from segd.render import render_review_components
    img, lesion, _body = make_phantom()
    mask = np.zeros(img.shape, np.uint8)
    mask[lesion] = 1
    mask[10:14, 20:24, 30:34] = 1
    mask[50:53, 60:63, 20:23] = 1
    im, ri = render_review_components(img, mask, (1.0, 1.0, 1.0))
    assert im.width <= 1568 and ri.kind == "review_components"
    assert len(ri.views) == 9                     # 3 components x (axial, coronal, sagittal)
    assert {v.axis for v in ri.views} == {"axial", "coronal", "sagittal"}
    im1, ri1 = render_review_components(img, (lesion & _dilate3d(lesion, 2)).astype(np.uint8),
                                        (1.0, 1.0, 1.0))
    assert len(ri1.views) == 3                    # single component -> one row


# ------------------------------------------------------------ real-data smoke (FakeNNI)
def test_marks_density_profile_off_by_default_and_fingerprinted():
    """Decision 5: the dense profile is a dataset-level option; MSTT/default marks are
    byte-identical to the frozen constants, the profile is fingerprinted into the cache."""
    from segd import marks as M
    img, _lesion, _body = make_phantom()
    ms_def = M.compute_marks(img, (1.0, 1.0, 1.0))
    ms_dense = M.compute_marks(img, (1.0, 1.0, 1.0), profile="dense")
    assert "profile" not in ms_def.params                       # default: untouched params
    assert ms_dense.params.get("profile") == "dense"
    assert ms_dense.params["fine_spacing_mm"] == 3.0
    assert ms_dense.params["visible_cap"] == 200
    n_fine = lambda ms: sum(1 for m in ms.marks.values() if m.level == "fine")  # noqa: E731
    assert n_fine(ms_dense) > n_fine(ms_def)                    # denser fine 3D thinning
    # cache fingerprint: a default-profile cache never validates a dense MarkSet
    assert M.cache_matches(ms_def, img.shape)
    assert not M.cache_matches(ms_dense, img.shape)
    assert M.cache_matches(ms_dense, img.shape, profile="dense")
    # visible() honours the profile cap without an explicit max_n
    assert len(ms_dense.visible("axial", img.shape[2] // 2, None, "fine")) <= 200
    with pytest.raises(ValueError):
        M.compute_marks(img, (1.0, 1.0, 1.0), profile="nope")


@pytest.mark.skipif(not (MAIN_DATA / "phase4/beamster/nifti_iso").is_dir(),
                    reason="real BEAMSTER data not mounted (read-only main tree)")
def test_beamster_smoke_fakenni_real_design_case(tmp_path, monkeypatch):
    """One real design case, FakeNNI backend, full agent-visible flow. Read-only on
    /home/ubuntu/agentic-segmentation/data (SEGD_DATA_ROOT remap, decisions 1/7)."""
    monkeypatch.setenv("SEGD_DATA_ROOT", str(MAIN_DATA))
    install_stubs(monkeypatch, force=False)       # real marks/render on the real volume
    rows = read_cases_phase4("beamster", "design")
    row = next(r for r in rows if r.case == "Dataset_A_001")

    from segd import ops
    from segd.case_state import CaseState
    from segd.data import load_case_geometry, load_mask
    g = load_case_geometry(row)
    # out_dir under the worktree results root (gitignored) so the CaseState mark cache
    # is shared across runs; the data itself is only ever read
    out_dir = PROJECT_ROOT / "results" / "_smoke_phase4" / f"pytest{id(tmp_path) % 100000}" / row.case
    cs = CaseState(row.case, g.image, g.affine, g.spacing, row.target_text, row.sequence,
                   out_dir=out_dir, nni_url=None,
                   budgets={"max_tool_calls": 50, "max_model_prompts": 20}, dataset="beamster")
    cs.initialize()
    try:
        assert cs.budgets == {"max_tool_calls": 50, "max_model_prompts": 20}
        assert not hasattr(cs, "n_lesions")       # GT count never enters the agent path

        out = ops.run_op(cs, "open", {})
        assert "all enhancing lesions" in out.text and "50 tool calls, 20 model prompts" in out.text
        ops.run_op(cs, "sheet", {"axis": "axial", "start": 100, "stop": 180})

        # a mark on/near a GT lesion (GT use stays in this OFFLINE test, CONTRACTS §0);
        # pick from the marks DRAWN in the new active view (resolve_ids requires that)
        gt = load_mask(row, cs.affine, cs.image.shape).astype(bool)
        centre = tuple(int(round(a.mean())) for a in np.nonzero(gt))
        ops.run_op(cs, "view", {"axis": "axial", "slice": centre[2], "marks": "fine"})
        drawn = cs.active_view.render_info.marks       # [(mark_id, panel, x, y), ...]
        assert drawn
        near = min(drawn, key=lambda d: sum((a - b) ** 2 for a, b in
                                            zip(cs.markset.marks[int(d[0])].ijk, centre)))
        out = ops.run_op(cs, "point", {"ids": [int(near[0])]})
        assert "mL:" in out.text                  # multi-lesion components line (decision 3)
        out = ops.run_op(cs, "finish", {"confidence": "low", "reason": "review"})
        assert not out.done
        code = re.search(r'review_code "(\d{4})"', out.text).group(1)
        out = ops.run_op(cs, "finish", {"confidence": "high", "reason": "ok", "review_code": code})
        assert out.done and cs.done
        meta_txt = (out_dir / "case_meta.json").read_text().replace("'", '"')
        assert '"dataset": "beamster"' in meta_txt
    finally:
        cs.close()
