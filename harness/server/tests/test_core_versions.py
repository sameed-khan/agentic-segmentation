"""Version DAG, undo/checkout, compact version storage, versions.npz round trip."""
from __future__ import annotations

import numpy as np
import pytest

from segd.case_state import OpError, pack_region, unpack_region, vid_label
from _helpers import install_stubs, lesion_center, make_case_state


def test_pack_region_round_trip():
    mask = np.zeros((20, 30, 40), np.uint8)
    mask[5:9, 10:20, 30:38] = 1
    bbox, packed, nvox = pack_region(mask)
    assert bbox == ((5, 9), (10, 20), (30, 38))
    assert nvox == 4 * 10 * 8
    assert (unpack_region(bbox, packed, nvox, mask.shape) == mask).all()
    # full-volume mask packs to ceil(nvox/8) bytes
    full = np.ones((13, 13, 13), np.uint8)
    bbox, packed, nvox = pack_region(full)
    assert len(packed) == (13 ** 3 + 7) // 8
    assert (unpack_region(bbox, packed, nvox, full.shape) == full).all()


def test_pack_region_empty():
    bbox, packed, nvox = pack_region(np.zeros((5, 5, 5), np.uint8))
    assert bbox is None and packed == b"" and nvox == 0
    assert (unpack_region(bbox, packed, nvox, (5, 5, 5)) == 0).all()


def test_version_dag_and_undo(tmp_path, monkeypatch):
    install_stubs(monkeypatch, force=True)
    cs, _lesion = make_case_state(tmp_path, monkeypatch)
    assert cs.current == 0 and vid_label(0) == "v0"
    assert (cs.current_mask() == 0).all()

    m1 = np.zeros(cs.shape, np.uint8)
    m1[:, :, 20] = 1
    assert vid_label(cs.commit_mask(m1)) == "v1"
    m2 = m1.copy()
    m2[:, :, 40] = 1
    assert vid_label(cs.commit_mask(m2)) == "v2"
    assert cs.versions[2].parent == 1
    assert (cs.version_mask(1) == m1).all()      # versions are immutable
    assert (cs.version_mask(2) == m2).all()

    # checkout v1 and branch: v3's parent is v1 (DAG, not a chain)
    assert vid_label(cs.checkout("v1")) == "v1"
    assert (cs.current_mask() == m1).all()
    m3 = m1.copy()
    m3[:, :, 60] = 1
    assert vid_label(cs.commit_mask(m3)) == "v3"
    assert cs.versions[3].parent == 1
    assert (cs.version_mask(1) == m1).all()      # still immutable after branching

    # undo = checkout parent
    assert vid_label(cs.undo()) == "v1"
    assert vid_label(cs.undo()) == "v0"
    with pytest.raises(OpError, match="no parent"):
        cs.undo()
    with pytest.raises(OpError, match="unknown version"):
        cs.checkout("v99")
    with pytest.raises(OpError, match="unknown version"):
        cs.checkout("banana")
    with pytest.raises(OpError, match="unknown version"):
        cs.checkout(7)


def test_versions_npz_round_trip(tmp_path, monkeypatch):
    install_stubs(monkeypatch, force=True)
    cs, _lesion = make_case_state(tmp_path, monkeypatch)
    masks = {}
    for vid, k in enumerate((10, 30, 50), start=1):
        m = np.zeros(cs.shape, np.uint8)
        m[:, :, k] = 1
        masks[cs.commit_mask(m)] = m
    cs.flush_versions()
    path = cs.out_dir / "versions.npz"
    assert path.exists()
    z = np.load(path)
    assert list(z["ids"]) == [0, 1, 2, 3]
    assert list(z["parents"]) == [-1, 0, 1, 2]
    for vid, m in masks.items():
        bb = z[f"v{vid}_bbox"]
        bbox = tuple((int(bb[a][0]), int(bb[a][1])) for a in range(3))
        back = unpack_region(bbox, z[f"v{vid}_packed"].tobytes(), int(z[f"v{vid}_nvox"]), cs.shape)
        assert (back == m).all()


def test_versions_npz_flushed_every_n(tmp_path, monkeypatch):
    install_stubs(monkeypatch, force=True)
    from segd import config
    monkeypatch.setattr(config, "VERSIONS_NPZ_EVERY", 2)
    cs, _lesion = make_case_state(tmp_path, monkeypatch)
    path = cs.out_dir / "versions.npz"
    assert not path.exists()
    for k in range(4):
        m = np.zeros(cs.shape, np.uint8)
        m[:, :, 5 * k] = 1
        cs.commit_mask(m)
    assert path.exists()          # flushed automatically during commits
    z = np.load(path)
    assert len(list(z["ids"])) == len(cs.versions)


def test_session_space_mask_round_trip_with_crop(tmp_path, monkeypatch):
    """Coordinate translation both ways, including a nonzero upload-crop offset."""
    install_stubs(monkeypatch, force=True)
    cs, lesion = make_case_state(tmp_path, monkeypatch, crop_forced=True)
    assert cs.crop_offset != (0, 0, 0)
    assert cs.crop_shape != cs.shape
    assert all(cs.crop_shape[i] <= cs.shape[i] for i in range(3))

    mask = lesion.astype(np.uint8)
    cs.commit_mask(mask)
    session = cs.current_mask_in_session_space()
    assert session.shape == cs.crop_shape
    # session crop is the mask restricted to the crop window
    sl = tuple(slice(o, o + n) for o, n in zip(cs.crop_offset, cs.crop_shape))
    assert (session == mask[sl]).all()
    assert (cs.session_mask_to_full(session) == mask).all()

    # ijk translation both ways
    ijk = lesion_center(lesion)
    s = cs.full_to_session_ijk(ijk)
    assert all(sx >= 0 for sx in s)
    assert cs.session_to_full_ijk(s) == ijk
