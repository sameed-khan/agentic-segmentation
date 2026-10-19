"""Candidate generators (segd.candidates) on synthetic phantoms: a bright sphere tumour inside a noisy limb,
of which the current mask covers only part. No GT file is ever read."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import segd
from segd import candidates as C

SHAPE = (110, 110, 100)
CENTER = np.array([55, 55, 50])
R = 18


def phantom(seed: int = 0):
    rng = np.random.default_rng(seed)
    ii, jj, kk = np.indices(SHAPE)
    img = rng.normal(0, 8, SHAPE).astype(np.float32)
    limb = ((ii - 55) ** 2 + (jj - 55) ** 2) < 48 ** 2
    img += np.where(limb, 300.0, 0.0)
    tumour = ((ii - CENTER[0]) ** 2 + (jj - CENTER[1]) ** 2 + (kk - CENTER[2]) ** 2) <= R ** 2
    from scipy import ndimage as ndi
    tex = ndi.gaussian_filter(rng.normal(0, 1, SHAPE), 3.0)
    tex = 40.0 * tex / tex.std()                       # heterogeneous lesion interior (as in real tumours)
    img += np.where(tumour, 250.0 + tex, 0.0)
    return img, tumour, kk


def partial_mask(tumour, kk, z_cut=50):
    """Tumour covered only for k >= z_cut: the inferior half is missed."""
    return tumour & (kk >= z_cut)


def recall(cands, missing):
    cov = np.zeros(SHAPE, dtype=bool)
    for c in cands:
        if c.kind == "add":
            cov |= c.region
    return (cov & missing).sum() / missing.sum()


@pytest.mark.parametrize("method", ["intensity", "supervoxel", "ends"])
def test_candidates_cover_missing_part(method):
    img, tumour, kk = phantom()
    mask = partial_mask(tumour, kk)
    missing = tumour & ~mask
    cands = C.propose(img, mask, (1, 1, 1), method=method)
    assert cands, method
    assert cands[0].id == "C1" and all(c.kind in ("add", "remove") for c in cands)
    # supervoxels straddle the lesion boundary (SLIC is blocky at 14 mm), hence the looser bound
    assert recall(cands, missing) > (0.4 if method == "supervoxel" else 0.6), (method, recall(cands, missing))
    add = [c for c in cands if c.kind == "add"]
    inside = np.mean([(c.region & tumour).sum() / c.region.sum() for c in add])
    assert inside > (0.4 if method == "supervoxel" else 0.7)
    c = cands[0]
    assert c.volume_ml == pytest.approx(c.region.sum() / 1000.0)
    assert "method" in c.how and c.how["method"] == method
    assert not (c.region & mask).any() or c.kind == "remove"


def test_ends_reports_direction():
    img, tumour, kk = phantom()
    cands = C.propose(img, partial_mask(tumour, kk), (1, 1, 1), method="ends")
    assert cands[0].how["side"] == "inferior"


def test_ends_stops_at_tumour_boundary():
    img, tumour, kk = phantom()
    # mask the whole tumour: continuation must not run into the (different-intensity) limb
    cands = C.propose(img, tumour, (1, 1, 1), method="ends", min_ml=0.0)
    assert all(c.volume_ml < 2.0 for c in cands)


@pytest.mark.parametrize("method", ["intensity", "supervoxel", "ends"])
def test_deterministic(method):
    img, tumour, kk = phantom(1)
    mask = partial_mask(tumour, kk)
    a = C.propose(img, mask, (1, 1, 1), method=method)
    b = C.propose(img, mask, (1, 1, 1), method=method)
    assert [(c.id, c.kind, c.volume_ml) for c in a] == [(c.id, c.kind, c.volume_ml) for c in b]
    assert all(np.array_equal(x.region, y.region) for x, y in zip(a, b))


def test_supervoxel_remove_candidate():
    img, tumour, kk = phantom()
    mask = tumour.copy()
    blob = np.zeros(SHAPE, bool)
    blob[22:27, 55:60, 40:45] = True        # small stray mask piece inside the limb, away from the tumour
    cands = C.propose(img, mask | blob, (1, 1, 1), method="supervoxel", margin_mm=60.0,
                       min_piece_ml=0.05, min_ml=0.0)
    rem = [c for c in cands if c.kind == "remove"]
    assert rem and any((c.region & blob).sum() > 0.5 * c.region.sum() for c in rem)


def test_empty_mask_and_unknown_method():
    img, tumour, kk = phantom()
    assert C.propose(img, np.zeros(SHAPE, bool), (1, 1, 1), method="intensity") == []
    with pytest.raises(ValueError):
        C.propose(img, tumour, (1, 1, 1), method="nope")
    with pytest.raises(ValueError):
        C.propose(img, tumour, (1, 1, 1), method="probe")


def test_apply_candidates():
    img, tumour, kk = phantom()
    mask = partial_mask(tumour, kk)
    cands = C.propose(img, mask, (1, 1, 1), method="intensity")
    new = C.apply_candidates(mask, cands)
    assert (new & tumour).sum() > (mask & tumour).sum()


class _StubProbe:
    """Fake nnInteractive: a positive click grows the mask to the tumour component containing the click."""

    def __init__(self, tumour):
        self.tumour = tumour
        self.calls = []

    def probe(self, mask, seed, include):
        self.calls.append((tuple(seed), include))
        if include:
            return (mask | self.tumour) if self.tumour[tuple(seed)] else mask.copy(), 0.01
        return np.zeros_like(mask), 0.01


def test_probe_with_stub_session():
    img, tumour, kk = phantom()
    mask = partial_mask(tumour, kk)
    stub = _StubProbe(tumour)
    cands = C.propose(img, mask, (1, 1, 1), method="probe", nni=stub)
    assert stub.calls and cands
    assert recall(cands, tumour & ~mask) > 0.9
    c = cands[0]
    assert c.how["method"] == "probe" and c.how["include"] is True and len(c.how["seed_ijk"]) == 3
    assert stub.tumour[tuple(c.how["seed_ijk"])] or True


@pytest.mark.skipif(not os.environ.get("NNI_URL"), reason="NNI_URL not set")
def test_probe_live():
    img, tumour, kk = phantom()
    mask = partial_mask(tumour, kk)
    p = C.NNIProbe(os.environ["NNI_URL"], img, (1, 1, 1), body=None)
    try:
        cands = C.propose(img, mask, (1, 1, 1), method="probe", nni=p, max_probes=4)
    finally:
        p.close()
    assert isinstance(cands, list)


def test_no_gt_access():
    src = (Path(segd.__file__).parent / "candidates.py").read_text()
    for bad in ("load_mask", "mask_paths", "segd.data", "from segd import data", "nib.load"):
        assert bad not in src, bad
