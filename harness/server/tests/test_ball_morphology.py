"""ball_morphology must equal scipy binary morphology with op_edit's ellipsoid structure (border_value=0)."""
import numpy as np
from scipy import ndimage

from segd.ops import ball_morphology

OPS = {"open": ndimage.binary_opening, "close": ndimage.binary_closing,
       "dilate": ndimage.binary_dilation, "erode": ndimage.binary_erosion}


def _struct(r_vox):
    axes = [np.arange(-r, r + 1) / max(r, 1) for r in r_vox]
    g = np.meshgrid(*axes, indexing="ij")
    return sum(x ** 2 for x in g) <= 1.0 + 1e-9


def test_ball_morphology_matches_scipy():
    rng = np.random.default_rng(0)
    for t in range(20):
        sh = tuple(int(x) for x in rng.integers(20, 45, 3))
        m = ndimage.gaussian_filter(rng.random(sh), rng.uniform(1, 4)) > rng.uniform(0.45, 0.55)
        if t % 3 == 0:
            m[:3] = True   # mask touching the array border
        r = tuple(int(x) for x in rng.integers(1, 6, 3))
        for op, f in OPS.items():
            assert (f(m, structure=_struct(r)) == ball_morphology(m, op, r)).all(), (t, op, r)
