"""Round-trip check of the remote nnInteractive server through the SSH port-forward.

Synthetic MRI-like phantom (noise + smooth background + bright ellipsoid lesion),
sized like real MSTT isotropic volumes. Measures set_image upload time, single
point-prompt latency, and sanity Dice vs the phantom lesion.

Run:  uv run --with nninteractive-client --with numpy harness/checks/remote_nni_check.py [shape...]
"""
import sys, time
import numpy as np
from nnInteractive.inference.remote import nnInteractiveRemoteInferenceSession
from nnInteractive.inference.remote.serialization import pack_array

URL = "http://127.0.0.1:1527"
shapes = [(231, 231, 343)] if len(sys.argv) < 2 else [tuple(map(int, s.split("x"))) for s in sys.argv[1:]]

rng = np.random.default_rng(0)
for shape in shapes:
    X, Y, Z = shape
    xx, yy, zz = np.meshgrid(*(np.linspace(-1, 1, n, dtype=np.float32) for n in shape), indexing="ij")
    body = (xx**2 + yy**2) < 0.8          # cylinder "limb"
    lesion = ((xx - 0.2) / 0.15) ** 2 + ((yy + 0.1) / 0.12) ** 2 + (zz / 0.25) ** 2 < 1
    img = np.where(body, 300.0, 0.0) + np.where(lesion, 500.0, 0.0)
    img = (img + rng.normal(0, 40, shape) * body).astype(np.float16)[None]
    packed = pack_array(img)
    print(f"shape={shape} voxels={X*Y*Z/1e6:.1f}M  float16 raw={img.nbytes/1e6:.1f}MB  packed={len(packed)/1e6:.1f}MB", flush=True)

    t0 = time.time()
    s = nnInteractiveRemoteInferenceSession(URL, read_timeout=300, write_timeout=600)
    t1 = time.time(); s.set_image(img); t2 = time.time()
    tb = np.zeros(shape, dtype=np.uint8); s.set_target_buffer(tb); t3 = time.time()
    c = [int(round((v + 1) / 2 * (n - 1))) for v, n in zip((0.2, -0.1, 0.0), shape)]
    lat = []
    for i in range(3):
        a = time.time(); s.add_point_interaction(c, include_interaction=True); lat.append(time.time() - a)
    inter = np.logical_and(tb > 0, lesion).sum(); dice = 2 * inter / ((tb > 0).sum() + lesion.sum())
    print(f"  claim={t1-t0:.2f}s set_image={t2-t1:.2f}s set_target={t3-t2:.2f}s "
          f"point_latency={[round(x,2) for x in lat]}s dice_vs_phantom={dice:.3f}", flush=True)
    try:
        s.close()
    except Exception:
        pass
