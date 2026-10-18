"""Tasks 2+3: load each design case, check image/mask agreement, compute float16 packed upload size,
and render the three orthogonal slices through the GT centroid (for orientation verification).

uv run --project harness/server eval/v2/data_check.py [split]
"""
import json, sys, time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from nnInteractive.inference.remote.serialization import pack_array

from segd.data import read_cases, load_image, load_mask, PROJECT_ROOT

split = sys.argv[1] if len(sys.argv) > 1 else "design"
out = PROJECT_ROOT / "eval" / "v2" / "data_check"
(out / "slices").mkdir(parents=True, exist_ok=True)
recs = []


def to_u8(a, lo, hi):
    return (np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)


for row in read_cases(split):
    t0 = time.time()
    rec = {"case": row.case, "seq": row.sequence}
    try:
        img, aff, sp = load_image(row)
        m = load_mask(row, aff, img.shape)
        lab, n = ndi.label(m)
        sizes = ndi.sum(m, lab, range(1, n + 1)) if n else []
        c = np.array(ndi.center_of_mass(m)).round().astype(int)
        packed = len(pack_array(img.astype(np.float16)[None]))
        rec.update(shape=list(img.shape), spacing=sp, affine_identity=bool(np.allclose(aff, np.eye(4))),
                   img_min=float(img.min()), img_max=float(img.max()),
                   mask_ml=float(m.sum() * np.prod(sp) / 1000), n_components=int(n),
                   largest_frac=float(max(sizes) / m.sum()) if n else 0.0,
                   centroid=c.tolist(), centroid_in_mask=bool(m[tuple(c)]),
                   packed_mb=round(packed / 1e6, 1), flag_upload=packed > 90e6, ok=True)
        lo, hi = np.percentile(img[img > 0], [0.5, 99.5]) if (img > 0).any() else (img.min(), img.max())
        panels = []
        for ax in range(3):
            sl = np.take(img, c[ax], axis=ax)
            ms = np.take(m, c[ax], axis=ax)
            g = to_u8(sl, lo, hi)
            rgb = np.stack([g] * 3, -1)
            edge = ms.astype(bool) & ~ndi.binary_erosion(ms.astype(bool))
            rgb[edge] = [255, 0, 0]
            # display: array [r, c] -> transpose so first remaining axis is horizontal, flip vertical
            rgb = np.flipud(np.transpose(rgb, (1, 0, 2)))
            panels.append(Image.fromarray(rgb))
        h = max(p.height for p in panels)
        canvas = Image.new("RGB", (sum(p.width for p in panels) + 20, h), (40, 40, 40))
        x = 0
        for p in panels:
            canvas.paste(p, (x, 0)); x += p.width + 10
        canvas.save(out / "slices" / f"{row.case}.png")
    except Exception as e:
        rec.update(ok=False, error=repr(e))
    rec["sec"] = round(time.time() - t0, 1)
    print(json.dumps(rec), flush=True)
    recs.append(rec)

(out / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
