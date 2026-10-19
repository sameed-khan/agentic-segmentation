"""Build the candidate-decision replay set from an agent run (E16).

    uv run --project harness/server python harness/study/prepare_candidate_replay.py [results/phase1-10-high-cand]

For every first-stage seg_finish review that showed candidates, record: the two review images the agent saw
(3x5 review grid, candidate sheet), the review text, and per candidate a GT label (precision = fraction of the
candidate volume inside GT; label 'lesion' if > 0.5). Candidates are recomputed with the same deterministic
GT-free generator on the reviewed mask version and matched to the logged ids/volumes (asserted).
Output: results/candidate_replay/manifest.jsonl (offline; GT-derived labels never go to the model).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval" / "v2"))
import score as S  # noqa: E402
from segd import candidates as C  # noqa: E402
from segd.data import get_case, load_image, load_mask  # noqa: E402
from segd.marks import body_mask  # noqa: E402


def main() -> None:
    run = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "results" / "phase1-10-high-cand").resolve()
    out = ROOT / "results" / "candidate_replay"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_dir in sorted(p for p in run.iterdir() if p.is_dir() and len(p.name) == 3):
        traj = [json.loads(l) for l in open(case_dir / "trajectory.jsonl")]
        revs = [t for t in traj if t["op"] == "finish" and t.get("candidates") and not t.get("error")]
        if not revs:
            continue
        row = get_case(case_dir.name)
        img, aff, sp = load_image(row)
        gt = load_mask(row) > 0
        body = body_mask(img)
        z = np.load(case_dir / "versions.npz")
        accepted = sorted({i for t in traj if t["op"] == "accept" and not t.get("error") for i in t["args"].get("ids", [])})
        for r in revs:
            items = r["candidates"]["items"] if isinstance(r["candidates"], dict) else r["candidates"]
            if not items:
                continue
            vid = int(str(r["parent"]).lstrip("v"))
            m = S.unpack_version(z, vid, img.shape)
            cands = C.propose(img, m, sp, method="intensity", max_n=5, cache={"body": body})
            assert [c.id for c in cands] == [i["id"] for i in items], (case_dir.name, vid)
            for c, i in zip(cands, items):
                assert abs(c.volume_ml - float(i["volume_ml"])) < 0.05, (case_dir.name, c.id)
            labels = []
            for c in cands:
                prec = float((c.region & gt).sum() / max(int(c.region.sum()), 1))
                labels.append({"id": c.id, "volume_ml": round(c.volume_ml, 1), "precision": round(prec, 3),
                               "label": "lesion" if prec > 0.5 else "normal",
                               "region_vox": int(c.region.sum()), "inter_vox": int((c.region & gt).sum()),
                               "overlap_mask_vox": int((c.region & m).sum())})
            # candidates are disjoint add-regions outside the mask -> Dice of any accepted subset is additive
            for a in range(len(cands)):
                for b in range(a + 1, len(cands)):
                    assert not (cands[a].region & cands[b].region).any(), (case_dir.name, "overlap")
            gtn = int(gt.sum())
            d0 = 2 * int((m & gt).sum()) / (int(m.sum()) + gtn)
            rows.append({"key": f"{case_dir.name}_v{vid}", "case": case_dir.name, "histotype": row.histotype,
                         "sequence": row.sequence, "version": f"v{vid}", "dice_reviewed": round(d0, 4),
                         "mask_vox": int(m.sum()), "gt_vox": gtn, "mask_gt_vox": int((m & gt).sum()),
                         "images": [str((case_dir / p).relative_to(ROOT)) for p in r["renders"]],
                         "review_text": r.get("observation", ""), "candidates": labels,
                         "agent_accepted": accepted, "source_run": str(run.relative_to(ROOT))})
            print(rows[-1]["key"], [(l["id"], l["label"]) for l in labels])
    with open(out / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n = sum(len(r["candidates"]) for r in rows)
    k = sum(l["label"] == "lesion" for r in rows for l in r["candidates"])
    print(f"{len(rows)} reviews, {n} candidates, {k} lesion-labelled -> {out / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
