"""Build the 30-case design (development) manifest for MSTT-199, 6 per histotype.

Source of truth: Kaggle file listing of reasat/mstt-199 (isotropic), saved as
kaggle_mstt199_isotropic_files.tsv. Each case has exactly one annotated sequence
(T1 for most lipomatous, T2 otherwise). 15 cases have three-rater masks (_J/_K/_N);
these are EXCLUDED from the design set and reserved for the test set (inter-rater ceiling).
Remaining cases form the test pool. Deterministic (seed 2026).
"""
import csv, random, collections
rows = [l.rstrip("\n").split("\t") for l in open("kaggle_mstt199_isotropic_files.tsv")]
size = {n: int(b) for n, b in rows}
cases = collections.defaultdict(list)
for n in size:
    if n.startswith("isotropic/mask/"):
        stem = n.split("/")[-1].removesuffix(".nii")
        c, seq, *rater = stem.split("_")
        cases[c].append((seq, rater[0] if rater else None, n))
out = []
for c, ms in sorted(cases.items()):
    seqs = {s for s, _, _ in ms}; assert len(seqs) == 1, (c, ms)
    seq = seqs.pop()
    img = f"isotropic/image/{c}_{seq}.nii/{c}_{seq}.nii"; assert img in size, img
    multi = any(r for _, r, _ in ms)
    out.append(dict(case=c, histotype=c[0], sequence=seq, multi_rater=int(multi),
                    image_kaggle=img, masks_kaggle=";".join(sorted(m for *_, m in ms)),
                    image_mb=round(size[img] / 1e6, 1)))
rng = random.Random(2026)
design = set()
for h in "FLMNV":
    pool = sorted(r["case"] for r in out if r["histotype"] == h and not r["multi_rater"])
    design |= set(rng.sample(pool, 6))
for r in out:
    r["split"] = "design" if r["case"] in design else "test"
with open("mstt_cases.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
cnt = collections.Counter((r["split"], r["histotype"], r["sequence"]) for r in out)
for k in sorted(cnt): print(k, cnt[k])
print("design image GB:", round(sum(r["image_mb"] for r in out if r["split"] == "design") / 1e3, 2))
