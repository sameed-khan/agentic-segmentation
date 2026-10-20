# Lesion-localizer candidates for the agentic segmentation harness

**Research checked:** 2026-10-20. Primary-source-first (papers, HF cards, official repos; fetched directly where cited).
No weights downloaded, no GPU jobs run. "Verified" = checked against the linked primary source, not reproduced.
Estimates and inferences are labeled. Builds on `pointer_models_gui_actor_molmo2.md`, `medgemma_1_5.md`,
`local_visuospatial_vlms.md` — does not repeat them.

## Ranked shortlist (top 5)

### 1. BiomedParse v2 — `microsoft/BiomedParse`, ckpt `biomedparse_v2.ckpt` (4.46 GB)
- **What:** text-prompted joint segmentation/detection/recognition foundation model, 9 modalities; v2 uses
  BoltzFormer with end-to-end 3D inference (slice-with-neighbouring-context). Paper: Zhao et al., Nat Methods 2025
  ([paper](https://doi.org/10.1038/s41592-024-02499-w)); v2 code branch
  ([repo, default branch v2](https://github.com/microsoft/BiomedParse)); checkpoint repo
  ([HF](https://huggingface.co/microsoft/BiomedParse), files: `biomedparse_v1.pt` 1.8 GB, `biomedparse_v2.ckpt` 4.46 GB).
- **Release / licence:** v2 code+checkpoint released ~Oct 2025 (repo update Sep 2026 noted in prior brief).
  Code Apache-2.0; **model for research/development use only, not for clinical use**
  ([HF card usage notice](https://huggingface.co/microsoft/BiomedParse)). Research licence OK per harness constraints.
- **Size/VRAM:** 4.46 GB checkpoint file (verified file listing). Peak inference VRAM **not published** — needs a
  one-case memory probe; prior brief estimates it fits comfortably in budget. CPU/GPU PyTorch, not an LLM endpoint.
- **Input:** full 3D volume (NIfTI; v2 3D notebook `inference_example_3D.ipynb`); we have the full 1 mm isotropic NIfTI —
  no slice-rendering adaptation needed. **Output:** mask + object-existence score.
- **MRI/lesion coverage (verified, HF card data-spec):** `MRI-FLAIR brain: edema, lower-grade glioma, tumor, tumor
  core, whole tumor`; `MRI-T1-Gd brain: enhancing tumor, tumor core`. Generic `MRI:` abdomen/brain/heart/prostate
  classes are **anatomy-only (no tumour)**; CT has `tumor` (abdomen/colon/liver); dermoscopy has `lesion/melanoma`.
  **No musculoskeletal / soft-tissue-tumour class.** Repo warns it "reasonably segments only targets learned in
  pretraining" and gives no external-dataset guarantee (prior brief; HF card: "no guarantee for the same performance
  on external datasets").
- **Evidence:** paper reports strong in-distribution results (often-cited brain-tumour Dice ~0.95–0.97 are v1
  brain-MRI examples, not a v2 generalisation benchmark — treat as transfer signal only). Challenge leaderboard #1 in
  2025 text-guided 3D biomedical segmentation (repo statement, prior brief).
- **Fit (a) MSTT-199 soft-tissue tumours:** MEDIUM-LOW. No matching class; best attempt is generic `tumor` prompt
  (a CT-trained token) applied to T1/T2/T2-FS extremity volumes — true zero-shot test, expect weak but informative
  results. **Fit (b) BEAMSTER brain mets:** MEDIUM-HIGH, best medical prior in this report: T1-Gd enhancing-tumor and
  FLAIR whole-tumor heads exist; caveat — trained on **glioma** subregions, and metastases (small, spherical,
  multiplicity) differ morphologically; BEAMSTER contrast-T1 matches the T1-Gd head's modality.
- **Leakage:** no — trained on public biomedical corpora (BiomedParseData); MSTT-199/BEAMSTER are new, not in it.
- **Study A plan:** bypass Study A's 2D slice harness; run 3D inference on the 30 design volumes with prompts
  `["tumor"]`, `["enhancing tumor", "tumor core"]` (brain-context prompts only meaningful if tested on brain data —
  for MSTT use `tumor`); derive GT-largest-slice hit (mask centroid → point; empty mask = abstention) and per-case
  Dice. Conda env python 3.10, torch cu124 per v2 README; `hf_hub_download(repo_id="microsoft/BiomedParse",
  filename="biomedparse_v2.ckpt")`. Record VRAM peak, empty-mask rate, false-positive rate.

### 2. SAT-Nano (72-dataset) — `zzh99/SAT` (HF) + `zhaoziheng/SAT` (code)
- **What:** "Segment Anything with Text": knowledge-enhanced universal 3D segmentation, 497 classes across
  **MR/CT/PET**, 8 body regions, prompted by anatomical terminology. Paper: Zhao et al., npj Digital Medicine 2025
  ([arXiv 2312.17183](https://arxiv.org/html/2312.17183v5)); code ([GitHub](https://github.com/zhaoziheng/SAT));
  checkpoints ([HF zzh99/SAT](https://huggingface.co/zzh99/SAT): SAT-Pro, SAT-Nano + 5 Nano variants).
- **Release / licence:** 2024 checkpoints, 2025 journal version. Licence **not verified** — no licence file surfaced
  in fetched sources; check repo/HF before use (research use assumed, confirm).
- **Size/VRAM (author-reported):** SAT-Nano batchsize_3d=1 **~24 GB**, Pro ~34 GB (official README table). Nano fits
  the ≤30 GB budget solo; run serially with nnInteractive (1–5 GB) or reduce `max_queries`. Training needed 8×A100 —
  inference only here.
- **Input:** 3D NIfTI + jsonl (`image`, `label` = target terms, `dataset`, `modality: mri`); handles orientation/
  intensity/spacing normalisation internally. **Output:** per-class 3D masks → centroid = Study A point.
- **MRI/lesion coverage:** 497-class vocabulary over 72 public 3D datasets incl. MRI; exact soft-tissue-sarcoma or
  brain-metastasis class membership **unverified** — must check paper Table 12 / `data/dataset_config/72.json`
  (fetched repo file listing confirms it exists) for terms like `tumor`, `lesion`, BraTS classes before running.
  Prompt must be a **supported anatomical term**, not free text ("fibrous soft-tissue tumor" will likely fail verbatim;
  back off to `tumor`/`lesion`).
- **Evidence:** beats specialist nnU-Net-style baselines on many of 72 datasets in-distribution (paper radar plots);
  CVPR-2025 text-guided-3D-seg challenge baseline. No published MSTT/BEAMSTER numbers.
- **Fit (a) MSTT:** MEDIUM — depends on vocabulary check; extremity MRI under-represented in public 3D corpora.
  **Fit (b) BEAMSTER:** MEDIUM — BraTS-family data is almost certainly in the 72 datasets, so glioma-head transfer to
  mets is plausible; small-lesion sensitivity unknown.
- **Leakage:** no for MSTT/BEAMSTER (public-dataset training only); note BraTS overlap inflates any BraTS-adjacent
  brain evaluation — not our datasets, so fine.
- **Study A plan:** inspect `72.json` for supported tumour terms → build `demo.jsonl`-style file over 30 design NIfTIs
  (`modality: mri`); run `inference.py --vision_backbone UNET --batchsize_3d 1`; map each predicted mask to GT-slice
  hit + Dice. If vocabulary lacks tumour terms, record as vocabulary-reject (still a finding).

### 3. MolmoPoint-8B — `allenai/MolmoPoint-8B` (HF)
- **What:** successor to Molmo2-8B (our 0.667 baseline) replacing text-coordinate pointing with grounding tokens
  (`<PATCH>/<SUBPATCH>/<LOCATION>`: coarse-to-fine selection over visual features; 8→3 tokens/point; explicit
  stop). Blog ([Ai2, Mar 2026](https://allenai.org/blog/molmopoint)); card
  ([HF](https://huggingface.co/allenai/MolmoPoint-8B)); code in existing `allenai/molmo2` repo.
- **Release / licence:** Mar 2026; **Apache-2.0**, same third-party-data academic/non-commercial caveat as Molmo2
  ([HF card](https://huggingface.co/allenai/MolmoPoint-8B)). Research use OK.
- **Size/VRAM:** ~9B class (same envelope as Molmo2-8B: ~17.3 GB BF16 est., 8-bit/NF4 or serial scheduling as in
  prior brief). New decoding API: `return_pointing_metadata=True`, `build_logit_processor_from_inputs`,
  `extract_image_points` — **Study A parser must be rewritten** (not the `<points>` regex).
- **Input:** 2D slice PNG (drop-in for Study A raw/marks panels); multi-image supported. **Output:** pixel points.
- **Evidence (author-reported):** PointBench 70.7 avg (vs 68.7 Molmo2-8B), PixMo-Points F1 89.2 (vs 85.2); GUI variant
  61.1 ScreenSpot-Pro SOTA-among-fully-open; trains faster (20 F1 pts better at 8k examples). No medical data in
  training (same PixMo/natural-image recipe) — **no medical prior**, pure pointer upgrade.
- **Fit (a) MSTT:** MEDIUM-HIGH as a Molmo2 replacement — same harness, strictly better general pointer; whether
  +2–4 pts general gain transfers to MRI is the experiment. **Fit (b) BEAMSTER:** MEDIUM — small multi-lesion
  pointing untested; new stop-token and ordering may help multi-focal output (Molmo2 emitted ~1 location).
- **Leakage:** no.
- **Study A plan:** rerun identical 30-case raw + marks conditions; prompts "Point to the {histotype} soft-tissue
  tumor"; `max_new_tokens=200`, BF16; metrics hit@1/hit@3 (test multi-point emission explicitly — the key Molmo2
  weakness), malformed rate, latency. Compare paired vs Molmo2-8B 0.667/0.600 and frontier 0.911/0.700.

### 4. RadGrounder-detection — `lmb-freiburg/radgrounder` (HF) + GitHub
- **What:** the only verified **medical grounding VLM with box output on 2D MR slices**: PaliGemma-2 (3B) fine-tuned
  on report+VQA+grounding; detection variant loads with **stock transformers**
  (`PaliGemmaForConditionalGeneration.from_pretrained("models/detection")`); segmentation variant needs repo code.
  Paper: Salcan et al., MICCAI 2026 ([paper](https://papers.miccai.org/miccai-2026/paper/4764_paper.pdf),
  [arXiv](https://arxiv.org/html/2606.20477)); repo ([GitHub](https://github.com/lmb-freiburg/RadGrounder)); card
  ([HF](https://huggingface.co/lmb-freiburg/radgrounder)).
- **Release / licence:** 2026 (MICCAI paper). **Access risk: HF repo states weights "private until the paper is
  accepted — `hf auth login` with an account that has access"** — verify access first; may be gated/unavailable.
  Licences: `LICENSE_APACHE` + `LICENSE_MODELS` (model-weights terms need review).
- **Size/VRAM:** 3B PaliGemma-2 → ~6 GB BF16 est.; easiest fit in this report; runs alongside nnInteractive.
- **Input:** 2D CT/MR slices. **Output:** boxes (detection) / masks (segmentation) tied to finding text.
- **MRI/lesion coverage — the catch:** grounding supervision is **TotalSegmentator anatomy** (117 CT + 50 MRI classes
  harmonised to 121; 236k slice–text pairs, only 18.5k MRI) — i.e. **organs, not lesions/tumours**. No tumour-box
  training. Expect it to ground "femur", "muscle", "brain" well and "{histotype} tumor" poorly; value is (i) a
  same-harness medical-grounding architecture test, (ii) an **anatomical prior** (exclude normal structures).
- **Evidence:** SLAKE open/closed F1 ~0.86–0.91, VQA-RAD F1 0.50 (repo table); grounding accuracy numbers are on
  anatomy (gIoU evaluator in repo), not lesions.
- **Fit (a) MSTT:** LOW-MEDIUM as localizer; MEDIUM as anatomical excluder. **Fit (b) BEAMSTER:** LOW-MEDIUM (brain
  anatomy grounding may help slice-level triage, not met detection).
- **Leakage:** no (private Freiburg RefRad2D + public SLAKE/VQA-RAD).
- **Study A plan:** if access granted: prompt Study A slices with histotype phrase → box→centroid→hit@1; plus
  anatomy-prompt arm ("femur", "muscle", nearest large anatomy) to test excluder role. If access denied, record and
  skip.

### 5. Qwen3-VL-8B-Instruct — `Qwen/Qwen3-VL-8B-Instruct` (HF) [general-pointer arm; 30B-A3B-Q4 as stretch]
- **What:** strongest general open grounding-VLM family release Sep 2025: 2D grounding (boxes **and points**),
  3D grounding, visual-agent/tool use, native multi-image, 256K context. Card
  ([HF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)); report ([arXiv 2511.21631](https://arxiv.org/abs/2511.21631)).
- **Release / licence:** Sep 2025; **Apache-2.0**. Research use OK.
- **Size/VRAM:** 8.77B → ~17.5 GB BF16 est. (4-bit safer; same recipe as Molmo2). Stretch: `Qwen3-VL-30B-A3B-Instruct`
  (MoE, 3B active) RefCOCO 89.3 avg vs 91.1 for 32B-dense (report); official **FP8** (~30 GB — too tight) and
  **GGUF Q4** (`Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF`, ~15–16 GB + vision — fits 40 GB solo, vision quality unverified).
  32B-dense BF16 (~64 GB) ruled out; Q4-quant 32B (~16 GB) possible but unvalidated for grounding fidelity.
- **Input:** 2D PNG (Study A drop-in); multi-image/agent harness possible. **Output:** elicited box/point JSON.
- **Evidence:** report claims SOTA 2D grounding/counting (48.6 mAP ODinW-13); 8B-specific RefCOCO number **not
  extracted** (report PDF §3.2.4 covers grounding; verify before citing); EGM-Qwen3-VL-8B efficiency variant claims
  91.4 RefCOCO IoU (secondary source, unverified). No medical training.
- **Fit (a)/(b):** MEDIUM, same logic as MolmoPoint: better general grounder, no medical prior; 8B is the fair
  Molmo2-size comparator; 30B-A3B-Q4 tests "much better pointing" hypothesis within 40 GB.
- **Leakage:** no.
- **Study A plan:** 8B BF16 (flash_attention_2): identical raw/marks prompts requesting `{"x":, "y":}` 0–1000 point;
  metrics as MolmoPoint arm. Stretch: 30B-A3B GGUF-Q4 single-image only if 8B shows promise.

## Broader consideration table

| Candidate | ID / source | Verdict | Reason |
|---|---|---|---|
| LesionLocator (MIC-DKFZ) | `github.com/MIC-DKFZ/LesionLocator`, ckpt via Zenodo 15174217 | **Adjacent: test as promptable segmenter, not localizer** | Zero-shot universal lesion seg (CVPR 2025), pretrained PET/CT/**MRI** — but needs point/box prompts; doesn't solve localisation. Pair with a localizer (prompt = candidate box). 5-fold ensemble heavy; `-f 0 --disable_tta` for probe. CC BY-NC-SA dataset; check ckpt licence. |
| MedSAM2 | `wanglab/MedSAM2` ([HF](https://huggingface.co/wanglab/MedSAM2)) | **Adjacent, not a localizer** | Promptable 3D medical SAM; needs points/boxes; CT-lesion ckpt only; liver-MRI annotation demo ≠ soft-tissue/brain-mets evidence. Useful downstream of a localizer, like nnInteractive alternative. |
| TotalSegmentator-MRI | `wasserth/TotalSegmentator` (Apache-2.0) | **Anatomical prior only** | 50 MR anatomy classes, 616 routine MRIs; **no tumour/lesion classes**. Use to exclude normal structures; same role as RadGrounder-anatomy but non-VLM, zero prompt risk. Lightweight, runnable. |
| MONAI BraTS seg | `MONAI/brats_mri_segmentation` (HF; NGC Apache-2.0) | **BEAMSTER-adjacent baseline; likely input-mismatched** | Segments glioma ET/TC/WT but **requires 4 aligned modalities (T1c+T1+T2+FLAIR)**; BEAMSTER is contrast-T1 (likely single-sequence) → needs input-ablation test. No MSTT relevance. |
| Met-Seg (MICCAI PRIME 2024) | `xmindflow/Met-Seg` | **BEAMSTER candidate, weights unverified** | Two-stage detect→segment for brain mets on T1c/T1/FLAIR — closest task match for BEAMSTER incl. small mets. But: 2024 (not 2025+ SOTA), **no downloadable-weights link verified** in fetched sources, single-centre BraTS-Mets training. Verify weights + licence before any plan. |
| NKI BrainMets / DeepMedic+ | `nki-radiology/BrainMets`, `YixingHuang/DeepMedicPlus` | **Rejected for now** | Pipeline without clear downloadable weights (BrainMets) / weights on-request-only (DeepMedic+). Revisit if Met-Seg fails. |
| SegVol | `BAAI/SegVol` | **Rejected (CT-only)** | 90k unlabeled + 6k labeled **CT** volumes, 200+ CT anatomy classes; no MRI. |
| OmniLesion (FLARE 2026) | `Nucleo-Research/omnilesion` | **Rejected (CT-only + gated)** | Pan-cancer CT (0.812 DSC hidden-val), CC BY-NC-4.0 access-on-request; CT volumes only. |
| VISTA3D / NV-Segment-CTMR | `Project-MONAI/VISTA`, `nvidia/NV-Segment-CT` | **Rejected as localizers** | Auto (class-label) + point-prompt seg; **not text-prompted**; CT coverage (CTMR 345+ classes incl. MR, but label-selected, no lesion-text grounding). No tumour-text→location path. |
| MedGrounder / uMedGround | `aehrc/MedGrounder`, `Cocofeat/uMedGround` | **Rejected (chest-X-ray-only)** | MS-CXR/PadChest-GR phrase grounding (SOTA on those); no MRI, weights via Drive (MedGrounder) / code-only + datasets "coming soon" (uMedGround, TPAMI 2025). |
| LocAnyMed-200K | `MiliLab/LocAnyMed` (arXiv 2608.03322) | **Watch, not runnable** | Promising multimodal medical grounding dataset, but code/data/checkpoints **"will be released in subsequent updates"** — nothing to download. |
| MAISI / NV-Generate-MR-Brain | NVIDIA NGC/HF | **Rejected (generative, not localizer)** | MR generator = images only, no paired tumour masks; BraTS-seg bundle is the MONAI model above. |
| M3D-LaMed (prior brief) | `GoodBaiBai88/M3D-LaMed-Phi-3-4B` | **Deprioritised (kept from prior brief)** | 3D medical VLM but 1×32×256×256 `.npy` pipeline friction; no new evidence to promote it. |
| UI-TARS-1.5-7B, GUI-Actor-7B, Qwen3-VL-32B-dense-BF16, InternVL-78B-class | various | **Rejected on VRAM/domain** | 7B+ BF16 envelopes (14–17 GB) leave no safe margin with nnInteractive on 40 GB alongside harness; GUI-only domain (UI-TARS/GUI-Actor); 32B/78B BF16 impossible solo. Q4 variants unvalidated for grounding fidelity — except 30B-A3B-Q4 kept as stretch (MoE, official GGUF). |

## Test order & concrete Study A adaptations

1. **MolmoPoint-8B** (cheapest: 2D drop-in, rewrite parser) → 2. **BiomedParse v2** (best BEAMSTER prior; 3D volume path,
   memory probe first) → 3. **SAT-Nano** (vocabulary check first; 24 GB run) → 4. **Qwen3-VL-8B** (general-pointer
   comparator; +30B-A3B-Q4 stretch) → 5. **RadGrounder** (gated by access approval).
   Output→point rule for all: mask→centroid (largest component; empty=abstain), box→center, point→direct; snap to
   nearest mark in marks condition (lossless per Study A diagnostics); report hit@1, hit@3 (any-hit), >25 mm
   wrong-structure rate, empty/malformed rate, VRAM peak, latency; paired bootstrap CIs vs Molmo2-8B and frontier.

## Open questions
1. SAT vocabulary: does `72.json` contain any soft-tissue-sarcoma or meningioma/metastasis-adjacent MRI class?
   (Decides SAT's MSTT vs BEAMSTER emphasis.)
2. BiomedParse v2 prompt semantics on non-brain MRI: does the `tumor` token generalise across body regions, or is it
   region-locked? (One 30-case probe answers this.)
3. RadGrounder weight access: is `lmb-freiburg/radgrounder` publicly downloadable now, and under what model terms?
4. Met-Seg weights: downloadable? licence? (Determines whether BEAMSTER gets a task-matched detector baseline.)
5. Qwen3-VL-8B exact RefCOCO number + quantised grounding fidelity (BF16 vs Q4) — needed before claiming "much
   better pointing than Molmo2".
6. MSTT-199 acquisition sources (TCIA collections?) — confirm no overlap with SAT-72 / BiomedParseData public
   corpora for a formal leakage statement.

## Sources
- Kept: BiomedParse paper DOI 10.1038/s41592-024-02499-w; [v2 repo](https://github.com/microsoft/BiomedParse);
  [HF ckpt](https://huggingface.co/microsoft/BiomedParse) (4.46 GB file, MRI class spec, research-only notice).
- Kept: SAT paper arXiv:2312.17183; [code](https://github.com/zhaoziheng/SAT) (VRAM table, inference recipe);
  [HF zzh99/SAT](https://huggingface.co/zzh99/SAT) (Pro/Nano + variants).
- Kept: [MolmoPoint blog](https://allenai.org/blog/molmopoint) (70.7 PointBench, 89.2 PixMo-Points, GUI 61.1);
  [HF card](https://huggingface.co/allenai/MolmoPoint-8B) (Apache-2.0, grounding-token API).
- Kept: RadGrounder [paper](https://papers.miccai.org/miccai-2026/paper/4764_paper.pdf) / [arXiv](https://arxiv.org/html/2606.20477)
  (236k pairs, 18.5k MRI, TotalSegmentator-derived labels); [repo](https://github.com/lmb-freiburg/RadGrounder)
  (stock-transformers detection loading, gated access); [HF](https://huggingface.co/lmb-freiburg/radgrounder).
- Kept: LesionLocator [repo readme](https://github.com/MIC-DKFZ/LesionLocator/blob/main/readme.md)
  (promptable point/box, Zenodo ckpt, single-fold/TTA flags); CVPR 2025 paper page.
- Kept: Qwen3-VL [HF 8B card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) (Apache-2.0, grounding+agent claims);
  [tech report](https://arxiv.org/abs/2511.21631) (32B 91.1 / 30B-A3B 89.3 RefCOCO); [FP8/GGUF variants](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8).
- Kept: [wanglab/MedSAM2](https://huggingface.co/wanglab/MedSAM2); [TotalSegmentator repo](https://github.com/wasserth/TotalSegmentator)
  (50 MR classes, Apache-2.0); [MONAI BraTS card](https://huggingface.co/MONAI/brats_mri_segmentation) (4-modality
  requirement); [Met-Seg repo](https://github.com/xmindflow/Met-Seg); [OmniLesion card](https://huggingface.co/Nucleo-Research/omnilesion)
  (CT-only, CC BY-NC-4.0); [MedGrounder repo](https://github.com/aehrc/MedGrounder) (chest-X-ray-only).
- Rejected/deprioritized: secondary blog summaries and leaderboard aggregators (used only for discovery, never as
  evidence); EGM-Qwen3-VL-8B 91.4 claim (secondary source, unverified); Molmo2-8B 63.1-academic-mean (not grounding
  evidence); VISTA3D/NV-Segment marketing pages (no text-grounding path).
