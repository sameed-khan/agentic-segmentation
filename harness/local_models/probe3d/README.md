# probe3d — 3D lesion-localizer probes for Phase 5 Study A

Study A asks: **given the case's MRI + target text, does a model localize the lesion?**
These probes run the two 3D text-prompted candidates from
`research/literature/lesion_localizer_candidates.md` (shortlist #1 BiomedParse v2, #2 SAT-Nano)
on the **full 3D volumes** of the 30 MSTT-199 **design** cases (`eval/v2/mstt_cases.csv`, split=design;
25 T2 / 5 T1; histotypes F/L/M/N/V per `segd.data.HISTOTYPE_TEXT`), bypassing the 2D slice harness.
Baseline on the same cases (2D slice pointing, harness E30/§4): Molmo2-8B 0.667, frontier Sonnet 0.911 hit@1.

Repos and pins:
- **BiomedParse v2**: `github.com/microsoft/BiomedParse`, default branch **v2**, pinned commit
  `f571d5ccaa4f8c0479bb89e390a7ef0dfc0736f4` at `~/BiomedParse`; checkpoint `biomedparse_v2.ckpt`
  (HF `microsoft/BiomedParse` @ `e473e5b2b1a3f44649734afd3dc7cf1770aaa9e2`) in `~/models/BiomedParse/`.
- **SAT-Nano (72-dataset)**: `github.com/zhaoziheng/SAT` @ `5676be5a62f5cd517264b948c6198e38ddfeb0e2`
  at `~/SAT`; checkpoints `Nano/nano.pth` + `Nano/nano_text_encoder.pth`
  (HF `zzh99/SAT` @ `56d1377859292e0cf2e9d0d7ff8ff696763725ca`) in `~/models/SAT/`;
  text-encoder base `FremyCompany/BioLORD-2023-C` @ `4bb5abd13930002136e3a5750b84474763c775c5`.
- Weights recorded in `~/models/weights_manifest_extra_3d.json`.

## Layout

```
probe3d/
├── common.py                  shared case loading (no GT), GPU wait/guards, VRAM sampler
├── score_probe3d.py           common OFFLINE scorer (GT via segd.data.load_mask)
├── biomedparse/               uv project (py3.10, torch 2.6.0+cu124 per README inference.txt)
│   └── run_biomedparse.py     runner (README 3D example pipeline; one [SEP] joint forward/case)
└── sat/                       uv project (py3.10, torch 2.1.2+cu121, SAT requirements.txt pins)
    └── run_sat.py             runner (official Inference_Dataset + inference_engine per case + back-map)
```

Outputs per model-prompt arm: `results/phase5/probe3d/<model>-<prompt>/`
`preds/<case>.nii.gz` (binary mask, case's canonical 1 mm [i,j,k] grid) + `runs.jsonl`
(per-case existence score, runtime, peak VRAM) + scorer-written `cases.csv`, `score.json`, `report.md`.

## Environments (created once)

```bash
export PATH=$HOME/.local/bin:$PATH
uv sync --project harness/local_models/probe3d/biomedparse   # torch 2.6.0 (cu124), hydra 1.3.2, transformers 4.40.0
uv sync --project harness/local_models/probe3d/sat           # SAT requirements.txt pins: numpy 1.21.5, monai 1.1.0,
                                                             # transformers 4.21.3, nibabel 4.0.2 + torch 2.1.2+cu121
```

Environment notes:
- The SAT venv patches `transformers/utils/hub.py` (`get_from_cache`/`http_get`): the 2026 HF hub
  answers resolve redirects with a **relative** `Location`, which transformers 4.21.3's downloader
  cannot follow (MissingSchema). Patch makes the redirect absolute; nothing else changed.
- SAT's customized U-Net backend is installed editable from the repo clone
  (`model/dynamic-network-architectures-main`) per README S1.
- BioLORD-2023-C is fetched by transformers 4.21.3 at first text-encoder load into the old-style
  `~/.cache/huggingface/transformers` cache (a copy is pinned in `~/models/BioLORD-2023-C`).
- Python/torch versions: biomedparse env = Python 3.10, torch 2.6.0+cu124; sat env = Python 3.10,
  torch 2.1.2+cu121. GPU: A100-SXM4-40GB (sm_80), driver 580.105.08 / CUDA 13.0.

## Run (GPU SHARED — read the rules)

GPU is shared with running studies: never kill processes/sessions you did not start; wrap every GPU
job in `flock /tmp/gpu_probe.lock`; check `nvidia-smi` free >= 22 GB before starting; one model at a
time; batch size 1. Both runners also wait for a free-GPU window and hard-abort below 22 GiB.

```bash
# 1) SAT-Nano (~24 GB per README table; batchsize_3d=1)
nvidia-smi    # confirm >= 26 GB free (model load ~4 GiB on top of the 22 GiB rule)
flock /tmp/gpu_probe.lock env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run --project harness/local_models/probe3d/sat torchrun --nproc_per_node=1 --master_port 1235 \
  harness/local_models/probe3d/sat/run_sat.py --limit 1        # smoke
flock /tmp/gpu_probe.lock env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run --project harness/local_models/probe3d/sat torchrun --nproc_per_node=1 --master_port 1235 \
  harness/local_models/probe3d/sat/run_sat.py                  # full 30 cases

# 2) BiomedParse v2
flock /tmp/gpu_probe.lock \
  uv run --project harness/local_models/probe3d/biomedparse python \
  harness/local_models/probe3d/biomedparse/run_biomedparse.py --limit 1   # smoke
flock /tmp/gpu_probe.lock \
  uv run --project harness/local_models/probe3d/biomedparse python \
  harness/local_models/probe3d/biomedparse/run_biomedparse.py             # full 30 cases

# 3) score (offline; CPU; harness/server env; GT via segd.data.load_mask)
SEGD_DATA_ROOT=/home/ubuntu/agentic-segmentation/data \
  uv run --project harness/server python harness/local_models/probe3d/score_probe3d.py --all
```

Inputs are read-only from `/home/ubuntu/agentic-segmentation/data` and the main tree's
`results/phase2_dev/prepare/manifest.jsonl` (k_star only, scorer side). Runners never read GT.

## Prompts

**BiomedParse v2** (all prompts joined with `[SEP]` in one forward per case, README pattern):
`tumor` (primary); the case's histotype text (`fibrous|lipomatous|myxoid soft-tissue tumor`,
`peripheral nerve sheath tumor`, `vascular soft-tissue tumor`); lesion-like MRI class names from the
HF-card data spec / README supported-tasks list: `tumor core`, `whole tumor`, `enhancing tumor`,
`lower-grade glioma`, `edema`, `brain tumor`, `stroke lesion`, `vestibular schwannoma`.
The per-prompt object-existence probability (per slice; max/mean recorded in runs.jsonl)
gates the masks at 0.5 exactly as `inference.postprocess` does.

**SAT-Nano** — vocabulary check FIRST (declared in the runner docstring + report): the label
vocabulary is the union of the SAT-DS repo's 72 dataset jsonl label lists (514 unique terms;
`data/dataset_config/72.json` itself holds only sampling/augmentation config). Tumour-like terms:
`brain tumor` / `enhancing brain tumor` / `necrotic brain tumor core` (MRI, BraTS),
`vestibular schwannoma` (MRI, CrossMoDA), `liver tumor` (MRI via ATLAS), bare `tumor` (PET only,
autoPET), `head and neck tumor` (PET), `kidney tumor` / `lung tumor` / `nasopharyngeal tumor` /
`colon cancer` (CT). **No `lesion`, `sarcoma`, `neoplasm` or soft-tissue/extremity class exists.**
Terms run: `tumor`, `brain tumor`, `vestibular schwannoma` (joint jsonl inference, per-class masks).

## Declared deviations / preprocessing

- BiomedParse intensity preprocessing (README ships no raw-NIfTI converter): the repo's own MRI
  recipe (`inference_utils/processing_utils.process_intensity_image`, non-CT branch) applied
  volumetrically — clip to the [0.5, 99.5] percentile over nonzero voxels, min-max scale to
  [0, 255], uint8 (challenge NPZ format: `imgs` uint8 [0,255], canonical orientation).
- BiomedParse `postprocess(..., do_nms=False)`: prompts are alternative phrasings of the same
  target, not mutually-exclusive classes; the README's per-slice NMS would suppress overlapping
  tumour classes. The object-existence gate (>0.5) is kept.
- SAT back-mapping: SAT writes masks in its PREPROCESSED grid (RAS, spacing (1,1,3),
  foreground-cropped, identity affine); the runner maps them back to the case's 1 mm grid
  (crop offset + nearest-neighbour x3 k-upsample) and verifies the mapping against SAT's saved
  preprocessed image per case.
- SAT image copies: cases are copied to `<workdir>/images/<case>.nii.gz` (read-only data untouched)
  so SAT's `sample_id` (filename minus `.nii.gz`) equals the case id.

## Scoring rules (score_probe3d.py)

Per case (1 mm isotropic, arrays [i,j,k], k = axial, k* from the prepare manifest):
- **3D hit**: deepest point of the LARGEST predicted component — argmax over the component of the
  GT EDT-depth map (ties -> first row-major index) — lies inside GT.
- **k\*-slice hit**: same rule on slice k* over the FULL predicted mask restricted to that slice
  (empty on k* = miss).
- **Dice** (3D, full prediction vs GT; largest-component Dice also recorded).
- **empty-prediction rate**; **distance from the 3D point to GT** in mm (0 if inside; closest
  predicted voxel distance also recorded as a stable miss-distance diagnostic).
- Per-case runtime and peak VRAM come from runs.jsonl (torch `max_memory_allocated` per case +
  pynvml process sampling).
