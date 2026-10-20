#!/usr/bin/env python
"""probe3d runner: SAT-Nano (72-dataset) on the 30 MSTT-199 design cases (Study A probe).

Uses the official SAT repo (`~/SAT` @ 5676be5a62f5cd517264b948c6198e38ddfeb0e2)
inference path verbatim: data.inference_dataset.Inference_Dataset (monai RAS +
Spacing (1,1,3) + foreground crop + MRI percentile/z-score normalization),
evaluate.inference_engine.inference (288x288x96 patches, gaussian accumulation,
sigmoid > 0.5), vision backbone UNET + text encoder 'ours' with the Nano
checkpoints from HF zzh99/SAT (README S4/S5, batchsize_3d=1 ~= 24 GB).

Differences from `torchrun inference.py --datasets_jsonl all.jsonl` (declared):
  - the case loop happens in THIS script (model + text encoder loaded once,
    exactly as inference.py does) so per-case runtime and peak VRAM are recorded;
  - case images are copied to <workdir>/images/<case>.nii.gz so SAT's sample_id
    (image filename minus .nii.gz) equals the case id;
  - the per-class predictions SAT writes in its PREPROCESSED grid (RAS,
    spacing (1,1,3), foreground-cropped, identity affine) are mapped BACK to the
    case's 1 mm isotropic canonical grid here (crop-offset + nearest-neighbour
    x3 k-upsample), because Study A scores masks in the case grid. The mapping is
    verified per case against SAT's saved preprocessed image (img_<id>.nii.gz).

SAT vocabulary check (data/dataset_config/72.json holds sampling/augmentation
config only; the label vocabulary is the union of the SAT-DS repo's 72 dataset
jsonl label lists, 514 unique terms). Tumour/lesion-like terms found:
  'brain tumor', 'enhancing brain tumor', 'necrotic brain tumor core' (MRI, BraTS),
  'vestibular schwannoma' (MRI, CrossMoDA), 'liver tumor' (MRI via ATLAS, CT via MSD),
  'tumor' (PET only, autoPET), 'head and neck tumor' (PET only),
  'kidney tumor' (CT only), 'lung tumor' (CT), 'nasopharyngeal tumor' (CT),
  'colon cancer' (CT).
  NO 'lesion', 'sarcoma', 'neoplasm' or any soft-tissue/extremity class exists.
Terms run here (best 1-3 MRI-valid): 'tumor' (generic, PET-attested),
'brain tumor' (most frequent MRI tumour term), 'vestibular schwannoma' (MRI,
nerve-sheath histology - closest match to histotype N).

Run (GPU): flock /tmp/gpu_probe.lock torchrun --nproc_per_node=1 --master_port 1235 \
             run_sat.py
Inputs: read-only case NIfTIs (no GT). Outputs: results/phase5/probe3d/sat-<term>/
  preds/<case>.nii.gz + runs.jsonl in THIS worktree.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import torch

SAT_REPO = Path("/home/ubuntu/SAT")
sys.path.insert(0, str(SAT_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # probe3d/common.py

import common  # noqa: E402

# Best 1-3 vocabulary terms (see module docstring for the full vocabulary findings).
SAT_TERMS = ["tumor", "brain tumor", "vestibular schwannoma"]
MODALITY = "mri"


def build_backmap(case_image_path: Path, orig_k: int):
    """Recompute SAT's preprocessing chain (same monai, same env) to recover the
    foreground bbox in the spacing-(1,1,3) grid; return (backmap, spaced_img)
    where backmap maps a SAT-grid prediction (H', W', D') to the original
    [i, j, k] 1 mm grid (nearest neighbour, background outside the crop)."""
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    )

    loader = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(axcodes="RAS", keys=["image"]),          # no-op: identity RAS affine
        Spacingd(keys=["image"], pixdim=(1, 1, 3), mode=("bilinear")),
    ])
    spaced = loader({"image": str(case_image_path)})["image"]  # (1, H, W, D_sp)

    from monai.transforms import CropForeground
    bbox_start, bbox_end = CropForeground().compute_bounding_box(spaced)
    cropped = spaced.numpy()[0, bbox_start[0]:bbox_end[0],
                             bbox_start[1]:bbox_end[1],
                             bbox_start[2]:bbox_end[2]]
    # SAT's load_image applies its Normalization AFTER the crop; mirror it so the
    # per-case verification below compares like with like.
    from data.inference_dataset import Normalization
    cropped_norm = Normalization(torch.from_numpy(cropped.copy()), MODALITY).numpy()

    def backmap(pred: np.ndarray) -> np.ndarray:
        h, w, d = pred.shape
        full_spaced = np.zeros(spaced.shape[1:], dtype=pred.dtype)
        full_spaced[bbox_start[0]:bbox_start[0] + h,
                    bbox_start[1]:bbox_start[1] + w,
                    bbox_start[2]:bbox_start[2] + d] = pred
        # nearest-neighbour k upsample x3 back to the 1 mm grid
        up = np.repeat(full_spaced, 3, axis=2)[:, :, :orig_k]   # (H, W, K)
        return up

    return backmap, cropped_norm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(Path.home() / "models/SAT/Nano/nano.pth"))
    ap.add_argument("--text-encoder-checkpoint", default=str(Path.home() / "models/SAT/Nano/nano_text_encoder.pth"))
    ap.add_argument("--vision-backbone", default="UNET")       # README: SAT-Nano 72-dataset
    ap.add_argument("--text-encoder", default="ours")
    ap.add_argument("--text-encoder-partial-load", default="True")
    ap.add_argument("--batchsize-3d", type=int, default=1)     # ~24 GB per README table
    ap.add_argument("--max-queries", type=int, default=256)
    ap.add_argument("--crop-size", type=int, nargs="+", default=[288, 288, 96])
    ap.add_argument("--patch-size", type=int, nargs="+", default=[32, 32, 32])
    ap.add_argument("--deep-supervision", default="False")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cases-csv", default=str(common.DEFAULT_CASES_CSV))
    ap.add_argument("--workdir", default="/tmp/probe3d_sat")
    ap.add_argument("--min-free-gib", type=float, default=22.0,
                    help="free GiB required BEFORE model load (task rule: >= 22 GiB)")
    ap.add_argument("--postload-free-gib", type=float, default=18.0,
                    help="abort if free GiB after model load is below this (OOM headroom)")
    ap.add_argument("--wait-gpu-min", type=float, default=180.0,
                    help="minutes to wait for a free-GPU window before aborting")
    args = ap.parse_args()

    cases = common.read_design_cases(Path(args.cases_csv))
    if args.limit:
        cases = cases[: args.limit]

    # wait for a shared-GPU window BEFORE taking the lock / loading the model:
    # require min_free_gib so that the >= 22 GiB guard also holds AFTER model load.
    common.wait_gpu_free(min_gib=args.min_free_gib, timeout_min=args.wait_gpu_min)

    # exact inference.py setup (torchrun single process)
    gpu_id = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(gpu_id)
    device = torch.device("cuda", gpu_id)
    torch.distributed.init_process_group(backend="nccl", init_method="env://",
                                         timeout=datetime.timedelta(seconds=7200))

    from model.build_model import build_maskformer, load_checkpoint
    from model.text_encoder import Text_Encoder
    from data.inference_dataset import Inference_Dataset, collate_fn
    from evaluate.inference_engine import inference

    model = build_maskformer(args, device, gpu_id)
    text_encoder = Text_Encoder(
        text_encoder=args.text_encoder, checkpoint=args.text_encoder_checkpoint,
        partial_load=True, open_bert_layer=12, open_modality_embed=False,
        gpu_id=gpu_id, device=device,
    )
    model, _, _ = load_checkpoint(checkpoint=args.checkpoint, resume=False,
                                  partial_load=True, model=model, device=device)

    common.assert_gpu_free(min_gib=args.postload_free_gib)   # OOM headroom after load

    workdir = Path(args.workdir)
    img_dir = workdir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    for ci, case in enumerate(cases):
        # resume: all three arm masks exist -> skip
        if all((common.arm_dir("sat", t) / "preds" / f"{case['case']}.nii.gz").exists()
               for t in SAT_TERMS):
            print(f"[{ci + 1}/{len(cases)}] {case['case']} already done, skip", flush=True)
            continue

        t0 = time.time()
        # case image copy with <case>.nii.gz name -> SAT sample_id == case id
        case_nii = img_dir / f"{case['case']}.nii.gz"
        if not case_nii.exists():
            im = nib.load(str(case["image_path"]))
            im = nib.as_closest_canonical(im)
            nib.save(nib.Nifti1Image(np.asarray(im.dataobj, dtype=np.float32), im.affine),
                     str(case_nii))

        # per-case single-line jsonl (identical preprocessing to the demo)
        case_jsonl = workdir / f"{case['case']}.jsonl"
        with open(case_jsonl, "w") as f:
            f.write(json.dumps({
                "image": str(case_nii),
                "label": SAT_TERMS,
                "modality": MODALITY,
                "dataset": "mstt",
            }) + "\n")

        backmap, spaced_img = build_backmap(case_nii, orig_k=int(nib.load(str(case_nii)).shape[2]))

        testset = Inference_Dataset(str(case_jsonl), max_queries=args.max_queries,
                                    batch_size=args.batchsize_3d)
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(testset)
        testloader = DataLoader(testset, sampler=sampler, batch_size=1,
                                pin_memory=False, num_workers=0, collate_fn=collate_fn)  # README defaults

        torch.cuda.reset_peak_memory_stats()
        case_nib_dir = workdir / "raw_out" / case["case"]
        case_nib_dir.mkdir(parents=True, exist_ok=True)   # inference.py main() pre-creates rcd_dir
        try:
            inference(model=model, text_encoder=text_encoder, device=device,
                      testset=testset, testloader=testloader, nib_dir=str(case_nib_dir))
        except torch.cuda.OutOfMemoryError:
            print(f"[{ci + 1}/{len(cases)}] {case['case']} CUDA OOM; recording + continuing", flush=True)
            torch.cuda.empty_cache()
            for term in SAT_TERMS:
                common.append_runs_jsonl(common.arm_dir("sat", term), {
                    "case": case["case"], "prompt": term, "histotype": case["histotype"],
                    "sequence": case["sequence"], "status": "oom",
                    "model_inference_s": None, "case_wall_s": time.time() - t0,
                    "peak_vram_gib": None})
            continue
        infer_s = time.time() - t0
        vram_case = torch.cuda.max_memory_allocated() / 1024**3

        # map the saved per-class masks back to the case's 1 mm grid
        seg_dir = case_nib_dir / "mstt" / f"seg_{case['case']}"
        saved_img = case_nib_dir / "mstt" / f"img_{case['case']}.nii.gz"
        if saved_img.exists():
            theirs = np.asarray(nib.load(str(saved_img)).dataobj)
            mine = spaced_img   # recomputed cropped preprocessed image (H, W, D_sp)
            if theirs.shape != mine.shape or not np.allclose(theirs, mine, atol=1e-3):
                diff = float(np.abs(np.asarray(theirs, dtype=np.float64)
                                    - np.asarray(mine, dtype=np.float64)).max()) \
                    if theirs.shape == mine.shape else float("nan")
                print(f"WARNING {case['case']}: preprocessed-image mismatch "
                      f"(shape {theirs.shape} vs {mine.shape}, max abs diff {diff:.4g}); "
                      "back-map may be misaligned", file=sys.stderr)

        for term in SAT_TERMS:
            pred = np.asarray(nib.load(str(seg_dir / f"{term}.nii.gz")).dataobj).astype(np.uint8)
            out = backmap(pred)
            assert out.shape == tuple(nib.load(str(case_nii)).shape), (out.shape, "case grid")
            arm = common.arm_dir("sat", term)
            (arm / "preds").mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(out.astype(np.uint8), np.eye(4)),
                     str(arm / "preds" / f"{case['case']}.nii.gz"))
            common.append_runs_jsonl(arm, {
                "case": case["case"], "prompt": term, "histotype": case["histotype"],
                "sequence": case["sequence"],
                "n_pred_voxels": int(out.sum()),
                "model_inference_s": infer_s,
                "case_wall_s": time.time() - t0,
                "peak_vram_gib": vram_case,
            })
        print(f"[{ci + 1}/{len(cases)}] {case['case']} done in {infer_s:.1f}s "
              f"(peak {vram_case:.2f} GiB)", flush=True)

    print("DONE")


if __name__ == "__main__":
    main()
