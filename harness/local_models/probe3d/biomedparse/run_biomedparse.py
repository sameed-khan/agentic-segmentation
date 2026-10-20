#!/usr/bin/env python
"""probe3d runner: BiomedParse v2 on the 30 MSTT-199 design cases (Study A probe).

Follows the BiomedParse v2 README / inference_example_3D pipeline exactly
(repo `microsoft/BiomedParse` branch v2 @ f571d5ccaa4f8c0479bb89e390a7ef0dfc0736f4):
  - 3D volume -> process_input(512) (pad to square + bicubic resize per slice)
  - prompts joined with "[SEP]" in ONE forward per case (README multi-prompt pattern)
  - model(input, mode="eval", slice_batch_size=BS)
  - pred_gmasks -> bicubic interpolate to 512 -> postprocess -> process_output back to
    the original voxel grid.

Declared deviations (recorded in the arm reports):
  - postprocess(..., do_nms=False): the README's per-slice NMS resolves
    mutually-exclusive challenge classes; our prompts are alternative phrasings of the
    same target scored independently, and NMS would suppress overlapping tumour
    classes (e.g. `tumor` vs `whole tumor`). The object-existence gate (> 0.5) is kept.
  - Raw-NIfTI intensity preprocessing (the README states raw intensities need
    modality-appropriate preprocessing and does not ship a 3D NIfTI converter): the
    repo's own MRI recipe from inference_utils/processing_utils.process_intensity_image
    applied volumetrically - clip to the [0.5, 99.5] percentile over nonzero voxels,
    min-max scale to [0, 255], uint8.

Inputs (read-only): the case NIfTIs under $SEGD_DATA_ROOT (default
/home/ubuntu/agentic-segmentation/data), case list from eval/v2/mstt_cases.csv
(split == design). NO ground truth is read here.

Outputs: results/phase5/probe3d/biomedparse-<prompt>/ in THIS worktree:
  preds/<case>.nii.gz  binary mask, canonical 1 mm grid ([i, j, k] = loaded NIfTI)
  runs.jsonl           per-case row: prompt, existence scores, runtime, VRAM
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import torch

BP_REPO = Path("/home/ubuntu/BiomedParse")   # pinned v2 checkout (see README)
sys.path.insert(0, str(BP_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # probe3d/common.py

import common  # noqa: E402

# Primary prompt first; secondary histotype text is added per case; then the
# lesion-like MRI class names from the README / HF card data spec.
FIXED_PROMPTS = [
    "tumor",                        # primary
    "tumor core",                   # HF-card MRI-FLAIR brain / MRI-T1-Gd
    "whole tumor",                  # HF-card MRI-FLAIR brain
    "enhancing tumor",              # HF-card MRI-T1-Gd brain
    "lower-grade glioma",           # HF-card MRI-FLAIR brain
    "edema",                        # HF-card MRI-FLAIR brain
    "brain tumor",                  # README v2 supported tasks (MRI neuro)
    "stroke lesion",                # README v2 supported tasks (MRI neuro)
    "vestibular schwannoma",        # README v2 supported tasks (MRI neuro)
]


def preprocess_mri_uint8(vol: np.ndarray) -> np.ndarray:
    """Repo MRI recipe (processing_utils.process_intensity_image, non-CT branch),
    applied to the whole 3D volume: percentile clip over nonzero voxels -> [0,255]."""
    nz = vol[vol > 0]
    lo, hi = np.percentile(nz, 0.5), np.percentile(nz, 99.5)
    v = np.clip(vol, lo, hi)
    v = (v - v.min()) / (v.max() - v.min()) * 255.0
    return v.astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(Path.home() / "models/BiomedParse/biomedparse_v2.ckpt"))
    ap.add_argument("--slice-batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only first N cases (smoke test)")
    ap.add_argument("--cases-csv", default=str(common.DEFAULT_CASES_CSV))
    ap.add_argument("--min-free-gib", type=float, default=26.0,
                    help="free GiB required BEFORE model load (so the 22 GiB task rule "
                         "still holds after weights/contexts)")
    ap.add_argument("--wait-gpu-min", type=float, default=180.0)
    args = ap.parse_args()

    cases = common.read_design_cases(Path(args.cases_csv))
    if args.limit:
        cases = cases[: args.limit]
    print(f"{len(cases)} design cases")

    # wait for a shared-GPU window BEFORE taking the lock / loading the model
    common.wait_gpu_free(min_gib=args.min_free_gib, timeout_min=args.wait_gpu_min)

    # ---- model (README example path) ---------------------------------------
    from hydra import compose
    from hydra.core.global_hydra import GlobalHydra
    import hydra
    from huggingface_hub import hf_hub_download  # noqa: F401  (README import surface)
    from utils import process_input, process_output
    from inference import postprocess

    GlobalHydra.instance().clear()
    hydra.initialize_config_dir(
        config_dir=str(BP_REPO / "configs" / "model"), version_base=None, job_name="probe3d"
    )
    cfg = compose(config_name="biomedparse_3D")
    model = hydra.utils.instantiate(cfg, _convert_="object")
    ckpt = args.checkpoint
    if not Path(ckpt).exists():
        ckpt = hf_hub_download("microsoft/BiomedParse", "biomedparse_v2.ckpt")
    missing = model.load_pretrained(ckpt)
    if missing.missing_keys or missing.unexpected_keys:
        print(f"WARNING load_pretrained: missing={len(missing.missing_keys)} "
              f"unexpected={len(missing.unexpected_keys)}", file=sys.stderr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    common.assert_gpu_free(min_gib=22.0)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    sampler = common.VramSampler()
    with sampler:
        for ci, case in enumerate(cases):
            prompts = ["tumor", case["target_text"]] + FIXED_PROMPTS[1:]
            arm_of = {p: common.arm_dir("biomedparse", p) for p in set(prompts)}

            # resume: skip cases whose every arm mask already exists
            if all((a / "preds" / f"{case['case']}.nii.gz").exists() for a in arm_of.values()):
                print(f"[{ci + 1}/{len(cases)}] {case['case']} already done, skip")
                continue

            t0 = time.time()
            img = nib.load(str(case["image_path"]))
            img = nib.as_closest_canonical(img)
            vol = np.asarray(img.dataobj, dtype=np.float32)
            vol_u8 = preprocess_mri_uint8(vol)

            text = "[SEP]".join(prompts)                      # README: multi-prompt joint text
            imgs, pad_width, padded_size, valid_axis = process_input(vol_u8, 512)
            imgs = imgs.to(device).int()                       # README: int cast
            input_tensor = {"image": imgs.unsqueeze(0), "text": [text]}

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                output = model(input_tensor, mode="eval", slice_batch_size=args.slice_batch_size)
            model_s = float(output["predictions"].get("inference_time", float("nan")))

            import torch.nn.functional as F
            mask_preds = output["predictions"]["pred_gmasks"]
            existence = output["predictions"]["object_existence"]        # (P, D) logits
            mask_preds = F.interpolate(mask_preds, size=(512, 512), mode="bicubic",
                                       align_corners=False, antialias=True)
            # do_nms=False: prompts are alternative phrasings of one target, not
            # mutually-exclusive classes (see module docstring).
            mask_preds = postprocess(mask_preds, existence, do_nms=False)  # (P, D, 512, 512) {0,1}

            ex_sig = torch.sigmoid(existence).float().cpu().numpy()
            mask_preds_np = mask_preds.cpu().numpy()
            vram_case = (torch.cuda.max_memory_allocated() / 1024**3) if device.type == "cuda" else 0.0

            for pi, prompt in enumerate(prompts):
                class_mask = mask_preds_np[pi]                 # (D, 512, 512)
                out = process_output(torch.from_numpy(class_mask), pad_width, padded_size, valid_axis)
                out = np.asarray(out).astype(np.uint8)         # back in [i, j, k] grid
                assert out.shape == vol.shape, (out.shape, vol.shape)
                arm = arm_of[prompt]
                (arm / "preds").mkdir(parents=True, exist_ok=True)
                nib.save(nib.Nifti1Image(out, np.eye(4)),
                         str(arm / "preds" / f"{case['case']}.nii.gz"))
                common.append_runs_jsonl(arm, {
                    "case": case["case"], "prompt": prompt, "histotype": case["histotype"],
                    "sequence": case["sequence"],
                    "n_pred_voxels": int(out.sum()),
                    "existence_prob_max": float(ex_sig[pi].max()),
                    "existence_prob_mean": float(ex_sig[pi].mean()),
                    "model_inference_s": model_s,
                    "case_wall_s": time.time() - t0,
                    "peak_vram_gib": vram_case,
                })
            print(f"[{ci + 1}/{len(cases)}] {case['case']} done in {time.time() - t0:.1f}s "
                  f"(model {model_s:.1f}s, peak {vram_case:.2f} GiB)")

    print("DONE")


if __name__ == "__main__":
    main()
