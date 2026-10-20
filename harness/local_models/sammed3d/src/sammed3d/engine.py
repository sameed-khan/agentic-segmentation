"""The real SAM-Med3D backend (torch): official checkpoint via medim, official
preprocessing, one patch forward per call (lazy import — ``--fake`` never loads torch).

Model: ``medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=...)``
(medim 0.1.2 registers SAM-Med3D as a ViT-B 3D SAM with image_size 128, patch 16;
the turbo checkpoint is the officially published ``sam_med3d_turbo.pth``, sha256
pinned in checkpoints/ and verified at load). The forward mirrors the official
``utils/infer_utils.py::sam_model_infer``: image_encoder on the 128^3 ROI, prompt
encoder on ``points=(coords (1,N,3), labels (1,N))`` + ``masks=prev_low_res`` (32^3
logits), mask decoder with ``multimask_output=False``, logits upsampled to 128^3 with
trilinear + align_corners=False. The final binary mask is ``sigmoid > 0.5``.

Determinism: eval mode, no dropout, torch.manual_seed(233) at load (official val
seed); no test-time tuning.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

CHECKPOINT_NAME = "sam_med3d_turbo.pth"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "checkpoints"
# pin: download via scripts/fetch_checkpoint.py; verified before loading
PINNED_SHA256 = "899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class SamMed3DEngine:
    kind = "real"

    def __init__(self, checkpoint: str | None = None, device: str | None = None):
        import torch
        import medim

        path = Path(checkpoint) if checkpoint else DEFAULT_CACHE_DIR / CHECKPOINT_NAME
        if not path.is_file():
            raise FileNotFoundError(
                f"SAM-Med3D checkpoint not found at {path}; run "
                "`uv run python scripts/fetch_checkpoint.py` in harness/local_models/sammed3d")
        got = _sha256(path)
        if got != PINNED_SHA256:
            raise RuntimeError(f"checkpoint sha256 mismatch: got {got}, pinned {PINNED_SHA256}")
        self.checkpoint_path = path
        torch.manual_seed(233)                     # official val seed; eval model
        self.torch = torch
        t0 = time.time()
        self.model = medim.create_model("SAM-Med3D", pretrained=True,
                                        checkpoint_path=str(path))
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.info = {"mode": "real", "checkpoint": path.name, "checkpoint_sha256": got,
                     "device": self.device, "load_s": round(time.time() - t0, 1),
                     "license": ("SAM-Med3D repo: Apache-2.0 (LICENSE); no separate weights "
                                 "licence stated in the repo README — check "
                                 "huggingface.co/blueyo0/SAM-Med3D before redistributing")}

    # ------------------------------------------------------------------ API
    def prepare(self, image: np.ndarray, spacing) -> np.ndarray:
        """Resample the session image to 1.5 mm isotropic (official step 1; the scipy
        implementation in geometry is bit-identical to torchio's Resample — verified)."""
        from sammed3d import geometry
        return geometry.resample_np(image, spacing)

    def predict_patch(self, patch_img, points_local, labels, mask_prompt) -> tuple:
        """One patch forward, exactly the official ``sam_model_infer`` per-click step.

        Returns (logits_128 float32, logits_lowres float32 32^3): the upsampled logits
        (trilinear, align_corners=False, as official) and the raw decoder output used
        as the next call's low-res mask prompt (official iterative carry-over).
        """
        import torch
        import torch.nn.functional as F
        from sammed3d import geometry

        model, device = self.model, self.device
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(patch_img, dtype=np.float32))
            x = x[None, None].to(device)                      # (1,1,128,128,128)
            image_embeddings = model.image_encoder(x)
            coords = torch.zeros(1, points_local.shape[0], 3, device=device)
            coords[0] = torch.as_tensor(np.asarray(points_local, dtype=np.float32),
                                        device=device) if points_local.shape[0] else coords[0]
            labels_t = torch.zeros(1, points_local.shape[0], dtype=torch.int64, device=device)
            if points_local.shape[0]:
                labels_t[0] = torch.as_tensor(np.asarray(labels, dtype=np.int64), device=device)
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=(coords, labels_t), boxes=None,
                masks=torch.as_tensor(np.asarray(mask_prompt, dtype=np.float32),
                                      device=device)[None, None],
            )
            low_res_masks, _ = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
            )
            # official: interpolate the low-res logits to the ROI size (trilinear)
            logits128 = F.interpolate(low_res_masks, size=x.shape[-3:], mode="trilinear",
                                      align_corners=False)
        return (logits128[0, 0].cpu().numpy().astype(np.float32),
                low_res_masks[0, 0].cpu().numpy().astype(np.float32))
