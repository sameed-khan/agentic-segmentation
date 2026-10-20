"""Shared helpers for the probe3d 3D lesion-localizer runners (Phase 5 Study A).

Runner-side only: NO ground-truth access here (no manifest k_star, no mask files).
GT enters only through probe3d/score_probe3d.py, which resolves masks via
segd.data.load_mask.

Data convention (harness/server/src/segd/data.py): MSTT-199 isotropic NIfTI are
1 mm, identity affine, nominal RAS; arrays are indexed [i, j, k] after
nibabel as_closest_canonical; k is the axial axis (k_star in the prepare
manifest refers to mask[:, :, k]).
"""
from __future__ import annotations

import csv
import json
import os
import threading
import time
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[3]          # .../pi-worktree-*/ (this git worktree)
PROBE3D_ROOT = Path(__file__).resolve().parent          # harness/local_models/probe3d
RESULTS_ROOT = WORKTREE / "results" / "phase5" / "probe3d"

# Read-only inputs live in the main tree (this worktree has no data/ checkout).
DEFAULT_DATA_ROOT = Path("/home/ubuntu/agentic-segmentation/data")
DEFAULT_CASES_CSV = Path("/home/ubuntu/agentic-segmentation/eval/v2/mstt_cases.csv")

# Same mapping as segd.data.HISTOTYPE_TEXT (kept literal here so GPU envs do not
# need the segd package installed).
HISTOTYPE_TEXT = {
    "F": "fibrous soft-tissue tumor",
    "L": "lipomatous soft-tissue tumor",
    "M": "myxoid soft-tissue tumor",
    "N": "peripheral nerve sheath tumor",
    "V": "vascular soft-tissue tumor",
}


def read_design_cases(cases_csv: Path | None = None) -> list[dict]:
    """Design-split cases for the probe: case, histotype (letter), target text,
    sequence and absolute image path. No GT columns are read."""
    csv_path = Path(cases_csv or DEFAULT_CASES_CSV)
    out = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r["split"] != "design":
                continue
            out.append({
                "case": r["case"],
                "histotype": r["histotype"],
                "target_text": HISTOTYPE_TEXT[r["histotype"]],
                "sequence": r["sequence"],
                "image_path": Path(os.environ.get("SEGD_DATA_ROOT", str(DEFAULT_DATA_ROOT)))
                / "mstt199" / "image" / Path(r["image_kaggle"]).name,
            })
    out.sort(key=lambda c: c["case"])
    return out


def slugify(text: str) -> str:
    """Prompt text -> filesystem-safe arm suffix (spaces/underscores -> dash)."""
    return text.strip().lower().replace(" ", "-").replace("_", "-")


def arm_dir(model: str, prompt: str) -> Path:
    return RESULTS_ROOT / f"{model}-{slugify(prompt)}"


class VramSampler:
    """Sample the process's GPU memory (pynvml) in a background thread; peak in bytes."""

    def __init__(self, interval_s: float = 2.0):
        self.interval_s = interval_s
        self._peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self, pid: int, handle: int) -> None:
        import pynvml
        while not self._stop.is_set():
            try:
                used = pynvml.nvmlDeviceGetComputeRunningProcesses_v3(handle)
                for p in used:
                    if p.pid == pid:
                        self._peak = max(self._peak, p.usedGpuMemory or 0)
            except Exception:
                pass
            time.sleep(self.interval_s)

    def __enter__(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._pid = os.getpid()
            self._thread = threading.Thread(target=self._run, args=(self._pid, self._handle), daemon=True)
            self._thread.start()
        except Exception:
            self._thread = None
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        try:
            import pynvml
            pynvml.nvmlShutdown()
        except Exception:
            pass

    @property
    def peak_bytes(self) -> int:
        return self._peak


def gpu_free_bytes() -> int:
    """Free GPU memory in bytes via pynvml (fallback: torch)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        free = info.free
        pynvml.nvmlShutdown()
        return int(free)
    except Exception:
        import torch
        free, _total = torch.cuda.mem_get_info(0)
        return int(free)


def assert_gpu_free(min_gib: float = 22.0) -> None:
    free = gpu_free_bytes() / 1024**3
    if free < min_gib:
        raise RuntimeError(
            f"GPU free memory {free:.1f} GiB < required {min_gib} GiB "
            "(GPU is shared with running studies; aborting instead of disturbing them)."
        )


def wait_gpu_free(min_gib: float, timeout_min: float) -> None:
    """Poll until the GPU has >= min_gib free (task rule: >= 22 GiB before starting);
    abort politely on timeout. flock is NOT held while waiting — acquire the lock
    only once the check passes."""
    deadline = time.time() + timeout_min * 60
    while True:
        free = gpu_free_bytes() / 1024**3
        if free >= min_gib:
            print(f"GPU free memory {free:.1f} GiB >= {min_gib} GiB, proceeding")
            return
        if time.time() > deadline:
            raise RuntimeError(
                f"GPU free memory {free:.1f} GiB < required {min_gib} GiB after "
                f"{timeout_min:.0f} min of waiting (shared GPU stays busy)."
            )
        print(f"GPU free memory {free:.1f} GiB < {min_gib} GiB; waiting 60 s")
        time.sleep(60)


def append_runs_jsonl(arm: Path, row: dict) -> None:
    arm.mkdir(parents=True, exist_ok=True)
    (arm / "preds").mkdir(exist_ok=True)
    with open(arm / "runs.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def load_runs(arm: Path) -> list[dict]:
    path = arm / "runs.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]
