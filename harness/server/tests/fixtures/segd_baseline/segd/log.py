"""Per-case artifact logging: trajectory.jsonl, renders/, versions.npz, case_meta.json.

Layout under ``results/<run_id>/<case>/`` (CONTRACTS §4/§5.7). Pure filesystem
helpers; the segd core (case_state/ops) owns the content that goes into records.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int(time.time() * 1000) % 1000:03d}Z"


def _json_default(o):
    """JSON encoder for numpy scalars/arrays (e.g. candidate 'how' metadata)."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class TrajectoryLogger:
    """One logger per open case. All writes are small appends; nothing is buffered."""

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.renders_dir = self.out_dir / "renders"
        self.renders_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_path = self.out_dir / "trajectory.jsonl"
        self._render_seq = 0

    # ---------------------------------------------------------------- renders
    def save_render(self, image, op: str) -> str:
        """Save a PIL image as ``renders/NNN_<op>.png``; returns the relative path."""
        self._render_seq += 1
        rel = f"renders/{self._render_seq:03d}_{op}.png"
        image.save(self.out_dir / rel)
        return rel

    # ------------------------------------------------------------ trajectory
    def record(
        self,
        op: str,
        args: dict,
        resolved: list | None = None,
        timings: dict | None = None,
        parent: str | None = None,
        child: str | None = None,
        observation: str | None = None,
        renders: list[str] | None = None,
        error: str | None = None,
        error_kind: str | None = None,
        extra: dict | None = None,
    ) -> None:
        rec = {
            "ts": _now_iso(),
            "op": op,
            "args": args,
            "resolved": resolved,        # resolved voxel coordinates ([i, j, k]); never shown to the agent
            "timings": timings or {},
            "parent": parent,
            "child": child,
            "observation": observation,
            "renders": renders or [],
            "error": error,
            # "op" = agent-facing error, "budget" = rejected call (exhausted budget),
            # "internal" = unexpected exception (error carries the full traceback) so
            # scoring can tag `harness` (reviewer finding #2).
            "error_kind": error_kind,
        }
        if extra:
            rec.update(extra)             # op-specific fields (e.g. candidates, accepted ids)
        with open(self.trajectory_path, "a") as f:
            f.write(json.dumps(rec, separators=(",", ":"), default=_json_default) + "\n")

    # -------------------------------------------------------------- versions
    def flush_versions(self, records: list[dict], shape) -> None:
        """Write versions.npz (compressed) from CaseState version records.

        Each record is a dict with keys: id (int), parent (int | None), bbox ((3,2) half-open or
        None for an empty mask), packed (bytes, np.packbits of the bbox-cropped mask), nvox (int).
        """
        arrays: dict[str, np.ndarray] = {
            "shape": np.asarray(shape, dtype=np.int64),
            "ids": np.asarray([r["id"] for r in records], dtype=np.int64),
            "parents": np.asarray([-1 if r["parent"] is None else r["parent"] for r in records], dtype=np.int64),
        }
        for r in records:
            bb = np.full((3, 2), -1, dtype=np.int64) if r["bbox"] is None else np.asarray(r["bbox"], dtype=np.int64)
            arrays[f"v{r['id']}_bbox"] = bb
            arrays[f"v{r['id']}_nvox"] = np.asarray(r["nvox"], dtype=np.int64)
            arrays[f"v{r['id']}_packed"] = np.frombuffer(r["packed"], dtype=np.uint8)
        np.savez_compressed(self.out_dir / "versions.npz", **arrays)

    # ------------------------------------------------------------------ meta
    def write_case_meta(self, meta: dict) -> None:
        with open(self.out_dir / "case_meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
