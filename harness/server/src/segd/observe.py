"""Compact agent-facing observations (CONTRACTS §4): mask statistics, budget counters,
version line. Observation text never contains voxel or mm coordinates (slice indices
are allowed: they are navigation, not pointing)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from segd.case_state import OpError, vid_label


def mask_stats(mask: np.ndarray, spacing: tuple) -> dict:
    """Volume (mL), component count, largest-component fraction, slice extent per axis.

    ``component_ml`` (Phase 4): volumes of up to the 10 largest connected components,
    mL, descending — derived from the CURRENT mask only (never GT). Empty mask: []."""
    m = mask > 0
    nvox = int(m.sum())
    out = {"volume_ml": 0.0, "n_components": 0, "largest_frac": 0.0, "extent": None,
           "component_ml": []}
    if nvox == 0:
        return out
    voxel_mm3 = float(np.prod(spacing))
    out["volume_ml"] = nvox * voxel_mm3 / 1000.0
    labels, n = ndimage.label(m, structure=ndimage.generate_binary_structure(3, 1))
    out["n_components"] = int(n)
    sizes = np.bincount(labels.ravel())
    if n:
        out["largest_frac"] = float(sizes[1:].max() / nvox)
        ml = (sizes[1:].astype(np.float64) * voxel_mm3 / 1000.0)
        ml.sort()
        out["component_ml"] = [round(float(v), 3) for v in ml[::-1][:10]]
    extent = {}
    for axis_name, a in (("sagittal", 0), ("coronal", 1), ("axial", 2)):
        idx = np.nonzero(m.any(axis=tuple(i for i in range(3) if i != a)))[0]
        extent[axis_name] = (int(idx.min()), int(idx.max()))
    out["extent"] = extent
    return out


def is_multi_lesion(cs) -> bool:
    """True on Phase 4 (non-MSTT) datasets: one multi-component target object holding ALL
    lesions. Gates every dataset-aware observation/review change; 'mstt' (and any CaseState
    without the attribute, e.g. the pre-dataset reference tree) keeps byte-identical text."""
    return getattr(cs, "dataset", "mstt") != "mstt"


def _fmt_ml(v: float) -> str:
    if v >= 10:
        return f"{v:.1f}"
    if v >= 0.1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def components_text_multi(st: dict) -> str:
    """Phase 4 components line: component count + volumes of up to the 10 largest
    components (mask_stats.component_ml, current mask only — never the GT lesion count).
    'components 4 (mL: 2.10, 0.60, 0.12, 0.05)'; beyond 10: '; N smaller'."""
    n = st["n_components"]
    if n == 0:
        return "components 0"
    vols = st.get("component_ml") or []
    tail = f"; {n - len(vols)} smaller" if n > len(vols) else ""
    return f"components {n} (mL: {', '.join(_fmt_ml(v) for v in vols)}{tail})"


def components_text_for(cs, st: dict) -> str:
    """components_text on MSTT (byte-identical), components_text_multi on Phase 4."""
    return components_text_multi(st) if is_multi_lesion(cs) else components_text(st)


def components_text(st: dict) -> str:
    """'components 12 (largest 99.6%; 11 others total 2.3 mL)'. The percentage has one
    decimal and is never rounded up to 100.0 while other components exist."""
    n = st["n_components"]
    if n == 0:
        return "components 0"
    pct = 100 * st["largest_frac"]
    if n == 1:
        return f"components 1 (largest {pct:.1f}%)"
    pct = min(round(pct, 1), 99.9)
    other_ml = st["volume_ml"] * (1.0 - st["largest_frac"])
    others = "1 other totals" if n == 2 else f"{n - 1} others total"
    return f"components {n} (largest {pct:.1f}%; {others} {other_ml:.1f} mL)"


def _fmt_extent(extent) -> str:
    if extent is None:
        return "extent empty"
    return (f"extent axial {extent['axial'][0]}\u2013{extent['axial'][1]}, "
            f"coronal {extent['coronal'][0]}\u2013{extent['coronal'][1]}, "
            f"sagittal {extent['sagittal'][0]}\u2013{extent['sagittal'][1]}")


def budget_line(cs) -> str:
    """Budget counters shown in EVERY observation and agent-facing error text (decision
    on finding #3): 'tool calls 12/40, prompts 4/15'."""
    # shown counts never exceed the maximum: the final seg_finish is allowed after the
    # budget is exhausted but must not display 41/40
    tc = min(cs.tool_calls, cs.budgets['max_tool_calls'])
    mp = min(cs.model_prompts, cs.budgets['max_model_prompts'])
    return f"tool calls {tc}/{cs.budgets['max_tool_calls']}, " \
           f"prompts {mp}/{cs.budgets['max_model_prompts']}"


def _observation_body(cs, new: np.ndarray, parent: np.ndarray) -> str:
    """Volume/components/extent/change line shared by mutation and restore observations."""
    st = mask_stats(new, cs.spacing)
    added_vox = int(((new > 0) & ~(parent > 0)).sum())
    removed_vox = int(((parent > 0) & ~(new > 0)).sum())
    mm3 = float(np.prod(cs.spacing)) / 1000.0
    if st["volume_ml"] == 0.0 and added_vox == 0 and removed_vox == 0:
        comp_txt = "components 0"
    else:
        comp_txt = components_text_for(cs, st)
    if added_vox == 0 and removed_vox == 0:
        change_txt = "change none"
    else:
        add_txt = f"{added_vox * mm3:+.1f} mL"
        rem_txt = f"{-removed_vox * mm3:+.1f} mL" if removed_vox else "0.0 mL"
        change_txt = f"change {add_txt} / {rem_txt}"
    return f"volume {st['volume_ml']:.1f} mL | {comp_txt} | {_fmt_extent(st['extent'])} | {change_txt}"


def observation_after_mutation(cs, parent_vid: int, new_vid: int) -> str:
    """The compact per-mutation observation line (CONTRACTS §4), e.g.

    version v5 (parent v4) | volume 123.4 mL | components 2 (largest 98.0%; 1 other totals 2.5 mL) |
    extent axial 101–163, ... | change +12.3 mL / −0.4 mL | tool calls 9/40, prompts 4/15
    """
    parent = cs.version_mask(parent_vid) if parent_vid != new_vid else np.zeros(cs.shape, np.uint8)
    new = cs.version_mask(new_vid)
    return (f"version {vid_label(new_vid)} (parent {vid_label(parent_vid)}) | "
            f"{_observation_body(cs, new, parent)} | {budget_line(cs)}")


def observation_after_restore(cs, restored_vid: int, was_vid: int) -> str:
    """Undo/checkout observation (finding #8): 'version v0 (restored; was v1)'. The
    change is measured against the version that was CURRENT before the restore, so
    re-checking out the current version reports 'change none' instead of the whole mask
    as added."""
    new = cs.version_mask(restored_vid)
    was = cs.version_mask(was_vid)
    return (f"version {vid_label(restored_vid)} (restored; was {vid_label(was_vid)}) | "
            f"{_observation_body(cs, new, was)} | {budget_line(cs)}")


def observation_for_view(cs, axis: str, slice_idx: int, crop: str, overlay: str, marks_level: str) -> str:
    """Header line for non-mutating navigation ops (sheet/view/overview)."""
    from segd import config
    return (f"{axis} {slice_idx}/{cs.shape[config.FIXED_AXIS[axis]]} | crop {crop} | "
            f"marks {marks_level} | overlay {overlay} | {vid_label(cs.current)} | {budget_line(cs)}")


def require_nonempty(mask: np.ndarray, what: str) -> None:
    if not (mask > 0).any():
        raise OpError(f"{what}: current mask is empty")
