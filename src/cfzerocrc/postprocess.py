from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import ndimage as ndi


@dataclass
class Candidate3D:
    label: int
    voxel_count: int
    z_start: int
    z_end: int
    bbox_xyzxyz: list[int]
    mean_score: float
    max_score: float

    def to_dict(self) -> dict:
        return asdict(self)


def robust_normalize(x: np.ndarray, mask: np.ndarray | None = None, q_low: float = 1.0, q_high: float = 99.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    values = x[mask > 0] if mask is not None and np.any(mask) else x.reshape(-1)
    if values.size == 0:
        return np.zeros_like(x)
    lo, hi = np.percentile(values, [q_low, q_high])
    if hi <= lo + 1e-8:
        return np.zeros_like(x)
    out = (x - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def top_fraction_mean(x: np.ndarray, fraction: float = 0.05, mask: np.ndarray | None = None) -> float:
    values = x[mask > 0] if mask is not None and np.any(mask) else x.reshape(-1)
    if values.size == 0:
        return 0.0
    k = max(1, int(np.ceil(values.size * fraction)))
    idx = np.argpartition(values, -k)[-k:]
    return float(values[idx].mean())


def threshold_map(x: np.ndarray, mask: np.ndarray | None, method: str, mad_k: float, percentile: float) -> np.ndarray:
    valid = x[mask > 0] if mask is not None and np.any(mask) else x.reshape(-1)
    if valid.size == 0:
        return np.zeros_like(x, dtype=np.uint8)

    if method == "mad":
        med = float(np.median(valid))
        mad = float(np.median(np.abs(valid - med))) + 1e-8
        threshold = med + mad_k * 1.4826 * mad
    elif method == "percentile":
        threshold = float(np.percentile(valid, percentile))
    else:
        raise ValueError(f"Unknown threshold method: {method}")

    out = x > threshold
    if mask is not None:
        out &= mask.astype(bool)
    return out.astype(np.uint8)


def clean_mask_2d(mask: np.ndarray, min_area: int = 24, closing_iterations: int = 1) -> np.ndarray:
    mask = mask.astype(bool)
    if closing_iterations > 0:
        mask = ndi.binary_closing(mask, iterations=closing_iterations)
    mask = ndi.binary_fill_holes(mask)
    labels, n = ndi.label(mask)
    if n == 0:
        return mask.astype(np.uint8)
    sizes = ndi.sum(mask, labels, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes >= min_area
    return keep[labels].astype(np.uint8)


def smooth_z(volume: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    if sigma <= 0:
        return volume.astype(np.float32)
    return ndi.gaussian_filter1d(volume.astype(np.float32), sigma=sigma, axis=2, mode="nearest")


def connected_components_3d(mask: np.ndarray, heatmap: np.ndarray, min_voxels: int = 100) -> tuple[np.ndarray, list[Candidate3D]]:
    structure = ndi.generate_binary_structure(3, 2)
    labels, n = ndi.label(mask.astype(bool), structure=structure)
    kept = np.zeros_like(labels, dtype=np.int32)
    candidates: list[Candidate3D] = []
    new_label = 0

    for old_label in range(1, n + 1):
        coords = np.argwhere(labels == old_label)
        if coords.shape[0] < min_voxels:
            continue
        new_label += 1
        kept[labels == old_label] = new_label
        x0, y0, z0 = coords.min(axis=0)
        x1, y1, z1 = coords.max(axis=0)
        scores = heatmap[labels == old_label]
        candidates.append(
            Candidate3D(
                label=new_label,
                voxel_count=int(coords.shape[0]),
                z_start=int(z0),
                z_end=int(z1),
                bbox_xyzxyz=[int(x0), int(y0), int(z0), int(x1), int(y1), int(z1)],
                mean_score=float(scores.mean()),
                max_score=float(scores.max()),
            )
        )
    candidates.sort(key=lambda c: c.max_score, reverse=True)
    return kept, candidates
