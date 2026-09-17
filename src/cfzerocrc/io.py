from __future__ import annotations

from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def load_nifti(path: str | Path) -> Tuple[nib.Nifti1Image, np.ndarray]:
    """Load a CT volume in closest-canonical orientation as float32 HU."""
    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume, got shape {data.shape} from {path}")
    return img, data


def save_nifti_like(reference: nib.Nifti1Image, data: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = nib.Nifti1Image(data, affine=reference.affine, header=reference.header.copy())
    nib.save(out, str(path))


def window_ct(slice_hu: np.ndarray, low: float = -160.0, high: float = 240.0) -> np.ndarray:
    x = np.clip(slice_hu.astype(np.float32), low, high)
    return (x - low) / max(high - low, 1e-6)


def to_rgb_pil(slice_hu: np.ndarray, resolution: int, low: float, high: float) -> Image.Image:
    x = window_ct(slice_hu, low=low, high=high)
    x = (x * 255.0).round().clip(0, 255).astype(np.uint8)
    rgb = np.repeat(x[..., None], 3, axis=-1)
    image = Image.fromarray(rgb, mode="RGB")
    if image.size != (resolution, resolution):
        image = image.resize((resolution, resolution), Image.Resampling.BILINEAR)
    return image


def body_mask_2d(slice_hu: np.ndarray, threshold_hu: float = -500.0) -> np.ndarray:
    """Simple label-free body mask: threshold, cleanup, keep largest component."""
    mask = np.asarray(slice_hu > threshold_hu, dtype=bool)
    mask = ndi.binary_fill_holes(mask)
    mask = ndi.binary_opening(mask, iterations=1)
    labels, n = ndi.label(mask)
    if n == 0:
        return mask.astype(np.uint8)
    sizes = ndi.sum(mask, labels, index=np.arange(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    mask = labels == keep
    mask = ndi.binary_closing(mask, iterations=2)
    mask = ndi.binary_fill_holes(mask)
    return mask.astype(np.uint8)


def resize_float_map(arr: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    image = Image.fromarray(arr.astype(np.float32), mode="F")
    image = image.resize((w, h), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)
