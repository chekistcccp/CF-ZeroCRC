import numpy as np

from cfzerocrc.postprocess import connected_components_3d, robust_normalize, top_fraction_mean


def test_robust_normalize_range():
    x = np.arange(100, dtype=np.float32).reshape(10, 10)
    y = robust_normalize(x)
    assert y.min() >= 0.0
    assert y.max() <= 1.0


def test_top_fraction_mean():
    x = np.arange(100, dtype=np.float32).reshape(10, 10)
    score = top_fraction_mean(x, 0.1)
    assert score > 80


def test_connected_components_3d_filters_small_component():
    mask = np.zeros((20, 20, 10), dtype=np.uint8)
    mask[2:8, 2:8, 2:7] = 1
    mask[15, 15, 1] = 1
    heat = mask.astype(np.float32)
    labels, candidates = connected_components_3d(mask, heat, min_voxels=20)
    assert len(candidates) == 1
    assert labels.max() == 1
