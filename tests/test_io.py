import numpy as np

from cfzerocrc.io import resize_float_map


def test_resized_map_is_writable_for_in_place_body_masking():
    source = np.arange(16, dtype=np.float32).reshape(4, 4)
    resized = resize_float_map(source, (8, 8))
    assert resized.shape == (8, 8)
    assert resized.dtype == np.float32
    assert resized.flags.writeable
    resized *= np.ones((8, 8), dtype=np.uint8)
