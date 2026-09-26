import io
import tarfile

import pytest

from cfzerocrc.datasets import msi_pairs, prepare_msd


def test_msi_pairing_uses_numeric_prefix_and_fixed_directories(tmp_path):
    images = tmp_path / 'data'
    labels = tmp_path / 'data2'
    images.mkdir()
    labels.mkdir()
    (images / '10001_men_jing_mai(0.5mm_low)_20180808084217.nii.gz').touch()
    (labels / '10001.nii.gz').touch()
    assert [(case, image.name, label.name) for case, image, label in msi_pairs(tmp_path)] == [
        ('10001', '10001_men_jing_mai(0.5mm_low)_20180808084217.nii.gz', '10001.nii.gz')
    ]


def test_msi_pairing_fails_on_missing_label(tmp_path):
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data2').mkdir()
    (tmp_path / 'data' / '10001_scan.nii.gz').touch()
    with pytest.raises(ValueError, match='missing labels'):
        msi_pairs(tmp_path)


def test_prepare_msd_extracts_tar(tmp_path):
    archive = tmp_path / 'Task10_Colon.tar'
    with tarfile.open(archive, 'w') as tar:
        for name in ['Task10_Colon/imagesTr/colon_001.nii.gz', 'Task10_Colon/labelsTr/colon_001.nii.gz']:
            payload = b'example'
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    images, labels = prepare_msd(tmp_path)
    assert (images / 'colon_001.nii.gz').read_bytes() == b'example'
    assert (labels / 'colon_001.nii.gz').read_bytes() == b'example'


def test_prepare_msd_rejects_path_traversal(tmp_path):
    archive = tmp_path / 'Task10_Colon.tar'
    with tarfile.open(archive, 'w') as tar:
        info = tarfile.TarInfo('../outside')
        info.size = 1
        tar.addfile(info, io.BytesIO(b'x'))
    with pytest.raises(ValueError, match='Unsafe tar member'):
        prepare_msd(tmp_path)
