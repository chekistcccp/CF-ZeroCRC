import numpy as np
import nibabel as nib

from scripts.evaluate_msd import evaluate_case


def test_evaluation_aligns_same_shape_flipped_affine(tmp_path):
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    affine[0, 3] = 11.0
    ct = np.full((12, 12, 5), 50, dtype=np.int16)
    label = np.zeros_like(ct, dtype=np.uint8)
    label[3:5, 5:7, 2:4] = 1
    image_path = tmp_path / 'image.nii.gz'
    label_path = tmp_path / 'label.nii.gz'
    nib.save(nib.Nifti1Image(ct, affine), image_path)
    nib.save(nib.Nifti1Image(label, affine), label_path)
    canonical_label = nib.as_closest_canonical(nib.load(label_path))
    aligned = np.asarray(canonical_label.dataobj)
    heat = aligned.astype(np.float32)
    nib.save(nib.Nifti1Image(heat, canonical_label.affine), tmp_path / 'case_heatmap.nii.gz')
    nib.save(nib.Nifti1Image(aligned, canonical_label.affine), tmp_path / 'case_mask.nii.gz')
    result = evaluate_case('case', image_path, label_path, tmp_path)
    assert result['dice'] == 1.0
    assert result['pointing'] == 1
    assert result['auroc'] == 1.0
