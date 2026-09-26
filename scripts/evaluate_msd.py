#!/usr/bin/env python
"""Evaluate complete MSD or MSI cohorts in prediction NIfTI space."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from sklearn.metrics import average_precision_score, roc_auc_score

from cfzerocrc.datasets import msi_pairs, nifti_files, nifti_id
from cfzerocrc.io import body_mask_2d


def aligned_array(path: Path, reference: nib.Nifti1Image, order: int) -> np.ndarray:
    image = nib.load(str(path))
    if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-4):
        image = resample_from_to(image, reference, order=order)
    return np.asarray(image.get_fdata(dtype=np.float32))


def bbox_from_mask(mask: np.ndarray) -> np.ndarray | None:
    coords = np.argwhere(mask)
    return np.concatenate([coords.min(axis=0), coords.max(axis=0)]) if coords.size else None


def bbox_iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    lo, hi = np.maximum(a[:3], b[:3]), np.minimum(a[3:], b[3:])
    intersection = float(np.prod(np.maximum(hi - lo + 1, 0)))
    va = float(np.prod(a[3:] - a[:3] + 1))
    vb = float(np.prod(b[3:] - b[:3] + 1))
    return intersection / max(va + vb - intersection, 1e-8)


def evaluate_case(case: str, image_path: Path, label_path: Path, pred_dir: Path) -> dict:
    heat_path = pred_dir / f'{case}_heatmap.nii.gz'
    mask_path = pred_dir / f'{case}_mask.nii.gz'
    heat_img = nib.load(str(heat_path))
    heat = np.asarray(heat_img.get_fdata(dtype=np.float32))
    pred = aligned_array(mask_path, heat_img, order=0) > 0.5
    raw_gt = nib.load(str(label_path))
    raw_count = int(np.count_nonzero(np.asanyarray(raw_gt.dataobj)))
    gt = aligned_array(label_path, heat_img, order=0) > 0.5
    if raw_count and not gt.any():
        raise ValueError(f'Label {label_path} became empty after affine alignment to {heat_path}')
    ct = aligned_array(image_path, heat_img, order=1)
    body = np.stack([body_mask_2d(ct[:, :, z]) for z in range(ct.shape[2])], axis=2).astype(bool)
    if np.any(gt & ~body):
        raise ValueError(f'Ground truth lies outside the body mask for {case}; check CT/label alignment')
    if not np.isfinite(heat).all() or not body.any():
        raise ValueError(f'Invalid heatmap or empty body mask for {case}')
    y = gt[body].astype(np.uint8)
    score = heat[body]
    auc = float(roc_auc_score(y, score)) if np.unique(y).size == 2 else None
    auprc = float(average_precision_score(y, score)) if y.any() else None
    overlap = int(np.logical_and(gt, pred).sum())
    dice = 2.0 * overlap / max(int(gt.sum() + pred.sum()), 1)
    pointing = int(gt[np.unravel_index(int(np.argmax(np.where(body, heat, -np.inf))), heat.shape)]) if np.any(score > 0) else 0
    return {'case': case, 'positive_voxels': int(gt.sum()), 'auroc': auc, 'auprc': auprc,
            'dice': dice, 'pointing': pointing, 'bbox_iou_3d': bbox_iou(bbox_from_mask(pred), bbox_from_mask(gt))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['msd', 'msi'], default='msd')
    parser.add_argument('--pred-dir', required=True)
    parser.add_argument('--gt-dir', required=True, help='MSD labelsTr, or MSI root containing data/ and data2/')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    pred_dir, gt_dir = Path(args.pred_dir), Path(args.gt_dir)
    manifest_path = pred_dir / 'manifest.json'
    if not manifest_path.exists():
        raise SystemExit(f'Missing inference manifest: {manifest_path}')
    manifest = {row['case']: row for row in json.loads(manifest_path.read_text(encoding='utf-8'))}
    if args.dataset == 'msi':
        expected = msi_pairs(gt_dir)
    else:
        labels = nifti_files(gt_dir)
        expected = [(nifti_id(label), Path(manifest[nifti_id(label)]['input']) if nifti_id(label) in manifest else None, label) for label in labels]
    if not expected:
        raise SystemExit(f'No labels found in {gt_dir}')
    missing = [case for case, _, _ in expected if case not in manifest or not (pred_dir / f'{case}_heatmap.nii.gz').exists() or not (pred_dir / f'{case}_mask.nii.gz').exists()]
    if missing:
        raise SystemExit(f'Missing predictions for {len(missing)} cases: {missing}')
    rows = []
    for case, image, label in expected:
        if image is None or not image.exists():
            raise FileNotFoundError(f'Missing CT for {case}: {image}')
        rows.append(evaluate_case(case, image, label, pred_dir))

    def mean(key: str) -> float | None:
        values = [row[key] for row in rows if row[key] is not None]
        return float(np.mean(values)) if values else None

    summary = {'n_cases': len(rows), 'n_positive_cases': sum(row['positive_voxels'] > 0 for row in rows),
               'macro_auroc': mean('auroc'), 'macro_auprc': mean('auprc'), 'mean_dice': mean('dice'),
               'pointing_accuracy': mean('pointing'), 'mean_bbox_iou_3d': mean('bbox_iou_3d')}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'summary': summary, 'cases': rows}, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
