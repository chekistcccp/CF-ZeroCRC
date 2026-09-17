#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def case_id(path: Path) -> str:
    return path.name[:-7] if path.name.endswith('.nii.gz') else path.stem


def bbox_from_mask(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    return np.concatenate([lo, hi])


def bbox_iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    a0, a1 = a[:3], a[3:]
    b0, b1 = b[:3], b[3:]
    inter0 = np.maximum(a0, b0)
    inter1 = np.minimum(a1, b1)
    inter = np.maximum(inter1 - inter0 + 1, 0)
    inter_v = float(np.prod(inter))
    va = float(np.prod(a1 - a0 + 1))
    vb = float(np.prod(b1 - b0 + 1))
    return inter_v / max(va + vb - inter_v, 1e-8)


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate CF-ZeroCRC maps against MSD Colon labels.')
    parser.add_argument('--pred-dir', required=True)
    parser.add_argument('--gt-dir', required=True)
    parser.add_argument('--output', default='outputs/msd_metrics.json')
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    gt_files = sorted(gt_dir.glob('*.nii.gz'))
    if not gt_files:
        raise SystemExit(f'No labels found in {gt_dir}')

    rows = []
    pooled_y = []
    pooled_s = []

    for gt_path in gt_files:
        cid = case_id(gt_path)
        heat_path = pred_dir / f'{cid}_heatmap.nii.gz'
        mask_path = pred_dir / f'{cid}_mask.nii.gz'
        if not heat_path.exists() or not mask_path.exists():
            continue

        gt = np.asarray(nib.load(str(gt_path)).get_fdata()) > 0
        heat = np.asarray(nib.load(str(heat_path)).get_fdata(), dtype=np.float32)
        pred = np.asarray(nib.load(str(mask_path)).get_fdata()) > 0
        if gt.shape != heat.shape:
            raise ValueError(f'Shape mismatch for {cid}: GT {gt.shape}, heatmap {heat.shape}')

        y = gt.reshape(-1).astype(np.uint8)
        s = heat.reshape(-1)
        pooled_y.append(y)
        pooled_s.append(s)

        auc = float(roc_auc_score(y, s)) if np.unique(y).size > 1 else float('nan')
        auprc = float(average_precision_score(y, s)) if y.sum() > 0 else float('nan')
        inter = float(np.logical_and(gt, pred).sum())
        dice = 2.0 * inter / max(float(gt.sum() + pred.sum()), 1e-8)
        max_idx = np.unravel_index(int(np.argmax(heat)), heat.shape)
        pointing = float(gt[max_idx])
        biou = bbox_iou(bbox_from_mask(pred), bbox_from_mask(gt))

        rows.append({'case': cid, 'auroc': auc, 'auprc': auprc, 'dice': dice, 'pointing': pointing, 'bbox_iou_3d': biou})

    if not rows:
        raise SystemExit('No prediction/label pairs found.')

    pooled_y_arr = np.concatenate(pooled_y)
    pooled_s_arr = np.concatenate(pooled_s)
    summary = {
        'n_cases': len(rows),
        'macro_auroc': float(np.nanmean([r['auroc'] for r in rows])),
        'macro_auprc': float(np.nanmean([r['auprc'] for r in rows])),
        'mean_dice': float(np.mean([r['dice'] for r in rows])),
        'pointing_accuracy': float(np.mean([r['pointing'] for r in rows])),
        'mean_bbox_iou_3d': float(np.mean([r['bbox_iou_3d'] for r in rows])),
        'pooled_auroc': float(roc_auc_score(pooled_y_arr, pooled_s_arr)),
        'pooled_auprc': float(average_precision_score(pooled_y_arr, pooled_s_arr)),
    }

    payload = {'summary': summary, 'cases': rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
