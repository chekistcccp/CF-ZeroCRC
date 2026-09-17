from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .io import body_mask_2d, load_nifti, resize_float_map, save_nifti_like, to_rgb_pil
from .model import SD35CounterfactualEngine
from .postprocess import clean_mask_2d, connected_components_3d, robust_normalize, smooth_z, threshold_map, top_fraction_mean


def _case_id(path: Path) -> str:
    name = path.name
    if name.endswith('.nii.gz'):
        return name[:-7]
    return path.stem


def list_nifti(input_path: str | Path) -> list[Path]:
    p = Path(input_path)
    if p.is_file():
        return [p]
    files = sorted(list(p.rglob('*.nii.gz')) + list(p.rglob('*.nii')))
    return list(dict.fromkeys(files))


def _select_candidates(scores: dict[int, float], n_slices: int, cfg: dict[str, Any]) -> list[int]:
    if not scores:
        return []
    idxs = np.array(sorted(scores), dtype=int)
    vals = np.array([scores[i] for i in idxs], dtype=np.float32)
    threshold = np.percentile(vals, float(cfg['candidate_percentile']))
    selected = set(int(i) for i, v in zip(idxs, vals) if v >= threshold)
    min_count = int(cfg.get('min_candidate_slices', 12))
    if len(selected) < min_count:
        order = idxs[np.argsort(vals)[::-1]]
        selected.update(int(i) for i in order[:min(min_count, len(order))])
    expand = int(cfg.get('expand_slices', 3))
    expanded: set[int] = set()
    for i in selected:
        for j in range(max(0, i - expand), min(n_slices, i + expand + 1)):
            expanded.add(j)
    return sorted(expanded)


def process_case(path: str | Path, engine: SD35CounterfactualEngine, cfg: dict[str, Any], output_dir: str | Path, mode: str = 'coarse_fine') -> dict[str, Any]:
    path = Path(path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    case = _case_id(path)
    ref_img, volume = load_nifti(path)
    h, w, n_slices = volume.shape
    low = float(cfg['ct']['window_low'])
    high = float(cfg['ct']['window_high'])
    body_thr = float(cfg['ct'].get('body_threshold_hu', -500.0))

    body = np.zeros((h, w, n_slices), dtype=np.uint8)
    for z in range(n_slices):
        body[:, :, z] = body_mask_2d(volume[:, :, z], body_thr)

    coarse_scores: dict[int, float] = {}
    if mode == 'coarse_fine':
        coarse_cfg = cfg['coarse']
        resolution = int(coarse_cfg['resolution'])
        stride = int(coarse_cfg['stride_z'])
        seed = int(cfg['flow']['coarse_seed'])
        levels = [float(x) for x in cfg['flow']['coarse_noise_levels']]
        top_fraction = float(coarse_cfg.get('top_fraction_for_score', 0.05))
        for z in range(0, n_slices, stride):
            if body[:, :, z].sum() < 64:
                continue
            pil = to_rgb_pil(volume[:, :, z], resolution, low, high)
            m = engine.flow_map(pil, resolution, levels, seed)
            body_small = resize_float_map(body[:, :, z].astype(np.float32), (resolution, resolution)) > 0.5
            m = m * body_small
            coarse_scores[z] = top_fraction_mean(m, top_fraction, body_small)
        candidate_slices = _select_candidates(coarse_scores, n_slices, cfg['coarse'])
    elif mode == 'exhaustive':
        candidate_slices = [z for z in range(n_slices) if body[:, :, z].sum() >= 64]
    else:
        raise ValueError("mode must be 'coarse_fine' or 'exhaustive'")

    heatmap = np.zeros((h, w, n_slices), dtype=np.float32)
    fine_scores: dict[int, float] = {}
    fine_resolution = int(cfg['fine']['resolution'])
    top_fraction = float(cfg['coarse'].get('top_fraction_for_score', 0.05))

    for z in candidate_slices:
        pil = to_rgb_pil(volume[:, :, z], fine_resolution, low, high)
        m, _ = engine.fine_map(pil)
        m = resize_float_map(m, (h, w))
        m *= body[:, :, z]
        m = robust_normalize(m, body[:, :, z])
        heatmap[:, :, z] = m
        fine_scores[z] = top_fraction_mean(m, top_fraction, body[:, :, z])

    heatmap = smooth_z(heatmap, sigma=float(cfg['postprocess'].get('z_sigma', 1.0)))
    heatmap *= body
    heatmap = robust_normalize(heatmap, body)

    mask = np.zeros_like(body, dtype=np.uint8)
    pp = cfg['postprocess']
    for z in range(n_slices):
        binary = threshold_map(heatmap[:, :, z], body[:, :, z], method=str(pp.get('threshold_method', 'mad')), mad_k=float(pp.get('mad_k', 3.5)), percentile=float(pp.get('percentile', 99.0)))
        mask[:, :, z] = clean_mask_2d(binary, min_area=int(pp.get('min_2d_area', 24)), closing_iterations=int(pp.get('closing_iterations', 1)))

    labels, candidates = connected_components_3d(mask, heatmap, min_voxels=int(pp.get('min_3d_voxels', 100)))
    mask = (labels > 0).astype(np.uint8)

    heatmap_path = out_dir / f'{case}_heatmap.nii.gz'
    mask_path = out_dir / f'{case}_mask.nii.gz'
    save_nifti_like(ref_img, heatmap.astype(np.float32), heatmap_path)
    save_nifti_like(ref_img, mask.astype(np.uint8), mask_path)

    csv_path = out_dir / f'{case}_slices.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['slice', 'coarse_score', 'fine_score', 'processed_fine'])
        for z in range(n_slices):
            writer.writerow([z, coarse_scores.get(z, ''), fine_scores.get(z, ''), int(z in fine_scores)])

    candidates_path = out_dir / f'{case}_candidates.json'
    candidates_path.write_text(json.dumps([c.to_dict() for c in candidates], ensure_ascii=False, indent=2), encoding='utf-8')

    return {'case': case, 'input': str(path), 'heatmap': str(heatmap_path), 'mask': str(mask_path), 'candidates': str(candidates_path), 'fine_slices': len(candidate_slices), 'num_candidates': len(candidates)}
