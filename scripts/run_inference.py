#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from tqdm import tqdm

from cfzerocrc.model import SD35CounterfactualEngine
from cfzerocrc.pipeline import list_nifti, process_case
from cfzerocrc.datasets import msi_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description='Run CF-ZeroCRC inference on NIfTI CT volumes.')
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--input', required=True, help='NIfTI file or directory')
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['coarse_fine', 'exhaustive'], default='coarse_fine')
    parser.add_argument('--dataset', choices=['msd', 'msi', 'generic'], default='generic')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    if args.dataset == 'msi':
        cases = msi_pairs(Path(args.input))
    else:
        cases = [(None, path, None) for path in list_nifti(args.input)]
    if not cases:
        raise SystemExit(f'No .nii/.nii.gz files found under {args.input}')

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    engine = SD35CounterfactualEngine(cfg)

    manifest = []
    for case_id, path, label_path in tqdm(cases, desc='Cases'):
        result = process_case(path, engine, cfg, out, mode=args.mode, case_id=case_id, label_path=label_path)
        manifest.append(result)
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    print(f'Finished {len(manifest)} cases. Results: {out}')


if __name__ == '__main__':
    main()
