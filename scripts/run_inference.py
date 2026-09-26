#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml
from tqdm import tqdm

from cfzerocrc.model import SD35CounterfactualEngine
from cfzerocrc.pipeline import list_nifti, process_case
from cfzerocrc.datasets import msi_pairs
from cfzerocrc.datasets import nifti_id


def write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temporary.replace(path)


def complete_case(output: Path, case: str) -> bool:
    return all((output / f'{case}{suffix}').is_file() for suffix in
               ('_heatmap.nii.gz', '_mask.nii.gz', '_slices.csv', '_candidates.json'))


def main() -> None:
    parser = argparse.ArgumentParser(description='Run CF-ZeroCRC inference on NIfTI CT volumes.')
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--input', required=True, help='NIfTI file or directory')
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['coarse_fine', 'exhaustive'], default='coarse_fine')
    parser.add_argument('--dataset', choices=['msd', 'msi', 'generic'], default='generic')
    parser.add_argument('--resume', action='store_true', help='Skip complete cases from a matching prior run')
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
    metadata = {'config_sha256': hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                'input': str(Path(args.input).resolve()), 'dataset': args.dataset, 'mode': args.mode}
    metadata_path = out / 'run_metadata.json'
    manifest_path = out / 'manifest.json'
    previous: dict[str, dict] = {}
    if args.resume and metadata_path.is_file() and manifest_path.is_file():
        if json.loads(metadata_path.read_text(encoding='utf-8')) == metadata:
            previous = {row['case']: row for row in json.loads(manifest_path.read_text(encoding='utf-8'))}
    write_json_atomic(metadata_path, metadata)

    manifest = []
    engine = None
    skipped = 0
    for case_id, path, label_path in tqdm(cases, desc='Cases'):
        case = case_id or nifti_id(path)
        old = previous.get(case)
        if old and old.get('input') == str(path) and old.get('label') == (str(label_path) if label_path else None) and complete_case(out, case):
            manifest.append(old)
            skipped += 1
            continue
        if engine is None:
            engine = SD35CounterfactualEngine(cfg)
        result = process_case(path, engine, cfg, out, mode=args.mode, case_id=case_id, label_path=label_path)
        manifest.append(result)
        write_json_atomic(manifest_path, manifest)

    write_json_atomic(manifest_path, manifest)
    print(f'Finished {len(manifest)} cases ({skipped} resumed). Results: {out}')


if __name__ == '__main__':
    main()
