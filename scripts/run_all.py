#!/usr/bin/env python
"""Run MSD and MSI preparation, inference, and evaluation in one command."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from cfzerocrc.datasets import msi_pairs, prepare_msd


def run_all(root: Path, runner=subprocess.run) -> None:
    root = root.resolve()
    config_path = root / 'configs/default.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    model_path = Path(config['model']['path'])
    if not model_path.is_absolute():
        model_path = root / model_path

    images, labels = prepare_msd(root / 'data/MSD')
    private_pairs = msi_pairs(root / 'data/MSI')
    print(f'MSD: {images} / {labels}', flush=True)
    print(f'MSI: {len(private_pairs)} paired cases', flush=True)

    def run(script: str, *args: str) -> None:
        print(f'Running {script}: {" ".join(args)}', flush=True)
        runner([sys.executable, str(root / 'scripts' / script), *args], cwd=root, check=True)

    if not (model_path / 'model_index.json').is_file():
        run('download_model.py', '--output', str(model_path))

    run('run_inference.py', '--dataset', 'msd', '--config', str(config_path),
        '--input', str(images), '--output', str(root / 'outputs/msd'),
        '--mode', 'exhaustive', '--resume')
    run('evaluate_msd.py', '--dataset', 'msd', '--pred-dir', str(root / 'outputs/msd'),
        '--gt-dir', str(labels), '--output', str(root / 'outputs/msd_metrics.json'))
    run('run_inference.py', '--dataset', 'msi', '--config', str(config_path),
        '--input', str(root / 'data/MSI'), '--output', str(root / 'outputs/msi'),
        '--mode', 'coarse_fine', '--resume')
    run('evaluate_msd.py', '--dataset', 'msi', '--pred-dir', str(root / 'outputs/msi'),
        '--gt-dir', str(root / 'data/MSI'), '--output', str(root / 'outputs/msi_metrics.json'))
    print('All experiments completed. Metrics: outputs/msd_metrics.json and outputs/msi_metrics.json', flush=True)


if __name__ == '__main__':
    run_all(Path(__file__).resolve().parents[1])
