#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from modelscope import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description='Download SD3.5 Medium Diffusers files from ModelScope.')
    parser.add_argument('--model-id', default='stabilityai/stable-diffusion-3.5-medium')
    parser.add_argument('--output', default='models/sd35-medium')
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    allow_patterns = [
        'model_index.json', 'scheduler/*', 'transformer/*', 'vae/*',
        'text_encoder/*', 'text_encoder_2/*', 'text_encoder_3/*',
        'tokenizer/*', 'tokenizer_2/*', 'tokenizer_3/*', 'LICENSE*', 'README*'
    ]
    path = snapshot_download(args.model_id, local_dir=str(output), allow_patterns=allow_patterns)
    print(f'Model downloaded to: {path}')


if __name__ == '__main__':
    main()
