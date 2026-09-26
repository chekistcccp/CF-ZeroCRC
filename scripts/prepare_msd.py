#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from cfzerocrc.datasets import prepare_msd


def main() -> None:
    parser = argparse.ArgumentParser(description='Safely extract/find MSD Task10 Colon training data.')
    parser.add_argument('--root', default='data/MSD')
    args = parser.parse_args()
    images, labels = prepare_msd(Path(args.root))
    print(f'imagesTr: {images}')
    print(f'labelsTr: {labels}')


if __name__ == '__main__':
    main()
