from __future__ import annotations

import re
import tarfile
from pathlib import Path
from shutil import copyfileobj


def nifti_id(path: Path) -> str:
    return path.name[:-7] if path.name.endswith('.nii.gz') else path.stem


def nifti_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob('*') if p.is_file() and (p.name.endswith('.nii.gz') or p.name.endswith('.nii')))


def msi_pairs(root: Path) -> list[tuple[str, Path, Path]]:
    """Pair data/123*.nii.gz images with data2/123.nii.gz labels."""
    image_dir, label_dir = root / 'data', root / 'data2'
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f'Expected MSI directories {image_dir} and {label_dir}')
    labels: dict[str, Path] = {}
    images: dict[str, Path] = {}
    for path in nifti_files(image_dir):
        stem = nifti_id(path)
        match = re.match(r'^(\d+).+', stem)
        if not match:
            raise ValueError(f'MSI image name must start with digits followed by other characters: {path}')
        case_id = match.group(1)
        if case_id in images:
            raise ValueError(f'Duplicate MSI image for {case_id}: {images[case_id]} and {path}')
        images[case_id] = path
    for path in nifti_files(label_dir):
        case_id = nifti_id(path)
        if not re.fullmatch(r'\d+', case_id):
            raise ValueError(f'MSI label name must be digits only: {path}')
        if case_id in labels:
            raise ValueError(f'Duplicate MSI label for {case_id}: {labels[case_id]} and {path}')
        labels[case_id] = path
    missing_labels = sorted(images.keys() - labels.keys())
    missing_images = sorted(labels.keys() - images.keys())
    if missing_labels or missing_images:
        raise ValueError(f'MSI pairs incomplete: missing labels for {missing_labels}; missing images for {missing_images}')
    if not images:
        raise ValueError(f'No MSI image/label pairs found in {root}')
    return [(case_id, images[case_id], labels[case_id]) for case_id in sorted(images)]


def prepare_msd(root: Path) -> tuple[Path, Path]:
    """Find an extracted Task10_Colon, or safely extract the one tar archive in root."""
    root.mkdir(parents=True, exist_ok=True)

    def locate() -> tuple[Path, Path] | None:
        for images in sorted(root.rglob('imagesTr')):
            labels = images.parent / 'labelsTr'
            if images.is_dir() and labels.is_dir() and nifti_files(images) and nifti_files(labels):
                return images, labels
        return None

    found = locate()
    if found:
        return found
    archives = sorted(p for p in root.iterdir() if p.is_file() and (p.name.endswith('.tar') or p.name.endswith('.tar.gz') or p.name.endswith('.tgz')))
    if len(archives) != 1:
        raise ValueError(f'Expected exactly one MSD tar archive in {root}, found {len(archives)}')
    archive = archives[0]
    destination = root.resolve()
    with tarfile.open(archive, 'r:*') as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination) or not (member.isfile() or member.isdir()):
                raise ValueError(f'Unsafe tar member: {member.name}')
        for member in tar.getmembers():
            target = destination / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise ValueError(f'Cannot read tar member: {member.name}')
                with source, target.open('wb') as output:
                    copyfileobj(source, output)
    found = locate()
    if not found:
        raise ValueError(f'No imagesTr/labelsTr pair found after extracting {archive}')
    return found
