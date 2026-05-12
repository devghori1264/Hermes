from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import Iterable

import requests

from src.data.sourcing.source_registry import DatasetArtifact, ExtractedFile


def compute_checksum(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_checksum(path: Path, checksum: str, algorithm: str) -> bool:
    digest = compute_checksum(path, algorithm)
    return digest == checksum


def download_artifact(artifact: DatasetArtifact, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / artifact.file_name
    if target_path.exists() and artifact.checksum and artifact.checksum_algorithm:
        if verify_checksum(target_path, artifact.checksum, artifact.checksum_algorithm):
            return target_path

    with requests.get(artifact.url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with target_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    if artifact.checksum and artifact.checksum_algorithm:
        if not verify_checksum(target_path, artifact.checksum, artifact.checksum_algorithm):
            raise ValueError("checksum verification failed")

    return target_path


def extract_zip(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(output_dir)


def verify_extracted_files(root_dir: Path, files: Iterable[ExtractedFile]) -> None:
    for entry in files:
        target = root_dir / entry.path
        if not target.exists():
            raise FileNotFoundError(f"missing extracted file: {entry.path}")
        if not verify_checksum(target, entry.checksum, entry.checksum_algorithm):
            raise ValueError(f"checksum mismatch: {entry.path}")
