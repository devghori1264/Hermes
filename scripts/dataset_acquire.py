from __future__ import annotations

import argparse
from pathlib import Path

from src.data.sourcing.acquire import download_artifact, extract_zip, verify_extracted_files
from src.data.sourcing.source_registry import load_dataset_sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify dataset artifacts")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-dir", default="datasets/raw")
    parser.add_argument("--source-registry", default="evidence/dataset_sources.json")
    parser.add_argument("--artifact-id", default="")
    args = parser.parse_args()

    registry = load_dataset_sources(Path(args.source_registry))
    source = registry.get(args.dataset_id)
    if source is None:
        raise ValueError("dataset_id not found")

    output_dir = Path(args.output_dir) / args.dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)

    for artifact in source.artifacts:
        if args.artifact_id and artifact.artifact_id != args.artifact_id:
            continue
        archive_path = download_artifact(artifact, output_dir)
        if artifact.extract and artifact.extract.kind == "zip":
            extract_zip(archive_path, output_dir)
            verify_extracted_files(output_dir, artifact.extract.files)


if __name__ == "__main__":
    main()
