from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExtractedFile:
    path: str
    checksum: str
    checksum_algorithm: str


@dataclass(frozen=True)
class ExtractSpec:
    kind: str
    files: list[ExtractedFile]


@dataclass(frozen=True)
class DatasetArtifact:
    artifact_id: str
    url: str
    file_name: str
    checksum: str | None = None
    checksum_algorithm: str | None = None
    extract: ExtractSpec | None = None


@dataclass(frozen=True)
class DatasetSource:
    dataset_id: str
    domain: str
    title: str
    homepage_url: str
    license_name: str | None
    license_url: str | None
    artifacts: list[DatasetArtifact]


@dataclass(frozen=True)
class DatasetSourceRegistry:
    created_utc: str
    sources: list[DatasetSource]

    def get(self, dataset_id: str) -> DatasetSource | None:
        for source in self.sources:
            if source.dataset_id == dataset_id:
                return source
        return None


def _parse_extract(payload: dict[str, Any]) -> ExtractSpec | None:
    extract = payload.get("extract")
    if not isinstance(extract, dict):
        return None
    files_payload = extract.get("files", [])
    files: list[ExtractedFile] = []
    for entry in files_payload:
        if not isinstance(entry, dict):
            continue
        files.append(
            ExtractedFile(
                path=str(entry.get("path", "")),
                checksum=str(entry.get("checksum", "")),
                checksum_algorithm=str(entry.get("checksum_algorithm", "")),
            )
        )
    return ExtractSpec(kind=str(extract.get("kind", "")), files=files)


def _parse_artifact(payload: dict[str, Any]) -> DatasetArtifact:
    return DatasetArtifact(
        artifact_id=str(payload.get("artifact_id", "")),
        url=str(payload.get("url", "")),
        file_name=str(payload.get("file_name", "")),
        checksum=payload.get("checksum"),
        checksum_algorithm=payload.get("checksum_algorithm"),
        extract=_parse_extract(payload),
    )


def load_dataset_sources(path: Path) -> DatasetSourceRegistry:
    payload = json.loads(path.read_text(encoding="utf8"))
    sources: list[DatasetSource] = []
    for entry in payload.get("sources", []):
        artifacts = [_parse_artifact(item) for item in entry.get("artifacts", [])]
        sources.append(
            DatasetSource(
                dataset_id=str(entry.get("dataset_id", "")),
                domain=str(entry.get("domain", "")),
                title=str(entry.get("title", "")),
                homepage_url=str(entry.get("homepage_url", "")),
                license_name=entry.get("license_name"),
                license_url=entry.get("license_url"),
                artifacts=artifacts,
            )
        )
    return DatasetSourceRegistry(created_utc=str(payload.get("created_utc", "")), sources=sources)
