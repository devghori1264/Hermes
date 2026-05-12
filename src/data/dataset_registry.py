from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from src.domain.contracts import validate_dataset_snapshot


@dataclass(frozen=True)
class DatasetSnapshot:
    dataset_id: str
    version: str
    license_name: str
    source_uri: str
    snapshot_utc: str
    content_hash: str
    schema_hash: str
    extra: dict[str, Any]


class DatasetRegistry:
    def __init__(self) -> None:
        self._snapshots: dict[str, DatasetSnapshot] = {}

    def _key(self, dataset_id: str, version: str) -> str:
        return f"{dataset_id}:{version}"

    def register(self, snapshot: DatasetSnapshot) -> None:
        key = self._key(snapshot.dataset_id, snapshot.version)
        if key in self._snapshots:
            raise ValueError("snapshot already registered")
        self._snapshots[key] = snapshot

    def get(self, dataset_id: str, version: str) -> DatasetSnapshot | None:
        return self._snapshots.get(self._key(dataset_id, version))

    def list_all(self) -> list[DatasetSnapshot]:
        return list(self._snapshots.values())

    def load_json_snapshot(self, path: Path) -> DatasetSnapshot:
        raw = json.loads(path.read_text(encoding="utf8"))
        validate_dataset_snapshot(raw)
        return DatasetSnapshot(
            dataset_id=str(raw["dataset_id"]),
            version=str(raw["version"]),
            license_name=str(raw["license_name"]),
            source_uri=str(raw["source_uri"]),
            snapshot_utc=str(raw["snapshot_utc"]),
            content_hash=str(raw["content_hash"]),
            schema_hash=str(raw["schema_hash"]),
            extra=dict(raw.get("extra", {})),
        )

    def register_from_json(self, path: Path) -> DatasetSnapshot:
        snapshot = self.load_json_snapshot(path)
        self.register(snapshot)
        return snapshot
