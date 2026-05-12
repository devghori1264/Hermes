from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FeatureKey:
    entity_type: str
    entity_id: str
    feature_name: str
    feature_version: str
    modality: str
    namespace: str = "default"

    def to_key(self) -> str:
        return ":".join(
            [
                self.namespace,
                self.entity_type,
                self.entity_id,
                self.feature_name,
                self.feature_version,
                self.modality,
            ]
        )


@dataclass(frozen=True)
class FeatureRecord:
    key: FeatureKey
    vector: list[float]
    created_at_utc: str
    metadata: dict[str, Any] = field(default_factory=dict)
    checksum: str | None = None


def now_utc() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def compute_vector_checksum(vector: list[float]) -> str:
    payload = ",".join(f"{value:.8f}" for value in vector).encode("utf8")
    return hashlib.sha256(payload).hexdigest()


def build_feature_record(
    key: FeatureKey,
    vector: list[float],
    metadata: dict[str, Any] | None = None,
) -> FeatureRecord:
    return FeatureRecord(
        key=key,
        vector=vector,
        created_at_utc=now_utc(),
        metadata=metadata or {},
        checksum=compute_vector_checksum(vector),
    )


class FeatureStore:
    def get(self, key: FeatureKey) -> FeatureRecord | None:
        raise NotImplementedError

    def put(self, record: FeatureRecord) -> None:
        raise NotImplementedError

    def batch_get(self, keys: list[FeatureKey]) -> list[FeatureRecord | None]:
        return [self.get(key) for key in keys]

    def batch_put(self, records: list[FeatureRecord]) -> None:
        for record in records:
            self.put(record)

    def count(self) -> int:
        raise NotImplementedError


class InMemoryFeatureStore(FeatureStore):
    def __init__(self) -> None:
        self._records: dict[str, FeatureRecord] = {}

    def get(self, key: FeatureKey) -> FeatureRecord | None:
        return self._records.get(key.to_key())

    def put(self, record: FeatureRecord) -> None:
        self._records[record.key.to_key()] = record

    def count(self) -> int:
        return len(self._records)


class JsonlFeatureStore(FeatureStore):
    def __init__(self, data_path: Path, index_path: Path | None = None) -> None:
        self.data_path = data_path
        self.index_path = index_path or data_path.with_suffix(".index.json")
        self._index: dict[str, int] = {}
        self._initialize()

    def _initialize(self) -> None:
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        if self.index_path.exists():
            raw = json.loads(self.index_path.read_text(encoding="utf8"))
            self._index = {str(key): int(value) for key, value in raw.items()}
            return
        if self.data_path.exists():
            self._index = self._build_index()
            self._persist_index()

    def _build_index(self) -> dict[str, int]:
        index: dict[str, int] = {}
        with self.data_path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                raw = json.loads(line)
                key = raw.get("key")
                if key:
                    index[str(key)] = int(offset)
        return index

    def _persist_index(self) -> None:
        self.index_path.write_text(json.dumps(self._index, ensure_ascii=True), encoding="utf8")

    def _record_from_payload(self, payload: dict[str, Any]) -> FeatureRecord:
        key = FeatureKey(
            namespace=str(payload["key"]).split(":")[0],
            entity_type=str(payload["entity_type"]),
            entity_id=str(payload["entity_id"]),
            feature_name=str(payload["feature_name"]),
            feature_version=str(payload["feature_version"]),
            modality=str(payload["modality"]),
        )
        return FeatureRecord(
            key=key,
            vector=list(payload["vector"]),
            created_at_utc=str(payload["created_at_utc"]),
            metadata=dict(payload.get("metadata", {})),
            checksum=str(payload.get("checksum")) if payload.get("checksum") else None,
        )

    def get(self, key: FeatureKey) -> FeatureRecord | None:
        key_str = key.to_key()
        offset = self._index.get(key_str)
        if offset is None:
            return None
        with self.data_path.open("rb") as handle:
            handle.seek(offset)
            line = handle.readline()
            if not line:
                return None
            raw = json.loads(line)
        return self._record_from_payload(raw)

    def put(self, record: FeatureRecord) -> None:
        key_str = record.key.to_key()
        payload = {
            "key": key_str,
            "entity_type": record.key.entity_type,
            "entity_id": record.key.entity_id,
            "feature_name": record.key.feature_name,
            "feature_version": record.key.feature_version,
            "modality": record.key.modality,
            "vector": record.vector,
            "created_at_utc": record.created_at_utc,
            "metadata": record.metadata,
            "checksum": record.checksum,
        }
        with self.data_path.open("ab") as handle:
            offset = handle.tell()
            handle.write(json.dumps(payload, ensure_ascii=True).encode("utf8"))
            handle.write(b"\n")
        self._index[key_str] = int(offset)
        self._persist_index()

    def count(self) -> int:
        return len(self._index)


def build_feature_store(path: str | None) -> FeatureStore:
    if path:
        return JsonlFeatureStore(Path(path))
    return InMemoryFeatureStore()
