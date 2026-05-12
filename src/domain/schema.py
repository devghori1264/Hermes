from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModalityPayload:
    text: dict[str, Any] = field(default_factory=dict)
    image: dict[str, Any] = field(default_factory=dict)
    audio: dict[str, Any] = field(default_factory=dict)
    video: dict[str, Any] = field(default_factory=dict)
    sequence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AssetRef:
    uri: str
    mime_type: str
    checksum: str | None = None
    size_bytes: int | None = None
    width_px: int | None = None
    height_px: int | None = None
    duration_ms: int | None = None
    codec: str | None = None


@dataclass(frozen=True)
class ModalityAssets:
    text: list[AssetRef] = field(default_factory=list)
    image: list[AssetRef] = field(default_factory=list)
    audio: list[AssetRef] = field(default_factory=list)
    video: list[AssetRef] = field(default_factory=list)


@dataclass(frozen=True)
class UniversalItemSchema:
    item_id: str
    domain: str
    locale: str
    modalities: ModalityPayload
    provenance: dict[str, Any]
    assets: ModalityAssets = field(default_factory=ModalityAssets)


@dataclass(frozen=True)
class UniversalUserSchema:
    user_id: str
    locale: str
    attributes: dict[str, Any] = field(default_factory=dict)
    preferences: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UniversalSessionSchema:
    session_id: str
    user_id: str | None
    started_at_utc: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UniversalEventSchema:
    event_id: str
    event_type: str
    user_id: str | None
    item_id: str | None
    session_id: str | None
    occurred_at_utc: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetSnapshotSchema:
    dataset_id: str
    version: str
    license_name: str
    source_uri: str
    snapshot_utc: str
    content_hash: str
    schema_hash: str
    extra: dict[str, Any] = field(default_factory=dict)
