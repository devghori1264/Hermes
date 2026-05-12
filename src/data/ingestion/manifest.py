from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from jsonschema import validate


INGESTION_MANIFEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "dataset_id": {"type": "string"},
        "version": {"type": "string"},
        "domain": {"type": "string"},
        "source_uri": {"type": "string"},
        "license_name": {"type": "string"},
        "snapshot_utc": {"type": "string"},
        "locale_default": {"type": "string"},
        "expected_columns": {"type": "array", "items": {"type": "string"}},
        "output_items_path": {"type": "string"},
        "output_snapshot_path": {"type": "string"},
        "record_limit": {"type": ["number", "null"]},
        "strict_validation": {"type": "boolean"},
    },
    "required": [
        "dataset_id",
        "version",
        "domain",
        "source_uri",
        "license_name",
        "snapshot_utc",
        "locale_default",
        "expected_columns",
        "output_items_path",
        "output_snapshot_path",
    ],
    "additionalProperties": True,
}


@dataclass(frozen=True)
class IngestionManifest:
    dataset_id: str
    version: str
    domain: str
    source_uri: str
    license_name: str
    snapshot_utc: str
    locale_default: str
    expected_columns: list[str]
    output_items_path: str
    output_snapshot_path: str
    record_limit: int | None
    strict_validation: bool


@dataclass(frozen=True)
class ManifestLoadResult:
    manifest: IngestionManifest
    raw: dict[str, Any]


def load_manifest(path: Path) -> ManifestLoadResult:
    raw = json.loads(path.read_text(encoding="utf8"))
    validate(instance=raw, schema=INGESTION_MANIFEST_SCHEMA)
    manifest = IngestionManifest(
        dataset_id=str(raw["dataset_id"]),
        version=str(raw["version"]),
        domain=str(raw["domain"]),
        source_uri=str(raw["source_uri"]),
        license_name=str(raw["license_name"]),
        snapshot_utc=str(raw["snapshot_utc"]),
        locale_default=str(raw["locale_default"]),
        expected_columns=list(raw["expected_columns"]),
        output_items_path=str(raw["output_items_path"]),
        output_snapshot_path=str(raw["output_snapshot_path"]),
        record_limit=int(raw["record_limit"]) if raw.get("record_limit") is not None else None,
        strict_validation=bool(raw.get("strict_validation", True)),
    )
    return ManifestLoadResult(manifest=manifest, raw=raw)
