from __future__ import annotations

from typing import Any

from jsonschema import validate


ASSET_REF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "uri": {"type": "string"},
        "mime_type": {"type": "string"},
        "checksum": {"type": ["string", "null"]},
        "size_bytes": {"type": ["number", "null"]},
        "width_px": {"type": ["number", "null"]},
        "height_px": {"type": ["number", "null"]},
        "duration_ms": {"type": ["number", "null"]},
        "codec": {"type": ["string", "null"]},
    },
    "required": ["uri", "mime_type"],
    "additionalProperties": True,
}


MODALITY_ASSETS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "array", "items": ASSET_REF_SCHEMA},
        "image": {"type": "array", "items": ASSET_REF_SCHEMA},
        "audio": {"type": "array", "items": ASSET_REF_SCHEMA},
        "video": {"type": "array", "items": ASSET_REF_SCHEMA},
    },
    "required": [],
    "additionalProperties": True,
}


MODALITY_PAYLOAD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "object"},
        "image": {"type": "object"},
        "audio": {"type": "object"},
        "video": {"type": "object"},
        "sequence": {"type": "object"},
    },
    "required": ["text", "image", "audio", "video", "sequence"],
    "additionalProperties": True,
}


UNIVERSAL_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "item_id": {"type": "string"},
        "domain": {"type": "string"},
        "locale": {"type": "string"},
        "modalities": MODALITY_PAYLOAD_SCHEMA,
        "assets": MODALITY_ASSETS_SCHEMA,
        "provenance": {"type": "object"},
    },
    "required": ["item_id", "domain", "locale", "modalities", "provenance"],
    "additionalProperties": True,
}

CONTEXT_ENRICHMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "device_type": {"type": "string"},
        "network_type": {"type": "string"},
        "timezone_offset": {"type": "number"},
        "app_version": {"type": "string"},
        "cohort": {"type": "string"}
    },
    "required": ["device_type", "cohort"],
    "additionalProperties": True,
}


UNIVERSAL_USER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "user_id": {"type": "string"},
        "locale": {"type": "string"},
        "attributes": {"type": "object"},
        "preferences": {"type": "object"},
    },
    "required": ["user_id", "locale"],
    "additionalProperties": True,
}


UNIVERSAL_SESSION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "session_id": {"type": "string"},
        "user_id": {"type": ["string", "null"]},
        "started_at_utc": {"type": "string"},
        "context": CONTEXT_ENRICHMENT_SCHEMA,
    },
    "required": ["session_id", "started_at_utc", "context"],
    "additionalProperties": True,
}


KNOWN_EVENT_TYPES = {
    "impression",
    "click",
    "dwell",
    "purchase",
    "save",
    "hide",
    "skip",
    "complaint",
}


UNIVERSAL_EVENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "event_id": {"type": "string"},
        "event_type": {"type": "string"},
        "user_id": {"type": ["string", "null"]},
        "item_id": {"type": ["string", "null"]},
        "session_id": {"type": ["string", "null"]},
        "occurred_at_utc": {"type": "string"},
        "properties": {"type": "object"},
    },
    "required": ["event_id", "event_type", "occurred_at_utc"],
    "additionalProperties": True,
}


DATASET_SNAPSHOT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "dataset_id": {"type": "string"},
        "version": {"type": "string"},
        "license_name": {"type": "string"},
        "source_uri": {"type": "string"},
        "snapshot_utc": {"type": "string"},
        "content_hash": {"type": "string"},
        "schema_hash": {"type": "string"},
        "extra": {"type": "object"},
    },
    "required": [
        "dataset_id",
        "version",
        "license_name",
        "source_uri",
        "snapshot_utc",
        "content_hash",
        "schema_hash",
    ],
    "additionalProperties": True,
}


def validate_item(payload: dict[str, Any]) -> None:
    validate(instance=payload, schema=UNIVERSAL_ITEM_SCHEMA)


def validate_user(payload: dict[str, Any]) -> None:
    validate(instance=payload, schema=UNIVERSAL_USER_SCHEMA)


def validate_session(payload: dict[str, Any]) -> None:
    validate(instance=payload, schema=UNIVERSAL_SESSION_SCHEMA)


def validate_event(payload: dict[str, Any], strict: bool = False) -> None:
    validate(instance=payload, schema=UNIVERSAL_EVENT_SCHEMA)
    if strict:
        event_type = payload.get("event_type")
        if event_type not in KNOWN_EVENT_TYPES:
            raise ValueError("unknown event_type")


def validate_dataset_snapshot(payload: dict[str, Any]) -> None:
    validate(instance=payload, schema=DATASET_SNAPSHOT_SCHEMA)
