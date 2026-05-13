from src.domain.contracts import (
    validate_dataset_snapshot,
    validate_event,
    validate_item,
    validate_session,
    validate_user,
)


def test_validate_item() -> None:
    payload = {
        "item_id": "item_1",
        "domain": "movies",
        "locale": "en",
        "modalities": {
            "text": {"title": "Example"},
            "image": {},
            "audio": {},
            "video": {},
            "sequence": {},
        },
        "assets": {"image": [], "text": [], "audio": [], "video": []},
        "provenance": {"source": "unit_test"},
    }
    validate_item(payload)


def test_validate_user() -> None:
    payload = {"user_id": "user_1", "locale": "en"}
    validate_user(payload)


def test_validate_session() -> None:
    payload = {
        "session_id": "session_1", 
        "started_at_utc": "2026_05_08T00:00:00Z",
        "context": {"device_type": "mobile", "cohort": "control"}
    }
    validate_session(payload)


def test_validate_event_strict() -> None:
    payload = {
        "event_id": "event_1",
        "event_type": "click",
        "occurred_at_utc": "2026_05_08T00:00:00Z",
    }
    validate_event(payload, strict=True)


def test_validate_dataset_snapshot() -> None:
    payload = {
        "dataset_id": "movies",
        "version": "v1",
        "license_name": "cc_by_4_0",
        "source_uri": "https://example.com/movies",
        "snapshot_utc": "2026_05_08T00:00:00Z",
        "content_hash": "hash_a",
        "schema_hash": "schema_a",
        "extra": {},
    }
    validate_dataset_snapshot(payload)
