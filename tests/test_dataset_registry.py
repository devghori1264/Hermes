import json

from src.data.dataset_registry import DatasetRegistry, DatasetSnapshot


def test_register_and_get() -> None:
    registry = DatasetRegistry()
    snapshot = DatasetSnapshot(
        dataset_id="movies",
        version="v1",
        license_name="cc_by_4_0",
        source_uri="https://example.com/movies",
        snapshot_utc="2026_05_08T00:00:00Z",
        content_hash="hash_a",
        schema_hash="schema_a",
        extra={},
    )
    registry.register(snapshot)
    loaded = registry.get("movies", "v1")
    assert loaded == snapshot


def test_register_duplicate_raises() -> None:
    registry = DatasetRegistry()
    snapshot = DatasetSnapshot(
        dataset_id="movies",
        version="v1",
        license_name="cc_by_4_0",
        source_uri="https://example.com/movies",
        snapshot_utc="2026_05_08T00:00:00Z",
        content_hash="hash_a",
        schema_hash="schema_a",
        extra={},
    )
    registry.register(snapshot)
    try:
        registry.register(snapshot)
        assert False
    except ValueError:
        assert True


def test_load_json_snapshot(tmp_path) -> None:
    payload = {
        "dataset_id": "movies",
        "version": "v1",
        "license_name": "cc_by_4_0",
        "source_uri": "https://example.com/movies",
        "snapshot_utc": "2026_05_08T00:00:00Z",
        "content_hash": "hash_a",
        "schema_hash": "schema_a",
        "extra": {"note": "test"},
    }
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(payload), encoding="utf8")

    registry = DatasetRegistry()
    snapshot = registry.load_json_snapshot(path)
    assert snapshot.dataset_id == "movies"
    assert snapshot.extra.get("note") == "test"
