from pathlib import Path

from src.features.feature_store import (
    FeatureKey,
    InMemoryFeatureStore,
    JsonlFeatureStore,
    build_feature_record,
)


def test_in_memory_store_round_trip() -> None:
    store = InMemoryFeatureStore()
    key = FeatureKey(
        entity_type="item",
        entity_id="item_1",
        feature_name="embedding",
        feature_version="v1",
        modality="text",
    )
    record = build_feature_record(key, [0.1, 0.2], {"source": "unit"})
    store.put(record)
    loaded = store.get(key)
    assert loaded is not None
    assert loaded.vector == [0.1, 0.2]
    assert store.count() == 1


def test_jsonl_store_round_trip(tmp_path) -> None:
    data_path = Path(tmp_path / "store.jsonl")
    store = JsonlFeatureStore(data_path)
    key = FeatureKey(
        entity_type="item",
        entity_id="item_1",
        feature_name="embedding",
        feature_version="v1",
        modality="text",
    )
    record = build_feature_record(key, [0.2, 0.3], {"source": "unit"})
    store.put(record)

    reloaded = JsonlFeatureStore(data_path)
    loaded = reloaded.get(key)
    assert loaded is not None
    assert loaded.vector == [0.2, 0.3]
    assert reloaded.count() == 1
