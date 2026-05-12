import json
from pathlib import Path

from src.data.ingestion.manifest import load_manifest


def test_load_manifest(tmp_path) -> None:
    payload = {
        "dataset_id": "movies_catalog",
        "version": "v1",
        "domain": "movies",
        "source_uri": "data.csv",
        "license_name": "cc_by_4_0",
        "snapshot_utc": "2026_05_08T00:00:00Z",
        "locale_default": "en",
        "expected_columns": ["movie_title", "comb"],
        "output_items_path": "out.jsonl",
        "output_snapshot_path": "snapshot.json",
        "record_limit": 5,
        "strict_validation": False,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf8")

    loaded = load_manifest(Path(path))
    assert loaded.manifest.dataset_id == "movies_catalog"
    assert loaded.manifest.record_limit == 5
