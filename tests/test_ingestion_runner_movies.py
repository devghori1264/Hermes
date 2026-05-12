import json
from pathlib import Path

from src.data.ingestion.runner import run_ingestion


def test_run_ingestion_movies(tmp_path) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("movie_title,comb\nExample,example", encoding="utf8")

    manifest_payload = {
        "dataset_id": "movies_catalog",
        "version": "v1",
        "domain": "movies",
        "source_uri": str(csv_path),
        "license_name": "cc_by_4_0",
        "snapshot_utc": "2026_05_08T00:00:00Z",
        "locale_default": "en",
        "expected_columns": ["movie_title", "comb"],
        "output_items_path": str(tmp_path / "items.jsonl"),
        "output_snapshot_path": str(tmp_path / "snapshot.json"),
        "record_limit": None,
        "strict_validation": True,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf8")

    result = run_ingestion(Path(manifest_path))
    assert result.output_items_path.exists()
    assert result.dataset_snapshot.dataset_id == "movies_catalog"
    assert result.report.normalized_rows == 1
