import json
from pathlib import Path

from src.data.ingestion.runner import run_ingestion


def test_amazon_reviews_2023_meta_ingestion(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "meta.jsonl"
    record = {
        "parent_asin": "B000TEST",
        "title": "Test Product",
        "features": ["Feature one", "Feature two"],
        "description": ["Description text"],
        "categories": ["Category"],
        "images": [{"hi_res": "https://example.com/image.jpg"}],
        "details": {"Brand": "Example"},
        "store": "Example Store",
    }
    jsonl_path.write_text(json.dumps(record) + "\n", encoding="utf8")

    output_items = tmp_path / "items.jsonl"
    output_snapshot = tmp_path / "snapshot.json"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "amazon_reviews_2023_meta",
                "version": "v1",
                "domain": "commerce",
                "source_uri": str(jsonl_path),
                "license_name": "local",
                "snapshot_utc": "2026_05_08T10:22:09Z",
                "locale_default": "en-US",
                "expected_columns": [
                    "parent_asin",
                    "title",
                    "features",
                    "description",
                    "categories",
                    "images",
                ],
                "output_items_path": str(output_items),
                "output_snapshot_path": str(output_snapshot),
                "record_limit": None,
                "strict_validation": True,
            }
        ),
        encoding="utf8",
    )

    result = run_ingestion(manifest_path)
    assert result.report.total_rows == 1

    items = output_items.read_text(encoding="utf8").strip().splitlines()
    payload = json.loads(items[0])
    assert payload["domain"] == "commerce"
    assert payload["modalities"]["text"]["title"] == "Test Product"
    assert payload["assets"]["image"]
