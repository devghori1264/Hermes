import json
from pathlib import Path

from src.data.ingestion.runner import run_ingestion


def test_mapped_tabular_ingestion(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("id,title,summary\n1,Alpha,First item\n", encoding="utf8")

    output_items = tmp_path / "items.jsonl"
    output_snapshot = tmp_path / "snapshot.json"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "mapped_tabular",
                "version": "v1",
                "domain": "generic",
                "source_uri": str(csv_path),
                "license_name": "local",
                "snapshot_utc": "2026_05_08T10:22:09Z",
                "locale_default": "en-US",
                "expected_columns": ["id", "title", "summary"],
                "output_items_path": str(output_items),
                "output_snapshot_path": str(output_snapshot),
                "record_limit": None,
                "strict_validation": True,
                "delimiter": ",",
                "mapping": {
                    "id_column": "id",
                    "title_column": "title",
                    "text_columns": ["summary"]
                }
            }
        ),
        encoding="utf8",
    )

    result = run_ingestion(manifest_path)
    assert result.report.total_rows == 1

    items = output_items.read_text(encoding="utf8").strip().splitlines()
    payload = json.loads(items[0])
    assert payload["modalities"]["text"]["title"] == "Alpha"
