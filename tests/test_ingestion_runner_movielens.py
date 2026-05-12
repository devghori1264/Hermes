import json
from pathlib import Path

from src.data.ingestion.runner import run_ingestion


def test_movielens_movies_ingestion(tmp_path: Path) -> None:
    csv_path = tmp_path / "movies.csv"
    csv_path.write_text(
        "movieId,title,genres\n1,Toy Story (1995),Adventure|Animation\n",
        encoding="utf8",
    )

    output_items = tmp_path / "items.jsonl"
    output_snapshot = tmp_path / "snapshot.json"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "movielens_movies",
                "version": "v1",
                "domain": "movies",
                "source_uri": str(csv_path),
                "license_name": "MovieLens usage license",
                "snapshot_utc": "2026_05_08T10:22:09Z",
                "locale_default": "en-US",
                "expected_columns": ["movieId", "title", "genres"],
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
    assert len(items) == 1
    payload = json.loads(items[0])
    assert payload["domain"] == "movies"
    assert payload["modalities"]["text"]["title"] == "Toy Story (1995)"
