from pathlib import Path

from src.data.catalog_repository import CatalogRepository


def test_dataset_snapshot_loading(tmp_path) -> None:
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("movie_title,comb\nexample,example", encoding="utf8")
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(
        "{\n"
        "  \"dataset_id\": \"movies_catalog\",\n"
        "  \"version\": \"v1\",\n"
        "  \"license_name\": \"cc_by_4_0\",\n"
        "  \"source_uri\": \"data.csv\",\n"
        "  \"snapshot_utc\": \"2026_05_08T00:00:00Z\",\n"
        "  \"content_hash\": \"hash_a\",\n"
        "  \"schema_hash\": \"schema_a\"\n"
        "}\n",
        encoding="utf8",
    )

    repo = CatalogRepository(Path(csv_path), snapshot_path=Path(snapshot_path))
    snapshot = repo.dataset_snapshot()
    assert snapshot is not None
    assert snapshot.dataset_id == "movies_catalog"
