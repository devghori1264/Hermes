import json
from pathlib import Path

from src.services.vector_index_service import VectorIndexService


def test_vector_index_service_search_and_snapshot(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "domain": "movies",
                "dimension": 3,
                "items": [
                    {"item_id": "0", "vector": [1.0, 0.0, 0.0]},
                    {"item_id": "1", "vector": [0.0, 1.0, 0.0]},
                    {"item_id": "2", "vector": [0.0, 0.0, 1.0]},
                ],
            }
        ),
        encoding="utf8",
    )

    service = VectorIndexService(index_path)
    snapshot = service.snapshot()
    assert snapshot.item_count == 3
    assert snapshot.dimension == 3

    results = service.search([1.0, 0.0, 0.0], top_k=2)
    assert results[0].item_id == "0"
    assert len(results) == 2
