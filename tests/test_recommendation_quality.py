from pathlib import Path

import pytest

from src.services.recommendation_service import RecommendationService


@pytest.mark.quality
def test_recommendation_quality_on_full_catalog(full_catalog_csv: Path) -> None:
    service = RecommendationService(full_catalog_csv)
    recs = service.recommend_titles("avatar")
    assert isinstance(recs, list)
    assert len(recs) <= 10
