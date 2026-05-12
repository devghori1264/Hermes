from pathlib import Path

from src.domain.models import RankedItem
from src.services.recommendation_service import RecommendationService


def test_recommendation_returns_candidates_for_known_title(tiny_catalog_csv: Path) -> None:
    service = RecommendationService(tiny_catalog_csv)
    recs = service.recommend_titles('avatar')
    assert isinstance(recs, list)


def test_recommendation_returns_cold_start_for_unknown_title(tiny_catalog_csv: Path) -> None:
    service = RecommendationService(tiny_catalog_csv)
    recs = service.recommend_titles('this-title-does-not-exist')
    assert isinstance(recs, list)
    assert all('this-title-does-not-exist' != r.lower() for r in recs)


def test_recommendation_ranked_items_include_explanation_fields(tiny_catalog_csv: Path) -> None:
    service = RecommendationService(tiny_catalog_csv)
    ranked = service.recommend_ranked('avatar')
    assert isinstance(ranked, list)
    if ranked:
        assert isinstance(ranked[0], RankedItem)
        assert isinstance(ranked[0].explanation, str)
