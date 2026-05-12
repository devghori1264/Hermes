from __future__ import annotations

import pytest

from src.domain.models import RankedItem
from src.evaluation.fairness import (
    FairnessAuditConfig,
    FairnessAuditEngine,
    FairnessAuditReport,
    _gini_coefficient,
)


def _make_item(
    item_id: str,
    title: str,
    genre: str = "unknown",
    provider: str = "unknown",
    score: float = 1.0,
) -> RankedItem:
    return RankedItem(
        item_id=item_id,
        title=title,
        score=score,
        explanation="test",
        metadata={"genre": genre, "provider": provider},
    )


def _balanced_items() -> list[RankedItem]:
    return [
        _make_item("1", "Action Movie A", genre="action", provider="studio_a"),
        _make_item("2", "Drama Film B", genre="drama", provider="studio_b"),
        _make_item("3", "Action Movie C", genre="action", provider="studio_a"),
        _make_item("4", "Drama Film D", genre="drama", provider="studio_b"),
        _make_item("5", "Comedy Show E", genre="comedy", provider="studio_c"),
        _make_item("6", "Comedy Show F", genre="comedy", provider="studio_c"),
    ]


def _skewed_items() -> list[RankedItem]:
    return [
        _make_item("1", "Action A", genre="action", provider="mega_studio"),
        _make_item("2", "Action B", genre="action", provider="mega_studio"),
        _make_item("3", "Action C", genre="action", provider="mega_studio"),
        _make_item("4", "Action D", genre="action", provider="mega_studio"),
        _make_item("5", "Drama E", genre="drama", provider="indie"),
        _make_item("6", "Comedy F", genre="comedy", provider="indie"),
    ]


class TestGiniCoefficient:
    def test_equal_distribution(self) -> None:
        assert abs(_gini_coefficient([10.0, 10.0, 10.0])) < 1e-9

    def test_maximum_inequality(self) -> None:
        gini = _gini_coefficient([0.0, 0.0, 100.0])
        assert gini > 0.5

    def test_single_value(self) -> None:
        assert _gini_coefficient([42.0]) == 0.0

    def test_empty(self) -> None:
        assert _gini_coefficient([]) == 0.0

    def test_all_zeros(self) -> None:
        assert _gini_coefficient([0.0, 0.0, 0.0]) == 0.0


class TestCohortExposure:
    def test_balanced_cohorts_have_small_gap(self) -> None:
        engine = FairnessAuditEngine()
        items = _balanced_items()
        catalog_counts = {"action": 2, "drama": 2, "comedy": 2}
        report = engine.audit(items, catalog_counts, {})
        for record in report.cohort_exposure:
            assert abs(record.gap) < 0.01

    def test_skewed_cohorts_have_large_gap(self) -> None:
        engine = FairnessAuditEngine()
        items = _skewed_items()
        catalog_counts = {"action": 33, "drama": 33, "comedy": 34}
        report = engine.audit(items, catalog_counts, {})
        action_record = next(r for r in report.cohort_exposure if r.cohort_id == "action")
        assert action_record.gap > 0.1

    def test_missing_cohort_in_recommendations(self) -> None:
        engine = FairnessAuditEngine()
        items = [_make_item("1", "Action", genre="action")]
        catalog_counts = {"action": 50, "drama": 50}
        report = engine.audit(items, catalog_counts, {})
        drama_record = next(r for r in report.cohort_exposure if r.cohort_id == "drama")
        assert drama_record.item_count == 0
        assert drama_record.exposure_share == 0.0
        assert drama_record.gap < 0.0

    def test_empty_items_returns_empty_exposure(self) -> None:
        engine = FairnessAuditEngine()
        report = engine.audit([], {"action": 10}, {})
        assert report.cohort_exposure == []


class TestLongTailCoverage:
    def test_coverage_with_long_tail_items(self) -> None:
        engine = FairnessAuditEngine(FairnessAuditConfig(long_tail_percentile=0.50))
        items = [_make_item("low_1", "Niche A"), _make_item("high_1", "Popular B")]
        popularity = {
            "low_1": 0.01,
            "low_2": 0.02,
            "low_3": 0.03,
            "high_1": 0.90,
            "high_2": 0.95,
        }
        report = engine.audit(items, {}, popularity)
        assert report.long_tail_coverage.recommended_long_tail_items >= 1
        assert report.long_tail_coverage.coverage_fraction > 0.0

    def test_no_long_tail_items_in_recommendations(self) -> None:
        engine = FairnessAuditEngine(FairnessAuditConfig(long_tail_percentile=0.20))
        items = [_make_item("high_1", "Popular A"), _make_item("high_2", "Popular B")]
        popularity = {
            "low_1": 0.01,
            "high_1": 0.90,
            "high_2": 0.95,
        }
        report = engine.audit(items, {}, popularity)
        assert report.long_tail_coverage.long_tail_ratio == 0.0

    def test_empty_popularity_returns_zeros(self) -> None:
        engine = FairnessAuditEngine()
        items = [_make_item("1", "A")]
        report = engine.audit(items, {}, {})
        assert report.long_tail_coverage.total_long_tail_items == 0
        assert report.long_tail_coverage.coverage_fraction == 0.0


class TestProviderConcentration:
    def test_equal_providers_low_gini(self) -> None:
        engine = FairnessAuditEngine()
        items = _balanced_items()
        report = engine.audit(items, {}, {})
        assert report.provider_concentration.gini_coefficient < 0.2

    def test_single_provider_high_concentration(self) -> None:
        engine = FairnessAuditEngine()
        items = [
            _make_item("1", "A", provider="monopoly"),
            _make_item("2", "B", provider="monopoly"),
            _make_item("3", "C", provider="monopoly"),
        ]
        report = engine.audit(items, {}, {})
        assert report.provider_concentration.top_provider_share == 1.0

    def test_empty_items_zero_concentration(self) -> None:
        engine = FairnessAuditEngine()
        report = engine.audit([], {}, {})
        assert report.provider_concentration.provider_count == 0
        assert report.provider_concentration.gini_coefficient == 0.0


class TestFairnessAuditReport:
    def test_passes_when_within_thresholds(self) -> None:
        engine = FairnessAuditEngine(
            FairnessAuditConfig(max_exposure_gap=0.50, min_long_tail_coverage=0.0)
        )
        items = _balanced_items()
        catalog_counts = {"action": 2, "drama": 2, "comedy": 2}
        report = engine.audit(items, catalog_counts, {})
        assert report.passes_exposure_threshold is True

    def test_fails_when_gap_exceeds_threshold(self) -> None:
        engine = FairnessAuditEngine(
            FairnessAuditConfig(max_exposure_gap=0.01)
        )
        items = _skewed_items()
        catalog_counts = {"action": 33, "drama": 33, "comedy": 34}
        report = engine.audit(items, catalog_counts, {})
        assert report.passes_exposure_threshold is False

    def test_coverage_threshold_enforcement(self) -> None:
        engine = FairnessAuditEngine(
            FairnessAuditConfig(min_long_tail_coverage=0.99)
        )
        items = [_make_item("high_1", "Popular")]
        popularity = {"high_1": 0.95, "low_1": 0.01}
        report = engine.audit(items, {}, popularity)
        assert report.passes_coverage_threshold is False

    def test_threshold_config_is_recorded(self) -> None:
        config = FairnessAuditConfig(max_exposure_gap=0.15, min_long_tail_coverage=0.10)
        engine = FairnessAuditEngine(config)
        report = engine.audit([], {}, {})
        assert report.threshold_config["max_exposure_gap"] == 0.15
        assert report.threshold_config["min_long_tail_coverage"] == 0.10


class TestCustomCohortAndProviderKeys:
    def test_custom_cohort_key(self) -> None:
        config = FairnessAuditConfig(cohort_key="category")
        engine = FairnessAuditEngine(config)
        items = [
            RankedItem(item_id="1", title="A", score=1.0, explanation="", metadata={"category": "electronics"}),
            RankedItem(item_id="2", title="B", score=1.0, explanation="", metadata={"category": "books"}),
        ]
        catalog_counts = {"electronics": 50, "books": 50}
        report = engine.audit(items, catalog_counts, {})
        cohort_ids = {r.cohort_id for r in report.cohort_exposure}
        assert "electronics" in cohort_ids
        assert "books" in cohort_ids

    def test_custom_provider_key(self) -> None:
        config = FairnessAuditConfig(provider_key="creator")
        engine = FairnessAuditEngine(config)
        items = [
            RankedItem(item_id="1", title="A", score=1.0, explanation="", metadata={"creator": "alice"}),
            RankedItem(item_id="2", title="B", score=1.0, explanation="", metadata={"creator": "bob"}),
        ]
        report = engine.audit(items, {}, {})
        assert report.provider_concentration.provider_count == 2

    def test_missing_metadata_falls_back_to_unknown(self) -> None:
        engine = FairnessAuditEngine()
        items = [
            RankedItem(item_id="1", title="A", score=1.0, explanation="", metadata={}),
        ]
        catalog_counts = {"unknown": 10}
        report = engine.audit(items, catalog_counts, {})
        assert report.cohort_exposure[0].cohort_id == "unknown"
