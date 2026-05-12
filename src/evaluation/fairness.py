from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Any, Iterable

from src.domain.models import RankedItem


@dataclass(frozen=True)
class CohortExposureRecord:
    """A single cohort's exposure statistics within a batch of impressions.

    ``exposure_share`` is the fraction of total slot positions occupied
    by items belonging to this cohort.  ``ideal_share`` is the target
    share based on the cohort's representation in the catalog.
    ``gap`` is the signed difference ``exposure_share - ideal_share``,
    where negative values indicate under exposure.
    """
    cohort_id: str
    item_count: int
    exposure_share: float
    ideal_share: float
    gap: float


@dataclass(frozen=True)
class LongTailCoverageRecord:
    """Measures how well the recommendation surface covers long tail
    items that have low historical popularity.

    ``long_tail_ratio`` is the fraction of recommended items that fall
    in the long tail segment.  ``total_long_tail_items`` is the size
    of the long tail pool in the catalog.  ``coverage_fraction`` is
    the fraction of that pool surfaced at least once.
    """
    long_tail_ratio: float
    total_long_tail_items: int
    recommended_long_tail_items: int
    coverage_fraction: float


@dataclass(frozen=True)
class ProviderConcentrationRecord:
    """Gini coefficient based measure of how evenly recommendation
    exposure is distributed across content providers.

    A Gini of 0.0 means perfectly equal distribution.  A Gini of 1.0
    means all exposure goes to a single provider.
    """
    provider_count: int
    gini_coefficient: float
    top_provider_share: float


@dataclass(frozen=True)
class FairnessAuditReport:
    """The complete output of a fairness audit pass.

    This report is designed to be stored alongside experiment artifacts
    so that every model evaluation includes a fairness accounting.
    """
    cohort_exposure: list[CohortExposureRecord]
    long_tail_coverage: LongTailCoverageRecord
    provider_concentration: ProviderConcentrationRecord
    exposure_parity_gap: float
    passes_exposure_threshold: bool
    passes_coverage_threshold: bool
    threshold_config: dict[str, float]


@dataclass(frozen=True)
class FairnessAuditConfig:
    """Configurable thresholds for the fairness audit engine.

    ``max_exposure_gap``: the maximum absolute gap between any cohort's
    actual exposure share and its ideal share.  Values above this cause
    the audit to fail.

    ``min_long_tail_coverage``: the minimum fraction of long tail items
    that must appear in recommendations for the audit to pass.

    ``long_tail_percentile``: items below this popularity percentile
    are classified as long tail.

    ``cohort_key``: the metadata key used to assign items to cohorts.

    ``provider_key``: the metadata key used to identify the content
    provider or creator of each item.
    """
    max_exposure_gap: float = 0.10
    min_long_tail_coverage: float = 0.05
    long_tail_percentile: float = 0.20
    cohort_key: str = "genre"
    provider_key: str = "provider"


def _gini_coefficient(values: list[float]) -> float:
    """Compute the Gini coefficient for a list of non negative values.

    Uses the relative mean absolute difference formula.
    """
    if not values:
        return 0.0
    n = len(values)
    if n == 1:
        return 0.0
    total = sum(values)
    if total <= 0.0:
        return 0.0
    sorted_values = sorted(values)
    cumulative = 0.0
    for i, v in enumerate(sorted_values):
        cumulative += (2.0 * (i + 1) - n - 1) * v
    return cumulative / (n * total)


class FairnessAuditEngine:
    """Engine for auditing recommendation fairness across cohorts,
    long tail coverage, and provider concentration.

    The engine operates on batches of ``RankedItem`` impressions and
    compares them against catalog level distributions to detect
    systematic under or over exposure.

    Usage::

        engine = FairnessAuditEngine(config=FairnessAuditConfig())
        report = engine.audit(
            recommended_items=items,
            catalog_cohort_counts={"action": 120, "drama": 80, "comedy": 50},
            catalog_popularity_scores={"item_1": 0.95, "item_2": 0.02, ...},
            catalog_provider_counts={"studio_a": 40, "studio_b": 60},
        )
    """

    def __init__(self, config: FairnessAuditConfig | None = None) -> None:
        self._config = config or FairnessAuditConfig()

    @property
    def config(self) -> FairnessAuditConfig:
        return self._config

    def audit(
        self,
        recommended_items: Iterable[RankedItem],
        catalog_cohort_counts: dict[str, int],
        catalog_popularity_scores: dict[str, float],
        catalog_provider_counts: dict[str, int] | None = None,
    ) -> FairnessAuditReport:
        items = list(recommended_items)
        cohort_exposure = self._compute_cohort_exposure(items, catalog_cohort_counts)
        long_tail = self._compute_long_tail_coverage(items, catalog_popularity_scores)
        provider_concentration = self._compute_provider_concentration(
            items, catalog_provider_counts or {}
        )
        max_gap = max((abs(record.gap) for record in cohort_exposure), default=0.0)
        passes_exposure = max_gap <= self._config.max_exposure_gap
        passes_coverage = long_tail.coverage_fraction >= self._config.min_long_tail_coverage

        return FairnessAuditReport(
            cohort_exposure=cohort_exposure,
            long_tail_coverage=long_tail,
            provider_concentration=provider_concentration,
            exposure_parity_gap=float(max_gap),
            passes_exposure_threshold=passes_exposure,
            passes_coverage_threshold=passes_coverage,
            threshold_config={
                "max_exposure_gap": self._config.max_exposure_gap,
                "min_long_tail_coverage": self._config.min_long_tail_coverage,
                "long_tail_percentile": self._config.long_tail_percentile,
            },
        )

    def _extract_cohort(self, item: RankedItem) -> str:
        metadata = item.metadata if isinstance(item.metadata, dict) else {}
        value = metadata.get(self._config.cohort_key)
        if value:
            return str(value)
        return "unknown"

    def _extract_provider(self, item: RankedItem) -> str:
        metadata = item.metadata if isinstance(item.metadata, dict) else {}
        value = metadata.get(self._config.provider_key)
        if value:
            return str(value)
        return "unknown"

    def _compute_cohort_exposure(
        self,
        items: list[RankedItem],
        catalog_cohort_counts: dict[str, int],
    ) -> list[CohortExposureRecord]:
        if not items:
            return []

        recommended_cohorts: dict[str, int] = defaultdict(int)
        for item in items:
            cohort = self._extract_cohort(item)
            recommended_cohorts[cohort] += 1

        total_recommended = len(items)
        total_catalog = sum(catalog_cohort_counts.values()) if catalog_cohort_counts else 1
        if total_catalog <= 0:
            total_catalog = 1

        all_cohorts = set(recommended_cohorts.keys()) | set(catalog_cohort_counts.keys())
        records: list[CohortExposureRecord] = []
        for cohort_id in sorted(all_cohorts):
            item_count = recommended_cohorts.get(cohort_id, 0)
            exposure_share = item_count / total_recommended if total_recommended > 0 else 0.0
            ideal_share = catalog_cohort_counts.get(cohort_id, 0) / total_catalog
            gap = exposure_share - ideal_share
            records.append(
                CohortExposureRecord(
                    cohort_id=cohort_id,
                    item_count=item_count,
                    exposure_share=float(exposure_share),
                    ideal_share=float(ideal_share),
                    gap=float(gap),
                )
            )
        return records

    def _compute_long_tail_coverage(
        self,
        items: list[RankedItem],
        catalog_popularity_scores: dict[str, float],
    ) -> LongTailCoverageRecord:
        if not catalog_popularity_scores:
            return LongTailCoverageRecord(
                long_tail_ratio=0.0,
                total_long_tail_items=0,
                recommended_long_tail_items=0,
                coverage_fraction=0.0,
            )

        sorted_scores = sorted(catalog_popularity_scores.values())
        threshold_index = int(len(sorted_scores) * self._config.long_tail_percentile)
        threshold_index = max(0, min(threshold_index, len(sorted_scores) - 1))
        popularity_threshold = sorted_scores[threshold_index]

        long_tail_items = {
            item_id
            for item_id, score in catalog_popularity_scores.items()
            if score <= popularity_threshold
        }
        total_long_tail = len(long_tail_items)

        recommended_ids = {item.item_id for item in items}
        recommended_long_tail = recommended_ids & long_tail_items
        recommended_long_tail_count = len(recommended_long_tail)

        long_tail_ratio = recommended_long_tail_count / len(items) if items else 0.0
        coverage = recommended_long_tail_count / total_long_tail if total_long_tail > 0 else 0.0

        return LongTailCoverageRecord(
            long_tail_ratio=float(long_tail_ratio),
            total_long_tail_items=total_long_tail,
            recommended_long_tail_items=recommended_long_tail_count,
            coverage_fraction=float(coverage),
        )

    def _compute_provider_concentration(
        self,
        items: list[RankedItem],
        catalog_provider_counts: dict[str, int],
    ) -> ProviderConcentrationRecord:
        if not items:
            return ProviderConcentrationRecord(
                provider_count=0,
                gini_coefficient=0.0,
                top_provider_share=0.0,
            )

        provider_exposure: dict[str, int] = defaultdict(int)
        for item in items:
            provider = self._extract_provider(item)
            provider_exposure[provider] += 1

        exposure_values = list(provider_exposure.values())
        total = sum(exposure_values)
        gini = _gini_coefficient(exposure_values)
        top_share = max(exposure_values) / total if total > 0 else 0.0

        return ProviderConcentrationRecord(
            provider_count=len(provider_exposure),
            gini_coefficient=float(gini),
            top_provider_share=float(top_share),
        )
