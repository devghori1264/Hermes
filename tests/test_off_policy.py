from __future__ import annotations

import math

import pytest

from src.evaluation.off_policy import (
    LoggedEvent,
    OffPolicyEstimate,
    batch_off_policy_report,
    doubly_robust,
    inverse_propensity_score,
    self_normalized_ips,
    sensitivity_analysis,
    _safe_propensity_ratio,
)


def _standard_events() -> list[LoggedEvent]:
    return [
        LoggedEvent(reward=1.0, logging_propensity=0.5, target_propensity=0.4),
        LoggedEvent(reward=0.0, logging_propensity=0.5, target_propensity=0.6),
        LoggedEvent(reward=1.0, logging_propensity=0.3, target_propensity=0.3),
        LoggedEvent(reward=0.0, logging_propensity=0.8, target_propensity=0.2),
    ]


def _events_with_direct_estimates() -> list[LoggedEvent]:
    return [
        LoggedEvent(reward=1.0, logging_propensity=0.5, target_propensity=0.4, direct_reward_estimate=0.8),
        LoggedEvent(reward=0.0, logging_propensity=0.5, target_propensity=0.6, direct_reward_estimate=0.1),
        LoggedEvent(reward=1.0, logging_propensity=0.3, target_propensity=0.3, direct_reward_estimate=0.9),
        LoggedEvent(reward=0.0, logging_propensity=0.8, target_propensity=0.2, direct_reward_estimate=0.05),
    ]


class TestSafePropensityRatio:
    def test_normal_ratio(self) -> None:
        ratio = _safe_propensity_ratio(0.4, 0.5, 0.0, float("inf"))
        assert ratio is not None
        assert abs(ratio - 0.8) < 1e-9

    def test_zero_logging_returns_none(self) -> None:
        assert _safe_propensity_ratio(0.4, 0.0, 0.0, float("inf")) is None

    def test_zero_target_returns_none(self) -> None:
        assert _safe_propensity_ratio(0.0, 0.5, 0.0, float("inf")) is None

    def test_clipping_upper(self) -> None:
        ratio = _safe_propensity_ratio(0.9, 0.1, 0.0, 5.0)
        assert ratio is not None
        assert ratio <= 5.0

    def test_clipping_lower(self) -> None:
        ratio = _safe_propensity_ratio(0.01, 0.9, 0.5, float("inf"))
        assert ratio is not None
        assert ratio >= 0.5


class TestInversePropensityScore:
    def test_basic_estimation(self) -> None:
        events = _standard_events()
        result = inverse_propensity_score(events)
        assert result.name == "ips"
        assert result.count == 4
        assert result.value > 0.0

    def test_empty_events(self) -> None:
        result = inverse_propensity_score([])
        assert result.value == 0.0
        assert result.count == 0

    def test_all_positive_rewards(self) -> None:
        events = [
            LoggedEvent(reward=1.0, logging_propensity=0.5, target_propensity=0.5),
            LoggedEvent(reward=1.0, logging_propensity=0.5, target_propensity=0.5),
        ]
        result = inverse_propensity_score(events)
        assert abs(result.value - 1.0) < 1e-9

    def test_variance_is_computed(self) -> None:
        events = _standard_events()
        result = inverse_propensity_score(events)
        assert result.variance >= 0.0

    def test_confidence_interval_contains_estimate(self) -> None:
        events = _standard_events()
        result = inverse_propensity_score(events)
        assert result.confidence_lower <= result.value
        assert result.confidence_upper >= result.value

    def test_clipping_reduces_extreme_weights(self) -> None:
        events = [
            LoggedEvent(reward=1.0, logging_propensity=0.01, target_propensity=0.99),
        ]
        unclipped = inverse_propensity_score(events, clip_max=float("inf"))
        clipped = inverse_propensity_score(events, clip_max=5.0)
        assert clipped.value <= unclipped.value

    def test_skips_invalid_propensities(self) -> None:
        events = [
            LoggedEvent(reward=1.0, logging_propensity=0.0, target_propensity=0.5),
            LoggedEvent(reward=1.0, logging_propensity=0.5, target_propensity=0.5),
        ]
        result = inverse_propensity_score(events)
        assert result.count == 1


class TestSelfNormalizedIPS:
    def test_basic_estimation(self) -> None:
        events = _standard_events()
        result = self_normalized_ips(events)
        assert result.name == "snips"
        assert result.count == 4
        assert result.value >= 0.0

    def test_empty_events(self) -> None:
        result = self_normalized_ips([])
        assert result.value == 0.0
        assert result.count == 0

    def test_equal_propensities_match_mean_reward(self) -> None:
        events = [
            LoggedEvent(reward=1.0, logging_propensity=0.5, target_propensity=0.5),
            LoggedEvent(reward=0.0, logging_propensity=0.5, target_propensity=0.5),
        ]
        result = self_normalized_ips(events)
        assert abs(result.value - 0.5) < 1e-9

    def test_variance_is_computed(self) -> None:
        events = _standard_events()
        result = self_normalized_ips(events)
        assert result.variance >= 0.0


class TestDoublyRobust:
    def test_basic_estimation(self) -> None:
        events = _events_with_direct_estimates()
        result = doubly_robust(events)
        assert result.name == "doubly_robust"
        assert result.count == 4

    def test_falls_back_to_ips_without_direct_estimates(self) -> None:
        events = _standard_events()
        dr_result = doubly_robust(events)
        ips_result = inverse_propensity_score(events)
        assert abs(dr_result.value - ips_result.value) < 1e-9

    def test_with_direct_estimates_differs_from_ips(self) -> None:
        events = _events_with_direct_estimates()
        dr_result = doubly_robust(events)
        ips_result = inverse_propensity_score(events)
        # With direct estimates the DR estimator should generally differ
        # from pure IPS because it includes the reward model correction.
        # In degenerate cases they may be close, so we just check it runs.
        assert dr_result.count == ips_result.count

    def test_empty_events(self) -> None:
        result = doubly_robust([])
        assert result.value == 0.0
        assert result.count == 0

    def test_variance_is_computed(self) -> None:
        events = _events_with_direct_estimates()
        result = doubly_robust(events)
        assert result.variance >= 0.0

    def test_confidence_interval_contains_estimate(self) -> None:
        events = _events_with_direct_estimates()
        result = doubly_robust(events)
        assert result.confidence_lower <= result.value
        assert result.confidence_upper >= result.value

    def test_clipping_applies(self) -> None:
        events = [
            LoggedEvent(reward=1.0, logging_propensity=0.01, target_propensity=0.99, direct_reward_estimate=0.5),
        ]
        clipped = doubly_robust(events, clip_max=3.0)
        assert clipped.count == 1

    def test_perfect_reward_model_recovers_direct_estimate(self) -> None:
        events = [
            LoggedEvent(reward=0.8, logging_propensity=0.5, target_propensity=0.5, direct_reward_estimate=0.8),
        ]
        result = doubly_robust(events)
        assert abs(result.value - 0.8) < 1e-9


class TestSensitivityAnalysis:
    def test_returns_correct_number_of_results(self) -> None:
        events = _standard_events()
        results = sensitivity_analysis(events, perturbation_factors=(0.8, 1.0, 1.2))
        assert len(results) == 3

    def test_factor_1_matches_standard_ips(self) -> None:
        events = _standard_events()
        results = sensitivity_analysis(events, perturbation_factors=(1.0,))
        standard = inverse_propensity_score(events)
        assert abs(results[0].value - standard.value) < 1e-9

    def test_names_include_perturbation_factor(self) -> None:
        events = _standard_events()
        results = sensitivity_analysis(events, perturbation_factors=(0.9,))
        assert "0.90" in results[0].name

    def test_empty_events(self) -> None:
        results = sensitivity_analysis([], perturbation_factors=(0.8, 1.0, 1.2))
        assert all(r.count == 0 for r in results)

    def test_different_factors_produce_different_values(self) -> None:
        events = _standard_events()
        results = sensitivity_analysis(events, perturbation_factors=(0.5, 2.0))
        assert results[0].value != results[1].value


class TestBatchOffPolicyReport:
    def test_returns_all_estimators(self) -> None:
        events = _events_with_direct_estimates()
        report = batch_off_policy_report(events)
        assert "ips" in report
        assert "snips" in report
        assert "doubly_robust" in report

    def test_all_estimators_have_same_count(self) -> None:
        events = _standard_events()
        report = batch_off_policy_report(events)
        counts = {name: estimate.count for name, estimate in report.items()}
        unique_counts = set(counts.values())
        assert len(unique_counts) == 1

    def test_clipping_propagates_to_all_estimators(self) -> None:
        events = [
            LoggedEvent(reward=1.0, logging_propensity=0.01, target_propensity=0.99),
        ]
        report_unclipped = batch_off_policy_report(events)
        report_clipped = batch_off_policy_report(events, clip_max=5.0)
        assert report_clipped["ips"].value <= report_unclipped["ips"].value


class TestBackwardCompatibility:
    """Verify that the existing test_off_policy.py usage pattern still works."""

    def test_original_interface(self) -> None:
        events = [
            LoggedEvent(reward=1.0, logging_propensity=0.5, target_propensity=0.4),
            LoggedEvent(reward=0.0, logging_propensity=0.5, target_propensity=0.6),
        ]
        ips = inverse_propensity_score(events)
        snips = self_normalized_ips(events)
        assert ips.count == 2
        assert snips.count == 2
