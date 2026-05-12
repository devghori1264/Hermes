import pytest

from src.ranking.delayed_feedback import DelayDistribution, DelayedFeedbackAdjuster, DelayFeedbackConfig


def test_adjuster_weights_negative() -> None:
    distribution = DelayDistribution.from_histogram({1.0: 80, 24.0: 20})
    adjuster = DelayedFeedbackAdjuster(distribution)

    short_delay = adjuster.adjust(label=0, delay_hours=1.0)
    long_delay = adjuster.adjust(label=0, delay_hours=24.0)

    assert short_delay.weight <= long_delay.weight
    assert short_delay.reason == "delayed_negative:cdf"


def test_adjuster_positive_label() -> None:
    distribution = DelayDistribution.from_histogram({1.0: 80, 24.0: 20})
    adjuster = DelayedFeedbackAdjuster(distribution)
    result = adjuster.adjust(label=1, delay_hours=2.0)
    assert result.weight == 1.0


@pytest.mark.parametrize(
    "strategy",
    DelayedFeedbackAdjuster.supported_strategies(),
)
def test_adjuster_strategy_variants(strategy: str) -> None:
    distribution = DelayDistribution.from_histogram({1.0: 80, 24.0: 20})
    adjuster = DelayedFeedbackAdjuster(distribution, DelayFeedbackConfig(strategy=strategy))
    short_delay = adjuster.adjust(label=0, delay_hours=1.0)
    long_delay = adjuster.adjust(label=0, delay_hours=24.0)
    assert short_delay.weight <= long_delay.weight
    assert short_delay.reason == f"delayed_negative:{strategy}"


def test_adjuster_rejects_invalid_strategy() -> None:
    distribution = DelayDistribution.from_histogram({1.0: 80, 24.0: 20})
    with pytest.raises(ValueError, match="unsupported delayed feedback strategy"):
        DelayedFeedbackAdjuster(distribution, DelayFeedbackConfig(strategy="unknown"))
