from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class DelayBucket:
    max_hours: float
    probability: float


@dataclass(frozen=True)
class DelayDistribution:
    buckets: list[DelayBucket]

    def normalized(self) -> "DelayDistribution":
        total = sum(bucket.probability for bucket in self.buckets)
        if math.isclose(total, 0.0):
            return self
        normalized = [
            DelayBucket(max_hours=bucket.max_hours, probability=bucket.probability / total)
            for bucket in self.buckets
        ]
        return DelayDistribution(buckets=sorted(normalized, key=lambda bucket: bucket.max_hours))

    def cdf(self, hours: float) -> float:
        total = 0.0
        for bucket in self.buckets:
            if hours <= bucket.max_hours:
                total += bucket.probability
                break
            total += bucket.probability
        return min(max(total, 0.0), 1.0)

    def survival(self, hours: float) -> float:
        return max(0.0, 1.0 - self.cdf(hours))

    @classmethod
    def from_histogram(cls, histogram: dict[float, int]) -> "DelayDistribution":
        buckets = [DelayBucket(max_hours=hours, probability=float(count)) for hours, count in histogram.items()]
        return cls(buckets=buckets).normalized()


@dataclass(frozen=True)
class DelayFeedbackConfig:
    negative_weight_floor: float = 0.05
    negative_weight_cap: float = 1.0
    strategy: str = "cdf"
    survival_power: float = 1.5
    defer_power: float = 2.0
    defuse_alpha: float = 0.6


@dataclass(frozen=True)
class AdjustedLabel:
    label: int
    weight: float
    delay_hours: float | None
    reason: str


class DelayedFeedbackAdjuster:
    _SUPPORTED_STRATEGIES = ("cdf", "survival_weighted", "defer", "defuse")

    def __init__(self, distribution: DelayDistribution, config: DelayFeedbackConfig | None = None) -> None:
        self._distribution = distribution.normalized()
        self._config = config or DelayFeedbackConfig()
        self._strategy = self._normalize_strategy(self._config.strategy)

    @classmethod
    def supported_strategies(cls) -> tuple[str, ...]:
        return cls._SUPPORTED_STRATEGIES

    @classmethod
    def _normalize_strategy(cls, strategy: str) -> str:
        normalized = strategy.strip().lower()
        if normalized not in cls._SUPPORTED_STRATEGIES:
            supported = ", ".join(cls._SUPPORTED_STRATEGIES)
            raise ValueError(f"unsupported delayed feedback strategy: {strategy}. supported strategies: {supported}")
        return normalized

    def _strategy_weight(self, cdf: float, survival: float) -> float:
        if self._strategy == "cdf":
            return cdf
        if self._strategy == "survival_weighted":
            return 1.0 - math.pow(survival, max(self._config.survival_power, 0.0))
        if self._strategy == "defer":
            return math.pow(cdf, max(self._config.defer_power, 0.0))
        if self._strategy == "defuse":
            alpha = min(max(self._config.defuse_alpha, 0.0), 1.0)
            defer_like = math.pow(cdf, max(self._config.defer_power, 0.0))
            survival_like = 1.0 - math.pow(survival, max(self._config.survival_power, 0.0))
            return alpha * defer_like + (1.0 - alpha) * survival_like
        return cdf

    def adjust(self, label: int, delay_hours: float | None) -> AdjustedLabel:
        if label == 1:
            return AdjustedLabel(label=1, weight=1.0, delay_hours=delay_hours, reason="positive")
        if delay_hours is None:
            return AdjustedLabel(label=0, weight=self._config.negative_weight_floor, delay_hours=None, reason="missing_delay")
        cdf = self._distribution.cdf(delay_hours)
        survival = self._distribution.survival(delay_hours)
        raw_weight = self._strategy_weight(cdf, survival)
        weight = min(max(raw_weight, self._config.negative_weight_floor), self._config.negative_weight_cap)
        return AdjustedLabel(label=0, weight=weight, delay_hours=delay_hours, reason=f"delayed_negative:{self._strategy}")

    def adjust_many(self, labels: Iterable[int], delays: Iterable[float | None]) -> list[AdjustedLabel]:
        results: list[AdjustedLabel] = []
        for label, delay in zip(labels, delays):
            results.append(self.adjust(label, delay))
        return results
