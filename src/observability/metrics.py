from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import time
from typing import Any


@dataclass
class RequestTrace:
    operation: str
    started_at: float = field(default_factory=time.perf_counter)
    request_id: str | None = None

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000


@dataclass(frozen=True)
class MetricSample:
    name: str
    value: float
    tags: dict[str, str]


@dataclass(frozen=True)
class HistogramSnapshot:
    count: int
    p50: float
    p95: float
    p99: float

    def to_dict(self) -> dict[str, float]:
        return {
            "count": float(self.count),
            "p50": float(self.p50),
            "p95": float(self.p95),
            "p99": float(self.p99),
        }


@dataclass(frozen=True)
class MetricsSnapshot:
    counters: dict[str, int]
    histograms: dict[str, HistogramSnapshot]

    def to_dict(self) -> dict[str, Any]:
        return {
            "counters": {key: int(value) for key, value in self.counters.items()},
            "histograms": {key: value.to_dict() for key, value in self.histograms.items()},
        }


def _metric_key(name: str, tags: dict[str, str] | None) -> str:
    if not tags:
        return name
    parts = [name]
    for key in sorted(tags):
        parts.append(f"{key}={tags[key]}")
    return "|".join(parts)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percentile))
    return float(ordered[max(0, min(index, len(ordered) - 1))])


class MetricsRegistry:
    def __init__(self, max_samples: int = 2048) -> None:
        self._counters: dict[str, int] = defaultdict(int)
        self._histograms: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=max_samples))

    def increment(self, name: str, value: int = 1, tags: dict[str, str] | None = None) -> None:
        key = _metric_key(name, tags)
        self._counters[key] += int(value)

    def record(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        key = _metric_key(name, tags)
        self._histograms[key].append(float(value))

    def snapshot(self) -> MetricsSnapshot:
        histograms: dict[str, HistogramSnapshot] = {}
        for key, values in self._histograms.items():
            samples = list(values)
            histograms[key] = HistogramSnapshot(
                count=len(samples),
                p50=_percentile(samples, 0.50),
                p95=_percentile(samples, 0.95),
                p99=_percentile(samples, 0.99),
            )
        return MetricsSnapshot(counters=dict(self._counters), histograms=histograms)
