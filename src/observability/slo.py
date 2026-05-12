from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SLO:
    latency_p95_ms: int
    error_rate_percent: float


PROFILE_SLOS = {
    "lean": SLO(latency_p95_ms=450, error_rate_percent=1.5),
    "mid": SLO(latency_p95_ms=300, error_rate_percent=1.0),
    "aggressive": SLO(latency_p95_ms=220, error_rate_percent=0.7),
}
