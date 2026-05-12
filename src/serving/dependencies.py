from __future__ import annotations

from dataclasses import dataclass

from src.observability.metrics import MetricsRegistry
from src.serving.reliability import CircuitBreaker, RateLimiter
from src.serving.security import SecurityGuard
from src.serving.cache_manager import CacheManager
from src.model_registry.versioning import ModelVersionRegistry


@dataclass(frozen=True)
class ServingDependencies:
    metrics: MetricsRegistry
    rate_limiter: RateLimiter
    tmdb_breaker: CircuitBreaker
    sentiment_breaker: CircuitBreaker
    musicbrainz_breaker: CircuitBreaker
    openlibrary_breaker: CircuitBreaker
    security_guard: SecurityGuard
    cache_manager: CacheManager
    model_registry: ModelVersionRegistry


