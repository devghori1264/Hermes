from __future__ import annotations

import os
from pathlib import Path

# Disable Hugging Face warnings for unauthenticated requests
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from flask import Flask

from src.api.routes import create_blueprint
from src.config import load_settings
from src.observability.metrics import MetricsRegistry
from src.observability.telemetry import setup_telemetry
from src.serving.dependencies import ServingDependencies
from src.serving.reliability import CircuitBreaker, RateLimiter
from src.serving.security import SecurityGuard, SecurityGuardConfig
from src.serving.cache_manager import CacheManager, build_cache_manager, CacheMetricsEvent
from src.model_registry.versioning import ModelVersionRegistry


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).resolve().parent.parent / "templates"),
        static_folder=str(Path(__file__).resolve().parent.parent / "static"),
    )
    base_path = Path(__file__).resolve().parent.parent
    settings = load_settings()
    metrics = MetricsRegistry()
    
    def on_cache_metric(event: CacheMetricsEvent) -> None:
        if event.hit:
            metrics.increment("cache.hits", tags={"scope": event.scope})
        else:
            metrics.increment("cache.misses", tags={"scope": event.scope})
        metrics.record("cache.latency", event.timestamp, tags={"scope": event.scope, "op": event.operation})

    cache_manager = build_cache_manager(
        profile=settings.profile,
        enabled=True,
        metrics_callback=on_cache_metric
    )

    deps = ServingDependencies(
        metrics=metrics,
        rate_limiter=RateLimiter(
            capacity=settings.rate_limit_capacity,
            refill_per_second=settings.rate_limit_refill_per_second,
        ),
        tmdb_breaker=CircuitBreaker(
            failure_threshold=settings.tmdb_breaker_failures,
            recovery_timeout_seconds=settings.tmdb_breaker_timeout_seconds,
        ),
        sentiment_breaker=CircuitBreaker(
            failure_threshold=settings.sentiment_breaker_failures,
            recovery_timeout_seconds=settings.sentiment_breaker_timeout_seconds,
        ),
        musicbrainz_breaker=CircuitBreaker(
            failure_threshold=settings.musicbrainz_breaker_failures,
            recovery_timeout_seconds=settings.musicbrainz_breaker_timeout_seconds,
        ),
        openlibrary_breaker=CircuitBreaker(
            failure_threshold=settings.openlibrary_breaker_failures,
            recovery_timeout_seconds=settings.openlibrary_breaker_timeout_seconds,
        ),
        security_guard=SecurityGuard(
            SecurityGuardConfig(
                max_query_length=settings.max_query_length,
                max_payload_field_length=settings.max_payload_field_length,
                max_payload_fields=settings.max_payload_fields,
            )
        ),
        cache_manager=cache_manager,
        model_registry=ModelVersionRegistry(),
    )
    setup_telemetry(app, metrics, settings.profile)
    app.register_blueprint(create_blueprint(base_path, deps, settings))
    return app
