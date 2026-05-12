from __future__ import annotations

from pathlib import Path

from flask import Flask
import pytest

from src.api import routes as routes_module
from src.api.routes import create_blueprint
from src.config import Settings
from src.domain.models import RankedItem
from src.observability.metrics import MetricsRegistry
from src.serving.dependencies import ServingDependencies
from src.serving.reliability import CircuitBreaker, RateLimiter
from src.serving.security import SecurityGuard, SecurityGuardConfig


class _DummyRecommendationService:
    def __init__(self, _data_path: Path, **kwargs) -> None:
        pass

    def suggestions(self) -> list[str]:
        return ["avatar"]

    def recommend_ranked(self, movie_title: str, context=None) -> list[RankedItem]:
        return [
            RankedItem(
                item_id="1",
                title=f"{movie_title} sequel",
                score=0.88,
                explanation="ranked",
                metadata={"signals": {"text": 0.72, "multimodal": 0.16}},
            )
        ]


class _DummySentimentService:
    def __init__(self, *args, **kwargs) -> None:
        pass


class _DummyTmdbService:
    def __init__(self, *args, **kwargs) -> None:
        pass


@pytest.fixture
def _settings() -> Settings:
    return Settings(
        app_env="test",
        debug=False,
        host="127.0.0.1",
        port=5000,
        tmdb_api_key="",
        request_timeout_seconds=3,
        profile="lean",
        dataset_snapshot_path="datasets/dataset_snapshot.json",
        feature_store_path="",
        rate_limit_capacity=100,
        rate_limit_refill_per_second=100,
        tmdb_breaker_failures=5,
        tmdb_breaker_timeout_seconds=30,
        sentiment_breaker_failures=3,
        sentiment_breaker_timeout_seconds=20,
        enable_model_encoders=False,
        text_encoder_model_id="test",
        florence_model_id="test",
        vector_index_path="",
        ranking_model_path="",
        max_query_length=180,
        max_payload_field_length=6000,
        max_payload_fields=80,
        musicbrainz_breaker_failures=3,
        musicbrainz_breaker_timeout_seconds=30,
        openlibrary_breaker_failures=3,
        openlibrary_breaker_timeout_seconds=30,
        music_catalog_path="datasets/music_catalog.csv",
        books_catalog_path="datasets/books_catalog.csv",
        internal_gate_token="",
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, _settings: Settings):
    monkeypatch.setattr(routes_module, "RecommendationService", _DummyRecommendationService)
    monkeypatch.setattr(routes_module, "SentimentService", _DummySentimentService)
    monkeypatch.setattr(routes_module, "TmdbService", _DummyTmdbService)

    from src.serving.cache_manager import build_cache_manager
    from src.model_registry.versioning import ModelVersionRegistry

    deps = ServingDependencies(
        metrics=MetricsRegistry(),
        rate_limiter=RateLimiter(capacity=100, refill_per_second=100),
        tmdb_breaker=CircuitBreaker(),
        sentiment_breaker=CircuitBreaker(),
        musicbrainz_breaker=CircuitBreaker(),
        openlibrary_breaker=CircuitBreaker(),
        security_guard=SecurityGuard(SecurityGuardConfig()),
        cache_manager=build_cache_manager("lean"),
        model_registry=ModelVersionRegistry(),
    )

    app = Flask(__name__)
    app.register_blueprint(create_blueprint(Path.cwd(), deps, _settings))
    return app.test_client()


def test_explanation_endpoint_returns_rationale_payload(client) -> None:
    response = client.post(
        "/api/recommend/explanations",
        json={"title": "avatar"},
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["query_title"] == "avatar"
    assert "summary" in body
    assert body["items"][0]["title"] == "avatar sequel"
    assert "primary_signals" in body["items"][0]


def test_explanation_endpoint_blocks_pii_payload(client) -> None:
    response = client.post(
        "/api/recommend/explanations",
        json={"title": "email me at person@example.com"},
    )

    assert response.status_code == 400
    body = response.get_json()
    assert body["error"] == "invalid request"
    assert body["reason"] == "pii_detected"


@pytest.fixture
def _settings_gated() -> Settings:
    return Settings(
        app_env="test",
        debug=False,
        host="127.0.0.1",
        port=5000,
        tmdb_api_key="",
        request_timeout_seconds=3,
        profile="lean",
        dataset_snapshot_path="datasets/dataset_snapshot.json",
        feature_store_path="",
        rate_limit_capacity=100,
        rate_limit_refill_per_second=100,
        tmdb_breaker_failures=5,
        tmdb_breaker_timeout_seconds=30,
        sentiment_breaker_failures=3,
        sentiment_breaker_timeout_seconds=20,
        enable_model_encoders=False,
        text_encoder_model_id="test",
        florence_model_id="test",
        vector_index_path="",
        ranking_model_path="",
        max_query_length=180,
        max_payload_field_length=6000,
        max_payload_fields=80,
        musicbrainz_breaker_failures=3,
        musicbrainz_breaker_timeout_seconds=30,
        openlibrary_breaker_failures=3,
        openlibrary_breaker_timeout_seconds=30,
        music_catalog_path="datasets/music_catalog.csv",
        books_catalog_path="datasets/books_catalog.csv",
        internal_gate_token="ci-gate-token-fixed",
    )


@pytest.fixture
def client_gated(monkeypatch: pytest.MonkeyPatch, _settings_gated: Settings):
    monkeypatch.setattr(routes_module, "RecommendationService", _DummyRecommendationService)
    monkeypatch.setattr(routes_module, "SentimentService", _DummySentimentService)
    monkeypatch.setattr(routes_module, "TmdbService", _DummyTmdbService)

    from src.serving.cache_manager import build_cache_manager
    from src.model_registry.versioning import ModelVersionRegistry

    deps = ServingDependencies(
        metrics=MetricsRegistry(),
        rate_limiter=RateLimiter(capacity=100, refill_per_second=100),
        tmdb_breaker=CircuitBreaker(),
        sentiment_breaker=CircuitBreaker(),
        musicbrainz_breaker=CircuitBreaker(),
        openlibrary_breaker=CircuitBreaker(),
        security_guard=SecurityGuard(SecurityGuardConfig()),
        cache_manager=build_cache_manager("lean"),
        model_registry=ModelVersionRegistry(),
    )

    app = Flask(__name__)
    app.register_blueprint(create_blueprint(Path.cwd(), deps, _settings_gated))
    return app.test_client()


def test_api_gate_blocks_without_header(client_gated) -> None:
    response = client_gated.post(
        "/api/recommend/explanations",
        json={"title": "avatar"},
    )
    assert response.status_code == 403


def test_api_gate_allows_with_valid_header(client_gated) -> None:
    response = client_gated.post(
        "/api/recommend/explanations",
        json={"title": "avatar"},
        headers={"X-Admin-Access-Token": "ci-gate-token-fixed"},
    )
    assert response.status_code == 200
