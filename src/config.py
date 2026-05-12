from __future__ import annotations

from dataclasses import dataclass
import os


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default) == "1"


def _env_int(name: str, default: str) -> int:
    return int(os.getenv(name, default))


def _env_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    app_env: str
    debug: bool
    host: str
    port: int
    tmdb_api_key: str
    request_timeout_seconds: int
    profile: str
    dataset_snapshot_path: str
    feature_store_path: str
    rate_limit_capacity: int
    rate_limit_refill_per_second: float
    tmdb_breaker_failures: int
    tmdb_breaker_timeout_seconds: int
    sentiment_breaker_failures: int
    sentiment_breaker_timeout_seconds: int
    enable_model_encoders: bool
    text_encoder_model_id: str
    florence_model_id: str
    vector_index_path: str
    ranking_model_path: str
    max_query_length: int
    max_payload_field_length: int
    max_payload_fields: int
    musicbrainz_breaker_failures: int
    musicbrainz_breaker_timeout_seconds: int
    openlibrary_breaker_failures: int
    openlibrary_breaker_timeout_seconds: int
    music_catalog_path: str
    books_catalog_path: str
    internal_gate_token: str


def load_settings() -> Settings:
    return Settings(
        app_env=_env_str("APP_ENV", "dev"),
        debug=_env_bool("FLASK_DEBUG", "0"),
        host=_env_str("APP_HOST", "127.0.0.1"),
        port=_env_int("APP_PORT", "5000"),
        tmdb_api_key=_env_str("TMDB_API_KEY", ""),
        request_timeout_seconds=_env_int("REQUEST_TIMEOUT_SECONDS", "8"),
        profile=_env_str("DEPLOYMENT_PROFILE", "lean"),
        dataset_snapshot_path=_env_str("DATASET_SNAPSHOT_PATH", "datasets/dataset_snapshot.json"),
        feature_store_path=_env_str("FEATURE_STORE_PATH", ""),
        rate_limit_capacity=_env_int("RATE_LIMIT_CAPACITY", "60"),
        rate_limit_refill_per_second=_env_float("RATE_LIMIT_REFILL_PER_SECOND", "30"),
        tmdb_breaker_failures=_env_int("TMDB_BREAKER_FAILURES", "5"),
        tmdb_breaker_timeout_seconds=_env_int("TMDB_BREAKER_TIMEOUT_SECONDS", "30"),
        sentiment_breaker_failures=_env_int("SENTIMENT_BREAKER_FAILURES", "3"),
        sentiment_breaker_timeout_seconds=_env_int("SENTIMENT_BREAKER_TIMEOUT_SECONDS", "20"),
        enable_model_encoders=_env_bool("ENABLE_MODEL_ENCODERS", "0"),
        text_encoder_model_id=_env_str("TEXT_ENCODER_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2"),
        florence_model_id=_env_str("FLORENCE_MODEL_ID", "microsoft/Florence-2-base"),
        vector_index_path=_env_str("VECTOR_INDEX_PATH", ""),
        ranking_model_path=_env_str("RANKING_MODEL_PATH", ""),
        max_query_length=_env_int("MAX_QUERY_LENGTH", "180"),
        max_payload_field_length=_env_int("MAX_PAYLOAD_FIELD_LENGTH", "6000"),
        max_payload_fields=_env_int("MAX_PAYLOAD_FIELDS", "80"),
        musicbrainz_breaker_failures=_env_int("MUSICBRAINZ_BREAKER_FAILURES", "3"),
        musicbrainz_breaker_timeout_seconds=_env_int("MUSICBRAINZ_BREAKER_TIMEOUT_SECONDS", "30"),
        openlibrary_breaker_failures=_env_int("OPENLIBRARY_BREAKER_FAILURES", "3"),
        openlibrary_breaker_timeout_seconds=_env_int("OPENLIBRARY_BREAKER_TIMEOUT_SECONDS", "30"),
        music_catalog_path=_env_str("MUSIC_CATALOG_PATH", "datasets/music_catalog.csv"),
        books_catalog_path=_env_str("BOOKS_CATALOG_PATH", "datasets/books_catalog.csv"),
        internal_gate_token=_env_str("INTERNAL_GATE_TOKEN", ""),
    )
