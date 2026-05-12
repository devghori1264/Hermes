from __future__ import annotations

import pytest

from src.serving.cache_manager import (
    CacheManager,
    CacheMetricsEvent,
    CachePolicy,
    build_cache_key,
    build_cache_manager,
)


class FakeClock:
    """Deterministic clock for testing cache TTL and expiration behavior."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def _basic_manager(clock: FakeClock | None = None) -> CacheManager:
    policies = {
        "items": CachePolicy(ttl_seconds=60, max_entries=5),
        "users": CachePolicy(ttl_seconds=30, max_entries=3),
    }
    return CacheManager(policies=policies, time_fn=clock or FakeClock())


class TestBuildCacheKey:
    def test_deterministic(self) -> None:
        a = build_cache_key("scope", "user_1", "query")
        b = build_cache_key("scope", "user_1", "query")
        assert a == b

    def test_different_parts_differ(self) -> None:
        a = build_cache_key("scope", "user_1")
        b = build_cache_key("scope", "user_2")
        assert a != b

    def test_key_length(self) -> None:
        key = build_cache_key("a", "b", "c")
        assert len(key) == 16

    def test_order_matters(self) -> None:
        a = build_cache_key("x", "y")
        b = build_cache_key("y", "x")
        assert a != b


class TestCacheManagerBasicOperations:
    def test_set_and_get(self) -> None:
        manager = _basic_manager()
        manager.set("items", "k1", "value_1")
        assert manager.get("items", "k1") == "value_1"

    def test_get_missing_key(self) -> None:
        manager = _basic_manager()
        assert manager.get("items", "missing") is None

    def test_get_unknown_scope(self) -> None:
        manager = _basic_manager()
        assert manager.get("nonexistent", "k1") is None

    def test_set_unknown_scope_no_error(self) -> None:
        manager = _basic_manager()
        manager.set("nonexistent", "k1", "v1")

    def test_overwrite_existing_key(self) -> None:
        manager = _basic_manager()
        manager.set("items", "k1", "old")
        manager.set("items", "k1", "new")
        assert manager.get("items", "k1") == "new"


class TestCacheManagerTTL:
    def test_entry_expires_after_ttl(self) -> None:
        clock = FakeClock()
        manager = _basic_manager(clock)
        manager.set("items", "k1", "value")
        clock.advance(61)
        assert manager.get("items", "k1") is None

    def test_entry_alive_before_ttl(self) -> None:
        clock = FakeClock()
        manager = _basic_manager(clock)
        manager.set("items", "k1", "value")
        clock.advance(30)
        assert manager.get("items", "k1") == "value"

    def test_custom_ttl_overrides_policy(self) -> None:
        clock = FakeClock()
        manager = _basic_manager(clock)
        manager.set("items", "k1", "value", ttl_seconds=10)
        clock.advance(11)
        assert manager.get("items", "k1") is None


class TestCacheManagerEviction:
    def test_lru_eviction_when_max_exceeded(self) -> None:
        clock = FakeClock()
        manager = _basic_manager(clock)
        for i in range(6):
            manager.set("items", f"k{i}", f"v{i}")
        assert manager.get("items", "k0") is None
        assert manager.get("items", "k5") == "v5"

    def test_eviction_counter_increments(self) -> None:
        clock = FakeClock()
        manager = _basic_manager(clock)
        for i in range(6):
            manager.set("items", f"k{i}", f"v{i}")
        snap = manager.snapshot()
        assert snap["items"]["evictions"] >= 1


class TestCacheManagerInvalidation:
    def test_invalidate_single_key(self) -> None:
        manager = _basic_manager()
        manager.set("items", "k1", "v1")
        manager.invalidate("items", "k1")
        assert manager.get("items", "k1") is None

    def test_invalidate_all_keys_in_scope(self) -> None:
        manager = _basic_manager()
        manager.set("items", "k1", "v1")
        manager.set("items", "k2", "v2")
        manager.invalidate("items")
        assert manager.get("items", "k1") is None
        assert manager.get("items", "k2") is None

    def test_invalidate_unknown_scope_no_error(self) -> None:
        manager = _basic_manager()
        manager.invalidate("nonexistent", "k1")

    def test_invalidate_unknown_key_no_error(self) -> None:
        manager = _basic_manager()
        manager.invalidate("items", "not_there")


class TestCacheManagerDisabled:
    def test_disabled_get_returns_none(self) -> None:
        manager = CacheManager({"items": CachePolicy(60, 5)}, enabled=False)
        manager.set("items", "k1", "v1")
        assert manager.get("items", "k1") is None

    def test_disabled_set_is_noop(self) -> None:
        manager = CacheManager({"items": CachePolicy(60, 5)}, enabled=False)
        manager.set("items", "k1", "v1")
        snap = manager.snapshot()
        assert snap["items"]["sets"] == 0


class TestCacheManagerSnapshot:
    def test_snapshot_contains_all_scopes(self) -> None:
        manager = _basic_manager()
        snap = manager.snapshot()
        assert "items" in snap
        assert "users" in snap

    def test_snapshot_tracks_hits_and_misses(self) -> None:
        manager = _basic_manager()
        manager.set("items", "k1", "v1")
        manager.get("items", "k1")
        manager.get("items", "missing")
        snap = manager.snapshot()
        assert snap["items"]["hits"] == 1
        assert snap["items"]["misses"] == 1


class TestCacheManagerHitRate:
    def test_hit_rate_with_mixed_operations(self) -> None:
        manager = _basic_manager()
        manager.set("items", "k1", "v1")
        manager.get("items", "k1")
        manager.get("items", "missing")
        rate = manager.hit_rate("items")
        assert abs(rate - 0.5) < 1e-9

    def test_hit_rate_no_operations(self) -> None:
        manager = _basic_manager()
        assert manager.hit_rate("items") == 0.0

    def test_hit_rate_unknown_scope(self) -> None:
        manager = _basic_manager()
        assert manager.hit_rate("nonexistent") == 0.0


class TestCacheManagerWarmup:
    def test_warmup_preloads_entries(self) -> None:
        manager = _basic_manager()
        count = manager.warmup("items", {"k1": "v1", "k2": "v2"})
        assert count == 2
        assert manager.get("items", "k1") == "v1"
        assert manager.get("items", "k2") == "v2"

    def test_warmup_with_custom_ttl(self) -> None:
        clock = FakeClock()
        manager = _basic_manager(clock)
        manager.warmup("items", {"k1": "v1"}, ttl_seconds=10)
        clock.advance(11)
        assert manager.get("items", "k1") is None

    def test_warmup_disabled_returns_zero(self) -> None:
        manager = CacheManager({"items": CachePolicy(60, 5)}, enabled=False)
        count = manager.warmup("items", {"k1": "v1"})
        assert count == 0

    def test_warmup_unknown_scope_returns_zero(self) -> None:
        manager = _basic_manager()
        count = manager.warmup("nonexistent", {"k1": "v1"})
        assert count == 0


class TestCacheManagerMetricsCallback:
    def test_callback_receives_get_events(self) -> None:
        events: list[CacheMetricsEvent] = []
        manager = CacheManager(
            {"items": CachePolicy(60, 5)},
            time_fn=FakeClock(),
            metrics_callback=events.append,
        )
        manager.set("items", "k1", "v1")
        manager.get("items", "k1")
        manager.get("items", "missing")
        get_events = [e for e in events if e.operation == "get"]
        assert len(get_events) == 2
        assert get_events[0].hit is True
        assert get_events[1].hit is False

    def test_callback_receives_set_events(self) -> None:
        events: list[CacheMetricsEvent] = []
        manager = CacheManager(
            {"items": CachePolicy(60, 5)},
            time_fn=FakeClock(),
            metrics_callback=events.append,
        )
        manager.set("items", "k1", "v1")
        set_events = [e for e in events if e.operation == "set"]
        assert len(set_events) == 1

    def test_callback_receives_invalidation_events(self) -> None:
        events: list[CacheMetricsEvent] = []
        manager = CacheManager(
            {"items": CachePolicy(60, 5)},
            time_fn=FakeClock(),
            metrics_callback=events.append,
        )
        manager.set("items", "k1", "v1")
        manager.invalidate("items", "k1")
        invalidate_events = [e for e in events if e.operation == "invalidate"]
        assert len(invalidate_events) == 1


class TestCacheManagerModelUpdate:
    def test_on_model_update_clears_scope(self) -> None:
        manager = _basic_manager()
        manager.set("items", "k1", "v1")
        manager.on_model_update("items")
        assert manager.get("items", "k1") is None

    def test_on_model_update_clears_all_scopes(self) -> None:
        manager = _basic_manager()
        manager.set("items", "k1", "v1")
        manager.set("users", "k2", "v2")
        manager.on_model_update()
        assert manager.get("items", "k1") is None
        assert manager.get("users", "k2") is None


class TestCacheManagerScopes:
    def test_scopes_returns_registered_names(self) -> None:
        manager = _basic_manager()
        scopes = manager.scopes()
        assert "items" in scopes
        assert "users" in scopes


class TestBuildCacheManager:
    def test_aggressive_profile(self) -> None:
        manager = build_cache_manager("aggressive")
        assert "recommendations" in manager.scopes()
        assert "retrieval_vector" in manager.scopes()
        assert "cold_start" in manager.scopes()

    def test_mid_profile(self) -> None:
        manager = build_cache_manager("mid")
        assert "recommendations" in manager.scopes()

    def test_default_profile(self) -> None:
        manager = build_cache_manager("lean")
        assert "recommendations" in manager.scopes()

    def test_disabled_manager(self) -> None:
        manager = build_cache_manager("lean", enabled=False)
        manager.set("recommendations", "k1", "v1")
        assert manager.get("recommendations", "k1") is None
