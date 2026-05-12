from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import time
from typing import Any, Callable


@dataclass(frozen=True)
class CachePolicy:
    """Configuration for a single cache scope.

    ``ttl_seconds``: maximum age of an entry before it is considered
    expired and removed on the next access or purge cycle.

    ``max_entries``: the upper bound on the number of entries stored
    in this scope.  When exceeded, the least recently used entry is
    evicted.
    """
    ttl_seconds: int
    max_entries: int


@dataclass(frozen=True)
class CacheScopeSnapshot:
    """Point in time statistics for a single cache scope.

    These counters are monotonically increasing over the lifetime
    of the cache.  Downstream observability systems can compute
    rates by differencing consecutive snapshots.
    """
    entries: int
    hits: int
    misses: int
    sets: int
    evictions: int
    expirations: int


@dataclass(frozen=True)
class CacheMetricsEvent:
    """A structured event emitted whenever a cache operation occurs.

    Designed for consumption by an observability layer that records
    per operation metrics for dashboarding and alerting.
    """
    scope: str
    operation: str
    key: str
    hit: bool
    timestamp: float


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float
    updated_at: float
    hit_count: int = 0


class _ScopedCache:
    def __init__(self, policy: CachePolicy, time_fn: Callable[[], float]) -> None:
        self._policy = policy
        self._time_fn = time_fn
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._sets = 0
        self._evictions = 0
        self._expirations = 0

    def get(self, key: str) -> Any | None:
        now = self._time_fn()
        entry = self._entries.get(key)
        if entry is None:
            self._misses += 1
            return None
        if entry.expires_at <= now:
            self._expirations += 1
            self._misses += 1
            del self._entries[key]
            return None
        entry.hit_count += 1
        entry.updated_at = now
        self._entries.move_to_end(key)
        self._hits += 1
        return entry.value

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        now = self._time_fn()
        ttl = max(1, int(ttl_seconds if ttl_seconds is not None else self._policy.ttl_seconds))
        expires_at = now + ttl
        self._entries[key] = _CacheEntry(value=value, expires_at=expires_at, updated_at=now)
        self._entries.move_to_end(key)
        self._sets += 1
        self._evict_if_needed()

    def invalidate(self, key: str) -> bool:
        if key in self._entries:
            del self._entries[key]
            return True
        return False

    def clear(self) -> None:
        self._entries.clear()

    def entry_count(self) -> int:
        return len(self._entries)

    def snapshot(self) -> CacheScopeSnapshot:
        self._purge_expired()
        return CacheScopeSnapshot(
            entries=len(self._entries),
            hits=self._hits,
            misses=self._misses,
            sets=self._sets,
            evictions=self._evictions,
            expirations=self._expirations,
        )

    def _purge_expired(self) -> None:
        now = self._time_fn()
        expired = [key for key, entry in self._entries.items() if entry.expires_at <= now]
        for key in expired:
            del self._entries[key]
            self._expirations += 1

    def _evict_if_needed(self) -> None:
        self._purge_expired()
        while len(self._entries) > self._policy.max_entries:
            self._entries.popitem(last=False)
            self._evictions += 1


def build_cache_key(*parts: str) -> str:
    """Generate a deterministic, collision resistant cache key from
    an ordered sequence of string components.

    Uses SHA256 truncated to 16 hex characters for compactness while
    maintaining negligible collision probability at the scale of
    recommendation request caches.

    Example usage::

        key = build_cache_key("movies", "user_42", "the matrix")
    """
    combined = "|".join(parts)
    digest = hashlib.sha256(combined.encode("utf8")).hexdigest()
    return digest[:16]


class CacheManager:
    """Multi scope cache manager with LRU eviction, TTL expiration,
    metrics emission, and invalidation hooks.

    Each scope (for example ``recommendations``, ``retrieval_vector``,
    ``cold_start``) is independently configured with its own TTL and
    max entries policy.

    The manager can be disabled at construction time so that all
    operations become no ops, which is useful in testing and profiling
    without removing cache call sites.

    A ``metrics_callback`` can be provided to receive structured
    ``CacheMetricsEvent`` objects on every cache operation, allowing
    integration with any observability backend.
    """

    def __init__(
        self,
        policies: dict[str, CachePolicy],
        *,
        enabled: bool = True,
        time_fn: Callable[[], float] | None = None,
        metrics_callback: Callable[[CacheMetricsEvent], None] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self._time_fn = time_fn or time.monotonic
        self._scopes = {scope: _ScopedCache(policy, self._time_fn) for scope, policy in policies.items()}
        self._metrics_callback = metrics_callback

    def _emit(self, scope: str, operation: str, key: str, hit: bool) -> None:
        if self._metrics_callback is not None:
            event = CacheMetricsEvent(
                scope=scope,
                operation=operation,
                key=key,
                hit=hit,
                timestamp=self._time_fn(),
            )
            self._metrics_callback(event)

    def get(self, scope: str, key: str) -> Any | None:
        if not self.enabled:
            return None
        cache = self._scopes.get(scope)
        if cache is None:
            return None
        result = cache.get(key)
        self._emit(scope, "get", key, hit=result is not None)
        return result

    def set(self, scope: str, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        if not self.enabled:
            return
        cache = self._scopes.get(scope)
        if cache is None:
            return
        cache.set(key, value, ttl_seconds=ttl_seconds)
        self._emit(scope, "set", key, hit=False)

    def invalidate(self, scope: str, key: str | None = None) -> None:
        cache = self._scopes.get(scope)
        if cache is None:
            return
        if key is None:
            cache.clear()
            self._emit(scope, "invalidate_all", "", hit=False)
            return
        removed = cache.invalidate(key)
        self._emit(scope, "invalidate", key, hit=removed)

    def warmup(self, scope: str, entries: dict[str, Any], ttl_seconds: int | None = None) -> int:
        """Preload a batch of entries into a cache scope.

        Returns the number of entries actually written.  This is useful
        for populating caches at startup with known high traffic keys
        to avoid a cold start penalty on the first requests.
        """
        if not self.enabled:
            return 0
        cache = self._scopes.get(scope)
        if cache is None:
            return 0
        count = 0
        for key, value in entries.items():
            cache.set(key, value, ttl_seconds=ttl_seconds)
            count += 1
        return count

    def hit_rate(self, scope: str) -> float:
        """Compute the hit rate for a specific cache scope.

        Returns a value between 0.0 and 1.0.  If no operations have
        occurred, returns 0.0.
        """
        cache = self._scopes.get(scope)
        if cache is None:
            return 0.0
        snap = cache.snapshot()
        total = snap.hits + snap.misses
        if total == 0:
            return 0.0
        return snap.hits / total

    def snapshot(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for scope, cache in self._scopes.items():
            snap = cache.snapshot()
            out[scope] = {
                "entries": snap.entries,
                "hits": snap.hits,
                "misses": snap.misses,
                "sets": snap.sets,
                "evictions": snap.evictions,
                "expirations": snap.expirations,
            }
        return out

    def scopes(self) -> list[str]:
        """Return the list of registered cache scope names."""
        return list(self._scopes.keys())

    def on_model_update(self, scope: str | None = None) -> None:
        """Invalidation hook intended to be called when a model artifact
        is updated or rolled back.

        If ``scope`` is provided, only that scope is cleared.  If None,
        all scopes are cleared.  This ensures that stale cached
        predictions from a previous model version are not served after
        a deployment.
        """
        if scope is not None:
            self.invalidate(scope)
            return
        for scope_name in list(self._scopes.keys()):
            self.invalidate(scope_name)


def build_cache_manager(
    profile: str,
    *,
    enabled: bool = True,
    default_ttl_seconds: int = 120,
    metrics_callback: Callable[[CacheMetricsEvent], None] | None = None,
) -> CacheManager:
    ttl = max(1, int(default_ttl_seconds))
    profile_key = profile.strip().lower()
    if profile_key == "aggressive":
        policies = {
            "recommendations": CachePolicy(ttl_seconds=ttl, max_entries=2048),
            "retrieval_vector": CachePolicy(ttl_seconds=max(ttl, 300), max_entries=4096),
            "cold_start": CachePolicy(ttl_seconds=max(ttl, 300), max_entries=1024),
        }
    elif profile_key == "mid":
        policies = {
            "recommendations": CachePolicy(ttl_seconds=ttl, max_entries=1024),
            "retrieval_vector": CachePolicy(ttl_seconds=max(ttl, 240), max_entries=2048),
            "cold_start": CachePolicy(ttl_seconds=max(ttl, 240), max_entries=768),
        }
    else:
        policies = {
            "recommendations": CachePolicy(ttl_seconds=ttl, max_entries=512),
            "retrieval_vector": CachePolicy(ttl_seconds=max(ttl, 180), max_entries=1024),
            "cold_start": CachePolicy(ttl_seconds=max(ttl, 180), max_entries=512),
        }
    return CacheManager(policies=policies, enabled=enabled, metrics_callback=metrics_callback)
