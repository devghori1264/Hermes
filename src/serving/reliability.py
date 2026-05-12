from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable


class CircuitBreakerOpen(RuntimeError):
    pass


@dataclass
class TokenBucket:
    capacity: float
    tokens: float
    refill_rate: float
    updated_at: float

    def refill(self, now: float) -> None:
        delta = max(0.0, now - self.updated_at)
        self.tokens = min(self.capacity, self.tokens + delta * self.refill_rate)
        self.updated_at = now


class RateLimiter:
    def __init__(
        self,
        capacity: int,
        refill_per_second: float,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self._capacity = float(capacity)
        self._refill_per_second = float(refill_per_second)
        self._time_fn = time_fn or time.monotonic
        self._buckets: dict[str, TokenBucket] = {}

    def allow(self, key: str) -> bool:
        now = self._time_fn()
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = TokenBucket(
                capacity=self._capacity,
                tokens=self._capacity,
                refill_rate=self._refill_per_second,
                updated_at=now,
            )
            self._buckets[key] = bucket
        bucket.refill(now)
        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return True
        return False


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_seconds: int = 30,
        half_open_successes: int = 2,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self._failure_threshold = int(failure_threshold)
        self._recovery_timeout = int(recovery_timeout_seconds)
        self._half_open_successes = int(half_open_successes)
        self._time_fn = time_fn or time.monotonic
        self._state = "closed"
        self._failure_count = 0
        self._success_count = 0
        self._opened_at: float | None = None

    def state(self) -> str:
        return self._state

    def call(self, func, *args, **kwargs):
        self._before_call()
        try:
            result = func(*args, **kwargs)
        except Exception as error:
            self._on_failure()
            raise error
        self._on_success()
        return result

    def _before_call(self) -> None:
        if self._state == "closed":
            return
        if self._state == "open":
            if self._opened_at is None:
                raise CircuitBreakerOpen("open")
            elapsed = self._time_fn() - self._opened_at
            if elapsed < self._recovery_timeout:
                raise CircuitBreakerOpen("open")
            self._state = "half_open"
            self._success_count = 0
            return
        if self._state == "half_open":
            return
        raise CircuitBreakerOpen("invalid_state")

    def _on_failure(self) -> None:
        if self._state == "half_open":
            self._trip()
            return
        self._failure_count += 1
        if self._failure_count >= self._failure_threshold:
            self._trip()

    def _on_success(self) -> None:
        if self._state == "half_open":
            self._success_count += 1
            if self._success_count >= self._half_open_successes:
                self._reset()
            return
        self._failure_count = 0

    def _trip(self) -> None:
        self._state = "open"
        self._opened_at = self._time_fn()
        self._failure_count = 0
        self._success_count = 0

    def _reset(self) -> None:
        self._state = "closed"
        self._opened_at = None
        self._failure_count = 0
        self._success_count = 0
