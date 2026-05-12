from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable


@dataclass(frozen=True)
class FallbackEvent:
    stage: str
    attempt: int
    error_type: str
    error_message: str
    fallback_used: bool
    fallback_succeeded: bool
    elapsed_ms: float
    timestamp: float


@dataclass(frozen=True)
class StageResult:
    value: Any
    degraded: bool
    events: list[FallbackEvent] = field(default_factory=list)


@dataclass(frozen=True)
class StageSpec:
    name: str
    primary: Callable[[Any], Any]
    fallback: Callable[[Any], Any]
    max_retries: int = 1
    timeout_seconds: float = 5.0


@dataclass(frozen=True)
class ChainResult:
    final_value: Any
    stage_results: list[StageResult]
    total_elapsed_ms: float
    degraded_stages: list[str]
    fully_healthy: bool


class StageExecutor:
    def __init__(self, *, time_fn: Callable[[], float] | None = None) -> None:
        self._time_fn = time_fn or time.perf_counter

    def execute(self, spec: StageSpec, stage_input: Any) -> StageResult:
        start = self._time_fn()
        last_error: Exception | None = None
        events: list[FallbackEvent] = []
        total_attempts = max(1, spec.max_retries + 1)

        for attempt in range(1, total_attempts + 1):
            try:
                result = spec.primary(stage_input)
                return StageResult(value=result, degraded=False, events=events)
            except Exception as exc:
                last_error = exc
                elapsed = (self._time_fn() - start) * 1000.0
                events.append(FallbackEvent(
                    stage=spec.name,
                    attempt=attempt,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    fallback_used=False,
                    fallback_succeeded=False,
                    elapsed_ms=float(elapsed),
                    timestamp=float(self._time_fn()),
                ))

        try:
            fallback_result = spec.fallback(stage_input)
            elapsed = (self._time_fn() - start) * 1000.0
            events.append(FallbackEvent(
                stage=spec.name,
                attempt=total_attempts,
                error_type=type(last_error).__name__ if last_error else "unknown",
                error_message=str(last_error) if last_error else "unknown",
                fallback_used=True,
                fallback_succeeded=True,
                elapsed_ms=float(elapsed),
                timestamp=float(self._time_fn()),
            ))
            return StageResult(value=fallback_result, degraded=True, events=events)
        except Exception as fallback_exc:
            elapsed = (self._time_fn() - start) * 1000.0
            events.append(FallbackEvent(
                stage=spec.name,
                attempt=total_attempts,
                error_type=type(fallback_exc).__name__,
                error_message=str(fallback_exc),
                fallback_used=True,
                fallback_succeeded=False,
                elapsed_ms=float(elapsed),
                timestamp=float(self._time_fn()),
            ))
            raise


class StageFallbackChain:
    def __init__(
        self,
        *,
        time_fn: Callable[[], float] | None = None,
        event_callback: Callable[[FallbackEvent], None] | None = None,
    ) -> None:
        self._time_fn = time_fn or time.perf_counter
        self._executor = StageExecutor(time_fn=self._time_fn)
        self._stages: list[StageSpec] = []
        self._event_callback = event_callback

    def add_stage(self, spec: StageSpec) -> None:
        self._stages.append(spec)

    @property
    def stage_count(self) -> int:
        return len(self._stages)

    @property
    def stage_names(self) -> list[str]:
        return [s.name for s in self._stages]

    def execute(self, initial_input: Any) -> ChainResult:
        chain_start = self._time_fn()
        current_input = initial_input
        stage_results: list[StageResult] = []
        degraded_names: list[str] = []

        for spec in self._stages:
            result = self._executor.execute(spec, current_input)
            stage_results.append(result)
            if result.degraded:
                degraded_names.append(spec.name)
            for event in result.events:
                if self._event_callback is not None:
                    self._event_callback(event)
            current_input = result.value

        total_elapsed = (self._time_fn() - chain_start) * 1000.0

        return ChainResult(
            final_value=current_input,
            stage_results=stage_results,
            total_elapsed_ms=float(total_elapsed),
            degraded_stages=degraded_names,
            fully_healthy=len(degraded_names) == 0,
        )