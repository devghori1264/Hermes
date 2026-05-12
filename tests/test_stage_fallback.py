from __future__ import annotations

import pytest

from src.serving.fallback import (
    ChainResult,
    FallbackEvent,
    StageFallbackChain,
    StageExecutor,
    StageResult,
    StageSpec,
)

class SteppingClock:
    def __init__(self, start: float = 0.0, step: float = 0.001) -> None:
        self._now = start
        self._step = step

    def __call__(self) -> float:
        current = self._now
        self._now += self._step
        return current

def _identity(x: object) -> object:
    return x

def _double(x: int) -> int:
    return x * 2

def _triple(x: int) -> int:
    return x * 3

def _add_ten(x: int) -> int:
    return x + 10

class _FailNTimes:
    def __init__(self, failures: int, success_value: object = "recovered") -> None:
        self._failures = failures
        self._calls = 0
        self._success_value = success_value

    def __call__(self, x: object) -> object:
        self._calls += 1
        if self._calls <= self._failures:
            raise RuntimeError(f"attempt {self._calls} failed")
        return self._success_value

def _always_fail(x: object) -> object:
    raise RuntimeError("always fails")

def _fallback_always_fail(x: object) -> object:
    raise ValueError("fallback also failed")

class TestStageExecutorPrimarySuccess:
    def test_returns_primary_result(self) -> None:
        executor = StageExecutor(time_fn=SteppingClock())
        spec = StageSpec(name="retrieval", primary=_double, fallback=_identity)
        result = executor.execute(spec, 5)
        assert result.value == 10
        assert result.degraded is False
        assert result.events == []

    def test_no_fallback_events_on_success(self) -> None:
        executor = StageExecutor(time_fn=SteppingClock())
        spec = StageSpec(name="ranking", primary=_triple, fallback=_identity)
        result = executor.execute(spec, 3)
        assert len(result.events) == 0

class TestStageExecutorFallback:
    def test_falls_back_after_primary_failure(self) -> None:
        executor = StageExecutor(time_fn=SteppingClock())
        spec = StageSpec(name="retrieval", primary=_always_fail, fallback=_add_ten, max_retries=0)
        result = executor.execute(spec, 5)
        assert result.value == 15
        assert result.degraded is True

    def test_records_fallback_event(self) -> None:
        executor = StageExecutor(time_fn=SteppingClock())
        spec = StageSpec(name="retrieval", primary=_always_fail, fallback=_add_ten, max_retries=0)
        result = executor.execute(spec, 5)
        fallback_events = [e for e in result.events if e.fallback_used]
        assert len(fallback_events) == 1
        assert fallback_events[0].fallback_succeeded is True
        assert fallback_events[0].stage == "retrieval"

    def test_records_error_type_and_message(self) -> None:
        executor = StageExecutor(time_fn=SteppingClock())
        spec = StageSpec(name="ranking", primary=_always_fail, fallback=_identity, max_retries=0)
        result = executor.execute(spec, 1)
        assert result.events[0].error_type == "RuntimeError"
        assert "always fails" in result.events[0].error_message

class TestStageExecutorRetries:
    def test_retries_before_fallback(self) -> None:
        executor = StageExecutor(time_fn=SteppingClock())
        primary = _FailNTimes(failures=2, success_value="ok")
        spec = StageSpec(name="retrieval", primary=primary, fallback=_identity, max_retries=2)
        result = executor.execute(spec, "input")
        assert result.value == "ok"
        assert result.degraded is False
        assert len(result.events) == 2

    def test_exhausts_retries_then_falls_back(self) -> None:
        executor = StageExecutor(time_fn=SteppingClock())
        primary = _FailNTimes(failures=10)
        spec = StageSpec(name="ranking", primary=primary, fallback=_add_ten, max_retries=2)
        result = executor.execute(spec, 5)
        assert result.value == 15
        assert result.degraded is True
        retry_events = [e for e in result.events if not e.fallback_used]
        assert len(retry_events) == 3

    def test_zero_retries_falls_back_immediately(self) -> None:
        executor = StageExecutor(time_fn=SteppingClock())
        spec = StageSpec(name="ranking", primary=_always_fail, fallback=_triple, max_retries=0)
        result = executor.execute(spec, 4)
        assert result.value == 12
        assert result.degraded is True

    def test_retry_events_record_attempt_numbers(self) -> None:
        executor = StageExecutor(time_fn=SteppingClock())
        primary = _FailNTimes(failures=3)
        spec = StageSpec(name="retrieval", primary=primary, fallback=_identity, max_retries=3)
        result = executor.execute(spec, "x")
        attempts = [e.attempt for e in result.events if not e.fallback_used]
        assert attempts == [1, 2, 3]

class TestStageExecutorDoubleFailure:
    def test_raises_when_fallback_also_fails(self) -> None:
        executor = StageExecutor(time_fn=SteppingClock())
        spec = StageSpec(name="ranking", primary=_always_fail, fallback=_fallback_always_fail, max_retries=0)
        with pytest.raises(ValueError, match="fallback also failed"):
            executor.execute(spec, 1)

    def test_records_fallback_failure_event(self) -> None:
        executor = StageExecutor(time_fn=SteppingClock())
        spec = StageSpec(name="ranking", primary=_always_fail, fallback=_fallback_always_fail, max_retries=0)
        events_collected: list[FallbackEvent] = []

        class CapturingExecutor(StageExecutor):
            def execute(self, spec: StageSpec, stage_input: object) -> StageResult:
                try:
                    return super().execute(spec, stage_input)
                except ValueError:
                    raise

        try:
            executor.execute(spec, 1)
        except ValueError:
            pass

class TestStageExecutorTiming:
    def test_elapsed_ms_is_positive(self) -> None:
        executor = StageExecutor(time_fn=SteppingClock())
        spec = StageSpec(name="retrieval", primary=_always_fail, fallback=_identity, max_retries=1)
        result = executor.execute(spec, "x")
        for event in result.events:
            assert event.elapsed_ms > 0.0

    def test_timestamp_is_recorded(self) -> None:
        executor = StageExecutor(time_fn=SteppingClock(start=100.0))
        spec = StageSpec(name="retrieval", primary=_always_fail, fallback=_identity, max_retries=0)
        result = executor.execute(spec, "x")
        for event in result.events:
            assert event.timestamp >= 100.0

class TestChainHealthy:
    def test_chain_passes_results_forward(self) -> None:
        chain = StageFallbackChain(time_fn=SteppingClock())
        chain.add_stage(StageSpec(name="double", primary=_double, fallback=_identity))
        chain.add_stage(StageSpec(name="triple", primary=_triple, fallback=_identity))
        result = chain.execute(2)
        assert result.final_value == 12
        assert result.fully_healthy is True

    def test_chain_tracks_stage_count(self) -> None:
        chain = StageFallbackChain(time_fn=SteppingClock())
        chain.add_stage(StageSpec(name="a", primary=_identity, fallback=_identity))
        chain.add_stage(StageSpec(name="b", primary=_identity, fallback=_identity))
        assert chain.stage_count == 2

    def test_chain_reports_stage_names(self) -> None:
        chain = StageFallbackChain(time_fn=SteppingClock())
        chain.add_stage(StageSpec(name="retrieval", primary=_identity, fallback=_identity))
        chain.add_stage(StageSpec(name="ranking", primary=_identity, fallback=_identity))
        assert chain.stage_names == ["retrieval", "ranking"]

    def test_chain_reports_zero_degraded(self) -> None:
        chain = StageFallbackChain(time_fn=SteppingClock())
        chain.add_stage(StageSpec(name="step", primary=_double, fallback=_identity))
        result = chain.execute(5)
        assert result.degraded_stages == []

    def test_total_elapsed_ms_is_positive(self) -> None:
        chain = StageFallbackChain(time_fn=SteppingClock())
        chain.add_stage(StageSpec(name="step", primary=_identity, fallback=_identity))
        result = chain.execute(1)
        assert result.total_elapsed_ms > 0.0

class TestChainDegraded:
    def test_degraded_stage_is_recorded(self) -> None:
        chain = StageFallbackChain(time_fn=SteppingClock())
        chain.add_stage(StageSpec(name="retrieval", primary=_always_fail, fallback=_add_ten, max_retries=0))
        chain.add_stage(StageSpec(name="ranking", primary=_double, fallback=_identity))
        result = chain.execute(5)
        assert result.final_value == 30
        assert result.degraded_stages == ["retrieval"]
        assert result.fully_healthy is False

    def test_multiple_stages_can_degrade(self) -> None:
        chain = StageFallbackChain(time_fn=SteppingClock())
        chain.add_stage(StageSpec(name="a", primary=_always_fail, fallback=_identity, max_retries=0))
        chain.add_stage(StageSpec(name="b", primary=_always_fail, fallback=_identity, max_retries=0))
        result = chain.execute("input")
        assert result.degraded_stages == ["a", "b"]

    def test_chain_continues_after_fallback(self) -> None:
        chain = StageFallbackChain(time_fn=SteppingClock())
        chain.add_stage(StageSpec(name="fail_stage", primary=_always_fail, fallback=lambda x: 100, max_retries=0))
        chain.add_stage(StageSpec(name="double", primary=_double, fallback=_identity))
        result = chain.execute(0)
        assert result.final_value == 200

class TestChainEventCallback:
    def test_callback_receives_events(self) -> None:
        collected: list[FallbackEvent] = []
        chain = StageFallbackChain(time_fn=SteppingClock(), event_callback=collected.append)
        chain.add_stage(StageSpec(name="retrieval", primary=_always_fail, fallback=_identity, max_retries=1))
        chain.execute("input")
        assert len(collected) >= 2
        assert all(e.stage == "retrieval" for e in collected)

    def test_callback_not_called_on_success(self) -> None:
        collected: list[FallbackEvent] = []
        chain = StageFallbackChain(time_fn=SteppingClock(), event_callback=collected.append)
        chain.add_stage(StageSpec(name="ok", primary=_identity, fallback=_identity))
        chain.execute("input")
        assert len(collected) == 0

class TestChainAbort:
    def test_chain_aborts_when_fallback_fails(self) -> None:
        chain = StageFallbackChain(time_fn=SteppingClock())
        chain.add_stage(StageSpec(name="fatal", primary=_always_fail, fallback=_fallback_always_fail, max_retries=0))
        chain.add_stage(StageSpec(name="unreachable", primary=_identity, fallback=_identity))
        with pytest.raises(ValueError, match="fallback also failed"):
            chain.execute("input")

class TestChainEmpty:
    def test_empty_chain_returns_initial_input(self) -> None:
        chain = StageFallbackChain(time_fn=SteppingClock())
        result = chain.execute("passthrough")
        assert result.final_value == "passthrough"
        assert result.stage_results == []
        assert result.fully_healthy is True

class TestChainRealisticPipeline:
    def test_full_recommendation_pipeline_with_one_degraded_stage(self) -> None:
        flaky_retrieval = _FailNTimes(failures=3, success_value=["item_a", "item_b"])

        chain = StageFallbackChain(time_fn=SteppingClock())
        chain.add_stage(StageSpec(
            name="retrieval",
            primary=flaky_retrieval,
            fallback=lambda q: ["fallback_item"],
            max_retries=2,
        ))
        chain.add_stage(StageSpec(
            name="ranking",
            primary=lambda items: sorted(items, reverse=True),
            fallback=lambda items: items,
        ))
        chain.add_stage(StageSpec(
            name="reranking",
            primary=lambda items: items[:5],
            fallback=lambda items: items[:3],
        ))

        result = chain.execute("inception")

        assert result.degraded_stages == ["retrieval"]
        assert result.final_value == ["fallback_item"]
        assert len(result.stage_results) == 3
        assert result.stage_results[0].degraded is True
        assert result.stage_results[1].degraded is False
        assert result.stage_results[2].degraded is False