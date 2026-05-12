from __future__ import annotations

import pytest

from src.observability.tracing import (
    CompletedTrace,
    InMemoryTraceExporter,
    Span,
    SpanEvent,
    SpanStatus,
    TraceContext,
    TracingEngine,
    WaterfallEntry,
    _generate_span_id,
    _generate_trace_id,
)

class SteppingClock:
    def __init__(self, start: float = 0.0, step: float = 0.001) -> None:
        self._now = start
        self._step = step

    def __call__(self) -> float:
        current = self._now
        self._now += self._step
        return current

class TestIdentifiers:
    def test_trace_id_is_32_hex_chars(self) -> None:
        trace_id = _generate_trace_id()
        assert len(trace_id) == 32
        int(trace_id, 16)

    def test_span_id_is_16_hex_chars(self) -> None:
        span_id = _generate_span_id()
        assert len(span_id) == 16
        int(span_id, 16)

    def test_trace_ids_are_unique(self) -> None:
        ids = {_generate_trace_id() for _ in range(100)}
        assert len(ids) == 100

    def test_span_ids_are_unique(self) -> None:
        ids = {_generate_span_id() for _ in range(100)}
        assert len(ids) == 100

class TestSpan:
    def test_set_attribute(self) -> None:
        span = Span(span_id="s1", trace_id="t1", parent_span_id=None, name="op", start_time=0.0)
        span.set_attribute("key", "value")
        assert span.attributes["key"] == "value"

    def test_set_attributes_bulk(self) -> None:
        span = Span(span_id="s1", trace_id="t1", parent_span_id=None, name="op", start_time=0.0)
        span.set_attributes({"a": 1, "b": 2})
        assert span.attributes["a"] == 1
        assert span.attributes["b"] == 2

    def test_set_attribute_noop_after_end(self) -> None:
        span = Span(span_id="s1", trace_id="t1", parent_span_id=None, name="op", start_time=0.0, end_time=1.0)
        span.set_attribute("key", "value")
        assert "key" not in span.attributes

    def test_add_event(self) -> None:
        span = Span(span_id="s1", trace_id="t1", parent_span_id=None, name="op", start_time=0.0)
        span.add_event("cache_miss", {"scope": "retrieval"}, timestamp=0.5)
        assert len(span.events) == 1
        assert span.events[0].name == "cache_miss"
        assert span.events[0].timestamp == 0.5
        assert span.events[0].attributes["scope"] == "retrieval"

    def test_add_event_noop_after_end(self) -> None:
        span = Span(span_id="s1", trace_id="t1", parent_span_id=None, name="op", start_time=0.0, end_time=1.0)
        span.add_event("should_not_appear")
        assert len(span.events) == 0

    def test_set_error(self) -> None:
        span = Span(span_id="s1", trace_id="t1", parent_span_id=None, name="op", start_time=0.0)
        span.set_error(ValueError("test failure"))
        assert span.is_error
        assert span.attributes["error.message"] == "test failure"
        assert span.attributes["error.type"] == "ValueError"

    def test_set_error_with_string(self) -> None:
        span = Span(span_id="s1", trace_id="t1", parent_span_id=None, name="op", start_time=0.0)
        span.set_error("something broke")
        assert span.is_error
        assert span.attributes["error.message"] == "something broke"
        assert span.attributes["error.type"] == "str"

    def test_duration_ms(self) -> None:
        span = Span(span_id="s1", trace_id="t1", parent_span_id=None, name="op", start_time=1.0, end_time=1.050)
        assert abs(span.duration_ms - 50.0) < 0.01

    def test_duration_ms_zero_when_not_ended(self) -> None:
        span = Span(span_id="s1", trace_id="t1", parent_span_id=None, name="op", start_time=1.0)
        assert span.duration_ms == 0.0

    def test_is_ended(self) -> None:
        span = Span(span_id="s1", trace_id="t1", parent_span_id=None, name="op", start_time=0.0)
        assert span.is_ended is False
        span.end_time = 1.0
        assert span.is_ended is True

class TestInMemoryTraceExporter:
    def test_export_and_retrieve(self) -> None:
        exporter = InMemoryTraceExporter(max_traces=10)
        trace = CompletedTrace(
            trace_id="t1", root_span_id="s1", spans=(), total_duration_ms=5.0, span_count=1, error_count=0,
        )
        exporter.export(trace)
        assert exporter.count == 1
        assert exporter.find("t1") is not None

    def test_evicts_oldest_when_full(self) -> None:
        exporter = InMemoryTraceExporter(max_traces=2)
        for i in range(3):
            trace = CompletedTrace(
                trace_id=f"t{i}", root_span_id=f"s{i}", spans=(), total_duration_ms=0.0, span_count=0, error_count=0,
            )
            exporter.export(trace)
        assert exporter.count == 2
        assert exporter.find("t0") is None
        assert exporter.find("t1") is not None
        assert exporter.find("t2") is not None

    def test_find_returns_none_for_unknown(self) -> None:
        exporter = InMemoryTraceExporter()
        assert exporter.find("nonexistent") is None

    def test_clear(self) -> None:
        exporter = InMemoryTraceExporter()
        trace = CompletedTrace(
            trace_id="t1", root_span_id="s1", spans=(), total_duration_ms=0.0, span_count=0, error_count=0,
        )
        exporter.export(trace)
        exporter.clear()
        assert exporter.count == 0

class TestTracingEngineLifecycle:
    def test_start_and_end_trace(self) -> None:
        clock = SteppingClock()
        engine = TracingEngine(time_fn=clock)
        ctx = engine.start_trace("recommend")
        assert engine.active_trace_count() == 1
        trace = engine.end_trace(ctx)
        assert engine.active_trace_count() == 0
        assert trace.trace_id == ctx.trace_id
        assert trace.span_count == 1

    def test_root_span_gets_ok_status(self) -> None:
        clock = SteppingClock()
        engine = TracingEngine(time_fn=clock)
        ctx = engine.start_trace("recommend")
        trace = engine.end_trace(ctx)
        assert trace.spans[0].status == "ok"

    def test_start_trace_with_attributes(self) -> None:
        clock = SteppingClock()
        engine = TracingEngine(time_fn=clock)
        ctx = engine.start_trace("recommend", attributes={"user_id": "u42"})
        trace = engine.end_trace(ctx)
        assert trace.spans[0].attributes["user_id"] == "u42"

    def test_end_trace_for_unknown_context(self) -> None:
        engine = TracingEngine(time_fn=SteppingClock())
        ctx = TraceContext(trace_id="unknown", root_span_id="s1")
        trace = engine.end_trace(ctx)
        assert trace.span_count == 0

    def test_total_duration_ms(self) -> None:
        clock = SteppingClock(step=0.010)
        engine = TracingEngine(time_fn=clock)
        ctx = engine.start_trace("recommend")
        trace = engine.end_trace(ctx)
        assert trace.total_duration_ms > 0.0

class TestTracingEngineChildSpans:
    def test_child_span_has_correct_parent(self) -> None:
        engine = TracingEngine(time_fn=SteppingClock())
        ctx = engine.start_trace("recommend")
        child = engine.start_child_span(ctx, "retrieval")
        assert child.parent_span_id == ctx.root_span_id
        assert child.trace_id == ctx.trace_id

    def test_child_span_depth(self) -> None:
        engine = TracingEngine(time_fn=SteppingClock())
        ctx = engine.start_trace("recommend")
        child = engine.start_child_span(ctx, "retrieval")
        assert child.depth == 1

    def test_grandchild_span_depth(self) -> None:
        engine = TracingEngine(time_fn=SteppingClock())
        ctx = engine.start_trace("recommend")
        child = engine.start_child_span(ctx, "retrieval")
        grandchild = engine.start_child_span(ctx, "cold_start", parent_span_id=child.span_id)
        assert grandchild.depth == 2
        assert grandchild.parent_span_id == child.span_id

    def test_child_span_with_attributes(self) -> None:
        engine = TracingEngine(time_fn=SteppingClock())
        ctx = engine.start_trace("recommend")
        child = engine.start_child_span(ctx, "ranking", attributes={"objective": "pairwise_hinge"})
        assert child.attributes["objective"] == "pairwise_hinge"

    def test_end_span_sets_ok_status(self) -> None:
        engine = TracingEngine(time_fn=SteppingClock())
        ctx = engine.start_trace("recommend")
        child = engine.start_child_span(ctx, "retrieval")
        engine.end_span(child)
        assert child.status == "ok"
        assert child.is_ended

    def test_end_span_with_explicit_status(self) -> None:
        engine = TracingEngine(time_fn=SteppingClock())
        ctx = engine.start_trace("recommend")
        child = engine.start_child_span(ctx, "retrieval")
        engine.end_span(child, status="error")
        assert child.status == "error"

    def test_end_span_is_idempotent(self) -> None:
        engine = TracingEngine(time_fn=SteppingClock())
        ctx = engine.start_trace("recommend")
        child = engine.start_child_span(ctx, "retrieval")
        engine.end_span(child)
        end_time = child.end_time
        engine.end_span(child)
        assert child.end_time == end_time

    def test_unended_child_spans_marked_as_error(self) -> None:
        engine = TracingEngine(time_fn=SteppingClock())
        ctx = engine.start_trace("recommend")
        engine.start_child_span(ctx, "retrieval")
        trace = engine.end_trace(ctx)
        retrieval_span = next(s for s in trace.spans if s.name == "retrieval")
        assert retrieval_span.is_error
        assert "not explicitly ended" in retrieval_span.attributes.get("error.message", "")

    def test_multiple_child_spans_appear_in_trace(self) -> None:
        engine = TracingEngine(time_fn=SteppingClock())
        ctx = engine.start_trace("recommend")
        ret = engine.start_child_span(ctx, "retrieval")
        engine.end_span(ret)
        rank = engine.start_child_span(ctx, "ranking")
        engine.end_span(rank)
        rerank = engine.start_child_span(ctx, "reranking")
        engine.end_span(rerank)
        trace = engine.end_trace(ctx)
        assert trace.span_count == 4
        names = [s.name for s in trace.spans]
        assert "retrieval" in names
        assert "ranking" in names
        assert "reranking" in names

class TestTracingEngineErrors:
    def test_error_count_in_completed_trace(self) -> None:
        engine = TracingEngine(time_fn=SteppingClock())
        ctx = engine.start_trace("recommend")
        child = engine.start_child_span(ctx, "retrieval")
        child.set_error(RuntimeError("timeout"))
        engine.end_span(child, status="error")
        trace = engine.end_trace(ctx)
        assert trace.error_count == 1

    def test_error_spans_extraction(self) -> None:
        engine = TracingEngine(time_fn=SteppingClock())
        ctx = engine.start_trace("recommend")
        good = engine.start_child_span(ctx, "retrieval")
        engine.end_span(good)
        bad = engine.start_child_span(ctx, "ranking")
        bad.set_error(ValueError("nan weights"))
        engine.end_span(bad, status="error")
        trace = engine.end_trace(ctx)
        errors = engine.error_spans(trace.trace_id)
        assert len(errors) == 1
        assert errors[0].name == "ranking"

    def test_error_spans_for_unknown_trace(self) -> None:
        engine = TracingEngine(time_fn=SteppingClock())
        assert engine.error_spans("nonexistent") == []

class TestWaterfall:
    def test_waterfall_entries_ordered_by_start_time(self) -> None:
        clock = SteppingClock()
        engine = TracingEngine(time_fn=clock)
        ctx = engine.start_trace("recommend")
        ret = engine.start_child_span(ctx, "retrieval")
        engine.end_span(ret)
        rank = engine.start_child_span(ctx, "ranking")
        engine.end_span(rank)
        trace = engine.end_trace(ctx)
        waterfall = engine.waterfall(trace.trace_id)
        assert len(waterfall) == 3
        offsets = [entry.offset_ms for entry in waterfall]
        assert offsets == sorted(offsets)

    def test_waterfall_indentation_reflects_depth(self) -> None:
        clock = SteppingClock()
        engine = TracingEngine(time_fn=clock)
        ctx = engine.start_trace("recommend")
        child = engine.start_child_span(ctx, "retrieval")
        grandchild = engine.start_child_span(ctx, "cold_start", parent_span_id=child.span_id)
        engine.end_span(grandchild)
        engine.end_span(child)
        trace = engine.end_trace(ctx)
        waterfall = engine.waterfall(trace.trace_id)
        root_entry = next(e for e in waterfall if e.name == "recommend")
        child_entry = next(e for e in waterfall if e.name == "retrieval")
        grandchild_entry = next(e for e in waterfall if e.name == "cold_start")
        assert root_entry.indent == 0
        assert child_entry.indent == 1
        assert grandchild_entry.indent == 2

    def test_waterfall_bar_width_reflects_duration(self) -> None:
        clock = SteppingClock(step=0.010)
        engine = TracingEngine(time_fn=clock)
        ctx = engine.start_trace("recommend")
        child = engine.start_child_span(ctx, "retrieval")
        engine.end_span(child)
        trace = engine.end_trace(ctx)
        waterfall = engine.waterfall(trace.trace_id)
        retrieval_entry = next(e for e in waterfall if e.name == "retrieval")
        assert retrieval_entry.bar_width_ms > 0.0

    def test_waterfall_for_unknown_trace(self) -> None:
        engine = TracingEngine(time_fn=SteppingClock())
        assert engine.waterfall("nonexistent") == []

    def test_waterfall_includes_event_count(self) -> None:
        engine = TracingEngine(time_fn=SteppingClock())
        ctx = engine.start_trace("recommend")
        child = engine.start_child_span(ctx, "retrieval")
        child.add_event("cache_miss", {"scope": "vectors"})
        child.add_event("fallback_activated")
        engine.end_span(child)
        trace = engine.end_trace(ctx)
        waterfall = engine.waterfall(trace.trace_id)
        retrieval_entry = next(e for e in waterfall if e.name == "retrieval")
        assert retrieval_entry.event_count == 2

class TestStageLatencyBreakdown:
    def test_breakdown_per_stage(self) -> None:
        clock = SteppingClock(step=0.005)
        engine = TracingEngine(time_fn=clock)
        ctx = engine.start_trace("recommend")
        ret = engine.start_child_span(ctx, "retrieval")
        engine.end_span(ret)
        rank = engine.start_child_span(ctx, "ranking")
        engine.end_span(rank)
        trace = engine.end_trace(ctx)
        breakdown = engine.stage_latency_breakdown(trace.trace_id)
        assert "retrieval" in breakdown
        assert "ranking" in breakdown
        assert breakdown["retrieval"] > 0.0

    def test_duplicate_stage_names_summed(self) -> None:
        clock = SteppingClock(step=0.005)
        engine = TracingEngine(time_fn=clock)
        ctx = engine.start_trace("recommend")
        r1 = engine.start_child_span(ctx, "retrieval")
        engine.end_span(r1)
        r2 = engine.start_child_span(ctx, "retrieval")
        engine.end_span(r2)
        trace = engine.end_trace(ctx)
        breakdown = engine.stage_latency_breakdown(trace.trace_id)
        assert breakdown["retrieval"] > r1.duration_ms

    def test_breakdown_for_unknown_trace(self) -> None:
        engine = TracingEngine(time_fn=SteppingClock())
        assert engine.stage_latency_breakdown("nonexistent") == {}

class TestConcurrentTraces:
    def test_multiple_traces_in_flight(self) -> None:
        engine = TracingEngine(time_fn=SteppingClock())
        ctx_a = engine.start_trace("recommend_a")
        ctx_b = engine.start_trace("recommend_b")
        assert engine.active_trace_count() == 2
        engine.start_child_span(ctx_a, "retrieval_a")
        engine.start_child_span(ctx_b, "retrieval_b")
        trace_a = engine.end_trace(ctx_a)
        assert engine.active_trace_count() == 1
        trace_b = engine.end_trace(ctx_b)
        assert engine.active_trace_count() == 0
        a_names = {s.name for s in trace_a.spans}
        b_names = {s.name for s in trace_b.spans}
        assert "retrieval_a" in a_names
        assert "retrieval_a" not in b_names
        assert "retrieval_b" in b_names
        assert "retrieval_b" not in a_names

class TestFullPipelineTrace:
    def test_realistic_recommendation_trace(self) -> None:
        clock = SteppingClock(step=0.002)
        exporter = InMemoryTraceExporter()
        engine = TracingEngine(exporter=exporter, time_fn=clock)

        ctx = engine.start_trace("recommend", attributes={"user_id": "u42", "query": "inception"})

        retrieval = engine.start_child_span(ctx, "retrieval", attributes={"strategy": "hybrid"})
        retrieval.add_event("cache_miss", {"scope": "retrieval_vector"})
        cold = engine.start_child_span(ctx, "cold_start_fallback", parent_span_id=retrieval.span_id)
        cold.set_attribute("mode", "new_user")
        cold.set_attribute("candidates_returned", 15)
        engine.end_span(cold)
        engine.end_span(retrieval)

        ranking = engine.start_child_span(ctx, "ranking", attributes={"objective": "pairwise_hinge"})
        ranking.set_attribute("candidates_scored", 50)
        engine.end_span(ranking)

        reranking = engine.start_child_span(ctx, "reranking")
        fairness = engine.start_child_span(ctx, "fairness_check", parent_span_id=reranking.span_id)
        fairness.set_attribute("exposure_gap", 0.03)
        fairness.set_attribute("passes_threshold", True)
        engine.end_span(fairness)
        diversity = engine.start_child_span(ctx, "diversity_rerank", parent_span_id=reranking.span_id)
        engine.end_span(diversity)
        engine.end_span(reranking)

        serving = engine.start_child_span(ctx, "cache_write")
        engine.end_span(serving)

        trace = engine.end_trace(ctx)

        assert trace.span_count == 8
        assert trace.error_count == 0
        assert trace.total_duration_ms > 0.0

        waterfall = engine.waterfall(trace.trace_id)
        assert len(waterfall) == 8

        breakdown = engine.stage_latency_breakdown(trace.trace_id)
        assert "retrieval" in breakdown
        assert "ranking" in breakdown
        assert "reranking" in breakdown

        cold_entry = next(e for e in waterfall if e.name == "cold_start_fallback")
        assert cold_entry.indent == 2
        fairness_entry = next(e for e in waterfall if e.name == "fairness_check")
        assert fairness_entry.indent == 2
