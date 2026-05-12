from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Callable, Protocol
import uuid

def _generate_trace_id() -> str:
    return uuid.uuid4().hex


def _generate_span_id() -> str:
    return uuid.uuid4().hex[:16]

class SpanStatus(str, Enum):
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass(frozen=True)
class SpanEvent:
    name: str
    timestamp: float
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TraceContext:
    trace_id: str
    root_span_id: str


@dataclass
class Span:
    span_id: str
    trace_id: str
    parent_span_id: str | None
    name: str
    start_time: float
    end_time: float | None = None
    status: str = SpanStatus.UNSET.value
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    depth: int = 0

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000.0

    @property
    def is_ended(self) -> bool:
        return self.end_time is not None

    @property
    def is_error(self) -> bool:
        return self.status == SpanStatus.ERROR.value

    def set_attribute(self, key: str, value: Any) -> None:
        if self.is_ended:
            return
        self.attributes[key] = value

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        if self.is_ended:
            return
        self.attributes.update(attributes)

    def add_event(self, name: str, attributes: dict[str, Any] | None = None, *, timestamp: float | None = None) -> None:
        if self.is_ended:
            return
        event = SpanEvent(
            name=name,
            timestamp=timestamp if timestamp is not None else time.perf_counter(),
            attributes=dict(attributes) if attributes else {},
        )
        self.events.append(event)

    def set_error(self, error: Exception | str) -> None:
        if self.is_ended:
            return
        self.status = SpanStatus.ERROR.value
        error_message = str(error)
        error_type = type(error).__name__ if isinstance(error, Exception) else "str"
        self.set_attribute("error.message", error_message)
        self.set_attribute("error.type", error_type)


@dataclass(frozen=True)
class CompletedTrace:
    trace_id: str
    root_span_id: str
    spans: tuple[Span, ...]
    total_duration_ms: float
    span_count: int
    error_count: int


@dataclass(frozen=True)
class WaterfallEntry:
    span_id: str
    name: str
    indent: int
    offset_ms: float
    bar_width_ms: float
    status: str
    attributes: dict[str, Any]
    event_count: int

class TraceExporter(Protocol):
    def export(self, trace: CompletedTrace) -> None: ...


class InMemoryTraceExporter:

    def __init__(self, max_traces: int = 256) -> None:
        self._max_traces = max(1, max_traces)
        self._traces: list[CompletedTrace] = []

    def export(self, trace: CompletedTrace) -> None:
        self._traces.append(trace)
        while len(self._traces) > self._max_traces:
            self._traces.pop(0)

    def traces(self) -> list[CompletedTrace]:
        return list(self._traces)

    def find(self, trace_id: str) -> CompletedTrace | None:
        for trace in self._traces:
            if trace.trace_id == trace_id:
                return trace
        return None

    def clear(self) -> None:
        self._traces.clear()

    @property
    def count(self) -> int:
        return len(self._traces)

class TracingEngine:
    def __init__(
        self,
        *,
        exporter: TraceExporter | None = None,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self._exporter = exporter or InMemoryTraceExporter()
        self._time_fn = time_fn or time.perf_counter
        self._active_spans: dict[str, dict[str, Span]] = {}
        self._trace_roots: dict[str, str] = {}

    @property
    def exporter(self) -> TraceExporter:
        return self._exporter

    def start_trace(self, operation: str, *, attributes: dict[str, Any] | None = None) -> TraceContext:
        trace_id = _generate_trace_id()
        root_span_id = _generate_span_id()
        root_span = Span(
            span_id=root_span_id,
            trace_id=trace_id,
            parent_span_id=None,
            name=operation,
            start_time=self._time_fn(),
            depth=0,
        )
        if attributes:
            root_span.set_attributes(attributes)
        self._active_spans[trace_id] = {root_span_id: root_span}
        self._trace_roots[trace_id] = root_span_id
        return TraceContext(trace_id=trace_id, root_span_id=root_span_id)

    def end_trace(self, context: TraceContext) -> CompletedTrace:
        trace_spans = self._active_spans.get(context.trace_id)
        if trace_spans is None:
            return CompletedTrace(
                trace_id=context.trace_id,
                root_span_id=context.root_span_id,
                spans=(),
                total_duration_ms=0.0,
                span_count=0,
                error_count=0,
            )

        now = self._time_fn()
        for span in trace_spans.values():
            if not span.is_ended:
                if span.span_id == context.root_span_id:
                    if span.status == SpanStatus.UNSET.value:
                        span.status = SpanStatus.OK.value
                else:
                    if span.status == SpanStatus.UNSET.value:
                        span.status = SpanStatus.ERROR.value
                        span.attributes["error.message"] = "span was not explicitly ended"
                span.end_time = now

        sorted_spans = sorted(trace_spans.values(), key=lambda s: s.start_time)
        root_span = trace_spans.get(context.root_span_id)
        total_duration = root_span.duration_ms if root_span else 0.0
        error_count = sum(1 for s in sorted_spans if s.is_error)

        completed = CompletedTrace(
            trace_id=context.trace_id,
            root_span_id=context.root_span_id,
            spans=tuple(sorted_spans),
            total_duration_ms=float(total_duration),
            span_count=len(sorted_spans),
            error_count=error_count,
        )

        self._exporter.export(completed)
        del self._active_spans[context.trace_id]
        self._trace_roots.pop(context.trace_id, None)
        return completed

    def start_child_span(
        self,
        context: TraceContext,
        name: str,
        *,
        parent_span_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        trace_spans = self._active_spans.get(context.trace_id, {})
        effective_parent_id = parent_span_id or context.root_span_id
        parent = trace_spans.get(effective_parent_id)
        parent_depth = parent.depth if parent else 0

        span_id = _generate_span_id()
        span = Span(
            span_id=span_id,
            trace_id=context.trace_id,
            parent_span_id=effective_parent_id,
            name=name,
            start_time=self._time_fn(),
            depth=parent_depth + 1,
        )
        if attributes:
            span.set_attributes(attributes)

        if context.trace_id not in self._active_spans:
            self._active_spans[context.trace_id] = {}
        self._active_spans[context.trace_id][span_id] = span
        return span

    def end_span(self, span: Span, *, status: str | None = None) -> None:
        if span.is_ended:
            return
        span.end_time = self._time_fn()
        if status is not None:
            span.status = status
        elif span.status == SpanStatus.UNSET.value:
            span.status = SpanStatus.OK.value

    def waterfall(self, trace_id: str) -> list[WaterfallEntry]:
        if not isinstance(self._exporter, InMemoryTraceExporter):
            return []
        trace = self._exporter.find(trace_id)
        if trace is None:
            return []
        if not trace.spans:
            return []

        root_start = trace.spans[0].start_time
        entries: list[WaterfallEntry] = []
        for span in trace.spans:
            offset = (span.start_time - root_start) * 1000.0
            entries.append(WaterfallEntry(
                span_id=span.span_id,
                name=span.name,
                indent=span.depth,
                offset_ms=float(offset),
                bar_width_ms=float(span.duration_ms),
                status=span.status,
                attributes=dict(span.attributes),
                event_count=len(span.events),
            ))
        return entries

    def active_trace_count(self) -> int:
        return len(self._active_spans)

    def stage_latency_breakdown(self, trace_id: str) -> dict[str, float]:
        if not isinstance(self._exporter, InMemoryTraceExporter):
            return {}
        trace = self._exporter.find(trace_id)
        if trace is None:
            return {}
        breakdown: dict[str, float] = {}
        for span in trace.spans:
            breakdown[span.name] = breakdown.get(span.name, 0.0) + span.duration_ms
        return breakdown

    def error_spans(self, trace_id: str) -> list[Span]:
        if not isinstance(self._exporter, InMemoryTraceExporter):
            return []
        trace = self._exporter.find(trace_id)
        if trace is None:
            return []
        return [span for span in trace.spans if span.is_error]