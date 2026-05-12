from __future__ import annotations

import uuid

from flask import Flask, Response, g, request

from src.observability.metrics import MetricsRegistry, RequestTrace
from src.observability.slo import PROFILE_SLOS, SLO


def _request_id() -> str:
    return uuid.uuid4().hex


def _slo_for_profile(profile: str) -> SLO:
    return PROFILE_SLOS.get(profile, PROFILE_SLOS["lean"])


def setup_telemetry(app: Flask, metrics: MetricsRegistry, profile: str) -> None:
    slo = _slo_for_profile(profile)

    @app.before_request
    def _start_trace() -> None:
        g.request_id = _request_id()
        g.request_trace = RequestTrace(
            operation=request.path,
            request_id=g.request_id,
        )

    @app.after_request
    def _end_trace(response: Response) -> Response:
        trace = getattr(g, "request_trace", None)
        if trace is not None:
            elapsed = trace.elapsed_ms()
            metrics.record(
                "http.latency_ms",
                elapsed,
                tags={"path": request.path, "method": request.method},
            )
            metrics.increment(
                "http.requests",
                tags={"path": request.path, "method": request.method, "status": str(response.status_code)},
            )
            if response.status_code >= 500:
                metrics.increment("http.errors", tags={"path": request.path})
            if elapsed > slo.latency_p95_ms:
                metrics.increment("http.slo_violations", tags={"path": request.path})
        response.headers["X-Request-ID"] = getattr(g, "request_id", "")
        return response

    @app.teardown_request
    def _teardown_trace(error: Exception | None) -> None:
        if error is None:
            return
        metrics.increment("http.errors", tags={"path": request.path})
