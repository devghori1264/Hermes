# Platform Operations

## Deployment Profiles

1. Lean profile uses constrained fanout, compressed models, and strict latency budgets.
2. Mid profile uses full pipeline with scheduled retraining and standard cache layers.
3. Aggressive profile uses wide fanout, frequent refresh, and expanded experiments.

## Reliability Controls

1. Rate limits protect the public surface and reduce abuse.
2. Circuit breakers isolate external dependency failures.
3. Cache layers provide deterministic fallback for critical endpoints.
4. Telemetry records latency, error rate, and service health.
5. Security guard enforces input validation, abuse signature detection, and personal data protection.

## Recovery Playbooks

1. Model rollback uses signed artifact versions with audit trail.
2. Feature fallback switches to deterministic embeddings when encoders fail.
3. Dependency outage fallback returns cached candidates and error banners.

## Standard Validation Commands

1. Fast developer tests.

```bash
.venv/bin/python -m pytest -m "not quality"
```

2. Quality recommendation check with full catalog.

```bash
.venv/bin/python -m pytest tests/test_recommendation_quality.py -m quality
```

3. Security and explanation focused verification.

```bash
.venv/bin/python -m pytest tests/test_security_controls.py tests/test_conversational_explanation_service.py
```

## Release Acceptance Gate

1. Data evidence gate.
2. Model artifact gate.
3. Retrieval and ranking gate.
4. Security and privacy gate.
5. Explanation endpoint gate.
6. Operational readiness gate.

Each gate requires command evidence and artifact references inside release records.

## Security Operations Notes

1. Query surfaces reject personal data payloads.
2. Injection signatures are blocked before recommendation execution.
3. Security denials emit `security.blocked` metric with reason tag.
4. Recommendation explanation endpoint follows the same security validation path.

## Current Release Focus

1. Keep default test cycle under ten seconds for daily engineering flow.
2. Run quality marker tests before every release candidate.
3. Keep plan and operations records aligned with implemented behavior.
