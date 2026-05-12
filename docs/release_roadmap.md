# Release Roadmap

## 1. Release Objective

Deliver a verifiable recommendation platform release with complete artifact lineage, guarded serving interfaces, explanation capability, and repeatable quality evidence.

## 2. Scope

1. Data evidence and checksums.
2. Ingestion and schema consistency.
3. Feature extraction and embedding training.
4. Retrieval ranking diversity stack.
5. Conversational explanation endpoint.
6. Security privacy abuse controls.
7. Fast and quality test split.
8. Operational runbook and deployment profiles.

## 3. Execution Phases

### 3.1 Phase A foundation closure

1. Confirm all completed engineering milestones.
2. Confirm todo state alignment with code reality.
3. Confirm plan and operations records consistency.

### 3.2 Phase B release validation

1. Execute fast test matrix.
2. Execute quality test matrix.
3. Verify explanation endpoint contract.
4. Verify security denial behavior for malicious and personal data payloads.
5. Verify telemetry counters for rate limit, dependency circuit open, and security blocked paths.

### 3.3 Phase C release candidate packaging

1. Freeze model artifact references.
2. Freeze profile configuration references.
3. Publish acceptance evidence with command outputs and timestamps.
4. Prepare rollback procedure for model and index artifacts.

## 4. Acceptance Checklist

1. All mandatory tests pass.
2. No missing critical module in serving path.
3. Explanation endpoint returns deterministic rationale fields.
4. Security guard blocks malicious payload classes.
5. Privacy protection blocks obvious personal data patterns.
6. Operations and plan records reflect final implementation status.

## 5. Command Matrix

1. Fast path command.

```bash
.venv/bin/python -m pytest -m "not quality"
```

2. Quality path command.

```bash
.venv/bin/python -m pytest -m quality
```

3. Security and explanation path command.

```bash
.venv/bin/python -m pytest tests/test_security_controls.py tests/test_conversational_explanation_service.py
```

## 6. Sign Off Roles

1. Engineering reviewer validates architecture and code quality.
2. Data reviewer validates source evidence and checksums.
3. Safety reviewer validates security privacy abuse controls.
4. Product reviewer validates recommendation and explanation behavior.

## 7. Current Status

1. Engineering milestones complete through security and explanation layers.
2. Release documentation is now active and versioned in repository.
3. Final sign off work remains active.

## 8. Update Timestamp

Updated on 2026 05 09.
