# Scientific Governance

## Reproducibility Controls

1. Dataset snapshots are versioned, signed, and referenced in experiment manifests.
2. Feature schema changes require a version increment, parity report, and backfill plan.
3. Model configuration, random seeds, and environment hashes are captured per run.
4. Every experiment stores dataset hash, schema hash, and code revision in the manifest.

## Statistical Policy

1. Every KPI claim includes confidence intervals and sample size.
2. Online changes require guardrail checks, automatic rollback gates, and audit logs.
3. Offline metrics are treated as screening, online tests remain the final gate.

## Data Rights and Provenance

1. Each dataset includes license name, source uri, snapshot time, and content hash.
2. Data ingestion records row level provenance and domain attribution.
3. Any restricted or unknown license is blocked from production use.

## Privacy and Safety

1. User identifiers are pseudonymized in all stored data.
2. Sensitive attributes are excluded by default and require explicit approval.
3. Explanations must be grounded in recorded signals and avoid sensitive inference.
