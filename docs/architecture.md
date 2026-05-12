# Architecture Specification

## Service Boundaries

1. api layer handles HTTP contracts and backward compatibility routes.
2. services layer orchestrates recommendation, TMDB access, and sentiment analysis.
3. data layer manages catalog loading and caching.
4. features layer builds text and multimodal vectors with fallback contracts.
5. ranking and policy layers implement staged scoring and reranking.
6. observability layer tracks request traces and stage latencies.

## Migration Sequence

1. keep `/home`, `/similarity`, `/recommend` behavior stable.
2. move recommendation logic out of `main.py` into `src/services`.
3. route TMDB calls through backend proxy endpoints.
4. harden contracts with deterministic parsing and tests.
