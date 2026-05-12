from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable

from src.retrieval.types import RetrievalCandidate


@dataclass(frozen=True)
class BlendInput:
    name: str
    candidates: list[RetrievalCandidate]
    weight: float = 1.0
    normalization: str = "minmax"


@dataclass(frozen=True)
class BlendResult:
    candidates: list[RetrievalCandidate]
    sources: list[str]


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _zscore(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if math.isclose(std, 0.0):
        return [0.0 for _ in values]
    return [(value - mean) / std for value in values]


def _sigmoid(values: list[float]) -> list[float]:
    return [1 / (1 + math.exp(-value)) for value in values]


def normalize(values: list[float], method: str) -> list[float]:
    if method == "minmax":
        return _minmax(values)
    if method == "zscore":
        return _zscore(values)
    if method == "sigmoid":
        return _sigmoid(values)
    if method == "none":
        return list(values)
    raise ValueError("unknown normalization")


class CandidateBlender:
    def blend(self, inputs: Iterable[BlendInput], top_k: int) -> BlendResult:
        merged: dict[str, RetrievalCandidate] = {}
        sources: list[str] = []

        for item in inputs:
            if not item.candidates:
                continue
            sources.append(item.name)
            scores = [candidate.score for candidate in item.candidates]
            normalized = normalize(scores, item.normalization)
            for candidate, score in zip(item.candidates, normalized):
                weighted = float(score) * item.weight
                existing = merged.get(candidate.item_id)
                if existing is None:
                    merged[candidate.item_id] = RetrievalCandidate(
                        item_id=candidate.item_id,
                        score=weighted,
                        source=item.name,
                        metadata={
                            "sources": {item.name: weighted},
                            "raw_scores": {item.name: candidate.score},
                        },
                    )
                    continue
                combined_sources = dict(existing.metadata.get("sources", {}))
                combined_sources[item.name] = combined_sources.get(item.name, 0.0) + weighted
                combined_raw = dict(existing.metadata.get("raw_scores", {}))
                combined_raw[item.name] = candidate.score
                merged[candidate.item_id] = RetrievalCandidate(
                    item_id=candidate.item_id,
                    score=existing.score + weighted,
                    source="blend",
                    metadata={
                        "sources": combined_sources,
                        "raw_scores": combined_raw,
                    },
                )

        blended = list(merged.values())
        blended.sort(key=lambda candidate: candidate.score, reverse=True)
        return BlendResult(candidates=blended[:top_k], sources=sources)
