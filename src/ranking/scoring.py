from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.domain.models import Candidate


@dataclass(frozen=True)
class ScoreWeights:
    base: float = 1.0
    text: float = 0.4
    multimodal: float = 0.4
    popularity: float = 0.2
    recency: float = 0.1
    novelty: float = 0.1


@dataclass(frozen=True)
class ScoreBreakdown:
    total: float
    base: float
    text: float
    multimodal: float
    popularity: float
    recency: float
    novelty: float

    def to_metadata(self) -> dict[str, float]:
        return {
            "total": self.total,
            "base": self.base,
            "text": self.text,
            "multimodal": self.multimodal,
            "popularity": self.popularity,
            "recency": self.recency,
            "novelty": self.novelty,
        }


def _signal(candidate: Candidate, name: str) -> float:
    metadata = candidate.metadata or {}
    signals = metadata.get("signals", {}) if isinstance(metadata, dict) else {}
    value = signals.get(name, 0.0) if isinstance(signals, dict) else 0.0
    return float(value)


def compute_score(candidate: Candidate, weights: ScoreWeights) -> ScoreBreakdown:
    base = float(candidate.score)
    text = _signal(candidate, "text")
    multimodal = _signal(candidate, "multimodal")
    popularity = _signal(candidate, "popularity")
    recency = _signal(candidate, "recency")
    novelty = _signal(candidate, "novelty")
    total = (
        base * weights.base
        + text * weights.text
        + multimodal * weights.multimodal
        + popularity * weights.popularity
        + recency * weights.recency
        + novelty * weights.novelty
    )
    return ScoreBreakdown(
        total=total,
        base=base,
        text=text,
        multimodal=multimodal,
        popularity=popularity,
        recency=recency,
        novelty=novelty,
    )
