from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievalFilter:
    domain: str | None = None
    allowed_item_ids: set[str] | None = None


@dataclass(frozen=True)
class RetrievalQuery:
    embedding: list[float]
    top_k: int
    filters: RetrievalFilter | None = None


@dataclass(frozen=True)
class RetrievalCandidate:
    item_id: str
    score: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorItem:
    item_id: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    domain: str | None = None
    locale: str | None = None
