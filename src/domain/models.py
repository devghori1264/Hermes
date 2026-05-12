from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ItemRecord:
    item_id: str
    domain: str
    title: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecommendationContext:
    query_item_title: str
    user_id: str | None = None
    domain: str = "movies"
    experiment_group: str = "control"
    timestamp_utc: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class Candidate:
    item_id: str
    title: str
    score: float
    channel: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RankedItem:
    item_id: str
    title: str
    score: float
    explanation: str
    metadata: dict[str, Any] = field(default_factory=dict)
