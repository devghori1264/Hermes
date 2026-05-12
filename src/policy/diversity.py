from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.domain.models import RankedItem


@dataclass(frozen=True)
class DiversityConfig:
    max_per_group: int = 2
    relevance_weight: float = 0.75
    diversity_weight: float = 0.25
    group_keys: tuple[str, ...] = ("category", "genre", "topic", "domain", "cluster_id")


class DiversityOptimizer:
    def __init__(self, config: DiversityConfig | None = None) -> None:
        self.config = config or DiversityConfig()

    def _group_key(self, item: RankedItem) -> str:
        metadata = item.metadata if isinstance(item.metadata, dict) else {}
        for key in self.config.group_keys:
            value = metadata.get(key)
            if value:
                return str(value)
        title = item.title.strip().lower()
        if not title:
            return "unknown"
        return title.split()[0]

    def _diversity_penalty(self, item: RankedItem, seen_groups: dict[str, int]) -> float:
        group = self._group_key(item)
        count = seen_groups.get(group, 0)
        if count == 0:
            return 0.0
        return float(count)

    def rerank(self, ranked_items: list[RankedItem]) -> list[RankedItem]:
        if not ranked_items:
            return []

        remaining = list(ranked_items)
        seen_groups: dict[str, int] = {}
        output: list[RankedItem] = []

        while remaining:
            best_index = 0
            best_score = float("-inf")
            for index, item in enumerate(remaining):
                group = self._group_key(item)
                group_count = seen_groups.get(group, 0)
                if group_count >= self.config.max_per_group:
                    continue
                penalty = self._diversity_penalty(item, seen_groups)
                score = (item.score * self.config.relevance_weight) - (penalty * self.config.diversity_weight)
                if score > best_score:
                    best_score = score
                    best_index = index
            chosen = remaining.pop(best_index)
            group = self._group_key(chosen)
            seen_groups[group] = seen_groups.get(group, 0) + 1
            output.append(chosen)
        return output
