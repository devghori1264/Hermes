from __future__ import annotations
from dataclasses import dataclass
import re

@dataclass(frozen=True)
class EntityRecord:
    id: str
    domain: str
    name: str
    attributes: dict[str, str]

class EntityResolutionPipeline:
    def __init__(self) -> None:
        self.canonical_map: dict[str, str] = {}
        self.entity_blocks: dict[str, list[EntityRecord]] = {}

    def _normalize_name(self, name: str) -> str:
        name = str(name).lower().strip()
        name = re.sub(r"[^\w\s]", "", name)
        name = re.sub(r"\s+", " ", name)
        return name

    def _blocking_key(self, record: EntityRecord) -> str:
        norm = self._normalize_name(record.name)
        parts = norm.split()
        if not parts:
            return "unknown"
        return parts[0][:3]

    def _jaccard_similarity(self, s1: str, s2: str) -> float:
        set1 = set(s1.split())
        set2 = set(s2.split())
        if not set1 and not set2:
            return 1.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return float(intersection) / float(union)

    def _match_score(self, r1: EntityRecord, r2: EntityRecord) -> float:
        n1 = self._normalize_name(r1.name)
        n2 = self._normalize_name(r2.name)
        if n1 == n2:
            return 1.0
        return self._jaccard_similarity(n1, n2)

    def add_records(self, records: list[EntityRecord]) -> None:
        for record in records:
            b_key = self._blocking_key(record)
            if b_key not in self.entity_blocks:
                self.entity_blocks[b_key] = []
            self.entity_blocks[b_key].append(record)

    def resolve(self, threshold: float = 0.8) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for b_key, block in self.entity_blocks.items():
            for i, r1 in enumerate(block):
                if r1.id in resolved:
                    continue
                resolved[r1.id] = r1.id
                for j in range(i + 1, len(block)):
                    r2 = block[j]
                    if r2.id in resolved:
                        continue
                    if self._match_score(r1, r2) >= threshold:
                        resolved[r2.id] = r1.id
        self.canonical_map.update(resolved)
        return dict(self.canonical_map)

    def get_canonical_id(self, entity_id: str) -> str:
        return self.canonical_map.get(entity_id, entity_id)
