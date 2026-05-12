from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict

@dataclass(frozen=True)
class ProvenanceRecord:
    source_id: str
    original_id: str
    ingestion_timestamp: str
    transformation_version: str

@dataclass(frozen=True)
class ResolvedEntity:
    canonical_id: str
    domain: str
    merged_attributes: dict[str, str]
    provenance: list[ProvenanceRecord]

class EntityResolver:
    def __init__(self) -> None:
        self.canonical_map: dict[str, str] = {}
        self.entities: dict[str, ResolvedEntity] = {}

    def resolve(self, records: list[dict[str, str]], domain: str) -> list[ResolvedEntity]:
        groups = defaultdict(list)
        for record in records:
            key = self._generate_match_key(record)
            groups[key].append(record)

        resolved_list = []
        for key, group in groups.items():
            canonical_id = f"{domain}_{key}"
            merged_attrs = {}
            provenance_list = []
            for item in group:
                for k, v in item.items():
                    if k not in merged_attrs and v:
                        merged_attrs[k] = v
                provenance_list.append(
                    ProvenanceRecord(
                        source_id=item.get("source", "unknown"),
                        original_id=item.get("id", "unknown"),
                        ingestion_timestamp=item.get("timestamp", "unknown"),
                        transformation_version="1.0"
                    )
                )
            entity = ResolvedEntity(
                canonical_id=canonical_id,
                domain=domain,
                merged_attributes=merged_attrs,
                provenance=provenance_list
            )
            self.entities[canonical_id] = entity
            resolved_list.append(entity)
        return resolved_list

    def _generate_match_key(self, record: dict[str, str]) -> str:
        title = record.get("title", "").strip().lower()
        year = record.get("year", "").strip()
        import re
        title_clean = re.sub(r"[^a-z0-9]", "", title)
        return f"{title_clean}_{year}" if year else title_clean
