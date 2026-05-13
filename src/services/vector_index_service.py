from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from src.retrieval.index import FaissVectorIndex
from src.retrieval.types import RetrievalCandidate, RetrievalFilter, RetrievalQuery, VectorItem


@dataclass(frozen=True)
class VectorIndexSnapshot:
    domain: str
    dimension: int
    item_count: int
    source_path: str


class VectorIndexService:
    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path
        self._index: Any = None
        self._payload: dict[str, Any] | None = None

    def available(self) -> bool:
        return self.index_path.exists()

    def load(self) -> None:
        if self._index is not None and self._payload is not None:
            return
        payload = json.loads(self.index_path.read_text(encoding="utf8"))
        domain = str(payload.get("domain", "unknown"))
        items = payload.get("items", [])
        if not isinstance(items, list):
            raise ValueError("invalid index payload")
        if not items:
            raise ValueError("empty index payload")

        dimension = len(items[0].get("vector", [])) if isinstance(items[0], dict) else 0
        try:
            import faiss
            index = FaissVectorIndex(dimension=dimension, name="trained_vector_index", index_type="Flat")
        except ImportError:
            from src.retrieval.index import InMemoryVectorIndex
            index = InMemoryVectorIndex(dimension=dimension, name="trained_vector_index")
            
        for entry in items:
            if not isinstance(entry, dict):
                continue
            vector = entry.get("vector", [])
            if not isinstance(vector, list) or not vector:
                continue
            index.add(
                VectorItem(
                    item_id=str(entry.get("item_id", "")),
                    vector=[float(value) for value in vector],
                    domain=domain,
                    metadata={"source": "trained_vector_index"},
                )
            )
        self._index = index
        self._payload = payload

    def snapshot(self) -> VectorIndexSnapshot:
        self.load()
        assert self._payload is not None
        items = self._payload.get("items", [])
        dimension = int(self._payload.get("dimension", 0))
        return VectorIndexSnapshot(
            domain=str(self._payload.get("domain", "unknown")),
            dimension=dimension,
            item_count=len(items),
            source_path=str(self.index_path),
        )

    def search(
        self,
        embedding: list[float],
        top_k: int = 10,
        domain: str | None = None,
        allowed_item_ids: set[str] | None = None,
    ) -> list[RetrievalCandidate]:
        self.load()
        assert self._index is not None
        filters = RetrievalFilter(domain=domain, allowed_item_ids=allowed_item_ids)
        query = RetrievalQuery(embedding=embedding, top_k=top_k, filters=filters)
        return self._index.search(query)

    def similar_item_ids(self, item_id: str, top_k: int = 10, domain: str | None = None) -> list[RetrievalCandidate]:
        self.load()
        assert self._payload is not None
        items = self._payload.get("items", [])
        target = None
        for entry in items:
            if isinstance(entry, dict) and str(entry.get("item_id", "")) == item_id:
                target = entry
                break
        if target is None:
            return []
        vector = target.get("vector", [])
        if not isinstance(vector, list) or not vector:
            return []
        allowed = {str(entry.get("item_id", "")) for entry in items if isinstance(entry, dict)}
        allowed.discard(item_id)
        return self.search([float(value) for value in vector], top_k=top_k, domain=domain, allowed_item_ids=allowed)
